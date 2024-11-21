using VectorLanelet
using Lanelet2
using Lanelet2.Routing
using Lanelet2.Core
using AutomotiveSimulator
using Graphs
using MetaGraphs
using NestedGraphsIO
using GraphNeuralNetworks
using GraphIO.GraphML
using Transformers
using Flux
using Statistics

using Wandb, Logging

"""
Load and prepare the lanelet map and graph data
"""
function load_map_data()
    # Load the lanelet map
    example_file = joinpath(@__DIR__, "../res","location0.osm")
    projector = Projection.UtmProjector(Lanelet2.Io.Origin(49, 8.4))
    llmap = Lanelet2.Io.load(example_file, projector)

    # Use passable lanelet submap
    traffic_rules = TrafficRules.create(TrafficRules.Locations.Germany, TrafficRules.Participants.Vehicle)
    rg = RoutingGraph(llmap, traffic_rules)
    llmap = rg.passableLaneletSubmap()
    lanelet_roadway = LaneletRoadway(llmap)

    # Load the meta graph
    gml_file_path = joinpath(@__DIR__, "../res/location0.osm.gml")
    g_meta = open(gml_file_path, "r") do io
        NestedGraphsIO.loadgraph(io, "G", GraphMLFormat(), MGFormat())
    end

    return lanelet_roadway, g_meta
end

"""
Prepare agent features and labels from lanelet centerlines, as well as the mean and std of the features
"""
function prepare_agent_features(lanelet_roadway::LaneletRoadway; n_train::Int=70)
    agt_features = Vector{Matrix{Float64}}()
    pos_agt = Vector{Vector{Float64}}()
    labels = Vector{Vector{Float64}}()
    for lanelet in values(lanelet_roadway.lanelets)
        curve = lanelet.curve
        push!(agt_features, hcat([curve[1].pos.x, curve[1].pos.y], [curve[2].pos.x, curve[2].pos.y])')
        push!(pos_agt, [curve[2].pos.x, curve[2].pos.y])
        push!(labels, [curve[end].pos.x, curve[end].pos.y])
    end

    agt_features = Float32.(cat(agt_features..., dims=3))
    
    # Normalize agent features to zero mean and unit variance
    # TODO: μ and σ could differ in other maps
    μ, σ = VectorLanelet.calculate_mean_and_std(agt_features; dims=(1,3))       # μ and σ of size (1,2,1)
    agt_features = (agt_features .- μ) ./ σ

    agt_features_train = agt_features[:,:,1:n_train]
    agt_features_test = agt_features[:,:,n_train+1:end]

    labels = Float32.(hcat(labels...))
    
    # Split labels
    labels_train = labels[:,1:n_train]
    labels_test = labels[:,n_train+1:end]
    @assert size(agt_features_train) == (2,2,n_train)
    @assert size(labels_train) == (2, n_train)

    # Reshape the μ and σ to Vector
    μ = reshape(μ,:)
    σ = reshape(σ,:)
    return agt_features_train, agt_features_test, labels_train, labels_test, μ, σ
end

"""
Prepare map features stored in GNNGraph(fulled-connected graph)
"""
function prepare_map_features(lanelet_roadway, g_meta, μ, σ)
    polyline_graphs = GNNGraph[]
    llt_pos = []
    for v in 1:nv(g_meta)
        lanelet_attr = Lanelet2.extract_graphml_attributes(get_prop(g_meta, v, :info))
        lanelet_tag = LaneletTag(lanelet_attr.lanelet_id, lanelet_attr.inverted)
        lanelet = lanelet_roadway[lanelet_tag]
        
        centerline = lanelet.curve
        num_nodes = length(centerline) - 1
        g_fc = complete_digraph(num_nodes) |> GNNGraph
        
        # Iterate over points in centerline to get vector-level features
        polyline_features = []
        for i in 1:num_nodes
            # Get start and end points of each polyline segment
            start_point = centerline[i]
            end_point = centerline[i+1]
            
            # Extract x,y coordinates for start and end points
            start_x = start_point.pos.x
            start_y = start_point.pos.y
            end_x = end_point.pos.x 
            end_y = end_point.pos.y
            
            # Create feature vector with start and end coordinates
            push!(polyline_features, Float32[start_x, start_y, end_x, end_y])
        end
        # Convert to matrix format
        polyline_features = reduce(hcat, polyline_features)       # feature matrix:(4, num_nodes)
        
        # Normalize polyline features to zero mean and unit variance
        μ_reshaped = repeat(μ, 2)
        σ_reshaped = repeat(σ, 2)
        polyline_features = (polyline_features .- μ_reshaped) ./ σ_reshaped
        
        g_fc.ndata.x = polyline_features
        push!(polyline_graphs, g_fc)
    end

    # Create heterogeneous graph
    g_all = batch(polyline_graphs)
    @assert size(g_all.x, 1) == 4
    @assert g_all.num_graphs == nv(g_meta)
    g_hetero = GNNHeteroGraph(
        (:lanelet, :right, :lanelet) => extract_gml_src_dst(g_meta, "Right"),
        (:lanelet, :left, :lanelet) => extract_gml_src_dst(g_meta, "Left"),
        (:lanelet, :suc, :lanelet) => extract_gml_src_dst(g_meta, "Successor"),
        (:lanelet, :adj_left, :lanelet) => extract_gml_src_dst(g_meta, "AdjacentLeft"),
        (:lanelet, :adj_right, :lanelet) => extract_gml_src_dst(g_meta, "AdjacentRight")
    )
    
    return g_all, g_hetero
end

"""
Main encoding pipeline
"""

function run_training(config::Dict{String, Any})
    lanelet_roadway, g_meta = load_map_data()
    
    # Prepare data
    @info "Preparing agent features"
    agt_features_train, agt_features_test, labels_train, labels_test, μ, σ = prepare_agent_features(lanelet_roadway)
    @info "Preparing map features"
    g_all, g_hetero = prepare_map_features(lanelet_roadway, g_meta, μ, σ)
    
    # Initialize model
    model = LaneletPredictor(config)
    
    # Training setup
    opt = Flux.setup(Adam(config["learning_rate"]), model)
    num_epochs = config["num_epochs"]
    
    train_loader = Flux.DataLoader(
        (agt_features_train, labels_train),
        batchsize=config["batch_size"],
        shuffle=true
    )

    # Training loop
    @info "Start training"
    for epoch in 1:num_epochs
        train_loss = 0.
        train_batches = 0
        for (agt_batch, label_batch) in train_loader
            loss, grad = Flux.withgradient(model) do m
                predictions = m(agt_batch, g_all, g_hetero, μ, σ)
                Flux.mse(predictions, label_batch)
            end
            
            train_loss += loss
            train_batches += 1
            Flux.update!(opt, model, grad[1])
        end
        
        if epoch % 10 == 0
           @info "Epoch $epoch: Train loss = $(train_loss/train_batches)"
        end
    end
    
end
include("../src/config.jl")
run_training(config)