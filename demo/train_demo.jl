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
using MLUtils

using Wandb, Logging, Dates

"""
Load and prepare the lanelet map and graph data
"""
function load_map_data()
    # Load the lanelet map
    example_file = joinpath(@__DIR__, "../res","location0.osm")
    projector = Projection.UtmProjector(Lanelet2.Io.Origin(50.99, 6.90))
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
Prepare agent features and labels from lanelet centerlines
"""
function prepare_agent_features(lanelet_roadway::LaneletRoadway)
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
    # μ, σ = VectorLanelet.calculate_mean_and_std(agt_features; dims=(1,3))       # μ and σ of size (1,2,1)
    # agt_features = (agt_features .- μ) ./ σ

    labels = Float32.(hcat(labels...))
    
    
    # labels_train = labels[:,1:n_train]
    # labels_test = labels[:,n_train+1:end]


    # labels_train = labels[:,1:n_train]
    # labels_test = labels[:,n_train+1:end]

    # Reshape the μ and σ to Vector
    # μ = reshape(μ,:)
    # σ = reshape(σ,:)
    return agt_features, labels
end

"""
Prepare map features stored in polyline level GNNGraph(fulled-connected graph)
"""
function prepare_map_features(lanelet_roadway, g_meta)
    polyline_graphs = GNNGraph[]
    llt_pos = []

    # Traverse all the lanelets according to vertex order
    # Mapping: v -> lanelet_id
    for v in 1:nv(g_meta)
        lanelet_attr = Lanelet2.extract_graphml_attributes(get_prop(g_meta, v, :info))
        lanelet_tag = LaneletTag(lanelet_attr.lanelet_id, lanelet_attr.inverted)
        lanelet = lanelet_roadway[lanelet_tag]

        # Check if the lanelets' order is aligned with the nodes' order(in location0 map)
        v == 14 && @assert lanelet.tag == LaneletTag(1707, false)
        
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
        # μ_reshaped = repeat(cpu(μ), 2)
        # σ_reshaped = repeat(cpu(σ), 2)
        # polyline_features = (polyline_features .- μ_reshaped) ./ σ_reshaped
        
        g_fc.ndata.x = polyline_features
        push!(polyline_graphs, g_fc)
    end

    # Create heterogeneous graph
    g_polyline = batch(polyline_graphs)
    @assert size(g_polyline.x, 1) == 4
    @assert g_polyline.num_graphs == nv(g_meta)
    g_heteromap = GNNHeteroGraph(
        (:lanelet, :right, :lanelet) => extract_gml_src_dst(g_meta, "Right"),
        (:lanelet, :left, :lanelet) => extract_gml_src_dst(g_meta, "Left"),
        (:lanelet, :suc, :lanelet) => extract_gml_src_dst(g_meta, "Successor"),
        (:lanelet, :adj_left, :lanelet) => extract_gml_src_dst(g_meta, "AdjacentLeft"),
        (:lanelet, :adj_right, :lanelet) => extract_gml_src_dst(g_meta, "AdjacentRight")
    )
    
    return g_polyline, g_heteromap
end

function run_training(wblogger::WandbLogger, config::Dict{String, Any})
    device = config["use_cuda"] ? gpu : cpu
    lanelet_roadway, g_meta = load_map_data()
    
    # Prepare data
    @info "Preparing agent features on $(device)"
    agt_features, labels = prepare_agent_features(lanelet_roadway) |> device
    @info "Preparing map features on $(device)"
    g_polyline, g_heteromap = prepare_map_features(lanelet_roadway, g_meta) |> device
    
    # Initialize model
    @info "Initializing model and moving to $(device)"
    model = LaneletPredictor(agt_features, g_polyline.x) |> device
    
    # Training setup
    opt = Flux.setup(Adam(config["learning_rate"]), model)
    num_epochs = config["num_epochs"]

    # Split training data
    train_data, val_data = splitobs((agt_features, labels), at=0.9)
    
    train_loader = Flux.DataLoader(
        train_data,
        batchsize=config["batch_size"],
        shuffle=true
    )

    # Initial logging
    for (type, data) in ("train" => train_data, "val" => val_data)
        let (x, y) = data
            Flux.reset!(model)
            loss = loss_fn(model, x, y, g_polyline, g_heteromap)
            epoch = 0
            logging_callback(wblogger, type, epoch, cpu(loss)..., log_step_increment=0)
        end
    end

    # Training loop
    @info "Start training"
    for epoch in 1:num_epochs
        Flux.reset!(model)
        
        # Training
        for (x, y) in train_loader
            loss, grad = Flux.withgradient(model) do m
                loss_fn(m, x, y, g_polyline, g_heteromap)
            end
            
            Flux.update!(opt, model, grad[1])
            logging_callback(wblogger, "train", epoch, cpu(loss)..., log_step_increment=length(y))  # log_step_increment = batch size
        end
    

        # Validation
        let (x, y) = val_data
            Flux.reset!(model)
            values = loss_fn(model, x, y, g_polyline, g_heteromap)

            # logging
            logging_callback(wblogger, "val", epoch, cpu(values)..., log_step_increment=length(y))
        end

    end
    
    # Save the trained model
    @info "Saving the trained model state"
    model_path = joinpath(@__DIR__, "../models/model_$(now()).jld2")
    save_model_state(model, model_path)
end


include("../src/config.jl")

wblogger = WandbLogger(
    project = "VectorLanelet",
    name = "demo-$(now())",
    config = config
)
run_training(wblogger, config)
close(wblogger)