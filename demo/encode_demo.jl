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
Prepare agent features and labels from lanelet centerlines
"""
function prepare_agent_features(lanelet_roadway::LaneletRoadway; n_train::Int=70)
    agt_features = [
        hcat([curve[1].pos.x, curve[1].pos.y], [curve[2].pos.x, curve[2].pos.y]) 
        for curve in (lanelet.curve for lanelet in values(lanelet_roadway.lanelets))
    ]
    
    pos_agt = [
        [curve[2].pos.x, curve[2].pos.y]
        for curve in (lanelet.curve for lanelet in values(lanelet_roadway.lanelets))
    ]
    
    labels = [
        [curve[end].pos.x, curve[end].pos.y]
        for curve in (lanelet.curve for lanelet in values(lanelet_roadway.lanelets))
    ]

    # Split data
    agt_features = Float32.(cat(agt_features..., dims=3))
    agt_features_train = agt_features[:,:,1:n_train]
    agt_features_test = agt_features[:,:,n_train+1:end]

    # Format and split labels
    labels = Float32.(hcat(labels...))
    labels_train = labels[:,1:n_train]
    labels_test = labels[:,n_train+1:end]
    @assert size(agt_features_train) == (2,2,n_train)
    @assert size(labels_train) == (2, n_train)
    return agt_features_train, agt_features_test, labels_train, labels_test
end

"""
Prepare map features stored in GNNGraph(fulled-connected graph)
"""
function prepare_map_features(lanelet_roadway, g_meta)
    poliline_graphs = [
        let
            lanelet_attr = Lanelet2.extract_graphml_attributes(get_prop(g_meta, v, :info))
            lanelet_tag = LaneletTag(lanelet_attr.lanelet_id, lanelet_attr.inverted)
            lanelet = lanelet_roadway[lanelet_tag]
            
            centerline = lanelet.curve
            num_nodes = length(centerline) - 1
            g_fc = complete_digraph(num_nodes) |> GNNGraph
            
            llt_features = [
                Float32[
                    centerline[i].pos.x, 
                    centerline[i].pos.y,
                    centerline[i+1].pos.x, 
                    centerline[i+1].pos.y
                ] for i in 1:num_nodes
            ]
            
            g_fc.ndata.x = reduce(hcat, llt_features)
            g_fc
        end
        for v in 1:nv(g_meta)
    ]

    # Create heterogeneous graph
    g_all = batch(poliline_graphs)
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
function main()
    lanelet_roadway, g_meta = load_map_data()
    
    # Prepare data
    @info "Preparing agent features"
    agt_features_train, agt_features_test, labels_train, labels_test = prepare_agent_features(lanelet_roadway)
    @info "Preparing map features"
    g_all, g_hetero = prepare_map_features(lanelet_roadway, g_meta)
    
    # Initialize model
    actor_config = Dict("n_actor" => 128, "din_actor" => 2)
    map_config = Dict("n_map" => 128, "num_scales" => 6)
    model = LaneletPredictor(actor_config, map_config)
    
    # Training setup
    opt = Flux.setup(Adam(1e-2), model)
    num_epochs = 100
    
    train_loader = Flux.DataLoader(
        (agt_features_train, labels_train),
        batchsize=10,
        shuffle=true
    )

    # Training loop
    @info "Start training"
    for epoch in 1:num_epochs
        epoch_loss = 0.
        num_batches = 0
        for (agt_batch, label_batch) in train_loader
            loss, grad = Flux.withgradient(model) do m
                predictions = m(agt_batch, g_all, g_hetero)
                Flux.mse(predictions, label_batch)
            end
            
            epoch_loss += loss
            num_batches += 1
            Flux.update!(opt, model, grad[1])
        end

        avg_loss = epoch_loss / num_batches

        # Print progress every 10 epochs
        if epoch % 10 == 0
           @info "Epoch $epoch: Average loss = $avg_loss"
        end
    end
    
    # TODO: Accuracy
    
end
