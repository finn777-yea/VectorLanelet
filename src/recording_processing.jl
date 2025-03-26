"""
    create_prediction_dataset(dataset::Dataset, prediction_horizon=5)

Create a prediction dataset with the following structure:
- Each entry represents all vehicles in a specific frame
- X contains current positions/states of all vehicles in the frame
- Y contains future positions of all vehicles after prediction_horizon frames
- No fixed vehicle count - uses actual number of vehicles in each frame

Returns:
- frame_features: List where each element contains all vehicles' data for one frame
- frame_targets: List where each element contains future positions for one frame
- frame_metadata: List of (recordingId, frame) to track which frame each sample represents
"""
function get_agent_prediction_dataset(dataset::Dataset, frame_limit=200,prediction_horizon=5)
    frame_features = []  # Each element contains all vehicles in one frame
    frame_targets = []   # Each element contains future positions for all the corresponding vehicles
    frame_metadata = []  # Tracking which frame each entry represents

    # Group by frames
    frame_groups = groupby(dataset.tracks, [:recordingId, :frame])

    for (key, frame_df) in Iterators.take(pairs(frame_groups), frame_limit)
        current_recording = key.recordingId
        current_frame = key.frame

        # Find the future frame data
        future_frame = current_frame + prediction_horizon
        future_frame_data = filter(row ->
            row.recordingId == current_recording &&
            row.frame == future_frame,
            dataset.tracks
        )

        nrow(future_frame_data) == 0 && continue

        # Store all the vehs in the current frame
        vehicles_features = []
        vehicles_targets = []
        track_ids_in_frame = []

        # Process each vehicle in the current frame
        for vehicle_idx in 1:nrow(frame_df)
            vehicle = frame_df[vehicle_idx, :]

            # Skip the vehicles that don't have future positions
            future_vehicle = filter(row -> row.trackId == vehicle.trackId, future_frame_data)
            nrow(future_vehicle) == 0 && continue

            # Extract vehicle features
            vehicle_feature = [
                vehicle.xCenter,        # x position
                vehicle.yCenter,        # y position
                # vehicle.trackId,        # Vehicle ID
                vehicle.xVelocity,      # x velocity
                vehicle.yVelocity,      # y velocity
                vehicle.heading,        # heading
                vehicle.width,          # width
                vehicle.length          # length
            ]

            vehicle_target = [
                # vehicle.trackId,                # Vehicle ID for tracking
                future_vehicle[1, :xCenter],    # Future x position
                future_vehicle[1, :yCenter]     # Future y position
            ]

            push!(vehicles_features, vehicle_feature)
            push!(vehicles_targets, vehicle_target)
            push!(track_ids_in_frame, vehicle.trackId)
        end

        vehicles_features = hcat(vehicles_features...)
        vehicles_targets = hcat(vehicles_targets...)
        @assert size(vehicles_features, 2) == size(vehicles_targets, 2)
        push!(frame_features, vehicles_features)
        push!(frame_targets, vehicles_targets)
        push!(frame_metadata, (;current_recording, current_frame, track_ids_in_frame))
    end

    return frame_features, frame_targets, frame_metadata
end



### Map related
# Export the routing graphs from dataset.maps to MetaDiGraph
# Return the routing graph in MetaDiGraph type, stored in Dict
function get_map_features(
    dataset::Dataset,
    recording_ids::Vector{Int},
    export_dir::String = joinpath(@__DIR__, "..", "res", "highd_routing_graphs")
)
    !isdir(export_dir) && mkdir(export_dir)

    polylines_graphs = Dict{Int, GNNGraph}()
    hetero_routing_graphs = Dict{Int, GNNHeteroGraph}()
    for i in recording_ids

        lanelet_roadway = dataset.maps[i]
        g_meta = get_routing_graph_in_MetaDiGraph(lanelet_roadway, export_dir, i)
        polylines_graph = VectorLanelet.get_polylines_graph(lanelet_roadway, g_meta)
        g_heteromap = GNNHeteroGraph(
            (:lanelet, :right, :lanelet) => VectorLanelet.extract_gml_src_dst(g_meta, "Right"),
            (:lanelet, :left, :lanelet) => VectorLanelet.extract_gml_src_dst(g_meta, "Left"),
            # (:lanelet, :suc, :lanelet) => VectorLanelet.extract_gml_src_dst(g_meta, "Successor"),
            # TODO: whether to put adjacent left/right here
            # TODO: handle no edge case
            # (:lanelet, :adj_left, :lanelet) => VectorLanelet.extract_gml_src_dst(g_meta, "AdjacentLeft"),
            # (:lanelet, :adj_right, :lanelet) => VectorLanelet.extract_gml_src_dst(g_meta, "AdjacentRight"),
            dir = :out
        )

        polylines_graphs[i] = polylines_graph
        hetero_routing_graphs[i] = g_heteromap
    end

    return polylines_graphs, hetero_routing_graphs
end

function get_routing_graph_in_MetaDiGraph(
    lanelet_roadway::LaneletRoadway,
    export_dir::String,
    recording_id::Int
)
    gml_file_path = joinpath(export_dir, "hd_recording_$(recording_id).gml")
    traffic_rules = Lanelet2.TrafficRules.create(Lanelet2.TrafficRules.Locations.Germany, Lanelet2.TrafficRules.Participants.Vehicle)
    routing_graph = RoutingGraph(lanelet_roadway.lanelet_map, traffic_rules)
    routing_graph.exportGraphML(gml_file_path)

    # Load gml to MetaGraphs
    g_meta = open(gml_file_path, "r") do io
        NestedGraphsIO.loadgraph(io, "G", GraphMLFormat(), MGFormat())
    end

    return g_meta
end



"""
    prepare_recording_data(dataset, recording_ids, config)
Prepare the recording data
expect the recording data to be saved in the following format:
- X: num_scenarios x (2, num_agents)
- Y: num_scenarios x (2, num_agents)

Returns:
- X: agt_features, agt_current_pos, polyline_graphs, g_heteromaps, llt_pos
- Y: labels
- μ: μ_agt, μ_map
- σ: σ_agt, σ_map
"""
function prepare_recording_data(config)
    # Load the dataset
    recording_ids = config["recording_ids"]
    dataset = load_dataset(config["recording_path"], recording_ids)

    # Agent data
    agent_data_path = config["recording_agent_data_path"]
    X, Y, metadatas = load(agent_data_path, "X", "Y", "metadatas")
    agt_features = X

    # Map data
    polylines_graphs, hetero_routing_graphs = get_map_features(dataset, recording_ids)
    polylines_graphs = [polylines_graphs[m.current_recording] for m in metadatas]
    hetero_routing_graphs = [hetero_routing_graphs[m.current_recording] for m in metadatas]
    # Retrieve agt_pos and llt_pos
    agt_pos = [scenario[1:2, :] for scenario in agt_features]
    llt_pos = [pg.gdata.pos for pg in polylines_graphs]

    # Calculate the statistics for normalization
    # TODO: is it better to calculate them seperately for each scenario
    μ_agt, σ_agt = VectorLanelet.calculate_mean_and_std(hcat(agt_features...); dims=2)
    μ_map, σ_map = VectorLanelet.calculate_mean_and_std(hcat(llt_pos...); dims=2)
    μ = (;μ_agt, μ_map)
    σ = (;σ_agt, σ_map)

    X = (;agt_features, agt_pos, polylines_graphs, hetero_routing_graphs, llt_pos)

    return X, Y, μ, σ
end

function load_dataset(dataset_path, recording_ids)
    data_infos = HighDRecordings(recording_ids, dataset_path)
    dataset = read(Adapter([data_infos]))
    return dataset
end
