"""
    Load and prepare the lanelet map and graph data
"""
function load_map_data()
    # Load the lanelet map
    location0_file = joinpath(@__DIR__, "../res","location0.osm")
    projector = Projection.UtmProjector(Lanelet2.Io.Origin(50.9904, 6.9003))
    llmap = Lanelet2.Io.load(location0_file, projector)

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
    Group agents that are close to each other into scenarios based on spatial proximity
"""
function cluster_agents_into_scenarios(positions::Matrix{Float32}, distance_threshold::Float32=0.001f0)
    num_agents = size(positions, 2)
    assigned = falses(num_agents)
    scenarios = Vector{Vector{Int}}()

    for i in 1:num_agents
        if assigned[i]
            continue
        end

        # Start a new scenario
        current_scenario = [i]
        assigned[i] = true

        # Find nearby agents
        for j in (i+1):num_agents
            if assigned[j]
                continue
            end

            # Check distance between agents i and j using their initial positions
            dist = sqrt(sum((positions[:,i] - positions[:,j]).^2))
            if dist < distance_threshold
                push!(current_scenario, j)
                assigned[j] = true
            end
        end

        push!(scenarios, current_scenario)
    end

    return scenarios
end

"""
Prepare agent features and agent end position from lanelet centerlines
    - agt_features: num_scenarios x (channels, time_step, num_agents)
    - agt_pos_end: num_scenarios x (2, num_agents)
"""
function prepare_agent_features(lanelet_roadway::LaneletRoadway, save_features::Bool=false; cluster_thrd::Float32=20.0f0)

    temp_features = Vector{Matrix{Float32}}()
    temp_pos_end = Vector{Vector{Float32}}()
    for lanelet in values(lanelet_roadway.lanelets)

        curve = lanelet.curve
        push!(temp_features, hcat([curve[1].pos.x, curve[1].pos.y], [curve[2].pos.x, curve[2].pos.y]))
        push!(temp_pos_end, [curve[end].pos.x, curve[end].pos.y])
    end

    # Get initial positions of all agents
    initial_positions = hcat([f[:,1] for f in temp_features]...)

    # Group agents into scenarios
    scenarios = cluster_agents_into_scenarios(initial_positions, cluster_thrd)
    @show length(scenarios)

    # Reorganize features and end positions by scenarios
    agt_features_by_scenario = []
    agt_pos_end_by_scenario = []

    for scenario_indices in scenarios
        scenario_features = cat([temp_features[i] for i in scenario_indices]..., dims=3)
        scenario_pos_end = hcat([temp_pos_end[i] for i in scenario_indices]...)

        push!(agt_features_by_scenario, scenario_features)
        push!(agt_pos_end_by_scenario, scenario_pos_end)
    end

    if save_features
        cache_path = joinpath(@__DIR__, "../res/agent_features.jld2")
        @info "Saving agent features to $(cache_path)"
        jldsave(cache_path, agt_features=agt_features, agt_pos_end=agt_pos_end)
    end

    return agt_features_by_scenario, agt_pos_end_by_scenario
end

"""
Prepare map features stored in polyline level GNNGraph(fulled-connected graph)
    vector features: (6, num_vectors) -> polyline_graphs.x
        - start_x, start_y, end_x, end_y
        - attribute features for object type
        - polyline id
    routing_graph -> g_heteromap
"""
function prepare_map_features(lanelet_roadway, g_meta, save_features::Bool=false)
    polylines_graph = get_polylines_graph(lanelet_roadway, g_meta)
    # Compute the mean and std
    # Only use the start x and start y of each vector for mean and std
    μ, σ = VectorLanelet.calculate_mean_and_std(polyline_graphs.x[1:2, :]; dims=2)

    g_heteromap = GNNHeteroGraph(
        (:lanelet, :right, :lanelet) => extract_gml_src_dst(g_meta, "Right"),
        (:lanelet, :left, :lanelet) => extract_gml_src_dst(g_meta, "Left"),
        (:lanelet, :suc, :lanelet) => extract_gml_src_dst(g_meta, "Successor"),
        # TODO: whether to put adjacent left/right here
        (:lanelet, :adj_left, :lanelet) => extract_gml_src_dst(g_meta, "AdjacentLeft"),
        (:lanelet, :adj_right, :lanelet) => extract_gml_src_dst(g_meta, "AdjacentRight"),
        dir = :out
    )

    if save_features
        # Save the map features
        cache_path = joinpath(@__DIR__, "../res/map_features.jld2")
        @info "Saving map features to $(cache_path)"
        jldsave(cache_path, map_features=polyline_graphs.x,
        polyline_graphs=polyline_graphs, g_heteromap=g_heteromap, μ=μ, σ=σ)
    end

    return polylines_graph, g_heteromap, μ, σ
end
"""
    get_polylines_graph(lanelet_roadway, g_meta)
return batched fully-connected graphs, each graph represents a lanelet
"""
function get_polylines_graph(lanelet_roadway, g_meta)
    polylines_graph = GNNGraph[]

    # Traverse all the lanelets according to vertex order
    # The vertex order is acquired from gml file
    # Mapping: v -> lanelet_id
    for v in 1:nv(g_meta)
        lanelet_attr = Lanelet2.extract_graphml_attributes(get_prop(g_meta, v, :info))
        lanelet_id = lanelet_attr.lanelet_id
        lanelet_tag = LaneletTag(lanelet_id, lanelet_attr.inverted)
        lanelet = lanelet_roadway[lanelet_tag]

        # Check if the lanelets' order is aligned with the vertices' order (in location0 map)
        # v == 14 && @assert lanelet.tag == LaneletTag(1707, false)

        centerline = lanelet.curve
        num_vectors = length(centerline) - 1
        g_fc = complete_digraph(num_vectors) |> GNNGraph

        # Calculate the midpoint coordinates of the lanelet
        llt_midpoint = calculate_llt_midpoint(centerline)

        # Iterate over points in centerline to get vector-level features
        polyline_features = []
        for i in 1:num_vectors
            # TODO: normalize the vector features according to agent-centric
            # Get start and end points of each polyline segment
            start_point = centerline[i]
            end_point = centerline[i+1]
            # TODO: complete node(vector) features
            # Extract x,y coordinates for start and end points
            start_x = start_point.pos.x
            start_y = start_point.pos.y
            end_x = end_point.pos.x
            end_y = end_point.pos.y

            # Create feature vector with start and end coordinates
            push!(polyline_features, Float32[start_x, start_y, end_x, end_y])
        end
        polyline_features = reduce(hcat, polyline_features)       # feature matrix:(4, num_vectors)

        g_fc.ndata.x = polyline_features
        g_fc.gdata.id = lanelet_id
        g_fc.gdata.pos = llt_midpoint
        push!(polylines_graph, g_fc)
    end

    polylines_graph = batch(polylines_graph)
    @assert size(polylines_graph.x, 1) == 4 "Vector features should have $(size(polylines_graph.x, 1)) channels"

    return polylines_graph
end

# Calculate the midpoint coordinates of a given lanelet by its centerline
function calculate_llt_midpoint(centerline)
    num_points = length(centerline)
    mid_idx = div(num_points, 2)
    mid_point = centerline[mid_idx]
    return reshape([mid_point.pos.x, mid_point.pos.y], 2, 1)    # reshape for gnngraph.gdata
end

function agent_features_upsample(agt_features, upsample_size::Int=20)
    agt_features = permutedims(agt_features, (2, 1, 3))
    agt_features = upsample_linear(agt_features, size=upsample_size)
    agt_features = permutedims(agt_features, (2, 1, 3))
    return agt_features
end

function prepare_data(config, device::Function)
    lanelet_roadway, g_meta = load_map_data()

    @info "Preparing agent features on $(device)"
    agt_features, agt_pos_end = prepare_agent_features(lanelet_roadway, cluster_thrd=config["cluster_thrd"]) |> device
    agt_features_upsampled = map(agent_features_upsample, agt_features) |> device

    @info "Preparing map features on $(device)"
    polylines_graph, g_heteromap, μ, σ = prepare_map_features(lanelet_roadway, g_meta) |> device

    labels = config["predict_current_pos"] ? agt_features[:, 2, :] : agt_pos_end

    agent_data = (; agt_features_upsampled, agt_features)
    map_data = (; polylines_graph, g_heteromap, μ, σ)
    data = (; agent_data, map_data, labels)
    return data
end