"""
Load and prepare the lanelet map and graph data
"""
function load_map_data()
    # Load the lanelet map
    location0_file = joinpath(@__DIR__, "../res","location0.osm")
    projector = Projection.UtmProjector(Lanelet2.Io.Origin(50.99, 6.90))
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
function cluster_agents_into_scenarios(positions::Matrix{Float32}, distance_threshold::Float32=30.0f0)
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
function prepare_agent_features(lanelet_roadway::LaneletRoadway, save_features::Bool=false)
    # First collect all agent features as before
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
    scenarios = cluster_agents_into_scenarios(initial_positions)

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
# TODO: is it possible to acess only one partial routing graph
function prepare_map_features(lanelet_roadway, g_meta, save_features::Bool=false)
    polyline_graphs = GNNGraph[]
    pos_llt = []

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
        push!(pos_llt, llt_midpoint)

        # Iterate over points in centerline to get vector-level features
        polyline_features = []
        for i in 1:num_vectors
            # Get start and end points of each polyline segment
            start_point = centerline[i]
            end_point = centerline[i+1]
            # TODO: node(vector) features completion
            # Extract x,y coordinates for start and end points
            start_x = start_point.pos.x
            start_y = start_point.pos.y
            end_x = end_point.pos.x
            end_y = end_point.pos.y

            # Create feature vector with start and end coordinates
            push!(polyline_features, Float32[start_x, start_y, end_x, end_y])
        end
        # Convert to matrix format
        polyline_features = reduce(hcat, polyline_features)       # feature matrix:(4, num_vectors)

        g_fc.ndata.x = polyline_features
        g_fc.gdata.id = lanelet_id
        push!(polyline_graphs, g_fc)
    end

    pos_llt = hcat(pos_llt...)

    polyline_graphs = batch(polyline_graphs)
    @assert size(polyline_graphs.x, 1) == 4
    @assert polyline_graphs.num_graphs == nv(g_meta)

    # Compute the mean and std
    # Only use the start x and start y of each vector for mean and std
    μ, σ = VectorLanelet.calculate_mean_and_std(polyline_graphs.x[1:2, :]; dims=2)

    g_heteromap = GNNHeteroGraph(
        (:lanelet, :right, :lanelet) => extract_gml_src_dst(g_meta, "Right"),
        (:lanelet, :left, :lanelet) => extract_gml_src_dst(g_meta, "Left"),
        (:lanelet, :suc, :lanelet) => extract_gml_src_dst(g_meta, "Successor"),
        (:lanelet, :adj_left, :lanelet) => extract_gml_src_dst(g_meta, "AdjacentLeft"),
        (:lanelet, :adj_right, :lanelet) => extract_gml_src_dst(g_meta, "AdjacentRight"),
        dir = :in
    )

    if save_features
        # Save the map features
        cache_path = joinpath(@__DIR__, "../res/map_features.jld2")
        @info "Saving map features to $(cache_path)"
        jldsave(cache_path, map_features=polyline_graphs.x,
        polyline_graphs=polyline_graphs, g_heteromap=g_heteromap, μ=μ, σ=σ)
    end

    return polyline_graphs, g_heteromap, pos_llt, μ, σ
end

# Calculate the midpoint coordinates of a given lanelet by its centerline
function calculate_llt_midpoint(centerline)
    num_points = length(centerline)
    mid_idx = div(num_points, 2)
    mid_point = centerline[mid_idx]
    return [mid_point.pos.x, mid_point.pos.y]
end

function agent_features_upsample(agt_features)
    agt_features = permutedims(agt_features, (2, 1, 3))
    agt_features = upsample_linear(agt_features, size=10)
    agt_features = permutedims(agt_features, (2, 1, 3))
    return agt_features
end

function prepare_data(lanelet_roadway, g_meta, save_features::Bool=false)

end


"""
    Preprocess data for DataLoader
expect data to include:
    - agent_data: (agt_features_upsampled)
    - map_data: (polyline_graphs, g_heteromap, llt_pos)
    - labels: (2, timesteps, num_agents)
"""
function preprocess_data(data, overfit::Bool=false)
    num_scenarios = length(data.agent_data.agt_features_upsampled)
    agent_data, map_data, labels = data
    agt_current_pos = [i[:,end,:] for i in agent_data.agt_features_upsampled]
    polyline_graphs = [map_data.polyline_graphs for _ in 1:num_scenarios]
    g_heteromap = [map_data.g_heteromap for _ in 1:num_scenarios]
    llt_pos = [map_data.llt_pos for _ in 1:num_scenarios]


    if overfit
        @info "Performing overfitting"
        X = (agent_data.agt_features_upsampled[1,:], agt_current_pos[1,:],
        polyline_graphs[1,:], g_heteromap[1,:], llt_pos[1,:])
        Y = labels[1,:]
    else
        X = (;agent_data.agt_features_upsampled, agt_current_pos,
        polyline_graphs, g_heteromap, llt_pos)
        Y = labels
    end

    return X, Y
end