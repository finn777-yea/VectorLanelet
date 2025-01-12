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
Prepare agent features and labels from lanelet centerlines
    - agent features: (2, 2, B)   (channels, time_step, batch_size)
    - labels: (2, B)            (channels, batch_size)

"""
function prepare_agent_features(lanelet_roadway::LaneletRoadway, save_features::Bool=false)
    agt_features = Vector{Matrix{Float32}}()
    pos_agt = Vector{Vector{Float32}}()
    labels = Vector{Vector{Float32}}()
    for lanelet in values(lanelet_roadway.lanelets)
        curve = lanelet.curve
        push!(agt_features, hcat([curve[1].pos.x, curve[1].pos.y], [curve[2].pos.x, curve[2].pos.y]))
        push!(pos_agt, [curve[2].pos.x, curve[2].pos.y])
        push!(labels, [curve[end].pos.x, curve[end].pos.y])
    end

    agt_features = cat(agt_features..., dims=3)
    pos_agt = hcat(pos_agt...)
    labels = hcat(labels...)

    if save_features
        # Save the agent features
        cache_path = joinpath(@__DIR__, "../res/agent_features.jld2")
        @info "Saving agent features to $(cache_path)"
        jldsave(cache_path, agt_features=agt_features, labels=labels)
    end

    return agt_features, pos_agt, labels
end

"""
Prepare map features stored in polyline level GNNGraph(fulled-connected graph)
    vector features: (4, num_vectors) -> polyline_graphs.x
    routing_graph -> g_heteromap
"""
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
        v == 14 && @assert lanelet.tag == LaneletTag(1707, false)
        
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
        (:lanelet, :adj_right, :lanelet) => extract_gml_src_dst(g_meta, "AdjacentRight")
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
