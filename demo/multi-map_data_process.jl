"""
    Processed data for training
        - agent features and labels
        - vector features
        - bachted fully connected graph: polyline_graphs
        - heterogenous graph: g_heteromap
"""

using Lanelet2
using AutomotiveSimulator
using NestedGraphsIO


"""
Load lanelet roadway for each map, store LaneletRoadway in a dictionary
"""
function load_lanelet_roadways(map_dir::String, save::Bool=false, save_dir::String="")
    # Load the lanelet map
    # TODO: Is it necessary to asign different origin for each map
    origin = Lanelet2.Io.Origin(50.99, 6.90)
    map_files = readdir(map_dir)
    lanelet_roadways = Dict{String, LaneletRoadway}()
    for map_file in map_files
        map_file_path = joinpath(map_dir, map_file)
        projector = Projection.UtmProjector(origin)
        llmap = Lanelet2.Io.load(map_file_path, projector)

        # Use passable lanelet submap
        traffic_rules = TrafficRules.create(TrafficRules.Locations.Germany, TrafficRules.Participants.Vehicle)
        rg = RoutingGraph(llmap, traffic_rules)
        llmap = rg.passableLaneletSubmap()
        lanelet_roadways[map_file] = LaneletRoadway(llmap)
    end

    return lanelet_roadways
end

"""
Load g_meta from gml files, stores the routing graph for each map
"""
function load_g_meta(gml_dir::String, save::Bool=false, save_dir::String="")
    g_meta = Dict{String, MetaDiGraph}()
    for gml_file in gml_files
        gml_file_path = joinpath(gml_dir, gml_file)
        g_meta[map_file] = open(gml_file_path, "r") do io
            NestedGraphsIO.loadgraph(io, "G", GraphMLFormat(), MGFormat())
        end
    end

    if save
        # Save the map features
        save_file = joinpath(save_dir, "g_meta.jld2")
        @info "Saving g_meta to $(save_file)"
        jldsave(save_file, g_meta=g_meta)
    end
    return g_meta
end

"""
Prepare agent features and labels from lanelet centerlines
    - agent features: (2, 2, B)   (channels, time_step, batch_size)
    - labels: (2, B)            (channels, batch_size)

"""
function prepare_agent_features(lanelet_roadways::Dict{String, LaneletRoadway}, save_features::Bool=false)
    # Initialize dictionaries to store features for each map
    map_agt_features = Dict{String, Array{Float32, 3}}()
    map_labels = Dict{String, Matrix{Float32}}()
    
    # Process each map separately
    for (map_name, roadway) in lanelet_roadways
        agt_features = Vector{Matrix{Float32}}()
        pos_agt = Vector{Vector{Float32}}()
        labels = Vector{Vector{Float32}}()
        
        for lanelet in values(roadway.lanelets)
            curve = lanelet.curve
            push!(agt_features, hcat([curve[1].pos.x, curve[1].pos.y], [curve[2].pos.x, curve[2].pos.y]))
            push!(pos_agt, [curve[2].pos.x, curve[2].pos.y])
            push!(labels, [curve[end].pos.x, curve[end].pos.y])
        end

        # Store concatenated features for this map
        map_agt_features[map_name] = cat(agt_features..., dims=3)
        map_labels[map_name] = Float32.(hcat(labels...))
    end

    if save_features
        cache_path = joinpath(@__DIR__, "../res/agent_features.jld2")
        @info "Saving agent features to $(cache_path)"
        jldsave(cache_path, map_agt_features=map_agt_features, map_labels=map_labels)
    end

    return map_agt_features, map_labels
end