"""
    Combine VectorSubGraph and MapNet to encode a single map
"""

using Lanelet2
using Lanelet2.Routing
using Lanelet2.Core
using AutomotiveSimulator
using Graphs
using MetaGraphs
using NestedGraphsIO
using GraphNeuralNetworks
using GraphIO.GraphML
using VectorLanelet
# ------Data preparation -------

# Load the map
example_file = joinpath(@__DIR__, "../res","location0.osm")
projector = Projection.UtmProjector(Lanelet2.Io.Origin(49, 8.4))
llmap = Lanelet2.Io.load(example_file, projector)

# Use passable lanelet submap instead of lanelet map
traffic_rules = TrafficRules.create(TrafficRules.Locations.Germany, TrafficRules.Participants.Vehicle)
rg = RoutingGraph(llmap, traffic_rules)
llmap = rg.passableLaneletSubmap()
lanelet_roadway = LaneletRoadway(llmap)

# Load the g_meta MetaDiGraph
gml_file_path = joinpath(@__DIR__, "../res/location0.osm.gml")
g_meta = open(gml_file_path, "r") do io
    NestedGraphsIO.loadgraph(io, "G", GraphMLFormat(), MGFormat())
end
@assert typeof(g_meta) == MetaDiGraph{Int64, Float64}

# Traverse all the lanelets according to vertex order
# Mapping: v -> lanelet_id
lanelet_graphs = GNNGraph[]
llt_pos = []
for v in 1:nv(g_meta)
    lanelet_attr = Lanelet2.extract_graphml_attributes(get_prop(g_meta, v, :info))
    lanelet_tag = LaneletTag(lanelet_attr.lanelet_id, lanelet_attr.inverted)
    lanelet = lanelet_roadway[lanelet_tag]
    
    centerline = lanelet.curve
    num_nodes = length(centerline) - 1
    g_fc = complete_digraph(num_nodes) |> GNNGraph
    
    # Iterate over points in centerline to get vector-level features
    llt_features = []
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
        push!(llt_features, [start_x, start_y, end_x, end_y])
    end
    # Convert to matrix format
    llt_features = reduce(hcat, llt_features)       # feature matrix:(4, num_nodes)
    g_fc.ndata.x = llt_features
    push!(lanelet_graphs, g_fc)
end
g_all = batch(lanelet_graphs)


# Construct HeteroGNNGraph based on g_meta
left_rel = (:lanelet, :left, :lanelet)
right_rel = (:lanelet, :right, :lanelet)
suc_rel = (:lanelet, :suc, :lanelet)
adj_left_rel = (:lanelet, :adj_left, :lanelet)
adj_right_rel = (:lanelet, :adj_right, :lanelet)

g_hetero = GNNHeteroGraph(
    right_rel => extract_gml_src_dst(g_meta, "Right"),
    left_rel => extract_gml_src_dst(g_meta, "Left"),
    suc_rel => extract_gml_src_dst(g_meta, "Successor"),
    adj_left_rel => extract_gml_src_dst(g_meta, "AdjacentLeft"),
    adj_right_rel => extract_gml_src_dst(g_meta, "AdjacentRight")
    )
    
    
# ------ Map encoder --------

in_channel = 4      # [ds, de]
vsg = VectorSubGraph(in_channel)

config = Dict{String, Any}()
config["n_map"] = 128
config["num_scales"] = 6
mapnet = MapNet(config)
    
emb_lanelets = vsg(g_all, g_all.x)
@assert size(emb_lanelets, 2) == g_all.num_graphs

g_hetero[:lanelet].x = emb_lanelets
emb_map = mapnet(g_hetero)
@assert size(emb_map) == (128, 105)