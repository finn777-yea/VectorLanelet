using VectorLanelet
using Graphs
using GraphNeuralNetworks

lanelet_roadway, g_meta = VectorLanelet.load_map_data()
polyline_graphs, g_heteromap, μ, σ = VectorLanelet.prepare_map_features(lanelet_roadway, g_meta)
@show μ, σ

# Node index same in the gml file
polyline_graphs
@assert polyline_graphs.gdata.id[73] == 1747
@assert polyline_graphs.gdata.id[82] == 1762

polyline_graphs_part = getgraph(polyline_graphs, 1:2)
polyline_graphs_part.x

pline = PolylineEncoder(4, 8, μ, σ)

clusters= graph_indicator(polyline_graphs_part)
max_pool = GlobalPool(max)
pooled_vecs = max_pool(polyline_graphs_part, polyline_graphs_part.x)  # Pooling within each graph
pooled_vecs[:, clusters]    # Duplication

emb_vecs_part = pline(polyline_graphs_part, polyline_graphs_part.x)
emb_vecs = pline(polyline_graphs, polyline_graphs.x)

# Node index in g_hetero the same as in gml file
suc_relation = (:lanelet, :suc, :lanelet)
left_relation = (:lanelet, :left, :lanelet)
@assert has_edge(g_heteromap, suc_relation, 78, 82)
@assert has_edge(g_heteromap, left_relation, 7, 8)



map_encoder = MapEncoder(8, 8, 1)
emb_map = map_encoder(g_heteromap, emb_vecs)



