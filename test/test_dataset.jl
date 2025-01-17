using VectorLanelet

lanelet_roadway, g_meta = VectorLanelet.load_map_data()
polyline_graphs, g_heteromap, llt_pos, μ, σ = VectorLanelet.prepare_map_features(lanelet_roadway, g_meta)
size(llt_pos, 2) == g_heteromap.num_nodes[:lanelet]
