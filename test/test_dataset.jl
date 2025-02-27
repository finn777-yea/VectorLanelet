using VectorLanelet
using Graphs
using GraphNeuralNetworks

lanelet_roadway, g_meta = VectorLanelet.load_map_data()
polyline_graphs, g_heteromap, llt_pos, μ, σ = VectorLanelet.prepare_map_features(lanelet_roadway, g_meta)
size(llt_pos, 2) == g_heteromap.num_nodes[:lanelet]

# Test the node order in polyline_graphs and g_heteromap
polyline_graphs.gdata.id

edge_index(g_heteromap, (:lanelet, :right, :lanelet))

# Test clustered agt features
agt_features, agt_pos_end = VectorLanelet.prepare_agent_features(lanelet_roadway)
agt_pos_end[1]
length(agt_features)
cat(agt_features..., dims=3)

using MLUtils
dataloader = DataLoader(agt_features, batchsize=5, shuffle=true)

for batch in dataloader
    @show size(batch)
    @show size(batch[1])
end


