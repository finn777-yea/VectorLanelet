using GraphNeuralNetworks
using VectorLanelet: load_map_data, prepare_agent_features, create_filtered_interaction_graphs, InteractionGraphModel
using LinearAlgebra
using Graphs.SimpleGraphs
using Flux

# Check star_digraph connectivity
g = star_digraph(3) |> GNNGraph
src = edge_index(g)[1]
dst = edge_index(g)[2]

x = Float32[
    1.0 0.0 0.0;
    0.0 1.0 0.0;
    0.0 0.0 1.0;
]

@assert all(src .== 1)

g.ndata.x = x
gcn = GCNConv(3=>3)
@show gcn(g, g.ndata.x)

g_undirected = star_graph(3) |> GNNGraph
edge_index(g_undirected)
@assert g_undirected.num_edges == 2g.num_edges

g_undirected.ndata.x = x
@show gcn(g_undirected, g_undirected.ndata.x)

