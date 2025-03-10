using JLD2
using GraphNeuralNetworks
using Flux

g, node_features = JLD2.load(joinpath(@__DIR__, "../node_features.jld2"), "g", "node_features")
src, dst = GraphNeuralNetworks.edge_index(g)
isolated_nodes = setdiff(1:g.num_nodes, dst)
g_empty = GNNGraph(src, dst)

n_in = size(node_features, 1)
e_in = size(g.edata.e, 1)
head_dim = 32
num_heads = 2
ln = LayerNorm(head_dim*num_heads)

# Self-loops
gat_with_self_loops = GATConv(n_in=>head_dim, heads=num_heads, add_self_loops=true, bias=true)
x = gat_with_self_loops(g_empty, node_features)
x = ln(x)

# No self-loops
gat_no_self_loops = GATConv(n_in=>head_dim, heads=num_heads, add_self_loops=false, bias=true)
x = gat_no_self_loops(g_empty, node_features)
x[:,isolated_nodes] == zeros32(64,1305)
x = ln(x)

# GATv2Conv
gatv2_conv = GATv2Conv((n_in,e_in)=>head_dim, heads=num_heads, add_self_loops=true, bias=true)