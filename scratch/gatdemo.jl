using JLD2
using GraphNeuralNetworks
using Flux
using Graphs

g, node_features = JLD2.load(joinpath(@__DIR__, "node_features.jld2"), "g", "node_features")
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


# Adding self-loops
n_in = 8
e_in = 2
head_dim = 4
num_heads = 2

src1 = [1,2,5]
dst1 = [2,3,6]
gat_no_slp = GATConv((n_in,e_in)=>head_dim, heads=num_heads, add_self_loops=false, bias=true)

# Use function add_self_loops
g_no_edge_data = GNNGraph(src, dst, dir=:in; ndata=rand(Float32, n_in, 4))
g_added_sl = add_self_loops(g_no_edge_data)
@assert has_self_loops(g_added_sl)
g_added_sl.edata.e = rand(Float32, e_in, 7)
g_added_sl.edata = DataStore(7, Dict(:e => rand(Float32, e_in, 7)))

edge_index(g_added_sl)
x = gat_no_slp(g_added_sl, g_added_sl.x, g_added_sl.edata.e)

# Manually change src and dst
src_with_self_loops = [1,1,2,2,3,3,4]
dst_with_self_loops = [1,2,2,3,3,4,4]
g_with_self_loops = GNNGraph(src_with_self_loops, dst_with_self_loops, dir=:in; ndata=rand(Float32, n_in, 4), 
                edata=rand(Float32, e_in, length(src_with_self_loops)))
@assert has_self_loops(g_with_self_loops)
x = gat_no_slp(g_with_self_loops, g_with_self_loops.x, g_with_self_loops.edata.e)

# Use add_edges
g = GNNGraph(src1, dst1, dir=:in; ndata=rand(Float32, n_in, 6), edata=rand(Float32, e_in, length(src1)))
self_loop_nodes = [1:g.num_nodes;]
g_added_edges = add_edges(g, self_loop_nodes, self_loop_nodes, edata=zeros(Float32, e_in, length(self_loop_nodes)))
has_self_loops(g_added_edges)
edge_index(g_added_edges)

x = gat_no_slp(g_added_edges, g_added_edges.x, g_added_edges.edata.e)