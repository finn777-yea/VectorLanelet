using GraphNeuralNetworks
using Flux
using Graphs

n_in = 8
e_in = 2
head_dim = 4
num_heads = 2

src = [1,2,3]
dst = [2,3,4]

gat_no_slp = GATConv(n_in=>head_dim, heads=num_heads, add_self_loops=false, bias=true)

g_out = GNNGraph(src, dst, dir=:out; ndata=rand(Float32, n_in, 4), edata=rand(Float32, e_in, 3))   # 1 has only outcoming edge
@assert !is_bidirected(g_out)
@assert !has_self_loops(g_out)
adjacency_list(g_out)
@assert inneighbors(g_out, 1) == []
@assert outneighbors(g_out, 1) == [2]

x = gat_no_slp(g_out, g_out.x)
@assert x[:,1] == zeros(Float32, head_dim*num_heads)

g_in = GNNGraph(src, dst, dir=:in; ndata=rand(Float32, n_in, 4), edata=rand(Float32, e_in, 3))
@assert edge_index(g_in) == edge_index(g_out)
@assert inneighbors(g_in, 1) == []      # the underlying graph structure is not changed compared to g_out

x = gat_no_slp(g_in, g_in.x)
# The dir parameter doesn't create new edges
# it just reverses the direction of message flow along existing edges.
# Since node 1 has no incoming edges in the original graph, it receives no messages in either direction setting.
@assert x[:,1] == zeros(Float32, head_dim*num_heads)

src_with_self_loops = [1,1,2,2,3,3,4]
dst_with_self_loops = [1,2,2,3,3,4,4]
g_with_self_loops = GNNGraph(src_with_self_loops, dst_with_self_loops, dir=:in; ndata=rand(Float32, n_in, 4), edata=rand(Float32, e_in, length(src_with_self_loops)))   # 1 is isolated
inneighbors(g_with_self_loops, 1)
outneighbors(g_with_self_loops, 1)

@assert has_self_loops(g_with_self_loops)
x = gat_no_slp(g_with_self_loops, g_with_self_loops.x)
@assert x[:,1] != zeros(Float32, head_dim*num_heads)