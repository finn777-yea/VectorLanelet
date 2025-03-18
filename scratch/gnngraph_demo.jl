using GraphNeuralNetworks
using Flux
using Graphs

n_in = 8
e_in = 2
head_dim = 4
num_heads = 2

src = [1,1,1,1]
dst = [2,3,4,5]

gat_no_slp = GATConv(n_in=>head_dim, heads=num_heads, add_self_loops=false, bias=true)

g_out = GNNGraph(src, dst, dir=:out; ndata=rand(Float32, n_in, 5), edata=rand(Float32, e_in, length(src)))   # 1 has only outcoming edge
@assert !is_bidirected(g_out)
@assert !has_self_loops(g_out)
adjacency_list(g_out)
@assert inneighbors(g_out, 1) == []
@assert outneighbors(g_out, 1) == [2,3,4,5]

x = gat_no_slp(g_out, g_out.x)
@assert x[:,1] == zeros(Float32, head_dim*num_heads)

# -------- THIS IS CORRECT!!!!!!!--------
# The expected-to-be-updated nodes should be the dst
g_out_rev = GNNGraph(dst, src, dir=:out; ndata=rand(Float32, n_in, 5), edata=rand(Float32, e_in, length(src)))
x = gat_no_slp(g_out_rev, g_out_rev.x)

# Seems wrong for non-bidirected graph
g_in = GNNGraph(src, dst, dir=:in; ndata=rand(Float32, n_in, 5), edata=rand(Float32, e_in, length(src)))
@assert is_directed(g_in)
@assert edge_index(g_in) == edge_index(g_out)
@assert inneighbors(g_in, 1) == []      # the underlying graph structure is not changed compared to g_out

x = gat_no_slp(g_in, g_in.x)
@assert x[:,1] == zeros(Float32, head_dim*num_heads)

# Bidirected graph
g_bi = to_bidirected(g_in)
@assert is_bidirected(g_bi)
edge_index(g_bi) == ([1, 2, 2, 3, 3, 4], [2, 1, 3, 2, 4, 3])
g_bi.ndata.x = rand(Float32, n_in, 4)
x = gat_no_slp(g_bi, g_bi.x)
# Now the msg is able to pass to node 1
@assert x[:,1] != zeros(Float32, head_dim*num_heads)

src_with_self_loops = [1,1,2,2,3,3,4]
dst_with_self_loops = [1,2,2,3,3,4,4]
g_with_self_loops = GNNGraph(src_with_self_loops, dst_with_self_loops, dir=:in; ndata=rand(Float32, n_in, 4), edata=rand(Float32, e_in, length(src_with_self_loops)))   # 1 is isolated
inneighbors(g_with_self_loops, 1)
outneighbors(g_with_self_loops, 1)

@assert has_self_loops(g_with_self_loops)
x = gat_no_slp(g_with_self_loops, g_with_self_loops.x)
@assert x[:,1] != zeros(Float32, head_dim*num_heads)