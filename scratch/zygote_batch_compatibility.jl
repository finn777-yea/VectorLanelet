using Zygote
using GNNGraphs
using Flux
using CUDA
# Zygote.@nograd batch

g1 = GNNGraph([1,2,3], [2,3,4])
g2 = GNNGraph([1,2,3], [2,4,5])

function test_fn(x)
    graphs = [g1, g2]
    gs = batch(graphs)

    @show typeof(gs)
    return sum(gs.num_nodes)
end

# This will error
gradient(test_fn, 1.0)

