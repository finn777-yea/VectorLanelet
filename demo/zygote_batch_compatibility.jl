using Zygote
using GNNGraphs

# Create some simple graphs
g1 = GNNGraph([1,2,3], [2,3,4])
g2 = GNNGraph([1,2,3], [2,4,5])

# Try to differentiate through batching
function test_fn(x)
    graphs = [g1, g2]
    batched = batch(graphs)
    return sum(batched.num_nodes)
end

# This will error
gradient(test_fn, 1.0)