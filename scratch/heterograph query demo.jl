using GraphNeuralNetworks

"""
    check if the batching of heterograph works
"""

# Define the relationships
left_rel = (:lanelet, :left, :lanelet)
right_rel = (:lanelet, :right, :lanelet)
pre_rel = (:lanelet, :pre, :lanelet)
suc_rel = (:lanelet, :suc, :lanelet)

# g1: 15 nodes
g1 = GNNHeteroGraph(
        left_rel => ([1,2,3,2], [1,3,9,1]),
        right_rel => ([1,1,2,3], [7,13,5,7]),
        pre_rel => ([1,3,4,5], [7,15,2,3]),
        suc_rel => ([1,1,2,3], [3,10,12,9])
    )
g1[:lanelet].x = rand(Float32, 32, 15)
g1[:lanelet].ctr = rand(1.0:5.0, 2, 15)

# g2: 6 nodes
g2 = GNNHeteroGraph(
        left_rel => ([1,3,4], [2,4,5]),
        right_rel => ([2,3,5], [1,4,6]),
        pre_rel => ([1,2,3], [4,5,6]),
        suc_rel => ([4,5,6], [1,2,3])
    )
g2[:lanelet].x = rand(Float32, 32, 6)
g2[:lanelet].ctr = rand(1.0:5.0, 2, 6)

g12 = batch([g1,g2])

edge_index(g2, left_rel)
edge_index(g12, left_rel)

# TODO: maybe turn the indicator to indices
graph_indicator(g12)[:lanelet]

# ------ Pairwise distances -------

# 3 and 2 agts in 2 batches
agt_feat = rand(Float32, 32, 5)
agt_idc = [[1,2,3], [4,5]]
batch_size = length(agt_idc)
agt_ctr = [rand(1.0:5.0, 2, 3), rand(1.0:5.0, 2, 2)]
node_ctr = [g1[:lanelet].ctr, g2[:lanelet].ctr]

for i in 1:batch_size
    @show dist = reshape(agt_ctr[i], (2,1,:)) .- reshape(node_ctr[i], (2,:,1)) 
    @show dist |> size
end