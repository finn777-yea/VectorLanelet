using Test
using VectorLanelet: create_filtered_interaction_graph, InteractionGraphModel
using GraphNeuralNetworks
using Random
using Flux
@testset "create_filtered_interaction_graph" begin
    # Test more complex batch case with different sized samples
"""
Sample 1: 3 agents and 4 contexts
Match(dist_thrd=1.0): a1/c1, a1/c2, a2/c1
                    y
                    ^
                    |
                    |
                    |
---c4---c3---c2---c1/a1---a2---a3--->  x
                    |
                    |

Sample 2: 5 agents and 2 contexts
Match(dist_thrd=1.0): a1/c2
            y
            ^
            |
            |
            a1---a2---a3---a4---a5
            |
            |
-------c1---c2------------------->  x
"""
    agt_pos1 = Float32[
        0 1 2;    # x coordinates
        0 0 0     # y coordinates
    ]
    ctx_pos1 = Float32[
        0 -1 -2 -3;  # x coordinates
        0  0  0  0   # y coordinates
    ]

    agt_pos2 = Float32[
        0 1 2 3 4;  # x coordinates
        1 1 1 1 1   # y coordinates
    ]
    ctx_pos2 = Float32[
        -1 0;      # x coordinates
         0 0       # y coordinates
    ]

    # Test CPU batch processing
    agt_pos = [agt_pos1, agt_pos2]
    ctx_pos = [ctx_pos1, ctx_pos2]
    g_agent = create_filtered_interaction_graph(agt_pos, ctx_pos, 2.0)

    # Test GPU batch processing
    agt_pos = [agt_pos1, agt_pos2] |> gpu
    ctx_pos = [ctx_pos1, ctx_pos2] |> gpu
    g_agent = create_filtered_interaction_graph(agt_pos, ctx_pos, 1.0)
    @test g_agent isa GNNGraph
    @test g_agent.edata.e isa AbstractMatrix

    # Test edge connections for distance threshold 1.0
    expected_src = [1, 1, 2, 4]
    expected_dst = [9, 9, 10, 14]
    src, dst = edge_index(g_agent)
    @test src |> sort == expected_src
    @test dst |> sort == expected_dst
    @show g_agent.edata.e
    @show g_agent.num_nodes
end

# No edge in the first scenario
agt_pos1 = Float32[
    5 6 7;    # x coordinates
    0 0 0     # y coordinates
]
ctx_pos1 = Float32[
    0 -1 -2 -3;  # x coordinates
    0  0  0  0   # y coordinates
]

# Edges in the second scenario
# Match(dist_thrd=1.0): a3/c1, a4/c2
agt_pos2 = Float32[
    0 1 2 3 4;  # x coordinates
    1 1 1 1 1   # y coordinates
]
ctx_pos2 = Float32[
    2 3;      # x coordinates
    0 0       # y coordinates
]
agts_pos = [agt_pos1, agt_pos2] |> gpu
ctxs_pos = [ctx_pos1, ctx_pos2] |> gpu
g_agent = create_filtered_interaction_graph(agts_pos, ctxs_pos, 1.0)
@assert edge_index(g_agent) == ([6,7], [13,14])
@show g_agent.edata.e



# ------ InteractionGraphModel ------
g_agent = create_filtered_interaction_graph([agt_pos1], [ctx_pos1], 1.0)
n_in = 64
e_in = 2
out_dim = 64
num_heads = 2

# using edge features and self_loop not compatible
model = InteractionGraphModel(n_in, e_in, out_dim, num_heads=num_heads, self_loop=true)

# Test forward pass with no interactions
agt_features = randn(Float32, n_in, 3) # 64 features, 3 agents
ctx_features = randn(Float32, n_in, 4) # 64 features, 4 context nodes
agt_pos = [agt_pos1]
ctx_pos = [ctx_pos1]
dist_thrd = 1.0f0

out = model((agt_features, agt_pos, ctx_features, ctx_pos, dist_thrd))
