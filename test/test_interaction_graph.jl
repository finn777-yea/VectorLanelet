using Test
using VectorLanelet: create_filtered_interaction_graph, InteractionGraphModel
using GraphNeuralNetworks
using Random
using Flux
@testset "create_filtered_interaction_graph" begin
    # Test more complex batch case with different sized samples
"""
Sample 1: 3 agents and 4 contexts
                    y
                    ^
                    |
                    |
                    |
---c4---c3---c2---c1/a1---a2---a3--->  x
                    |
                    |

Sample 2: 5 agents and 2 contexts
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
    @test g_agent isa GNNGraph
    @test edge_index(g_agent) isa Tuple

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

