using Test
using VectorLanelet: load_map_data, prepare_agent_features, create_filtered_interaction_graphs, InteractionGraphModel
using GraphNeuralNetworks
using Random

"""
    Position of agents and lanelets:
                         y
                         ^
                         |
                         |                   
                         |
---l5---l4---l3---l2---l1/a1---a2---a3---a4--->  x
                         |
"""

@testset "Interaction Graph" begin
    agt_pos = [Float32[i, 0] for i in 0:3]
    llt_pos = [Float32[-i, 0] for i in 0:4]

    agt_pos = hcat(agt_pos...)
    llt_pos = hcat(llt_pos...)
    n_in = 64        # Number of features for each node
    e_in = 1
    out_dim = 64
    num_heads = 4
    num_layers = 2
    num_agents = size(agt_pos, 2)
    num_llts = size(llt_pos, 2)
    agent_emb = rand(Float32, n_in, num_agents)
    map_emb = rand(Float32, n_in, num_llts)
    dist_thrd = 2.0

    @testset "create_filtered_interaction_graphs with features" begin
        g_agent = create_filtered_interaction_graphs(agt_pos, llt_pos, dist_thrd)
        @test all(g_agent.edata.d .< dist_thrd)
        @assert typeof(g_agent.edata.d) == Matrix{Float32}
        @show edge_index(g_agent)
    end

    # @testset "InteractionGraphModel" begin
    #     interaction = InteractionGraphModel(n_in, e_in, out_dim, num_heads=num_heads)
    #     res = interaction(agent_emb, agt_pos, map_emb, llt_pos, 100.0)
    # end
end
