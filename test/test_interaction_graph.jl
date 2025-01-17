using Test
using VectorLanelet: load_map_data, prepare_agent_features, create_filtered_interaction_graphs, InteractionGraphModel
using GraphNeuralNetworks
using Random


@testset "Interaction Graph" begin
    lanelet_roadway, g_meta = load_map_data()
    agt_features, agt_pos, labels = prepare_agent_features(lanelet_roadway)

    llt_pos = rand(Float32, (2, 128))
    n_in = 64        # Number of features for each node
    e_in = 1
    out_dim = 64
    num_heads = 4
    num_layers = 2
    num_agents = size(agt_pos, 2)
    agent_emb = rand(Float32, n_in, num_agents)
    map_emb = rand(Float32, n_in, size(llt_pos, 2))

    @testset "create_filtered_interaction_graphs with features" begin
        g_agent = create_filtered_interaction_graphs(agt_pos, llt_pos, 100.0)
        @test all(g_agent.edata.d .< 100.0)
        @test typeof(g_agent.edata.d) == Matrix{Float32}
    end
    
    # @testset "create_filtered_interaction_graphs without features" begin
        #     g_agent = create_filtered_interaction_graphs(agt_pos, llt_pos, 100.0)
        #     @test g_agent.num_graphs == num_agents
        #     @test all(g_agent.edata.d .< 100.0)
        # end
        
    # DEPRECATED!!!
    # @testset "prepare_interaction_feautures" begin
    #     using VectorLanelet: prepare_interaction_feautures
    #     emb_agt = [0.1 0.2; 0.1 0.2]
    #     emb_ctx = [1.0 2.0 3.0; 1.0 2.0 3.0]

    #     agt_g1 = star_digraph(3) |> GNNGraph
    #     agt_g1.ndata.ind = [1, 1, 2]    # 1 is agt, 2, 3 are ctx nodes
    #     agt_g2 = star_digraph(2) |> GNNGraph
    #     agt_g2.ndata.ind = [2, 3]    # 2 is agt, 3 is ctx node
    #     g_all = batch([agt_g1, agt_g2])

        
    #     expected_ndata = Float32[0.1 1.0 2.0 0.2 3.0; 0.1 1.0 2.0 0.2 3.0]
    #     res = prepare_interaction_feautures(emb_agt, emb_ctx, g_all)
    #     @show eltype(res)
    #     @test res == expected_ndata
    # end

    @testset "InteractionGraphModel" begin
        interaction = InteractionGraphModel(n_in, e_in, out_dim, num_heads=num_heads)
        res = interaction(agent_emb, agt_pos, map_emb, llt_pos, 100.0)
    end
end
