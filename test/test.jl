using Test
using VectorLanelet
using GraphNeuralNetworks
using CUDA
using Flux

config = Dict{String, Any}()
config["n_actor"] = 128         # the output feature dim each actor
config["n_map"] = 128           # the output feature dim each lane
config["num_scales"] = 6
config["din_actor"] = 2

@testset "create_residual_block" begin
    res_block = create_residual_block(3, 32, stride=1)
    @test res_block(rand(Float32, 10, 3, 4)) |> size == (10,32,4)
end
@testset "create_group_block" begin
    group_block1 = create_group_block(1, 3, 32)
    @test group_block1(rand(Float32, 10, 3, 4)) |> size == (10,32,4)
    group_block2 = create_group_block(2, 32, 64)
    @test group_block2(rand(Float32, 10, 32, 4)) |> size == (5,64,4)
end

@testset "create_node_encoder" begin
    node_encoder = create_node_encoder(3, 32)
    @test node_encoder(rand(Float32, 3, 10)) |> size == (32,10)
end

@testset "PolylineEncoder" begin
    # Single graph
    g = GNNGraph([1,2], [2,3], ndata=rand(Float32, 4, 3))
    output_channel = 32
    μ = rand(Float32, 2)
    σ = rand(Float32, 2)
    pline = PolylineEncoder(4, output_channel, μ, σ)
    @test pline(g, g.ndata.x) |> size == (output_channel,1)

    # Batch of graphs
    data = [rand_graph(3,6, ndata=(;x=rand(Float32, 4, 3))) for _ in 1:10]
    g = batch(data)
    @assert g.num_graphs == 10
    pline = PolylineEncoder(4, output_channel, μ, σ)
    @test pline(g, g.ndata.x) |> size == (output_channel, 10)
end

@testset "ActorNet_Simp" begin
    agt_features = rand(Float32, 2, 10, 32)
    μ = rand(Float32, 2)
    σ = rand(Float32, 2)
    actornet = VectorLanelet.ActorNet_Simp(2, [64, 128], μ, σ)

    @test actornet.agt_preprocess(agt_features) |> size == (10, 2, 32)

    @test actornet.groups[1](actornet.agt_preprocess(agt_features)) |> size == (10, 64, 32)
    @test actornet.groups[2](
        actornet.groups[1](
            actornet.agt_preprocess(agt_features)
        )
    ) |> size == (5, 128, 32)

    @test actornet.lateral[1](
        actornet.groups[1](
            actornet.agt_preprocess(agt_features)
        )
    ) |> size == (10, 128, 32)
    
    @test actornet.lateral[2](
        actornet.groups[2](
            actornet.groups[1](
                actornet.agt_preprocess(agt_features)
            )
        )
    ) |> size == (5, 128, 32)
    @test actornet(agt_features) |> size == (128, 32)
end

@testset "MapEncoder" begin
    mapenc = VectorLanelet.MapEncoder(64, 64, 4)
    left_rel = (:lanelet, :left, :lanelet)
    right_rel = (:lanelet, :right, :lanelet)
    pre_rel = (:lanelet, :pre, :lanelet)
    suc_rel = (:lanelet, :suc, :lanelet)
    adj_left_rel = (:lanelet, :adj_left, :lanelet)
    adj_right_rel = (:lanelet, :adj_right, :lanelet)

    g = GNNHeteroGraph(
        left_rel => ([1,2,2,3], [1,3,2,9]),
        right_rel => ([1,1,2,3], [7,13,5,7]),
        pre_rel => ([1,3,4,5], [7,15,2,3]),
        suc_rel => ([1,1,2,3], [3,10,12,9]),
        adj_left_rel => ([1,1,2,3], [3,10,12,9]),
        adj_right_rel => ([1,1,2,3], [3,10,12,9])
    )
    g[:lanelet].x = rand(Float32, 64, 15)
    @test size(mapenc(g, g[:lanelet].x)) == (64, 15)
end

@testset "LaneletPredictor" begin
    
    # Test input dimensions
    num_agts = 32
    num_timesteps = 2
    agt_features = rand(Float32, 2, num_timesteps, num_agts)
    
    g1 = GNNGraph([1,2], [2,3], ndata=rand(Float32, 4, 3))
    g2 = GNNGraph([1,2,4], [2,3,1], ndata=rand(Float32, 4, 4))
    g_polyline = batch([g1, g2])
    vector_features = g_polyline.ndata.x
    g_heteromap = GNNHeteroGraph(
        (:lanelet, :left, :lanelet) => ([1], [2]),
        (:lanelet, :right, :lanelet) => ([1], [2]),
        (:lanelet, :pre, :lanelet) => ([1], [2]),
        (:lanelet, :suc, :lanelet) => ([1], [2]),
        (:lanelet, :adj_left, :lanelet) => ([1], [2]),
        (:lanelet, :adj_right, :lanelet) => ([1], [2])
        )
        
    # Test forward pass
    μ, σ = VectorLanelet.calculate_mean_and_std(vector_features[1:2, :]; dims=2)
    model = LaneletPredictor(μ, σ)
    out = model(agt_features, g_polyline, g_heteromap)
    
    # Test output dimensions
    @test size(out) == (2, num_agts)
end