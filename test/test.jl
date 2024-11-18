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

@testset "Res1d" begin
    res1d = Res1d(3, 32)
    @test res1d(rand(Float32, 10, 3, 4)) |> size == (10,32,4)
end

# @testset "ActorNet" begin
#     actornet = ActorNet(config)
#     batch_size = 32
#     n_actor = config["n_actor"]
#     input = rand(Float32, 20, 2, batch_size)      # input: (timesteps, features, batch)
    
#     # Test the output of each group
#     @test size(actornet.groups[1](input)) == (20, 32, batch_size)
#     @test size(actornet.groups[2](actornet.groups[1](input))) == (10, 64, batch_size)
#     @test size(actornet.groups[3](actornet.groups[2](actornet.groups[1](input)))) == (5, 128, batch_size)
    
#     # Test the output of lateral connections
#     @test size(actornet.lateral[1](actornet.groups[1](input))) == (20, 128, batch_size)
#     @test size(actornet.lateral[2](actornet.groups[2](actornet.groups[1](input)))) == (10, 128, batch_size)
#     @test size(actornet.lateral[3](actornet.groups[3](actornet.groups[2](actornet.groups[1](input))))) == (5, 128, batch_size)
    
#     # Test the final output
#     @test size(actornet(input)) == (n_actor, batch_size)
# end


# @testset "MapNet" begin
#     @testset "Random input" begin
#         left_rel = (:lanelet, :left, :lanelet)
#         right_rel = (:lanelet, :right, :lanelet)
#         pre_rel = (:lanelet, :pre, :lanelet)
#         suc_rel = (:lanelet, :suc, :lanelet)

#         g = GNNHeteroGraph(
#         left_rel => ([1,2,2,3], [1,3,2,9]),
#         right_rel => ([1,1,2,3], [7,13,5,7]),
#         pre_rel => ([1,3,4,5], [7,15,2,3]),
#         suc_rel => ([1,1,2,3], [3,10,12,9]))

#         g[:lanelet].feat = rand(Float32, 64, 15)
#         g[:lanelet].ctrs = rand(Float32, 64, 15)
        
#         mapnet = MapNet(config)
        
#         # Test input module
#         input_feat = mapnet.input(g[:lanelet].ctrs)
#         @test size(input_feat) == (128, 15)
        
#         # Test segmentation module
#         seg_feat = mapnet.seg(g[:lanelet].feat)
#         @test size(seg_feat) == (128, 15)
        
#         # Test fusion module
#         feat, _ = mapnet(g)
#         @test size(feat) == (128, 15)     # 15 nodes in the graph
#     end
# end

@testset "LaneletPredictor" begin
    actor_config = Dict{String, Any}(
        "n_actor" => 128,
        "din_actor" => 2
    )
    
    map_config = Dict{String, Any}(
        "n_map" => 128,
        "num_scales" => 6
    )
    
    model = LaneletPredictor(actor_config, map_config)
    
    # Test input dimensions
    num_agts = 32
    num_timesteps = 2
    agt_features = rand(Float32, num_timesteps, 2, num_agts)  # (timesteps, features, batch)
    
    g1 = GNNGraph([1,2], [2,3], ndata=rand(Float32, 4, 3))
    g2 = GNNGraph([1,2,4], [2,3,1], ndata=rand(Float32, 4, 4))
    g_all = batch([g1, g2])
    g_hetero = GNNHeteroGraph(
        (:lanelet, :left, :lanelet) => ([1], [2]),
        (:lanelet, :right, :lanelet) => ([1], [2]),
        (:lanelet, :pre, :lanelet) => ([1], [2]),
        (:lanelet, :suc, :lanelet) => ([1], [2]),
        (:lanelet, :adj_left, :lanelet) => ([1], [2]),
        (:lanelet, :adj_right, :lanelet) => ([1], [2])
    )
    
    # Test forward pass
    out = model(agt_features, g_all, g_hetero)
    
    # Test output dimensions
    @test size(out) == (2, num_agts)
end

@testset "Data encoding" begin
    include("../demo/encode_demo.jl")
    lanelet_roadway, g_meta = load_map_data()
    
    # Prepare data
    n_train = 70
    agt_features_train, agt_features_test, labels_train, labels_test = prepare_agent_features(lanelet_roadway, n_train=n_train)
    g_all, g_hetero = prepare_map_features(lanelet_roadway, g_meta)
    
    # Initialize model
    actor_config = Dict("n_actor" => 128, "din_actor" => 2)
    map_config = Dict("n_map" => 128, "num_scales" => 6)
    model = LaneletPredictor(actor_config, map_config)

    prediction = model(agt_features_train, g_all, g_hetero)
    @test size(prediction) == (2,70)
end