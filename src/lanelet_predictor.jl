using Flux
using Transformers

"""
Complete model architecture combining all components
"""
struct LaneletPredictor
    actornet::ActorNet_Simp
    vsg::VectorSubGraph
    mapnet::MapNet
    transformer::Transformer
    pred_head::PredictionHead
end

Flux.@layer LaneletPredictor

function LaneletPredictor(config::Dict)
    actornet = ActorNet_Simp(config["actor_config"])
    vsg = VectorSubGraph(config["vsg_in_channel"])
    mapnet = MapNet(config["map_config"])

    # Transformer setup
    num_layer = config["transformer_num_layer"]
    hidden_size = config["transformer_hidden_size"]
    num_head = config["transformer_num_head"]
    head_hidden_size = div(hidden_size, num_head)
    intermediate_size = 2hidden_size

    transformer = Transformer(Layers.TransformerBlock,
        num_layer, relu, num_head, hidden_size, head_hidden_size, intermediate_size)
    pred_head = PredictionHead(hidden_size)

    LaneletPredictor(actornet, vsg, mapnet, transformer, pred_head)
end

"""
Forward pass for LaneletPredictor
Takes raw agent features and graph data as input, returns predictions
"""
function (model::LaneletPredictor)(agt_features, g_all, g_hetero, μ, σ)
    emb_actor = model.actornet(agt_features)

    emb_lanelets = model.vsg(g_all, g_all.x)
    g_hetero = deepcopy(g_hetero)
    g_hetero[:lanelet].x = emb_lanelets
    emb_map = model.mapnet(g_hetero)

    @assert size(emb_actor) == (128, size(agt_features, 3))
    @assert size(emb_map) == (128, g_all.num_graphs)

    # Fusion and prediction
    x = hcat(emb_actor, emb_map)
    emb_fuse = model.transformer((; hidden_state = x)).hidden_state     # (channel, tokens, batches)
    predictions = model.pred_head(emb_fuse[:, 1:size(emb_actor, 2), 1], μ, σ)     # Only take agent features for prediction

    return predictions
end