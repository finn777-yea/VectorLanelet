"""
Complete model architecture combining all components
"""

# ---- LaneletPredictor ----
struct LaneletPredictor
    actornet::ActorNet
    ple::PolylineEncoder
    mapenc::MapEncoder
    transformer::Transformer
    cross_attention
    pred_head
end

Flux.@layer LaneletPredictor

"""
    agt_features: (timesteps, 2, num_agents)
    map_features: (4, num_vectors)
"""
function LaneletPredictor(config::Dict{String, Any}, μ, σ)
    actornet = ActorNet(config["actornet_in_channels"], config["group_out_channels"], μ, σ)
    ple = PolylineEncoder(config["ple_in_channels"], config["ple_out_channels"], μ, σ, config["ple_num_layers"], config["ple_hidden_unit"])
    mapenc = MapEncoder(config["mapenc_hidden_unit"], config["mapenc_hidden_unit"], config["mapenc_num_layers"])

    # Transformer setup
    transformer = create_transformer_block(config["transformer_num_layer"], config["transformer_hidden_unit"], config["transformer_num_head"])
    cross_attention = MultiHeadAttention(64, nheads=2)

    # Prediction head
    pred_head = create_prediction_head(config["transformer_hidden_unit"], μ, σ)
    # pred_head = Dense(config["transformer_hidden_unit"] => 2)
    LaneletPredictor(actornet, ple, mapenc, transformer, cross_attention, pred_head)
end

"""
Forward pass for LaneletPredictor
Takes raw agent features and graph data as input, returns predictions

- agt_features: (channels, timesteps, num_agents)
- g_polylines: A batch of polylines: each graph -> lanelet, node -> vector
- g_heteromaps: A batch of maps: each graph -> map, node -> lanelet
"""
# TODO: For agents from certain map, provide corresponding map
function (model::LaneletPredictor)(agt_features::AbstractArray, agt_pos, polyline_graphs::GNNGraph, g_heteromaps::GNNHeteroGraph, llt_pos)
    emb_actor = model.actornet(agt_features)

    emb_lanelets = model.ple(polyline_graphs, polyline_graphs.x)
    # g_heteromaps = deepcopy(g_heteromaps)
    # g_heteromaps[:lanelet].x = emb_lanelets
    emb_map = model.mapenc(g_heteromaps, emb_lanelets)

    @assert size(emb_actor) == (64, size(agt_features, 3))
    # @assert size(emb_map) == (64, polyline_graphs.num_grafphs)

    # Fusion and prediction
    x = hcat(emb_actor, emb_map)
    # TODO: Add distance mask to it
    emb_fuse = model.transformer((; hidden_state = x)).hidden_state     # (channel, tokens, batches)
    # emb_fuse, _ = model.cross_attention(x)

    predictions = model.pred_head(emb_fuse[:, 1:size(emb_actor, 2)])     # Only take agent features for prediction

    return predictions
end
