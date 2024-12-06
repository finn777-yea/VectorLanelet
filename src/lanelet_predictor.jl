using Flux
using Transformers

"""
Complete model architecture combining all components
"""

# ---- ActorNet_Simp ----
struct ActorNet_Simp
    groups::Chain
    output_block::Chain
    lateral::Chain
end

Flux.@layer ActorNet_Simp

function ActorNet_Simp(in_channels, group_out_channels::Vector{Int}=[64, 128])
    out_channels = group_out_channels[end]
    num_groups = length(group_out_channels)
    groups = []
    for i in eachindex(group_out_channels)
        if i == 1
            push!(groups, create_group_block(i, in_channels, group_out_channels[i]))
        else
            push!(groups, create_group_block(i, group_out_channels[i-1], group_out_channels[i])) 
        end
    end
    groups = Chain(groups...)

    lateral = []
    for i in eachindex(group_out_channels)
        lat_connection = Chain(
            Conv((1,), group_out_channels[i]=>out_channels, stride=1),
            GroupNorm(out_channels, gcd(32, out_channels)),
            relu
        )
        push!(lateral, lat_connection)
    end
    lateral = Chain(lateral...)

    output_block = create_residual_block(out_channels, out_channels, stride=1)

    ActorNet_Simp(groups, output_block, lateral)
end

function (actornet::ActorNet_Simp)(agt_features)
    outputs = Flux.activations(actornet.groups, agt_features)

    out = actornet.lateral[end](outputs[end])
    for i in range(length(outputs)-1, 1, step=-1)
        out = upsample_linear(out, 2, align_corners=false)
        out = out .+ actornet.lateral[i](outputs[i])
    end

    out = actornet.output_block(out)
    return @view out[end, :, :]
end

# ---- PolylineEncoder ----
struct PolylineEncoder
    layers::Chain
    output_layer::Dense
end

Flux.@layer PolylineEncoder

function PolylineEncoder(in_channels, out_channels, num_layers::Int=3, hidden_unit::Int=64)
    layers = []
    for i in 1:num_layers
        push!(layers, create_node_encoder(in_channels, hidden_unit))
        in_channels = hidden_unit * 2
    end
    layers = Chain(layers...)
    output_layer = Dense(hidden_unit * 2, out_channels)
    PolylineEncoder(layers, output_layer)
end


"""
Forward pass for PolylineEncoder
    Takes a batch of graphs and node features as input_features
    NodeEncoder -> MaxPooling -> Duplication -> Concatenate -> OutputLayer -> MaxPooling
    
    - clusters = [1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, ...]
"""
function (pline::PolylineEncoder)(g::GNNGraph, X::AbstractMatrix)
    max_pool = GlobalPool(max)
    clusters = graph_indicator(g)
    for layer in pline.layers
        X = layer(X)
        agg_data = max_pool(g, X)[:, clusters]
        X = vcat(X, agg_data)
    end
    X = pline.output_layer(X)
    out_data = max_pool(g, X)
    return out_data
end

# ---- MapEncoder ----
struct MapEncoder
    


# ---- LaneletPredictor ----
struct LaneletPredictor
    actornet::ActorNet_Simp
    vsg::VectorSubGraph
    mapnet::MapNet
    transformer::Transformer
    pred_head::PredictionHead
end

Flux.@layer LaneletPredictor

function LaneletPredictor(config::Dict)
    actornet = ActorNet_Simp(2, [64, 128])
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