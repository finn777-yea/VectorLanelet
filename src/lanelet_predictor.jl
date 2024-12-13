using Flux
using Transformers

"""
Complete model architecture combining all components
"""

# ---- ActorNet_Simp ----
struct ActorNet_Simp
    agt_preprocess::Chain
    groups::Chain
    output_block::Chain
    lateral::Chain
end

Flux.@layer ActorNet_Simp

function ActorNet_Simp(in_channels, group_out_channels::Vector{Int}, μ::Union{Vector, CuArray}, σ::Union{Vector, CuArray})
    agt_preprocess = create_agt_preprocess_block(μ, σ)
    out_channels = group_out_channels[end]
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

    ActorNet_Simp(agt_preprocess, groups, output_block, lateral)
end

function (actornet::ActorNet_Simp)(agt_features)
    agt_features = actornet.agt_preprocess(agt_features)
    @assert size(agt_features, 2) == 2      # [x,y] in the 2nd dimension
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
    vec_preprocess::Chain
    layers::Chain
    output_layer::Dense
end

Flux.@layer PolylineEncoder

function PolylineEncoder(in_channels, out_channels, μ::Union{Vector, CuArray}, σ::Union{Vector, CuArray}, num_layers::Int=3, hidden_unit::Int=64)
    vec_preprocess = VectorLanelet.create_map_preprocess_block(μ, σ)
    layers = []
    for i in 1:num_layers
        push!(layers, create_node_encoder(in_channels, hidden_unit))
        in_channels = hidden_unit * 2
    end
    layers = Chain(layers...)
    output_layer = Dense(hidden_unit * 2, out_channels)
    PolylineEncoder(vec_preprocess, layers, output_layer)
end


"""
Forward pass for PolylineEncoder
    Takes a batch of graphs and node features as input_features
    NodeEncoder -> MaxPooling -> Duplication -> Concatenate -> OutputLayer -> MaxPooling
    
    - g: Fully connected graph, representing a lanelet, with nodes as vectors of centerline
    - vector_features: (2, num_vectors)
"""
function (pline::PolylineEncoder)(g::GNNGraph, vector_features::AbstractMatrix)
    vector_features = pline.vec_preprocess(vector_features)
    max_pool = GlobalPool(max)
    clusters = graph_indicator(g)
    for layer in pline.layers
        vector_features = layer(vector_features)
        agg_data = max_pool(g, vector_features)[:, clusters]
        vector_features = vcat(vector_features, agg_data)
    end
    vector_features = pline.output_layer(vector_features)
    out_data = max_pool(g, vector_features)
    return out_data
end

# ---- MapEncoder ----
struct MapEncoder
    layers
end

Flux.@layer MapEncoder

function MapEncoder(in_channels::Int=64, out_channels::Int=64, num_layers::Int=4)
    layers = []
    
    for _ in 1:num_layers
        layer = (
            dense1 = Dense(in_channels=>out_channels),
            heteroconv = create_hetero_conv(out_channels, out_channels),
            norm = GroupNorm(out_channels, gcd(32, out_channels)),
            dense2 = Dense(out_channels=>out_channels, relu)
        )
        push!(layers, layer)
        in_channels = out_channels  # For next iteration
    end
    MapEncoder(layers)
end

function (mapenc::MapEncoder)(g::GNNHeteroGraph, x::AbstractMatrix)
    x = g[:lanelet].x

    for layer in mapenc.layers
        temp = x
        
        x = layer.dense1(x)
        x = layer.heteroconv(g, (;lanelet = x)).lanelet
        x = layer.norm(x)
        x = relu.(x)
        x = layer.dense2(x)
        x = x .+ temp  # Residual connection
        x = relu.(x)
    end

    return x
end

# ---- LaneletPredictor ----
struct LaneletPredictor
    actornet::ActorNet_Simp
    ple::PolylineEncoder
    mapenc::MapEncoder
    transformer::Transformer
    pred_head::Chain
end

Flux.@layer LaneletPredictor

"""
    agt_features: (timesteps, 2, num_agents)
    map_features: (4, num_vectors)
"""
function LaneletPredictor(μ, σ)
    actornet = ActorNet_Simp(2, [16, 64], μ, σ)
    ple = PolylineEncoder(4, 64, μ, σ, 3, 64)
    mapenc = MapEncoder(64, 64, 4)

    # Transformer setup
    num_layer = 3
    hidden_size = 64
    num_head = 4
    transformer = create_transformer_block(num_layer, hidden_size, num_head)

    pred_head = create_prediction_head(hidden_size, μ, σ)

    LaneletPredictor(actornet, ple, mapenc, transformer, pred_head)
end

"""
Forward pass for LaneletPredictor
Takes raw agent features and graph data as input, returns predictions
"""
function (model::LaneletPredictor)(agt_features::AbstractArray, g_polyline::GNNGraph, g_heteromap::GNNHeteroGraph)
    emb_actor = model.actornet(agt_features)
    
    emb_lanelets = model.ple(g_polyline, g_polyline.x)
    g_heteromap = deepcopy(g_heteromap)
    g_heteromap[:lanelet].x = emb_lanelets
    emb_map = model.mapenc(g_heteromap, emb_lanelets)

    @assert size(emb_actor) == (64, size(agt_features, 3))
    @assert size(emb_map) == (64, g_polyline.num_graphs)

    # Fusion and prediction
    x = hcat(emb_actor, emb_map)
    emb_fuse = model.transformer((; hidden_state = x)).hidden_state     # (channel, tokens, batches)
    
    predictions = model.pred_head(emb_fuse[:, 1:size(emb_actor, 2), 1])     # Only take agent features for prediction

    return predictions
end