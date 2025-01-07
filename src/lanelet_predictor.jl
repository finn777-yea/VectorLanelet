using Flux
using Transformers

"""
Complete model architecture combining all components
"""

# ---- ActorNet_Simp ----
struct ActorNet_Simp
    agt_preprocess::Chain
    groups::Chain
    lateral::Chain
    output_block::Chain
end

Flux.@layer ActorNet_Simp

function ActorNet_Simp(in_channels, group_out_channels::Vector{Int}, μ::Union{Vector, CuArray}, σ::Union{Vector, CuArray}, kernel_size::Int=3)
    agt_preprocess = create_agt_preprocess_block(μ, σ)
    out_channels = group_out_channels[end]
    groups = []
    for i in eachindex(group_out_channels)
        if i == 1
            push!(groups, create_group_block(i, in_channels, group_out_channels[i], kernel_size=kernel_size))
        else
            push!(groups, create_group_block(i, group_out_channels[i-1], group_out_channels[i], kernel_size=kernel_size))
        end
    end
    groups = Chain(groups...)

    lateral = []
    for i in eachindex(group_out_channels)
        lat_connection = Chain(
            Conv((1,), group_out_channels[i]=>out_channels, stride=1),
            GroupNorm(out_channels, gcd(32, out_channels))
        )
        push!(lateral, lat_connection)
    end
    lateral = Chain(lateral...)

    output_block = create_residual_block(out_channels, out_channels, kernel_size=kernel_size, stride=1)

    ActorNet_Simp(agt_preprocess, groups, lateral, output_block)
end

function (actornet::ActorNet_Simp)(agt_features)
    agt_features = actornet.agt_preprocess(agt_features)
    @assert size(agt_features, 2) == 2      # Channel dimension:[x,y] in the 2nd dimension
    outputs = Flux.activations(actornet.groups, agt_features)

    out = actornet.lateral[end](outputs[end])
    for i in range(length(outputs)-1, 1, step=-1)
        out = upsample_linear(out, 2, align_corners=false)
        out = out .+ actornet.lateral[i](outputs[i])
    end

    out = actornet.output_block(out)        # (timesteps, channels, num_agents)
    out = out[end, :, :]
    return out
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

"""
    g: GNNHeteroGraph
    x: (pline_out_channels, num_lanelets)
"""

function (mapenc::MapEncoder)(g::GNNHeteroGraph, x::AbstractMatrix)
    # x = g[:lanelet].x

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
    cross_attention
    pred_head
end

Flux.@layer LaneletPredictor

"""
    agt_features: (timesteps, 2, num_agents)
    map_features: (4, num_vectors)
"""
function LaneletPredictor(config::Dict{String, Any}, μ, σ)
    actornet = ActorNet_Simp(config["actornet_in_channels"], config["group_out_channels"], μ, σ)
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
function (model::LaneletPredictor)(agt_features::AbstractArray, polyline_graphs::GNNGraph, g_heteromaps::GNNHeteroGraph)
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



# ------ SpatialAttention ------
struct SpatialAttention
    agt
    dist
    query
    ctx
    out
    norm
end

Flux.@layer SpatialAttention

function SpatialAttention(agt_dim::Int, ctx_dim::Int)
    ng = 1

    agt = Dense(agt_dim => ctx_dim, bias=false)
    
    dist = Chain(
        Dense(2 => ctx_dim),
        relu,
        Dense(ctx_dim => ctx_dim),
        GroupNorm(ctx_dim, gcd(ng, ctx_dim)),
        relu
    )
    
    query = Chain(
        Dense(agt_dim => ctx_dim),
        GroupNorm(ctx_dim, gcd(ng, ctx_dim)),
        relu
    )

    ctx = Chain(
        Dense(3*ctx_dim => agt_dim),
        GroupNorm(agt_dim, gcd(ng, agt_dim)),
        relu,
        Dense(agt_dim => agt_dim, bias=false)
    )

    out = Chain(
        Dense(ctx_dim => agt_dim),
        GroupNorm(agt_dim, gcd(ng, agt_dim))
    )
    norm = GroupNorm(agt_dim, gcd(ng, agt_dim))
    SpatialAttention(agt, dist, query, ctx, out, norm)
end

"""
    agt_features: (agt_channels, timesteps, num_agents)
    ctx_features: (ctx_channels, num_lanelets)
    agt_pos: (2, num_agts)        # num_agents in one batch
    ctx_pos: (2, num_ctxs)      # num_lanelets in total

    Returns:
    - updated_agt_features: (agt_channels, num_agents)
"""
function (att::SpatialAttention)(agts::AbstractArray, ctxs::AbstractArray, agt_pos::AbstractArray, ctx_pos::AbstractArray)
    res = agts
    
    # Handle empty context case
    if size(ctxs, 2) == 0
        agts = att.agt(agts)
        agts = relu.(agts)
        agts = att.out(agts)
        agts = agts + res
        agts = relu.(agts)
        return agts
    end
    
    # Calculate distance between agt and ctx
    # TODO: Add distance threshold here
    dist = reshape(agt_pos, (2, :, 1)) .- reshape(ctx_pos, (2, 1, :))       # (2, num_agts, num_ctx)
    dist = reshape(dist, (2, :))    # (2, num_agts*num_ctx)
    dist = att.dist(dist)   # (ctx_dim, num_agts*num_ctx)
    
    # Get query features from agents and expand
    query = att.query(agts)
    query = reshape(query, (:, :, 1))
    query = repeat(query, 1, 1, num_ctx)
    
    # Expand context features
    # Result: (n_ctx, num_agents, num_contexts)
    ctx_expanded = reshape(ctxs, (:, 1, :))
    ctx_expanded = repeat(ctx_expanded, 1, num_agents, 1)
    
    # Concatenate distance, query and context features along feature dimension
    # Result: (3*n_ctx, num_agents, num_contexts)
    ctx_combined = vcat(dist, query, ctx_expanded)
    
    # Process through context network
    # Reshape to (3*n_ctx, num_pairs) for the network
    ctx_combined = reshape(ctx_combined, (:, :))
    ctx_combined = att.ctx(ctx_combined)
    # Reshape back to (n_agt, num_agents, num_contexts)
    ctx_combined = reshape(ctx_combined, (:, num_agents, num_ctx))
    
    # Sum over context dimension
    ctx_combined = sum(ctx_combined, dims=3)
    ctx_combined = dropdims(ctx_combined, dims=3)
    
    # Process agent features
    agts = att.agt(agts)
    agts = agts + ctx_combined
    agts = att.norm(agts)
    agts = relu.(agts)
    
    # Final linear layer with residual connection
    agts = att.out(agts)
    agts = agts + res
    agts = relu.(agts)
    
    return agts
end