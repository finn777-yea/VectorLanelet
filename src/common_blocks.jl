# ---- ActorNet_Simp ----
"""
Agent trajectory encoder

ActorNet(in_channels, group_out_channels::Vector{Int}, μ::Union{Vector, CuArray}, σ::Union{Vector, CuArray};
    kernel_size::Int=3, norm::String="GN", ng::Int=32)

NOTE: only works with timesteps that can be divided by 4, due to stride=2 in some layers
"""
struct ActorNet
    agt_preprocess::Chain
    groups::Chain
    lateral::Chain
    output_block::Chain
end

Flux.@layer ActorNet

function ActorNet(in_channels, group_out_channels::Vector{Int}, μ::Union{Vector, CuArray}, σ::Union{Vector, CuArray};
    kernel_size::Int=3, norm::String="GN", ng::Int=32)
    agt_preprocess = create_agt_preprocess_block(μ, σ)
    out_channels = group_out_channels[end]
    groups = []
    for i in eachindex(group_out_channels)
        if i == 1
            push!(groups, create_group_block(i, in_channels, group_out_channels[i], kernel_size=kernel_size, norm=norm, ng=ng))
        else
            push!(groups, create_group_block(i, group_out_channels[i-1], group_out_channels[i], kernel_size=kernel_size, norm=norm, ng=ng))
        end
    end
    groups = Chain(groups...)

    lateral = []
    for i in eachindex(group_out_channels)
        lat_connection = Chain(
            Conv((1,), group_out_channels[i]=>out_channels, stride=1),
            GroupNorm(out_channels, gcd(ng, out_channels))
        )
        push!(lateral, lat_connection)
    end
    lateral = Chain(lateral...)

    output_block = create_residual_block(out_channels, out_channels, kernel_size=kernel_size, stride=1, ng=ng)

    ActorNet(agt_preprocess, groups, lateral, output_block)
end

function (actornet::ActorNet)(agt_features)
    agt_features = actornet.agt_preprocess(agt_features)
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
"""
    PolylineEncoder
    Takes a batch of fully connected graphs and node features as input_features
    NodeEncoder -> MaxPooling -> Duplication -> Concatenate -> OutputLayer -> MaxPooling

    - g: Fully connected graph, representing a lanelet, with nodes as vectors of centerline
    - vector_features: (2, num_vectors)
"""
struct PolylineEncoder
    vec_preprocess::Chain
    layers::Chain
    output_layer::Dense
end

Flux.@layer PolylineEncoder

function PolylineEncoder(
    in_channels,
    hidden_unit,
    μ::Union{Vector, CuArray},
    σ::Union{Vector, CuArray};
    num_layers::Int=3,
    norm::String="LN"
)
    vec_preprocess = VectorLanelet.create_map_preprocess_block(μ, σ)
    layers = []
    for _ in 1:num_layers
        push!(layers, create_node_encoder(in_channels, hidden_unit, norm))
        in_channels = hidden_unit * 2
    end
    layers = Chain(layers...)
    output_layer = Dense(hidden_unit * 2, out_channels)     # multiply by 2 due to concatenation
    PolylineEncoder(vec_preprocess, layers, output_layer)
end


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

function MapEncoder(in_channels::Int=64, out_channels::Int=64, num_layers::Int=4; ng=32)
    layers = []

    for _ in 1:num_layers
        layer = (
            dense1 = Dense(in_channels=>out_channels),
            heteroconv = create_hetero_conv(out_channels, out_channels),
            norm = GroupNorm(out_channels, gcd(ng, out_channels)),
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
