function create_residual_block(in_channels , out_channels; kernel_size=2, stride, norm="GN", ng=32, act=true)
    filter = (kernel_size,)
    
    # Main convolution branch
    main_branch = Chain(
        Conv(filter, in_channels => out_channels, stride=stride, pad=SamePad()),
        norm == "GN" ? GroupNorm(out_channels, gcd(ng, out_channels)) : BatchNorm(out_channels),
        relu,
        Conv(filter, out_channels => out_channels, stride=1, pad=SamePad()),
        norm == "GN" ? GroupNorm(out_channels, gcd(ng, out_channels)) : BatchNorm(out_channels)
    )
    
    # Identity/downsample branch
    if stride != 1 || out_channels != in_channels
        identity_branch = Chain(
            Conv((1,), in_channels=>out_channels, stride=stride),
            norm == "GN" ? GroupNorm(out_channels, gcd(ng, out_channels)) : BatchNorm(out_channels)
        )
    else
        identity_branch = identity
    end
    
    # Combine branches with skip connection
    residual = Chain(
        Parallel(+, main_branch, identity_branch),
        act ? relu : identity
    )
    
    return residual
end

"""
Each group contains a first layer with stride 2 and the following layers with stride 1.
    - i: group index
    - input_channels: input channels of the group
    - output_channels: output channels of the group
"""
function create_group_block(i, input_channels, output_channels)
    first_layer = if i == 1
        create_residual_block(input_channels, output_channels, stride=1)
    else
        create_residual_block(input_channels, output_channels, stride=2)
    end
    second_layer = create_residual_block(output_channels, output_channels, stride=1)
    return Chain(first_layer, second_layer)
end

"""
    Encoder for the node features at vector level
    Contains 2 dense layers and batch/layer normalization
"""

function create_node_encoder(in_channels, out_channels, norm="layernorm")
    norm_layer = norm == "layernorm" ? LayerNorm : BatchNorm
    linear1 = Dense(in_channels, out_channels)
    norm1 = norm_layer(out_channels)
    linear2 = Dense(out_channels, out_channels)
    norm2 = norm_layer(out_channels)

    shotcut = if in_channels != out_channels
        Chain(Dense(in_channels, out_channels), norm_layer(out_channels))
    else
        identity
    end

    residual = Parallel(+, Chain(linear1, norm1, relu, linear2, norm2), shotcut)
    encoder = Chain(residual, relu)
    return encoder
end

"""
    Preprocess block for agent features
    μ: (2, )
    σ: (2, )
    μ and σ are computed from the map features(vector-level)
    map features: (4, num_vectors) channels: x_start, y_start, x_end, y_end

    1. normalize agent features to zero mean and unit variance
    2. transpose agent features for subsequent convolution
"""
function create_agt_preprocess_block(μ, σ)
    preprocess = Chain(
        Base.Fix2(.-, μ),
        Base.Fix2(./, σ),
        x -> permutedims(x, (2, 1, 3))
    )
    return preprocess
end

# vector features: (4, B)
function create_map_preprocess_block(μ, σ)
    # Reshape μ and σ to be aligned with the dimension of vector features
    # TODO: generalize repeat times
    μ = repeat(μ, 2)
    σ = repeat(σ, 2)

    preprocess = Chain(
        Base.Fix2(.-, μ),
        Base.Fix2(./, σ)
    )
    return preprocess
end

function create_transformer_block(num_layer, hidden_size, num_head)
    head_hidden_size = div(hidden_size, num_head)
    intermediate_size = 2hidden_size

    transformer = Transformer(
        Layers.TransformerBlock,        # Transformer encoder block
        num_layer, 
        relu, 
        num_head, 
        hidden_size, 
        head_hidden_size, 
        intermediate_size
    )
    return transformer
end

function create_inverse_normalization_block(μ, σ)
    inverse_normalize = Chain(
        Base.Fix2(.*, σ),
        Base.Fix2(.+, μ)
    )
    return inverse_normalize
end

function create_hetero_conv(in_channels, out_channels)
    left_rel = (:lanelet, :left, :lanelet)
    right_rel = (:lanelet, :right, :lanelet)
    adj_left_rel = (:lanelet, :adj_left, :lanelet)
    adj_right_rel = (:lanelet, :adj_right, :lanelet)
    suc_rel = (:lanelet, :suc, :lanelet)
    
    heteroconv = HeteroGraphConv(
        left_rel => GATConv(in_channels=>out_channels),
        right_rel => GATConv(in_channels=>out_channels),
        adj_left_rel => GATConv(in_channels=>out_channels),
        adj_right_rel => GATConv(in_channels=>out_channels),
        suc_rel => GATConv(in_channels=>out_channels)
    )
end

function create_prediction_head(input_dim::Int, μ, σ)
    dense = Dense(input_dim => 2)
    postprocess = Chain(
        Base.Fix2(.*, σ),
        Base.Fix2(.+, μ)
    )
    return Chain(
        dense,
        postprocess
    )
end