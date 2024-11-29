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
function create_group_block(input_channels, output_channels; kernel_size=2, stride=2, norm="GN", ng=32)
    first_layer = create_residual_block(input_channels, output_channels, 
            kernel_size=kernel_size, stride=stride, norm=norm, ng=ng)
    second_layer = create_residual_block(output_channels, output_channels, 
            kernel_size=kernel_size, stride=1, norm=norm, ng=ng)
    return Chain(first_layer, second_layer)
end

"""
    Encoder for the node features at vector level
    Contains 2 dense layers and batch/layer normalization
"""

function create_node_encoder(in_channels, out_channels, norm="layernorm ")
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
