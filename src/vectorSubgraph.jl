struct VectorSubGraph{T<:Int}
    num_layers::T
    hidden_unit::T
    out_channel::T
    layers::Chain
    linear::Dense
end

Flux.@layer VectorSubGraph

function VectorSubGraph(in_channel::Int, num_layers::Int=3, hidden_unit::Int=64)
    out_channel = hidden_unit
    layers = []
    for i in num_layers
        # TODO: replace MLP with Attention
        push!(layers, MLP(in_channel, hidden_unit; hidden_unit=hidden_unit, activation="relu", norm="layer"))
        in_channel = hidden_unit * 2
    end
    layers = Chain(layers...)
    linear = Dense(hidden_unit *2, hidden_unit)     # output layer without activation and normalization
    return VectorSubGraph(num_layers, hidden_unit, out_channel, layers, linear)
end

function (vsg::VectorSubGraph)(g::GNNGraph, X::AbstractMatrix)
    max_pool = GlobalPool(max)
    for layer in vsg.layers
        X = layer(X)
        agg_data = max_pool(g, X)
        clusters = graph_indicator(g)

        agg_data = agg_data[:, clusters]
        X = vcat(X, agg_data)
    end
    X = vsg.linear(X)
    out_data = max_pool(g, X)
    return out_data
end

# MLP
struct MLP
    linear1::Dense
    linear2::Dense
    norm1
    norm2
    act1
    act2
    shortcut
end

Flux.@layer MLP

function MLP(in_channel::Int, out_channel::Int; hidden_unit::Int=64, bias::Bool=true, activation::String="relu", norm::String="layer")
    # Define the activation function
    act_layer = if activation == "relu"
        relu
    # elseif activation == "relu6"
        # x -> min.(max.(x, 0), 6)
    elseif activation == "leaky"
        leakyrelu
    else
        error("Unsupported activation function")
    end

    # Define the normalization function
    norm_layer = if norm == "layer"
        LayerNorm
    elseif norm == "batch"
        BatchNorm
    else
        error("Unsupported normalization function")
    end

    # Create layers
    linear1 = Dense(in_channel, hidden_unit, bias=bias)
    linear2 = Dense(hidden_unit, out_channel, bias=bias)
    
    norm1 = norm_layer(hidden_unit)
    norm2 = norm_layer(out_channel)
    
    act1 = act_layer
    act2 = act_layer
    
    shortcut = if in_channel != out_channel
        Chain(Dense(in_channel, out_channel, bias=bias), norm_layer(out_channel))
    else
        nothing
    end

    MLP(linear1, linear2, norm1, norm2, act1, act2, shortcut)
end

function (m::MLP)(x)
    out = m.linear1(x)
    out = m.norm1(out)
    out = m.act1(out)
    out = m.linear2(out)
    out = m.norm2(out)

    if !isnothing(m.shortcut)
        out += m.shortcut(x)
    else
        out += x
    end
    
    m.act2(out)
end


# in_channels = 10
# vsg = VectorSubGraph(in_channels)

# Create a list of GNNGraphs of 10 polylines
# data = [rand_graph(3,6, ndata=(;x=rand(Float32, 10, 3))) for _ in 1:10]
# g = batch(data)
# @assert g.num_graphs == 10

# # Run forward pass
# g12 = getgraph(g, 1:2)

# output = vsg(g12)

# # Check output shape: 
# @assert size(output) == (vsg.out_channel, 2)