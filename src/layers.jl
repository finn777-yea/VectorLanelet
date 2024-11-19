"""Customized layers which norm and act can be determined"""


# ------ Res1d -----------
struct Res1d
    layers::Chain
    downsample::Union{Chain, Nothing}
    act::Bool
end

Flux.@layer Res1d

function Res1d(n_in, n_out; kernel_size=3, stride=1, norm="GN", ng=32, act=true)
    filter = (kernel_size,)
    conv1 = Conv(filter, n_in => n_out, stride=stride, pad=SamePad())
    norm1 = norm == "GN" ? GroupNorm(n_out, gcd(ng, n_out)) : BatchNorm(n_out)
    conv2 = Conv(filter, n_out => n_out, stride=1, pad=SamePad())
    norm2 = norm == "GN" ? GroupNorm(n_out, gcd(ng, n_out)) : BatchNorm(n_out)

    layers = Chain(
        conv1,
        norm1,
        relu,
        conv2,
        norm2
    )
    
    if stride != 1 || n_out != n_in
        downsample = Chain(
            Conv((1,), n_in=>n_out, stride=stride),
            norm == "GN" ? GroupNorm(n_out, gcd(ng, n_out)) : BatchNorm(n_out)
        )
    else
        downsample = nothing
    end


    Res1d(layers, downsample, act)
end

### Forward function
function (res1d::Res1d)(X)
    out = res1d.layers(X)

    if !isnothing(res1d.downsample)
        X = res1d.downsample(X)
    end
    
    # TODO: Refactor it using skip connection
    out = out .+ X  # In-place addition for residual connection

    out = res1d.act ? relu.(out) : out
    return out
end



# ------ Conv1d --------
struct Conv1d{T<:Conv}
    norm
    conv::T
    act::Bool
end
Flux.@layer Conv1d

function Conv1d(n_in, n_out; kernel_size=3, stride=1, norm="GN", ng=32, act=true)
    @assert in(norm, ["GN", "BN", "SyncBN"])
    conv = Conv((kernel_size,), n_in=>n_out, pad=SamePad(), stride=stride)
    if norm == "GN"
        norm = GroupNorm(n_out, gcd(ng, n_out))
    # TODO: add batch norm
    elseif norm == "BN"
        nothing
    end
    Conv1d(norm, conv, act)
end

function (conv1d::Conv1d)(X)
    out = conv1d.conv(X)
    out = conv1d.norm(out)
    if conv1d.act
        out = relu.(out)
    end
    return out
end



# ------ Linear ----------
# TODO: add norm selection
struct Linear
    layers::NamedTuple
    act::Bool

    function Linear(layers::NamedTuple, act::Bool)
        new(layers, act)
    end
end

Flux.@layer Linear

function Linear(n_in, n_out; norm="GN", ng=32, act=true)
    linear = Dense(n_in=>n_out)
    norm = GroupNorm(n_out, gcd(ng, n_out))
    layers = (linear=linear, norm=norm)
    Linear(layers, act)
end

function (linear::Linear)(X)
    l = linear.layers
    out = l.linear(X)
    out = l.norm(out)
    if linear.act
        out = relu.(out)
    end
    return out
end

"""
Simple prediction head using a dense layer
"""
struct PredictionHead
    dense::Dense
end

Flux.@layer PredictionHead

function PredictionHead(input_dim::Int)
    return PredictionHead(Dense(input_dim => 2))
end

function (m::PredictionHead)(x)
    # Expect x to be of shape (hidden_size, num_samples)
    return m.dense(x)
end