using Flux.NNlib
using Flux
using Transformers
using Transformers.Layers
using NeuralAttentionlib
using NeuralAttentionlib.Masks

# Mask
n_state = 3
max_seq_len = 4
batch_size = 8

qk_dim = 16 # query key dim per head
v_dim = 64 # value dim per head
out_dim = 64 # output dim in total
n_heads = 2

mha = MultiHeadAttention(n_state=>(qk_dim * n_heads, v_dim * n_heads)=>out_dim,nheads=n_heads)
q = rand(n_state, max_seq_len, batch_size)
attention_scores = mha(q)[2]
mask = make_causal_mask(q)

neginf = typemin(eltype(attention_scores))
masked_attention_scores = ifelse.(mask, attention_scores, neginf)

# Cross attention
# att_op = 
head = 2
hidden_size = 8
head_dim = div(hidden_size, head)
csatt = Layers.CrossAttention(head, hidden_size)
query = rand(Float32, hidden_size, 1, 2)        # length = 5
key_value = rand(Float32, hidden_size, 1, 2)    # length = 7

mask = SymLengthMask(1)
bool_mask = trues(3,5,2) .* mask
# shape: q_len, k_len, batch
# apply same mask across different heads

# Forward pass
output = csatt((; hidden_state=query,
memory=key_value,
attention_mask=mask))
output.hidden_state
output.attention_mask
@assert output.memory == key_value
@assert output.hidden_state != query



# SpatialRelationalAttention
struct SpatialRelationalAttention <: AbstractAttenOp
    dist_net::Chain
    query_net::Chain
    ctx_net::Chain
    dist_threshold::Float32
end

function (op::SpatialRelationalAttention)(q, k, v, mask=nothing)
    # 1. Calculate pairwise distances and create distance mask
    dist = compute_pairwise_distances(q_positions, k_positions)
    dist_mask = dist .<= op.dist_threshold
    
    # 2. Process distances through dist_net
    dist_features = op.dist_net(dist)
    
    # 3. Process queries
    query_features = op.query_net(q)
    
    # 4. Concatenate features
    combined_features = vcat(dist_features, query_features, v)
    
    # 5. Final context processing
    attention = op.ctx_net(combined_features)
    
    return attention
end

"""
    Wrap cross attention
"""
struct SpatialCrossAttention
    cross_attention::CrossAttention
    norm::GroupNorm
    residual_proj::Dense
    activation::Function
end

function SpatialCrossAttention(n_agt::Int, n_ctx::Int; dist_th::Float32=5.0f0)
    # Create the custom attention operator
    dist_net = Chain(
        Dense(2, n_ctx),
        relu,
        Dense(n_ctx, n_ctx)
    )
    
    query_net = Dense(n_agt, n_ctx)
    
    ctx_net = Chain(
        Dense(3 * n_ctx, n_agt),
        Dense(n_agt, n_agt, bias=false)
    )
    
    attention_op = SpatialRelationalAttention(
        dist_net,
        query_net,
        ctx_net,
        dist_th
    )
    
    # Create CrossAttention with custom operator
    cross_attention = CrossAttention(
        attention_op,
        1,  # single head in this case
        n_agt,
        n_ctx
    )
    
    # Additional components
    norm = GroupNorm(1, n_agt)  # equivalent to the PyTorch version's ng=1
    residual_proj = Dense(n_agt, n_agt)
    
    return SpatialCrossAttention(
        cross_attention,
        norm,
        residual_proj,
        relu
    )
end

function (m::SpatialCrossAttention)(
    agts::AbstractArray,
    agt_idcs::Vector,
    agt_ctrs::Vector,
    ctx::AbstractArray,
    ctx_idcs::Vector,
    ctx_ctrs::Vector
)
    # Handle empty context case
    if isempty(ctx)
        return residual_forward(m, agts)
    end
    
    # 1. Create attention mask based on distances
    mask = create_distance_mask(agt_ctrs, ctx_ctrs, m.cross_attention.attention_op.dist_threshold)
    # shape: num_agts x num_ctxs
    
    # 2. Apply cross attention
    res = agts  # Store residual
    out = m.cross_attention((
        query = agts,
        memory = ctx,
        attention_mask = mask
    )).hidden_state
    
    # 3. Post-processing
    out = m.norm(out)
    out = m.activation.(out)
    out = m.residual_proj(out)
    out += res
    out = m.activation.(out)
    
    return out
end

# Helper function for empty context case
function residual_forward(m::SpatialCrossAttention, agts::AbstractArray)
    res = agts
    out = m.residual_proj(agts)
    out = m.activation.(out)
    out += res
    out = m.activation.(out)
    return out
end
