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