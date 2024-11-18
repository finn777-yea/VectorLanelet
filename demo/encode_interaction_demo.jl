include("encode_agent_demo.jl")
include("encode_map_demo.jl")

using Transformers
using Flux

@assert size(emb_actor) == (128, 70)
@assert size(emb_map) == (128, 105)

### 1. Concatenate the embeddings and apply self-Attention
# TODO: add locality and positional encoding

num_layer = 3
hidden_size = 128       # input dimension
num_head = 2
head_hidden_size = div(hidden_size, num_head)
intermediate_size = 2hidden_size

trf_blocks = Transformer(Layers.TransformerBlock,
    num_layer, relu, num_head, hidden_size, head_hidden_size, intermediate_size)
x = hcat(emb_actor, emb_map)

emb_fuse = trf_blocks((; hidden_state = x)).hidden_state
@assert size(emb_fuse) == (128, 175, 1)

### 2. Use FuseNet in LanneGCN (cross-attention + self-attention)
# TODO: add threshold
# TODO: add distance

# A2M
hidden_size = 128
num_head = 2
# TODO: change attention type
crs_att1 = Layers.CrossAttention(num_head, hidden_size)
linear = Linear(hidden_size, hidden_size)
