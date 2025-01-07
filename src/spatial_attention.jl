"""
    create_distance_mask(agt_ctrs, ctx_ctrs, threshold::Float64)

Create distance matrix and boolean mask based on pairwise distances between agent centers and context centers.

Parameters:
- agt_ctrs(query): 2×n matrix where each column represents an agent's (x,y) coordinates
- ctx_ctrs(key_value): 2×m matrix where each column represents a context's (x,y) coordinates
- threshold: Maximum distance threshold

Returns:
- Tuple of (distances, mask) where:
  - distances: m×n matrix of pairwise distances (contexts × agents)
  - mask: m×n Boolean matrix where True indicates pairs within threshold distance
"""
function create_distance_mask(agt_ctrs::Matrix{Float64}, ctx_ctrs::Matrix{Float64}, threshold::Float64)
    if threshold < 0
        throw(ArgumentError("Threshold must be non-negative"))
    end
    if size(agt_ctrs, 1) != 2 || size(ctx_ctrs, 1) != 2
        throw(ArgumentError("Input matrices must have 2 rows (x,y coordinates)"))
    end
    
    # Get number of agents and contexts
    n_agents = size(agt_ctrs, 2)
    n_contexts = size(ctx_ctrs, 2)
    
    # Compute pairwise distances between agent centers and context centers
    distances = zeros(n_contexts, n_agents)
    for i in 1:n_contexts, j in 1:n_agents
        distances[i,j] = norm(ctx_ctrs[:,i] - agt_ctrs[:,j])
    end
    return distances, distances .<= threshold
end

using Transformers.Layers
head = 2
hidden_size = 8
ca = Layers.CrossAttention(head, hidden_size)

# TODO: Apply distance mask to it
agt_feat = rand(Float32, hidden_size, 1, 2)
ctx_feat = rand(Float32, hidden_size, 1, 2)
agt_ctrs = [0.0 1.0;   # x coordinates
            0.0 1.0]    # y coordinates
ctx_ctrs = [0.0 2.0;   # x coordinates
            1.0 2.0]    # y coordinates
mask = create_distance_mask(agt_ctrs, ctx_ctrs, 1.0)

struct GraphSpatialAttention
    agt_encoder::Chain
    ctx_encoder::Chain
    attention::Chain
end

Flux.@layer GraphSpatialAttention