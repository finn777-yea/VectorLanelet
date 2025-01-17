function create_filtered_interaction_graphs(agt_pos, ctx_pos, distance_threshold::Real)
    num_agt = size(agt_pos, 2)
    num_ctx = size(ctx_pos, 2)
    dist = reshape(agt_pos, 2,:,1) .- reshape(ctx_pos, 2,1,:)
    # [dist]: (2, num_agt, num_ctx)
    dist = sqrt.(sum(dist.^2, dims=1))[1,:,:]
    
    @assert size(dist) == (num_agt, num_ctx) "Distance matrix size is not correct"
    mask = dist .<= distance_threshold
    @assert size(mask) == (num_agt, num_ctx) "Mask size is not correct"
    
    
    indices = findall(mask) |> cpu
    
    # TODO: Configure src and dst correctly: the aggregation happens at target nodes
    # TODO: Make it GPU-friendly: avoid indices[i]
    src = [idx[1] for idx in indices]  # agent indices
    dst = [idx[2] + num_agt for idx in indices]  # context indices
    edge_ind = (dst, src)
    
    # Normalize edge data
    edata = reshape(dist[mask], 1, :)
    μ, σ = calculate_mean_and_std(edata, dims=2)
    edata = (edata .- μ) ./ σ

    inter_graph = GNNGraph(
        edge_ind,
        num_nodes = num_agt + num_ctx,
        edata = (;d = edata)
    )
    @assert inter_graph.num_nodes == num_agt + num_ctx "Number of nodes is not correct"
    return inter_graph
end

# """
#     Create interaction graphs with node indices
#     Return:
#         - inter_graphs: GNNGraphs with node indices
#         the ith graph in inter_graphs has ndata.ind storing:
#             global index of the agt node
#             global indices of the valid ctx nodes
# """
# function create_filtered_interaction_graphs(agt_pos, ctx_pos, distance_threshold::Real)
#     inter_graphs = GNNGraph[]
#     for agt_ind in axes(agt_pos, 2)
#         diff = agt_pos[:, agt_ind] .- ctx_pos
#         distances = sqrt.(sum(diff.^2, dims=1))
#         @show eltype(distances)
#         mask = distances .<= distance_threshold
#         # [distance]/[mask]: 1xnum_ctx
#         @assert length(vec(mask)) == size(ctx_pos, 2) "Mask size is not correct"
        
#         valid_indices = findall(vec(mask))      # Global indices of the ctx nodes
#         g = star_digraph(length(valid_indices) + 1) |> GNNGraph
#         g.ndata.ind = vcat(agt_ind, valid_indices)
#         g.edata.d = reshape(distances[mask], 1, :)
#         @show eltype(g.edata.d)
#         push!(inter_graphs, g)
#     end
#     return batch(inter_graphs)
# end

# TODO: Make it GPU compatible
# function prepare_interaction_feautures(emb_agt, emb_ctx, inter_graphs)
#     ndata = zeros(Float32, size(emb_agt, 1), inter_graphs.num_nodes)
#     clusters = graph_indicator(inter_graphs)
#     # Replace in-place updates with functional construction
#     ndata = mapreduce(hcat, 1:inter_graphs.num_graphs) do cluster
#         cluster_nodes = findall(==(cluster), clusters)
#         inds = inter_graphs.ndata.ind[cluster_nodes]
#         # First node is agent, rest are map nodes
#         @assert inds[1] == cluster "Agent node index is not correct"
        
#         # Concatenate agent and context features for this cluster
#         [emb_agt[:, inds[1]] reduce(hcat, [emb_ctx[:, i] for i in inds[2:end]])]
#     end
#     return ndata
# end

"""
A GAT based interaction graph model
Parameters:
- gat: GATConv layer
- norm: GroupNorm layer
- output: Dense layer
"""
struct InteractionGraphModel
    gat
    norm
    output
end

Flux.@layer InteractionGraphModel
"""
    Create a GAT based interaction graph model
Parameters:
- n_in: Input node feature dimension
- e_in: Input edge feature dimension
- head_dim: Dimension of each attention head
- num_heads: Number of attention heads
"""
function InteractionGraphModel(n_in::Int, e_in::Int, out_dim::Int; num_heads::Int=2)
    head_dim = div(out_dim, num_heads)
    gat = GATConv((n_in, e_in)=>head_dim, heads=num_heads, add_self_loops=false)
    norm = GroupNorm(out_dim, gcd(1, out_dim))
    output = Dense(out_dim=>out_dim)
    return InteractionGraphModel(gat, norm, output)
end

"""
    InteractionGraphModel forward pass
Parameters:
- g: agt-centered GNNGraphs
- node_features: Concatenated agt-centered graphs node features
- edge_features: Distances
Returns:
- Output agts features, shape (channels, num_agts)
"""
# TODO: normalize edge features?
function (interaction::InteractionGraphModel)(agt_features, agt_pos, ctx_features, ctx_pos, dist_thrd)
    num_agts = size(agt_pos, 2)
    num_ctx = size(ctx_pos, 2)
    g = create_filtered_interaction_graphs(agt_pos, ctx_pos, dist_thrd)
    @assert g.num_nodes == num_agts + num_ctx "Number of nodes is not correct"
    node_features = hcat(agt_features, ctx_features)
    
    res = node_features
    # TODO: Configure the type of edge features
    x = interaction.gat(g, node_features, g.edata.d)
    x = interaction.norm(x)
    x = relu(x)
    x = interaction.output(x)
    x = res + x

    # Return the corresponding agt features
    return x[:, 1:num_agts]
end

# ------ LaneletFusionPred -------
struct LaneletFusionPred
    actornet::ActorNet_Simp
    ple::PolylineEncoder
    mapenc::MapEncoder
    a2m::InteractionGraphModel
    m2m::MapEncoder
    m2a::InteractionGraphModel
    a2a::InteractionGraphModel
    pred_head
    dist_thrd
end

Flux.@layer LaneletFusionPred

function LaneletFusionPred(config::Dict{String, Any}, μ, σ)
    actornet = ActorNet_Simp(config["actornet_in_channels"], config["group_out_channels"], μ, σ)
    ple = PolylineEncoder(config["ple_in_channels"], config["ple_out_channels"], μ, σ, config["ple_num_layers"], config["ple_hidden_unit"])
    mapenc = MapEncoder(config["mapenc_hidden_unit"], config["mapenc_hidden_unit"], config["mapenc_num_layers"])

    # Fusion setup
    # TODO: Make it multiple layers
    a2m = InteractionGraphModel(config["fusion_n_in"], config["fusion_e_in"], config["fusion_out_dim"])
    m2m = MapEncoder(config["fusion_out_dim"], config["fusion_out_dim"], 2, ng=1)
    m2a = InteractionGraphModel(config["fusion_n_in"], config["fusion_e_in"], config["fusion_out_dim"])
    a2a = InteractionGraphModel(config["fusion_n_in"], config["fusion_e_in"], config["fusion_out_dim"])


    # Prediction head
    pred_head = create_prediction_head(config["fusion_out_dim"], μ, σ)

    dist_thrd = (;agent = config["agent_dist_thrd"], llt = config["llt_dist_thrd"])
    LaneletFusionPred(actornet, ple, mapenc, a2m, m2m, m2a, a2a, pred_head, dist_thrd)
end

"""
    Forward pass for LaneletFusionPred
Parameters:
    - agt_features: (channels, timesteps, num_agents)
    - pos_agt: (2, num_agents)
    - polyline_graphs: each graph -> lanelet, node -> vector
    - g_heteromaps: each graph -> map, node -> lanelet
    - pos_llt: (2, num_lanelets)
    - map2agent_graphs: GNNGraphs with edge data
    - agent2map_graphs: GNNGraphs with edge data
"""
# TODO: Use profile to check the efficiency
function (model::LaneletFusionPred)(agt_features, agt_pos, polyline_graphs, g_heteromaps, llt_pos)
    emb_actor = model.actornet(agt_features)
    emb_lanelets = model.ple(polyline_graphs, polyline_graphs.x)
    emb_map = model.mapenc(g_heteromaps, emb_lanelets)

    # TODO: Extend the fusion module to contain 2 layers of fuse attention
    emb_map = model.a2m(emb_map, llt_pos, emb_actor, agt_pos, model.dist_thrd.llt)
    @assert size(emb_map,2) == g_heteromaps.num_nodes[:lanelet]
    emb_map = model.m2m(g_heteromaps, emb_map)

    # Assign the updated emb_map to map2agent_graphs
    emb_actor = model.m2a(emb_actor, agt_pos, emb_map, llt_pos, model.dist_thrd.agent)
    

    emb_actor = model.a2a(emb_actor, agt_pos, emb_actor, agt_pos, model.dist_thrd.agent)
    @assert size(emb_actor) == (64, size(agt_pos, 2))
    
    predictions = model.pred_head(emb_actor)
    return predictions
end