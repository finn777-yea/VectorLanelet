function create_filtered_interaction_graphs(agt_pos, ctx_pos, distance_threshold::Real, normalize_dist::Bool=false)
    num_agt = size(agt_pos, 2)
    num_ctx = size(ctx_pos, 2)
    dist = reshape(agt_pos, 2,:,1) .- reshape(ctx_pos, 2,1,:)
    # [dist]: (2, num_agt, num_ctx)
    dist = sqrt.(sum(dist.^2, dims=1))[1,:,:]

    @assert size(dist) == (num_agt, num_ctx) "Distance matrix size is not correct"
    mask = dist .<= distance_threshold
    @assert size(mask) == (num_agt, num_ctx) "Mask size is not correct"
    indices = findall(mask) |> cpu
    
    # Handle empty case
    isempty(indices) && return GNNGraph(num_agt + num_ctx, dir=:in)

    # TODO: Make it GPU-friendly: avoid indices[i]
    src = [idx[1] for idx in indices]  # agent indices
    dst = [idx[2] + num_agt for idx in indices]  # context indices
    edge_ind = (src, dst)
    
    # Normalize edge data
    edata = reshape(dist[mask], 1, :)
    if normalize_dist
        μ, σ = calculate_mean_and_std(edata, dims=2)
        edata = (edata .- μ) ./ σ
    end
    # Configure src and dst correctly: the aggregation happens at target nodes
    inter_graph = GNNGraph(
        edge_ind,
        num_nodes = num_agt + num_ctx,
        edata = (;d = edata),
        dir = :in
    )
    @assert inter_graph.num_nodes == num_agt + num_ctx "Number of nodes is not correct"
    return inter_graph
end

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
    agt_res
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
    ng = 32
    head_dim = div(out_dim, num_heads)
    gat = GATConv((n_in, e_in)=>head_dim, heads=num_heads, add_self_loops=false)
    # TODO: Config layer norm
    norm = GroupNorm(out_dim, gcd(1, out_dim))      # LayerNorm
    # norm = LayerNorm(out_dim)
    output = Dense(out_dim=>out_dim)
    agt_res = SkipConnection(
        Chain(
            Dense(out_dim=>out_dim),
            relu,
            Dense(out_dim=>out_dim),
            GroupNorm(out_dim, gcd(32, out_dim)),
            relu
        ),
        +
    )
    return InteractionGraphModel(gat, norm, output, agt_res)
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
# TODO: Empty ctx handling
function (interaction::InteractionGraphModel)(data)
    agt_features, agt_pos, ctx_features, ctx_pos, dist_thrd = data
    num_agts = size(agt_pos, 2)
    num_ctx = size(ctx_pos, 2)
    g = create_filtered_interaction_graphs(agt_pos, ctx_pos, dist_thrd)
    @assert g.num_nodes == num_agts + num_ctx "Number of nodes is not correct"
    
    if g.num_edges == 0
        # TODO: Pass agt_features through a res block
        agt_features = interaction.agt_res(agt_features)
        agt_features = relu(agt_features)
        return agt_features
    end
    
    node_features = hcat(agt_features, ctx_features)
    res = node_features
    # TODO: Configure the type of edge features
    x = interaction.gat(g, node_features, g.edata.d)
    x = interaction.norm(x)
    x = relu(x)
    x = interaction.output(x)
    x = res + x
    x = relu(x)

    # Return the corresponding agt features
    return x[:, 1:num_agts]
end

# ------ LaneletFusionPred -------
struct LaneletFusionPred
    actornet::ActorNet_Simp
    ple::PolylineEncoder
    mapenc::MapEncoder
    a2m::Chain
    m2m::MapEncoder
    m2a::Chain
    a2a::Chain
    pred_head
    dist_thrd
end

Flux.@layer LaneletFusionPred

function LaneletFusionPred(config::Dict{String, Any}, μ, σ)
    actornet = ActorNet_Simp(config["actornet_in_channels"], config["group_out_channels"], μ, σ)
    ple = PolylineEncoder(config["ple_in_channels"], config["ple_out_channels"], μ, σ, config["ple_num_layers"], config["ple_hidden_unit"])
    mapenc = MapEncoder(config["mapenc_hidden_unit"], config["mapenc_hidden_unit"], config["mapenc_num_layers"])

    # Fusion setup
    a2m_layers = InteractionGraphModel[]
    m2a_layers = InteractionGraphModel[]
    a2a_layers = InteractionGraphModel[]
    for _ in config["fusion_num_layers"]
        push!(a2m_layers, InteractionGraphModel(config["fusion_n_in"], config["fusion_e_in"], config["fusion_out_dim"]))
        push!(m2a_layers, InteractionGraphModel(config["fusion_n_in"], config["fusion_e_in"], config["fusion_out_dim"]))
        push!(a2a_layers, InteractionGraphModel(config["fusion_n_in"], config["fusion_e_in"], config["fusion_out_dim"]))
    end
    a2m = Chain(a2m_layers...)
    m2m = MapEncoder(config["fusion_out_dim"], config["fusion_out_dim"], 2, ng=1)
    m2a = Chain(m2a_layers...)
    a2a = Chain(a2a_layers...)


    # Prediction head
    pred_head = create_prediction_head(config["fusion_out_dim"], μ, σ)

    dist_thrd = (;a2m = config["agent2map_dist_thrd"], m2a = config["map2agent_dist_thrd"], a2a = config["agent2agent_dist_thrd"])
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
    emb_map = model.a2m((emb_map, llt_pos, emb_actor, agt_pos, model.dist_thrd.a2m))
    @assert size(emb_map,2) == g_heteromaps.num_nodes[:lanelet]
    emb_map = model.m2m(g_heteromaps, emb_map)

    # Assign the updated emb_map to map2agent_graphs
    emb_actor = model.m2a((emb_actor, agt_pos, emb_map, llt_pos, model.dist_thrd.m2a))
    emb_actor = model.a2a((emb_actor, agt_pos, emb_actor, agt_pos, model.dist_thrd.a2a))
    @assert size(emb_actor) == (64, size(agt_pos, 2))

    predictions = model.pred_head(emb_actor)
    return predictions
end