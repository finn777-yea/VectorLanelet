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
function InteractionGraphModel(n_in::Int, e_in::Int, out_dim::Int; num_heads::Int=2, norm="GN", ng=1)
    head_dim = div(out_dim, num_heads)
    gat = GATv2Conv((n_in, e_in)=>head_dim, heads=num_heads, add_self_loops=false, concat=true, bias=true)
    # gat = GATv2Conv(n_in=>head_dim, heads=num_heads, add_self_loops=true, bias=true, concat=true)
    norm = norm == "GN" ? GroupNorm(out_dim, gcd(ng, out_dim)) : LayerNorm(out_dim)
    output = Dense(out_dim=>out_dim)
    agt_res = SkipConnection(
        Chain(
            Dense(out_dim=>out_dim),
            relu,
            Dense(out_dim=>out_dim),
            # GroupNorm(out_dim, gcd(ng, out_dim)),
            LayerNorm(out_dim),
            relu
        ),
        +
    )
    return InteractionGraphModel(gat, norm, output, agt_res)
end

"""
    InteractionGraphModel forward pass
    Parameters:
- data: (agt_features, agt_pos, ctx_features, ctx_pos, dist_thrd)
- agt_features: (channels, timesteps, num_agents)
- ctx_features: (channels, timesteps, num_ctxs)

- agt_pos:  num_scenarios x (2, num_agents)
- ctx_pos: num_scenarios x (2, num_ctxs)
- dist_thrd: Distance threshold for creating edges

    Returns:
- Output agts features, shape (channels, num_agts)
"""
function (interaction::InteractionGraphModel)(data)
    agt_features, ctx_features, g = data
    num_agts = size(agt_features,2)
    num_ctx = size(ctx_features,2)

    @assert g.num_nodes == num_agts + num_ctx "Number of nodes is not correct"

    if g.num_edges == g.num_nodes       # only self-loops
        @info "No interaction"
        agt_features = interaction.agt_res(agt_features)
        agt_features = relu(agt_features)
        return agt_features
    end

    node_features = hcat(agt_features, ctx_features)
    res = node_features
    # Save the node_features using JLD2
    # jldsave(joinpath(@__DIR__, "../node_features.jld2"), node_features=cpu(node_features), g=cpu(g))
    x = interaction.gat(g, node_features, g.edata.e)
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
    actornet::ActorNet
    ple::PolylineEncoder
    mapenc::MapEncoder
    a2m::Chain
    m2m::MapEncoder
    m2a::Chain
    a2a::Chain
    pred_head
end

Flux.@layer LaneletFusionPred

function LaneletFusionPred(config::Dict{String, Any}, μ, σ)
    actornet = ActorNet(config["actornet_in_channels"], config["group_out_channels"], μ, σ;
        kernel_size=config["actornet_kernel_size"], norm=config["actornet_norm"], ng=config["actornet_ng"]
    )

    ple = PolylineEncoder(config["ple_in_channels"], config["ple_hidden_channels"], μ, σ;
        num_layers=config["ple_num_layers"], norm=config["ple_norm"]
    )

    mapenc = MapEncoder(config["mapenc_hidden_channels"], config["mapenc_hidden_channels"], config["mapenc_num_layers"])

    # Fusion setup
    a2m_layers = InteractionGraphModel[]
    m2a_layers = InteractionGraphModel[]
    a2a_layers = InteractionGraphModel[]
    for _ in config["fusion_num_layers"]
        push!(a2m_layers, InteractionGraphModel(config["fusion_n_in"], config["fusion_e_in"], config["fusion_out_dim"];
            num_heads=config["fusion_num_heads"], norm=config["fusion_norm"], ng=config["fusion_ng"]))
        push!(m2a_layers, InteractionGraphModel(config["fusion_n_in"], config["fusion_e_in"], config["fusion_out_dim"];
            num_heads=config["fusion_num_heads"], norm=config["fusion_norm"], ng=config["fusion_ng"]))
        push!(a2a_layers, InteractionGraphModel(config["fusion_n_in"], config["fusion_e_in"], config["fusion_out_dim"]))
    end
    a2m = Chain(a2m_layers...)
    m2m = MapEncoder(config["fusion_out_dim"], config["fusion_out_dim"], 2, ng=1)
    m2a = Chain(m2a_layers...)
    a2a = Chain(a2a_layers...)


    # Prediction head
    pred_head = create_prediction_head(config["fusion_out_dim"], μ, σ)

    LaneletFusionPred(actornet, ple, mapenc, a2m, m2m, m2a, a2a, pred_head)
end

"""
    Forward pass for LaneletFusionPred
Parameters:
    - agt_features: num_scenarios x (channels, timesteps, num_agents)
    - agt_pos: num_scenarios x (2, num_agents)
    - polyline_graphs: graph -> lanelet, node -> vector
        num_scenarios x batched polyline graphs
    - g_heteromaps: graph -> map, node -> lanelet
        num_scenarios x routing_graph
    - llt_pos: num_scenarios x (2, num_lanelets)
"""

# TODO: Use profile to check the efficiency of the forward function
function (model::LaneletFusionPred)(agt_features::Union{Vector{<:AbstractArray}, SubArray{<:AbstractArray}},
    polyline_graphs::Union{Vector{<:GNNGraph}, SubArray{<:GNNGraph}}, g_heteromaps, ga2m, gm2a, ga2a)

    batch_size = length(agt_features)
    # Concatenate agt features for batch processing
    agt_features = cat(agt_features..., dims=3)

    # TODO: Ego encoder
    emb_actor = model.actornet(agt_features)
    # TODO: Consider how to process the polyline_graphs(vector of batched fully-connected graphs)
    emb_lanelets = model.ple(polyline_graphs[1], polyline_graphs[1].x)
    # Duplicate emb_map num_scenarios times
    emb_lanelets = repeat(emb_lanelets,1,batch_size)     # (channels, num_scenarios x num_llts)
    emb_map = model.mapenc(g_heteromaps, emb_lanelets)        # (channels, num_scenarios x num_llts)

    emb_map = model.a2m((emb_map, emb_actor, ga2m))
    emb_map = model.m2m(g_heteromaps, emb_map)      # (channels, num_scenarios x num_llts)

    # Assign the updated emb_map to map2agent_graphs
    emb_actor = model.m2a((emb_actor, emb_map, gm2a))      # (channels, num_scenarios x num_agents)
    emb_actor = model.a2a((emb_actor, emb_actor, ga2a))

    @assert size(emb_actor) == (64, size(agt_features, 3))
    predictions = model.pred_head(emb_actor)      # (2, num_scenarios x num_agents)
    return predictions
end

"""
    Preprocess data for LaneletFusionPred
expect data to include:
    - agent_data: (agt_features_upsampled, agt_features)
    - map_data: (polyline_graphs, g_heteromap, llt_pos)
    - labels: (2, timesteps, num_agents)
"""
function preprocess_data(::LaneletFusionPred, data; overfit::Bool=false, overfit_idx::Int=1)
    num_scenarios = length(data.agent_data.agt_features_upsampled)
    agent_data, map_data, labels = data

    agt_current_pos = [i[:,end,:] for i in agent_data.agt_features_upsampled]
    # duplicate
    polyline_graphs = [map_data.polyline_graphs for _ in 1:num_scenarios]
    g_heteromaps = [map_data.g_heteromap for _ in 1:num_scenarios]
    llt_pos = [map_data.llt_pos for _ in 1:num_scenarios]


    if overfit
        @info "Performing overfitting"
        training_x = (;agt_features_upsampled=agent_data.agt_features_upsampled[overfit_idx,:],
        agt_current_pos=agt_current_pos[overfit_idx,:],
        polyline_graphs=polyline_graphs[overfit_idx,:], g_heteromaps=g_heteromaps[overfit_idx,:], llt_pos=llt_pos[overfit_idx,:])
        training_y = labels[overfit_idx,:]
    else
        training_x = (;agent_data.agt_features_upsampled, agent_data.agt_features, agt_current_pos,
            polyline_graphs, g_heteromaps, llt_pos)
        training_y = labels
    end

    return training_x, training_y
end
