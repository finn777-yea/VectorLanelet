"""
Encoder for static actor features
    - preprocess: normalize the actor positions feature using μ and σ computed from the map features
    - layers: 2 dense layers with layer normalization
"""
function create_static_actor_net(
    in_dim,
    hidden_dim,
    μ::Union{Vector, CuArray},
    σ::Union{Vector, CuArray},
    num_layers=2,
    norm="LN",
)
    preprocess = Chain(
        Base.Fix2(.-, μ),
        Base.Fix2(./, σ)
    )
    layers = []
    for _ in 1:num_layers
        push!(layers, create_node_encoder(in_dim, hidden_dim, norm))
        in_dim = hidden_dim
    end
    return Chain(preprocess, layers...)
end

struct LaneletStaticFusionPred
    actornet::Chain
    ple::PolylineEncoder
    mapenc::MapEncoder
    a2m::Chain
    m2m::MapEncoder
    m2a::Chain
    a2a::Chain
    pred_head
end

Flux.@layer LaneletStaticFusionPred

function LaneletStaticFusionPred(config::Dict{String, Any}, μ, σ)
    actornet = create_static_actor_net(config["actornet_in_channels"], config["actornet_hidden_channels"], μ.μ_agt, σ.σ_agt)

    ple = PolylineEncoder(config["ple_in_channels"], config["ple_hidden_channels"], μ.μ_map, σ.σ_map;
        num_layers=config["ple_num_layers"], norm=config["ple_norm"]
    )

    mapenc = MapEncoder(
        config["mapenc_hidden_channels"],
        config["mapenc_hidden_channels"],
        config["mapenc_num_layers"],
        config["routing_relations"]
        )

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
    pred_head = create_prediction_head(config["fusion_out_dim"], μ.μ_map, σ.σ_map)

    LaneletStaticFusionPred(actornet, ple, mapenc, a2m, m2m, m2a, a2a, pred_head)
end

"""
    Forward pass for LaneletStaticFusionPred
Parameters:
    - agt_features: num_scenarios x (channels, num_agents)
    - polyline_graphs: graph -> lanelet, node -> vector
        num_scenarios x batched polyline graphs
    - g_heteromaps: graph -> map, node -> lanelet
        num_scenarios x routing_graph
    - llt_pos: num_scenarios x (2, num_lanelets)
"""

# TODO: Use profile to check the efficiency of the forward function
function (model::LaneletStaticFusionPred)(
    agt_features::Union{Vector{<:AbstractArray}, SubArray{<:AbstractArray}},
    polylines_graphs,
    g_heteromaps,
    ga2m, gm2a, ga2a
    )
    batch_size = length(agt_features)
    # Concatenate agt features for batch processing
    agt_features = cat(agt_features..., dims=2)

    # TODO: Ego encoder
    emb_actor = model.actornet(agt_features)
    # TODO: Consider how to process the polyline_graphs(vector of batched fully-connected graphs)
    emb_lanelets = model.ple(polylines_graphs, polylines_graphs.x)

    emb_map = model.mapenc(g_heteromaps, emb_lanelets)        # (channels, num_scenarios x num_llts)
    emb_map = model.a2m((emb_map, emb_actor, ga2m))
    emb_map = model.m2m(g_heteromaps, emb_map)      # (channels, num_scenarios x num_llts)

    # Assign the updated emb_map to map2agent_graphs
    emb_actor = model.m2a((emb_actor, emb_map, gm2a))      # (channels, num_scenarios x num_agents)
    emb_actor = model.a2a((emb_actor, emb_actor, ga2a))

    @assert size(emb_actor) == (64, size(agt_features, 2))
    predictions = model.pred_head(emb_actor)      # (2, num_scenarios x num_agents)
    return predictions
end

function preprocess_data(::LaneletStaticFusionPred, data; overfit::Bool=false, overfit_idx::Int=1)
    num_scenarios = length(data.agent_data.agt_features_upsampled)
    agent_data, map_data, labels = data

    agt_current_pos = [i[:,end,:] for i in agent_data.agt_features_upsampled]
    agt_features = agt_current_pos
    # duplicate
    polyline_graphs = [map_data.polyline_graphs for _ in 1:num_scenarios]
    g_heteromaps = [map_data.g_heteromap for _ in 1:num_scenarios]
    llt_pos = [map_data.llt_pos for _ in 1:num_scenarios]


    if overfit
        @info "Performing overfitting"
        training_x = (;agt_features=agt_features[overfit_idx,:], agt_current_pos=agt_current_pos[overfit_idx,:],
        polyline_graphs=polyline_graphs[overfit_idx,:], g_heteromaps=g_heteromaps[overfit_idx,:], llt_pos=llt_pos[overfit_idx,:])
        training_y = labels[overfit_idx,:]
    else
        training_x = (;agt_features, agt_current_pos,
            polyline_graphs, g_heteromaps, llt_pos)
        training_y = labels
    end

    return training_x, training_y
end

"""
Called after dataloader, take care of the interface with the model
1. batching after dataloader
2. create interaction interaction graphs

expect the train_data_x to contain the following fields:
- agt_features: num_scenarios x (2, num_agents)
- agt_pos: num_scenarios x (2, num_agents)
- polylines_graphs: num_scenarios x  polyline graphs
- hetero_routing_graphs: num_scenarios x routing_graph
- llt_pos: num_scenarios x (2, num_lanelets)
"""
function collate_data(::LaneletStaticFusionPred, train_data_x::NamedTuple, config::Dict{String, Any})
    ga2m, gm2a, ga2a = create_interaction_graphs(
        train_data_x.agt_pos,
        train_data_x.llt_pos,
    config["agent2map_dist_thrd"],
    config["map2agent_dist_thrd"],
    config["agent2agent_dist_thrd"]
    )

    # return only the input args for the model forward pass
    return(
        train_data_x.agt_features,
        Flux.batch(train_data_x.polylines_graphs),
        Flux.batch(train_data_x.hetero_routing_graphs),
        ga2m,
        gm2a,
        ga2a
    )
end