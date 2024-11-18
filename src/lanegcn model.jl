config = Dict{String, Any}()
config["n_actor"] = 128
config["n_map"] = 128
config["num_scales"] = 6
config["din_actor"] = 2



# ------ Net -------
struct Net
    config
    actor_net
    map_net
    a2m
    m2m
    m2a
    a2a
    pred_net
end

function Net(config)
    actor_net = ActorNet(config)
    map_net = MapNet(config)
    a2m = A2M(config)
    m2m = M2M(config)
    m2a = M2A(config)
    a2a = A2A(config)
    pred_net = PredNet(config)

    Net(config, actor_net, map_net, a2m, m2m, m2a, a2a, pred_net)
end

"""
expect:
    actors in batch
    actor_idcs::List[Tensor]
    actor_ctrs

    graphs in batch
    node_idcs
    node_ctrs

"""
function (net::Net)(data::Dict)
    # construct actor feature
    # actors: (features, timesteps, num_actors_in_batch)
    actors, actor_idcs = actor_gather(gpu(data["feat"]))
    actor_ctrs = gpu(data["ctrs"])
    actors = net.actor_net(actors)

    # construct map feature
    nodes, node_idcs, node_ctrs = net.map_net(data["graph"])
    # actor-map fusion cycle
    nodes = net.a2m(nodes, graph, actors, actor_idcs, actor_ctrs)

end



# combine a batch of actors into one
function actor_gather(actors::Vector)
    batch_size = length(actors)
    num_actors = [length(x) for x in actors]

    # expect actors: (features, timesteps, num_actors)
    actors = cat(actors..., dims=3)


    actor_idcs = []
    count = 0
    for i in range(1, batch_size)
        idcs = range(count, count + num_actors[i]-1) |> collect |> gpu
        push!(actor_idcs, idcs)
        count += num_actors[i]
    end
    return actors, actor_idcs
end


# ------ ActorNet -------
struct ActorNet
    cfg::Dict
    groups
    lateral
    output
end

function ActorNet(config::Dict)
    norm = "GN"
    ng = 1

    n_in = config["din_actor"]
    n_out = [32, 64, 128]
    blocks = [Res1d, Res1d, Res1d]
    num_blocks = [2, 2, 2]

    ### groups
    groups = []
    for i in eachindex(num_blocks)                      # i is the group index
        group = []
        if i == 1
            push!(group, blocks[i](n_in, n_out[i], norm=norm, ng=ng))
        else
            push!(group, blocks[i](n_in, n_out[i], stride=2, norm=norm, ng=ng))
        end
        # initialize the sequent blocks in this group
        for j in range(2, num_blocks[i])                # j is the block index
            push!(group, blocks[i](n_out[i], n_out[i], norm=norm, ng=ng))
        end
        push!(groups, Chain(group...))
        n_in = n_out[i]
    end
    groups = Chain(groups...)

    ### lateral
    lateral = []
    n_actor = config["n_actor"]
    for i in eachindex(num_blocks)
        lat_connection = Conv1d(n_out[i], n_actor, norm=norm, ng=ng, act=false)
        push!(lateral, lat_connection)
    end
    lateral = Chain(lateral...)

    output = Res1d(n_actor, n_actor, norm=norm, ng=ng)
    ActorNet(config, groups, lateral, output)
end

# expect actors:(traj length, features=3, batch)
function (actornet::ActorNet)(actors)
    @assert size(actors, 2) == config["din_actor"]
    out = actors

    outputs = []

    for i in eachindex(actornet.groups)
        out = actornet.groups[i](out)
        push!(outputs, out)
    end

    out = actornet.lateral[end](outputs[end])
    for i in range(length(outputs)-1, 1, step=-1)
        # Upsample the temperal dimension with scale factor 2
        out = upsample_linear(out, 2, align_corners=false)
        @show size(out)
        out .+= actornet.lateral[i](outputs[i])
    end

    # extract the last element of the temperal sequence
    out = actornet.output(out)[end,:,:]
    return out
end



# ------ MapNet -------
struct MapNet
    cfg
    input
    seg
    fuse
end

Flux.@layer MapNet

function MapNet(config)
    n_map = config["n_map"]
    n_scales = config["num_scales"]
    norm = "GN"
    ng = 1

    input = Chain(
        Dense(64=>n_map, relu),
        Linear(n_map, n_map, norm=norm, ng=ng, act=false)
    )

    seg = Chain(
        Dense(2=>n_map, relu),
        Linear(n_map, n_map, norm=norm, ng=ng, act=false)
    )

    fuse = Dict()
    ks = ["ctr", "norm", "ctr2", "heteroconv"]
    for key in ks
        fuse[key] = []
    end

    # Aggregate heterogenously
    left_rel = (:lanelet, :left, :lanelet)
    right_rel = (:lanelet, :right, :lanelet)
    adj_left_rel = (:lanelet, :adj_left, :lanelet)
    adj_right_rel = (:lanelet, :adj_right, :lanelet)
    suc_rel = (:lanelet, :suc, :lanelet)

    heteroconv = HeteroGraphConv(
            left_rel => GATConv(n_map=>n_map),
            right_rel => GATConv(n_map=>n_map),
            adj_left_rel => GATConv(n_map=>n_map),
            adj_right_rel => GATConv(n_map=>n_map),
            suc_rel => GATConv(n_map=>n_map)
        )

    for i in 1:4
        for key in keys(fuse)
            if key in ["norm"]
                push!(fuse[key], GroupNorm(n_map, gcd(ng, n_map)))
            elseif key in ["ctr2"]
                push!(fuse[key], Linear(n_map, n_map, norm=norm, ng=ng))
            elseif key in ["heteroconv"]
                push!(fuse[key], heteroconv)
            else
                push!(fuse[key], Dense(n_map=>n_map))
            end
        end
    end

    for key in keys(fuse)
        fuse[key] = Chain(fuse[key]...)
    end

    MapNet(config, input, seg, fuse)
end

"""
Expect heterograph with 5 relation types
Original Input:
    - g[:lanelet].ctrs: center position of the lane
    - g[:lanelet].feat: start/end position of the lane 

New Input:
    - g[:lanelet].x:(64, num_lanelets)
"""

function (mapnet::MapNet)(g::GNNHeteroGraph)
    # TODO: check if node features are empty

    ### Node features
    feat = mapnet.input(g[:lanelet].x)
    feat = relu.(feat)

    ### Fusion aggregation
    # TODO: check if a certain relation exists

    # feat: (128, num_nodes)
    res = feat

    for i in eachindex(mapnet.fuse["ctr"])
        temp = (;lanelet = mapnet.fuse["ctr"][i](feat))
        temp = mapnet.fuse["heteroconv"][i](g, temp).lanelet

        feat = mapnet.fuse["norm"][i](temp)
        feat = relu.(feat)
        feat = mapnet.fuse["ctr2"][i](feat)
        feat = feat + res
        feat = relu.(feat)

        res = feat
    end

    return feat
end


    # for i in eachindex(mapnet.fuse["ctr"])      # 4 LaneConv blocks
    #     temp = mapnet.fuse["ctr"][i](feat)
    #     # TODO: aggregate pre and suc
    #     for key in keys(mapnet.fuse)
    #         if startswith(key, "pre") || startswith(key, "suc")
    #             k1 = key[1:3]
    #             k2 = Int(key[])
    #         end
    #     end

    #     # aggregate left/right
    #     left_rel = (:lanelet, :left, :lanelet)
    #     if left_rel in g.etypes
    #         # msg transformation of the incoming nodes
    #         msg(xi, xj, e) = mapnet.fuse["left"][i](xj)
    #         x_left = propagate(msg, g_left, +, xj=g.x)       # simply summation of the incoming msgs
    #         # could use degree(g) for validation

    #     end

    #     if g_right.num_nodes !== nothing
    #         msg(xi, xj, e) = mapnet.fuse["right"][i](xj)
    #         x_right = propagate(msg, g_right, +, xj=g_right.x)

    #     end
    # end



# ------ A2M -------
struct A2M
    config
    meta
    att
end

function A2M(config)
    n_map = config["n_map"]
    norm = "GN"
    ng = 1

    meta = Linear(n_map +4, n_map, norm=norm, ng=ng)
    att = []
    for i in range(1,2)
        push!(att, Att(n_map, config["n_actor"]))
    end
    att = Chain(att...)
    A2M(config, meta, att)
end

function (a2m::A2M)(node_feat, graph, actors, actor_idcs, actor_ctrs)
    # meta: (4, num_nodes)
    meta = cat(
        (graph.turn,
        # expect graph.control/intersect as Vector
        Flux.unsqueeze(graph.control,1),
        Flux.unsqueeze(graph.intersect,1)
        ),
        dims=1
    )
    node_feat = cat((node_feat,meta), dims=1)
    node_feat = a2m.meta(node_feat)

    for i in eachindex(a2m.att)
        feat = a2m.att[i](
            node_feat,
            graph.idcs.
            graph.ctrs,
            actors,
            actor_idcs,
            actor_ctrs,
            a2m.config["actor2map_dist"]
        )
    end
    return feat     # 
end


# ------ M2M -------
    """
    Similar to MapNet, but without input and seg module
    """
struct M2M
    config
    fuse
end

function M2M(config)
    n_map = config["n_map"]
    norm = "GM"
    ng = 1

    ks = ["ctr", "norm", "ctr2", "left", "right"]
    for i in range(1, config["num_scales"])
        push!(ks, "pre$(i)")
        push!(ks, "suc$(i)")
    end

    for i in range(1, config["num_scales"])
        push!(ks, "pre$(i)")
        push!(ks, "suc$(i)")
    end

    fuse = Dict()
    for key in ks
        fuse[key] = []
    end

    for i in range(1,4)
        for key in keys(fuse)
            if key in ["norm"]
                push!(fuse[key], GroupNorm(n_map, gcd(ng, n_map)))
            elseif key in ["ctr2"]
                push!(fuse[key], Linear(n_map, n_map, norm=norm, ng=ng))
            else
                push!(fuse[key], Dense(n_map, n_map))
            end
        end
    end

    for key in keys(fuse)
        fuse[key] = Chain(fuse[key]...)
    end
end

function (m2m::M2M)(feat, graph)
    res = feat


end



# ------ M2A -------
struct M2A
    config
    att
end

function M2A(config)
    n_map = config["n_map"]
    n_actor = config["n_actor"]
    norm = "GN"
    ng = 1

    meta = Linear(n_map +4, n_map, norm=norm, ng=ng)
    att = []
    for i in range(1,2)
        push!(att, Att(n_actor, n_map))
    end
    att = Chain(att...)
    M2A(config, att)
end

function (m2a::M2A)(actors, acotr_idcs, actor_ctrs, node_idcs, node_ctrs)

end



# ------ A2A -------

struct A2A
    config
    att
end

function A2A(config)
    n_map = config["n_map"]
    n_actor = config["n_actor"]
    norm = "GN"
    ng = 1

    meta = Linear(n_map +4, n_map, norm=norm, ng=ng)
    att = []
    for i in range(1,2)
        push!(att, Att(n_actor, n_map))
    end
    att = Chain(att...)
    A2M(config, meta, att)
end

function (a2a::A2A)(actors, actor_idcs, actor_ctrs)

end








# ------ Att -------
    """
    Attention block to pass context nodes information to target nodes
    This is used in Actor2Map, Actor2Actor, Map2Actor and Map2Map
    """
struct Att
    agt
    linear
    query
    dist
    ctx
end

function Att(n_agt::Int, n_ctx::Int)
    agt = Dense(n_agt => n_agt)
    linear = Linear(n_agt, n_agt, norm = norm, ng=ng, act=false)
    dist = Chain(
        Dense(2 => n_ctx),
        relu,
        Linear(n_ctx, n_ctx, norm=norm, ng=ng)
    )
    query = Linear(n_agt,n_ctx, norm=norm, ng=ng)
    ctx = Chain(
        Linear(3*n_ctx, n_agt, norm=norm, ng=ng),
        Dense(n_agt, n_agt)
    )
    Att(agt, linear, query, dist, ctx)
end

# Each batch corresponds to a scenario, containing different number of actors and lanes
function (att::Att)(agts, agt_idcs::Vector, agt_ctrs, ctx, ctx_idcs::Vector, ctx_ctrs, dist_th::Float32)
    res = agt_ctrs
    if length(ctx) == 0
        agts = att.agt(agts)
        agts = relu.(agts)
        agts = att.linear(agts)
        agts += res
        agts = relu.(agts)
        return agts
    end

    batch_size = length(agt_idcs)
    hi, wi = [], []
    hi_count, wi_count = 0, 0

    for i in 1:batch_size
        # agt_ctrs[:,:,i]: 2 x num_agts
        # ctx_ctrs[:,:,i]: 2 x num_ctxs
        # TODO: reshape the tensor to compute pairwise distances
        dist = reshape(agt_ctrs[:,:,i], 2,1,:) - reshape(ctx_ctrs[:,:,i], 2,:,1)
        
        dist = sqrt(sum((dist^2), dims=1))      # dist: num_ctxs x num_agts
        mask = dist .<= dist_th

        idcs = findall(!iszero, mask)       # return a CartesianIndex
        if length(idcs) == 0
            continue
        end

        # TODO: maintain global indices of both agt and ctx
        # hi = ...
        # wi = ...
    end

    dist = agt_ctrs[hi] - ctx_ctrs[wi]
    dist = att.dist(dist)

    query = att.query(agts[hi])
    return agts
end

