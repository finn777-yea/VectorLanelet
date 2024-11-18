struct ActorNet_Simp
    cfg::Dict
    groups
    lateral
    output
end

Flux.@layer ActorNet_Simp

function ActorNet_Simp(config::Dict)
    norm = "GN"
    ng = 1

    n_in = config["din_actor"]
    n_out = [64, 128]
    blocks = [Res1d, Res1d]
    num_blocks = [2, 2]

    ### groups
    groups = []
    for i in eachindex(num_blocks)      # i is the group index
        group = []
        if i == 1
            push!(group, blocks[i](n_in, n_out[i], kernel_size=2, norm=norm, ng=ng))
        else
            push!(group, blocks[i](n_in, n_out[i], kernel_size=2, stride=2, norm=norm, ng=ng))
        end
        # initialize the sequent blocks in this group
        for j in range(2, num_blocks[i])        # j is the block index
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
    ActorNet_Simp(config, groups, lateral, output)
end

# Expect actors:(traj length, din_actor=2, batch)
function (actornet::ActorNet_Simp)(actors)
    out = actors
    # TODO: Replace push! by non-mutating array operation
    outputs = Flux.activations(actornet.groups, out)

    out = actornet.lateral[end](outputs[end])
    for i in range(length(outputs)-1, 1, step=-1)
        out = upsample_linear(out, 2, align_corners=false)
        out = out .+ actornet.lateral[i](outputs[i])
    end

    # extract the last element of the temperal sequence
    out = @view actornet.output(out)[end,:,:]
    
    return out
end