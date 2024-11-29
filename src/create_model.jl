"""
    - in_channels: input channels of the actor net
    - num_groups: number of groups in the actor net
"""

function create_actor_net(in_channels, group_out_channels::Vector{Int})
    out_channels = group_out_channels[end]
    num_groups = length(group_out_channels)
    groups = []
    for i in 1:num_groups
        if i == 1
            push!(groups, create_group_block(in_channels, group_out_channels[i]))
        else
            push!(groups, create_group_block(group_out_channels[i-1], group_out_channels[i])) 
        end
    end
    group_chain = Chain(groups...)

    lateral = [Conv((1,), group_out_channels[i]=>out_channels, stride=1) for i in 1:num_groups]
    
    output_block = create_residual_block(out_channels, out_channels, stride=1)
    
    actor_net = Chain(
        group_chain,
    
        # TODO:Lateral connection
        # output out group_chain is only the output of the last group

        output_block,
        x -> @view x[end,:,:]
    )

    return actor_net
end

function create_vector_subgraph(g::GNNGraph, in_channels, out_channels, num_layers::Int=3)
    layers = []
    for i in 1:num_layers
        push!(layers, create_node_encoder(in_channels, out_channels))
        in_channels = out_channels * 2
    end
    output_layer = Dense(out_channels, out_channels)

    # TODO: maxpooling
    clusters = graph_indicator(g)
end


function create_map_net(in_channels, out_channels, num_layers::Int=3)
    layers = []
    for i in 1:num_layers
        push!(layers, )
    end
end
