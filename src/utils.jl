function extract_gml_src_dst(g, rel_type::String)
    edges = filter_edges(g, :relation, rel_type)

    src_list = Int64[]
    dst_list = Int64[]
    for e in edges
        push!(src_list, src(e))
        push!(dst_list, dst(e))
    end
    return src_list, dst_list
end

"""
    Calculate the mean and standard deviation using map features(x, y coordinates)
    data: (2, num_vectors)
"""
function calculate_mean_and_std(data; dims=2)
    μ = mean(data, dims=dims) |> vec
    σ = std(data, dims=dims) |> vec
    return μ, σ
end

function indices_to_matrix(indices)
    tuple_idc = map(Tuple, indices)
    return reduce(hcat, map(collect, tuple_idc))
end

function plot_predictions(agent_features, labels, pred; save::Bool=false)
    indices = 1:2:size(agent_features, 3)
    x_t1 = agent_features[1, 1, indices]  # x-coordinates at timestep 1
    y_t1 = agent_features[2, 1, indices]  # y-coordinates at timestep 1
    x_t2 = agent_features[1, 2, indices]  # x-coordinates at timestep 2
    y_t2 = agent_features[2, 2, indices]  # y-coordinates at timestep 2

    x_end = labels[1, indices]
    y_end = labels[2, indices]

    # Initialize the plot
    plot(title="Agent Positions at Two Timesteps",
        xlabel="X", ylabel="Y", legend=:top, grid=:true, size=(800, 600))

    # Plot timestep 1 as a scatter plot with distinct styling
    scatter!(x_t1, y_t1,
            label="Timestep 1",
            markershape=:circle,
            color=:blue,
            alpha=0.8,
            markersize=4)

    # Plot timestep 2 as a scatter plot with arrows pointing from timestep 1
    scatter!(x_t2, y_t2,
            label="Timestep 2",
            markershape=:diamond,
            color=:red,
            alpha=0.8,
            markersize=4)

    scatter!(x_end, y_end,
            label="End",
            markershape=:star,
            color=:green,
            alpha=0.8,
            markersize=4)

    # pred = predictor(agent_features, polyline_graphs, g_heteromaps)
    x_pred = pred[1, indices]
    y_pred = pred[2, indices]
    scatter!(x_pred, y_pred,
            label="Prediction",
            markershape=:circle,
            color=:yellow,
            alpha=0.8,
            markersize=4)

    for i in eachindex(x_t1)
        plot!([x_end[i], x_pred[i]], [y_end[i], y_pred[i]],
                color=:gray, linewidth=1.0, alpha=0.6, label=""
        )
    end
    if save
        savefig("predictions.png")
    else
        display(current())
    end
end