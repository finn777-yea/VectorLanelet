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