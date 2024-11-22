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

function calculate_mean_and_std(data; dims)
    μ = mean(data, dims=dims)
    σ = std(data, dims=dims)
    return μ, σ
end