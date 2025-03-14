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

"""
    Create a GNNGraph for each sample in the batch
Parameters:
- agt_pos: [(2, num_agts_in_sample1), (2, num_agts_in_sample2), ...]
- ctx_pos: [(2, num_ctxs_in_sample1), (2, num_ctxs_in_sample2), ...]
- distance_threshold: Distance threshold for creating edges
- normalize_dist: Whether to normalize the distance
Returns:
- g: GNNGraph for each sample in the batch
"""
function create_filtered_interaction_graph(agt_pos::Vector{T}, ctx_pos::Vector{T},
    distance_threshold::Real,
    normalize_dist::Bool=false,
    store_edata::Bool=false,
    add_self_loops::Bool=false
    ) where T <: AbstractMatrix

    # Process each sample independently
    @assert length(agt_pos) == length(ctx_pos) "Number of samples must match"

    # Calculate cumulative sums for offset calculation
    agt_counts = cumsum([0; [size(agt, 2) for agt in agt_pos]])
    ctx_counts = cumsum([0; [size(ctx, 2) for ctx in ctx_pos]])
    total_agts = agt_counts[end]
    total_ctxs = ctx_counts[end]

    all_src = Int[]
    all_dst = Int[]

    # Process each sample independently
    for (i, (agt, ctx)) in enumerate(zip(agt_pos, ctx_pos))
        # Calculate pairwise distances
        dist = reshape(agt, 2, :, 1) .- reshape(ctx, 2, 1, :)    # (2, num_agt, num_ctx)
        dist = sqrt.(sum(dist.^2, dims=1))[1,:,:]                # (num_agt, num_ctx)

        mask = dist .<= distance_threshold

        # findall() for BitMatrix much faster than CuArray
        indices = findall(mask) |> cpu

        # Retrieve across-samples src/dst indices
        if !isempty(indices)
            indices = VectorLanelet.indices_to_matrix(indices)
            agt_idc = indices[1,:]
            ctx_idc = indices[2,:]
            agt_offset = agt_counts[i]
            ctx_offset = ctx_counts[i]

            # Replace append! with non-mutating vcat
            # Note: use Vector as src and dst to construct GNNGraph is faster
            all_src = vcat(all_src, agt_idc .+ agt_offset)
            all_dst = vcat(all_dst, ctx_idc .+ total_agts .+ ctx_offset)
        end
    end
    agt_pos = reduce(hcat, agt_pos)
    ctx_pos = reduce(hcat, ctx_pos)
    global_pos = hcat(agt_pos, ctx_pos)
    dist = global_pos[:,all_src] - global_pos[:,all_dst]    # (2, num_edges)

    # Handle empty case
    # no connection in each scenario
    if isempty(all_src)
        return GNNGraph(total_agts + total_ctxs, dir=:in)
    end

    # Process edge data
    if normalize_dist
        μ, σ = calculate_mean_and_std(dist, dims=2)       # dist: 2, num_edges
        dist = (dist .- μ) ./ (σ .+ 1e-6)
    end

    # Create single graph with all samples
    graph = GNNGraph(
        (all_src, all_dst),
        num_nodes = total_agts + total_ctxs,
        edata = store_edata ? dist : nothing,
        dir = :in       # msg flow from dst to src
    )
    # Add self-loops with edge data 0
    if add_self_loops
        self_loop_nodes = [1:graph.num_nodes;]
        graph = add_edges(graph, self_loop_nodes, self_loop_nodes, edata=zeros(Float32, size(dist, 1), length(self_loop_nodes)))
    end

    return graph
end

