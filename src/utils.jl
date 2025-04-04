function extract_gml_src_dst(g, rel_type::String)
    edges = filter_edges(g, :relation, rel_type)

    src_list = Int64[]
    dst_list = Int64[]

    isempty(edges) && return src_list, dst_list
    if !isempty(edges)
        for e in edges
            push!(src_list, src(e))
            push!(dst_list, dst(e))
        end
    else
        @info "No edges found for relation type $(rel_type)"
    end
    return src_list, dst_list
end

function create_gnn_heterograph(g_meta::MetaDiGraph, routing_relations::Vector{String})
    pairs = []
    for rel in routing_relations
        src_list, dst_list = extract_gml_src_dst(g_meta, rel)
        isempty(src_list) && isempty(dst_list) && continue
        pair = (:lanelet, Symbol(rel), :lanelet) => (src_list, dst_list)
        push!(pairs, pair)
    end
    g_heteromap = GNNHeteroGraph(
        pairs...,
        dir = :out
    )
    return g_heteromap
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
function create_filtered_interaction_graph(
    agt_pos::Union{Vector{<:AbstractArray}, SubArray{<:AbstractArray}},
    ctx_pos::Union{Vector{<:AbstractArray}, SubArray{<:AbstractArray}},
    distance_threshold::Real,
    normalize_dist::Bool=false,
    store_edata::Bool=true,
    add_self_loops::Bool=true
    )

    # Process each sample independently
    @assert length(agt_pos) == length(ctx_pos) "Number of samples must match"

    # Calculate cumulative sums for offset calculation
    agt_counts = cumsum([0; [size(agt, 2) for agt in agt_pos]])
    ctx_counts = cumsum([0; [size(ctx, 2) for ctx in ctx_pos]])
    total_agts = agt_counts[end]
    total_ctxs = ctx_counts[end]

    # scr for ctx nodes, while dst for agt nodes
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

            # Note: use Vector as src and dst to construct GNNGraph is faster
            all_dst = vcat(all_dst, agt_idc .+ agt_offset)
            all_src = vcat(all_src, ctx_idc .+ total_agts .+ ctx_offset)
        end
    end
    agt_pos = reduce(hcat, agt_pos)
    ctx_pos = reduce(hcat, ctx_pos)
    global_pos = hcat(agt_pos, ctx_pos)
    dist = global_pos[:,all_dst] - global_pos[:,all_src]    # (2, num_edges)

    # Handle empty case
    # no connection in each scenario
    if isempty(all_src)
        graph = GNNGraph(total_agts + total_ctxs, dir=:in)
        if add_self_loops
            self_loop_nodes = [1:graph.num_nodes;]
            # TODO: hard-coded the edge data being 2 dim
            e_dim = 2
            graph = add_edges(graph, self_loop_nodes, self_loop_nodes)
            graph.edata.e = zeros(Float32, e_dim, length(self_loop_nodes))
        end
    else
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
            dir = :out
        )
        # Add self-loops with edge data 0
        if add_self_loops
            self_loop_nodes = [1:graph.num_nodes;]
            # TODO: hard-coded the edge data being 2 dim
            graph = add_edges(graph, self_loop_nodes, self_loop_nodes, edata=zeros(Float32, 2, length(self_loop_nodes)))
        end
    end


    return graph
end

function create_interaction_graphs(agent_pos, llt_pos,
    a2m_dist_thrd::Real,
    m2a_dist_thrd::Real,
    a2a_dist_thrd::Real,
    )

    # Create interaction graphs
    ga2m_all = create_filtered_interaction_graph(llt_pos, agent_pos, a2m_dist_thrd)
    ga2m_all.num_edges == ga2m_all.num_nodes && @info "No connection from agents to map"

    gm2a_all = create_filtered_interaction_graph(agent_pos, llt_pos, m2a_dist_thrd)
    gm2a_all.num_edges == gm2a_all.num_nodes && @info "No connection from map to agents"

    ga2a_all = create_filtered_interaction_graph(agent_pos, agent_pos, a2a_dist_thrd)
    ga2a_all.num_edges == ga2a_all.num_nodes && @info "No connection from agents to agents"
    return ga2m_all, gm2a_all, ga2a_all
end
