"""
    .gml to GNNHeteroGraph
"""

using Graphs
using MetaGraphs
using NestedGraphsIO
using GraphNeuralNetworks
using GraphIO.GraphML

gml_file_path = joinpath(@__DIR__, "res/location0.osm.gml")
g_meta = open(gml_file_path, "r") do io
    NestedGraphsIO.loadgraph(io, "G", GraphMLFormat(), MGFormat())
end
@assert typeof(g_meta) == MetaDiGraph{Int64, Float64}


left_rel = (:lanelet, :left, :lanelet)
right_rel = (:lanelet, :right, :lanelet)
suc_rel = (:lanelet, :suc, :lanelet)

function extract_src_dst(g, rel_type::String)
    edges = filter_edges(g, :relation, rel_type)
    
    src_list = Int64[]
    dst_list = Int64[]
    for e in edges
        push!(src_list, src(e))
        push!(dst_list, dst(e))
    end
    return src_list, dst_list
end

g = GNNHeteroGraph(
    right_rel => extract_src_dst(g_meta, "Right"),
    left_rel => extract_src_dst(g_meta, "Left"),
    suc_rel => extract_src_dst(g_meta, "Successor")
)