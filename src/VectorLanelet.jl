module VectorLanelet

using Random
using Lanelet2
using Lanelet2.Routing
using Lanelet2.Core
using AutomotiveSimulator
using Flux
# using CUDA
using GraphNeuralNetworks
using Graphs
using MetaGraphs
using Transformers
using Statistics

export Res1d,
       Conv1d,
       Linear,
       PredictionHead
include("layers.jl")

export create_residual_block,
       create_group_block,
       create_node_encoder
include("layers_remaster.jl")

export create_actor_net
include("create_model.jl")

export VectorSubGraph
include("vectorSubgraph.jl")

export ActorNet_Simp
include("actor_simp.jl")

export MapNet
include("mapnet.jl")

export extract_gml_src_dst
include("utils.jl")


export LaneletPredictor
include("lanelet_predictor.jl") 

export logging_callback, loss_fn
include("training_utils.jl")

end     # VectorLanelet