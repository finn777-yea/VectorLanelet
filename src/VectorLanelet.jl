module VectorLanelet

using Random
using Lanelet2
using Lanelet2.Routing
using Lanelet2.Core
using AutomotiveSimulator
using Flux
using Zygote
using CUDA
using GraphNeuralNetworks
using Graphs
using MetaGraphs
using Transformers
using Statistics
using LinearAlgebra
using Plots

# Zygote.@nograd Flux.batch

export create_residual_block,
       create_group_block,
       create_node_encoder,
       create_hetero_conv,
       create_agt_preprocess_block,
       create_map_preprocess_block,
       create_transformer_block,
       create_prediction_head
include("layers_remaster.jl")


export ActorNet_Simp,
       PolylineEncoder,
       MapEncoder
include("common_blocks.jl")

export load_map_data,
       prepare_agent_features,
       prepare_map_features,
       agent_features_upsample
include("dataset_processing.jl")

export extract_gml_src_dst
include("utils.jl")

include("validation_plot.jl")

export ActorNet_Simp,
       PolylineEncoder,
       MapEncoder,
       LaneletPredictor
include("lanelet_predictor.jl")

export create_filtered_interaction_graphs,
       InteractionGraphModel,
       LaneletFusionPred
include("lanelet_fuse_predictor.jl")

export logging_callback,
       save_model_state
include("training_utils.jl")

end     # VectorLanelet