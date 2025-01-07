module VectorLanelet

using Random
using Lanelet2
using Lanelet2.Routing
using Lanelet2.Core
using AutomotiveSimulator
using Flux
using CUDA
using GraphNeuralNetworks
using Graphs
using MetaGraphs
using Transformers
using Statistics
using LinearAlgebra
using Plots

export Res1d,
       Conv1d,
       Linear,
       PredictionHead
include("layers.jl")

export create_residual_block,
       create_group_block,
       create_node_encoder,
       create_hetero_conv,
       create_agt_preprocess_block,
       create_map_preprocess_block,
       create_transformer_block,
       create_prediction_head
include("layers_remaster.jl")

# export create_actor_net,
#        create_vector_subgraph
# include("create_model.jl")

# export VectorSubGraph
# include("vectorSubgraph.jl")

export extract_gml_src_dst
include("utils.jl")

export ActorNet_Simp,
       PolylineEncoder,
       MapEncoder,
       LaneletPredictor
include("lanelet_predictor.jl") 

export logging_callback,
       save_model_state,
       load_map_data,
       prepare_agent_features,
       prepare_map_features
include("training_utils.jl")

# Export spatial attention functionality
export create_distance_mask
include("spatial_attention.jl")

end     # VectorLanelet