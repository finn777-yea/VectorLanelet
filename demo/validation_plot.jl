using VectorLanelet
using Flux
using Plots
using JLD2
using MLUtils

function agent_features_upsample(agt_features)
      agt_features = permutedims(agt_features, (2, 1, 3))
      agt_features = upsample_linear(agt_features, size=10)
      agt_features = permutedims(agt_features, (2, 1, 3))
      return agt_features
end

function setup_model(config::Dict{String, Any}, model_state_name::String)
    # 1. First prepare the normalization parameters
    lanelet_roadway, g_meta = load_map_data()
    _, _, μ, σ = prepare_map_features(lanelet_roadway, g_meta)
    
    # 2. Initialize model on CPU with the same parameters used during training
    model = LaneletPredictor(config, μ, σ)
    Flux.testmode!(model)
    
    # 3. Load the saved state (which was saved on CPU)
    model_state_path = joinpath(@__DIR__, "../models/$(model_state_name).jld2")
    loaded_model_state = JLD2.load(model_state_path, "model_state")
    Flux.loadmodel!(model, loaded_model_state)
    
    # 4. Only after loading, move to GPU if needed
    return model |> gpu
end

# Data preparation
lanelet_roadway, g_meta = load_map_data()

agt_features, pos_agt, labels = prepare_agent_features(lanelet_roadway)
agt_features_upsampled = agent_features_upsample(agt_features) |> gpu
pos_agt = pos_agt |> gpu
labels = labels |> gpu

polyline_graphs, g_heteromap, μ, σ = prepare_map_features(lanelet_roadway, g_meta) |> gpu

# Model inference
include("../src/config.jl")

model_state_name = "LaneletPredictor_with_Transformer_2025-01-06T17:46:50.287"
model = setup_model(config, model_state_name)
prediction = model(agt_features_upsampled, polyline_graphs, g_heteromap)

VectorLanelet.plot_predictions(cpu(agt_features), cpu(labels), cpu(prediction), save=true)
