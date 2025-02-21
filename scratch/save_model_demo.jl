using VectorLanelet
using Flux
using BSON

include("../src/config.jl")
model = LaneletPredictor(config)

# ------ BSON -------
# Save the model
model_path = joinpath(@__DIR__, "../res/model.bson")
BSON.@save model_path model

# Load the model
model = BSON.@load model_path

# ------ JLD2 -------
using JLD2
model_state = Flux.state(model)
model_state_path = joinpath(@__DIR__, "../res/model_state.jld2")
jldsave(model_state_path; model_state)

loaded_model_state = JLD2.load(model_state_path, "model_state")
model = LaneletPredictor(config, μ, σ)
Flux.loadmodel!(model, loaded_model_state)
