using LaneletPredictor
using Flux
using Plots

model = LaneletPredictor(config)
state_name = "model_state"
model_state_path = joinpath(@__DIR__, "../res/$(state_name).jld2")    
model_state = JLD2.load(model_state_path, "model_state")
Flux.loadmodel!(model, model_state)

# Plot the prediction of the model