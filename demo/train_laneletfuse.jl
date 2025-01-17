using VectorLanelet
using Lanelet2
using Lanelet2.Routing
using Lanelet2.Core
using AutomotiveSimulator
using Graphs
using MetaGraphs
using GraphNeuralNetworks
using Flux
using Statistics
using MLUtils

using Wandb, Logging, Dates
using JLD2

# TODO: Move to preprocess block
function agent_features_upsample(agt_features)
    agt_features = permutedims(agt_features, (2, 1, 3))
    agt_features = upsample_linear(agt_features, size=10)
    agt_features = permutedims(agt_features, (2, 1, 3))
    return agt_features
end

function run_training(wblogger::WandbLogger, config::Dict{String, Any}; model_name::String="", save_model::Bool=false, overfit::Bool=false)
    device = config["use_cuda"] ? gpu : cpu
    
    # Prepare data
    lanelet_roadway, g_meta = load_map_data()
    @info "Preparing agent features on $(device)"
    agt_features, agt_pos, labels = prepare_agent_features(lanelet_roadway) |> device
    agt_features_upsampled = agent_features_upsample(agt_features) |> device
    
    @info "Preparing map features on $(device)"
    polyline_graphs, g_heteromap, llt_pos, μ, σ = prepare_map_features(lanelet_roadway, g_meta) |> device

    # Initialize model
    @info "Initializing model and moving to $(device)"
    model = LaneletFusionPred(config, μ, σ) |> device
    
    # Training setup
    opt = Flux.setup(Adam(config["learning_rate"]), model)
    num_epochs = config["num_epochs"]
    
    function loss_fn(pred, y)
        mae = Flux.mae(pred, y)
        mse = Flux.mse(pred, y)
        return mae, mse
    end

    labels = labels
    train_data = (agt_features_upsampled, agt_pos, labels)
    batch_size = config["batch_size"]
    if overfit
        train_data = (agt_features_upsampled[:,:,1:1], agt_pos[:,1:1], labels[:,1:1])
        batch_size = 1
    end
    
    train_loader = Flux.DataLoader( 
        train_data,
        batchsize=batch_size,
        shuffle=true
    )

    # Initial logging
    # Flux.reset!(model)
    # pred = model(agt_features_upsampled, map2agent_graphs, polyline_graphs, g_heteromap, agent2map_graphs)
    # loss = loss_fn(pred, labels)
    # epoch = 0
    # logging_callback(wblogger, "train", epoch, cpu(loss)..., log_step_increment=0)

    # Training loop
    @info "Start training"
    for epoch in 1:num_epochs
        @show epoch
        Flux.reset!(model)
        
        # Training
        for (agt_feat, agt_pos, y) in train_loader  
            loss, grad = Flux.withgradient(model) do model
                pred = model(agt_feat, agt_pos, polyline_graphs, g_heteromap, llt_pos)
                loss_fn(pred, y)
            end
            
            Flux.update!(opt, model, grad[1])
            logging_callback(wblogger, "train", epoch, cpu(loss)..., log_step_increment=length(y))  # log_step_increment = batch size
        end
    end
    
    if save_model
        # Save the trained model
        @info "Saving the trained model state"
        model_name = "$(model_name)_$(now()).jld2"
        model_path = joinpath(@__DIR__, "../models/$(model_name)")
        save_model_state(cpu(model), model_path)
    else
        @info "Plotting predictions"
        pred = model(agt_features_upsampled, agt_pos, polyline_graphs, g_heteromap, llt_pos)
        VectorLanelet.plot_predictions(cpu(agt_features), cpu(labels), cpu(pred))
    end
end

include("../src/config.jl")

wblogger = WandbLogger(
    project = "VectorLanelet",
    name = "demo-$(now())",
    config = config
)
try
    run_training(wblogger, config; model_name="LaneletFusionPred", save_model=false)
finally
    close(wblogger)
end

