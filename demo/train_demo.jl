using VectorLanelet
using Lanelet2
using Lanelet2.Routing
using Lanelet2.Core
using AutomotiveSimulator
using Graphs
using MetaGraphs

using GraphNeuralNetworks
using GraphIO.GraphML
using Transformers
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
    lanelet_roadway, g_meta = load_map_data()
    # Prepare data
    @info "Preparing agent features on $(device)"
    agt_features, pos_agt, labels = prepare_agent_features(lanelet_roadway)
    agt_features_upsampled = agent_features_upsample(agt_features) |> device
    pos_agt = pos_agt |> device
    labels = labels |> device
    
    @info "Preparing map features on $(device)"
    polyline_graphs, g_heteromap, μ, σ = prepare_map_features(lanelet_roadway, g_meta) |> device
    
    # Initialize model
    @info "Initializing model and moving to $(device)"
    @show μ, σ
    model = LaneletPredictor(config, μ, σ) |> device
    
    # Training setup
    opt = Flux.setup(Adam(config["learning_rate"]), model)
    num_epochs = config["num_epochs"]
    
    function loss_fn(model, x, y, g_polyline, g_heteromap)
        pred = model(x, g_polyline, g_heteromap)
        mse = Flux.mse(pred, y)
        mae = Flux.mae(pred, y)
        return mae, mse
    end

    # Split training data
    labels = labels
    train_data, val_data = splitobs((agt_features_upsampled, labels), at=0.99)
    batch_size = config["batch_size"]
    if overfit
        train_data = (agt_features_upsampled[:,:,1:1], labels[:,1:1])
        batch_size = 1
    end
    
    train_loader = Flux.DataLoader( 
        train_data,
        batchsize=batch_size,
        shuffle=true
    )

    # Initial logging
    for (type, data) in ("train" => train_data, "val" => val_data)
        let (x, y) = data
            Flux.reset!(model)
            loss = loss_fn(model, x, y, polyline_graphs, g_heteromap)
            epoch = 0
            logging_callback(wblogger, type, epoch, cpu(loss)..., log_step_increment=0)
        end
    end

    # Training loop
    @info "Start training"
    for epoch in 1:num_epochs
        Flux.reset!(model)
        
        # Training
        for (x, y) in train_loader
            loss, grad = Flux.withgradient(model) do m
                loss_fn(m, x, y, polyline_graphs, g_heteromap)
            end
            
            Flux.update!(opt, model, grad[1])
            logging_callback(wblogger, "train", epoch, cpu(loss)..., log_step_increment=length(y))  # log_step_increment = batch size
        end
    

        # # Validation
        # let (x, y) = val_dataxw
        #     Flux.reset!(model)
        #     values = loss_fn(model, x, y, polyline_graphs, g_heteromap)

        #     # logging
        #     logging_callback(wblogger, "val", epoch, cpu(values)..., log_step_increment=length(y))
        # end

    end
    
    if save_model
        # Save the trained model
        @info "Saving the trained model state"
        model_name = "$(model_name)_$(now()).jld2"
        model_path = joinpath(@__DIR__, "../models/$(model_name)")
        save_model_state(cpu(model), model_path)
    else
        pred = model(agt_features_upsampled, polyline_graphs, g_heteromap)
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
    run_training(wblogger, config; model_name="LaneletPredictor_with_Transformer", save_model=true)
finally
    close(wblogger)
end

