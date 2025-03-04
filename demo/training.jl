using VectorLanelet
using Flux
using Wandb, Logging, Dates
using JLD2

# Create a new named tuple with the same fields but with batched g_heteromaps
function batch_heteromaps(train_data_x)
    return (
        agt_features_upsampled = train_data_x.agt_features_upsampled,
        agt_current_pos = train_data_x.agt_current_pos,
        polyline_graphs = train_data_x.polyline_graphs,
        g_heteromaps = Flux.batch(train_data_x.g_heteromaps),
        llt_pos = train_data_x.llt_pos,
    )
end

function setup_model(config::Dict{String, Any}, μ, σ)
    if config["model_name"] == "LaneletFusionPred"
        model = LaneletFusionPred(config, μ, σ)
    elseif config["model_name"] == "LaneletPredictor"
        model = LaneletPredictor(config, μ, σ)
    else
        throw(ArgumentError("Model name $(config["model_name"]) not supported"))
    end
    @info "Model $(config["model_name"]) initialized"
    return model
end

function run_training(wblogger::WandbLogger, config::Dict{String, Any})
    device = config["use_cuda"] ? gpu : cpu
    @info "Using device: $(device) for training"

    # Preparing data
    data = VectorLanelet.prepare_data(config, device)

    # Initialize model
    @info "Initializing model and moving to $(device)"
    model = setup_model(config, data.map_data.μ, data.map_data.σ) |> device

    # Training setup
    opt = Flux.setup(Adam(config["learning_rate"]), model)
    num_epochs = config["num_epochs"]

    function loss_fn(pred, y)
        y = reduce(hcat, y)
        mae = Flux.mae(pred, y)
        mse = Flux.mse(pred, y)
        return mae, mse
    end

    # DataLoader
    train_data = VectorLanelet.preprocess_data(data, overfit=config["overfit"], overfit_idx=config["overfit_idx"])
    batch_size = config["overfit"] ? 1 : config["batch_size"]

    train_loader = Flux.DataLoader(
        train_data,
        batchsize=batch_size,
        shuffle=true
    )

    # Initial logging
    @info "Initial logging"
    x, y = train_data
    pred = model(batch_heteromaps(x)...)
    loss = loss_fn(pred, y)
    epoch = 0
    logging_callback(wblogger, "train", epoch, cpu(loss)..., log_step_increment=0)

    # Training loop
    @info "Start training"
    for epoch in 1:num_epochs
        @show epoch

        # Training
        for (x, y) in train_loader
            x_batch = batch_heteromaps(x)
            loss, grad = Flux.withgradient(model) do model
                pred = model(x_batch...)
                loss_fn(pred, y)
            end

            Flux.update!(opt, model, grad[1])
            logging_callback(wblogger, "train", epoch, cpu(loss)..., log_step_increment=length(y))  # log_step_increment = batch size
        end
    end

    if config["save_model"]
        # Save the trained model
        @info "Saving the trained model state"
        model_name = "$(config["model_name"])_$(now()).jld2"
        model_path = joinpath(@__DIR__, "../models/$(model_name)")
        save_model_state(cpu(model), model_path)
    else
        @info "Plotting predictions"
        # TODO: retrieve the actual pred of the corresponding scene
        plot_data, plot_label = VectorLanelet.preprocess_data(data, overfit=true, overfit_idx=config["overfit_idx"])
        pred = model(batch_heteromaps(plot_data)...)
        @show num_veh = length(plot_label[1])
        VectorLanelet.plot_predictions(cpu(plot_data.agt_features_upsampled[1]), cpu(plot_label[1]), cpu(pred))
    end
end

include("../src/config.jl")

wblogger = WandbLogger(
    project = "VectorLanelet",
    name = "demo-$(now())",
    config = config
)
try
    run_training(wblogger, config)
finally
    close(wblogger)
end
