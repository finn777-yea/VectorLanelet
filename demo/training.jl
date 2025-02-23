using VectorLanelet
using Flux
using Wandb, Logging, Dates
using JLD2

# TODO: preprocess_data for different models
# X, Y = preprocess_data(factory, loaded_data.gb) |> device
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

    # Prepare data
    lanelet_roadway, g_meta = load_map_data()
    @info "Preparing agent features on $(device)"
    agt_features, agt_pos_end = prepare_agent_features(lanelet_roadway) |> device
    agt_features_upsampled = agent_features_upsample(agt_features) |> device

    @info "Preparing map features on $(device)"
    polyline_graphs, g_heteromap, llt_pos, μ, σ = prepare_map_features(lanelet_roadway, g_meta) |> device

    labels = config["predict_current_pos"] ? agt_features[:, 2, :] : agt_pos_end

    agent_data = (; agt_features_upsampled)
    map_data = (; polyline_graphs, g_heteromap, llt_pos)
    data = (; agent_data, map_data, labels)

    # Initialize model
    @info "Initializing model and moving to $(device)"
    model = setup_model(config, μ, σ) |> device

    # Training setup
    opt = Flux.setup(Adam(config["learning_rate"]), model)
    num_epochs = config["num_epochs"]

    function loss_fn(pred, y)
        mae = Flux.mae(pred, y)
        mse = Flux.mse(pred, y)
        return mae, mse
    end

    # DataLoader
    train_data = VectorLanelet.preprocess_data(data, config["overfit"])
    batch_size = config["overfit"] ? 1 : config["batch_size"]

    train_loader = Flux.DataLoader(
        train_data,
        batchsize=batch_size,
        shuffle=true
    )

    # Initial logging
    @info "Initial logging"
    x, y = train_data
    pred = model(x, map_data...)
    loss = loss_fn(pred, y)
    epoch = 0
    logging_callback(wblogger, "train", epoch, cpu(loss)..., log_step_increment=0)

    # Training loop
    @info "Start training"
    for epoch in 1:num_epochs
        @show epoch

        # Training
        for (x, y) in train_loader
            loss, grad = Flux.withgradient(model) do model
                pred = model(x, map_data...)
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
        pred = model(agent_data..., map_data...)
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
    run_training(wblogger, config)
finally
    close(wblogger)
end
