using VectorLanelet
using Flux
using MLUtils: splitobs
using Wandb, Logging, Dates
using JLD2

# Used for validation plotting
function cpu_view(x::SubArray)
    # Get the parent array and indices
    parent_array = cpu(parent(x))
    # Recreate the view with the same indices but on the CPU array
    return view(parent_array, parentindices(x)...)
end

function setup_model(config::Dict{String, Any}, μ, σ)
    if config["model_name"] == "LaneletFusionPred"
        model = LaneletFusionPred(config, μ, σ)
    elseif config["model_name"] == "LaneletStaticFusionPred"
        model = LaneletStaticFusionPred(config, μ, σ)
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

    # Load the saved recording data
    @info "Loading recording data and moving to $(device)"
    X, Y, μ, σ = prepare_recording_data(config) |> device

    # Initialize model
    @info "Initializing model and moving to $(device)"
    # TODO: normalize agt and map module seperately
    model = setup_model(config, μ, σ) |> device

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
    train_data, val_data = splitobs((X, Y), at=config["train_fraction"])
    batch_size = config["overfit"] ? 1 : config["batch_size"]

    train_loader = Flux.DataLoader(
        train_data,
        batchsize=batch_size,
        shuffle=true
    )

    # Initial logging
    @info "Initial logging"
    for (type, data) in ("train" => train_data, "val" => val_data)
        let (x, y) = data
            x = VectorLanelet.collate_data(model, x, config)
            pred = model(x...)
            loss = loss_fn(pred, y)

            epoch = 0
            logging_callback(wblogger, type, epoch, cpu(loss)..., log_step_increment=0)
        end
    end

    # Training loop
    @info "Start training"
    for epoch in 1:num_epochs
        @show epoch
        for (x, y) in train_loader
            x = VectorLanelet.collate_data(model, x, config)
            loss, grad = Flux.withgradient(model) do model
                pred = model(x...)
                loss_fn(pred, y)
            end

            Flux.update!(opt, model, grad[1])
            logging_callback(wblogger, "train", epoch, cpu(loss)..., log_step_increment=length(y))  # log_step_increment = batch size
        end

        # Validation
        let (x, y) = val_data
            x = VectorLanelet.collate_data(model, x, config)
            values = loss_fn(model(x...), y)

            # logging
            logging_callback(wblogger, "val", epoch, cpu(values)..., log_step_increment=0)
        end

    end

    # if config["save_model"]
    #     # Save the trained model
    #     @info "Saving the trained model state"
    #     model_name = "$(config["model_name"])_$(now()).jld2"
    #     model_path = joinpath(@__DIR__, "../models/$(model_name)")
    #     save_model_state(cpu(model), model_path)
    # else
    #     @info "Plotting predictions"
    #     # let (x, y) = train_data
    #     #     ga2m_all, gm2a_all, ga2a_all = create_interaction_graphs(x.agt_current_pos, x.llt_pos,
    #     #         config["agent2map_dist_thrd"], config["map2agent_dist_thrd"], config["agent2agent_dist_thrd"])
    #     #     pred = model(batch_heteromaps(x)..., ga2m_all, gm2a_all, ga2a_all)
    #     #     # Use cpu_view to move views of CuArray to cpu
    #     #     VectorLanelet.plot_predictions(cpu_view(x.agt_features), cpu_view(y), cpu(pred),
    #     #         grid_layout=config["plot_grid"], scenario_indices=config["plot_scenario_indices"])
    #     # end
    #     let (x, y) = val_data
    #         x = VectorLanelet.collate_data(model, x, config)
    #         pred = model(x...)
    #         # TODO: x[1] is agent features, which leads to poor validation result
    #         VectorLanelet.plot_predictions(cpu_view(x[1]), cpu_view(y), cpu(pred),
    #             grid_layout=config["plot_grid"], scenario_indices=nothing)
    #     end
    # end
end

include("../src/config.jl")

wblogger = WandbLogger(
    project = "VectorLanelet",
    name = "$(config["model_name"])-b$(config["batch_size"])-lr$(config["learning_rate"])-epochs$(config["num_epochs"])",
    config = config
)
try
    run_training(wblogger, config)
finally
    close(wblogger)
end
