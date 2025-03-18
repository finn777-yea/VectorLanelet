using VectorLanelet
using Flux
using MLUtils: splitobs
using Wandb, Logging, Dates
using JLD2

# Create a new named tuple with the same fields but with batched g_heteromaps
# for batching after dataloader
function batch_heteromaps(train_data_x)
    return (
        agt_features_upsampled = train_data_x.agt_features_upsampled,
        # agt_current_pos = train_data_x.agt_current_pos,
        polyline_graphs = train_data_x.polyline_graphs,
        g_heteromaps = Flux.batch(train_data_x.g_heteromaps),
        # llt_pos = train_data_x.llt_pos,
    )
end

function cpu_view(x::SubArray)
    # Get the parent array and indices
    parent_array = cpu(parent(x))
    # Recreate the view with the same indices but on the CPU array
    return view(parent_array, parentindices(x)...)
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
    X, Y = VectorLanelet.preprocess_data(data, overfit=config["overfit"], overfit_idx=config["overfit_idx"])
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

            # Create interaction graphs
            ga2m_all, gm2a_all, ga2a_all = create_interaction_graphs(x.agt_current_pos, x.llt_pos,
                config["agent2map_dist_thrd"], config["map2agent_dist_thrd"], config["agent2agent_dist_thrd"])
            pred = model(batch_heteromaps(x)..., ga2m_all, gm2a_all, ga2a_all)
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
            # Create interaction graphs after sampling
            ga2m, gm2a, ga2a = create_interaction_graphs(x.agt_current_pos, x.llt_pos,
                config["agent2map_dist_thrd"], config["map2agent_dist_thrd"], config["agent2agent_dist_thrd"])
            x = batch_heteromaps(x)
            loss, grad = Flux.withgradient(model) do model
                pred = model(x..., ga2m, gm2a, ga2a)
                loss_fn(pred, y)
            end

            Flux.update!(opt, model, grad[1])
            logging_callback(wblogger, "train", epoch, cpu(loss)..., log_step_increment=length(y))  # log_step_increment = batch size
        end

        # Validation
        let (x, y) = val_data
            inter_graphs = create_interaction_graphs(x.agt_current_pos, x.llt_pos,
                config["agent2map_dist_thrd"], config["map2agent_dist_thrd"], config["agent2agent_dist_thrd"])
            values = loss_fn(model(batch_heteromaps(x)..., inter_graphs...), y)

            # logging
            logging_callback(wblogger, "val", epoch, cpu(values)..., log_step_increment=0)
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
        # let (x, y) = train_data
        #     ga2m_all, gm2a_all, ga2a_all = create_interaction_graphs(x.agt_current_pos, x.llt_pos,
        #         config["agent2map_dist_thrd"], config["map2agent_dist_thrd"], config["agent2agent_dist_thrd"])
        #     pred = model(batch_heteromaps(x)..., ga2m_all, gm2a_all, ga2a_all)
        #     # Use cpu_view to move views of CuArray to cpu
        #     VectorLanelet.plot_predictions(cpu_view(x.agt_features), cpu_view(y), cpu(pred),
        #         grid_layout=config["plot_grid"], scenario_indices=config["plot_scenario_indices"])
        # end
        let (x, y) = val_data
            inter_graphs = create_interaction_graphs(x.agt_current_pos, x.llt_pos,
                config["agent2map_dist_thrd"], config["map2agent_dist_thrd"], config["agent2agent_dist_thrd"])
            pred = model(batch_heteromaps(x)..., inter_graphs...)
            VectorLanelet.plot_predictions(cpu_view(x.agt_features), cpu_view(y), cpu(pred),
                grid_layout=config["plot_grid"], scenario_indices=nothing)
        end
    end
end

include("../src/config.jl")

wblogger = WandbLogger(
    project = "VectorLanelet",
    name = "demo-b$(config["batch_size"])-lr$(config["learning_rate"])-epochs$(config["num_epochs"])",
    config = config
)
try
    run_training(wblogger, config)
finally
    close(wblogger)
end
