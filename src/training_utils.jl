using Logging
using Wandb
using JLD2

function logging_callback(logger, type, epoch, mse, mae; log_step_increment)
    with_logger(logger) do
        @info "$type/info" epoch = epoch log_step_increment = 0
        @info "$type/loss" mse = mse mae = mae log_step_increment = log_step_increment
    end
end

"""
Loss function for the model

# Arguments
- `model`: the model to evaluate
- `x`: agent features
- `y`: labels
- `g_all`: all graph
- `g_hetero`: heterogeneous graph
- `μ`: mean of the agt feature distribution
- `σ`: std of the agt feature distribution
"""

function loss_fn(model, x, y, g_polyline, g_heteromap)
    pred = model(x, g_polyline, g_heteromap)
    mse = Flux.mse(pred, y)
    mae = Flux.mae(pred, y)
    return mse, mae
end

function save_model_state(model, path)
    model_state = Flux.state(model)
    jldsave(path; model_state)
end