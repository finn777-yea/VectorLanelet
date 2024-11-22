using Logging
using Wandb

function logging_callback(logger, type, epoch, mse; log_step_increment)
    with_logger(logger) do
        @info "$type/info" epoch = epoch log_step_increment = log_step_increment
        @info "$type/loss" mse = mse log_step_increment = 0
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

# TODO: add mae
# TODO: add accuracy
function loss_fn(model, x, y, g_all, g_hetero, μ, σ)
    pred = model(x, g_all, g_hetero, μ, σ)
    return Flux.mse(pred, y)
end