using Logging
using Wandb
using JLD2
using NestedGraphsIO
using GraphIO.GraphML
using MetaGraphs

function logging_callback(logger, type, epoch, mae, mse; log_step_increment)
    with_logger(logger) do
        @info "$type/info" epoch = epoch log_step_increment = 0
        @info "$type/loss" mae = mae mse = mse log_step_increment = log_step_increment
    end
end

function save_model_state(model, path)
    model_state = Flux.state(model)
    jldsave(path; model_state)
end