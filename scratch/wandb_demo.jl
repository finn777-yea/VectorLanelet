using Wandb, Logging, Dates

lg = WandbLogger(
    project = "VectorLanelet",
    name = "demo-$(now())",
    config = Dict("learning_rate" => 0.01, "dropout" => 0.5)
)

global_logger(lg)

with_logger(lg) do
    for x in 1:50
        acc = log(1 + x + rand() * get_config(lg, "learning_rate") + rand())
        loss = 10 - log(1 + x + rand() + x * get_config(lg, "learning_rate") + rand())
        # Log metrics from your script to W&B
        @info "metrics" accuracy=acc loss=loss
        @info "epoch" epoch=x log_step_increment=2
    end
end
close(lg)