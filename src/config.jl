"""
Configuration for the model
"""
config = Dict{String, Any}()
config["vsg_in_channel"] = 4
config["actor_config"] = Dict("actor_in_channel" => 2, "actor_out_channel" => 128)
config["map_config"] = Dict("map_out_channel" => 128, "num_scales" => 3)

config["transformer_num_layer"] = 3
config["transformer_hidden_size"] = 128
config["transformer_num_head"] = 2

config["learning_rate"] = 1e-4
config["num_epochs"] = 100
config["batch_size"] = 8
