"""
Configuration for the model
"""
config = Dict{String, Any}()

config["transformer_num_layer"] = 3
config["transformer_num_head"] = 4

config["learning_rate"] = 1e-4
config["num_epochs"] = 50
config["batch_size"] = 16
config["use_cuda"] = true
