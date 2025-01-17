"""
Configuration for the model
"""
config = Dict{String, Any}()

# Training
config["learning_rate"] = 1e-3
config["num_epochs"] = 20
config["batch_size"] = 32
config["use_cuda"] = true

config["overfit"] = true
config["save_model"] = false
config["model_name"] = "lanelet_predictor"

# Logger
config["wandb_project"] = "VectorLanelet"
# config["wandb_name"] = "$(config["model_name"])-$(now())"

# Model

# μ and σ for map location0
config["μ"] = Float32[-65.87789, 83.178276]
config["σ"] = Float32[108.708725, 130.57153]

config["ple_in_channels"] = 4
config["ple_hidden_unit"] = 64
config["ple_out_channels"] = 64
config["ple_num_layers"] = 1

config["mapenc_hidden_unit"] = 64
config["mapenc_num_layers"] = 1

config["actornet_in_channels"] = 2
config["group_out_channels"] = [16, 64]
config["actornet_num_layers"] = 1

config["pred_head_hidden_unit"] = 64
config["pred_head_num_layers"] = 1

config["transformer_hidden_unit"] = 64
config["transformer_num_head"] = 2
config["transformer_num_layer"] = 3

# Fusion setup
config["fusion_n_in"] = 64        # Number of features for each node
config["fusion_e_in"] = 1
config["fusion_out_dim"] = 64
config["fusion_num_heads"] = 2
config["fusion_num_layers"] = 2
config["agent_dist_thrd"] = 100.0
config["llt_dist_thrd"] = 10.0
