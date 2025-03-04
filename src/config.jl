"""
Configuration for the model
"""
config = Dict{String, Any}()

### Data processing
config["cluster_thrd"] = 50.0f0

### Training
config["learning_rate"] = 4e-5
config["num_epochs"] = 100
config["batch_size"] = 12
config["use_cuda"] = true

config["overfit"] = false
config["overfit_idx"] = 2
config["save_model"] = false
config["predict_current_pos"] = false
config["model_name"] = "LaneletFusionPred"

### Logger
config["wandb_project"] = "VectorLanelet"
# config["wandb_name"] = "$(config["model_name"])-$(now())"

#### Model
# μ and σ for map location0
config["μ"] = Float32[-65.87789, 83.178276]
config["σ"] = Float32[108.708725, 130.57153]

config["ple_in_channels"] = 4
config["ple_hidden_unit"] = 64
config["ple_out_channels"] = 64
config["ple_num_layers"] = 1
config["ple_norm"] = "LN"

config["mapenc_hidden_unit"] = 64
config["mapenc_num_layers"] = 1
# TODO: check if this helps
config["mapenc_self_loop"] = true

config["actornet_in_channels"] = 2
config["group_out_channels"] = [16, 64] # TODO: Change to 3 groups
config["actornet_num_layers"] = 1
config["actornet_norm"] = "GN"
config["actornet_ng"] = 32
config["actornet_kernel_size"] = 3

config["pred_head_hidden_unit"] = 64
config["pred_head_num_layers"] = 1

config["transformer_hidden_unit"] = 64
config["transformer_num_head"] = 2
config["transformer_num_layer"] = 3

# Fusion setup
config["fusion_n_in"] = 64        # Number of features for each node
config["fusion_e_in"] = 2
config["fusion_out_dim"] = 64
config["fusion_num_heads"] = 2
config["fusion_num_layers"] = 2
config["fusion_norm"] = "GN"
config["fusion_ng"] = 1
config["agent2map_dist_thrd"] = 3.0      # in the paper: 7.0
config["map2agent_dist_thrd"] = 2.0      # in the paper: 6.0
config["agent2agent_dist_thrd"] = 3.0   # in the paper: 100.0
