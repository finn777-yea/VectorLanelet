"""
Configuration for the model
"""
config = Dict{String, Any}()

### Data processing
config["recording_ids"] = [1,2,3,4,5]
config["recording_path"] = joinpath(homedir(), "Documents", "highD-dataset-v1.0")
config["recording_agent_data_path"] = joinpath(@__DIR__, "..", "res", "highd_prediction_dataset_size500.jld2")

config["cluster_thrd"] = 30.0f0

### Plotting
config["plot_grid"] = true
config["plot_scenario_indices"] = [1,3,5,7]

### Training
config["learning_rate"] = 2e-4
config["num_epochs"] = 5
config["batch_size"] = 32
config["train_fraction"] = 0.8
config["use_cuda"] = true

config["overfit"] = false
config["overfit_idx"] = 1
config["save_model"] = false
config["predict_current_pos"] = false
config["model_name"] = "LaneletStaticFusionPred"
# config["model_name"] = "LaneletFusionPred"

### Logger
config["wandb_project"] = "VectorLanelet"
# config["wandb_name"] = "$(config["model_name"])-$(now())"

#### Model
# μ and σ for map location0
config["μ"] = Float32[-65.87789, 83.178276]
config["σ"] = Float32[108.708725, 130.57153]

config["ple_in_channels"] = 4
config["ple_hidden_channels"] = 64
config["ple_num_layers"] = 2
config["ple_norm"] = "LN"

config["mapenc_hidden_channels"] = 64       # should be the same as ple hidden unit
config["mapenc_num_layers"] = 2
# TODO: check if this helps
config["mapenc_self_loop"] = true

config["actornet_in_channels"] = 7
config["actornet_hidden_channels"] = 64
config["group_out_channels"] = [16, 32, 64]
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
config["fusion_norm"] = "LN"
config["fusion_ng"] = 1
config["agent2map_dist_thrd"] = 10.0      # in the paper: 7.0
config["map2agent_dist_thrd"] = 10.0      # in the paper: 6.0
config["agent2agent_dist_thrd"] = 100.0   # in the paper: 100.0
