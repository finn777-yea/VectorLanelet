using DroneDatasetAdapter
using VectorLanelet: create_prediction_dataset
using JLD2
using Lanelet2
using Lanelet2.Routing
using NestedGraphsIO, GraphIO.GraphML
using MetaGraphs
using GraphNeuralNetworks
using VectorLanelet

dataset_path = joinpath(homedir(), "Documents", "highD-dataset-v1.0")
recording_ids = [1,2,3,4,5]
data_infos = HighDRecordings(recording_ids, dataset_path)
dataset = read(Adapter([data_infos]))

### Included vehicle features
# vehicle.xCenter,        # x position
# vehicle.yCenter,        # y position
# vehicle.xVelocity,      # x velocity
# vehicle.yVelocity,      # y velocity
# vehicle.heading,        # heading
# vehicle.width,          # width
# vehicle.length          # length
X,Y,metadatas = get_agent_prediction_dataset(dataset, 500)

output_path = joinpath(@__DIR__,"..", "res", "highd_prediction_dataset_size$(length(X)).jld2")
jldsave(output_path; X, Y, metadatas)

X, Y, μ, σ = prepare_recording_data(dataset, recording_ids, config)
