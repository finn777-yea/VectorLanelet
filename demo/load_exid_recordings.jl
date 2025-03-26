using DroneDatasetAdapter
using VectorLanelet: create_prediction_dataset
using JLD2

dataset_path = joinpath(homedir(), "Documents", "exiD-dataset-v2.1")
recording_ids = [0,1,2,3,4,5]

data_infos = ExiDRecordings(recording_ids, dataset_path)
dataset = read(data_infos)

### Included vehicle features
# vehicle.xCenter,        # x position
# vehicle.yCenter,        # y position
# vehicle.xVelocity,      # x velocity
# vehicle.yVelocity,      # y velocity
# vehicle.heading,        # heading
# vehicle.width,          # width
# vehicle.length          # length
X,Y,metadata = create_prediction_dataset(dataset, 200)

output_path = joinpath(@__DIR__,"..", "res", "prediction_dataset_size$(length(X)).jld2")
jldsave(output_path; X, Y)