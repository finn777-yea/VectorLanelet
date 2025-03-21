using DroneDatasetAdapter
using JLD2
using Test

dataset_path = joinpath(homedir(), "Documents", "exiD-dataset-v2.1")
recording_ids = [0,1,2,3,4,5]
data_infos = ExiDRecordings(recording_ids, dataset_path)
dataset = read(data_infos)
data_path = joinpath(@__DIR__,"..", "res", "prediction_dataset_size200.jld2")

# The data is processed in file demo/load_recordings.jl
data = load(data_path)
X = data["X"]
Y = data["Y"]

@testset "Test loaded drone dataset" begin
    for i in 1:10
        df = filter(
        row -> row.recordingId == metadata[i].current_recording &&
        row.frame == metadata[i].current_frame,
        dataset.tracks
        )
        @test unique(df.recordingId)[1] == metadata[i].current_recording
        @test unique(df.frame)[1] == metadata[i].current_frame
        @test df.trackId == metadata[i].track_ids_in_frame
    
        @test X[i][1,:] == df.xCenter
        @test X[i][2,:] == df.yCenter
    end
end