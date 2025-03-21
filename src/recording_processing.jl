"""
    create_prediction_dataset(dataset::Dataset, prediction_horizon=5)

Create a prediction dataset with the following structure:
- Each entry represents all vehicles in a specific frame
- X contains current positions/states of all vehicles in the frame
- Y contains future positions of all vehicles after prediction_horizon frames
- No fixed vehicle count - uses actual number of vehicles in each frame

Returns:
- frame_features: List where each element contains all vehicles' data for one frame
- frame_targets: List where each element contains future positions for one frame
- frame_metadata: List of (recordingId, frame) to track which frame each sample represents
"""
function create_prediction_dataset(dataset::Dataset, frame_limit=200,prediction_horizon=5)
    frame_features = []  # Each element contains all vehicles in one frame
    frame_targets = []   # Each element contains future positions for all the corresponding vehicles
    frame_metadata = []  # Tracking which frame each entry represents

    # Group by frames
    frame_groups = groupby(dataset.tracks, [:recordingId, :frame])

    for (key, frame_df) in Iterators.take(pairs(frame_groups), frame_limit)
        current_recording = key.recordingId
        current_frame = key.frame

        # Find the future frame data
        future_frame = current_frame + prediction_horizon
        future_frame_data = filter(row ->
            row.recordingId == current_recording &&
            row.frame == future_frame,
            dataset.tracks
        )

        nrow(future_frame_data) == 0 && continue

        # Store all the vehs in the current frame
        vehicles_features = []
        vehicles_targets = []
        track_ids_in_frame = []

        # Process each vehicle in the current frame
        for vehicle_idx in 1:nrow(frame_df)
            vehicle = frame_df[vehicle_idx, :]

            # Skip the vehicles that don't have future positions
            future_vehicle = filter(row -> row.trackId == vehicle.trackId, future_frame_data)
            nrow(future_vehicle) == 0 && continue

            # Extract vehicle features
            vehicle_feature = [
                vehicle.xCenter,        # x position
                vehicle.yCenter,        # y position
                # vehicle.trackId,        # Vehicle ID
                vehicle.xVelocity,      # x velocity
                vehicle.yVelocity,      # y velocity
                vehicle.heading,        # heading
                vehicle.width,          # width
                vehicle.length          # length
            ]

            vehicle_target = [
                # vehicle.trackId,                # Vehicle ID for tracking
                future_vehicle[1, :xCenter],    # Future x position
                future_vehicle[1, :yCenter]     # Future y position
            ]

            push!(vehicles_features, vehicle_feature)
            push!(vehicles_targets, vehicle_target)
            push!(track_ids_in_frame, vehicle.trackId)
        end

        vehicles_features = hcat(vehicles_features...)
        vehicles_targets = hcat(vehicles_targets...)
        @assert size(vehicles_features, 2) == size(vehicles_targets, 2)
        push!(frame_features, vehicles_features)
        push!(frame_targets, vehicles_targets)
        push!(frame_metadata, (;current_recording, current_frame, track_ids_in_frame))
    end

    return frame_features, frame_targets, frame_metadata
end