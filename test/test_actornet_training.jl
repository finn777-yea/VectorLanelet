using VectorLanelet
using Flux

lanelet_roadway, g_meta = VectorLanelet.load_map_data()
polyline_graphs, g_heteromap, μ, σ = VectorLanelet.prepare_map_features(lanelet_roadway, g_meta)
agent_features, pos_agt, labels = VectorLanelet.prepare_agent_features(lanelet_roadway)

# Upsample the features along timestep axis
agent_features_upsampled = permutedims(agent_features, (2, 1, 3)) |>
x -> upsample_linear(x, size=10) |>
x -> permutedims(x, (2, 1, 3))

# Check the normalization result
agt_preprocess = create_agt_preprocess_block(μ, σ)
processed_agent_features = agt_preprocess(agent_features)

res_block_predictor = Chain(
    x -> permutedims(x, (2, 1, 3)),     # timestep, agent, feature
    create_residual_block(2, 32, kernel_size=3, stride=1, norm="BN", ng=16),
    x -> x[end, :, :],
    create_prediction_head(32, μ, σ)
)   |> gpu

group_predictor = Chain(
    x -> permutedims(x, (2, 1, 3)),
    create_group_block(1, 2, 32, kernel_size=3, norm="BN"),         # Both ResBlock with stride=1
    x -> x[end, :, :],
    create_prediction_head(32, μ, σ)
)   |> gpu

actornet_predictor = Chain(
    ActorNet_Simp(2, [16, 64], μ, σ),
    create_prediction_head(64, μ, σ),
)   |> gpu


# Training setup
function training(predictor, agent_features, labels=labels; overfit=false, valid_plot=true, save_valid_plot=false)
    opt = Flux.setup(Adam(1e-4), predictor)
    num_epochs = 500
    batch_size = 16

    # Split training data
    if overfit
        train_data = (agent_features[:,:,1:3], labels[:,1:3]) |> gpu
        batch_size = 1
    else
        train_data = (agent_features, labels) |> gpu
    end

    train_loader = Flux.DataLoader(
        train_data,
        batchsize=batch_size,
        shuffle=true
    )

    # Loss function
    loss_fn(m, x, y) = Flux.mse(m(x), y)

    # Training loop
    @info "Start training"
    for epoch in 1:num_epochs
        Flux.reset!(predictor)
        
        # Training
        loss = 0.0
        for (x, y) in train_loader
            loss, grad = Flux.withgradient(predictor) do m
                loss_fn(m, x, y)
            end
            
            Flux.update!(opt, predictor, grad[1])
        end
        if epoch % 100 == 0
            @info "Epoch: $epoch, Loss: $(cpu(loss))"
        end
    end
    
    if valid_plot
        VectorLanelet.plot_predictions(cpu(predictor), cpu(agent_features), cpu(labels), save=save_valid_plot)
    end
end

training(actornet_predictor, agent_features_upsampled, labels, overfit=false, valid_plot=true)


