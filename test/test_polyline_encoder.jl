using VectorLanelet
using GraphNeuralNetworks

using Random
Random.seed!(1234)

# Test the PolylineEncoder
using Flux
using Graphs
using GraphNeuralNetworks
using Statistics

# Create a simple synthetic dataset
function create_synthetic_data(num_samples=100)
    samples = []
    labels = []
    
    for _ in 1:num_samples
        # Create a simple path with 3-5 vectors
        num_vectors = rand(3:5)
        
        # Generate sequential points to form a path
        # Each vector has 2 features: [x, y]
        vector_features = zeros(2, num_vectors)
        for i in 1:num_vectors
            vector_features[1, i] = i + rand() * 0.2  # x coordinate with some noise
            vector_features[2, i] = i^2 + rand() * 0.2  # y coordinate following y=x^2 with noise
        end
        
        # Create a fully connected graph for these vectors
        g = complete_digraph(num_vectors) |> GNNGraph
        
        # Create a simple label (e.g., average position)
        label = mean(vector_features, dims=2)
        
        push!(samples, (g, vector_features))
        push!(labels, label)
    end
    
    return samples, labels
end

# Initialize the model
function init_model()
    # Calculate mean and std from your training data
    μ = [0.0, 0.0]  # placeholder values
    σ = [1.0, 1.0]  # placeholder values
    
    model = PolylineEncoder(2, 2, μ, σ, 2, 32)  # in_channels=2, out_channels=2
    return model
end

# Training loop
function train_model()
    # Create dataset
    train_data, train_labels = create_synthetic_data(100)
    
    # Initialize model
    model = init_model()
    
    # Setup optimizer
    opt = Flux.setup(Adam(0.01), model)
    
    # Define loss function
    loss_fn(g, vector_features, y) = Flux.mse(model(g, vector_features), y)
    
    # Split training data
    train_loader = Flux.DataLoader(
        (train_data, train_labels),
        batchsize=10,
        shuffle=true
    )
    
    # Training loop
    epochs = 50
    for epoch in 1:epochs
        total_loss = 0.0
        
        # Training
        for (batch_data, batch_labels) in train_loader
            for (sample, label) in zip(batch_data, batch_labels)
                g, vector_features = sample
                
                # Calculate gradients and update parameters
                loss, grad = Flux.withgradient(model) do m
                    loss_fn(g, vector_features, label)
                end
                
                Flux.update!(opt, model, grad[1])
                
                # Accumulate loss
                total_loss += loss
            end
        end
        
        # Print progress
        if epoch % 10 == 0
            println("Epoch $epoch: Average loss = $(total_loss / length(train_data))")
        end
    end
    
    return model
end

model = train_model()