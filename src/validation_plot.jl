function grid_layout_plot(agent_features, labels, pred, agent_counts, agent_offsets; save::Bool=false, scenario_indices=nothing)
    # Create a grid of plots
    n_scenarios = length(scenario_indices)
    n_cols = min(2, n_scenarios)  # Maximum 2 columns
    n_rows = ceil(Int, n_scenarios / n_cols)
    plots = []
    
    #  i is the plot index
    for (i, idx) in enumerate(scenario_indices)
        # Extract data for this scenario
        features = agent_features[idx]
        label = labels[idx]

        start_idx = agent_offsets[idx] + 1
        end_idx = start_idx + agent_counts[idx] - 1
        scenario_pred = pred[:, start_idx:end_idx]
        
        x_t1 = features[1, 1, :]  # x-coordinates at timestep 1
        y_t1 = features[2, 1, :]  # y-coordinates at timestep 1
        x_t2 = features[1, 2, :]  # x-coordinates at timestep 2
        y_t2 = features[2, 2, :]  # y-coordinates at timestep 2
        
        x_end = label[1, :]
        y_end = label[2, :]
        
        x_pred = scenario_pred[1, :]
        y_pred = scenario_pred[2, :]
        
        # Create individual plot
        p = plot(title="Scenario $idx",
            xlabel="X", ylabel="Y", legend=(i==1 ? :topright : false),
            grid=:true, size=(250, 200))
            
        # Plot points
        scatter!(p, x_t1, y_t1, label=(i==1 ? "Timestep 1" : ""), markershape=:circle, color=:blue, alpha=0.8, markersize=4)
        scatter!(p, x_t2, y_t2, label=(i==1 ? "Timestep 2" : ""), markershape=:diamond, color=:red, alpha=0.8, markersize=4)
        scatter!(p, x_end, y_end, label=(i==1 ? "End" : ""), markershape=:star, color=:green, alpha=0.8, markersize=4)
        scatter!(p, x_pred, y_pred, label=(i==1 ? "Prediction" : ""), markershape=:circle, color=:yellow, alpha=0.8, markersize=4)
        
        # Draw lines between end and prediction
        for j in eachindex(x_end)
            plot!(p, [x_end[j], x_pred[j]], [y_end[j], y_pred[j]], color=:gray, linewidth=1.0, alpha=0.6, label="")
        end
        
        push!(plots, p)
    end
    
    # Combine plots in a grid
    final_plot = plot(plots..., layout=(n_rows, n_cols), size=(350*n_cols, 320*n_rows))
    
    if save
        savefig(final_plot, "predictions_grid.png")
    else
        display(final_plot)
    end
    
    return final_plot
end

function single_plot(agent_features, labels, pred, agent_counts, agent_offsets; save::Bool=false, scenario_indices=nothing)
    # Single plot with selected scenarios
    p = plot(title="Agent Positions Across Scenarios",
        xlabel="X", ylabel="Y", legend=:topright, grid=:true, size=(800, 600))
        
    for (i, idx) in enumerate(scenario_indices)
        features = agent_features[idx]
        label = labels[idx]

        start_idx = agent_offsets[idx] + 1
        end_idx = start_idx + agent_counts[idx] - 1
        scenario_pred = pred[:, start_idx:end_idx]
        
        x_t1 = features[1, 1, :]  # x-coordinates at timestep 1
        y_t1 = features[2, 1, :]  # y-coordinates at timestep 1
        x_t2 = features[1, 2, :]  # x-coordinates at timestep 2
        y_t2 = features[2, 2, :]  # y-coordinates at timestep 2
        
        x_end = label[1, :]
        y_end = label[2, :]
        
        x_pred = scenario_pred[1, :]
        y_pred = scenario_pred[2, :]
        
        # Use different colors for each scenario
        scenario_color = i  # Plots.jl will cycle through colors
        
        # Plot with scenario-specific labels for the first agent in each scenario
        scatter!(p, x_t1, y_t1, 
                label=(i==1 ? "Timestep 1" : ""), 
                markershape=:circle, 
                color=scenario_color, 
                alpha=0.8, 
                markersize=4)
                
        scatter!(p, x_t2, y_t2, 
                label=(i==1 ? "Timestep 2" : ""), 
                markershape=:diamond, 
                color=scenario_color, 
                alpha=0.8, 
                markersize=4)
                
        scatter!(p, x_end, y_end, 
                label=(i==1 ? "End" : ""), 
                markershape=:star, 
                color=scenario_color, 
                alpha=0.8, 
                markersize=4)
                
        scatter!(p, x_pred, y_pred, 
                label=(i==1 ? "Prediction" : ""), 
                markershape=:circle, 
                color=scenario_color, 
                alpha=0.8, 
                markersize=4)
        
        # Draw lines between end and prediction
        for j in eachindex(x_end)
            plot!(p, [x_end[j], x_pred[j]], [y_end[j], y_pred[j]],
                    color=scenario_color, linewidth=1.0, alpha=0.6, label="")
        end
    end
    
    if save
        savefig(p, "predictions_combined.png")
    else
        display(p)
    end
    
    return p
end

function plot_predictions(agent_features, labels, pred; save::Bool=false, scenario_indices=nothing, grid_layout=true)
    # If scenario_indices is not provided, use all available scenarios
    if isnothing(scenario_indices)
        scenario_indices = 1:length(agent_features)
    end

    # Calculate the total number of agents across all scenarios to help with indexing
    agent_counts = [size(agent_features[i], 3) for i in eachindex(agent_features)]
    agent_offsets = [0; cumsum(agent_counts)[1:end-1]]
    
    if grid_layout
        grid_layout_plot(agent_features, labels, pred, agent_counts, agent_offsets; save, scenario_indices)
    else
        single_plot(agent_features, labels, pred, agent_counts, agent_offsets; save, scenario_indices)
    end
end