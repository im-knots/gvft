import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns
import pandas as pd

def visualize_simulation(Fx_series, Fy_series, W_series, Module_coords_series, Graphs, 
                         X_np, Y_np, x_np, y_np, config, output_dir, params, bio_prior=False):
    """Visualize simulation results with multi-panel figures."""
    idx, current_lam_W, D_F = params
    timesteps_to_store = list(range(0, config.timesteps_sim + 1, config.view_step))
    if config.timesteps_sim not in timesteps_to_store:
        timesteps_to_store.append(config.timesteps_sim)
    
    n_show = len(timesteps_to_store)
    row_labels = ["Flow Mag $|F|$", "Strength $W(x)$", "Flow Field $F(x)$", "Graph & Fields"]
    fig, axs = plt.subplots(len(row_labels), n_show, figsize=(4 * n_show, 13), constrained_layout=True)
    
    # Handle case where only one timestep is stored (reshape axes)
    if n_show == 1:
        axs = axs.reshape(-1, 1)
    
    prior_text = "Bio Prior + " if bio_prior else ""
    print(f"Generating visualization for {prior_text}lam_W={current_lam_W:.3f}, D_F={D_F:.3f}...")
    
    base_grid_size = 200  # For scaling adjustments
    
    # Calculate metrics for each timestep
    from metrics_utils import compute_field_metrics
    metrics_series = []
    
    for i, t_idx in enumerate(range(len(timesteps_to_store))):
        t = timesteps_to_store[t_idx]
        
        # Get fields at this timestep
        Fx_t = Fx_series[t_idx]
        Fy_t = Fy_series[t_idx]
        W_t = W_series[t_idx]
        F_mag_t = np.sqrt(Fx_t**2 + Fy_t**2)
        module_coords_t = Module_coords_series[t_idx]
        
        # Calculate metrics at this timestep
        if i > 0:  # Skip the first timestep (initial) for metrics comparison
            metrics = compute_field_metrics(Fx_t, Fy_t, W_t, W_series[0])
            metrics_series.append(metrics)
        
        # Get graph at this timestep (except for t=0)
        if t_idx > 0 and len(Graphs) > t_idx - 1:
            weights = Graphs[t_idx-1]
        else:
            weights = np.zeros((config.num_modules, config.num_modules))
        
        # Row 0: Flow Magnitude
        im0 = axs[0, i].imshow(F_mag_t, extent=[-1, 1, -1, 1], origin='lower', cmap='Blues', aspect='auto', vmin=0, vmax=1.0)
        axs[0, i].set_title(f"t={t}")
        axs[0, i].set_xticks([]); axs[0, i].set_yticks([])
        
        # Row 1: Synaptic Strength W
        im1 = axs[1, i].imshow(W_t, extent=[-1, 1, -1, 1], origin='lower', cmap='hot', aspect='auto', vmin=-2, vmax=2)
        axs[1, i].set_xticks([]); axs[1, i].set_yticks([])
        
        # Row 2: Flow Field Streamplot
        if np.any(Fx_t) or np.any(Fy_t):
            # Adjust streamplot density based on grid size to avoid overcrowding
            streamplot_density = 1.2 * (base_grid_size / config.grid_size)
            axs[2, i].streamplot(x_np, y_np, Fx_t, Fy_t, color=F_mag_t, cmap='Blues', 
                              density=streamplot_density, linewidth=0.8)
        axs[2, i].set_xlim(-1, 1); axs[2, i].set_ylim(-1, 1)
        axs[2, i].set_aspect('equal', adjustable='box')
        axs[2, i].set_xticks([]); axs[2, i].set_yticks([])
        
        # Row 3: Graph + Fields Overlay
        axs[3, i].imshow(W_t, extent=[-1, 1, -1, 1], origin='lower', cmap='hot', alpha=0.4, aspect='auto', vmin=-2, vmax=2)
        if np.any(Fx_t) or np.any(Fy_t):
            # Lighter streamplot in the background
            streamplot_density = 1.5 * (base_grid_size / config.grid_size)
            axs[3, i].streamplot(x_np, y_np, Fx_t, Fy_t, color='lightgray', 
                              density=streamplot_density, linewidth=0.5)
        axs[3, i].scatter(module_coords_t[:, 0], module_coords_t[:, 1], c='blue', s=50, edgecolors='black', zorder=3, label='Modules')
        
        connection_count_vis = 0
        for m in range(config.num_modules):
            for n in range(config.num_modules):
                if weights[m, n] > 0.01:
                    x0, y0 = module_coords_t[m]
                    x1, y1 = module_coords_t[n]
                    arrow_alpha = np.clip(weights[m, n] * 1.5, 0.1, 1.0)
                    arrow_width = np.clip(weights[m, n] * 1.5, 0.5, 1.5)
                    dx_arrow = (x1 - x0) * 0.85
                    dy_arrow = (y1 - y0) * 0.85
                    if abs(dx_arrow) > 1e-6 or abs(dy_arrow) > 1e-6:
                        axs[3, i].arrow(x0, y0, dx_arrow, dy_arrow, head_width=0.06, head_length=0.08, 
                                       length_includes_head=True, alpha=arrow_alpha, color='green', 
                                       linewidth=arrow_width, zorder=2)
                        connection_count_vis += 1
        
        axs[3, i].set_title(f"Connectivity ({connection_count_vis}) & Fields")
        axs[3, i].set_xlim(-1, 1); axs[3, i].set_ylim(-1, 1)
        axs[3, i].set_aspect('equal', adjustable='box')
    
    # Add row labels
    for row_idx, label in enumerate(row_labels):
        axs[row_idx, 0].set_ylabel(label, fontsize=14, labelpad=20)
        axs[row_idx, 0].yaxis.set_label_coords(-0.15, 0.5)
    
    # Add grid size and scaling info to the title
    scale_info = f"(grid: {config.grid_size}x{config.grid_size}, dt: {config.dt:.3f})" if config.use_parameter_scaling else ""
    bio_tag = "bio_prior_" if bio_prior else ""
    
    # Save visualization
    filename = f"simulation_{bio_tag}lamW_{current_lam_W:.3f}_DF_{D_F:.3f}_etacoeff_{config.eta_coeff:.3f}_grid{config.grid_size}.png"
    filepath = os.path.join(output_dir, filename)
    plt.suptitle(f'{prior_text}Full Simulation: lam_W={current_lam_W:.3f}, D_F={D_F:.3f}, eta_coeff={config.eta_coeff:.3f} {scale_info}', 
                fontsize=16, y=0.99)
    plt.savefig(filepath, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved visualization to {filepath}")
    
    # Create metrics evolution chart if we have metrics data
    if metrics_series:
        visualize_metrics_over_time(metrics_series, timesteps_to_store[1:], output_dir, 
                                    current_lam_W, D_F, config, bio_prior)
    
    return filepath

def visualize_metrics_over_time(metrics_series, timesteps, output_dir, 
                             current_lam_W, D_F, config, bio_prior=False):
    """Visualize metrics evolution over simulation time."""
    if not metrics_series:
        return
    
    # Extract metric names and values
    metric_names = list(metrics_series[0].keys())
    
    # Filter metrics to include only numeric values and exclude redundant stats
    filtered_metrics = []
    for metric_name in metric_names:
        # Check if metric is numeric in the first item
        if isinstance(metrics_series[0].get(metric_name), (int, float)) and not metric_name.endswith('_count'):
            filtered_metrics.append(metric_name)
    
    # Separate field metrics from connectome metrics
    field_metrics = [m for m in filtered_metrics if m not in 
                    ['edge_overlap', 'modularity_similarity', 'weight_correlation', 
                     'combined_fidelity', 'source_clustering', 'generated_clustering']]
    
    connectome_metrics = [m for m in filtered_metrics if m in 
                         ['edge_overlap', 'modularity_similarity', 'weight_correlation', 
                          'combined_fidelity']]
    
    # Create two separate figures if we have both types of metrics
    if field_metrics:
        # Create figure for field metrics
        fig1, axs1 = plt.subplots(1, len(field_metrics), figsize=(4 * len(field_metrics), 5))
        
        # Handle case of single metric
        if len(field_metrics) == 1:
            axs1 = [axs1]
        
        # Plot each field metric's evolution
        for i, metric_name in enumerate(field_metrics):
            values = [m.get(metric_name, 0) for m in metrics_series]
            axs1[i].plot(timesteps, values, 'o-', linewidth=2)
            axs1[i].set_title(metric_name.replace('_', ' ').title())
            axs1[i].set_xlabel('Timestep')
            axs1[i].set_ylabel('Value')
            axs1[i].grid(True, linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        
        # Add overall title
        prior_text = "Bio Prior + " if bio_prior else ""
        plt.suptitle(f'{prior_text}Field Metrics Evolution: lam_W={current_lam_W:.3f}, D_F={D_F:.3f}', 
                     fontsize=14, y=1.05)
        
        # Save figure
        bio_tag = "bio_prior_" if bio_prior else ""
        filename = f"field_metrics_evolution_{bio_tag}lamW_{current_lam_W:.3f}_DF_{D_F:.3f}_grid{config.grid_size}.png"
        filepath = os.path.join(output_dir, filename)
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved field metrics evolution chart to {filepath}")
    
    # Create figure for connectome metrics if available
    if connectome_metrics:
        fig2, axs2 = plt.subplots(1, len(connectome_metrics), figsize=(4 * len(connectome_metrics), 5))
        
        # Handle case of single metric
        if len(connectome_metrics) == 1:
            axs2 = [axs2]
        
        # Plot each connectome metric's evolution
        for i, metric_name in enumerate(connectome_metrics):
            values = [m.get(metric_name, 0) for m in metrics_series]
            axs2[i].plot(timesteps, values, 'o-', linewidth=2, color='green')
            axs2[i].set_title(metric_name.replace('_', ' ').title())
            axs2[i].set_xlabel('Timestep')
            axs2[i].set_ylabel('Fidelity [0-1]')
            axs2[i].set_ylim(0, 1)
            axs2[i].grid(True, linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        
        # Add overall title
        prior_text = "Bio Prior + " if bio_prior else ""
        plt.suptitle(f'{prior_text}Connectome Fidelity Evolution: lam_W={current_lam_W:.3f}, D_F={D_F:.3f}', 
                     fontsize=14, y=1.05)
        
        # Save figure
        bio_tag = "bio_prior_" if bio_prior else ""
        filename = f"connectome_fidelity_evolution_{bio_tag}lamW_{current_lam_W:.3f}_DF_{D_F:.3f}_grid{config.grid_size}.png"
        filepath = os.path.join(output_dir, filename)
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved connectome fidelity evolution chart to {filepath}")
        
        # Also create a combined panel showing both metrics
        if field_metrics and len(field_metrics + connectome_metrics) <= 6:
            # Create a single figure with all metrics
            all_metrics = field_metrics + connectome_metrics
            fig3, axs3 = plt.subplots(2, 3, figsize=(12, 8))
            axs3 = axs3.flatten()
            
            # Plot each metric
            for i, metric_name in enumerate(all_metrics):
                if i < len(axs3):
                    is_connectome = metric_name in connectome_metrics
                    values = [m.get(metric_name, 0) for m in metrics_series]
                    color = 'green' if is_connectome else 'blue'
                    axs3[i].plot(timesteps, values, 'o-', linewidth=2, color=color)
                    axs3[i].set_title(metric_name.replace('_', ' ').title())
                    axs3[i].set_xlabel('Timestep')
                    
                    if is_connectome:
                        axs3[i].set_ylabel('Fidelity [0-1]')
                        axs3[i].set_ylim(0, 1)
                    else:
                        axs3[i].set_ylabel('Value')
                        
                    axs3[i].grid(True, linestyle='--', alpha=0.7)
            
            # Hide unused subplots
            for i in range(len(all_metrics), len(axs3)):
                axs3[i].set_visible(False)
            
            plt.tight_layout()
            
            # Add overall title
            prior_text = "Bio Prior + " if bio_prior else ""
            plt.suptitle(f'{prior_text}Combined Metrics Evolution: lam_W={current_lam_W:.3f}, D_F={D_F:.3f}', 
                         fontsize=14, y=0.98)
            
            # Save figure
            bio_tag = "bio_prior_" if bio_prior else ""
            filename = f"combined_metrics_evolution_{bio_tag}lamW_{current_lam_W:.3f}_DF_{D_F:.3f}_grid{config.grid_size}.png"
            filepath = os.path.join(output_dir, filename)
            plt.savefig(filepath, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"Saved combined metrics evolution chart to {filepath}")

def visualize_phase_diagram(pattern_intensity_summary_2d, lam_W_values, D_F_values, config, output_dir, 
                           is_final=True, bio_prior=False, metric_name="hotspot_fraction"):
    """Visualize phase diagram from parameter sweep."""
    # Convert to CPU arrays
    pattern_intensity_summary_2d_cpu = pattern_intensity_summary_2d.cpu().numpy()
    lam_W_values_cpu = lam_W_values.cpu().numpy()
    D_F_values_cpu = D_F_values.cpu().numpy()
    
    # Create DataFrame for seaborn
    df_2d = pd.DataFrame(
        pattern_intensity_summary_2d_cpu,
        index=[f"{v:.3f}" for v in lam_W_values_cpu],
        columns=[f"{v:.3f}" for v in D_F_values_cpu]
    )
    
    plt.figure(figsize=(12, 9))
    
    # Customize colormap based on metric
    cmap = "viridis"
    if metric_name == "pattern_persistence":
        cmap = "plasma"
    elif metric_name == "structural_complexity":
        cmap = "cividis"
    elif metric_name == "flow_coherence":
        cmap = "magma"
    elif metric_name == "pattern_quality":
        cmap = "inferno"
    elif metric_name in ["combined_fidelity", "edge_overlap", "modularity_similarity", "weight_correlation"]:
        cmap = "RdYlGn"  # Red-Yellow-Green colormap is good for fidelity metrics (red=bad, green=good)
    
    # Set vmax/vmin based on metric
    vmin, vmax = None, None
    if metric_name in ["combined_fidelity", "edge_overlap", "modularity_similarity", "weight_correlation"]:
        vmin, vmax = 0, 1  # Fidelity measures are in [0,1]
    
    # Create heatmap
    sns.heatmap(df_2d, cmap=cmap, linewidths=0.5, vmin=vmin, vmax=vmax,
                cbar_kws={"label": f"{metric_name.replace('_', ' ').title()}"})
    
    # Format title with metric information
    metric_display = metric_name.replace('_', ' ').title()
    
    # Include scaling info in title
    scaling_info = f" (grid: {config.grid_size}x{config.grid_size}, dt: {config.dt:.3f})" if config.use_parameter_scaling else ""
    prior_text = "Bio Prior + " if bio_prior else ""
    plt.title(f"{prior_text}GVFT 2D Phase Diagram - {metric_display}\n(lam_eta={config.lam_eta:.2f}, eta_coeff={config.eta_coeff:.3f}){scaling_info}", fontsize=16)
    plt.xlabel("D_F (Flow Field Diffusion)", fontsize=12)
    plt.ylabel("lam_W (W Decay Rate)", fontsize=12)
    plt.tight_layout()
    
    # Decide filename based on whether this is final or intermediate
    bio_tag = "bio_prior_" if bio_prior else ""
    status = "final" if is_final else "intermediate"
    filename = f"phase_diagram_{metric_name}_{bio_tag}{status}_grid{config.grid_size}.png"
    
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=200)
    plt.close()
    
    return filepath

def visualize_multi_metric_phase_diagrams(metrics_tensors, lam_W_values, D_F_values, config, output_dir, 
                                         is_final=True, bio_prior=False):
    """Create a single figure with multiple phase diagrams for different metrics."""
    # Create a grid of subplots
    metric_names = list(metrics_tensors.keys())
    n_metrics = len(metric_names)
    
    # Calculate grid dimensions for the subplots
    if n_metrics <= 3:
        n_rows, n_cols = 1, n_metrics
    else:
        n_cols = min(3, n_metrics)
        n_rows = (n_metrics + n_cols - 1) // n_cols
    
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 5*n_rows))
    
    # Handle case where there's only one row
    if n_rows == 1 and n_cols == 1:
        axs = np.array([axs])
    elif n_rows == 1:
        axs = axs.reshape(1, -1)
    
    # Convert values to CPU numpy arrays
    lam_W_values_cpu = lam_W_values.cpu().numpy()
    D_F_values_cpu = D_F_values.cpu().numpy()
    
    # Create a DataFrame for each metric and plot
    for i, metric_name in enumerate(metric_names):
        row = i // n_cols
        col = i % n_cols
        
        # Convert tensor to numpy
        metric_data = metrics_tensors[metric_name].cpu().numpy()
        
        # Create DataFrame
        df = pd.DataFrame(
            metric_data,
            index=[f"{v:.3f}" for v in lam_W_values_cpu],
            columns=[f"{v:.3f}" for v in D_F_values_cpu]
        )
        
        # Customize colormap based on metric
        cmap = "viridis"
        if metric_name == "pattern_persistence":
            cmap = "plasma"
        elif metric_name == "structural_complexity":
            cmap = "cividis"
        elif metric_name == "flow_coherence":
            cmap = "magma"
        elif metric_name == "pattern_quality":
            cmap = "inferno"
        elif metric_name in ["combined_fidelity", "edge_overlap", "modularity_similarity", "weight_correlation"]:
            cmap = "RdYlGn"  # Red-Yellow-Green for fidelity metrics
        
        # Set vmax/vmin based on metric
        vmin, vmax = None, None
        if metric_name in ["combined_fidelity", "edge_overlap", "modularity_similarity", "weight_correlation"]:
            vmin, vmax = 0, 1  # Fidelity measures are in [0,1]
        
        # Create heatmap
        sns.heatmap(df, cmap=cmap, linewidths=0.5, ax=axs[row, col], vmin=vmin, vmax=vmax,
                    cbar_kws={"label": f"{metric_name.replace('_', ' ').title()}"})
        
        # Set titles and labels
        axs[row, col].set_title(f"{metric_name.replace('_', ' ').title()}", fontsize=12)
        axs[row, col].set_xlabel("D_F (Flow Field Diffusion)")
        axs[row, col].set_ylabel("lam_W (W Decay Rate)")
    
    # Handle empty subplots (if any)
    for i in range(len(metric_names), n_rows * n_cols):
        row = i // n_cols
        col = i % n_cols
        fig.delaxes(axs[row, col])
    
    # Add overall title
    scaling_info = f"(grid: {config.grid_size}x{config.grid_size}, dt: {config.dt:.3f})" if config.use_parameter_scaling else ""
    prior_text = "Bio Prior + " if bio_prior else ""
    plt.suptitle(f"{prior_text}GVFT Multiple Metrics Phase Diagrams\n(lam_eta={config.lam_eta:.2f}, eta_coeff={config.eta_coeff:.3f}) {scaling_info}", 
                fontsize=16, y=0.98)
    
    plt.tight_layout()
    
    # Decide filename
    bio_tag = "bio_prior_" if bio_prior else ""
    status = "final" if is_final else "intermediate"
    filename = f"multi_metric_phase_diagrams_{bio_tag}{status}_grid{config.grid_size}.png"
    
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=200, bbox_inches='tight')
    plt.close()
    
    return filepath