import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns
import pandas as pd
import networkx as nx
import sys

# Add the parent directory to sys.path to allow imports from there
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from graph_to_neuroml import GraphToNeuroML
    NEUROML_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import GraphToNeuroML: {e}")
    print("NeuroML export functionality will be disabled.")
    NEUROML_AVAILABLE = False

def visualize_simulation(Fx_series, Fy_series, W_series, Module_coords_series, Graphs, 
                         X_np, Y_np, x_np, y_np, config, output_dir, params, bio_prior=False,
                         NetworkX_graphs=None, Communities=None):
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
    
    # Create directory for NeuroML output if it doesn't exist and NeuroML export is available
    neuroml_converter = None
    if NEUROML_AVAILABLE:
        neuroml_output_dir = os.path.join(output_dir, "neuroml")
        os.makedirs(neuroml_output_dir, exist_ok=True)
        
        # Initialize GraphToNeuroML converter
        neuroml_converter = GraphToNeuroML()
    
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
        
        # Prefer using pre-built NetworkX graph if available
        G = None
        if NetworkX_graphs is not None and t_idx < len(NetworkX_graphs) and NetworkX_graphs[t_idx] is not None:
            G = NetworkX_graphs[t_idx]
            # Draw connection arrows from the NetworkX graph
            for u, v, data in G.edges(data=True):
                if 'weight' in data and data['weight'] > 0.01:
                    x0, y0 = module_coords_t[u]
                    x1, y1 = module_coords_t[v]
                    arrow_alpha = np.clip(data['weight'] * 1.5, 0.1, 1.0)
                    arrow_width = np.clip(data['weight'] * 1.5, 0.5, 1.5)
                    dx_arrow = (x1 - x0) * 0.85
                    dy_arrow = (y1 - y0) * 0.85
                    if abs(dx_arrow) > 1e-6 or abs(dy_arrow) > 1e-6:
                        axs[3, i].arrow(x0, y0, dx_arrow, dy_arrow, head_width=0.06, head_length=0.08, 
                                      length_includes_head=True, alpha=arrow_alpha, color='green', 
                                      linewidth=arrow_width, zorder=2)
        else:
            # Create NetworkX graph from weights matrix if not provided
            G = nx.DiGraph()
            
            # Add nodes with positions
            for node_idx in range(len(module_coords_t)):
                # Convert from tensor to numpy or list for position data
                if isinstance(module_coords_t, torch.Tensor):
                    pos = module_coords_t[node_idx].cpu().numpy()
                else:
                    pos = module_coords_t[node_idx]
                G.add_node(node_idx, pos=tuple(pos))
            
            # Add edges with weights and visualization
            for m in range(config.num_modules):
                for n in range(config.num_modules):
                    if weights[m, n] > 0.01:
                        # Add to NetworkX graph
                        weight_val = float(weights[m, n]) if isinstance(weights[m, n], (np.number, torch.Tensor)) else weights[m, n]
                        G.add_edge(m, n, weight=weight_val)
                        
                        # Also draw the arrows on the visualization
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
        
        # Add community information if available
        if Communities is not None and t_idx < len(Communities) and Communities[t_idx] is not None:
            community_data = Communities[t_idx]
            # Implement community visualization if needed
            pass
        
        # Export to NeuroML if the converter is available and graph has edges
        if neuroml_converter and t > 0 and G.number_of_edges() > 0:
            try:
                # Define NeuroML output filename
                bio_tag = "bio_prior_" if bio_prior else ""
                neuroml_filename = f"gvft_network_{bio_tag}lamW_{current_lam_W:.3f}_DF_{D_F:.3f}_t{t}.net.nml"
                neuroml_path = os.path.join(neuroml_output_dir, neuroml_filename)
                
                # Generate cell type assignments
                node_types = neuroml_converter.assign_cell_types(G)
                
                # Export to NeuroML
                neuroml_converter.export_to_neuroml(G, neuroml_path, node_types)
                
                # Also save cell type assignments to JSON for reference
                json_filename = f"cell_types_{bio_tag}lamW_{current_lam_W:.3f}_DF_{D_F:.3f}_t{t}.json"
                json_path = os.path.join(neuroml_output_dir, json_filename)
                neuroml_converter.export_cell_types_to_json(node_types, json_path)
                
                print(f"  Saved NeuroML for timestep t={t} to {neuroml_filename}")
            except Exception as e:
                print(f"  Error exporting to NeuroML for timestep t={t}: {e}")
        
        axs[3, i].set_xlim(-1, 1); axs[3, i].set_ylim(-1, 1)
        axs[3, i].set_aspect('equal', adjustable='box')
        axs[3, i].set_xticks([]); axs[3, i].set_yticks([])
    
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
    
    # Create figure with subplots
    fig, axs = plt.subplots(1, len(metric_names), figsize=(4 * len(metric_names), 5))
    
    # Handle case of single metric
    if len(metric_names) == 1:
        axs = [axs]
    
    # Plot each metric's evolution
    for i, metric_name in enumerate(metric_names):
        values = [m[metric_name] for m in metrics_series]
        axs[i].plot(timesteps, values, 'o-', linewidth=2)
        axs[i].set_title(metric_name.replace('_', ' ').title())
        axs[i].set_xlabel('Timestep')
        axs[i].set_ylabel('Value')
        axs[i].grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    
    # Add overall title
    prior_text = "Bio Prior + " if bio_prior else ""
    plt.suptitle(f'{prior_text}Metrics Evolution: lam_W={current_lam_W:.3f}, D_F={D_F:.3f}', 
                 fontsize=14, y=1.05)
    
    # Save figure
    bio_tag = "bio_prior_" if bio_prior else ""
    filename = f"metrics_evolution_{bio_tag}lamW_{current_lam_W:.3f}_DF_{D_F:.3f}_grid{config.grid_size}.png"
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved metrics evolution chart to {filepath}")

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
    
    # Create heatmap
    sns.heatmap(df_2d, cmap=cmap, linewidths=0.5,
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
        
        # Create heatmap
        sns.heatmap(df, cmap=cmap, linewidths=0.5, ax=axs[row, col],
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