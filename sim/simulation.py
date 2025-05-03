import torch
import time
import gc
import numpy as np
import os

from solvers import create_crank_nicolson_operators, solve_diffusion_equation
from field_utils import normalize_field, clip_field
from module_sampling import sample_module_positions, build_connectivity_graph
from visualization import visualize_simulation
from metrics_utils import compute_field_metrics

def run_full_simulation(params, config, apply_laplacian, domain, Fx_init, Fy_init, W_init, eta_init, output_dir):
    """Run a full simulation with selected parameters and create visualizations."""
    idx, current_lam_W, D_F = params
    total_simulations = len(params)
    
    # Check if bio priors were used (simple heuristic)
    bio_prior = False
    if torch.max(torch.abs(Fx_init)) > 0.5 or torch.max(torch.abs(W_init)) > 1.0:
        bio_prior = True
    
    prior_text = "Bio Prior + " if bio_prior else ""
    print(f"--- Starting full simulation {idx+1}/{total_simulations}: {prior_text}lam_W={current_lam_W:.3f}, D_F={D_F:.3f} ---")
    
    # Create Crank-Nicolson operators for each field
    F_A_inv_B, F_A_inv = create_crank_nicolson_operators(
        D_F, config.lam_F, config.dt, config.grid_size, apply_laplacian)
    W_A_inv_B, W_A_inv = create_crank_nicolson_operators(
        config.D_W, current_lam_W, config.dt, config.grid_size, apply_laplacian)
    eta_A_inv_B, eta_A_inv = create_crank_nicolson_operators(
        config.D_eta, config.lam_eta, config.dt, config.grid_size, apply_laplacian)
    
    # Reset fields
    Fx = Fx_init.clone()
    Fy = Fy_init.clone()
    W = W_init.clone()
    eta = eta_init.clone()
    
    # Initialize lists to store fields and graphs
    # Use lists to reduce memory usage during simulation
    Fx_series, Fy_series, W_series = [], [], []
    Graphs, Module_coords_series = [], []
    
    # Track metrics over time
    metrics_series = []
    
    # Only store fields at specified timesteps to save memory
    timesteps_to_store = list(range(0, config.timesteps_sim + 1, config.view_step))
    if config.timesteps_sim not in timesteps_to_store:
        timesteps_to_store.append(config.timesteps_sim)
    
    # Calculate initial metrics
    initial_metrics = compute_field_metrics(Fx, Fy, W, W_init)
    metrics_series.append(initial_metrics)
    
    # Initial module positions at t=0
    module_coords = sample_module_positions(
        Fx, Fy, W, config.num_modules, config.beta, config.gamma, domain['X'], domain['Y'])
    Module_coords_series.append(module_coords.clone().cpu().numpy())
    
    # Store initial fields
    if 0 in timesteps_to_store:
        Fx_series.append(Fx.clone().cpu().numpy())
        Fy_series.append(Fy.clone().cpu().numpy())
        W_series.append(W.clone().cpu().numpy())
    
    for t in range(1, config.timesteps_sim + 1):
        if t % max(1, config.timesteps_sim // 10) == 0:
            print(f"Full simulation {idx+1}/{total_simulations}: "
                  f"{prior_text}lam_W={current_lam_W:.3f}, D_F={D_F:.3f} - Timestep {t}/{config.timesteps_sim}")
        
        # Calculate field magnitudes and gradients
        F_mag = torch.sqrt(Fx**2 + Fy**2)
        Wy, Wx = torch.gradient(W)
        
        # Update eta field
        S_eta = 0.5 * (F_mag**2 + W**2)
        eta = solve_diffusion_equation(eta, eta_A_inv_B, eta_A_inv, S_eta, config.dt)
        eta = clip_field(eta, -5, 5)
        
        # Update Fx field
        noise_Fx = torch.randn_like(Fx) * config.noise_F
        S_Fx = config.beta_coupling * Wx + noise_Fx
        Fx = solve_diffusion_equation(Fx, F_A_inv_B, F_A_inv, S_Fx, config.dt)
        
        # Update Fy field
        noise_Fy = torch.randn_like(Fy) * config.noise_F
        S_Fy = config.beta_coupling * Wy + noise_Fy
        Fy = solve_diffusion_equation(Fy, F_A_inv_B, F_A_inv, S_Fy, config.dt)
        
        # Normalize flow field
        Fx, Fy = normalize_field(Fx, Fy)
        
        # Update W field
        noise_W_t = torch.randn_like(W) * config.noise_W
        S_W = config.eta_coeff * eta + config.alpha * F_mag + noise_W_t
        W = solve_diffusion_equation(W, W_A_inv_B, W_A_inv, S_W, config.dt)
        W = clip_field(W)
        
        # Normalize W field
        max_abs_W = torch.max(torch.abs(W))
        if max_abs_W > 1e-8:
            W = W / max_abs_W * 2.0
        else:
            W.fill_(0.0)
        
        # Calculate metrics at regular intervals
        if t % (config.timesteps_sim // 10) == 0 or t in timesteps_to_store:
            current_metrics = compute_field_metrics(Fx, Fy, W, W_init)
            metrics_series.append(current_metrics)
            
            # Print metrics progress every 1/5 of the simulation
            if t % (config.timesteps_sim // 5) == 0:
                print(f"  Metrics at t={t}:")
                for metric_name, value in current_metrics.items():
                    print(f"    {metric_name}: {value:.3f}")
            
        # Only compute module positions and store data at specified timesteps
        if t in timesteps_to_store:
            # Update module positions based on current fields
            module_coords = sample_module_positions(
                Fx, Fy, W, config.num_modules, config.beta, config.gamma, domain['X'], domain['Y'])
            
            # Build connectivity graph
            weights = build_connectivity_graph(module_coords, Fx, Fy, W, domain, config)
            
            # Store data at specified timesteps
            Module_coords_series.append(module_coords.clone().cpu().numpy())
            Graphs.append(weights.clone().cpu().numpy())
            Fx_series.append(Fx.clone().cpu().numpy())
            Fy_series.append(Fy.clone().cpu().numpy())
            W_series.append(W.clone().cpu().numpy())
    
    print(f"Full simulation {idx+1}/{total_simulations}: "
          f"{prior_text}lam_W={current_lam_W:.3f}, D_F={D_F:.3f} - Field updates completed. Starting visualization...")
    
    # Move tensors to CPU for visualization
    X_np = domain['X'].cpu().numpy()
    Y_np = domain['Y'].cpu().numpy()
    x_np = domain['x'].cpu().numpy()
    y_np = domain['y'].cpu().numpy()
    
    # Save the final metrics to a file
    metrics_file = os.path.join(output_dir, f"metrics_lamW_{current_lam_W:.3f}_DF_{D_F:.3f}{'_bio_prior' if bio_prior else ''}.csv")
    save_metrics_series(metrics_series, metrics_file, config)
    
    # Visualization
    visualize_simulation(
        Fx_series, Fy_series, W_series, Module_coords_series, Graphs,
        X_np, Y_np, x_np, y_np, config, output_dir, params, bio_prior=bio_prior
    )
    
    print(f"--- Completed full simulation {idx+1}/{total_simulations}: "
          f"{prior_text}lam_W={current_lam_W:.3f}, D_F={D_F:.3f} ---")
    
    # Force cleanup
    torch.cuda.empty_cache()
    gc.collect()
    
    return idx

def save_metrics_series(metrics_series, filename, config):
    """Save metrics evolution to a CSV file."""
    import pandas as pd
    import os
    
    # Create a list of timesteps
    timesteps = [0]
    step = config.timesteps_sim // (len(metrics_series) - 1) if len(metrics_series) > 1 else config.timesteps_sim
    for i in range(1, len(metrics_series)):
        timesteps.append(i * step)
    
    # Create DataFrame
    df = pd.DataFrame(metrics_series)
    
    # Add timestep column
    df['timestep'] = timesteps
    
    # Reorder columns to put timestep first
    cols = df.columns.tolist()
    cols = cols[-1:] + cols[:-1]
    df = df[cols]
    
    # Save to CSV
    df.to_csv(filename, index=False)
    print(f"Metrics saved to {filename}")

def select_simulation_parameters(pattern_intensity_summary_2d, config):
    """Select interesting parameter regimes for detailed simulations."""
    print("\nSelecting interesting parameter regimes for detailed simulations...")
    
    # Get CPU arrays
    pattern_intensity_cpu = pattern_intensity_summary_2d.cpu().numpy()
    lam_W_values_cpu = config.lam_W_values.cpu().numpy()
    D_F_values_cpu = config.D_F_values.cpu().numpy()
    
    # Method 1: Find row with maximum variance (most interesting transitions)
    row_std_dev = np.std(pattern_intensity_cpu, axis=1)
    chosen_lam_W_index = np.argmax(row_std_dev)
    chosen_lam_W = lam_W_values_cpu[chosen_lam_W_index]

    selected_row_values = pattern_intensity_cpu[chosen_lam_W_index]
    sorted_D_F_indices = np.argsort(selected_row_values)

    # Take low, medium, and high values from the most varying row
    low_idx = sorted_D_F_indices[0]
    mid_idx = sorted_D_F_indices[len(sorted_D_F_indices) // 2]
    high_idx = sorted_D_F_indices[-1]

    selected_D_F_indices = sorted(set([low_idx, mid_idx, high_idx]))
    selected_params = [(chosen_lam_W, D_F_values_cpu[idx]) for idx in selected_D_F_indices]
    
    # Method 2: Also find specific parameter pairs where value is around 0.5
    # These are likely interesting transition regions
    value_diff = np.abs(pattern_intensity_cpu - 0.5)
    closest_pairs = np.unravel_index(np.argsort(value_diff.ravel())[:3], value_diff.shape)
    
    for lam_idx, d_idx in zip(closest_pairs[0], closest_pairs[1]):
        pair = (lam_W_values_cpu[lam_idx], D_F_values_cpu[d_idx])
        if pair not in selected_params:
            selected_params.append(pair)
    
    # Limit to a reasonable number of simulations
    max_full_sims = 6
    if len(selected_params) > max_full_sims:
        selected_params = selected_params[:max_full_sims]

    print(f"Selected {len(selected_params)} parameter pairs for full simulation: {selected_params}")
    
    return selected_params