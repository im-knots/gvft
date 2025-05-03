import torch
import time
import gc
import numpy as np
import os

from solvers import create_crank_nicolson_operators, solve_diffusion_equation
from field_utils import normalize_field, clip_field
from module_sampling import sample_module_positions, build_connectivity_graph
from visualization import visualize_simulation
from metrics_utils import compute_field_metrics, compute_connectome_metrics

def save_metrics_series(metrics_series, output_file, config):
    """Save the metrics series to a CSV file for later analysis."""
    import pandas as pd
    import os
    
    # Create a DataFrame from the metrics series
    metrics_df = pd.DataFrame(metrics_series)
    
    # Add a timestep column
    timesteps = list(range(0, config.timesteps_sim + 1, config.view_step))
    if config.timesteps_sim not in timesteps:
        timesteps.append(config.timesteps_sim)
    
    # Handle case where metrics might have fewer entries than timesteps
    timesteps = timesteps[:len(metrics_series)]
    metrics_df.insert(0, 'timestep', timesteps)
    
    # Ensure the directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Save to CSV
    metrics_df.to_csv(output_file, index=False)
    print(f"Saved metrics to {output_file}")

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
    # Use biological positions if available, otherwise sample
    module_coords = sample_module_positions(
        Fx, Fy, W, config.num_modules, config.beta, config.gamma, domain['X'], domain['Y'], config=config)
    Module_coords_series.append(module_coords.clone().cpu().numpy())
    
    # Store initial fields
    if 0 in timesteps_to_store:
        Fx_series.append(Fx.clone().cpu().numpy())
        Fy_series.append(Fy.clone().cpu().numpy())
        W_series.append(W.clone().cpu().numpy())
    
    # Build initial connectivity graph for t=0
    initial_weights = build_connectivity_graph(module_coords, Fx, Fy, W, domain, config)
    Graphs.append(initial_weights.clone().cpu().numpy())
    
    # Add connectome fidelity metrics for initial state if source connections are available
    if hasattr(config, 'source_connections') and config.source_connections:
        connectome_metrics = compute_connectome_metrics(
            config.source_connections,
            module_coords.cpu().numpy(),
            Fx.cpu().numpy(), Fy.cpu().numpy(), W.cpu().numpy(),
            config
        )
        
        # Add these metrics to the current metrics
        initial_metrics.update(connectome_metrics)
        
        # Print initial connectome fidelity
        print(f"  Initial Connectome Fidelity Metrics:")
        for metric_name, value in connectome_metrics.items():
            if isinstance(value, (int, float)):
                print(f"    {metric_name}: {value:.3f}")
    
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
            
            # At t=0, we already have biological positions
            # For t>0, we use dynamic positioning if config.use_dynamic_positions is True
            # Otherwise, keep the same positions as t=0 to maintain structural similarity
            if t > 0 and hasattr(config, 'use_dynamic_positions') and config.use_dynamic_positions:
                # Update module positions based on current fields
                module_coords = sample_module_positions(
                    Fx, Fy, W, config.num_modules, config.beta, config.gamma, domain['X'], domain['Y'], config=config)
            
            # Build connectivity graph
            weights = build_connectivity_graph(module_coords, Fx, Fy, W, domain, config)
            
            # Add connectome fidelity metrics if source connections are available
            if hasattr(config, 'source_connections') and config.source_connections:
                connectome_metrics = compute_connectome_metrics(
                    config.source_connections,
                    module_coords.cpu().numpy(),
                    Fx.cpu().numpy(), Fy.cpu().numpy(), W.cpu().numpy(),
                    config
                )
                
                # Add these metrics to the current metrics
                current_metrics.update(connectome_metrics)
                
                # Print connectome fidelity metrics
                if t % (config.timesteps_sim // 5) == 0:
                    print(f"  Connectome Fidelity Metrics at t={t}:")
                    for metric_name, value in connectome_metrics.items():
                        if isinstance(value, (int, float)):
                            print(f"    {metric_name}: {value:.3f}")
            
            metrics_series.append(current_metrics)
            
            # Print metrics progress every 1/5 of the simulation
            if t % (config.timesteps_sim // 5) == 0:
                print(f"  Metrics at t={t}:")
                for metric_name, value in current_metrics.items():
                    if isinstance(value, (int, float)) and metric_name not in ['source_edge_count', 'generated_edge_count']:
                        print(f"    {metric_name}: {value:.3f}")
            
        # Only store data at specified timesteps
        if t in timesteps_to_store:
            # Store module positions and connectivity graphs
            Module_coords_series.append(module_coords.clone().cpu().numpy())
            
            # Get weights if not already calculated
            if 'weights' not in locals() or t not in timesteps_to_store:
                weights = build_connectivity_graph(module_coords, Fx, Fy, W, domain, config)
                
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

