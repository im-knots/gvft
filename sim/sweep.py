import torch
import time
import numpy as np
import gc
import os

from solvers import create_crank_nicolson_operators, solve_diffusion_equation
from field_utils import normalize_field, clip_field
from metrics_utils import compute_field_metrics

def optimize_batch_size(config):
    """Calculate an optimal batch size based on grid size and GPU memory."""
    base_grid_size = 200
    device = config.device
    
    if device.type != 'cuda':
        return 4  # Default for CPU
        
    try:
        total_memory = torch.cuda.get_device_properties(0).total_memory
        # Adjust available memory based on grid size
        # Larger grids need more memory per simulation
        memory_per_sim = (config.grid_size / base_grid_size)**2 * 100 * 1024 * 1024  # Base ~100MB for 100x100
        
        # Use 80% of available memory
        optimal_batch_size = max(1, int(total_memory * 0.8 / memory_per_sim))
        
        # Cap at reasonable values based on grid size
        if config.grid_size <= 100:
            return min(optimal_batch_size, 32)
        elif config.grid_size <= 200:
            return min(optimal_batch_size, 16)
        elif config.grid_size <= 500:
            return min(optimal_batch_size, 8)
        else:
            return min(optimal_batch_size, 4)
    except Exception as e:
        print(f"Error calculating batch size: {e}")
        # Conservative defaults based on grid size
        if config.grid_size <= 100:
            return 10
        elif config.grid_size <= 200:
            return 6
        elif config.grid_size <= 500:
            return 3
        else:
            return 1

def run_sweep_simulation(param_batch, config, apply_laplacian, Fx_init, Fy_init, W_init, eta_init):
    """Run a batch of parameter sweep simulations."""
    results = []
    batch_start_time = time.time()
    
    # Try to precompute eta operators which are fixed for all simulations
    try:
        eta_A_inv_B, eta_A_inv = create_crank_nicolson_operators(
            config.D_eta, config.lam_eta, config.dt, config.grid_size, apply_laplacian)
        precomputed_eta = True
    except Exception as e:
        print(f"Could not precompute eta operators: {e}")
        precomputed_eta = False
    
    for batch_idx, (lam_W_idx, D_F_idx) in enumerate(param_batch):
        lam_W = config.lam_W_values[lam_W_idx]
        D_F = config.D_F_values[D_F_idx]
        total_combinations = len(config.lam_W_values) * len(config.D_F_values)
        current_combination = lam_W_idx * len(config.D_F_values) + D_F_idx + 1
        
        sim_start_time = time.time()
        print(f"Starting sweep simulation {current_combination}/{total_combinations}: lam_W={lam_W:.3f}, D_F={D_F:.3f}")
        
        # Create Crank-Nicolson operators for each field
        F_A_inv_B, F_A_inv = create_crank_nicolson_operators(
            D_F, config.lam_F, config.dt, config.grid_size, apply_laplacian)
        W_A_inv_B, W_A_inv = create_crank_nicolson_operators(
            config.D_W, lam_W, config.dt, config.grid_size, apply_laplacian)
        
        # Only create eta operators if not precomputed
        if not precomputed_eta:
            eta_A_inv_B, eta_A_inv = create_crank_nicolson_operators(
                config.D_eta, config.lam_eta, config.dt, config.grid_size, apply_laplacian)
        
        # Reset fields for this parameter set
        Fx = Fx_init.clone()
        Fy = Fy_init.clone()
        W = W_init.clone()
        eta = eta_init.clone()
        
        # Store initial metrics for tracking performance
        initial_metrics = compute_field_metrics(Fx, Fy, W, W_init)
        metrics_over_time = [initial_metrics]
        
        # Run simulation
        for t in range(config.timesteps_sweep):
            if t % 100 == 0:
                print(f"Sweep simulation {current_combination}/{total_combinations}: "
                      f"lam_W={lam_W:.3f}, D_F={D_F:.3f} - Timestep {t}/{config.timesteps_sweep}")
                
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
            if max_abs_W > 1e-10:
                W = W / max_abs_W * 2.0
            else:
                W.fill_(0.0)
            
            # Compute metrics at regular intervals
            if t % (config.timesteps_sweep // 5) == 0 or t == config.timesteps_sweep - 1:
                current_metrics = compute_field_metrics(Fx, Fy, W, W_init)
                metrics_over_time.append(current_metrics)
        
        # Calculate final metrics
        final_metrics = compute_field_metrics(Fx, Fy, W, W_init)
        
        # Report simulation time and results
        sim_time = time.time() - sim_start_time
        print(f"Completed sweep simulation {current_combination}/{total_combinations}: "
              f"lam_W={lam_W:.3f}, D_F={D_F:.3f}")
        print(f"  Hotspot Fraction: {final_metrics['hotspot_fraction']:.3f}")
        print(f"  Pattern Persistence: {final_metrics['pattern_persistence']:.3f}")
        print(f"  Structural Complexity: {final_metrics['structural_complexity']:.3f}")
        print(f"  Flow Coherence: {final_metrics['flow_coherence']:.3f}")
        print(f"  Pattern Quality: {final_metrics['pattern_quality']:.3f}")
        print(f"  Simulation completed in {sim_time:.2f}s")
        
        results.append((lam_W_idx.item(), D_F_idx.item(), final_metrics))
        
        # Clear GPU memory after each simulation
        del F_A_inv_B, F_A_inv, W_A_inv_B, W_A_inv
        if not precomputed_eta:
            del eta_A_inv_B, eta_A_inv
        torch.cuda.empty_cache()
    
    # Report batch timing
    batch_time = time.time() - batch_start_time
    print(f"Batch of {len(param_batch)} simulations completed in {batch_time:.1f}s "
          f"({batch_time/len(param_batch):.1f}s per simulation)")
    
    return results

def run_parameter_sweep(config, apply_laplacian, Fx_init, Fy_init, W_init, eta_init, output_dir, 
                        save_checkpoints=False, save_intermediates=False):
    """Run full parameter sweep with batching."""
    global_start_time = time.time()
    
    # Determine optimal batch size
    batch_size = optimize_batch_size(config)
    print(f"\nUsing batch size: {batch_size} for grid size {config.grid_size}x{config.grid_size}")

    # Generate all parameter combinations
    param_combinations = [
        (torch.tensor(j, device=config.device), torch.tensor(i, device=config.device))
        for j in range(len(config.lam_W_values)) for i in range(len(config.D_F_values))
    ]

    # Create batches
    batches = [param_combinations[i:i + batch_size] for i in range(0, len(param_combinations), batch_size)]
    total_batches = len(batches)
    total_combinations = len(param_combinations)

    print(f"Total parameter combinations: {total_combinations}")
    print(f"Created {total_batches} batches with up to {batch_size} simulations per batch")

    # Initialize results tensors for all metrics
    metrics_names = ['hotspot_fraction', 'pattern_persistence', 'structural_complexity', 
                    'flow_coherence', 'pattern_quality']
    metrics_tensors = {}
    
    for metric in metrics_names:
        metrics_tensors[metric] = torch.zeros(
            len(config.lam_W_values), len(config.D_F_values), device=config.device)
    
    # Process batches with progress tracking
    completed_sims = 0
    for batch_idx, param_batch in enumerate(batches):
        batch_start_time = time.time()
        print(f"\nProcessing batch {batch_idx + 1}/{total_batches} with {len(param_batch)} simulations...")
        
        # Run batch
        batch_results = run_sweep_simulation(param_batch, config, apply_laplacian, Fx_init, Fy_init, W_init, eta_init)
        
        # Store results for all metrics
        for lam_W_idx, D_F_idx, metrics in batch_results:
            for metric_name, value in metrics.items():
                if metric_name in metrics_tensors:
                    # Fix: Convert value to a PyTorch tensor on the same device
                    metrics_tensors[metric_name][int(lam_W_idx), int(D_F_idx)] = torch.tensor(value, device=config.device)
            completed_sims += 1
        
        # Report progress
        batch_time = time.time() - batch_start_time
        elapsed_total = time.time() - global_start_time
        progress = 100.0 * completed_sims / total_combinations
        
        # Calculate performance metrics
        sims_per_second = completed_sims / elapsed_total
        remaining_sims = total_combinations - completed_sims
        estimated_remaining = remaining_sims / max(0.1, sims_per_second)
        
        print(f"Batch {batch_idx + 1} completed in {batch_time:.2f}s ({batch_time/len(param_batch):.2f}s per simulation)")
        print(f"Overall progress: {progress:.1f}% - {completed_sims}/{total_combinations} simulations")
        print(f"Time elapsed: {elapsed_total/60:.1f} minutes, Est. remaining: {estimated_remaining/60:.1f} minutes")
        print(f"Performance: {sims_per_second:.2f} simulations/second")
        
        # Save intermediate results every 2 batches or at the end (if enabled)
        is_final_batch = batch_idx == total_batches - 1
        is_bio_prior = torch.max(torch.abs(Fx_init)) > 0.5 or torch.max(torch.abs(W_init)) > 1.0
        bio_tag = "bio_prior_" if is_bio_prior else ""
        
        # Save checkpoints if enabled or at the end
        if (save_checkpoints and (batch_idx + 1) % 2 == 0) or is_final_batch:
            # Save all metrics as numpy files
            for metric_name, tensor in metrics_tensors.items():
                checkpoint_data = tensor.cpu().numpy()
                checkpoint_file = os.path.join(output_dir, f"{metric_name}_{bio_tag}grid{config.grid_size}_checkpoint_{batch_idx+1}.npy")
                np.save(checkpoint_file, checkpoint_data)
                
            # Only print message if checkpoints are actually saved
            if save_checkpoints or is_final_batch:
                print(f"Saved checkpoint metrics to {output_dir}")
        
        # Create visualizations if enabled or at the end
        if (save_intermediates and (batch_idx + 1) % 5 == 0) or is_final_batch:
            try:
                from visualization import visualize_phase_diagram, visualize_multi_metric_phase_diagrams
                
                # Only create visualizations if explicitly enabled or it's the final batch
                if save_intermediates or is_final_batch:
                    # Traditional visualization with hotspot fraction
                    visualize_phase_diagram(
                        metrics_tensors['hotspot_fraction'], 
                        config.lam_W_values, 
                        config.D_F_values,
                        config,
                        output_dir,
                        is_final=is_final_batch,
                        bio_prior=is_bio_prior,
                        metric_name="hotspot_fraction"
                    )
                    
                    # Create multi-metric visualization
                    visualize_multi_metric_phase_diagrams(
                        metrics_tensors,
                        config.lam_W_values,
                        config.D_F_values,
                        config,
                        output_dir,
                        is_final=is_final_batch,
                        bio_prior=is_bio_prior
                    )
                    
                    # Only print message if visualizations are actually saved
                    if save_intermediates or is_final_batch:
                        print(f"Saved {'intermediate' if not is_final_batch else 'final'} visualizations")
            except Exception as e:
                print(f"Warning: Could not create visualizations: {e}")
        
        # Force memory cleanup
        torch.cuda.empty_cache()
        gc.collect()
    
    return metrics_tensors

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

