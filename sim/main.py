import os
import time
import argparse
import torch
import numpy as np
import gc
import matplotlib.pyplot as plt

# Import modules
from config import GVFTConfig
from field_utils import build_domain, build_laplacian_operator, initialize_fields, load_neuroml_fields
from sweep import run_parameter_sweep
from simulation import run_full_simulation, select_simulation_parameters
from visualization import visualize_phase_diagram, visualize_multi_metric_phase_diagrams

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='GVFT Field Simulation with GPU Acceleration')
    parser.add_argument('--grid-size', type=int, default=200, help='Grid size for simulation')
    parser.add_argument('--no-scaling', action='store_true', help='Disable parameter scaling')
    parser.add_argument('--output-dir', type=str, default='../figures', help='Output directory for figures')
    parser.add_argument('--full-sims-only', action='store_true', help='Skip parameter sweep, run full simulations only')
    parser.add_argument('--sweep-only', action='store_true', help='Run parameter sweep only, skip full simulations')
    parser.add_argument('--neuroml-fields', type=str, help='Directory containing preprocessed NeuroML GVFT fields (from neuroml_to_gvft.py)')
    parser.add_argument('--neuroml-basename', type=str, default='PharyngealNetwork', help='Base name of the processed NeuroML files')
    parser.add_argument('--primary-metric', type=str, default='pattern_quality', 
                        choices=['hotspot_fraction', 'pattern_persistence', 'structural_complexity', 
                                'flow_coherence', 'pattern_quality'],
                        help='Primary metric to use for parameter selection')
    # New arguments for controlling output
    parser.add_argument('--save-checkpoints', action='store_true', help='Save checkpoint files during parameter sweep')
    parser.add_argument('--save-intermediates', action='store_true', help='Save intermediate visualizations during parameter sweep')
    
    return parser.parse_args()

def setup_environment(args):
    """Set up simulation environment."""
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Initialize configuration
    config = GVFTConfig(
        grid_size=args.grid_size,
        use_parameter_scaling=not args.no_scaling
    )
    
    # Print configuration
    config.print_config()
    
    return config

def main():
    """Main function to run GVFT simulation."""
    global_start_time = time.time()
    
    # Parse arguments
    args = parse_arguments()
    
    # Setup environment and configuration
    config = setup_environment(args)
    
    # Print CUDA info if available
    if config.device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory allocated: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
        print(f"Memory reserved: {torch.cuda.memory_reserved(0) / 1024**2:.2f} MB")
    
    # Setup domain
    domain = build_domain(config.grid_size, config.device)
    
    # Build Laplacian operator
    apply_laplacian = build_laplacian_operator(
        config.grid_size, domain['dx'], domain['dy'], config.device)
    
    # Flag to track if bio priors are used
    using_bio_priors = args.neuroml_fields is not None
    
    # Initialize fields - either from NeuroML or random
    if using_bio_priors:
        print(f"\nLoading fields from preprocessed NeuroML data: {args.neuroml_fields}")
        Fx_init, Fy_init, W_init, eta_init = load_neuroml_fields(
            args.neuroml_fields, args.neuroml_basename, config.grid_size, config.device)
        print("Successfully loaded biological GVFT fields as simulation priors")
    else:
        print("\nUsing random field initialization")
        filter_scale = 200 / config.grid_size if config.use_parameter_scaling else 1.0
        Fx_init, Fy_init, W_init, eta_init = initialize_fields(
            config.grid_size, config.device, filter_scale)
    
    # Run parameter sweep if not skipped
    if not args.full_sims_only:
        print("\nStarting 2D Parameter Sweep (lam_W vs D_F) with GPU acceleration...")
        print(f"Using {args.primary_metric} as the primary metric for parameter selection")
        print(f"Checkpoints: {'Enabled' if args.save_checkpoints else 'Disabled'}")
        print(f"Intermediate visualizations: {'Enabled' if args.save_intermediates else 'Disabled'}")
        
        metrics_tensors = run_parameter_sweep(
            config, apply_laplacian, Fx_init, Fy_init, W_init, eta_init, args.output_dir,
            save_checkpoints=args.save_checkpoints, save_intermediates=args.save_intermediates)
        
        # Visualize final phase diagram for the primary metric
        visualize_phase_diagram(
            metrics_tensors[args.primary_metric], 
            config.lam_W_values, 
            config.D_F_values,
            config,
            args.output_dir,
            is_final=True,
            bio_prior=using_bio_priors,
            metric_name=args.primary_metric
        )
        
        # Visualize all metrics in a multi-metric figure
        visualize_multi_metric_phase_diagrams(
            metrics_tensors,
            config.lam_W_values,
            config.D_F_values,
            config,
            args.output_dir,
            is_final=True,
            bio_prior=using_bio_priors
        )
        
        # Select parameters for full simulations based on the primary metric
        if not args.sweep_only:
            selected_params = select_simulation_parameters(metrics_tensors[args.primary_metric], config)
    else:
        # Hard-coded parameters if skipping sweep
        selected_params = [
            (0.3, 0.004),
            (0.2, 0.006),
            (0.4, 0.002)
        ]
        print(f"Using pre-defined parameters for full simulations: {selected_params}")
    
    # Run full simulations if not skipped
    if not args.sweep_only:
        print("\nRunning full simulations for detailed analysis...")
        
        full_simulation_params = [
            (idx, lam_W, D_F) for idx, (lam_W, D_F) in enumerate(selected_params)
        ]
        
        for sim_params in full_simulation_params:
            run_full_simulation(
                sim_params, config, apply_laplacian, domain,
                Fx_init, Fy_init, W_init, eta_init, args.output_dir
            )
    
    # Report total runtime
    total_time = (time.time() - global_start_time) / 60
    print(f"\nAll simulations completed in {total_time:.2f} minutes.")
    
    # Final summary with scaling information
    if config.use_parameter_scaling:
        print("\nSimulation used parameter scaling for grid consistency:")
        print(f"  Base grid: 200x200, Actual grid: {config.grid_size}x{config.grid_size}")
        print(f"  Scaling factor: {config.scale_factor:.3f}")
        print("\nScaled parameters:")
        print(f"  D_W: {config.D_W:.6f} (from {0.03:.6f})")
        print(f"  D_F: scaled range [{config.D_F_values[0].item():.6f} to {config.D_F_values[-1].item():.6f}]")
        print(f"  dt: {config.dt:.6f} (from {0.1:.6f})")
        print(f"  timesteps: {config.timesteps_sweep}/{config.timesteps_sim} (from 100/500)")

if __name__ == "__main__":
    main()