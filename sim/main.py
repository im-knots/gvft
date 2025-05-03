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
from sweep import run_parameter_sweep, select_simulation_parameters  # Changed import to get this from sweep.py
from simulation import run_full_simulation  # Only import run_full_simulation from simulation.py
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
                                'flow_coherence', 'pattern_quality', 'combined_fidelity'],
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

def load_biological_module_positions(neuroml_dir, basename):
    """Load the module positions from the NeuroML-derived positions."""
    try:
        from neuroml_to_gvft import parse_neuroml2_file, assign_synthetic_positions
        
        # First, try to load precomputed positions if available
        positions_file = os.path.join(neuroml_dir, f"{basename}_positions.npy")
        if os.path.exists(positions_file):
            print(f"Loading precomputed module positions from {positions_file}")
            module_positions = np.load(positions_file)
            return module_positions
        
        # If not available, parse the NeuroML file and compute positions
        neuroml_file_path = os.path.join(neuroml_dir, f"{basename}.net.nml")
        if os.path.exists(neuroml_file_path):
            print(f"Parsing NeuroML file to extract module positions: {neuroml_file_path}")
            data = parse_neuroml2_file(neuroml_file_path)
            if not data['neurons']:
                print("No neurons found in NeuroML file")
                return None
                
            # Compute positions using force-directed layout
            neurons_2d = assign_synthetic_positions(data['neurons'], data['connections'])
            
            # Extract positions as numpy array
            positions = []
            for neuron_id, neuron_data in neurons_2d.items():
                if 'position_2d' in neuron_data:
                    positions.append(neuron_data['position_2d'])
            
            if positions:
                module_positions = np.array(positions)
                
                # Save for future use
                np.save(positions_file, module_positions)
                print(f"Saved {len(module_positions)} module positions to {positions_file}")
                
                return module_positions
            else:
                print("No positions found in NeuroML data")
                return None
        else:
            print(f"NeuroML file not found: {neuroml_file_path}")
            return None
    except Exception as e:
        print(f"Error loading biological module positions: {e}")
        return None

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
        
        # Load the original connections to enable connectome fidelity metrics
        try:
            from neuroml_to_gvft import parse_neuroml2_file
            neuroml_file_path = os.path.join(args.neuroml_fields, f"{args.neuroml_basename}.net.nml")
            if os.path.exists(neuroml_file_path):
                data = parse_neuroml2_file(neuroml_file_path)
                if data and 'connections' in data:
                    config.source_connections = data['connections']
                    print(f"Loaded {len(data['connections'])} source connections for connectome fidelity metrics")
                    
                    # Load or compute module positions from NeuroML
                    module_positions = load_biological_module_positions(args.neuroml_fields, args.neuroml_basename)
                    if module_positions is not None:
                        config.bio_module_positions = torch.tensor(
                            module_positions, dtype=torch.float32, device=config.device)
                        config.num_modules = len(module_positions)
                        print(f"Will use {config.num_modules} biologically-positioned modules for simulation")
            else:
                print(f"NeuroML source file not found at {neuroml_file_path}")
                config.source_connections = None
        except Exception as e:
            print(f"Could not load source connections: {e}")
            config.source_connections = None
    else:
        print("\nUsing random field initialization")
        filter_scale = 200 / config.grid_size if config.use_parameter_scaling else 1.0
        Fx_init, Fy_init, W_init, eta_init = initialize_fields(
            config.grid_size, config.device, filter_scale)
        config.source_connections = None
    
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