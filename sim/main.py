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
from neuroml_integration import NeuroMLProcessor
import networkx as nx

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='GVFT Field Simulation with GPU Acceleration')
    parser.add_argument('--grid-size', type=int, default=200, help='Grid size for simulation')
    parser.add_argument('--no-scaling', action='store_true', help='Disable parameter scaling')
    parser.add_argument('--output-dir', type=str, default='../figures', help='Output directory for figures')
    parser.add_argument('--full-sims-only', action='store_true', help='Skip parameter sweep, run full simulations only')
    parser.add_argument('--sweep-only', action='store_true', help='Run parameter sweep only, skip full simulations')
    parser.add_argument('--neuroml-fields', type=str, help='Directory containing preprocessed NeuroML GVFT fields (from neuroml_to_gvft.py)')
    parser.add_argument('--neuroml-file', type=str, help='Path to NeuroML file to process directly')
    parser.add_argument('--neuroml-basename', type=str, default='PharyngealNetwork', help='Base name of the processed NeuroML files')
    parser.add_argument('--primary-metric', type=str, default='pattern_quality', 
                        choices=['hotspot_fraction', 'pattern_persistence', 'structural_complexity', 
                                'flow_coherence', 'pattern_quality'],
                        help='Primary metric to use for parameter selection')
    # New arguments for controlling output
    parser.add_argument('--save-checkpoints', action='store_true', help='Save checkpoint files during parameter sweep')
    parser.add_argument('--save-intermediates', action='store_true', help='Save intermediate visualizations during parameter sweep')
    parser.add_argument('--visualize-priors', action='store_true', help='Always visualize prior fields before simulation (default: true for NeuroML input)')
    
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

def visualize_prior_fields(Fx, Fy, W, eta, output_dir, basename="priors", community_field=None, neurons_2d=None, connections=None):
    """Visualize the initial field state before simulation."""
    print("Visualizing initial field states...")
    
    # Convert to numpy for visualization
    Fx_np = Fx.cpu().numpy()
    Fy_np = Fy.cpu().numpy()
    W_np = W.cpu().numpy()
    eta_np = eta.cpu().numpy()
    
    # Create figure with multiple panels
    n_fields = 4 if community_field is None else 5
    fig, axs = plt.subplots(2, (n_fields+1)//2, figsize=(4*((n_fields+1)//2), 8), constrained_layout=True)
    axs = axs.flatten()
    
    # Flow magnitude
    F_mag = np.sqrt(Fx_np**2 + Fy_np**2)
    axs[0].imshow(F_mag, extent=[-1, 1, -1, 1], origin='lower', cmap='Blues')
    axs[0].set_title("Flow Field Magnitude (|F|)")
    axs[0].set_xlim(-1, 1); axs[0].set_ylim(-1, 1)
    axs[0].set_xlabel('x'); axs[0].set_ylabel('y')
    
    # Synaptic strength
    im1 = axs[1].imshow(W_np, extent=[-1, 1, -1, 1], origin='lower', cmap='hot', vmin=-2, vmax=2)
    axs[1].set_title("Synaptic Strength (W)")
    axs[1].set_xlim(-1, 1); axs[1].set_ylim(-1, 1)
    axs[1].set_xlabel('x'); axs[1].set_ylabel('y')
    plt.colorbar(im1, ax=axs[1])
    
    # Neuromodulatory field
    im2 = axs[2].imshow(eta_np, extent=[-1, 1, -1, 1], origin='lower', cmap='viridis')
    axs[2].set_title("Neuromodulatory Field (η)")
    axs[2].set_xlim(-1, 1); axs[2].set_ylim(-1, 1)
    axs[2].set_xlabel('x'); axs[2].set_ylabel('y')
    plt.colorbar(im2, ax=axs[2])
    
    # Flow field (streamplot)
    x = np.linspace(-1, 1, Fx_np.shape[1])
    y = np.linspace(-1, 1, Fx_np.shape[0])
    X, Y = np.meshgrid(x, y)
    
    # Create streamplot with density based on grid size
    stream_density = min(1.0, 200 / max(Fx_np.shape))
    axs[3].streamplot(x, y, Fx_np, Fy_np, density=stream_density, color=F_mag, cmap='Blues')
    axs[3].set_title("Flow Field Direction")
    axs[3].set_xlim(-1, 1); axs[3].set_ylim(-1, 1)
    axs[3].set_xlabel('x'); axs[3].set_ylabel('y')
    
    # Add community field if available
    if community_field is not None:
        community_np = community_field.cpu().numpy()
        im3 = axs[4].imshow(community_np, extent=[-1, 1, -1, 1], origin='lower', cmap='tab10')
        axs[4].set_title("Community Field")
        axs[4].set_xlim(-1, 1); axs[4].set_ylim(-1, 1)
        axs[4].set_xlabel('x'); axs[4].set_ylabel('y')
        plt.colorbar(im3, ax=axs[4])
    
    plt.suptitle(f'Initial Field States: {basename}', fontsize=16)
    
    # Save figure
    filepath = os.path.join(output_dir, f"{basename}_initial_fields.png")
    plt.savefig(filepath, dpi=200, bbox_inches='tight')
    plt.close()
    
    # Create connectivity visualization if neuron data is available
    if neurons_2d is not None and connections is not None:
        visualize_connectivity(neurons_2d, connections, output_dir, basename)
    
    print(f"Initial field visualizations saved to {filepath}")
    
    return filepath

def visualize_connectivity(neurons_2d, connections, output_dir, basename="priors"):
    """Visualize the connectivity graph from NeuroML data."""
    print("Visualizing connectivity structure...")
    
    # Create NetworkX graph
    G = nx.DiGraph()
    
    # Add nodes with positions
    for neuron_id, neuron_data in neurons_2d.items():
        if 'position_2d' in neuron_data:
            G.add_node(neuron_id, pos=neuron_data['position_2d'], type=neuron_data['type'])
    
    # Add edges with properties
    for pre_id, post_id, weight, delay, excitatory in connections:
        if pre_id in G.nodes and post_id in G.nodes:
            G.add_edge(pre_id, post_id, weight=weight, delay=delay, excitatory=excitatory)
    
    # Get positions
    pos = nx.get_node_attributes(G, 'pos')
    
    # Create figure for connectivity
    plt.figure(figsize=(10, 10))
    
    # Color nodes by type
    node_types = nx.get_node_attributes(G, 'type')
    color_map = {'motor': 'red', 'sensory': 'green', 'interneuron': 'blue', 'default': 'gray'}
    node_colors = [color_map.get(node_types.get(node, 'default'), 'gray') for node in G.nodes()]
    
    # Draw nodes with colors by type
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=100, alpha=0.8)
    
    # Draw edges with colors by excitatory/inhibitory
    excitatory_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get('excitatory', True)]
    inhibitory_edges = [(u, v) for u, v, d in G.edges(data=True) if not d.get('excitatory', True)]
    
    # Draw excitatory edges
    nx.draw_networkx_edges(G, pos, edgelist=excitatory_edges, edge_color='green', alpha=0.6, 
                          width=[G[u][v]['weight']/5 for u, v in excitatory_edges], 
                          arrowsize=10, connectionstyle='arc3,rad=0.1')
    
    # Draw inhibitory edges
    nx.draw_networkx_edges(G, pos, edgelist=inhibitory_edges, edge_color='red', alpha=0.6, 
                          width=[G[u][v]['weight']/5 for u, v in inhibitory_edges], 
                          arrowsize=10, connectionstyle='arc3,rad=0.1')
    
    # Add labels with minimal overlap
    nx.draw_networkx_labels(G, pos, font_size=8, font_family='sans-serif', alpha=0.7)
    
    # Add legend for node types
    handles = []
    for type_name, color in color_map.items():
        if type_name in node_types.values():
            handles.append(plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, 
                                     markersize=10, label=type_name.capitalize()))
    
    # Add legend for edge types
    if excitatory_edges:
        handles.append(plt.Line2D([0], [0], color='green', lw=2, label='Excitatory'))
    if inhibitory_edges:
        handles.append(plt.Line2D([0], [0], color='red', lw=2, label='Inhibitory'))
    
    plt.legend(handles=handles, loc='upper right')
    
    # Set limits to match fields
    plt.xlim(-1.1, 1.1)
    plt.ylim(-1.1, 1.1)
    
    # Add title and adjust layout
    plt.title(f"Neural Connectivity Graph ({len(G.nodes())} neurons, {len(G.edges())} connections)")
    plt.axis('off')
    plt.tight_layout()
    
    # Save figure
    filepath = os.path.join(output_dir, f"{basename}_connectivity.png")
    plt.savefig(filepath, dpi=200, bbox_inches='tight')
    plt.close()
    
    # Create a second figure with community detection
    communities = list(nx.community.louvain_communities(G.to_undirected(), resolution=1.0))
    
    plt.figure(figsize=(10, 10))
    
    # Color nodes by community
    community_map = {}
    for idx, comm in enumerate(communities):
        for node in comm:
            community_map[node] = idx
    
    community_colors = [plt.cm.tab10(community_map.get(node, 0) % 10) for node in G.nodes()]
    
    # Draw nodes with colors by community
    nx.draw_networkx_nodes(G, pos, node_color=community_colors, node_size=100, alpha=0.8)
    
    # Draw all edges with same style
    nx.draw_networkx_edges(G, pos, alpha=0.4, width=0.5, 
                          arrowsize=8, connectionstyle='arc3,rad=0.1')
    
    # Add community borders
    for idx, community in enumerate(communities):
        # Get subgraph for this community
        community_nodes = list(community)
        if community_nodes:
            # Calculate the convex hull of community
            community_pos = np.array([pos[node] for node in community_nodes])
            if len(community_pos) > 2:  # Need at least 3 points for a hull
                try:
                    from scipy.spatial import ConvexHull
                    hull = ConvexHull(community_pos)
                    hull_points = community_pos[hull.vertices]
                    hull_points = np.append(hull_points, [hull_points[0]], axis=0)  # Close the polygon
                    plt.fill(hull_points[:, 0], hull_points[:, 1], alpha=0.2, 
                            color=plt.cm.tab10(idx % 10), zorder=0)
                except Exception as e:
                    print(f"Could not draw community hull: {e}")
    
    # Add legend for communities
    handles = []
    for idx, community in enumerate(communities[:min(10, len(communities))]):
        handles.append(plt.Line2D([0], [0], marker='o', color='w', 
                                 markerfacecolor=plt.cm.tab10(idx % 10), 
                                 markersize=10, label=f'Community {idx}'))
    
    plt.legend(handles=handles, loc='upper right')
    
    # Set limits to match fields
    plt.xlim(-1.1, 1.1)
    plt.ylim(-1.1, 1.1)
    
    # Add title and adjust layout
    plt.title(f"Neural Communities ({len(communities)} detected)")
    plt.axis('off')
    plt.tight_layout()
    
    # Save figure
    filepath = os.path.join(output_dir, f"{basename}_communities.png")
    plt.savefig(filepath, dpi=200, bbox_inches='tight')
    plt.close()
    
    print(f"Connectivity visualization saved to {output_dir}")
    
    return communities

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
    using_bio_priors = args.neuroml_fields is not None or args.neuroml_file is not None
    
    # Variables to store additional data for visualization
    neurons_2d = None
    connections = None
    community_field = None
    initial_modules = None
    initial_weights = None
    
    # Initialize fields - either from NeuroML or random
    if using_bio_priors:
        if args.neuroml_file:
            # Process NeuroML file directly
            print(f"\nProcessing NeuroML file: {args.neuroml_file}")
            processor = NeuroMLProcessor(grid_size=config.grid_size, device=config.device)
            neuroml_dir = os.path.dirname(args.neuroml_file)
            basename = os.path.basename(args.neuroml_file).split('.')[0]
            gvft_fields, data, neurons_2d = processor.process_and_generate_fields(neuroml_dir, basename)
            
            if gvft_fields:
                Fx_init = gvft_fields['flow_x']
                Fy_init = gvft_fields['flow_y']
                W_init = gvft_fields['strength']
                eta_init = gvft_fields['neuromod']
                community_field = gvft_fields.get('community')
                
                if data and 'connections' in data:
                    connections = data['connections']
                
                print("Successfully generated fields from NeuroML file")
                
                # Update config based on the actual neuron count and connectivity
                if neurons_2d and connections:
                    # Update number of modules to match neuron count
                    neuron_count = len(neurons_2d)
                    config.num_modules = neuron_count
                    
                    # Adjust top-k connections based on average connectivity
                    if connections and neuron_count > 0:
                        avg_connections_per_neuron = len(connections) / neuron_count
                        # Set top_k to be slightly higher than the average to ensure we don't lose connectivity
                        config.top_k = max(config.top_k, int(avg_connections_per_neuron * 1.5))
                        
                        # Cap top_k to a reasonable maximum to avoid excessive connections
                        config.top_k = min(config.top_k, 20)
                    
                    print(f"Updated configuration to match NeuroML complexity:")
                    print(f"  Number of modules (neurons): {config.num_modules}")
                    print(f"  Top-k connections per module: {config.top_k}")
                    
                    # Initialize module positions directly from NeuroML neuron positions
                    module_positions = []
                    neuron_ids = []
                    for neuron_id, neuron_data in neurons_2d.items():
                        if 'position_2d' in neuron_data:
                            pos = neuron_data['position_2d']
                            module_positions.append(pos)
                            neuron_ids.append(neuron_id)
                    
                    if module_positions:
                        # Convert to tensor
                        initial_modules = torch.tensor(module_positions, dtype=torch.float32, device=config.device)
                        
                        # Ensure positions are within domain bounds
                        X_min, X_max = domain['x'].min().item(), domain['x'].max().item()
                        Y_min, Y_max = domain['y'].min().item(), domain['y'].max().item()
                        
                        # Clip any out-of-bounds positions
                        initial_modules[:, 0] = torch.clamp(initial_modules[:, 0], X_min, X_max)
                        initial_modules[:, 1] = torch.clamp(initial_modules[:, 1], Y_min, Y_max)
                        
                        print(f"Initialized {len(initial_modules)} module positions from NeuroML data")
                        
                        # Create initial connectivity graph based on NeuroML connections
                        num_modules = len(initial_modules)
                        initial_weights = torch.zeros((num_modules, num_modules), device=config.device)
                        
                        # Create mapping from neuron_id to index
                        id_to_index = {neuron_id: i for i, neuron_id in enumerate(neuron_ids)}
                        
                        # Fill in the weights matrix directly from connections
                        for pre_id, post_id, weight, delay, excitatory in connections:
                            if pre_id in id_to_index and post_id in id_to_index:
                                i = id_to_index[pre_id]
                                j = id_to_index[post_id]
                                
                                # Scale the weight appropriately
                                w_ij = torch.clamp(torch.tensor(weight / config.lambda_val, device=config.device), 0, 1.0)
                                
                                # Store the connection weight
                                initial_weights[i, j] = w_ij
                
                # Always visualize fields from NeuroML
                visualize_prior_fields(Fx_init, Fy_init, W_init, eta_init, args.output_dir, 
                                       basename=basename, community_field=community_field,
                                       neurons_2d=neurons_2d, connections=connections)
            else:
                print("Falling back to random field initialization")
                filter_scale = 200 / config.grid_size if config.use_parameter_scaling else 1.0
                Fx_init, Fy_init, W_init, eta_init = initialize_fields(
                    config.grid_size, config.device, filter_scale)
                using_bio_priors = False
        else:
            # Load preprocessed fields
            print(f"\nLoading fields from preprocessed NeuroML data: {args.neuroml_fields}")
            Fx_init, Fy_init, W_init, eta_init = load_neuroml_fields(
                args.neuroml_fields, args.neuroml_basename, config.grid_size, config.device)
            print("Successfully loaded biological GVFT fields as simulation priors")
            
            # Visualize loaded fields if requested
            if args.visualize_priors:
                visualize_prior_fields(Fx_init, Fy_init, W_init, eta_init, args.output_dir, 
                                       basename=args.neuroml_basename)
    else:
        print("\nUsing random field initialization")
        filter_scale = 200 / config.grid_size if config.use_parameter_scaling else 1.0
        Fx_init, Fy_init, W_init, eta_init = initialize_fields(
            config.grid_size, config.device, filter_scale)
        
        # Visualize random fields if explicitly requested
        if args.visualize_priors:
            visualize_prior_fields(Fx_init, Fy_init, W_init, eta_init, args.output_dir, 
                                   basename="random")
    
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
                Fx_init, Fy_init, W_init, eta_init, args.output_dir,
                initial_modules=initial_modules if 'initial_modules' in locals() else None,
                initial_weights=initial_weights if 'initial_weights' in locals() else None
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