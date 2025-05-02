import os
import sys
import glob
import argparse
import numpy as np
import matplotlib.pyplot as plt
from lxml import etree
from scipy.ndimage import gaussian_filter, sobel
import networkx as nx

# ------- Top-Level Defaults (Constants) -------
DEFAULT_GRID_SIZE = 200
DEFAULT_DOMAIN_BOUNDS = [(-1, 1), (-1, 1)]
DEFAULT_SIGMA_FLOW_POTENTIAL = 15.0
DEFAULT_SIGMA_STRENGTH_DENSITY = 10.0
DEFAULT_SIGMA_ETA_DENSITY = 15.0
DEFAULT_CONNECTION_WEIGHT_SCALE = 5.0
DEFAULT_ACTIVITY_THRESHOLD = 0.05
DEFAULT_OUTPUT_FORMATS = ['npy', 'png']

# Neuron type weights for strength field
NEURON_TYPE_WEIGHTS = {
    'motor': 1.5,
    'sensory': 1.0,
    'interneuron': 0.8,
    'default': 1.0
}

# =====================================================
# Function Definitions
# =====================================================

def clean_neuron_id(raw_id, population):
    """Clean NeuroML neuron ID from ../<pop>/index/<comp> to <pop>_<index>."""
    if raw_id and '../' in raw_id:
        # Format: ../<population>/index/<component>
        parts = raw_id.split('/')
        if len(parts) >= 3:
            index = parts[-2]  # The index is the second-to-last part
            return f"{population}_{index}"
    return f"{population}_{raw_id}"

def parse_neuroml2_file(filename):
    """Parse a NeuroML2 file and extract neural populations, connections, and synaptic properties."""
    print(f"Parsing {filename}...")
    try:
        with open(filename, 'rb') as f:
            parser = etree.XMLParser(remove_comments=True, ns_clean=True, recover=True)
            tree = etree.parse(f, parser)

        root = tree.getroot()
        nsmap = {k if k is not None else 'neuroml': v for k, v in root.nsmap.items()}
        if 'neuroml' not in nsmap:
            default_uri = root.nsmap.get(None)
            if default_uri and 'neuroml.org/schema/neuroml2' in default_uri:
                nsmap['neuroml'] = default_uri
            else:
                print("Warning: Could not automatically detect NeuroML v2 namespace. Assuming standard 'neuroml' prefix.")
                nsmap['neuroml'] = 'http://www.neuroml.org/schema/neuroml2'

        # Extract neurons with types
        neurons = {}
        for population in root.xpath('//neuroml:population', namespaces=nsmap):
            pop_id = population.get('id')
            component_type = population.get('component', pop_id).lower()
            size = int(population.get('size', 1))
            neuron_type = 'motor' if 'motor' in component_type else 'sensory' if 'sensory' in component_type else 'interneuron' if 'inter' in component_type else 'default'
            for i in range(size):
                neuron_id = f"{pop_id}_{i}"
                neurons[neuron_id] = {'type': neuron_type}

        # Extract synaptic properties
        synapse_types = {}
        for synapse in root.xpath('//neuroml:expTwoSynapse', namespaces=nsmap):
            syn_id = synapse.get('id')
            tau_rise = float(synapse.get('tauRise', 0.1))
            tau_decay = float(synapse.get('tauDecay', 0.5))
            synapse_types[syn_id] = {'delay': (tau_rise + tau_decay) / 2, 'excitatory': True}
        for synapse in root.xpath('//neuroml:inhSynapse', namespaces=nsmap):
            syn_id = synapse.get('id')
            tau = float(synapse.get('tau', 0.5))
            synapse_types[syn_id] = {'delay': tau, 'excitatory': False}

        # Extract connections with synaptic properties
        connections = []
        for projection in root.xpath('//neuroml:projection', namespaces=nsmap):
            synapse_type = projection.get('synapse')
            presynaptic_pop = projection.get('presynapticPopulation')
            postsynaptic_pop = projection.get('postsynapticPopulation')
            if not presynaptic_pop or not postsynaptic_pop:
                continue
            for conn in projection.xpath('.//neuroml:connection', namespaces=nsmap):
                pre_id_raw = conn.get('preCellId')
                post_id_raw = conn.get('postCellId')
                pre_cell_id = clean_neuron_id(pre_id_raw, presynaptic_pop) if pre_id_raw else f"{presynaptic_pop}_0"
                post_cell_id = clean_neuron_id(post_id_raw, postsynaptic_pop) if post_id_raw else f"{postsynaptic_pop}_0"
                weight = float(conn.get('weight', 1.0)) * DEFAULT_CONNECTION_WEIGHT_SCALE
                syn_props = synapse_types.get(synapse_type, {'delay': 0.5, 'excitatory': True})
                connections.append((pre_cell_id, post_cell_id, weight, syn_props['delay'], syn_props['excitatory']))

        for electrical in root.xpath('//neuroml:electricalProjection', namespaces=nsmap):
            presynaptic_pop = electrical.get('presynapticPopulation')
            postsynaptic_pop = electrical.get('postsynapticPopulation')
            if not presynaptic_pop or not postsynaptic_pop:
                continue
            for conn in electrical.xpath('.//neuroml:electricalConnection', namespaces=nsmap):
                pre_id_raw = conn.get('preCell') or conn.get('preCellId')
                post_id_raw = conn.get('postCell') or conn.get('postCellId')
                pre_cell_id = clean_neuron_id(pre_id_raw, presynaptic_pop) if pre_id_raw else f"{presynaptic_pop}_0"
                post_cell_id = clean_neuron_id(post_id_raw, postsynaptic_pop) if post_id_raw else f"{postsynaptic_pop}_0"
                conductance = float(conn.get('weight', 1.0)) * DEFAULT_CONNECTION_WEIGHT_SCALE
                connections.append((pre_cell_id, post_cell_id, conductance, 0.1, True))

        print(f"Successfully parsed {len(neurons)} neurons and {len(connections)} connections.")
        return {'neurons': neurons, 'connections': connections}
    except FileNotFoundError:
        print(f"Error: File not found at {filename}")
        return {'neurons': {}, 'connections': []}
    except etree.XMLSyntaxError as e:
        print(f"Error parsing XML in {filename}: {e}")
        return {'neurons': {}, 'connections': []}
    except Exception as e:
        print(f"An unexpected error occurred while parsing {filename}: {e}")
        return {'neurons': {}, 'connections': []}

def assign_synthetic_positions(neurons):
    """Assign synthetic 2D positions to neurons in a grid layout."""
    if not neurons:
        print("Warning: No neurons to assign positions to.")
        return {}
    
    neuron_ids = list(neurons.keys())
    num_neurons = len(neuron_ids)
    neurons_2d = {}
    
    grid_dim = int(np.ceil(np.sqrt(num_neurons)))
    xs = np.linspace(DEFAULT_DOMAIN_BOUNDS[0][0] * 0.8, DEFAULT_DOMAIN_BOUNDS[0][1] * 0.8, grid_dim)
    ys = np.linspace(DEFAULT_DOMAIN_BOUNDS[1][0] * 0.8, DEFAULT_DOMAIN_BOUNDS[1][1] * 0.8, grid_dim)
    coords = [(x, y) for x in xs for y in ys]
    
    for i, neuron_id in enumerate(neuron_ids):
        neuron_data = neurons[neuron_id].copy()
        neuron_data['position_2d'] = coords[i % len(coords)]
        neurons_2d[neuron_id] = neuron_data
    
    return neurons_2d

def simulate_activity(G, num_steps=100):
    """Simulate simple firing rate activity to estimate correlations."""
    firing_rates = {node: 0.1 for node in G.nodes()}
    print("Nodes in graph:", list(G.nodes()))  # Debugging
    for _ in range(num_steps):
        new_rates = firing_rates.copy()
        for node in G.nodes():
            neighbors = list(G.predecessors(node))  # Use predecessors for DiGraph
            if neighbors:
                print(f"Neighbors of {node}: {neighbors}")  # Debugging
                incoming = sum(firing_rates[n] * G[n][node]['weight'] for n in neighbors)
                new_rates[node] = max(0, min(1, 0.1 * incoming + 0.9 * firing_rates[node]))
        firing_rates = new_rates
    return firing_rates

def place_gaussian(field, center_ix, center_iy, sigma, weight=1.0):
    """Places a Gaussian kernel onto the field."""
    grid_size = field.shape[0]
    if not (0 <= center_ix < grid_size and 0 <= center_iy < grid_size):
        return field
    x = np.arange(grid_size)
    y = np.arange(grid_size)
    X, Y = np.meshgrid(x, y)
    dist_sq = (X - center_ix)**2 + (Y - center_iy)**2
    sigma_sq = 2 * sigma**2
    if sigma_sq > 1e-9:
        gaussian = weight * np.exp(-dist_sq / sigma_sq)
        field += gaussian
    else:
        field[center_iy, center_ix] += weight
    return field

def generate_multi_scale_fields(neurons_2d, connections, config):
    """Generate GVFT fields using multi-scale feature extraction with dynamics."""
    grid_size = config['grid_size']
    domain_bounds = config['domain_bounds']
    sigma_flow_potential = config['sigma_flow_potential']
    sigma_strength_density = config['sigma_strength_density']
    sigma_eta_density = config['sigma_eta_density']
    activity_threshold = config['activity_threshold']

    if not neurons_2d:
        print("Warning: No 2D neuron positions provided. Cannot generate fields.")
        return None

    # Build graph for analysis
    G = nx.DiGraph()
    for pre_id, post_id, weight, delay, excitatory in connections:
        G.add_edge(pre_id, post_id, weight=weight, delay=delay, excitatory=excitatory)

    # Calculate centrality measures
    degree_centrality = nx.degree_centrality(G)
    betweenness_centrality = nx.betweenness_centrality(G)

    # Perform community detection
    communities = list(nx.community.louvain_communities(G.to_undirected(), resolution=1.0))
    community_map = {}
    for idx, comm in enumerate(communities):
        for neuron_id in comm:
            community_map[neuron_id] = idx

    # Simulate activity
    firing_rates = simulate_activity(G)

    # Map neurons to grid indices
    neuron_indices = {}
    for neuron_id, neuron_data in neurons_2d.items():
        if 'position_2d' in neuron_data:
            pos_x, pos_y = neuron_data['position_2d']
            pos_x = np.clip(pos_x, domain_bounds[0][0], domain_bounds[0][1])
            pos_y = np.clip(pos_y, domain_bounds[1][0], domain_bounds[1][1])
            i_x = min(grid_size - 1, int(np.floor((pos_x - domain_bounds[0][0]) / (domain_bounds[0][1] - domain_bounds[0][0]) * grid_size)))
            i_y = min(grid_size - 1, int(np.floor((pos_y - domain_bounds[1][0]) / (domain_bounds[1][1] - domain_bounds[1][0]) * grid_size)))
            i_x = max(0, i_x)
            i_y = max(0, i_y)
            neuron_indices[neuron_id] = (i_x, i_y)

    # Initialize fields
    flow_potential = np.zeros((grid_size, grid_size))
    strength_density = np.zeros((grid_size, grid_size))
    eta_density = np.zeros((grid_size, grid_size))
    community_field = np.zeros((grid_size, grid_size))

    # Strength field with centrality and neuron type weighting
    for neuron_id, (i_x, i_y) in neuron_indices.items():
        centrality_score = (degree_centrality.get(neuron_id, 0) + betweenness_centrality.get(neuron_id, 0)) / 2
        type_weight = NEURON_TYPE_WEIGHTS.get(neurons_2d[neuron_id]['type'], 1.0)
        strength_density = place_gaussian(strength_density, i_x, i_y, sigma_strength_density, weight=centrality_score * type_weight * 2)

    # Neuromodulatory field with activity modulation
    connection_count = 0
    for pre_id, post_id, weight, delay, excitatory in connections:
        if pre_id in neuron_indices and post_id in neuron_indices:
            pre_ix, pre_iy = neuron_indices[pre_id]
            post_ix, post_iy = neuron_indices[post_id]
            mid_ix = int((pre_ix + post_ix) / 2)
            mid_iy = int((pre_iy + post_iy) / 2)
            activity_weight = (firing_rates.get(pre_id, 0.1) + firing_rates.get(post_id, 0.1)) / 2
            eta_density = place_gaussian(eta_density, mid_ix, mid_iy, sigma_eta_density / (1 + delay), weight=abs(weight) * activity_weight)
            connection_count += 1
    print(f"Processed {connection_count} connections.")

    # Community field
    for neuron_id, (i_x, i_y) in neuron_indices.items():
        if neuron_id in community_map:
            community_field = place_gaussian(community_field, i_x, i_y, sigma_strength_density, weight=community_map[neuron_id] + 1)

    # Flow field between community centroids with synaptic directionality
    community_centroids = {}
    for comm_idx, community in enumerate(communities):
        x_coords = [neuron_indices[nid][0] for nid in community if nid in neuron_indices]
        y_coords = [neuron_indices[nid][1] for nid in community if nid in neuron_indices]
        if x_coords and y_coords:
            centroid_x = np.mean(x_coords)
            centroid_y = np.mean(y_coords)
            community_centroids[comm_idx] = (centroid_x, centroid_y)

    for i in range(len(community_centroids)):
        for j in range(i + 1, len(community_centroids)):
            if i in community_centroids and j in community_centroids:
                start_x, start_y = community_centroids[i]
                end_x, end_y = community_centroids[j]
                dx = (end_x - start_x) / max(1, abs(end_x - start_x)) if start_x != end_x else 0
                dy = (end_y - start_y) / max(1, abs(end_y - start_y)) if start_y != end_y else 0
                steps = int(max(abs(end_x - start_x), abs(end_y - start_y), 5))
                net_flow = 0
                for pre_id in communities[i]:
                    for post_id in communities[j]:
                        if G.has_edge(pre_id, post_id):
                            edge_data = G[pre_id][post_id]
                            flow_contrib = edge_data['weight'] * (1 if edge_data['excitatory'] else -1) / (1 + edge_data['delay'])
                            net_flow += flow_contrib
                for step in range(steps + 1):
                    interp_x = int(start_x + step * dx)
                    interp_y = int(start_y + step * dy)
                    if 0 <= interp_x < grid_size and 0 <= interp_y < grid_size:
                        flow_potential[interp_y, interp_x] += net_flow * (1 if step < steps / 2 else -1)

    print("Calculating Flow Field (F) = -Gradient(Potential)...")
    flow_potential_blurred = gaussian_filter(flow_potential, sigma=sigma_flow_potential)
    Fy = -sobel(flow_potential_blurred, axis=0, mode='reflect')
    Fx = -sobel(flow_potential_blurred, axis=1, mode='reflect')

    flow_mag = np.sqrt(Fx**2 + Fy**2)
    max_flow_mag = np.max(flow_mag)
    if max_flow_mag > 1e-9:
        Fx /= max_flow_mag
        Fy /= max_flow_mag
        print(f"Max flow magnitude after gradient: {max_flow_mag:.4f}")
    else:
        print("Warning: Max flow magnitude after gradient is near zero.")
        flow_potential_blurred += np.random.normal(0, 0.1, flow_potential_blurred.shape)
        Fy = -sobel(flow_potential_blurred, axis=0, mode='reflect')
        Fx = -sobel(flow_potential_blurred, axis=1, mode='reflect')
        flow_mag = np.sqrt(Fx**2 + Fy**2)
        max_flow_mag = np.max(flow_mag)
        if max_flow_mag > 1e-9:
            Fx /= max_flow_mag
            Fy /= max_flow_mag
            print(f"Max flow magnitude after noise addition: {max_flow_mag:.4f}")

    print("Finalizing Strength Field (W) from density...")
    W_final = gaussian_filter(strength_density, sigma=1.0)
    max_abs_w = np.max(np.abs(W_final))
    if max_abs_w > 1e-9:
        W_final = np.clip((W_final / max_abs_w) * 2.0, -2.0, 2.0)

    print("Finalizing Neuromodulatory Field (eta)...")
    eta_density_blurred = gaussian_filter(eta_density, sigma=sigma_eta_density)
    eta_final = np.where(eta_density_blurred > activity_threshold, eta_density_blurred, 0)
    max_eta = np.max(eta_final)
    if max_eta > 1e-9:
        eta_final /= max_eta

    print("Finalizing Community Field...")
    community_field_blurred = gaussian_filter(community_field, sigma=sigma_strength_density)
    max_comm = np.max(community_field_blurred)
    if max_comm > 1e-9:
        community_field_blurred /= max_comm

    gvft_fields = {
        'flow_x': Fx,
        'flow_y': Fy,
        'strength': W_final,
        'neuromod': eta_final,
        'community': community_field_blurred
    }

    print("GVFT field generation complete (Multi-Scale method with dynamics).")
    return gvft_fields

def visualize_fields(gvft_fields, neurons_2d, connections, output_path, config):
    """Visualize the generated GVFT fields."""
    if gvft_fields is None:
        print("Cannot visualize: GVFT fields are None.")
        return

    grid_size = config['grid_size']
    domain_bounds = config['domain_bounds']

    fig, axs = plt.subplots(2, 3, figsize=(20, 15))
    x = np.linspace(domain_bounds[0][0], domain_bounds[0][1], grid_size)
    y = np.linspace(domain_bounds[1][0], domain_bounds[1][1], grid_size)
    X, Y = np.meshgrid(x, y)

    valid_neuron_coords = []
    if neurons_2d:
        for neuron_id, neuron_data in neurons_2d.items():
            if 'position_2d' in neuron_data:
                valid_neuron_coords.append(neuron_data['position_2d'])
    valid_neuron_coords = np.array(valid_neuron_coords)

    flow_mag = np.sqrt(gvft_fields['flow_x']**2 + gvft_fields['flow_y']**2)
    vmax_flow = np.max(flow_mag) if np.max(flow_mag) > 1e-9 else 1.0
    im0 = axs[0, 0].imshow(flow_mag, extent=[domain_bounds[0][0], domain_bounds[0][1], domain_bounds[1][0], domain_bounds[1][1]], origin='lower', cmap='Blues', vmin=0, vmax=vmax_flow)
    axs[0, 0].set_title("Flow Field Magnitude (|F|)")
    fig.colorbar(im0, ax=axs[0, 0], fraction=0.046, pad=0.04)

    im1 = axs[0, 1].imshow(gvft_fields['strength'], extent=[domain_bounds[0][0], domain_bounds[0][1], domain_bounds[1][0], domain_bounds[1][1]], origin='lower', cmap='hot', vmin=-2, vmax=2)
    axs[0, 1].set_title("Synaptic Strength (W)")
    fig.colorbar(im1, ax=axs[0, 1], fraction=0.046, pad=0.04)

    im2 = axs[0, 2].imshow(gvft_fields['neuromod'], extent=[domain_bounds[0][0], domain_bounds[0][1], domain_bounds[1][0], domain_bounds[1][1]], origin='lower', cmap='viridis', vmin=0, vmax=1)
    axs[0, 2].set_title("Neuromodulatory Field (eta)")
    fig.colorbar(im2, ax=axs[0, 2], fraction=0.046, pad=0.04)

    im3 = axs[1, 0].imshow(gvft_fields['community'], extent=[domain_bounds[0][0], domain_bounds[0][1], domain_bounds[1][0], domain_bounds[1][1]], origin='lower', cmap='tab20', vmin=0, vmax=len(set(community_map.values())) + 1 if 'community_map' in locals() else 1)
    axs[1, 0].set_title("Community Field")
    fig.colorbar(im3, ax=axs[1, 0], fraction=0.046, pad=0.04)

    axs[1, 1].imshow(gvft_fields['strength'], extent=[domain_bounds[0][0], domain_bounds[0][1], domain_bounds[1][0], domain_bounds[1][1]], origin='lower', cmap='hot', alpha=0.5, vmin=-2, vmax=2)
    max_flow_vis = np.max(flow_mag)
    if max_flow_vis > 1e-6:
        density = max(0.8, 20 / np.sqrt(grid_size))
        try:
            axs[1, 1].streamplot(X, Y, gvft_fields['flow_x'], gvft_fields['flow_y'],
                                color='lightgray', density=density, linewidth=0.7, arrowstyle='->', broken_streamlines=False)
        except ValueError as e:
            print(f"Warning: Streamplot failed. Error: {e}")
            step = max(1, grid_size // 20)
            axs[1, 1].quiver(X[::step, ::step], Y[::step, ::step],
                            gvft_fields['flow_x'][::step, ::step], gvft_fields['flow_y'][::step, ::step],
                            color='lightgray', scale=max(1, max_flow_vis * 25), width=0.003)
    else:
        print("Skipping flow visualization in overlay plot due to near-zero magnitude.")

    if valid_neuron_coords.size > 0:
        axs[1, 1].scatter(valid_neuron_coords[:, 0], valid_neuron_coords[:, 1], c='cyan', s=25, alpha=0.8, edgecolors='black', linewidth=0.5, zorder=3)

    connection_count_vis = 0
    if connections and neurons_2d:
        for pre_id, post_id, weight, delay, excitatory in connections:
            if pre_id in neurons_2d and post_id in neurons_2d and 'position_2d' in neurons_2d[pre_id] and 'position_2d' in neurons_2d[post_id]:
                pre_x, pre_y = neurons_2d[pre_id]['position_2d']
                post_x, post_y = neurons_2d[post_id]['position_2d']
                color = 'green' if excitatory else 'red'
                scaled_lw = max(0.5, weight * 0.8) if weight is not None else 0.5
                axs[1, 1].plot([pre_x, post_x], [pre_y, post_y], color=color, linewidth=scaled_lw, alpha=0.4, zorder=2)
                connection_count_vis += 1

    axs[1, 1].set_title(f"Connectivity ({connection_count_vis}) & Fields")
    axs[1, 1].set_xlim(domain_bounds[0])
    axs[1, 1].set_ylim(domain_bounds[1])
    axs[1, 1].set_aspect('equal', adjustable='box')

    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    plt.suptitle(f"{os.path.basename(output_path).replace('_fields.png','')}\nGrid: {grid_size}x{grid_size}", fontsize=14)
    try:
        plt.savefig(output_path, dpi=150)
        print(f"Visualization saved to {output_path}")
    except Exception as e:
        print(f"Error saving visualization to {output_path}: {e}")
    plt.close()

def main():
    """Main function to run the script."""
    parser = argparse.ArgumentParser(description='Generate GVFT fields from NeuroML2 files (Multi-Scale Method with Dynamics)')
    parser.add_argument('input_dir', help='Directory containing NeuroML2 files')
    parser.add_argument('output_dir', help='Directory to save output fields')
    parser.add_argument('--grid-size', type=int, default=None, help=f'Size of the grid (default: {DEFAULT_GRID_SIZE})')
    parser.add_argument('--sigma-flow-potential', type=float, default=None, help=f'Sigma for flow potential field blurring/placement (default: {DEFAULT_SIGMA_FLOW_POTENTIAL})')
    parser.add_argument('--sigma-strength-density', type=float, default=None, help=f'Sigma for neuron density Gaussian kernels (W) (default: {DEFAULT_SIGMA_STRENGTH_DENSITY})')
    parser.add_argument('--sigma-eta-density', type=float, default=None, help=f'Sigma for connection density blurring (eta) (default: {DEFAULT_SIGMA_ETA_DENSITY})')
    parser.add_argument('--activity-threshold', type=float, default=None, help=f'Threshold for neuromodulatory field activity source (default: {DEFAULT_ACTIVITY_THRESHOLD})')
    parser.add_argument('--no-vis', action='store_true', help='Disable visualization')
    args = parser.parse_args()

    config = {
        'grid_size': args.grid_size if args.grid_size is not None else DEFAULT_GRID_SIZE,
        'domain_bounds': DEFAULT_DOMAIN_BOUNDS,
        'sigma_flow_potential': args.sigma_flow_potential if args.sigma_flow_potential is not None else DEFAULT_SIGMA_FLOW_POTENTIAL,
        'sigma_strength_density': args.sigma_strength_density if args.sigma_strength_density is not None else DEFAULT_SIGMA_STRENGTH_DENSITY,
        'sigma_eta_density': args.sigma_eta_density if args.sigma_eta_density is not None else DEFAULT_SIGMA_ETA_DENSITY,
        'activity_threshold': args.activity_threshold if args.activity_threshold is not None else DEFAULT_ACTIVITY_THRESHOLD,
        'visualize': not args.no_vis,
        'output_formats': DEFAULT_OUTPUT_FORMATS,
        'connection_weight_scale': DEFAULT_CONNECTION_WEIGHT_SCALE
    }

    input_dir, output_dir = args.input_dir, args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    neuroml_file_path = None
    possible_names = ["PharyngealNetwork.net.nml.xml", "PharyngealNetwork.net.nml"]
    for name in possible_names:
        test_path = os.path.join(input_dir, name)
        if os.path.exists(test_path):
            neuroml_file_path = test_path
            break
    
    if not neuroml_file_path:
        print("Specific files not found, searching for *PharyngealNetwork*.*")
        found_files = glob.glob(os.path.join(input_dir, "*PharyngealNetwork*.*"))
        if found_files:
            neuroml_file_path = found_files[0]
            print(f"Using fallback match: {os.path.basename(neuroml_file_path)}")
        else:
            print(f"Error: Target file containing 'PharyngealNetwork' not found in {input_dir}")
            all_files = glob.glob(os.path.join(input_dir, "*"))
            print("Files found in directory:", [os.path.basename(f) for f in all_files] if all_files else "None")
            return

    print(f"Processing file: {neuroml_file_path}")
    filename = os.path.basename(neuroml_file_path)
    basename = "".join(c if c.isalnum() else "_" for c in os.path.splitext(filename)[0])

    data = parse_neuroml2_file(neuroml_file_path)

    if not data or not data['neurons']:
        print(f"No neurons found or error parsing {filename}, skipping...")
        return

    neurons_2d = assign_synthetic_positions(data['neurons'])
    if not neurons_2d:
        print(f"Position assignment failed for {filename}, skipping...")
        return

    gvft_fields = generate_multi_scale_fields(neurons_2d, data['connections'], config)

    if gvft_fields is None:
        print(f"Field generation failed for {filename}, skipping...")
        return

    output_prefix = os.path.join(output_dir, basename)
    print(f"Saving fields with prefix: {output_prefix}")
    saved_files = []
    try:
        if 'npy' in config['output_formats']:
            np.save(f"{output_prefix}_flow_x.npy", gvft_fields['flow_x'])
            np.save(f"{output_prefix}_flow_y.npy", gvft_fields['flow_y'])
            np.save(f"{output_prefix}_strength.npy", gvft_fields['strength'])
            np.save(f"{output_prefix}_neuromod.npy", gvft_fields['neuromod'])
            np.save(f"{output_prefix}_community.npy", gvft_fields['community'])
            saved_files.extend([f"{basename}_flow_x.npy", f"{basename}_flow_y.npy", f"{basename}_strength.npy", f"{basename}_neuromod.npy", f"{basename}_community.npy"])

        if config['visualize'] and 'png' in config['output_formats']:
            vis_path = f"{output_prefix}_fields.png"
            visualize_fields(gvft_fields, neurons_2d, data['connections'], vis_path, config)
            saved_files.append(f"{basename}_fields.png")

        print(f"Successfully saved outputs for {filename}: {', '.join(saved_files)}")

    except NameError as e:
        if 'visualize_fields' in str(e):
            print("Error: visualize_fields function not found. Make sure it's defined in the script.")
        else:
            print(f"A NameError occurred: {e}")
    except Exception as e:
        print(f"Error saving output fields or visualization for {filename}: {e}")

    print("\nScript finished.")

if __name__ == "__main__":
    main()