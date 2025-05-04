import torch
import os
import numpy as np
import networkx as nx
from lxml import etree
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter, sobel
import glob

# --- NeuroML to GVFT Integration ---

class NeuroMLProcessor:
    """Process NeuroML2 files and extract GVFT fields."""
    
    def __init__(self, grid_size=200, device=torch.device('cpu')):
        """Initialize the processor with configuration."""
        self.grid_size = grid_size
        self.device = device
        self.domain_bounds = [(-1, 1), (-1, 1)]
        self.sigma_flow_potential = 15.0
        self.sigma_strength_density = 10.0
        self.sigma_eta_density = 15.0
        self.activity_threshold = 0.05
        self.neuron_type_weights = {
            'motor': 1.5,
            'sensory': 1.0,
            'interneuron': 0.8,
            'default': 1.0
        }
        
    def process_neuroml_file(self, filename):
        """Process a NeuroML2 file and extract neural populations, connections, and properties."""
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
                    pre_cell_id = self._clean_neuron_id(pre_id_raw, presynaptic_pop) if pre_id_raw else f"{presynaptic_pop}_0"
                    post_cell_id = self._clean_neuron_id(post_id_raw, postsynaptic_pop) if post_id_raw else f"{postsynaptic_pop}_0"
                    weight = float(conn.get('weight', 1.0)) * 5.0  # default weight scale
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
                    pre_cell_id = self._clean_neuron_id(pre_id_raw, presynaptic_pop) if pre_id_raw else f"{presynaptic_pop}_0"
                    post_cell_id = self._clean_neuron_id(post_id_raw, postsynaptic_pop) if post_id_raw else f"{postsynaptic_pop}_0"
                    conductance = float(conn.get('weight', 1.0)) * 5.0
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
    
    def _clean_neuron_id(self, raw_id, population):
        """Clean NeuroML neuron ID from ../<pop>/index/<comp> to <pop>_<index>."""
        if raw_id and '../' in raw_id:
            # Format: ../<population>/index/<component>
            parts = raw_id.split('/')
            if len(parts) >= 3:
                index = parts[-2]  # The index is the second-to-last part
                return f"{population}_{index}"
        return f"{population}_{raw_id}"
    
    def assign_synthetic_positions(self, neurons, connections):
        """Assign synthetic 2D positions to neurons using force-directed layout."""
        if not neurons:
            print("Warning: No neurons to assign positions to.")
            return {}
        
        # Create a graph for layout calculation
        G = nx.Graph()
        for neuron_id in neurons:
            G.add_node(neuron_id)
            
        # Add edges from connections
        for pre_id, post_id, weight, delay, excitatory in connections:
            if pre_id in neurons and post_id in neurons:
                G.add_edge(pre_id, post_id, weight=weight)
        
        # Use force-directed layout to position nodes
        try:
            # Calculate positions - try different layouts
            if len(neurons) < 50:
                # Smaller networks: use more precise layout
                pos = nx.spring_layout(G, k=0.15, iterations=100, seed=42)
            else:
                # Larger networks: faster algorithm
                pos = nx.kamada_kawai_layout(G)
                
            # Scale positions to fit domain bounds
            x_vals = [p[0] for p in pos.values()]
            y_vals = [p[1] for p in pos.values()]
            
            if x_vals and y_vals:
                min_x, max_x = min(x_vals), max(x_vals)
                min_y, max_y = min(y_vals), max(y_vals)
                
                # Avoid division by zero
                x_range = max_x - min_x if max_x > min_x else 1.0
                y_range = max_y - min_y if max_y > min_y else 1.0
                
                # Scale positions to fit domain
                domain_width = self.domain_bounds[0][1] - self.domain_bounds[0][0]
                domain_height = self.domain_bounds[1][1] - self.domain_bounds[1][0]
                
                # Leave some margin
                margin = 0.1
                scaled_pos = {}
                for node, p in pos.items():
                    scaled_x = self.domain_bounds[0][0] + margin * domain_width + \
                               ((p[0] - min_x) / x_range) * domain_width * (1 - 2 * margin)
                    scaled_y = self.domain_bounds[1][0] + margin * domain_height + \
                               ((p[1] - min_y) / y_range) * domain_height * (1 - 2 * margin)
                    scaled_pos[node] = (scaled_x, scaled_y)
                
                # Update neuron data with positions
                neurons_2d = {}
                for neuron_id, neuron_data in neurons.items():
                    neurons_2d[neuron_id] = neuron_data.copy()
                    if neuron_id in scaled_pos:
                        neurons_2d[neuron_id]['position_2d'] = scaled_pos[neuron_id]
                    else:
                        # Fallback for any missing nodes
                        neurons_2d[neuron_id]['position_2d'] = (0.0, 0.0)
                
                return neurons_2d
            else:
                print("Warning: Could not calculate layout - no valid positions.")
                return self._use_grid_fallback(neurons)
                
        except Exception as e:
            print(f"Error in force-directed layout: {e}")
            print("Falling back to grid layout")
            return self._use_grid_fallback(neurons)
    
    def _use_grid_fallback(self, neurons):
        """Fallback to grid layout if force-directed layout fails."""
        neuron_ids = list(neurons.keys())
        num_neurons = len(neuron_ids)
        neurons_2d = {}
        
        grid_dim = int(np.ceil(np.sqrt(num_neurons)))
        xs = np.linspace(self.domain_bounds[0][0] * 0.8, self.domain_bounds[0][1] * 0.8, grid_dim)
        ys = np.linspace(self.domain_bounds[1][0] * 0.8, self.domain_bounds[1][1] * 0.8, grid_dim)
        coords = [(x, y) for x in xs for y in ys]
        
        for i, neuron_id in enumerate(neuron_ids):
            neuron_data = neurons[neuron_id].copy()
            neuron_data['position_2d'] = coords[i % len(coords)]
            neurons_2d[neuron_id] = neuron_data
        
        return neurons_2d

    def simulate_activity(self, G):
        """Simulate simple firing rate activity to estimate correlations."""
        firing_rates = {node: 0.1 for node in G.nodes()}
        num_steps = 100
        
        for _ in range(num_steps):
            new_rates = firing_rates.copy()
            for node in G.nodes():
                neighbors = list(G.predecessors(node))  # Use predecessors for DiGraph
                if neighbors:
                    incoming = sum(firing_rates[n] * G[n][node]['weight'] for n in neighbors)
                    new_rates[node] = max(0, min(1, 0.1 * incoming + 0.9 * firing_rates[node]))
            firing_rates = new_rates
        
        return firing_rates

    def place_gaussian(self, field, center_ix, center_iy, sigma, weight=1.0):
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

    def generate_fields(self, neurons_2d, connections):
        """Generate GVFT fields from neuron positions and connectivity."""
        grid_size = self.grid_size
        
        if not neurons_2d:
            print("Warning: No 2D neuron positions provided. Cannot generate fields.")
            return None

        # Build graph for analysis
        G = nx.DiGraph()
        for pre_id, post_id, weight, delay, excitatory in connections:
            weight_value = weight
            if not excitatory:
                weight_value = -weight  # Use negative weights for inhibitory connections
            G.add_edge(pre_id, post_id, weight=weight_value, delay=delay, excitatory=excitatory)

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
        firing_rates = self.simulate_activity(G)

        # Map neurons to grid indices
        neuron_indices = {}
        for neuron_id, neuron_data in neurons_2d.items():
            if 'position_2d' in neuron_data:
                pos_x, pos_y = neuron_data['position_2d']
                pos_x = np.clip(pos_x, self.domain_bounds[0][0], self.domain_bounds[0][1])
                pos_y = np.clip(pos_y, self.domain_bounds[1][0], self.domain_bounds[1][1])
                i_x = min(grid_size - 1, int(np.floor((pos_x - self.domain_bounds[0][0]) / (self.domain_bounds[0][1] - self.domain_bounds[0][0]) * grid_size)))
                i_y = min(grid_size - 1, int(np.floor((pos_y - self.domain_bounds[1][0]) / (self.domain_bounds[1][1] - self.domain_bounds[1][0]) * grid_size)))
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
            type_weight = self.neuron_type_weights.get(neurons_2d[neuron_id]['type'], 1.0)
            strength_density = self.place_gaussian(strength_density, i_x, i_y, self.sigma_strength_density, weight=centrality_score * type_weight * 2)

        # Neuromodulatory field with activity modulation
        connection_count = 0
        for pre_id, post_id, weight, delay, excitatory in connections:
            if pre_id in neuron_indices and post_id in neuron_indices:
                pre_ix, pre_iy = neuron_indices[pre_id]
                post_ix, post_iy = neuron_indices[post_id]
                mid_ix = int((pre_ix + post_ix) / 2)
                mid_iy = int((pre_iy + post_iy) / 2)
                activity_weight = (firing_rates.get(pre_id, 0.1) + firing_rates.get(post_id, 0.1)) / 2
                eta_density = self.place_gaussian(eta_density, mid_ix, mid_iy, self.sigma_eta_density / (1 + delay), weight=abs(weight) * activity_weight)
                connection_count += 1

        # Community field
        for neuron_id, (i_x, i_y) in neuron_indices.items():
            if neuron_id in community_map:
                community_field = self.place_gaussian(community_field, i_x, i_y, self.sigma_strength_density, weight=community_map[neuron_id] + 1)

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
            for j in range(len(community_centroids)):
                if i != j and i in community_centroids and j in community_centroids:
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

        # For direct connections between neurons
        for pre_id, post_id, weight, delay, excitatory in connections:
            if pre_id in neuron_indices and post_id in neuron_indices:
                pre_ix, pre_iy = neuron_indices[pre_id]
                post_ix, post_iy = neuron_indices[post_id]
                
                # Skip if same position
                if pre_ix == post_ix and pre_iy == post_iy:
                    continue
                    
                dx = (post_ix - pre_ix) / max(1, abs(post_ix - pre_ix)) if post_ix != pre_ix else 0
                dy = (post_iy - pre_iy) / max(1, abs(post_iy - pre_iy)) if post_iy != pre_iy else 0
                steps = int(max(abs(post_ix - pre_ix), abs(post_iy - pre_iy), 3))
                
                flow_contrib = weight * (1 if excitatory else -1) / (1 + delay)
                
                for step in range(steps + 1):
                    interp_x = int(pre_ix + step * dx)
                    interp_y = int(pre_iy + step * dy)
                    if 0 <= interp_x < grid_size and 0 <= interp_y < grid_size:
                        flow_potential[interp_y, interp_x] += flow_contrib * (1 if step < steps / 2 else -1) * 0.5

        print("Calculating Flow Field (F) = -Gradient(Potential)...")
        flow_potential_blurred = gaussian_filter(flow_potential, sigma=self.sigma_flow_potential)
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
        eta_density_blurred = gaussian_filter(eta_density, sigma=self.sigma_eta_density)
        eta_final = np.where(eta_density_blurred > self.activity_threshold, eta_density_blurred, 0)
        max_eta = np.max(eta_final)
        if max_eta > 1e-9:
            eta_final /= max_eta

        print("Finalizing Community Field...")
        community_field_blurred = gaussian_filter(community_field, sigma=self.sigma_strength_density)
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
    
    def _sobel(self, arr, axis=0, mode='reflect'):
        """Apply Sobel operator along specified axis."""
        if axis == 0:
            kernel = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
        else:
            kernel = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
            
        from scipy.ndimage import convolve
        return convolve(arr, kernel, mode=mode)
    
    def load_neuroml_file(self, input_dir, basename=None):
        """Load a NeuroML file from the given directory."""
        if basename:
            # Try common filename patterns
            possible_names = [f"{basename}.net.nml.xml", f"{basename}.net.nml"]
            for name in possible_names:
                test_path = os.path.join(input_dir, name)
                if os.path.exists(test_path):
                    return test_path
                    
            # Search for files containing the basename
            found_files = glob.glob(os.path.join(input_dir, f"*{basename}*.*"))
            if found_files:
                return found_files[0]
        
        # Generic search for any NeuroML files
        for ext in ['.nml', '.xml']:
            found_files = glob.glob(os.path.join(input_dir, f"*{ext}"))
            if found_files:
                return found_files[0]
        
        return None
    
    def process_and_generate_fields(self, input_dir, basename=None):
        """Process a NeuroML file and generate GVFT fields."""
        # Find and load NeuroML file
        neuroml_file_path = self.load_neuroml_file(input_dir, basename)
        if not neuroml_file_path:
            print(f"Error: No NeuroML files found in {input_dir}")
            return None
            
        print(f"Processing file: {neuroml_file_path}")
        
        # Process NeuroML file
        data = self.process_neuroml_file(neuroml_file_path)
        if not data or not data['neurons']:
            print(f"No neurons found or error parsing {neuroml_file_path}, skipping...")
            return None
            
        # Assign positions to neurons using force-directed layout
        neurons_2d = self.assign_synthetic_positions(data['neurons'], data['connections'])
        if not neurons_2d:
            print(f"Position assignment failed for {neuroml_file_path}, skipping...")
            return None
            
        # Generate fields
        gvft_fields = self.generate_fields(neurons_2d, data['connections'])
        if gvft_fields is None:
            print(f"Field generation failed for {neuroml_file_path}, skipping...")
            return None
            
        # Convert numpy arrays to PyTorch tensors
        for key in gvft_fields:
            gvft_fields[key] = torch.tensor(gvft_fields[key], dtype=torch.float32, device=self.device)
            
        print(f"Field statistics after generation:")
        print(f"  Flow: min={gvft_fields['flow_x'].min().item():.3f}/{gvft_fields['flow_y'].min().item():.3f}, "
              f"max={gvft_fields['flow_x'].max().item():.3f}/{gvft_fields['flow_y'].max().item():.3f}")
        print(f"  Strength (W): min={gvft_fields['strength'].min().item():.3f}, max={gvft_fields['strength'].max().item():.3f}")
        print(f"  Neuromodulatory (eta): min={gvft_fields['neuromod'].min().item():.3f}, max={gvft_fields['neuromod'].max().item():.3f}")
        
        return gvft_fields, data, neurons_2d