import torch
import numpy as np
import scipy.ndimage as ndimage
import networkx as nx
from scipy.stats import pearsonr, spearmanr

def count_hotspots(W, threshold=1.5):
    """Count fraction of grid points where W exceeds threshold."""
    if isinstance(W, np.ndarray):
        W = torch.tensor(W)
    return torch.sum(W > threshold).float() / (W.shape[0] * W.shape[1])

def pattern_persistence(W, W_init):
    """Measure correlation between current field and initial field.
    
    Higher values indicate better preservation of initial patterns.
    
    Args:
        W: Current field state (tensor)
        W_init: Initial field state (tensor)
    
    Returns:
        Correlation coefficient between initial and current state
    """
    # Convert to numpy for correlation calculation
    if isinstance(W, torch.Tensor):
        W = W.detach().cpu().numpy()
    if isinstance(W_init, torch.Tensor):
        W_init = W_init.detach().cpu().numpy()
    
    # Flatten arrays
    W_flat = W.flatten()
    W_init_flat = W_init.flatten()
    
    # Calculate correlation
    corr = np.corrcoef(W_init_flat, W_flat)[0, 1]
    
    # Handle NaN (can happen if one field is constant)
    if np.isnan(corr):
        return 0.0
        
    return float(corr)

def structural_complexity(W):
    """Measure structural complexity using spatial frequency distribution.
    
    Higher values indicate more complex spatial patterns with a 
    balance of mid and high-frequency components.
    
    Args:
        W: Field to analyze (tensor or numpy array)
    
    Returns:
        Ratio of mid-to-high frequency energy to low frequency energy
    """
    # Convert to numpy if needed
    if isinstance(W, torch.Tensor):
        W = W.detach().cpu().numpy()
    
    # Compute 2D FFT
    W_fft = np.fft.fft2(W)
    W_fft_shifted = np.fft.fftshift(W_fft)
    magnitude = np.abs(W_fft_shifted)
    
    # Calculate energy at different frequency bands
    h, w = magnitude.shape
    center_y, center_x = h//2, w//2
    y, x = np.ogrid[-center_y:h-center_y, -center_x:w-center_x]
    dist_from_center = np.sqrt(x*x + y*y)
    
    # Define frequency bands (low, med, high)
    bands = [0, h//8, h//4, h//2]
    band_energy = []
    
    for i in range(len(bands)-1):
        mask = (dist_from_center >= bands[i]) & (dist_from_center < bands[i+1])
        band_energy.append(np.sum(magnitude[mask]))
    
    # Return ratio of mid-to-high frequency energy to low frequency energy
    # Higher values indicate more complex spatial patterns
    complexity = (band_energy[1] + band_energy[2]) / (band_energy[0] + 1e-10)
    
    # Normalize to a more interpretable range [0, 1]
    normalized_complexity = min(1.0, complexity / 10.0)
    
    return float(normalized_complexity)

def flow_coherence(Fx, Fy):
    """Measure directional coherence of flow field.
    
    Higher values indicate more organized, coherent flow patterns.
    
    Args:
        Fx, Fy: Flow field components (tensor or numpy arrays)
    
    Returns:
        Average local directional coherence [0, 1]
    """
    # Convert to numpy if needed
    if isinstance(Fx, torch.Tensor):
        Fx = Fx.detach().cpu().numpy()
        Fy = Fy.detach().cpu().numpy()
    
    # Calculate flow magnitude
    F_mag = np.sqrt(Fx**2 + Fy**2)
    mask = F_mag > 1e-6
    
    # Use gradient structure tensor approach (more efficient than pairwise comparison)
    # Compute normalized vector field
    Fx_norm = np.zeros_like(Fx)
    Fy_norm = np.zeros_like(Fy)
    Fx_norm[mask] = Fx[mask] / F_mag[mask]
    Fy_norm[mask] = Fy[mask] / F_mag[mask]
    
    # Compute local structure tensor components
    sigma = 2.0  # Smoothing scale
    Jxx = ndimage.gaussian_filter(Fx_norm * Fx_norm, sigma)
    Jxy = ndimage.gaussian_filter(Fx_norm * Fy_norm, sigma)
    Jyy = ndimage.gaussian_filter(Fy_norm * Fy_norm, sigma)
    
    # Compute coherence measure
    trace = Jxx + Jyy
    det = Jxx * Jyy - Jxy * Jxy
    # Avoid division by zero
    coherence = np.zeros_like(trace)
    valid = trace > 1e-6
    coherence[valid] = np.sqrt(1 - 4 * det[valid] / (trace[valid]**2 + 1e-10))
    
    # Return global average coherence
    valid_count = np.sum(valid)
    if valid_count > 0:
        return float(np.sum(coherence) / valid_count)
    else:
        return 0.0

def compute_field_metrics(Fx, Fy, W, W_init):
    """Compute all metrics for current field state.
    
    Args:
        Fx, Fy: Flow field components
        W: Current strength field
        W_init: Initial strength field
    
    Returns:
        Dictionary of metrics
    """
    metrics = {
        'hotspot_fraction': float(count_hotspots(W).item()) if isinstance(W, torch.Tensor) else float(count_hotspots(torch.tensor(W))),
        'pattern_persistence': float(pattern_persistence(W, W_init)),
        'structural_complexity': float(structural_complexity(W)),
        'flow_coherence': float(flow_coherence(Fx, Fy))
    }
    
    # Compute an overall "pattern quality" metric
    # Weighted combination of individual metrics
    metrics['pattern_quality'] = float(
        metrics['hotspot_fraction'] * 0.3 + 
        metrics['pattern_persistence'] * 0.3 + 
        metrics['structural_complexity'] * 0.2 +
        metrics['flow_coherence'] * 0.2
    )
    
    return metrics

# NEW FUNCTIONS FOR BIOLOGICAL CONNECTOME VALIDATION

def build_graph_from_connections(connections, neurons=None):
    """Build NetworkX graph from connection data.
    
    Args:
        connections: List of connection tuples (pre_id, post_id, weight, delay, excitatory)
        neurons: Optional dictionary of neuron properties
        
    Returns:
        NetworkX DiGraph representing the connectome
    """
    G = nx.DiGraph()
    
    # Add neurons as nodes
    if neurons:
        for neuron_id, props in neurons.items():
            G.add_node(neuron_id, **props)
    
    # Add connections as edges
    for pre_id, post_id, weight, delay, excitatory in connections:
        G.add_edge(pre_id, post_id, weight=weight, delay=delay, excitatory=excitatory)
    
    return G

def build_graph_from_gvft_fields(Fx, Fy, W, module_positions, config):
    """Build a graph from GVFT fields and module positions.
    
    Args:
        Fx, Fy: Flow field components
        W: Strength field
        module_positions: Array of module positions (Nx2)
        config: Dictionary with parameters (top_k, cos_threshold, lambda_val)
        
    Returns:
        NetworkX DiGraph representing the reconstructed connectome
    """
    # Convert to numpy if needed
    if isinstance(Fx, torch.Tensor):
        Fx = Fx.detach().cpu().numpy()
    if isinstance(Fy, torch.Tensor):
        Fy = Fy.detach().cpu().numpy()
    if isinstance(W, torch.Tensor):
        W = W.detach().cpu().numpy()
    
    # Create graph
    G = nx.DiGraph()
    
    num_modules = len(module_positions)
    grid_size = Fx.shape[0]
    domain_size = 2.0  # Assuming domain is [-1, 1] x [-1, 1]
    
    # Map module positions to grid indices
    module_indices = []
    for pos in module_positions:
        x, y = pos
        # Normalize to grid coordinates
        i_x = int((x + 1) / domain_size * grid_size)
        i_y = int((y + 1) / domain_size * grid_size)
        # Clamp to valid range
        i_x = max(0, min(grid_size - 1, i_x))
        i_y = max(0, min(grid_size - 1, i_y))
        module_indices.append((i_x, i_y))
    
    # Add modules as nodes
    for i in range(num_modules):
        G.add_node(i, pos=module_positions[i])
    
    # Generate weights based on field values
    for i in range(num_modules):
        i_x, i_y = module_indices[i]
        F_i = np.array([Fx[i_y, i_x], Fy[i_y, i_x]])
        W_i = W[i_y, i_x]
        
        # Calculate potential connections
        candidate_edges = []
        for j in range(num_modules):
            if i == j:
                continue
            
            r_i = module_positions[i]
            r_j = module_positions[j]
            delta = r_j - r_i
            norm_delta = np.linalg.norm(delta)
            norm_F = np.linalg.norm(F_i)
            
            if norm_delta < 1e-6 or norm_F < 1e-6:
                continue
            
            cos_sim = np.dot(delta, F_i) / (norm_delta * norm_F)
            cos_sim = np.clip(cos_sim, -1.0, 1.0)
            
            if cos_sim < config.get('cos_threshold', 0.2):
                continue
            
            rho = cos_sim * W_i
            w_ij = np.clip(rho / config.get('lambda_val', 0.2), 0, 1.0)
            candidate_edges.append((j, w_ij))
        
        # Sort and keep top-k edges
        top_k = config.get('top_k', 2)
        candidate_edges = sorted(candidate_edges, key=lambda item: item[1], reverse=True)[:top_k]
        for j, w_ij in candidate_edges:
            G.add_edge(i, j, weight=w_ij, excitatory=(w_ij > 0))
    
    return G

def connectome_fidelity(source_graph, generated_graph, metric_type='graph_edit_distance'):
    """Calculate fidelity between source biological connectome and generated connectome.
    
    Args:
        source_graph: NetworkX graph representing the biological connectome
        generated_graph: NetworkX graph derived from GVFT fields
        metric_type: Type of graph similarity metric to use
        
    Returns:
        Fidelity score between 0 and 1, where 1 is perfect fidelity
    """
    if metric_type == 'graph_edit_distance':
        # Compute normalized graph edit distance
        # (lower is better, so invert for fidelity)
        try:
            # Use edge substitution cost based on weight differences
            def edge_subst_cost(e1, e2):
                w1 = source_graph.edges[e1].get('weight', 1.0)
                w2 = generated_graph.edges[e2].get('weight', 1.0)
                return abs(w1 - w2)
            
            # Node substitution cost is always 0 (nodes are interchangeable)
            def node_subst_cost(n1, n2):
                return 0.0
            
            # Edge deletion/insertion cost is 1.0
            def edge_del_cost(e):
                return 1.0
                
            def edge_ins_cost(e):
                return 1.0
            
            # Compute graph edit distance with custom costs
            ged = nx.graph_edit_distance(
                source_graph, generated_graph,
                node_subst_cost=node_subst_cost,
                edge_subst_cost=edge_subst_cost,
                edge_del_cost=edge_del_cost,
                edge_ins_cost=edge_ins_cost
            )
            
            # Normalize by graph size (approximate max possible edit distance)
            max_possible_ged = (source_graph.number_of_edges() + 
                               generated_graph.number_of_edges())
            
            # Convert to fidelity score (1 - normalized distance)
            if max_possible_ged > 0:
                fidelity = 1.0 - ged / max_possible_ged
            else:
                fidelity = 0.0
                
            return max(0.0, min(1.0, fidelity))
            
        except Exception as e:
            print(f"Warning: Graph edit distance calculation failed: {e}")
            # Fall back to simpler method
            metric_type = 'edge_overlap'
    
    if metric_type == 'edge_overlap':
        # Calculate edge overlap (Jaccard similarity of edge sets)
        source_edges = set(source_graph.edges())
        generated_edges = set(generated_graph.edges())
        
        intersection = len(source_edges.intersection(generated_edges))
        union = len(source_edges.union(generated_edges))
        
        # Handle empty graphs
        if union == 0:
            return 0.0
            
        return intersection / union
    
    if metric_type == 'modularity_similarity':
        # Compare modularity structure between graphs
        try:
            # Get communities for both graphs
            source_communities = list(nx.community.louvain_communities(
                source_graph.to_undirected(), resolution=1.0))
            generated_communities = list(nx.community.louvain_communities(
                generated_graph.to_undirected(), resolution=1.0))
            
            # Calculate modularity scores
            source_modularity = nx.community.modularity(
                source_graph.to_undirected(), source_communities)
            generated_modularity = nx.community.modularity(
                generated_graph.to_undirected(), generated_communities)
            
            # Compare modularity values
            # Return similarity based on absolute difference
            modularity_diff = abs(source_modularity - generated_modularity)
            
            # Convert to similarity (1 - normalized difference)
            # Max modularity difference is 2.0 (range is -1 to 1)
            fidelity = 1.0 - modularity_diff / 2.0
            
            return max(0.0, min(1.0, fidelity))
            
        except Exception as e:
            print(f"Warning: Modularity comparison failed: {e}")
            return 0.0
    
    if metric_type == 'weight_correlation':
        # Compare edge weight distributions
        # Get common edges between graphs
        common_edges = set(source_graph.edges()).intersection(set(generated_graph.edges()))
        
        if not common_edges:
            return 0.0
            
        # Get weights for common edges
        source_weights = [source_graph.edges[e].get('weight', 1.0) for e in common_edges]
        generated_weights = [generated_graph.edges[e].get('weight', 1.0) for e in common_edges]
        
        # Calculate correlation
        if len(source_weights) > 1:
            try:
                # Use Spearman rank correlation (more robust to non-linear relationships)
                corr, _ = spearmanr(source_weights, generated_weights)
                
                # Handle NaN (can happen with constant weights)
                if np.isnan(corr):
                    corr = 0.0
                    
                # Convert to [0, 1] range
                fidelity = (corr + 1.0) / 2.0
                
                return max(0.0, min(1.0, fidelity))
                
            except Exception as e:
                print(f"Warning: Weight correlation calculation failed: {e}")
                return 0.0
        else:
            # Only one common edge, can't compute correlation
            return 0.5  # Neutral score
    
    # Compute combined fidelity score using multiple metrics
    if metric_type == 'combined':
        edge_score = connectome_fidelity(source_graph, generated_graph, 'edge_overlap')
        mod_score = connectome_fidelity(source_graph, generated_graph, 'modularity_similarity')
        weight_score = connectome_fidelity(source_graph, generated_graph, 'weight_correlation')
        
        # Weighted average of the three scores
        combined_score = 0.4 * edge_score + 0.3 * mod_score + 0.3 * weight_score
        
        return combined_score
    
    # Default to edge overlap if unknown metric type
    return connectome_fidelity(source_graph, generated_graph, 'edge_overlap')

def compute_connectome_metrics(source_connections, module_positions, Fx, Fy, W, config):
    """Compute comprehensive metrics comparing source connectome to GVFT-generated connectome.
    
    Args:
        source_connections: List of connection tuples from source connectome
        module_positions: Array of module positions
        Fx, Fy: Flow field components
        W: Strength field
        config: Dictionary with parameters
        
    Returns:
        Dictionary of connectome fidelity metrics
    """
    # Build source graph
    source_graph = build_graph_from_connections(source_connections)
    
    # Build graph from GVFT fields
    generated_graph = build_graph_from_gvft_fields(Fx, Fy, W, module_positions, config)
    
    # Compute metrics
    metrics = {
        'edge_overlap': connectome_fidelity(source_graph, generated_graph, 'edge_overlap'),
        'modularity_similarity': connectome_fidelity(source_graph, generated_graph, 'modularity_similarity'),
        'weight_correlation': connectome_fidelity(source_graph, generated_graph, 'weight_correlation'),
        'combined_fidelity': connectome_fidelity(source_graph, generated_graph, 'combined')
    }
    
    # Add graph statistics
    metrics['source_edge_count'] = source_graph.number_of_edges()
    metrics['generated_edge_count'] = generated_graph.number_of_edges()
    metrics['source_clustering'] = nx.average_clustering(source_graph.to_undirected())
    metrics['generated_clustering'] = nx.average_clustering(generated_graph.to_undirected())
    
    return metrics