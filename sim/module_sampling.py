import torch
import torch.nn.functional as F
import numpy as np
import networkx as nx

def compute_module_probability(Fx, Fy, W, beta, gamma):
    """Compute module placement probability based on fields"""
    F_mag = torch.sqrt(Fx**2 + Fy**2)
    logits = torch.clamp(beta * F_mag + gamma * W, -50, 50)
    prob = 1 / (1 + torch.exp(-logits))
    prob_sum = prob.sum()
    
    if prob_sum <= 1e-10:
        prob = torch.ones_like(prob) / prob.numel()
    else:
        prob = prob / prob_sum
        
    return prob

def sample_module_positions(Fx, Fy, W, num_modules, beta, gamma, X, Y, config=None):
    """Sample module positions based on field-guided probability using GPU-native operations
    
    If config contains bio_module_positions, those will be used instead of sampling.
    """
    # Check if we have biological positions available
    if config is not None and hasattr(config, 'bio_module_positions') and config.bio_module_positions is not None:
        bio_positions = config.bio_module_positions
        
        # If the number of modules doesn't match, we sample a subset or pad
        if len(bio_positions) != num_modules:
            if len(bio_positions) > num_modules:
                # Sample a subset of positions
                indices = torch.randperm(len(bio_positions), device=bio_positions.device)[:num_modules]
                return bio_positions[indices]
            else:
                # We need to add more positions
                # Use the provided positions and sample the rest
                prob = compute_module_probability(Fx, Fy, W, beta, gamma)
                flat_prob = prob.flatten()
                flat_prob = flat_prob / (flat_prob.sum() + 1e-10)
                
                device = Fx.device
                grid_size = Fx.shape[0]
                
                # Sample additional positions
                additional_count = num_modules - len(bio_positions)
                try:
                    indices = torch.multinomial(flat_prob, additional_count, replacement=False)
                except ValueError as e:
                    print(f"Error sampling module positions: {e}, Sum(p)={flat_prob.sum()}. Using uniform sampling.")
                    indices = torch.randperm(grid_size * grid_size, device=device)[:additional_count]
                
                yi, xi = torch.div(indices, grid_size, rounding_mode='floor'), indices % grid_size
                
                # Extract coordinates
                additional_coords = torch.stack([X[yi, xi], Y[yi, xi]], dim=1)
                
                # Combine with biological positions
                return torch.cat([bio_positions, additional_coords], dim=0)
        else:
            # Perfect match - use all biological positions
            return bio_positions
    
    # No biological positions available, use standard sampling
    prob = compute_module_probability(Fx, Fy, W, beta, gamma)
    flat_prob = prob.flatten()
    flat_prob = flat_prob / (flat_prob.sum() + 1e-10)
    
    device = Fx.device
    grid_size = Fx.shape[0]
    
    try:
        # Use torch.multinomial instead of numpy for GPU efficiency
        indices = torch.multinomial(flat_prob, num_modules, replacement=False)
    except ValueError as e:
        print(f"Error sampling module positions: {e}, Sum(p)={flat_prob.sum()}. Using uniform sampling.")
        indices = torch.randperm(grid_size * grid_size, device=device)[:num_modules]
    
    yi, xi = torch.div(indices, grid_size, rounding_mode='floor'), indices % grid_size
    
    # Extract coordinates
    module_coords = torch.stack([X[yi, xi], Y[yi, xi]], dim=1)
    
    # Apply force-directed refinement if we have more than 2 modules
    if num_modules > 2:
        module_coords = refine_module_positions_with_force_directed(
            module_coords, Fx, Fy, W, beta, gamma, device)
    
    return module_coords

def refine_module_positions_with_force_directed(initial_positions, Fx, Fy, W, beta, gamma, device):
    """Refine module positions using force-directed layout based on field values.
    
    This implements a light version of the force-directed layout algorithm to adjust
    module positions while preserving their general distribution according to field values.
    """
    # Convert tensors to CPU for NetworkX operations
    module_coords_np = initial_positions.cpu().numpy()
    Fx_np = Fx.cpu().numpy()
    Fy_np = Fy.cpu().numpy()
    W_np = W.cpu().numpy()
    
    num_modules = module_coords_np.shape[0]
    grid_size = Fx_np.shape[0]
    
    # Create a fully connected graph for modules
    G = nx.Graph()
    
    # Add nodes with positions
    for i in range(num_modules):
        G.add_node(i, pos=tuple(module_coords_np[i]))
    
    # Compute field-based weights between nodes
    for i in range(num_modules):
        # Get approximate field values at module position
        x, y = module_coords_np[i]
        # Convert [-1,1] coordinates to [0,grid_size-1] indices
        ix = int((x + 1) / 2 * (grid_size - 1))
        iy = int((y + 1) / 2 * (grid_size - 1))
        # Clamp to valid range
        ix = max(0, min(grid_size - 1, ix))
        iy = max(0, min(grid_size - 1, iy))
        
        # Get field values
        fx_i = Fx_np[iy, ix]
        fy_i = Fy_np[iy, ix]
        w_i = W_np[iy, ix]
        
        for j in range(i+1, num_modules):
            # Calculate a weight based on field coherence
            x2, y2 = module_coords_np[j]
            ix2 = int((x2 + 1) / 2 * (grid_size - 1))
            iy2 = int((y2 + 1) / 2 * (grid_size - 1))
            # Clamp to valid range
            ix2 = max(0, min(grid_size - 1, ix2))
            iy2 = max(0, min(grid_size - 1, iy2))
            
            # Get field values at second point
            fx_j = Fx_np[iy2, ix2]
            fy_j = Fy_np[iy2, ix2]
            w_j = W_np[iy2, ix2]
            
            # Calculate coherence as dot product of flow vectors
            f_dot = fx_i * fx_j + fy_i * fy_j
            
            # Calculate weight based on field coherence and strength
            weight = (1.0 + f_dot) * (abs(w_i) + abs(w_j)) / 4.0
            
            # Distance between points
            dist = np.sqrt((x - x2)**2 + (y - y2)**2)
            
            # Add edge with weight
            G.add_edge(i, j, weight=weight, distance=dist)
    
    # Apply force-directed layout with position constraints
    # k parameter controls the optimal distance between nodes
    pos = nx.spring_layout(
        G, 
        pos={i: tuple(pos) for i, pos in enumerate(module_coords_np)},
        k=0.15,  # Smaller k keeps nodes closer
        iterations=50,  # Fewer iterations to maintain field-based distribution
        weight='weight',
        fixed=None  # No fixed positions
    )
    
    # Convert back to tensor
    refined_positions = np.array([pos[i] for i in range(num_modules)])
    
    # Ensure positions stay within [-1, 1] bounds
    refined_positions = np.clip(refined_positions, -1, 1)
    
    # Convert back to torch tensor on the original device
    return torch.tensor(refined_positions, device=device, dtype=torch.float32)

def build_connectivity_graph(module_coords, Fx, Fy, W, domain, config):
    """Build connectivity graph between modules based on flow field alignment."""
    num_modules = module_coords.shape[0]
    device = Fx.device
    weights = torch.zeros((num_modules, num_modules), device=device)
    
    x, y = domain['x'], domain['y']
    cos_threshold = config.cos_threshold
    lambda_val = config.lambda_val
    top_k = config.top_k
    
    for i in range(num_modules):
        # Find closest grid points to module positions
        xi_idx = torch.argmin(torch.abs(x - module_coords[i, 0]))
        yi_idx = torch.argmin(torch.abs(y - module_coords[i, 1]))
        
        # Get field values at module position
        F_i = torch.tensor([Fx[yi_idx, xi_idx], Fy[yi_idx, xi_idx]], device=device)
        W_i = W[yi_idx, xi_idx]
        
        # Calculate potential connections
        candidate_edges = []
        for j in range(num_modules):
            if i == j: 
                continue
            
            r_i = module_coords[i]
            r_j = module_coords[j]
            delta = r_j - r_i
            norm_delta = torch.norm(delta)
            norm_F = torch.norm(F_i)
            
            if norm_delta < 1e-6 or norm_F < 1e-6: 
                continue
            
            cos_sim = torch.dot(delta, F_i) / (norm_delta * norm_F)
            cos_sim = torch.nan_to_num(torch.clamp(cos_sim, -1.0, 1.0))
            
            if cos_sim < cos_threshold: 
                continue
            
            rho = cos_sim * W_i
            w_ij = torch.clamp(rho / lambda_val, 0, 1.0)
            candidate_edges.append((j, w_ij.item()))
        
        # Sort and keep top-k edges
        candidate_edges = sorted(candidate_edges, key=lambda item: item[1], reverse=True)[:top_k]
        for j, w_ij in candidate_edges:
            weights[i, j] = w_ij
    
    return weights