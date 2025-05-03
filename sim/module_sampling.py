import torch
import torch.nn.functional as F
import numpy as np

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

def sample_module_positions(Fx, Fy, W, num_modules, beta, gamma, X, Y):
    """Sample module positions based on field-guided probability using GPU-native operations"""
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
    return module_coords

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