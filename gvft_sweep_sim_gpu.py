import os
import time
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import torch.nn.functional as F
import gc

# --- ENSURE FIGURE DIRECTORY ---
os.makedirs("figures", exist_ok=True)

# --- CHECK FOR CUDA GPU ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

if device.type == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory allocated: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
    print(f"Memory reserved: {torch.cuda.memory_reserved(0) / 1024**2:.2f} MB")

# --- BASE PARAMETERS ---
# Reference grid size for parameter scaling
base_grid_size = 200 
grid_size = 200  # Can be changed, parameters will scale accordingly
use_parameter_scaling = True  # Set to True to enable scaling

# Base time parameters (for grid_size=100)
base_timesteps_sweep = 100
base_timesteps_sim = 500
base_dt = 0.1

# Scale time parameters based on grid resolution if scaling is enabled
if use_parameter_scaling:
    # Scale factor based on grid resolution (squared relationship for diffusion)
    scale_factor = (base_grid_size / grid_size) ** 2
    
    # Scale timestep and increase iterations to maintain same simulation time
    dt = base_dt * scale_factor
    timesteps_sweep = int(base_timesteps_sweep / scale_factor)
    timesteps_sim = int(base_timesteps_sim / scale_factor)
    
    # Ensure minimum number of timesteps
    timesteps_sweep = max(timesteps_sweep, 50)
    timesteps_sim = max(timesteps_sim, 100)
else:
    # Use base parameters directly if scaling is disabled
    dt = base_dt
    timesteps_sweep = base_timesteps_sweep
    timesteps_sim = base_timesteps_sim

print(f"Grid size: {grid_size}x{grid_size}, dt: {dt}, Timesteps: {timesteps_sweep}/{timesteps_sim}")

# --- OTHER FIXED PARAMETERS ---
view_step = max(1, timesteps_sim // 10)  # Scale view steps too
num_modules = 10
top_k = 2
cos_threshold = 0.2
lambda_val = 0.2

# --- GVFT Field Parameters (Base values) ---
# These will be scaled according to grid resolution
base_D_W = 0.03    
base_lam_F = 0.007
base_beta = 0.5
base_gamma = 0.5
base_alpha = 1.5
base_beta_coupling = 0.1
base_D_eta = 0.02   
base_lam_eta = 0.1
base_eta_coeff = 0.05
base_noise_F = 0.0001
base_noise_W = 0.001

# Apply scaling to diffusion coefficients and rates if enabled
if use_parameter_scaling:
    # Diffusion coefficients scale with dx² (inverse to grid_size²)
    D_W = base_D_W * scale_factor
    D_eta = base_D_eta * scale_factor
    
    # Decay rates remain constant (time-based, not space-based)
    lam_F = base_lam_F
    lam_eta = base_lam_eta
    
    # These parameters affect field coupling, scale with grid size
    beta = base_beta
    gamma = base_gamma
    alpha = base_alpha
    beta_coupling = base_beta_coupling * scale_factor  # Scales with gradient magnitude
    eta_coeff = base_eta_coeff
    
    # Noise terms should scale with grid resolution too
    noise_F = base_noise_F / np.sqrt(scale_factor)
    noise_W = base_noise_W / np.sqrt(scale_factor)
else:
    # Use base parameters directly
    D_W = base_D_W
    lam_F = base_lam_F
    beta = base_beta
    gamma = base_gamma
    alpha = base_alpha
    beta_coupling = base_beta_coupling
    D_eta = base_D_eta
    lam_eta = base_lam_eta
    eta_coeff = base_eta_coeff
    noise_F = base_noise_F
    noise_W = base_noise_W

# --- Define Parameter Sweep Ranges ---
# These remain fixed, we explore the same parameter space regardless of grid size
D_F_values = torch.linspace(0.002, 0.007, 10).to(device)
lam_W_values = torch.linspace(0.1, 0.5, 10).to(device)

# Scale D_F values if needed
if use_parameter_scaling:
    D_F_values = D_F_values * scale_factor

# --- DOMAIN SETUP ---
torch.manual_seed(42)
np.random.seed(42)
x = torch.linspace(-1, 1, grid_size).to(device)
y = torch.linspace(-1, 1, grid_size).to(device)
dx = x[1] - x[0]
dy = y[1] - y[0]
Y, X = torch.meshgrid(y, x, indexing='ij')

# --- LAPLACIAN COMPUTATION ---
def build_laplacian_operator(grid_size, dx, dy):
    """Build a function that applies the Laplacian operator using PyTorch's conv2d"""
    # Create Laplacian kernel
    laplacian_kernel = torch.zeros((1, 1, 3, 3), device=device)
    laplacian_kernel[0, 0, 0, 1] = 1.0 / dy**2
    laplacian_kernel[0, 0, 1, 0] = 1.0 / dx**2
    laplacian_kernel[0, 0, 1, 2] = 1.0 / dx**2
    laplacian_kernel[0, 0, 2, 1] = 1.0 / dy**2
    laplacian_kernel[0, 0, 1, 1] = -2.0 * (1.0 / dx**2 + 1.0 / dy**2)
    
    def apply_laplacian(tensor):
        """Apply the Laplacian operator to a 2D tensor"""
        # Add batch and channel dimensions, pad, then convolve
        padded = F.pad(tensor.unsqueeze(0).unsqueeze(0), (1, 1, 1, 1), mode='replicate')
        return F.conv2d(padded, laplacian_kernel).squeeze()
    
    return apply_laplacian

apply_laplacian = build_laplacian_operator(grid_size, dx, dy)

# --- INITIALIZE COMMON STARTING FIELDS ---
# Gaussian filter function using PyTorch
def gaussian_filter(tensor, sigma=1.0):
    """Apply Gaussian filter to a tensor using PyTorch"""
    # Calculate kernel size based on sigma (3*sigma rule of thumb)
    kernel_size = int(2 * 3 * sigma + 1)
    kernel_size = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
    
    # Create Gaussian kernel
    x = torch.arange(-(kernel_size // 2), kernel_size // 2 + 1, device=device).float()
    kernel_1d = torch.exp(-x**2 / (2 * sigma**2))
    kernel_1d = kernel_1d / kernel_1d.sum()
    
    # Apply separable convolution
    tensor_padded = F.pad(tensor.unsqueeze(0).unsqueeze(0), 
                         (kernel_size // 2, kernel_size // 2, kernel_size // 2, kernel_size // 2),
                         mode='replicate')
    tensor_filtered = F.conv2d(tensor_padded, kernel_1d.view(1, 1, 1, -1).repeat(1, 1, 1, 1), padding=0)
    tensor_filtered = F.conv2d(tensor_filtered, kernel_1d.view(1, 1, -1, 1).repeat(1, 1, 1, 1), padding=0)
    
    return tensor_filtered.squeeze()

# Initialize fields with proper GPU tensors
# Scale the sigma for gaussian filters based on grid size to maintain similar features
filter_scale = base_grid_size / grid_size
potential = gaussian_filter(torch.randn(grid_size, grid_size, device=device), sigma=5*filter_scale)
Fy_init, Fx_init = torch.gradient(potential)
F_mag_init = torch.sqrt(Fx_init**2 + Fy_init**2)
mask = F_mag_init > 1e-6
Fx_init[mask] = Fx_init[mask] / F_mag_init[mask]
Fy_init[mask] = Fy_init[mask] / F_mag_init[mask]

W_init = gaussian_filter(torch.rand(grid_size, grid_size, device=device) - 0.5, sigma=3*filter_scale) * 0.1
eta_init = gaussian_filter(torch.rand(grid_size, grid_size, device=device) - 0.5, sigma=5*filter_scale) * 0.1

# --- UTILITY FUNCTIONS ---
def normalize_field(Fx, Fy, max_mag=1.0):
    """Normalize vector field to have magnitude <= max_mag"""
    F_mag = torch.sqrt(Fx**2 + Fy**2)
    mask = F_mag > 1e-6
    scale = torch.ones_like(F_mag)
    scale[mask] = torch.clamp(max_mag / F_mag[mask], max=1.0)
    return Fx * scale, Fy * scale

def clip_field(field, min_val=-10.0, max_val=10.0):
    """Clip field values to stay within bounds"""
    return torch.clamp(field, min_val, max_val)

def count_hotspots(W, threshold=1.5):
    """Count fraction of grid points where W exceeds threshold"""
    return torch.sum(W > threshold).float() / (grid_size * grid_size)

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

# --- CRANK-NICOLSON SOLVER FUNCTIONS ---
def solve_diffusion_equation(field, A_inv_B, A_inv, source_term):
    """Solve diffusion equation using pre-computed operators"""
    # Apply A⁻¹ * B to the field
    field_next = A_inv_B(field)
    
    # Add source term: field_next += A⁻¹ * dt * source
    field_next = field_next + A_inv(dt * source_term)
    
    return field_next

def create_crank_nicolson_operators(D, lam, dt, grid_size, apply_laplacian):
    """Create operators for Crank-Nicolson method solving (I - (dt/2)(D*∇² - λI))u = (I + (dt/2)(D*∇² - λI))u₀ + dt*S"""
    def A_op(u):
        """Apply (I - (dt/2)(D*∇² - λI)) to u"""
        return u - (dt/2) * (D * apply_laplacian(u) - lam * u)
    
    def A_inv(b):
        """Solve (I - (dt/2)(D*∇² - λI))x = b for x with optimized convergence"""
        # Initialize with b
        x = b.clone()
        
        # Define residual function
        def residual(x):
            return b - A_op(x)
        
        # Use conjugate gradient method
        r = residual(x)
        p = r.clone()
        rsold = (r * r).sum()
        
        # Skip iterations if already close to solution
        if torch.sqrt(rsold) < 1e-9:
            return x
        
        # Scale max iterations based on grid size
        max_cg_iterations = min(100, 20 * grid_size // base_grid_size)
        
        for i in range(max_cg_iterations):
            Ap = A_op(p)
            
            # Avoid division by zero
            pAp = (p * Ap).sum()
            if pAp < 1e-10:
                break
                
            alpha = rsold / pAp
            x = x + alpha * p
            r = r - alpha * Ap
            rsnew = (r * r).sum()
            
            # Scale convergence criteria with grid size
            # For larger grids, we can use a looser tolerance
            tol = 1e-10 * (base_grid_size / grid_size)
            if torch.sqrt(rsnew) < tol:
                break
                
            p = r + (rsnew / rsold) * p
            rsold = rsnew
            
        return x
    
    def B_op(u):
        """Apply (I + (dt/2)(D*∇² - λI)) to u"""
        return u + (dt/2) * (D * apply_laplacian(u) - lam * u)
    
    def A_inv_B(u):
        """Apply A⁻¹ * B to u"""
        return A_inv(B_op(u))
    
    return A_inv_B, A_inv

def optimize_batch_size():
    """Calculate an optimal batch size based on grid size and GPU memory"""
    if device.type != 'cuda':
        return 4  # Default for CPU
        
    try:
        total_memory = torch.cuda.get_device_properties(0).total_memory
        # Adjust available memory based on grid size
        # Larger grids need more memory per simulation
        memory_per_sim = (grid_size / base_grid_size)**2 * 100 * 1024 * 1024  # Base ~100MB for 100x100
        
        # Use 80% of available memory
        optimal_batch_size = max(1, int(total_memory * 0.8 / memory_per_sim))
        
        # Cap at reasonable values based on grid size
        if grid_size <= 100:
            return min(optimal_batch_size, 32)
        elif grid_size <= 200:
            return min(optimal_batch_size, 16)
        elif grid_size <= 500:
            return min(optimal_batch_size, 8)
        else:
            return min(optimal_batch_size, 4)
    except Exception as e:
        print(f"Error calculating batch size: {e}")
        # Conservative defaults based on grid size
        if grid_size <= 100:
            return 10
        elif grid_size <= 200:
            return 6
        elif grid_size <= 500:
            return 3
        else:
            return 1

# --- PARAMETER SWEEP FUNCTION ---
def run_sweep_simulation(param_batch):
    """Run a batch of parameter sweep simulations"""
    results = []
    batch_start_time = time.time()
    
    # Try to precompute eta operators which are fixed for all simulations
    try:
        eta_A_inv_B, eta_A_inv = create_crank_nicolson_operators(D_eta, lam_eta, dt, grid_size, apply_laplacian)
        precomputed_eta = True
    except Exception as e:
        print(f"Could not precompute eta operators: {e}")
        precomputed_eta = False
    
    for batch_idx, (lam_W_idx, D_F_idx) in enumerate(param_batch):
        lam_W = lam_W_values[lam_W_idx]
        D_F = D_F_values[D_F_idx]
        total_combinations = len(lam_W_values) * len(D_F_values)
        current_combination = lam_W_idx * len(D_F_values) + D_F_idx + 1
        
        sim_start_time = time.time()
        print(f"Starting sweep simulation {current_combination}/{total_combinations}: lam_W={lam_W:.3f}, D_F={D_F:.3f}")
        
        # Create Crank-Nicolson operators for each field
        F_A_inv_B, F_A_inv = create_crank_nicolson_operators(D_F, lam_F, dt, grid_size, apply_laplacian)
        W_A_inv_B, W_A_inv = create_crank_nicolson_operators(D_W, lam_W, dt, grid_size, apply_laplacian)
        
        # Only create eta operators if not precomputed
        if not precomputed_eta:
            eta_A_inv_B, eta_A_inv = create_crank_nicolson_operators(D_eta, lam_eta, dt, grid_size, apply_laplacian)
        
        # Reset fields for this parameter set
        Fx = Fx_init.clone()
        Fy = Fy_init.clone()
        W = W_init.clone()
        eta = eta_init.clone()
        
        # Run simulation
        for t in range(timesteps_sweep):
            if t % 100 == 0:
                print(f"Sweep simulation {current_combination}/{total_combinations}: lam_W={lam_W:.3f}, D_F={D_F:.3f} - Timestep {t}/{timesteps_sweep}")
                
            # Calculate field magnitudes and gradients
            F_mag = torch.sqrt(Fx**2 + Fy**2)
            Wy, Wx = torch.gradient(W)
            
            # Update eta field
            S_eta = 0.5 * (F_mag**2 + W**2)
            eta = solve_diffusion_equation(eta, eta_A_inv_B, eta_A_inv, S_eta)
            eta = clip_field(eta, -5, 5)
            
            # Update Fx field
            noise_Fx = torch.randn_like(Fx) * noise_F
            S_Fx = beta_coupling * Wx + noise_Fx
            Fx = solve_diffusion_equation(Fx, F_A_inv_B, F_A_inv, S_Fx)
            
            # Update Fy field
            noise_Fy = torch.randn_like(Fy) * noise_F
            S_Fy = beta_coupling * Wy + noise_Fy
            Fy = solve_diffusion_equation(Fy, F_A_inv_B, F_A_inv, S_Fy)
            
            # Normalize flow field
            Fx, Fy = normalize_field(Fx, Fy)
            
            # Update W field
            noise_W_t = torch.randn_like(W) * noise_W
            S_W = eta_coeff * eta + alpha * F_mag + noise_W_t
            W = solve_diffusion_equation(W, W_A_inv_B, W_A_inv, S_W)
            W = clip_field(W)
            
            # Normalize W field
            max_abs_W = torch.max(torch.abs(W))
            if max_abs_W > 1e-10:
                W = W / max_abs_W * 2.0
            else:
                W.fill_(0.0)
        
        # Calculate hotspot fraction
        hotspot_fraction = count_hotspots(W).item()
        
        # Report simulation time and results
        sim_time = time.time() - sim_start_time
        print(f"Completed sweep simulation {current_combination}/{total_combinations}: "
              f"lam_W={lam_W:.3f}, D_F={D_F:.3f}, Hotspot Fraction={hotspot_fraction:.3f}")
        print(f"  Simulation completed in {sim_time:.2f}s")
        
        results.append((lam_W_idx.item(), D_F_idx.item(), hotspot_fraction))
        
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

# --- FULL SIMULATION FUNCTION ---
def run_full_simulation(params):
    """Run a full simulation with selected parameters and create visualizations"""
    idx, current_lam_W, D_F = params
    total_simulations = len(selected_params_full)
    print(f"--- Starting full simulation {idx+1}/{total_simulations}: lam_W={current_lam_W:.3f}, D_F={D_F:.3f} ---")
    
    # Create Crank-Nicolson operators for each field
    F_A_inv_B, F_A_inv = create_crank_nicolson_operators(D_F, lam_F, dt, grid_size, apply_laplacian)
    W_A_inv_B, W_A_inv = create_crank_nicolson_operators(D_W, current_lam_W, dt, grid_size, apply_laplacian)
    eta_A_inv_B, eta_A_inv = create_crank_nicolson_operators(D_eta, lam_eta, dt, grid_size, apply_laplacian)
    
    # Reset fields
    Fx = Fx_init.clone()
    Fy = Fy_init.clone()
    W = W_init.clone()
    eta = eta_init.clone()
    
    # Initialize lists to store fields and graphs
    # Use lists to reduce memory usage during simulation
    Fx_series, Fy_series, W_series = [], [], []
    Graphs, Module_coords_series = [], []
    
    # Only store fields at specified timesteps to save memory
    timesteps_to_store = list(range(0, timesteps_sim + 1, view_step))
    if timesteps_sim not in timesteps_to_store:
        timesteps_to_store.append(timesteps_sim)
    
    # Initial module positions at t=0
    module_coords = sample_module_positions(Fx, Fy, W, num_modules, beta, gamma, X, Y)
    Module_coords_series.append(module_coords.clone().cpu().numpy())
    
    # Store initial fields
    if 0 in timesteps_to_store:
        Fx_series.append(Fx.clone().cpu().numpy())
        Fy_series.append(Fy.clone().cpu().numpy())
        W_series.append(W.clone().cpu().numpy())
    
    for t in range(1, timesteps_sim + 1):
        if t % max(1, timesteps_sim // 10) == 0:
            print(f"Full simulation {idx+1}/{total_simulations}: lam_W={current_lam_W:.3f}, D_F={D_F:.3f} - Timestep {t}/{timesteps_sim}")
        
        # Calculate field magnitudes and gradients
        F_mag = torch.sqrt(Fx**2 + Fy**2)
        Wy, Wx = torch.gradient(W)
        
        # Update eta field
        S_eta = 0.5 * (F_mag**2 + W**2)
        eta = solve_diffusion_equation(eta, eta_A_inv_B, eta_A_inv, S_eta)
        eta = clip_field(eta, -5, 5)
        
        # Update Fx field
        noise_Fx = torch.randn_like(Fx) * noise_F
        S_Fx = beta_coupling * Wx + noise_Fx
        Fx = solve_diffusion_equation(Fx, F_A_inv_B, F_A_inv, S_Fx)
        
        # Update Fy field
        noise_Fy = torch.randn_like(Fy) * noise_F
        S_Fy = beta_coupling * Wy + noise_Fy
        Fy = solve_diffusion_equation(Fy, F_A_inv_B, F_A_inv, S_Fy)
        
        # Normalize flow field
        Fx, Fy = normalize_field(Fx, Fy)
        
        # Update W field
        noise_W_t = torch.randn_like(W) * noise_W
        S_W = eta_coeff * eta + alpha * F_mag + noise_W_t
        W = solve_diffusion_equation(W, W_A_inv_B, W_A_inv, S_W)
        W = clip_field(W)
        
        # Normalize W field
        max_abs_W = torch.max(torch.abs(W))
        if max_abs_W > 1e-8:
            W = W / max_abs_W * 2.0
        else:
            W.fill_(0.0)
            
        # Only compute module positions and store data at specified timesteps
        if t in timesteps_to_store:
            # Update module positions based on current fields
            module_coords = sample_module_positions(Fx, Fy, W, num_modules, beta, gamma, X, Y)
            
            # Build connectivity graph
            weights = torch.zeros((num_modules, num_modules), device=device)
            
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
            
            # Store data at specified timesteps
            Module_coords_series.append(module_coords.clone().cpu().numpy())
            Graphs.append(weights.clone().cpu().numpy())
            Fx_series.append(Fx.clone().cpu().numpy())
            Fy_series.append(Fy.clone().cpu().numpy())
            W_series.append(W.clone().cpu().numpy())
    
    print(f"Full simulation {idx+1}/{total_simulations}: lam_W={current_lam_W:.3f}, D_F={D_F:.3f} - Field updates completed. Starting visualization...")
    
    # Move tensors to CPU for visualization
    X_np = X.cpu().numpy()
    Y_np = Y.cpu().numpy()
    x_np = x.cpu().numpy()
    y_np = y.cpu().numpy()
    
    # Visualization
    n_show = len(timesteps_to_store)
    row_labels = ["Flow Mag $|F|$", "Strength $W(x)$", "Flow Field $F(x)$", "Graph & Fields"]
    fig, axs = plt.subplots(len(row_labels), n_show, figsize=(4 * n_show, 13), constrained_layout=True)
    
    # Handle case where only one timestep is stored (reshape axes)
    if n_show == 1:
        axs = axs.reshape(-1, 1)
    
    print(f"Full simulation {idx+1}/{total_simulations}: Generating visualization for lam_W={current_lam_W:.3f}, D_F={D_F:.3f}...")
    
    for i, t_idx in enumerate(range(len(timesteps_to_store))):
        t = timesteps_to_store[t_idx]
        
        # Get fields at this timestep
        Fx_t = Fx_series[t_idx]
        Fy_t = Fy_series[t_idx]
        W_t = W_series[t_idx]
        F_mag_t = np.sqrt(Fx_t**2 + Fy_t**2)
        module_coords_t = Module_coords_series[t_idx]
        
        # Get graph at this timestep (except for t=0)
        if t_idx > 0:
            weights = Graphs[t_idx-1]
        else:
            weights = np.zeros((num_modules, num_modules))
        
        # Row 0: Flow Magnitude
        im0 = axs[0, i].imshow(F_mag_t, extent=[-1, 1, -1, 1], origin='lower', cmap='Blues', aspect='auto', vmin=0, vmax=1.0)
        axs[0, i].set_title(f"t={t}")
        axs[0, i].set_xticks([]); axs[0, i].set_yticks([])
        
        # Row 1: Synaptic Strength W
        im1 = axs[1, i].imshow(W_t, extent=[-1, 1, -1, 1], origin='lower', cmap='hot', aspect='auto', vmin=-2, vmax=2)
        axs[1, i].set_xticks([]); axs[1, i].set_yticks([])
        
        # Row 2: Flow Field Streamplot
        if np.any(Fx_t) or np.any(Fy_t):
            # Adjust streamplot density based on grid size to avoid overcrowding
            streamplot_density = 1.2 * (base_grid_size / grid_size)
            axs[2, i].streamplot(x_np, y_np, Fx_t, Fy_t, color=F_mag_t, cmap='Blues', 
                              density=streamplot_density, linewidth=0.8)
        axs[2, i].set_xlim(-1, 1); axs[2, i].set_ylim(-1, 1)
        axs[2, i].set_aspect('equal', adjustable='box')
        axs[2, i].set_xticks([]); axs[2, i].set_yticks([])
        
        # Row 3: Graph + Fields Overlay
        axs[3, i].imshow(W_t, extent=[-1, 1, -1, 1], origin='lower', cmap='hot', alpha=0.4, aspect='auto', vmin=-2, vmax=2)
        if np.any(Fx_t) or np.any(Fy_t):
            # Lighter streamplot in the background
            streamplot_density = 1.5 * (base_grid_size / grid_size)
            axs[3, i].streamplot(x_np, y_np, Fx_t, Fy_t, color='lightgray', 
                              density=streamplot_density, linewidth=0.5)
        axs[3, i].scatter(module_coords_t[:, 0], module_coords_t[:, 1], c='blue', s=50, edgecolors='black', zorder=3, label='Modules')
        
        for m in range(num_modules):
            for n in range(num_modules):
                if weights[m, n] > 0.01:
                    x0, y0 = module_coords_t[m]
                    x1, y1 = module_coords_t[n]
                    arrow_alpha = np.clip(weights[m, n] * 1.5, 0.1, 1.0)
                    arrow_width = np.clip(weights[m, n] * 1.5, 0.5, 1.5)
                    dx_arrow = (x1 - x0) * 0.85
                    dy_arrow = (y1 - y0) * 0.85
                    if abs(dx_arrow) > 1e-6 or abs(dy_arrow) > 1e-6:
                        axs[3, i].arrow(x0, y0, dx_arrow, dy_arrow, head_width=0.06, head_length=0.08, 
                                       length_includes_head=True, alpha=arrow_alpha, color='green', 
                                       linewidth=arrow_width, zorder=2)
        
        axs[3, i].set_xlim(-1, 1); axs[3, i].set_ylim(-1, 1)
        axs[3, i].set_aspect('equal', adjustable='box')
        axs[3, i].set_xticks([]); axs[3, i].set_yticks([])
    
    # Add row labels
    for row_idx, label in enumerate(row_labels):
        axs[row_idx, 0].set_ylabel(label, fontsize=14, labelpad=20)
        axs[row_idx, 0].yaxis.set_label_coords(-0.15, 0.5)
    
    # Add grid size and scaling info to the title
    scale_info = f"(grid: {grid_size}x{grid_size}, dt: {dt:.3f})" if use_parameter_scaling else ""
    
    # Save visualization
    filename = f"figures/simulation_lamW_{current_lam_W:.3f}_DF_{D_F:.3f}_etacoeff_{eta_coeff:.3f}_grid{grid_size}.png"
    plt.suptitle(f'Full Simulation: lam_W={current_lam_W:.3f}, D_F={D_F:.3f}, eta_coeff={eta_coeff:.3f} {scale_info}', 
                fontsize=16, y=0.99)
    plt.savefig(filename, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Full simulation {idx+1}/{total_simulations}: Saved visualization to {filename}")
    print(f"--- Completed full simulation {idx+1}/{total_simulations}: lam_W={current_lam_W:.3f}, D_F={D_F:.3f} ---")
    
    # Force cleanup
    torch.cuda.empty_cache()
    gc.collect()
    
    return idx

def main():
    global_start_time = time.time()
    print(f"Starting GVFT simulation with grid size {grid_size}x{grid_size} on {device}")
    
    # Print all scaling-related parameters for reference
    if use_parameter_scaling:
        print("\nScaled parameters for current grid size:")
        print(f"  D_W: {D_W:.6f} (from base: {base_D_W:.6f})")
        print(f"  D_eta: {D_eta:.6f} (from base: {base_D_eta:.6f})")
        print(f"  dt: {dt:.6f} (from base: {base_dt:.6f})")
        print(f"  timesteps_sweep: {timesteps_sweep} (from base: {base_timesteps_sweep})")
        print(f"  noise_F: {noise_F:.6f} (from base: {base_noise_F:.6f})")
        print(f"  noise_W: {noise_W:.6f} (from base: {base_noise_W:.6f})")
        print(f"  D_F values: [{D_F_values[0].item():.6f} to {D_F_values[-1].item():.6f}]")
    
    # Determine optimal batch size
    batch_size = optimize_batch_size()
    print(f"\nUsing batch size: {batch_size} for grid size {grid_size}x{grid_size}")

    # --- 2D PARAMETER SWEEP WITH GPU BATCHING ---
    print(f"\nStarting 2D Parameter Sweep (lam_W vs D_F) with GPU acceleration...")

    param_combinations = [
        (torch.tensor(j, device=device), torch.tensor(i, device=device))
        for j in range(len(lam_W_values)) for i in range(len(D_F_values))
    ]

    # Create batches
    batches = [param_combinations[i:i + batch_size] for i in range(0, len(param_combinations), batch_size)]
    total_batches = len(batches)
    total_combinations = len(param_combinations)

    print(f"Total parameter combinations: {total_combinations}")
    print(f"Created {total_batches} batches with up to {batch_size} simulations per batch")

    # Initialize results tensor
    pattern_intensity_summary_2d = torch.zeros(len(lam_W_values), len(D_F_values), device=device)
    
    # Process batches with progress tracking
    completed_sims = 0
    for batch_idx, param_batch in enumerate(batches):
        batch_start_time = time.time()
        print(f"\nProcessing batch {batch_idx + 1}/{total_batches} with {len(param_batch)} simulations...")
        
        # Run batch
        batch_results = run_sweep_simulation(param_batch)
        
        # Store results
        for lam_W_idx, D_F_idx, hotspot_fraction in batch_results:
            pattern_intensity_summary_2d[lam_W_idx, D_F_idx] = hotspot_fraction
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
        
        # Save intermediate results every 2 batches or at the end
        if (batch_idx + 1) % 2 == 0 or batch_idx == total_batches - 1:
            checkpoint_data = pattern_intensity_summary_2d.cpu().numpy()
            checkpoint_file = f"figures/phase_diagram_grid{grid_size}_checkpoint_{batch_idx+1}.npy"
            np.save(checkpoint_file, checkpoint_data)
            print(f"Saved checkpoint to {checkpoint_file}")
            
            # Create and save intermediate visualization every 5 batches or at the end
            if (batch_idx + 1) % 5 == 0 or batch_idx == total_batches - 1:
                plt.figure(figsize=(10, 8))
                sns.heatmap(checkpoint_data, cmap="viridis", linewidths=0.0,
                           cbar_kws={"label": "Hotspot Fraction"})
                plt.title(f"GVFT Phase Diagram - Intermediate ({completed_sims}/{total_combinations})")
                plt.tight_layout()
                plt.savefig(f"figures/phase_diagram_grid{grid_size}_intermediate_{batch_idx+1}.png", dpi=150)
                plt.close()
                print(f"Saved intermediate visualization")
        
        # Force memory cleanup
        torch.cuda.empty_cache()
        gc.collect()

    # Get final results
    pattern_intensity_summary_2d_cpu = pattern_intensity_summary_2d.cpu().numpy()

    # --- SAVE 2D PHASE DIAGRAM ---
    df_2d = pd.DataFrame(
        pattern_intensity_summary_2d_cpu,
        index=[f"{v:.3f}" for v in lam_W_values.cpu().numpy()],
        columns=[f"{v:.3f}" for v in D_F_values.cpu().numpy()]
    )

    plt.figure(figsize=(12, 9))
    sns.heatmap(df_2d, cmap="viridis", linewidths=0.5,
                cbar_kws={"label": "Fraction of W(x) Hotspots (Threshold=1.5)"})
    # Include scaling info in title
    scaling_info = f" (grid: {grid_size}x{grid_size}, dt: {dt:.3f})" if use_parameter_scaling else ""
    plt.title(f"GVFT 2D Phase Diagram (lam_eta={lam_eta:.2f}, eta_coeff={eta_coeff:.3f}){scaling_info}", fontsize=16)
    plt.xlabel("D_F (Flow Field Diffusion)", fontsize=12)
    plt.ylabel("lam_W (W Decay Rate)", fontsize=12)
    plt.tight_layout()
    plt.savefig(f"figures/phase_diagram_2D_lamW_DF_grid{grid_size}.png", dpi=200)
    plt.close()

    print(f"Saved 2D phase diagram to figures/phase_diagram_2D_lamW_DF_grid{grid_size}.png")

    # --- SELECT PARAMETERS FOR FULL SIMULATION ---
    print("\nSelecting interesting parameter regimes for detailed simulations...")
    
    # Method 1: Find row with maximum variance (most interesting transitions)
    row_std_dev = np.std(pattern_intensity_summary_2d_cpu, axis=1)
    chosen_lam_W_index = np.argmax(row_std_dev)
    chosen_lam_W = lam_W_values[chosen_lam_W_index].item()

    selected_row_values = pattern_intensity_summary_2d_cpu[chosen_lam_W_index]
    sorted_D_F_indices = np.argsort(selected_row_values)

    # Take low, medium, and high values from the most varying row
    low_idx = sorted_D_F_indices[0]
    mid_idx = sorted_D_F_indices[len(sorted_D_F_indices) // 2]
    high_idx = sorted_D_F_indices[-1]

    selected_D_F_indices = sorted(set([low_idx, mid_idx, high_idx]))
    global selected_params_full
    selected_params_full = [(chosen_lam_W, D_F_values[idx].item()) for idx in selected_D_F_indices]
    
    # Method 2: Also find specific parameter pairs where hotspot fraction is around 0.5
    # These are likely interesting transition regions
    hotspot_diff = np.abs(pattern_intensity_summary_2d_cpu - 0.5)
    closest_pairs = np.unravel_index(np.argsort(hotspot_diff.ravel())[:3], hotspot_diff.shape)
    
    for lam_idx, d_idx in zip(closest_pairs[0], closest_pairs[1]):
        pair = (lam_W_values[lam_idx].item(), D_F_values[d_idx].item())
        if pair not in selected_params_full:
            selected_params_full.append(pair)
    
    # Limit to a reasonable number of simulations
    max_full_sims = 6
    if len(selected_params_full) > max_full_sims:
        selected_params_full = selected_params_full[:max_full_sims]

    print(f"Selected {len(selected_params_full)} parameter pairs for full simulation: {selected_params_full}")

    # --- RUN FULL SIMULATIONS ---
    print("\nRunning full simulations for detailed analysis...")
    
    full_simulation_params = [
        (idx, lam_W, D_F) for idx, (lam_W, D_F) in enumerate(selected_params_full)
    ]

    for sim_params in full_simulation_params:
        run_full_simulation(sim_params)

    # Report total runtime
    total_time = (time.time() - global_start_time) / 60
    print(f"\nAll simulations completed in {total_time:.2f} minutes.")
    
    # Final summary with scaling information
    if use_parameter_scaling:
        print("\nSimulation used parameter scaling for grid consistency:")
        print(f"  Base grid: {base_grid_size}x{base_grid_size}, Actual grid: {grid_size}x{grid_size}")
        print(f"  Scaling factor: {scale_factor:.3f}")
        print("\nScaled parameters:")
        print(f"  D_W: {D_W:.6f} (from {base_D_W:.6f})")
        print(f"  D_F: scaled range [{D_F_values[0].item():.6f} to {D_F_values[-1].item():.6f}]")
        print(f"  dt: {dt:.6f} (from {base_dt:.6f})")
        print(f"  timesteps: {timesteps_sweep}/{timesteps_sim} (from {base_timesteps_sweep}/{base_timesteps_sim})")


if __name__ == "__main__":
    main()