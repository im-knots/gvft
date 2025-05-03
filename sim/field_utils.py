import torch
import torch.nn.functional as F
import numpy as np
import os
import glob

def build_domain(grid_size, device):
    """Set up the spatial domain for the simulation."""
    x = torch.linspace(-1, 1, grid_size).to(device)
    y = torch.linspace(-1, 1, grid_size).to(device)
    Y, X = torch.meshgrid(y, x, indexing='ij')
    dx = x[1] - x[0]
    dy = y[1] - y[0]
    
    return {
        'x': x,
        'y': y,
        'X': X,
        'Y': Y,
        'dx': dx,
        'dy': dy
    }

def build_laplacian_operator(grid_size, dx, dy, device):
    """Build a function that applies the Laplacian operator using PyTorch's conv2d."""
    # Create Laplacian kernel
    laplacian_kernel = torch.zeros((1, 1, 3, 3), device=device)
    laplacian_kernel[0, 0, 0, 1] = 1.0 / dy**2
    laplacian_kernel[0, 0, 1, 0] = 1.0 / dx**2
    laplacian_kernel[0, 0, 1, 2] = 1.0 / dx**2
    laplacian_kernel[0, 0, 2, 1] = 1.0 / dy**2
    laplacian_kernel[0, 0, 1, 1] = -2.0 * (1.0 / dx**2 + 1.0 / dy**2)
    
    def apply_laplacian(tensor):
        """Apply the Laplacian operator to a 2D tensor."""
        # Add batch and channel dimensions, pad, then convolve
        padded = F.pad(tensor.unsqueeze(0).unsqueeze(0), (1, 1, 1, 1), mode='replicate')
        return F.conv2d(padded, laplacian_kernel).squeeze()
    
    return apply_laplacian

def gaussian_filter(tensor, sigma=1.0):
    """Apply Gaussian filter to a tensor using PyTorch."""
    # Calculate kernel size based on sigma (3*sigma rule of thumb)
    kernel_size = int(2 * 3 * sigma + 1)
    kernel_size = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
    
    # Create Gaussian kernel
    device = tensor.device
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

def load_neuroml_fields(input_dir, basename, grid_size, device):
    """Load pre-processed GVFT fields derived from NeuroML data."""
    # Find relevant field files
    flow_x_pattern = os.path.join(input_dir, f"{basename}*_flow_x.npy")
    flow_y_pattern = os.path.join(input_dir, f"{basename}*_flow_y.npy")
    strength_pattern = os.path.join(input_dir, f"{basename}*_strength.npy")
    neuromod_pattern = os.path.join(input_dir, f"{basename}*_neuromod.npy")
    
    flow_x_files = glob.glob(flow_x_pattern)
    flow_y_files = glob.glob(flow_y_pattern)
    strength_files = glob.glob(strength_pattern)
    neuromod_files = glob.glob(neuromod_pattern)
    
    # Check if all required files exist
    if not (flow_x_files and flow_y_files and strength_files and neuromod_files):
        raise FileNotFoundError(f"Could not find all required field files in {input_dir} with basename {basename}")
    
    # Load the fields
    Fx_prior = np.load(flow_x_files[0])
    Fy_prior = np.load(flow_y_files[0])
    W_prior = np.load(strength_files[0])
    eta_prior = np.load(neuromod_files[0])
    
    # Check original dimensions
    orig_shape = Fx_prior.shape
    print(f"Loaded biological fields with shape {orig_shape}")
    
    # Resize if necessary
    if orig_shape[0] != grid_size:
        from scipy.ndimage import zoom
        
        scale_factor = grid_size / orig_shape[0]
        print(f"Resizing fields from {orig_shape} to {grid_size}x{grid_size}")
        
        Fx_prior = zoom(Fx_prior, scale_factor, order=1)
        Fy_prior = zoom(Fy_prior, scale_factor, order=1)
        W_prior = zoom(W_prior, scale_factor, order=1)
        eta_prior = zoom(eta_prior, scale_factor, order=1)
    
    # Normalize fields
    F_mag = np.sqrt(Fx_prior**2 + Fy_prior**2)
    max_mag = np.max(F_mag)
    if max_mag > 1e-6:
        Fx_prior = Fx_prior / max_mag
        Fy_prior = Fy_prior / max_mag
    
    max_W = np.max(np.abs(W_prior))
    if max_W > 1e-6:
        W_prior = W_prior / max_W * 2.0
    
    max_eta = np.max(np.abs(eta_prior))
    if max_eta > 1e-6:
        eta_prior = eta_prior / max_eta
    
    # Convert to PyTorch tensors on the specified device
    Fx_init = torch.tensor(Fx_prior, dtype=torch.float32, device=device)
    Fy_init = torch.tensor(Fy_prior, dtype=torch.float32, device=device)
    W_init = torch.tensor(W_prior, dtype=torch.float32, device=device)
    eta_init = torch.tensor(eta_prior, dtype=torch.float32, device=device)
    
    print(f"Field statistics after normalization:")
    print(f"  Flow: min={Fx_init.min().item():.3f}/{Fy_init.min().item():.3f}, max={Fx_init.max().item():.3f}/{Fy_init.max().item():.3f}")
    print(f"  Strength (W): min={W_init.min().item():.3f}, max={W_init.max().item():.3f}")
    print(f"  Neuromodulatory (eta): min={eta_init.min().item():.3f}, max={eta_init.max().item():.3f}")
    
    return Fx_init, Fy_init, W_init, eta_init

def initialize_fields(grid_size, device, filter_scale=1.0):
    """Initialize GVFT fields."""
    # Generate random potential field
    potential = gaussian_filter(torch.randn(grid_size, grid_size, device=device), sigma=5*filter_scale)
    
    # Calculate initial flow field from gradient of potential
    Fy_init, Fx_init = torch.gradient(potential)
    F_mag_init = torch.sqrt(Fx_init**2 + Fy_init**2)
    mask = F_mag_init > 1e-6
    Fx_init[mask] = Fx_init[mask] / F_mag_init[mask]
    Fy_init[mask] = Fy_init[mask] / F_mag_init[mask]
    
    # Initialize W and eta fields
    W_init = gaussian_filter(torch.rand(grid_size, grid_size, device=device) - 0.5, sigma=3*filter_scale) * 0.1
    eta_init = gaussian_filter(torch.rand(grid_size, grid_size, device=device) - 0.5, sigma=5*filter_scale) * 0.1
    
    return Fx_init, Fy_init, W_init, eta_init

def normalize_field(Fx, Fy, max_mag=1.0):
    """Normalize vector field to have magnitude <= max_mag."""
    F_mag = torch.sqrt(Fx**2 + Fy**2)
    mask = F_mag > 1e-6
    scale = torch.ones_like(F_mag)
    scale[mask] = torch.clamp(max_mag / F_mag[mask], max=1.0)
    return Fx * scale, Fy * scale

def clip_field(field, min_val=-10.0, max_val=10.0):
    """Clip field values to stay within bounds."""
    return torch.clamp(field, min_val, max_val)

def count_hotspots(W, threshold=1.5):
    """Count fraction of grid points where W exceeds threshold."""
    return torch.sum(W > threshold).float() / (W.shape[0] * W.shape[1])