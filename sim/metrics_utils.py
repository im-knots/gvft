import torch
import numpy as np
import scipy.ndimage as ndimage

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