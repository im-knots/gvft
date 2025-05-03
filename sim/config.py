import torch
import numpy as np

# --- BASE PARAMETERS ---
# Reference grid size for parameter scaling
BASE_GRID_SIZE = 500

# Base time parameters
BASE_TIMESTEPS_SWEEP = 50
BASE_TIMESTEPS_SIM = 50
BASE_DT = 0.1

# Default module parameters
DEFAULT_NUM_MODULES = 20
DEFAULT_TOP_K = 5
DEFAULT_COS_THRESHOLD = 0.2
DEFAULT_LAMBDA_VAL = 0.2

# --- GVFT Field Parameters (Base values) ---
# These will be scaled according to grid resolution
BASE_D_W = 0.03    
BASE_LAM_F = 0.007
BASE_BETA = 0.5
BASE_GAMMA = 0.5
BASE_ALPHA = 1.5
BASE_BETA_COUPLING = 0.1
BASE_D_ETA = 0.02   
BASE_LAM_ETA = 0.1
BASE_ETA_COEFF = 0.05
BASE_NOISE_F = 0.0001
BASE_NOISE_W = 0.001

class GVFTConfig:
    """Configuration class for GVFT simulations."""
    
    def __init__(self, grid_size=BASE_GRID_SIZE, use_parameter_scaling=True):
        # Store basic config
        self.grid_size = grid_size
        self.use_parameter_scaling = use_parameter_scaling
        
        # Determine device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Apply parameter scaling
        self._apply_scaling()
        
        # Parameter sweep ranges
        self._setup_parameter_ranges()
        
        # Derived parameters based on scaling
        self.view_step = max(1, self.timesteps_sim // 10)
        
        # Module parameters
        self.num_modules = DEFAULT_NUM_MODULES
        self.top_k = DEFAULT_TOP_K
        self.cos_threshold = DEFAULT_COS_THRESHOLD
        self.lambda_val = DEFAULT_LAMBDA_VAL
    
    def _apply_scaling(self):
        """Apply scaling to parameters based on grid resolution if enabled."""
        # Base parameters (not scaled)
        self.dt = BASE_DT
        self.timesteps_sweep = BASE_TIMESTEPS_SWEEP
        self.timesteps_sim = BASE_TIMESTEPS_SIM
        self.D_W = BASE_D_W
        self.lam_F = BASE_LAM_F
        self.beta = BASE_BETA
        self.gamma = BASE_GAMMA
        self.alpha = BASE_ALPHA
        self.beta_coupling = BASE_BETA_COUPLING
        self.D_eta = BASE_D_ETA
        self.lam_eta = BASE_LAM_ETA
        self.eta_coeff = BASE_ETA_COEFF
        self.noise_F = BASE_NOISE_F
        self.noise_W = BASE_NOISE_W
        
        if self.use_parameter_scaling:
            # Scale factor based on grid resolution (squared relationship for diffusion)
            self.scale_factor = (BASE_GRID_SIZE / self.grid_size) ** 2
            
            # Scale timestep and increase iterations to maintain same simulation time
            self.dt = BASE_DT * self.scale_factor
            self.timesteps_sweep = int(BASE_TIMESTEPS_SWEEP / self.scale_factor)
            self.timesteps_sim = int(BASE_TIMESTEPS_SIM / self.scale_factor)
            
            # Ensure minimum number of timesteps
            self.timesteps_sweep = max(self.timesteps_sweep, 50)
            self.timesteps_sim = max(self.timesteps_sim, 100)
            
            # Diffusion coefficients scale with dx² (inverse to grid_size²)
            self.D_W = BASE_D_W * self.scale_factor
            self.D_eta = BASE_D_ETA * self.scale_factor
            
            # These parameters affect field coupling, scale with grid size
            self.beta_coupling = BASE_BETA_COUPLING * self.scale_factor
            
            # Noise terms should scale with grid resolution too
            self.noise_F = BASE_NOISE_F / np.sqrt(self.scale_factor)
            self.noise_W = BASE_NOISE_W / np.sqrt(self.scale_factor)
    
    def _setup_parameter_ranges(self):
        """Set up parameter ranges for sweep."""
        # Base parameter ranges
        D_F_base = torch.linspace(0.002, 0.007, 10)
        lam_W_base = torch.linspace(0.1, 0.5, 10)
        
        # Apply scaling if needed
        if self.use_parameter_scaling:
            self.D_F_values = D_F_base * self.scale_factor
        else:
            self.D_F_values = D_F_base
            
        self.lam_W_values = lam_W_base
        
        # Move to device
        self.D_F_values = self.D_F_values.to(self.device)
        self.lam_W_values = self.lam_W_values.to(self.device)
    
    def print_config(self):
        """Print the current configuration."""
        print(f"\nGVFT Configuration Summary:")
        print(f"  Grid size: {self.grid_size}x{self.grid_size}")
        print(f"  Device: {self.device}")
        print(f"  Parameter scaling: {self.use_parameter_scaling}")
        
        if self.use_parameter_scaling:
            print(f"  Scale factor: {self.scale_factor:.3f}")
        
        print(f"\nTime parameters:")
        print(f"  dt: {self.dt:.6f}")
        print(f"  Timesteps (sweep): {self.timesteps_sweep}")
        print(f"  Timesteps (simulation): {self.timesteps_sim}")
        
        print(f"\nField parameters:")
        print(f"  D_W: {self.D_W:.6f}")
        print(f"  D_eta: {self.D_eta:.6f}")
        print(f"  lam_F: {self.lam_F:.6f}")
        print(f"  lam_eta: {self.lam_eta:.6f}")
        print(f"  alpha: {self.alpha:.3f}")
        print(f"  beta: {self.beta:.3f}")
        print(f"  gamma: {self.gamma:.3f}")
        print(f"  beta_coupling: {self.beta_coupling:.6f}")
        print(f"  eta_coeff: {self.eta_coeff:.3f}")
        print(f"  noise_F: {self.noise_F:.6f}")
        print(f"  noise_W: {self.noise_W:.6f}")
        
        print(f"\nSweep parameters:")
        print(f"  D_F range: {self.D_F_values[0].item():.6f} to {self.D_F_values[-1].item():.6f}")
        print(f"  lam_W range: {self.lam_W_values[0].item():.3f} to {self.lam_W_values[-1].item():.3f}")
        
        print(f"\nModule parameters:")
        print(f"  Number of modules: {self.num_modules}")
        print(f"  Top-k connections: {self.top_k}")
        print(f"  Cosine threshold: {self.cos_threshold}")
        print(f"  Lambda value: {self.lambda_val}")