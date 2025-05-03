import torch

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
        max_cg_iterations = min(100, 20 * grid_size // 200)
        
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
            tol = 1e-10 * (200 / grid_size)
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

def solve_diffusion_equation(field, A_inv_B, A_inv, source_term, dt):
    """Solve diffusion equation using pre-computed operators"""
    # Apply A⁻¹ * B to the field
    field_next = A_inv_B(field)
    
    # Add source term: field_next += A⁻¹ * dt * source
    field_next = field_next + A_inv(dt * source_term)
    
    return field_next