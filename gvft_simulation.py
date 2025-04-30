import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter, laplace

# === Parameters ===
grid_size = 100
timesteps = 10
num_modules = 10
top_k = 3
cos_threshold = 0.3
lambda_val = 0.2
kappa = 0.3

# === Setup Domain ===
np.random.seed(42)
x = np.linspace(-1, 1, grid_size)
y = np.linspace(-1, 1, grid_size)
X, Y = np.meshgrid(x, y)
dx = x[1] - x[0]
dy = y[1] - y[0]

# === Fixed Module Positions ===
indices = np.random.choice(grid_size * grid_size, size=num_modules, replace=False)
yi, xi = np.unravel_index(indices, (grid_size, grid_size))
module_coords = np.column_stack([X[yi, xi], Y[yi, xi]])

# === Evolve Field Function ===
def evolve_field(field, D, lam, steps):
    series = [field.copy()]
    for _ in range(steps - 1):
        lap = laplace(field)
        field += D * lap - lam * field
        field = np.clip(field, 0, 1)
        series.append(field.copy())
    return series

# === Seed Initial Fields ===
potential = gaussian_filter(np.random.randn(grid_size, grid_size), sigma=8)
Fy_0, Fx_0 = np.gradient(potential, dy, dx)
Fx_0 = gaussian_filter(Fx_0, sigma=1)
Fy_0 = gaussian_filter(Fy_0, sigma=1)

F_mag_0 = np.sqrt(Fx_0**2 + Fy_0**2)
W_0 = 0.6 * 1 / (1 + np.exp(-F_mag_0 * 5)) + 0.4 * gaussian_filter(np.random.rand(grid_size, grid_size), sigma=4)
W_0 = (W_0 - W_0.min()) / (W_0.max() - W_0.min())

# === Evolve Each Field Separately ===
Fx_series = evolve_field(Fx_0, D=0.005, lam=0.02, steps=timesteps)
Fy_series = evolve_field(Fy_0, D=0.015, lam=0.01, steps=timesteps)
W_series  = evolve_field(W_0,  D=0.01,  lam=0.01, steps=timesteps)

# === Instantiate Graphs from Fields ===
graph_series = []
for t in range(timesteps):
    Fx_t, Fy_t, W_t = Fx_series[t], Fy_series[t], W_series[t]
    weights = np.zeros((num_modules, num_modules))

    for i in range(num_modules):
        xi_idx = np.argmin(np.abs(x - module_coords[i, 0]))
        yi_idx = np.argmin(np.abs(y - module_coords[i, 1]))
        F_i = np.array([Fx_t[yi_idx, xi_idx], Fy_t[yi_idx, xi_idx]])
        W_i = W_t[yi_idx, xi_idx]

        candidate_edges = []
        for j in range(num_modules):
            if i == j:
                continue
            r_i = module_coords[i]
            r_j = module_coords[j]
            delta = r_j - r_i
            norm_delta = np.linalg.norm(delta)
            norm_F = np.linalg.norm(F_i)
            if norm_delta == 0 or norm_F == 0:
                continue
            cos_sim = np.dot(delta, F_i) / (norm_delta * norm_F)
            if cos_sim < cos_threshold:
                continue
            rho = cos_sim * W_i
            w_ij = np.clip(rho / lambda_val, 0, 1.0)
            candidate_edges.append((j, w_ij))

        candidate_edges = sorted(candidate_edges, key=lambda x: x[1], reverse=True)[:top_k]
        for j, w_ij in candidate_edges:
            weights[i, j] = w_ij

    graph_series.append(weights.copy())

# === Visualization ===
fig, axs = plt.subplots(4, timesteps, figsize=(4 * timesteps, 12))
for t in range(timesteps):
    F_mag_t = np.sqrt(Fx_series[t]**2 + Fy_series[t]**2)
    axs[0, t].imshow(F_mag_t, extent=[-1, 1, -1, 1], origin='lower', cmap='Blues')
    axs[0, t].set_title(f"Flow Mag t={t}")
    axs[0, t].axis('off')

    axs[1, t].imshow(W_series[t], extent=[-1, 1, -1, 1], origin='lower', cmap='hot')
    axs[1, t].set_title(f"W(x) t={t}")
    axs[1, t].axis('off')

    axs[2, t].quiver(X, Y, Fx_series[t], Fy_series[t], color='gray', alpha=0.5)
    axs[2, t].set_title(f"Flow Field t={t}")
    axs[2, t].axis('off')

    axs[3, t].imshow(W_series[t], extent=[-1, 1, -1, 1], origin='lower', cmap='hot', alpha=0.3)
    axs[3, t].quiver(X, Y, Fx_series[t], Fy_series[t], color='lightgray', alpha=0.3)
    axs[3, t].scatter(module_coords[:, 0], module_coords[:, 1], c='blue', s=40, zorder=3)
    for i in range(num_modules):
        for j in range(num_modules):
            if graph_series[t][i, j] > 0:
                x0, y0 = module_coords[i]
                x1, y1 = module_coords[j]
                alpha = min(1.0, graph_series[t][i, j])
                axs[3, t].arrow(x0, y0, x1 - x0, y1 - y0,
                                head_width=0.01, length_includes_head=True,
                                alpha=alpha, color='green', linewidth=0.8, zorder=2)
    axs[3, t].set_title(f"Graph t={t}")
    axs[3, t].axis('off')

plt.tight_layout()
plt.show()
