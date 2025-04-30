import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import laplace, gaussian_filter

# --- PARAMETERS ---
grid_size = 100
timesteps = 1000
view_step = 100
num_modules = 10
top_k = 2
cos_threshold = 0.2
lambda_val = 0.2
kappa = 0.3

beta_W = 0.95
gamma_F = 0.2

D_F = 0.03
D_W = 0.04
D_eta = 0.02

lam_F = 0.005
lam_W = 0.03
lam_eta = 0.01

# --- DOMAIN SETUP ---
np.random.seed(42)
x = np.linspace(-1, 1, grid_size)
y = np.linspace(-1, 1, grid_size)
X, Y = np.meshgrid(x, y)

# --- MODULE LOCATIONS ---
indices = np.random.choice(grid_size * grid_size, size=num_modules, replace=False)
yi, xi = np.unravel_index(indices, (grid_size, grid_size))
module_coords = np.column_stack([X[yi, xi], Y[yi, xi]])

# --- SOURCE FIELD FOR W ---
def source_field(X, Y, centers, amplitude=1.0, sigma=0.1):
    S = np.zeros_like(X)
    for (cx, cy) in centers:
        S += amplitude * np.exp(-((X - cx)**2 + (Y - cy)**2) / (2 * sigma**2))
    return S

S_W = source_field(X, Y, [(-0.5, -0.5), (0.5, 0.5)], amplitude=0.2, sigma=0.2)

# --- INITIAL FIELD SEEDING ---
potential = gaussian_filter(np.random.randn(grid_size, grid_size), sigma=10)
Fy_0, Fx_0 = np.gradient(potential, y[1] - y[0], x[1] - x[0])
F_mag_0 = np.sqrt(Fx_0**2 + Fy_0**2)

W_0 = 0.6 * 1 / (1 + np.exp(-F_mag_0 * 5)) + 0.4 * gaussian_filter(np.random.rand(grid_size, grid_size), sigma=4)
W_0 = (W_0 - W_0.min()) / (W_0.max() - W_0.min())

eta_0 = np.zeros_like(W_0)

# --- EVOLUTION ---
Fx_series, Fy_series, W_series, eta_series, Graphs = [Fx_0.copy()], [Fy_0.copy()], [W_0.copy()], [eta_0.copy()], []

for t in range(timesteps):
    Fx, Fy, W, eta = Fx_series[-1], Fy_series[-1], W_series[-1], eta_series[-1]

    F_mag = np.sqrt(Fx**2 + Fy**2)
    F_mag_norm = F_mag / (np.max(F_mag) + 1e-5)

    # Dynamic neuromodulation source based on local energy
    S_eta = (F_mag_norm**2 + W**2) / 2.0

    W_interact = beta_W * 1 / (1 + np.exp(-F_mag_norm * 5)) + (1 - beta_W) * W
    F_scale = 1 + gamma_F * (W - 0.5)

    Fx_next = Fx + D_F * laplace(Fx) - lam_F * Fx
    Fy_next = Fy + D_F * laplace(Fy) - lam_F * Fy
    Fx_next *= F_scale
    Fy_next *= F_scale

    eta_next = eta + D_eta * laplace(eta) - lam_eta * eta + S_eta
    W_next = W + D_W * laplace(W) - lam_W * W + S_W + eta
    W_next = (W_next + W_interact) / 2.0
    W_next = (W_next - W_next.min()) / (W_next.max() - W_next.min())

    Fx_series.append(Fx_next)
    Fy_series.append(Fy_next)
    W_series.append(W_next)
    eta_series.append(eta_next)

    # === CONNECTION INSTANTIATION ===
    weights = np.zeros((num_modules, num_modules))
    for i in range(num_modules):
        xi_idx = np.argmin(np.abs(x - module_coords[i, 0]))
        yi_idx = np.argmin(np.abs(y - module_coords[i, 1]))
        F_i = np.array([Fx_next[yi_idx, xi_idx], Fy_next[yi_idx, xi_idx]])
        W_i = W_next[yi_idx, xi_idx]

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

    Graphs.append(weights.copy())

# --- VISUALIZATION ---
timesteps_to_show = list(range(0, timesteps, view_step))
n_show = len(timesteps_to_show)
row_labels = ["Flow Mag", "W(x)", "η(x, t)", "Flow Field", "Graph"]

fig, axs = plt.subplots(len(row_labels), n_show, figsize=(4 * n_show, 16))

# Plot fields
for i, t in enumerate(timesteps_to_show):
    Fx_t = Fx_series[t]
    Fy_t = Fy_series[t]
    W_t = W_series[t]
    eta_t = eta_series[t]
    F_mag_t = np.sqrt(Fx_t**2 + Fy_t**2)
    weights = Graphs[t]

    axs[0, i].imshow(F_mag_t, extent=[-1, 1, -1, 1], origin='lower', cmap='Blues')
    axs[0, i].set_title(f"t={t}")
    axs[0, i].axis('off')

    axs[1, i].imshow(W_t, extent=[-1, 1, -1, 1], origin='lower', cmap='hot')
    axs[1, i].axis('off')

    axs[2, i].imshow(eta_t, extent=[-1, 1, -1, 1], origin='lower', cmap='PuRd')
    axs[2, i].axis('off')

    axs[3, i].streamplot(x, y, Fx_t, Fy_t, color='gray', density=1)
    axs[3, i].axis('off')

    axs[4, i].imshow(W_t, extent=[-1, 1, -1, 1], origin='lower', cmap='hot', alpha=0.3)
    axs[4, i].streamplot(x, y, Fx_t, Fy_t, color='lightgray', density=1.5)
    axs[4, i].scatter(module_coords[:, 0], module_coords[:, 1], c='blue', s=40, zorder=3)
    for m in range(num_modules):
        for n in range(num_modules):
            if weights[m, n] > 0:
                x0, y0 = module_coords[m]
                x1, y1 = module_coords[n]
                alpha = min(1.0, weights[m, n])
                axs[4, i].arrow(x0, y0, x1 - x0, y1 - y0,
                                head_width=0.01, length_includes_head=True,
                                alpha=alpha, color='green', linewidth=0.8, zorder=2)
    axs[4, i].axis('off')

# Manually position row labels using figure coordinates
# Calculate vertical centers using axes positions
fig.subplots_adjust(left=0.2, right=0.98, top=0.95, bottom=0.05, wspace=0.1, hspace=0.1)

for row_idx, label in enumerate(row_labels):
    pos = axs[row_idx, 0].get_position()
    y_center = (pos.y0 + pos.y1) / 2
    fig.text(0.13, y_center, label, fontsize=14, ha='right', va='center')

plt.show()


