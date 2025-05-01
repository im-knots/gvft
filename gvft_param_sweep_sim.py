import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy.ndimage import laplace, gaussian_filter
from scipy import sparse
from scipy.sparse.linalg import spsolve
from multiprocessing import Pool, cpu_count

# --- ENSURE FIGURE DIRECTORY ---
os.makedirs("figures", exist_ok=True)

# --- PARAMETERS ---
grid_size = 100
timesteps_sweep = 200
timesteps_sim = 500
view_step = 50
num_modules = 10
top_k = 2
cos_threshold = 0.2
lambda_val = 0.2
dt = 1.0

# --- GVFT Field Parameters ---
D_W = 0.03
lam_F = 0.007
beta = 0.5
gamma = 0.5
alpha = 0.8
beta_coupling = 0.1
D_eta = 0.02
lam_eta = 0.1
eta_coeff = 0.05
noise_F = 0.0001
noise_W = 0.001

# --- Define Parameter Sweep Ranges ---
D_F_values = np.linspace(0.005, 0.08, 50)
lam_W_values = np.linspace(0.01, 0.30, 50)

# --- DOMAIN SETUP ---
np.random.seed(42)
x = np.linspace(-1, 1, grid_size)
y = np.linspace(-1, 1, grid_size)
X, Y = np.meshgrid(x, y)
dx = x[1] - x[0]
dy = y[1] - y[0]

# --- Build the Laplacian Matrix ---
def build_laplacian_matrix(grid_size, dx, dy):
    N = grid_size * grid_size
    diagonals = []
    offsets = []

    main_diag = -2 * (1/dx**2 + 1/dy**2) * np.ones(N)
    diagonals.append(main_diag)
    offsets.append(0)

    x_diag = (1/dx**2) * np.ones(N - 1)
    x_diag[grid_size-1::grid_size] = 0
    diagonals.extend([x_diag, x_diag])
    offsets.extend([1, -1])

    y_diag = (1/dy**2) * np.ones(N - grid_size)
    diagonals.extend([y_diag, y_diag])
    offsets.extend([grid_size, -grid_size])

    L = sparse.diags(diagonals, offsets, shape=(N, N), format='csr')
    return L

L = build_laplacian_matrix(grid_size, dx, dy)

# --- Precompute Matrices for eta (constant across all simulations) ---
A_eta = sparse.eye(grid_size * grid_size) - (dt/2) * (D_eta * L - lam_eta * sparse.eye(grid_size * grid_size))
B_eta = sparse.eye(grid_size * grid_size) + (dt/2) * (D_eta * L - lam_eta * sparse.eye(grid_size * grid_size))

# --- FUNCTIONS ---
def compute_module_probability(Fx, Fy, W, beta, gamma):
    F_mag = np.sqrt(Fx**2 + Fy**2)
    logits = np.clip(beta * F_mag + gamma * W, -50, 50)
    prob = 1 / (1 + np.exp(-logits))
    prob_sum = prob.sum()
    if prob_sum <= 1e-10:
        prob.fill(1.0 / prob.size)
    else:
        prob = prob / prob_sum
    return prob

def sample_module_positions(Fx, Fy, W, num_modules, beta, gamma, X, Y):
    prob = compute_module_probability(Fx, Fy, W, beta, gamma)
    flat_prob = prob.flatten()
    flat_prob = flat_prob / (flat_prob.sum() + 1e-10)
    try:
        indices = np.random.choice(grid_size * grid_size, size=num_modules, replace=False, p=flat_prob)
    except ValueError as e:
        print(f"Error sampling module positions: {e}, Sum(p)={flat_prob.sum()}. Using uniform sampling.")
        indices = np.random.choice(grid_size * grid_size, size=num_modules, replace=False)
    yi, xi = np.unravel_index(indices, (grid_size, grid_size))
    return np.column_stack([X[yi, xi], Y[yi, xi]])

def normalize_field(Fx, Fy, max_mag=1.0):
    F_mag = np.sqrt(Fx**2 + Fy**2)
    F_mag = np.nan_to_num(F_mag, 0)
    mask = F_mag > 1e-6
    scale = np.ones_like(F_mag)
    scale[mask] = np.minimum(1.0, max_mag / F_mag[mask])
    Fx = Fx * scale
    Fy = Fy * scale
    return Fx, Fy

def clip_field(field, min_val=-10.0, max_val=10.0):
    return np.clip(field, min_val, max_val)

def count_hotspots(W, threshold=1.5):
    return np.sum(W > threshold) / (grid_size * grid_size)

# --- INITIALIZE COMMON STARTING FIELDS ---
potential = gaussian_filter(np.random.randn(grid_size, grid_size), sigma=5)
Fy_init, Fx_init = np.gradient(potential, dy, dx)
Fx_init, Fy_init = normalize_field(Fx_init, Fy_init)
F_mag_init = np.sqrt(Fx_init**2 + Fy_init**2)
W_init = gaussian_filter(np.random.rand(grid_size, grid_size) - 0.5, sigma=3) * 0.1
eta_init = gaussian_filter(np.random.rand(grid_size, grid_size) - 0.5, sigma=5) * 0.1

# --- Function for Parameter Sweep Simulation ---
def run_sweep_simulation(params):
    lam_W_idx, D_F_idx = params
    current_lam_W = lam_W_values[lam_W_idx]
    D_F = D_F_values[D_F_idx]
    total_combinations = len(lam_W_values) * len(D_F_values)
    current_combination = lam_W_idx * len(D_F_values) + D_F_idx + 1

    print(f"Starting sweep simulation {current_combination}/{total_combinations}: lam_W={current_lam_W:.3f}, D_F={D_F:.3f}")

    # Precompute matrices for F and W
    A_F = sparse.eye(grid_size * grid_size) - (dt/2) * (D_F * L - lam_F * sparse.eye(grid_size * grid_size))
    B_F = sparse.eye(grid_size * grid_size) + (dt/2) * (D_F * L - lam_F * sparse.eye(grid_size * grid_size))
    A_W = sparse.eye(grid_size * grid_size) - (dt/2) * (D_W * L - current_lam_W * sparse.eye(grid_size * grid_size))
    B_W = sparse.eye(grid_size * grid_size) + (dt/2) * (D_W * L - current_lam_W * sparse.eye(grid_size * grid_size))

    # Reset fields
    Fx = Fx_init.copy()
    Fy = Fy_init.copy()
    W = W_init.copy()
    eta = eta_init.copy()

    # Run simulation
    for t in range(timesteps_sweep):
        if t % 100 == 0:  # Print progress every 100 timesteps
            print(f"Sweep simulation {current_combination}/{total_combinations}: lam_W={current_lam_W:.3f}, D_F={D_F:.3f} - Timestep {t}/{timesteps_sweep}")

        Fx_flat = Fx.flatten()
        Fy_flat = Fy.flatten()
        W_flat = W.flatten()
        eta_flat = eta.flatten()

        # Update eta
        F_mag = np.sqrt(Fx**2 + Fy**2)
        S_eta = 0.5 * (F_mag**2 + W**2)
        rhs_eta = B_eta @ eta_flat + dt * S_eta.flatten()
        eta_next_flat = spsolve(A_eta, rhs_eta)
        eta_next = eta_next_flat.reshape(grid_size, grid_size)
        eta_next = clip_field(eta_next, -5, 5)

        # Update Fx
        Wy, Wx = np.gradient(W, dy, dx)
        noise_Fx = np.random.normal(0, noise_F, Fx.shape)
        rhs_Fx = B_F @ Fx_flat + dt * (beta_coupling * Wx.flatten() + noise_Fx.flatten())
        Fx_next_flat = spsolve(A_F, rhs_Fx)
        Fx_next = Fx_next_flat.reshape(grid_size, grid_size)

        # Update Fy
        noise_Fy = np.random.normal(0, noise_F, Fy.shape)
        rhs_Fy = B_F @ Fy_flat + dt * (beta_coupling * Wy.flatten() + noise_Fy.flatten())
        Fy_next_flat = spsolve(A_F, rhs_Fy)
        Fy_next = Fy_next_flat.reshape(grid_size, grid_size)

        Fx_next, Fy_next = normalize_field(Fx_next, Fy_next)

        # Update W
        noise_W_t = np.random.normal(0, noise_W, W.shape)
        rhs_W = B_W @ W_flat + dt * (eta_coeff * eta.flatten() + alpha * F_mag.flatten() + noise_W_t.flatten())
        W_next_flat = spsolve(A_W, rhs_W)
        W_next = W_next_flat.reshape(grid_size, grid_size)
        W_next = clip_field(W_next)
        max_abs_W = np.max(np.abs(W_next))
        if max_abs_W > 1e-8:
            W_next = W_next / max_abs_W * 2.0
        else:
            W_next.fill(0.0)

        Fx, Fy, W, eta = Fx_next, Fy_next, W_next, eta_next

    hotspot_fraction = count_hotspots(W)
    print(f"Completed sweep simulation {current_combination}/{total_combinations}: lam_W={current_lam_W:.3f}, D_F={D_F:.3f}, Hotspot Fraction={hotspot_fraction:.3f}")
    return (lam_W_idx, D_F_idx, hotspot_fraction)

# --- 2D PARAMETER SWEEP WITH PARALLELIZATION ---
print(f"Starting 2D Parameter Sweep (lam_W vs D_F) with Crank-Nicolson... Using {cpu_count()} processes.")
pattern_intensity_summary_2d = np.zeros((len(lam_W_values), len(D_F_values)))
param_combinations = [(j, i) for j in range(len(lam_W_values)) for i in range(len(D_F_values))]

# Use multiprocessing to parallelize the sweep
num_processes = cpu_count()
with Pool(processes=num_processes) as pool:
    results = pool.map(run_sweep_simulation, param_combinations)

# Collect results
for lam_W_idx, D_F_idx, hotspot_fraction in results:
    pattern_intensity_summary_2d[lam_W_idx, D_F_idx] = hotspot_fraction

print("Parameter sweep finished. Collecting and saving results...")

# --- SAVE 2D PHASE DIAGRAM ---
df_2d = pd.DataFrame(pattern_intensity_summary_2d,
                     index=[f"{v:.3f}" for v in lam_W_values],
                     columns=[f"{v:.3f}" for v in D_F_values])
plt.figure(figsize=(12, 9))
sns.heatmap(df_2d, annot=True, fmt=".3f", cmap="viridis", linewidths=.5,
            cbar_kws={"label": "Fraction of W(x) Hotspots (Threshold=1.5)"})
plt.title(f"GVFT 2D Phase Diagram (lam_eta={lam_eta:.2f}, eta_coeff={eta_coeff:.3f})", fontsize=16)
plt.xlabel("D_F (Flow Field Diffusion)", fontsize=12)
plt.ylabel("lam_W (W Decay Rate)", fontsize=12)
plt.tight_layout()
plt.savefig("figures/phase_diagram_2D_lamW_DF.png", dpi=200)
plt.close()
print(f"Saved 2D phase diagram to figures/phase_diagram_2D_lamW_DF.png (lam_eta={lam_eta:.2f}, eta_coeff={eta_coeff:.3f})")

# --- SELECT PARAMETERS FOR FULL SIMULATION ---
row_std_dev = np.std(pattern_intensity_summary_2d, axis=1)
row_mean = np.mean(pattern_intensity_summary_2d, axis=1)
valid_rows_mask = (row_std_dev > 0.01) & (row_mean < 0.99) & (row_mean > 0.01)

if np.any(valid_rows_mask):
    valid_indices = np.where(valid_rows_mask)[0]
    chosen_lam_W_index_relative = np.argmax(row_std_dev[valid_rows_mask])
    chosen_lam_W_index = valid_indices[chosen_lam_W_index_relative]
    chosen_lam_W = lam_W_values[chosen_lam_W_index]
    print(f"Selected lam_W = {chosen_lam_W:.3f} (row index {chosen_lam_W_index}) based on max standard deviation across D_F in non-saturated rows.")
else:
    chosen_lam_W_index = len(lam_W_values) // 2
    chosen_lam_W = lam_W_values[chosen_lam_W_index]
    print(f"Warning: No row with significant variation found. Choosing middle lam_W = {chosen_lam_W:.3f} (row index {chosen_lam_W_index}).")

selected_row_values = pattern_intensity_summary_2d[chosen_lam_W_index, :]
sorted_D_F_indices = np.argsort(selected_row_values)

if len(np.unique(selected_row_values)) < 3:
    print(f"Warning: Fewer than 3 unique hotspot values for lam_W={chosen_lam_W:.3f}. Selecting boundary and middle D_F values.")
    low_idx = 0
    mid_idx = len(D_F_values) // 2
    high_idx = len(D_F_values) - 1
else:
    low_idx = sorted_D_F_indices[0]
    mid_idx = sorted_D_F_indices[len(sorted_D_F_indices)//2]
    high_idx = sorted_D_F_indices[-1]

selected_D_F_indices = sorted(list(set([low_idx, mid_idx, high_idx])))

if len(selected_D_F_indices) < 3:
    print("Adjusting selection to ensure 3 distinct D_F values.")
    missing = 3 - len(selected_D_F_indices)
    all_indices = list(range(len(D_F_values)))
    available_indices = sorted(list(set(all_indices) - set(selected_D_F_indices)))
    candidates = sorted(available_indices, key=lambda x: abs(x - (len(D_F_values) // 2)))
    selected_D_F_indices.extend(candidates[:missing])
    selected_D_F_indices = sorted(list(set(selected_D_F_indices)))[:3]

selected_params_full = [(chosen_lam_W, D_F_values[idx]) for idx in selected_D_F_indices]
print(f"Selected parameter pairs (lam_W, D_F) for full simulation: {selected_params_full}")

# --- Function for Full Simulation ---
def run_full_simulation(params):
    idx, current_lam_W, D_F, dx, dy, x, y, X, Y, num_modules, beta, gamma = params
    total_simulations = len(selected_params_full)
    print(f"--- Starting full simulation {idx+1}/{total_simulations}: lam_W={current_lam_W:.3f}, D_F={D_F:.3f} ---")

    # Precompute matrices for F and W
    A_F = sparse.eye(grid_size * grid_size) - (dt/2) * (D_F * L - lam_F * sparse.eye(grid_size * grid_size))
    B_F = sparse.eye(grid_size * grid_size) + (dt/2) * (D_F * L - lam_F * sparse.eye(grid_size * grid_size))
    A_W = sparse.eye(grid_size * grid_size) - (dt/2) * (D_W * L - current_lam_W * sparse.eye(grid_size * grid_size))
    B_W = sparse.eye(grid_size * grid_size) + (dt/2) * (D_W * L - current_lam_W * sparse.eye(grid_size * grid_size))

    # Reset fields
    Fx = Fx_init.copy()
    Fy = Fy_init.copy()
    W = W_init.copy()
    eta = eta_init.copy()

    # Initialize lists to store fields and graphs
    Fx_series, Fy_series, W_series, eta_series, Graphs, Module_coords_series = [Fx.copy()], [Fy.copy()], [W.copy()], [eta.copy()], [], []

    # Initial module positions at t=0
    module_coords = sample_module_positions(Fx, Fy, W, num_modules, beta, gamma, X, Y)
    Module_coords_series.append(module_coords.copy())

    for t in range(timesteps_sim):
        if t % 100 == 0:  # Print progress every 100 timesteps
            print(f"Full simulation {idx+1}/{total_simulations}: lam_W={current_lam_W:.3f}, D_F={D_F:.3f} - Timestep {t}/{timesteps_sim}")

        Fx_flat = Fx.flatten()
        Fy_flat = Fy.flatten()
        W_flat = W.flatten()
        eta_flat = eta.flatten()

        # Update eta
        F_mag = np.sqrt(Fx**2 + Fy**2)
        S_eta = 0.5 * (F_mag**2 + W**2)
        rhs_eta = B_eta @ eta_flat + dt * S_eta.flatten()
        eta_next_flat = spsolve(A_eta, rhs_eta)
        eta_next = eta_next_flat.reshape(grid_size, grid_size)
        eta_next = clip_field(eta_next, -5, 5)

        # Update Fx
        Wy, Wx = np.gradient(W, dy, dx)
        noise_Fx = np.random.normal(0, noise_F, Fx.shape)
        rhs_Fx = B_F @ Fx_flat + dt * (beta_coupling * Wx.flatten() + noise_Fx.flatten())
        Fx_next_flat = spsolve(A_F, rhs_Fx)
        Fx_next = Fx_next_flat.reshape(grid_size, grid_size)

        # Update Fy
        noise_Fy = np.random.normal(0, noise_F, Fy.shape)
        rhs_Fy = B_F @ Fy_flat + dt * (beta_coupling * Wy.flatten() + noise_Fy.flatten())
        Fy_next_flat = spsolve(A_F, rhs_Fy)
        Fy_next = Fy_next_flat.reshape(grid_size, grid_size)

        Fx_next, Fy_next = normalize_field(Fx_next, Fy_next)

        # Update W
        noise_W_t = np.random.normal(0, noise_W, W.shape)
        rhs_W = B_W @ W_flat + dt * (eta_coeff * eta.flatten() + alpha * F_mag.flatten() + noise_W_t.flatten())
        W_next_flat = spsolve(A_W, rhs_W)
        W_next = W_next_flat.reshape(grid_size, grid_size)
        W_next = clip_field(W_next)
        max_abs_W = np.max(np.abs(W_next))
        if max_abs_W > 1e-8:
            W_next = W_next / max_abs_W * 2.0
        else:
            W_next.fill(0.0)

        # Update fields
        Fx, Fy, W, eta = Fx_next, Fy_next, W_next, eta_next

        # Update module positions based on current fields
        module_coords = sample_module_positions(Fx, Fy, W, num_modules, beta, gamma, X, Y)

        # Store fields and module positions
        Fx_series.append(Fx.copy())
        Fy_series.append(Fy.copy())
        W_series.append(W.copy())
        eta_series.append(eta.copy())
        Module_coords_series.append(module_coords.copy())

        # Graph generation
        weights = np.zeros((num_modules, num_modules))
        for i in range(num_modules):
            xi_idx = np.argmin(np.abs(x - module_coords[i, 0]))
            yi_idx = np.argmin(np.abs(y - module_coords[i, 1]))
            F_i = np.array([Fx[yi_idx, xi_idx], Fy[yi_idx, xi_idx]])
            W_i = W[yi_idx, xi_idx]

            candidate_edges = []
            for j in range(num_modules):
                if i == j: continue
                r_i = module_coords[i]
                r_j = module_coords[j]
                delta = r_j - r_i
                norm_delta = np.linalg.norm(delta)
                norm_F = np.linalg.norm(F_i)
                if norm_delta < 1e-6 or norm_F < 1e-6: continue
                if norm_delta * norm_F < 1e-9: cos_sim = 0.0
                else: cos_sim = np.dot(delta, F_i) / (norm_delta * norm_F)
                cos_sim = np.nan_to_num(np.clip(cos_sim, -1.0, 1.0))

                if cos_sim < cos_threshold: continue
                rho = cos_sim * W_i
                w_ij = np.clip(rho / lambda_val, 0, 1.0)
                candidate_edges.append((j, w_ij))

            candidate_edges = sorted(candidate_edges, key=lambda item: item[1], reverse=True)[:top_k]
            for j, w_ij in candidate_edges:
                weights[i, j] = w_ij
        Graphs.append(weights.copy())

    print(f"Full simulation {idx+1}/{total_simulations}: lam_W={current_lam_W:.3f}, D_F={D_F:.3f} - Field updates completed. Starting visualization...")

    # Visualization
    timesteps_to_show = list(range(0, timesteps_sim + 1, view_step))
    if timesteps_sim not in timesteps_to_show: timesteps_to_show.append(timesteps_sim)
    n_show = len(timesteps_to_show)
    row_labels = ["Flow Mag $|F|$", "Strength $W(x)$", "Flow Field $F(x)$", "Graph & Fields"]
    fig, axs = plt.subplots(len(row_labels), n_show, figsize=(4 * n_show, 13), constrained_layout=True)

    print(f"Full simulation {idx+1}/{total_simulations}: Generating visualization for lam_W={current_lam_W:.3f}, D_F={D_F:.3f}...")
    for i, t in enumerate(timesteps_to_show):
        if t >= len(Fx_series): t = len(Fx_series) - 1

        Fx_t = Fx_series[t]
        Fy_t = Fy_series[t]
        W_t = W_series[t]
        F_mag_t = np.sqrt(Fx_t**2 + Fy**2)
        module_coords_t = Module_coords_series[t]

        if t > 0:
            graph_index = t - 1
            if graph_index < len(Graphs):
                weights = Graphs[graph_index]
            else:
                print(f"Full simulation {idx+1}/{total_simulations}: Warning: Graph index {graph_index} out of bounds for t={t}. Using empty graph.")
                weights = np.zeros((num_modules, num_modules))
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
            axs[2, i].streamplot(x, y, Fx_t, Fy_t, color=F_mag_t, cmap='Blues', density=1.2, linewidth=0.8)
        axs[2, i].set_xlim(-1, 1); axs[2, i].set_ylim(-1, 1)
        axs[2, i].set_aspect('equal', adjustable='box')
        axs[2, i].set_xticks([]); axs[2, i].set_yticks([])

        # Row 3: Graph + Fields Overlay
        axs[3, i].imshow(W_t, extent=[-1, 1, -1, 1], origin='lower', cmap='hot', alpha=0.4, aspect='auto', vmin=-2, vmax=2)
        if np.any(Fx_t) or np.any(Fy_t):
            axs[3, i].streamplot(x, y, Fx_t, Fy_t, color='lightgray', density=1.5, linewidth=0.5)
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
                        axs[3, i].arrow(x0, y0, dx_arrow, dy_arrow, head_width=0.06, head_length=0.08, length_includes_head=True, alpha=arrow_alpha, color='green', linewidth=arrow_width, zorder=2)
        axs[3, i].set_xlim(-1, 1); axs[3, i].set_ylim(-1, 1)
        axs[3, i].set_aspect('equal', adjustable='box')
        axs[3, i].set_xticks([]); axs[3, i].set_yticks([])

    # Add row labels
    for row_idx, label in enumerate(row_labels):
        axs[row_idx, 0].set_ylabel(label, fontsize=14, labelpad=20)
        axs[row_idx, 0].yaxis.set_label_coords(-0.15, 0.5)

    # Save visualization
    filename = f"figures/simulation_lamW_{current_lam_W:.3f}_DF_{D_F:.3f}_etacoeff_{eta_coeff:.3f}.png"
    plt.suptitle(f'Full Simulation: lam_W={current_lam_W:.3f}, D_F={D_F:.3f}, eta_coeff={eta_coeff:.3f}', fontsize=16, y=0.99)
    plt.savefig(filename, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Full simulation {idx+1}/{total_simulations}: Saved visualization to {filename}")
    print(f"--- Completed full simulation {idx+1}/{total_simulations}: lam_W={current_lam_W:.3f}, D_F={D_F:.3f} ---")

# --- RUN FULL SIMULATIONS WITH PARALLELIZATION ---
print(f"\nStarting full simulations for selected parameters with Crank-Nicolson... Using {num_processes} processes.")
param_combinations_full = [(idx, lam_W, D_F, dx, dy, x, y, X, Y, num_modules, beta, gamma) for idx, (lam_W, D_F) in enumerate(selected_params_full)]

# Use multiprocessing to parallelize the full simulations
with Pool(processes=num_processes) as pool:
    pool.map(run_full_simulation, param_combinations_full)

print("\nAll simulations finished.")