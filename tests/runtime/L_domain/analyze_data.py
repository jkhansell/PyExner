import xarray as xr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Assuming 'data' is your xarray dataset
data = xr.open_dataset("L_domain_out.nc")
g = 9.81
total_time = 20.0
dt = 0.5
N_half = data.h.shape[1] // 2
time_array = np.arange(0, total_time, dt)

points = {
    "U1": [3.75, 0.125],
    "U2": [4.20, 0.375],
    "U3": [4.20, 0.125],
    "U4": [4.45, 0.375],
    "U5": [4.45, 0.125],
    "U6": [4.95, 0.375]
}

lines = {
    "S1": 4.1, 
    "S2": 4.2, 
    "S3": 4.3, 
    "S4": 4.4
}

# 1. Setup indices dictionary
point_indices = {}

for name, coords in points.items():
    px, py = coords
    # Find the index of the closest value
    idx_x = np.argmin(np.abs(data.x.values - px))
    idx_y = np.argmin(np.abs(data.y.values - py))
    
    # Store indices in a dictionary mapped to the point name
    point_indices[name] = (idx_y, idx_x)

# 2. Initialize z_vals as a dictionary of lists
# This creates: {"U1": [], "U2": [], ...}
z_vals = {name: [] for name in points.keys()}

# 3. Populate the dictionary over time
for time_step in range(len(time_array)):
    for name, (iy, ix) in point_indices.items():
        # Access the value and append to the specific point's list
        val = data.z_b.values[-1, iy, ix]
        z_vals[name].append(val)

# Optional: Convert lists to numpy arrays for easier plotting/math later
for name in z_vals:
    z_vals[name] = np.array(z_vals[name])
    
fig, axes = plt.subplots(3, 2, figsize=(12, 10), sharex=True, sharey=True)
axes = axes.flatten()  # Flatten to easily loop through them

for i, (name, values) in enumerate(z_vals.items()):
    ax = axes[i]
    ax.plot(time_array, values, color='tab:blue', linewidth=1.5)
    ax.set_title(f"Point {name}", fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Label only the outer plots to keep it clean
    if i >= 4: ax.set_xlabel("Time (s)")
    if i % 2 == 0: ax.set_ylabel("Z Value")

plt.suptitle("Individual Comparisons of Point Data", fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig("Gauge_points.png")
plt.close()

# 1. Setup indices for the lines (x-coordinates)
line_indices = {}
for name, x_coord in lines.items():
    idx_x = np.argmin(np.abs(data.x.values - x_coord))
    line_indices[name] = idx_x

# 2. Select the specific timesteps you want to visualize
# We find the nearest indices in the data for [0.0, 0.1, 0.3, 0.6]
target_t = [20.0]
t_indices = [np.argmin(np.abs(time_array - t)) for t in target_t]


# 3. Create the plot (2x2 grid for S1, S2, S3, S4)
fig, axes = plt.subplots(2, 2, figsize=(12, 10), sharex=True, sharey=True)
axes = axes.flatten()

for i, (name, ix) in enumerate(line_indices.items()):
    ax = axes[i]
    
    # Plot a line for each specific timestep
    # Extract Z values along the entire Y axis for the fixed X index
    # shape will be (len(dataset.y))
    z_profile = data.z_b.values[-1, :, ix]
    
    actual_time = time_array[-1]
    ax.plot(data.y.values, z_profile, label=f"last time")

    if name == "S1" or name == "S2":
        df = pd.read_csv(f"experimental_results/{name}.csv").sort_values("x")
        ax.plot(df.iloc[:,0], df.iloc[:,1], label="Experimental results", marker=".")

    ax.set_title(f"Cross-section at {name} (x={lines[name]})", fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize='small')
    ax.set_ylim(0.04, 0.14)

    # Axis labels
    if i >= 2: ax.set_xlabel("y-coordinate")
    if i % 2 == 0: ax.set_ylabel("Z Value")

plt.suptitle("Spatial Profiles along Y-axis at Different Timesteps", fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig("Line_Profiles.png")
plt.close()

for i, idt in enumerate(time_array):

    h_slice = data.h[i, 80]
    hu_slice = data.hu[i, 80]

    # Calculate velocity u = hu / h (handling potential division by zero)
    u_slice = np.where(h_slice > 1e-6, hu_slice / h_slice, 0.0)
    # Calculate Froude number
    froude = np.abs(u_slice) / np.sqrt(g * h_slice + 1e-10)

    # 2. Plotting
    fig = plt.figure(figsize=(10, 9)) # Increased height from 6 to 9
    gs = fig.add_gridspec(3, 1, height_ratios=[1.8, 1, 1]) # Added a third row

    ax_1 = fig.add_subplot(gs[0])
    ax_2 = fig.add_subplot(gs[1])
    ax_3 = fig.add_subplot(gs[2])

    # Water Level Plot
    ax_1.plot(data.x, data.h[i, 80], c="black", marker='o', 
            markerfacecolor='none', markeredgecolor="black", markersize=2)
    ax_1.set_ylabel(r"Water level $h+z$ [m]")
    ax_1.set_xticklabels([]) # Remove labels to avoid overlap with middle plot
    #ax_1.set_ylim(0.0, 0.4)

    # Bed Height Plot
    ax_2.plot(data.x, data.z_b[i, 80], c="black", marker='o', 
            markerfacecolor='none', markeredgecolor="black", markersize=2)
    ax_2.set_ylabel(r"Bed height $z$ [m]")
    ax_2.set_xticklabels([]) # Remove labels
    #ax_2.set_ylim(0.06, 0.11)

    # Froude Number Plot
    ax_3.plot(data.x, froude, c="black", marker='o', 
            markerfacecolor='none', markeredgecolor="black", markersize=2)
    ax_3.axhline(1.0, color='red', linestyle='--', alpha=0.5, label='Critical Flow (Fr=1)')
    ax_3.set_ylabel(r"Froude number $Fr$")
    ax_3.set_xlabel(r"Channel position $x$ [m]")

    plt.tight_layout()
    fig.savefig(f"dambreak_{i}.png", dpi=200)
    plt.close()

    