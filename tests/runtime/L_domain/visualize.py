import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import os

# 1. Load the data
data = xr.open_dataset("L_domain_out.nc")

# Dynamically find the time dimension name (usually 'time' or 't')
time_dim = 't'

if time_dim is None:
    # Fallback if time isn't explicitly a dimension name
    num_steps = len(data.z_b)
    print(f"Time dimension not explicitly found. Looping over length of z_b ({num_steps} steps).")
else:
    num_steps = len(data[time_dim])
    print(f"Found time dimension '{time_dim}' with {num_steps} steps.")

# Create an output directory to keep your workspace clean
os.makedirs("plots", exist_ok=True)

# 2. Loop over every time step
for i in range(num_steps):
    # Select the current time slice
    step_data = data.isel({time_dim: i})

    # 3. Calculate umag safely (avoiding division by zero)
    h = step_data.h
    u = np.where(h > 1e-5, step_data.hu / h, 0.0)
    v = np.where(h > 1e-5, step_data.hv / h, 0.0)
    umag = np.sqrt(u**2 + v**2)

    print(h.shape)
    print(np.where(step_data.h > 0.25))

    # 4. Set up the 2x2 plot grid
    fig, axs = plt.subplots(4, 1, figsize=(14, 5), sharex=True, sharey=True)
    axs = axs.ravel()  # Flatten the 2D array of axes to 1D

    # Define variables, titles, and color themes for the grid
    plot_configs = [
        {"data": h,            "title": f"Water Depth (h) - Step {i}", "cmap": "jet"},
        {"data": step_data.z_b,"title": f"Bed Elevation (z_b) - Step {i}", "cmap": "jet"},
        {"data": step_data.G,  "title": f"G - Step {i}",               "cmap": "jet"},
        {"data": umag,         "title": f"Velocity Magnitude (umag) - Step {i}", "cmap": "jet"}
    ]

    # 5. Populate the subplots
    for ax, config in zip(axs, plot_configs):
        im = ax.imshow(config["data"], cmap=config["cmap"])
        fig.colorbar(im, ax=ax, orientation="vertical", shrink=0.7)
        ax.set_title(config["title"], fontsize=12, fontweight="bold")
        ax.set_aspect("equal")

    plt.tight_layout()

    # 6. Save the grid and clean up memory
    fig.savefig(f"plots/grid_output_{i:04d}.png", dpi=250, bbox_inches="tight")
    plt.close(fig)
    plt.clf()

    if (i + 1) % 10 == 0 or (i + 1) == num_steps:
        print(f"Progress: [{i + 1}/{num_steps}] plots saved.")

print("All 2x2 grid plots generated successfully inside the 'plots/' directory!")