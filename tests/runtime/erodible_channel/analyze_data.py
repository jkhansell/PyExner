import xarray as xr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

"""
Location of gauges for the partial dam-break flow in a straight erodible
channel.
Gauge x (m) y (m)
G1 0.64 -0.5
G2 0.64 -0.165
G3 1.94 -0.99
G4 1.94 -0.33

three cross sections located at S1 (y = 0.2 m), S2 (y = 0.7
m), and S3 (y = 1.45 m).

"""

# constants
data = xr.open_dataset("erodible_channel_out.nc")

print(data)

g = 9.81
total_time = 20.0
dt = 0.25
N_half = data.h.shape[1] // 2
time_array = np.arange(0, total_time, dt)

markers = ["o", "s", "x", "+"]
colors = ["red", "orange", "green", "black"]

def experimental_envelope(exp_dfs, xcol="x", ycol="z", npts=300):
    x_common = np.linspace(
        min(df[xcol].min() for df in exp_dfs),
        max(df[xcol].max() for df in exp_dfs),
        npts
    )

    Y = []
    for df in exp_dfs:
        df = df.sort_values(xcol)
        y_interp = np.interp(x_common, df[xcol], df[ycol])
        Y.append(y_interp)

    Y = np.array(Y)
    return x_common, Y.min(axis=0), Y.max(axis=0)

def plot_line_profiles():

    lines = {
        "S1": 0.2,
        "S2": 0.7, 
        "S3": 1.45, 
    }

    z = data.z_b.isel(t=-1)

    line_indices = {}
    for name, y_coord in lines.items():
        y_target = y_coord
        iy = np.argmin(np.abs(data.y.values - y_target))
        
        line_indices[name] = iy

    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
   
    axes = axes.flatten()

    for i, (name, iy) in enumerate(line_indices.items()):
        ax = axes[i]
        
        # Plot a line for each specific timestep
        # Extract Z values along the entire Y axis for the fixed X index
        # shape will be (len(dataset.y))
        profile = z.isel(y=iy).sel(x=slice(1.0, 9.0))

        actual_time = time_array[-1]
        print(actual_time)
        # plot experimental envelope
        expfiles = [x for x in os.listdir("experimental_results") if name in x and "exp" in x and "S" in name]
        expdfs = []
        for j, file in enumerate(expfiles):
            df = pd.read_csv(os.path.join("experimental_results", file), skipinitialspace=True).sort_values("x")
            ax.scatter(df.iloc[:,0], df.iloc[:,1], label=f"Exp {file.split('exp')[-1][:-4]}", 
                marker=markers[j % len(markers)], 
                color="black",
                s=10,
                zorder=1
            )
            expdfs.append(df)

        if len(expdfs) > 0:
            x_common, Ymin, Ymax = experimental_envelope(expdfs)
            ax.fill_between(
                x_common, Ymin, Ymax, 
                color="gray", alpha=0.5,
                edgecolor="none",
                label="Experimental envelope" if i == 0 else None,
                zorder=0
            )

        # plot each series
        files = [x for x in os.listdir("experimental_results") if name in x and "exp" not in x]
        for j,file in enumerate(files):
            df = pd.read_csv(os.path.join("experimental_results", file), skipinitialspace=True).sort_values("x")
            ax.plot(df.iloc[:,0], df.iloc[:,1], label=f"{file[3:-4]}", c=colors[j % len(colors)], zorder=2)

        ax.plot(profile.x - 1.0, profile.values, label=f"PyExner", zorder=3, color="blue")

        ax.set_title(f"Cross-section at {name} (y={lines[name]})", fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize='small', ncol=2)
        ax.set_xlim(0.0, 8.0)

    fig.savefig("lines.png", dpi=250)
    plt.close()


def plot_time_series():
    points = {
        "G1": [0.64, -0.50],
        "G2": [0.64, -0.165],
        "G3": [1.94, -0.99],
        "G4": [1.94, -0.33],
    }

    # 1. Setup indices dictionary
    point_indices = {}

    for name, coords in points.items():
        px, py = coords

        px_target = 1.0 + px  
        py_target = py 

        # Find the index of the closest value
        idx_x = np.argmin(np.abs(data.x.values - px_target))
        idx_y = np.argmin(np.abs(data.y.values - py_target))
        
        # Store indices in a dictionary mapped to the point name
        point_indices[name] = (idx_y, idx_x)
    
    # 2. Initialize eta_vals as a dictionary of lists
    # This creates: {"G1": [], "G2": [], ...}
    eta_vals = {name: [] for name in points.keys()}

    # 3. Populate the dictionary over time
    for time_step in range(len(time_array)):
        for name, (iy, ix) in point_indices.items():
            # Access the value and append to the specific point's list
            val = data.z_b.values[time_step, iy, ix] + data.h.values[time_step, iy, ix]
            eta_vals[name].append(val)

    # Optional: Convert lists to numpy arrays for easier plotting/math later
    for name in eta_vals:
        eta_vals[name] = np.array(eta_vals[name])

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharey=True, sharex=True)
    axes = axes.flatten()  # Flatten to easily loop through them

    for i, (name, values) in enumerate(eta_vals.items()):
        
        ax = axes[i]

        files = [x for x in os.listdir("experimental_results") if name in x]
        for j,file in enumerate(files):
            df = pd.read_csv(os.path.join("experimental_results", file), skipinitialspace=True).sort_values("t" if "G" in name else "x")
            if "Exp" in file or "exp" in file:
                ax.scatter(df.iloc[:,0], df.iloc[:,1], label=f"{file[3:-4]}", marker=".", c=colors[j % len(colors)], zorder=2)
            else:
                ax.plot(df.iloc[:,0], df.iloc[:,1], label=f"{file[3:-4]}", c=colors[j % len(colors)], zorder=2)
             
        ax.plot(time_array, values, color='tab:blue', linewidth=1.5, label="PyExner")
        ax.set_title(f"Point {name}", fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize='small', ncol=1)
        
        # Label only the outer plots to keep it clean
        if i >= 2: ax.set_xlabel("Time (s)")
        if i % 2 == 0: ax.set_ylabel("Z")

    plt.suptitle("Individual Comparisons of Point Data", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig("Gauge_points.png", dpi=250)
    plt.close()

def plot_images():
    os.makedirs("dambreak", exist_ok=True)
    os.makedirs("fields", exist_ok=True)

    for i, idt in enumerate(time_array):
        print(f"Timestep: {i % 10}")

        x_sl = slice(-2.1, 11.9)
        y_sl = slice(1.8, -1.8)

        G = data.G[i].sel(x=x_sl, y=y_sl)
        h = data.h[i].sel(x=x_sl, y=y_sl)
        z = data.z[i].sel(x=x_sl, y=y_sl)
        z_b = data.z_b[i].sel(x=x_sl, y=y_sl)

        eps = 1e-12
        u = data.hu[i].sel(x=x_sl, y=y_sl) / (h + eps)
        v = data.hv[i].sel(x=x_sl, y=y_sl) / (h + eps)

        umag = np.sqrt(u**2 + v**2)

        fig, axs = plt.subplots(2, 2, figsize=(12, 6), constrained_layout=True)

        fig.suptitle(f"Time: {idt} s")

        extent = [G.x.min(), G.x.max(), G.y.min(), G.y.max()]

        # --- Free surface ---
        im0 = axs[0, 0].imshow(h + z_b + z,
                            cmap="jet",
                            extent=extent)
        axs[0, 0].set_title(r"Free surface $(h + z_b + z) [m]$")
        plt.colorbar(im0, ax=axs[0, 0])
        im0.set_clim(1e-1, 0.6)

        # --- Bed elevation ---
        im1 = axs[0, 1].imshow(z_b + z,
                            cmap="jet",
                            extent=extent)
        axs[0, 1].set_title(r"Bed elevation $(z_b + z) [m]$")
        plt.colorbar(im1, ax=axs[0, 1])
        im1.set_clim(0.1, 0.3)

        # --- Exner flux / G ---
        im2 = axs[1, 0].imshow(G,
                            cmap="jet",
                            extent=extent)
        axs[1, 0].set_title(r"Interaction factor $G [s^2/m]$")
        plt.colorbar(im2, ax=axs[1, 0])
        im2.set_clim(0.0, 0.002)

        # --- Velocity magnitude ---
        im3 = axs[1, 1].imshow(umag,
                            cmap="jet",
                            extent=extent)
        axs[1, 1].set_title(r"Velocity magnitude $|u| [m/s]$")
        plt.colorbar(im3, ax=axs[1, 1])
        im3.set_clim(0.0, 4.0)

        for ax in axs.flat:
            ax.axis("equal")

        plt.savefig(f"fields/fields_{i:04d}.png", dpi=250)
        plt.close()
        
        y_target = -0.5
        iy = np.argmin(np.abs(data.y.values - y_target))

        h_slice = data.h[i, iy]
        zb_slice = data.z_b[i, iy]
        z_slice = data.z[i, iy]
        hu_slice = data.hu[i, iy]

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
        ax_1.plot(data.x, h_slice + zb_slice + z_slice, c="black", marker='o', 
                markerfacecolor='none', markeredgecolor="black", markersize=2)
        ax_1.set_ylabel(r"Water level $h+z$ [m]")
        ax_1.set_xticklabels([]) # Remove labels to avoid overlap with middle plot
        #ax_1.set_ylim(0.0, 0.4)

        # Bed Height Plot
        ax_2.plot(data.x, zb_slice + z_slice, c="black", marker='o', 
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
        fig.savefig(f"dambreak/dambreak_{i}.png", dpi=200)
        plt.close()


if __name__ == "__main__":
    #plot_line_profiles()
    #print("Lines plotted")
    #plot_time_series()
    #print("Time series plotted")
    plot_images()
    print("Images plotted")