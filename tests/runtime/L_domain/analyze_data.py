import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from netCDF4 import Dataset

def process_and_plot():
    # 1. Load data
    nc_file = "L_domain_out.nc"
    if not os.path.exists(nc_file):
        print(f"Error: {nc_file} not found.")
        return

    # Use same loading procedure as notebook
    ds = Dataset(nc_file)
    data = xr.open_dataset(xr.backends.NetCDF4DataStore(ds))
    
    # 2. Configuration
    dt = 0.5
    t_max = 10.0
    nt = data.dims["t"]
    time_array = np.arange(0, nt * dt, dt)
    
    # Crop dataset
    mask = time_array <= t_max
    time_array = time_array[mask]
    data_cut = data.isel(t=mask)
    
    points = {
        "G1": [3.75, 0.125],
        "G2": [4.20, 0.125],
        "G3": [4.45, 0.125],
        "G4": [4.95, 0.125],
        "G5": [4.20, 0.375],
        "G6": [4.95, 0.375]
    }
    
    lines = {
        "S1": 4.1, 
        "S2": 4.4,
    }
    
    # 3. Points Analysis
    point_indices = {}
    for name, (px, py) in points.items():
        ix = np.argmin(np.abs(data_cut.x.values - px))
        iy = np.argmin(np.abs(data_cut.y.values - py))
        point_indices[name] = (iy, ix)
        
    # Plot point comparison (3x2 grid)
    fig, axes = plt.subplots(3, 2, figsize=(12, 10), sharex=True)
    axes = axes.flatten()
    
    for i, (name, (iy, ix)) in enumerate(point_indices.items()):
        ax = axes[i]
        
        # Simulation results (eta = h + z_b)
        # Simulation results (eta = h + z_b)
        sim_h = data_cut.h.values[:, iy, ix]
        sim_z_b = data_cut.z_b.values[:, iy, ix]
        eta = sim_h + sim_z_b
        
        ax.plot(time_array, eta, 'k-', label='PyExner', linewidth=1.5)
        
        # Overlay experimental and model data from CSV
        res_dir = "experimental_results"
        valid_models = ['exp', 'mcs', 'fvm']
        colors = {'exp': 'r', 'mcs': 'g', 'fvm': 'b'}
        markers = {'exp': 'o', 'mcs': '+', 'fvm': 'x'}
        
        for model in valid_models:
            csv_path = os.path.join(res_dir, f"{name}_{model}.csv")
            if os.path.exists(csv_path):
                df = pd.read_csv(csv_path)
                ax.plot(df['t'], df['n'], colors[model] + markers[model], label=model.upper(), markersize=3, alpha=0.7)
                
        ax.set_title(f"Gauge {name}", fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Axis limits
        if name == "G1":
            ax.set_ylim(0.08, 0.22)
        else:
            ax.set_ylim(0.08, 0.18)
        ax.set_xlim(0, 10)
            
        ax.set_ylabel(r"Free surface $\eta$ (m)")
        ax.set_xlabel("Time (s)")
        ax.legend(fontsize=8, loc='upper right')
            
    fig.suptitle("Simulation vs Experimental Data Comparison", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig("Gauges_Comparison.png", bbox_inches='tight')
    plt.close()
    
    # 4. Lines Analysis (S1, S2)
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
    target_t = 20.0
    t_idx = np.argmin(np.abs(time_array - target_t))
    
    for i, (name, x_val) in enumerate(lines.items()):
        ax = axes[i]
        ix = np.argmin(np.abs(data_cut.x.values - x_val))
        
        # Simulation profile at t=10s (current t_max)
        y_sim = data_cut.y.values
        zb_sim = data_cut.z_b.values[t_idx, :, ix]
        
        ax.plot(y_sim, zb_sim, 'k-', label='PyExner', linewidth=2)
        
        for model in valid_models:
            csv_path = os.path.join(res_dir, f"{name}_{model}.csv")
            if os.path.exists(csv_path):
                df = pd.read_csv(csv_path)
                ax.plot(df['y'], df['z'], colors[model] + markers[model], label=model.upper(), markersize=3, alpha=0.7)
                
        ax.set_title(f"Section {name} (x={x_val} m)")
        ax.grid(True, alpha=0.3)
        ax.set_ylabel(r"Bed elevation $z_b$ (m)")
        if i == 1:
            ax.set_xlabel("y (m)")
            
    axes[0].legend()
    plt.tight_layout()
    plt.savefig("Sections_Comparison.png", bbox_inches='tight')
    plt.close()

    # 5. Dambreak animations (80th slice)
    """g = 9.81
    for i in range(len(time_array)):
        h_slice = data_cut.h[i, 80]
        zb_slice = data_cut.z_b[i, 80]
        hu_slice = data_cut.hu[i, 80]
        
        u_slice = np.where(h_slice > 1e-6, hu_slice / h_slice, 0.0)
        froude = np.abs(u_slice) / np.sqrt(g * h_slice + 1e-10)
        
        fig = plt.figure(figsize=(10, 9))
        gs = fig.add_gridspec(3, 1, height_ratios=[1.8, 1, 1])
        
        ax1 = fig.add_subplot(gs[0])
        ax2 = fig.add_subplot(gs[1])
        ax3 = fig.add_subplot(gs[2])
        
        # Water Level
        ax1.plot(data_cut.x, h_slice + zb_slice, 'k-')
        ax1.set_ylabel(r"Water Level $h+z_b$ [m]")
        ax1.set_title(f"Time: {time_array[i]:.1f} s")
        
        # Bed Height
        ax2.plot(data_cut.x, zb_slice, 'k-')
        ax2.set_ylabel(r"Bed Height $z_b$ [m]")
        
        # Froude
        ax3.plot(data_cut.x, froude, 'k-')
        ax3.axhline(1.0, color='red', linestyle='--')
        ax3.set_ylabel(r"Froude Number $Fr$")
        ax3.set_xlabel(r"Distance $x$ [m]")
        
        plt.tight_layout()
        fig.savefig(f"dambreak_{i:03d}.png", dpi=150)
        plt.close()"""
    
    print("Analysis completed. Plots saved.")

if __name__ == "__main__":
    process_and_plot()