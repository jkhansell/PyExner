import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import xarray as xr
from netCDF4 import Dataset
import os
import imageio

################## General Setup ##################
# results folder
folder = 'results'
os.makedirs(folder, exist_ok=True)

# Enable or disable Monte Carlo runs parsing and plotting
include_monte_carlo = False 

# Main simulation
ds_main = Dataset(f"90_bend_channel_domain_out.nc") # Change if needed
data_main = xr.open_dataset(xr.backends.NetCDF4DataStore(ds_main))

# Monte Carlo runs
mc_datasets = []
if include_monte_carlo:
    num_runs = 3
    for i in range(num_runs):
        ds_mc = Dataset(f"run_{i+1}/90_bend_channel_domain__out_{i+1}.nc")
        data_mc = xr.open_dataset(xr.backends.NetCDF4DataStore(ds_mc))
        mc_datasets.append(data_mc)
else:
    print("Monte Carlo runs are disabled.")

# time stepping parameters (modify dt or t_max to change simulation window) # Change if needed
dt = 0.5
t_max = 115 # for analysis 

################## Points and sections of interest ##################
points = {
    "G1": [2, 2],
    "G2": [2.74, 0.69],
    "G3": [4.24, 0.69],
    "G4": [5.74, 0.69],
    "G5": [6.48, 1.95],
    "G6": [6.48, 3.45]
}

lines = {
    "S1": 6.34, 
    "S2": 6.6,
}

# find nearest grid indices for each point
point_indices = {}
for name, (px, py) in points.items():
    ix = np.argmin(np.abs(data_main.x.values - px))
    iy = np.argmin(np.abs(data_main.y.values - py))
    point_indices[name] = (iy, ix)

# find nearest x-index for each section
line_indices = {}
for name, px in lines.items():
    ix = np.argmin(np.abs(data_main.x.values - px))
    line_indices[name] = ix


def plot_domain():
    print("Plotting domain verification...")
    plt.figure(figsize=(8, 4), dpi=100)

    # initial water height field
    plt.imshow(data_main.h[0],
               origin='upper', 
               extent=[data_main.x.min(), data_main.x.max(),
                       data_main.y.min(), data_main.y.max()],
               aspect='auto',
               cmap='jet')

    plt.colorbar(label="Altura inicial del agua $h$ (m)")

    for name, (iy, ix) in point_indices.items():
        x_val = data_main.x.values[ix]
        y_val = data_main.y.values[iy]
        
        plt.scatter(x_val, y_val, color='red')
        plt.text(x_val, y_val, name, color='white', fontsize=9, fontweight='bold')

    for name, ix in line_indices.items():
        x_val = data_main.x.values[ix]
        plt.plot([x_val, x_val], [0.445, data_main.y.max()], color='r', linestyle='--', linewidth=1.5, alpha=0.8) 
        plt.text(x_val, 0.445, f' {name}', color='white', fontweight='bold', va='bottom', ha='right')

    plt.title("Puntos y secciones en el domino $(t=0)$")
    plt.xlabel("$x$ (m)")
    plt.ylabel("$y$ (m)")

    plt.tight_layout()
    file_path = os.path.join(folder, "domain.png")
    plt.savefig(file_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_time_series():
    print("Plotting points results...")
    fig, axes = plt.subplots(3, 2, figsize=(12, 10), sharex=True)
    axes = axes.flatten()

    nt = data_main.sizes["t"]
    time_array = np.arange(0, nt * dt, dt)
    mask = time_array <= 60
    time_array_cut = time_array[mask]
    
    data_cut_main = data_main.isel(t=mask)

    # Plot Main Simulation
    for i, name in enumerate(points.keys()):
        ax = axes[i]
        px, py = points[name]
        ds_point = data_cut_main.sel(x=px, y=py, method="nearest")

        eta_series = ds_point.h.values + ds_point.z_b.values

        if i == 0:
            ax.plot(time_array_cut, ds_point.h.values-0.33, linewidth=1.5, label="Simulación", color='blue')
        else:
            ax.plot(time_array_cut, eta_series, linewidth=1.5, label="Simulación", color='blue')
            
        if i == 0: ax.legend()

    # Plot Monte Carlo Runs
    if include_monte_carlo:
        all_runs_eta = {name: [] for name in points.keys()}
        for mc_data in mc_datasets:
            data_cut_mc = mc_data.isel(t=mask)
            for i, name in enumerate(points.keys()):
                ax = axes[i]
                px, py = points[name]
                ds_point = data_cut_mc.sel(x=px, y=py, method="nearest")
                eta_series = ds_point.h.values + ds_point.z_b.values
                if i == 0:
                    val = ds_point.h.values - 0.33
                else:
                    val = eta_series
                
                ax.plot(time_array_cut, val, linewidth=1, alpha=0.5, color='skyblue')
                all_runs_eta[name].append(val)

        # Plot MC Uncertainty Band and add label
        for i, name in enumerate(points.keys()):
            ax = axes[i]
            if all_runs_eta[name]:
                runs_matrix = np.array(all_runs_eta[name])
                eta_min = np.min(runs_matrix, axis=0)
                eta_max = np.max(runs_matrix, axis=0)
                ax.fill_between(time_array_cut, eta_min, eta_max, color='gray', alpha=0.2, zorder=1)
                
                if i == 0 and len(mc_datasets) > 0:
                    ax.plot([], [], color='skyblue', label='Monte Carlo')
                    ax.legend() # Refresh legend

    # Experimental data
    for i, name in enumerate(points.keys()):
        ax = axes[i]
        try:
            df = pd.read_csv(f"experimental_results/{name}_exp.csv")
            t_exp = df.iloc[:, 0].values
            eta_exp = df.iloc[:, 1].values
            ax.plot(t_exp, eta_exp, 'o', label="Experimental", markersize=2.5, color="orange")
        except FileNotFoundError:
            pass

        ax.set_xlim(0, 60)
        if i == 0:
            ax.set_ylim(0, 0.3)
        elif i in [1,2,3]:
            ax.set_ylim(0, 0.25)
        else:
            ax.set_ylim(0, 0.16)

        ax.set_title(f"Punto {name}", fontweight='bold')
        ax.grid(True, alpha=0.3)

        if i >= 4:
            ax.set_xlabel("Tiempo (s)")
        if i % 2 == 0:
            ax.set_ylabel("Superficie libre $\eta$ (m)")
            
        ax.legend(fontsize='small')

    plt.suptitle("Simulación vs Experimental", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    file_path = os.path.join(folder, "points_sim_vs_exp.png")
    plt.savefig(file_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_line_profiles():
    print("Plotting lines results...")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

    nt = data_main.sizes["t"]
    time_array = np.arange(0, nt * dt, dt)
    mask = time_array <= 115
    data_cut_main = data_main.isel(t=mask)

    # Plot main simulation
    for i, (name, x_val) in enumerate(lines.items()):
        ax = axes[i]
        ix = np.argmin(np.abs(data_cut_main.x.values - x_val))
        z_sim = data_cut_main.z_b.values[-1, 3:, ix]
        y_vals = data_cut_main.y.values[3:] 
        ax.plot(y_vals, z_sim, color='blue', linewidth=2, label="Simulación", zorder=10)

    # Plot Monte Carlo
    if include_monte_carlo:
        all_runs_zb = {name: [] for name in lines.keys()}
        for mc_data in mc_datasets:
            data_cut_mc = mc_data.isel(t=mask)
            for i, (name, x_val) in enumerate(lines.items()):
                ax = axes[i]
                ix = np.argmin(np.abs(data_cut_mc.x.values - x_val))
                z_sim = data_cut_mc.z_b.values[-1, :, ix]
                y_vals = data_cut_mc.y.values
                ax.plot(y_vals, z_sim, color='skyblue', linewidth=0.8, alpha=0.3)
                all_runs_zb[name].append(z_sim)

        # Plot MC Uncertainty Band
        for i, (name, x_val) in enumerate(lines.items()):
            ax = axes[i]
            if all_runs_zb[name]:
                y_vals = data_main.y.values
                runs_matrix = np.array(all_runs_zb[name])
                z_min = np.min(runs_matrix, axis=0)
                z_max = np.max(runs_matrix, axis=0)
                ax.fill_between(y_vals, z_min, z_max, color='gray', alpha=0.2, zorder=1)
                if i == 0 and len(mc_datasets) > 0:
                    ax.plot([], [], color='skyblue', label='Monte Carlo')

    # Experimental data
    for i, (name, x_val) in enumerate(lines.items()):
        ax = axes[i]
        try:
            df = pd.read_csv(f"experimental_results/{name}_exp.csv")
            df = df.sort_values(by=df.columns[0])
            y_exp = df.iloc[:, 0].values
            z_exp = df.iloc[:, 1].values
            ax.plot(y_exp, z_exp, 'o', label="Experimental", markersize=3.5, color="orange", zorder=15)
        except FileNotFoundError:
            pass

        ax.set_title(f"Sección {name} (x = {x_val} m)", fontweight='bold')
        ax.set_xlabel("y (m)")
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.set_ylabel("Altura del lecho $z_b$ (m)")
        
        ax.legend(fontsize='small', loc='best')

    plt.suptitle("Perfiles transversales del lecho (t = 115 s)", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    file_path = os.path.join(folder, "sections_sim_vs_exp.png")
    plt.savefig(file_path, dpi=300, bbox_inches='tight')
    plt.close()


def compute_metrics():
    print("Computing metrics...")
    all_results = []

    # Main data filtering
    nt = data_main.sizes["t"]
    time_array = np.arange(0, nt * dt, dt)
    mask = time_array <= t_max
    time_array_cut = time_array[mask]
    data_cut_main = data_main.isel(t=mask)

    # Point metrics
    eta_vals = {}
    for name in points.keys():
        px, py = points[name]
        ds_point = data_cut_main.sel(x=px, y=py, method="nearest")
        if name == 'G1':
            eta_vals[name] = ds_point.h.values - 0.33
        else: 
            eta_vals[name] = ds_point.h.values + ds_point.z_b.values

    for name in points.keys():
        try:
            df = pd.read_csv(f"experimental_results/{name}_exp.csv")
            t_exp = df.iloc[:, 0].values
            eta_exp = df.iloc[:, 1].values
            
            eta_sim_interp = np.interp(t_exp, time_array_cut, eta_vals[name])
            rmse = np.sqrt(np.mean((eta_sim_interp - eta_exp)**2))
            
            mask_exp = eta_exp != 0
            if np.any(mask_exp):
                mape = np.mean(np.abs((eta_sim_interp[mask_exp] - eta_exp[mask_exp]) / eta_exp[mask_exp])) * 100
            else:
                mape = np.nan
            
            all_results.append({
                "name": name,
                "type": "point",
                "RMSE": rmse,
                "MAPE (%)": mape
            })
        except FileNotFoundError:
            pass

    # Section metrics
    for name, ix in line_indices.items():
        try:
            z_sim = data_cut_main.z_b.values[-1, :, ix]
            y_sim = data_cut_main.y.values
            
            df = pd.read_csv(f"experimental_results/{name}_exp.csv")
            df = df.sort_values(by=df.columns[0])
            y_exp = df.iloc[:, 0].values
            z_exp = df.iloc[:, 1].values
            
            z_sim_interp = np.interp(y_exp, y_sim[::-1], z_sim[::-1])
            rmse = np.sqrt(np.mean((z_sim_interp - z_exp)**2))
            
            mask_exp = z_exp != 0
            if np.any(mask_exp):
                mape = np.mean(np.abs((z_sim_interp[mask_exp] - z_exp[mask_exp]) / z_exp[mask_exp])) * 100
            else:
                mape = np.nan
            
            all_results.append({
                "name": name,
                "type": "section",
                "RMSE": rmse,
                "MAPE (%)": mape
            })
        except FileNotFoundError:
            pass

    df_all = pd.DataFrame(all_results)
    file_path = os.path.join(folder, "metrics_all.csv")
    df_all.to_csv(file_path, index=False)


def plot_images():
    print("Generating frame images and GIFs...")
    g = 9.81

    nt = data_main.sizes["t"]
    time_array = np.arange(0, nt * dt, dt)
    mask = time_array <= t_max
    time_array_cut = time_array[mask]
    data_cut = data_main.isel(t=mask)

    y_target = 0.6925
    iy = np.argmin(np.abs(data_cut.y.values - y_target)) 

    frames_folder = "results/dambreak_frames"
    os.makedirs(frames_folder, exist_ok=True)

    frame_paths = []

    for i in range(data_cut.sizes["t"]):
        h_slice = data_cut.h.values[i, iy, :]
        hu_slice = data_cut.hu.values[i, iy, :]
        zb_slice = data_cut.z_b.values[i, iy, :]
        
        u_slice = np.zeros_like(h_slice)
        mask_h_slice = h_slice > 1e-6
        u_slice[mask_h_slice] = hu_slice[mask_h_slice] / h_slice[mask_h_slice]

        froude = np.full_like(h_slice, np.nan)
        froude[mask_h_slice] = np.abs(u_slice[mask_h_slice]) / np.sqrt(g * h_slice[mask_h_slice])
            
        eta = h_slice + zb_slice
        
        fig = plt.figure(figsize=(10, 8))
        gs = fig.add_gridspec(3, 1, height_ratios=[1.8, 1, 1])

        ax1 = fig.add_subplot(gs[0])
        ax2 = fig.add_subplot(gs[1])
        ax3 = fig.add_subplot(gs[2])

        ax1.plot(data_cut.x.values, eta, c="black", linewidth=2)
        ax1.set_ylabel(r"Superficie libre $\eta$ (m)")
        ax1.set_title(f"t = {time_array_cut[i]:.1f} s")
        ax1.set_xlim(0, 6.725)

        ax2.plot(data_cut.x.values, zb_slice, c="black", linewidth=2)
        ax2.set_ylabel(r"Altura del lecho $z_b$ (m)")
        ax2.set_xlim(0, 6.725)

        ax3.plot(data_cut.x.values, froude, c="black", linewidth=2)
        ax3.axhline(1.0, linestyle='--', alpha=0.5, c='r', label='Flujo crítico', linewidth=2)
        ax3.set_ylabel(r"Número de Froude")
        ax3.set_xlabel("x (m)")
        ax3.set_xlim(0, 6)

        plt.tight_layout()
        
        frame_path = f"{frames_folder}/frame_{i:03d}.png"
        fig.savefig(frame_path, dpi=200)
        plt.close()
        
        frame_paths.append(frame_path)

    images = [imageio.imread(fp) for fp in frame_paths]
    if len(images) > 0:
        imageio.mimsave("results/dambreak_h_z_Fr.gif", images, fps=10)

    frames_2d_folder = "results/dambreak_2D_frames"
    os.makedirs(frames_2d_folder, exist_ok=True)
    frame_paths = []

    eta_min = np.nanmin(data_cut.h.values + data_cut.z_b.values)
    eta_max = np.nanmax(data_cut.h.values + data_cut.z_b.values)

    for i in range(data_cut.sizes["t"]):
        eta = data_cut.h.values[i] + data_cut.z_b.values[i]
        
        plt.figure(figsize=(8, 3))
        plt.imshow(eta, origin='upper', extent=[data_cut.x.min(), data_cut.x.max(), data_cut.y.min(), data_cut.y.max()],
                   aspect='auto', cmap='jet', vmin=eta_min, vmax=eta_max)
        
        plt.colorbar(label="Superficie libre $\eta$ (m)")
        plt.title(f"t = {time_array_cut[i]:.1f} s")
        plt.xlabel("$x$ (m)")
        plt.ylabel("$y$ (m)")
        
        plt.tight_layout()
        frame_path = f"{frames_2d_folder}/frame_{i:03d}.png"
        plt.savefig(frame_path, dpi=200)
        plt.close()
        
        frame_paths.append(frame_path)

    images = [imageio.imread(fp) for fp in frame_paths]
    if len(images) > 0:
        imageio.mimsave("results/dambreak_2D.gif", images, fps=10)


def plot_fields_2x2():
    print("Generating 2x2 fields animations...")
    nt = data_main.sizes["t"]
    time_array = np.arange(0, nt * dt, dt)
    mask = time_array <= t_max
    time_array_cut = time_array[mask]
    data_cut = data_main.isel(t=mask)

    fields_folder = "results/fields"
    os.makedirs(fields_folder, exist_ok=True)
    field_paths = []

    # Get extent
    extent = [data_cut.x.min().values, data_cut.x.max().values, data_cut.y.min().values, data_cut.y.max().values]

    # Global min/max to prevent flickering
    try:
        z_global = data_cut.z.values
    except AttributeError:
        z_global = np.zeros_like(data_cut.h.values)
        
    try:
        G_global = data_cut.G.values
        G_min, G_max = np.nanmin(G_global), np.nanmax(G_global)
    except AttributeError:
        G_min, G_max = 0.0, 1.0

    h_total_global = data_cut.h.values + data_cut.z_b.values + z_global
    bed_global = data_cut.z_b.values + z_global
    
    eps = 1e-12
    u_global = data_cut.hu.values / (data_cut.h.values + eps)
    v_global = data_cut.hv.values / (data_cut.h.values + eps)
    umag_global = np.sqrt(u_global**2 + v_global**2)
    
    h_tot_min, h_tot_max = np.nanmin(h_total_global) + 1e-5, np.nanmax(h_total_global)
    if h_tot_min <= 0: h_tot_min = 1e-3
    bed_min, bed_max = np.nanmin(bed_global), np.nanmax(bed_global)
    umag_min, umag_max = np.nanmin(umag_global), np.nanmax(umag_global)

    if G_max <= G_min: G_max = G_min + 1e-3
    if h_tot_max <= h_tot_min: h_tot_max = h_tot_min + 1e-3
    if bed_max <= bed_min: bed_max = bed_min + 1e-3
    if umag_max <= umag_min: umag_max = umag_min + 1e-3

    from matplotlib.colors import LogNorm

    for i in range(data_cut.sizes["t"]):
        h_field = data_cut.h.values[i]
        zb_field = data_cut.z_b.values[i]
        hu_field = data_cut.hu.values[i]
        hv_field = data_cut.hv.values[i]
        
        try:
            G_field = data_cut.G.values[i]
        except AttributeError:
            G_field = np.zeros_like(h_field)
            
        try:
            z_field = data_cut.z.values[i]
        except AttributeError:
            z_field = np.zeros_like(h_field)

        eps = 1e-12
        u_field = hu_field / (h_field + eps)
        v_field = hv_field / (h_field + eps)
        umag_field = np.sqrt(u_field**2 + v_field**2)

        fig, axs = plt.subplots(2, 2, figsize=(12, 6), constrained_layout=True)
        fig.suptitle(f"Time: {time_array_cut[i]:.1f} s")

        h_total = h_field + zb_field + z_field
        
        # --- Free surface ---
        try:
            im0 = axs[0, 0].imshow(h_total, origin='upper', cmap="jet", norm=LogNorm(vmin=h_tot_min, vmax=h_tot_max), extent=extent)
        except Exception:
            im0 = axs[0, 0].imshow(h_total, origin='upper', cmap="jet", extent=extent, vmin=h_tot_min, vmax=h_tot_max)
            
        axs[0, 0].set_title(r"Free surface $(h + z_b + z) [m]$")
        plt.colorbar(im0, ax=axs[0, 0])

        # --- Bed elevation ---
        im1 = axs[0, 1].imshow(zb_field + z_field, origin='upper', cmap="jet", extent=extent, vmin=0.0, vmax=0.5)
        axs[0, 1].set_title(r"Bed elevation $(z_b + z) [m]$")
        plt.colorbar(im1, ax=axs[0, 1])

        # --- Exner flux / G ---
        im2 = axs[1, 0].imshow(G_field, origin='upper', cmap="jet", extent=extent, vmin=G_min, vmax=G_max)
        axs[1, 0].set_title(r"Interaction factor $G [s^2/m]$")
        plt.colorbar(im2, ax=axs[1, 0])

        # --- Velocity magnitude ---
        im3 = axs[1, 1].imshow(umag_field, origin='upper', cmap="jet", extent=extent, vmin=umag_min, vmax=umag_max)
        axs[1, 1].set_title(r"Velocity magnitude $|u| [m/s]$")
        plt.colorbar(im3, ax=axs[1, 1])

        for ax in axs.flat:
            ax.axis("equal")

        field_path = f"{fields_folder}/fields_{i:04d}.png"
        plt.savefig(field_path, dpi=200)
        plt.close(fig)
        field_paths.append(field_path)

    images_fields = [imageio.imread(fp) for fp in field_paths]
    if len(images_fields) > 0:
        imageio.mimsave("results/fields_2x2.gif", images_fields, fps=10)


if __name__ == "__main__":
    plot_domain()
    plot_time_series()
    plot_line_profiles()
    compute_metrics()
    plot_fields_2x2()
    print("Done!")
