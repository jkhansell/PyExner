# Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import xarray as xr
from netCDF4 import Dataset
import os
import imageio

# results folder
folder = 'results'
os.makedirs(folder, exist_ok=True)

# Monte Carlo runs
num_runs = 10
datasets = [] # dataset for runs

# dataset[0] is for 'exact' simulation
ds = Dataset(f"T_domain_out.nc")
data = xr.open_dataset(xr.backends.NetCDF4DataStore(ds))
datasets.append(data) 

# from dataset[1] to num_runs are the Monte Carlo runs
for i in range(num_runs):
    ds = Dataset(f"run_{i+1}/T_domain_out_{i+1}.nc")
    data = xr.open_dataset(xr.backends.NetCDF4DataStore(ds))
    datasets.append(data)


# time stepping parameters (modify dt or t_max to change simulation window)
dt = 0.25
t_max = 20.0 # for analysis

################## Points and sections of interest ##################
# define points of interest (modify coordinates to analyze other locations)
points = {
    "G1": [12.73, 4.1],
    "G2": [12.73, 4.435],
    "G3": [14.03, 3.61],
    "G4": [14.03, 4.27],
}

# define horizontal sections (change values or orientation if needed)
lines = {
    "S1": 4.8, 
    "S2": 5.3,
    "S3": 6.05,
}

# find nearest grid indices for each point
point_indices = {}
for name, (px, py) in points.items():
    ix = np.argmin(np.abs(datasets[0].x.values - px))
    iy = np.argmin(np.abs(datasets[0].y.values - py))
    point_indices[name] = (iy, ix)

# find nearest x-index for each section
line_indices = {}
for name, py in lines.items():
    iy = np.argmin(np.abs(datasets[0].y.values - py))
    line_indices[name] = iy

# domain verification
plt.figure(figsize=(8, 4), dpi=100)

# initial water height field
plt.imshow(datasets[0].h[0]+datasets[0].z[0],
           origin='upper', 
           extent=[datasets[0].x.min(), datasets[0].x.max(),
                   datasets[0].y.min(), datasets[0].y.max()],
           aspect='auto',
           cmap='jet')

plt.colorbar(label="Altura inicial del agua $h$ (m)")

# plot points
for name, (iy, ix) in point_indices.items():
    x_val = datasets[0].x.values[ix]
    y_val = datasets[0].y.values[iy]
    
    plt.scatter(x_val, y_val, color='red')
    plt.text(x_val, y_val, name, color='white', fontsize=9, fontweight='bold')

# plot horizontal sections
x_min_line = 12.59
x_max_line = datasets[0].x.max()

for name, iy in line_indices.items():
    y_val = datasets[0].y.values[iy]

    plt.hlines(y=y_val,
               xmin=x_min_line,
               xmax=x_max_line,
               colors='r',
               linestyles='--',
               linewidth=1.5,
               alpha=0.8)

    plt.text(x_max_line, y_val, f' {name}',
             color='white', fontweight='bold',
             va='top', ha='right')

plt.title("Puntos y secciones en el domino $(t=0)$")
plt.xlabel("$x$ (m)")
plt.ylabel("$y$ (m)")

plt.tight_layout()

# save domain figure
file_path = os.path.join(folder, "domain.png")
plt.savefig(file_path, dpi=300, bbox_inches='tight')
plt.close()

################## Points results ##################
fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
axes = axes.flatten()

# extract point names and coordinates
names = list(points.keys())
coords_x = [p[0] for p in points.values()]
coords_y = [p[1] for p in points.values()]

# store monte carlo runs
all_runs_eta = {name: [] for name in points.keys()}

# loop over datasets (each run)
for j, data in enumerate(datasets):

    nt = data.sizes["t"]
    time_array = np.arange(0, nt * dt, dt)

    # truncate time to desired window
    mask = time_array <= t_max
    time_array_cut = time_array[mask]

    data_cut = data.isel(t=mask)

    # plot
    for i, name in enumerate(points.keys()):
        ax = axes[i]

        px, py = points[name]

        # extract nearest point time series
        ds_point = data_cut.sel(x=px, y=py, method="nearest")

        # free surface = h + zb
        eta_series = ds_point.h.values + ds_point.z_b.values

        if j == 0:
            # main simulation line
            ax.plot(
                time_array_cut,
                eta_series,
                linewidth=1.5,
                label="Simulación",
                color='blue', 
                zorder=10 
            )
        else: 
            # monte carlo runs
            ax.plot(
                time_array_cut,
                eta_series,
                linewidth=1,
                alpha=0.5,
                color ='skyblue',
                label="Monte Carlo" if j == 1 else None
            )
            all_runs_eta[name].append(eta_series)

# compute and plot uncertainty band
for i, name in enumerate(points.keys()):
    ax = axes[i]
    if all_runs_eta[name]:
        runs_matrix = np.array(all_runs_eta[name])
        eta_min = np.min(runs_matrix, axis=0)
        eta_max = np.max(runs_matrix, axis=0)
        
        # shaded region
        ax.fill_between(time_array_cut, eta_min, eta_max, color='gray', alpha=0.2, zorder=1)
        
    if i == 0: ax.legend()

# experimental data
for i, name in enumerate(points.keys()):
    ax = axes[i]

    try:
        # load experimental data 
        df = pd.read_csv(f"experimental_results/{name}_exp.csv")

        t_exp = df.iloc[:, 0].values
        eta_exp = df.iloc[:, 1].values

        ax.plot(
            t_exp,
            eta_exp,
            'o',
            label="Experimental",
            markersize=2.5,
            color="orange"
        )

    except FileNotFoundError:
        print(f"No se encontró archivo para {name}")

    # axis limits (adjust if needed)
    ax.set_xlim(0, 20)

    if i in [0,1]:
        ax.set_ylim(0, 0.3)
    else:
        ax.set_ylim(0, 0.2)

    # style
    ax.set_title(f"Punto {name}", fontweight='bold')
    ax.grid(True, alpha=0.3)

    if i >= 2:
        ax.set_xlabel("Tiempo (s)")
    if i % 2 == 0:
        ax.set_ylabel("Superficie libre $\eta$ (m)")

    ax.legend(fontsize='small')

plt.suptitle("Simulación (múltiples corridas) vs Experimental", fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])

# save figure
file_path = os.path.join(folder, "points_sim_vs_exp.png")
plt.savefig(file_path, dpi=300, bbox_inches='tight')
plt.close()


################### Section results ##################
fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharey=True)
axes = axes.flatten()

# container for monte carlo profiles
all_runs_zb = {name: [] for name in lines.keys()}

# loop over datasets
x_min_line = 12.59
x_max_line = 20
for j, data in enumerate(datasets):
    nt = data.sizes["t"]
    time_array = np.arange(0, nt * dt, dt)
    mask = time_array <= 20.0
    data_cut = data.isel(t=mask)
    
    t_idy = -1

    for i, (name, y_val) in enumerate(lines.items()):
        ax = axes[i]
        
        iy = np.argmin(np.abs(data_cut.y.values - y_val))
        
        z_sim = data_cut.z_b.values[t_idy, iy, :]

        x_vals_full = data_cut.x.values
        mask_x = x_vals_full >= x_min_line

        x_vals = x_vals_full[mask_x]
        z_sim = z_sim[mask_x]

        if j == 0:
            mask_x = (x_vals >= x_min_line) & (x_vals <= x_max_line)
            ax.plot(x_vals[mask_x], z_sim[mask_x], color='blue', linewidth=2,
                    label="Simulación", zorder=10)
        else:
            mask_x = (x_vals >= x_min_line) & (x_vals <= x_max_line)
            ax.plot(x_vals[mask_x], z_sim[mask_x], color='skyblue', linewidth=0.8, alpha=0.3,
                    label="Monte Carlo" if j == 1 else None)
            all_runs_zb[name].append(z_sim[mask_x])
            
# uncertainty band
for i, name in enumerate(lines.keys()):
    ax = axes[i]
    if all_runs_zb[name]:
        nt = data.sizes["t"]
        time_array = np.arange(0, nt * dt, dt)
        mask = time_array <= 20.0
        data_cut = data.isel(t=mask)
        mask_x = (x_vals >= x_min_line) & (x_vals <= x_max_line)

        runs_matrix = np.array(all_runs_zb[name])
        z_min = np.min(runs_matrix, axis=0)
        z_max = np.max(runs_matrix, axis=0)
        
        ax.fill_between(x_vals[mask_x], z_min, z_max,
                        color='gray', alpha=0.2, zorder=1)

# experimental data
markers = ["o", "s", "x", "+"]

exp_data = {name: [] for name in lines.keys()}

for i, (name, y_val) in enumerate(lines.items()):
    ax = axes[i]

    for k in range(4):  # exp1, exp2, exp3, exp4
        try:
            df = pd.read_csv(f"experimental_results/{name}_exp{k+1}.csv")
            df = df.sort_values(by=df.columns[0])

            x_exp = df.iloc[:, 0].values + 12.09
            z_exp = df.iloc[:, 1].values

            ax.plot(x_exp, z_exp,
                    markers[k],
                    label=f"Experimental {k+1}",
                    markersize=3,
                    color='darkorange',
                    linestyle="None",
                    zorder=15)

            exp_data[name].append((x_exp, z_exp))

        except FileNotFoundError:
            print(f"No se encontró archivo {name}_exp{k+1}")

    ax.set_title(f"Sección {name} (y = {y_val} m)", fontweight='bold')
    ax.set_xlabel("x (m)")
    ax.grid(True, alpha=0.3)
    
    if i == 0:
        ax.set_ylabel("Altura del lecho $z_b$ (m)")

# experimental uncertainty band 
for i, name in enumerate(lines.keys()):
    ax = axes[i]

    if exp_data[name]:
        x_ref = exp_data[name][0][0]

        z_interp_list = []

        for x_exp, z_exp in exp_data[name]:
            # interpolate with the reference
            z_interp = np.interp(x_ref, x_exp, z_exp)
            z_interp_list.append(z_interp)

        z_matrix = np.array(z_interp_list)

        z_min = np.min(z_matrix, axis=0)
        z_max = np.max(z_matrix, axis=0)

        ax.fill_between(x_ref, z_min, z_max,
                        color='orange', alpha=0.15, zorder=5)

ax_empty = axes[3]
ax_empty.axis('off')

handles, labels = axes[0].get_legend_handles_labels()
ax_empty.legend(handles, labels, loc='center', fontsize='medium')

plt.suptitle("Perfiles transversales del lecho (t = 20 s)", fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])

# save figure
file_path = os.path.join(folder, "sections_sim_vs_exp.png")
plt.savefig(file_path, dpi=300, bbox_inches='tight')
plt.close()

################## Points and sections RMSE - MAPE ##################
# metrics container
all_results = []

# points metrics
data = datasets[0]
nt = data.sizes["t"]
time_array = np.arange(0, nt * dt, dt)
mask = time_array <= t_max
time_array_cut = time_array[mask]
data_cut = data.isel(t=mask)

# build simulation time series
eta_vals = {}
for name in points.keys():
    px, py = points[name]
    ds_point = data_cut.sel(x=px, y=py, method="nearest")
    eta_vals[name] = ds_point.h.values + ds_point.z_b.values

# compute metrics for points
for name in points.keys():
    try:
        df = pd.read_csv(f"experimental_results/{name}_exp.csv")
        df = df.dropna()
        t_exp = df.iloc[:, 0].values
        eta_exp = df.iloc[:, 1].values
        
        eta_sim_interp = np.interp(t_exp, time_array_cut, eta_vals[name]) # must interpolate to match with experimental data
        
        rmse = np.sqrt(np.mean((eta_sim_interp - eta_exp)**2))
        
        mask = eta_exp != 0
        mape = np.mean(np.abs((eta_sim_interp[mask] - eta_exp[mask]) / eta_exp[mask])) * 100
        
        # append result with type
        all_results.append({
            "name": name,
            "type": "point",
            "RMSE": rmse,
            "MAPE (%)": mape
        })
        
    except FileNotFoundError:
        print(f"No se encontró archivo para {name}")

# sections metrics (exp1–exp4)
for name, y_val in lines.items():
    rmse_list = []
    mape_list = []

    iy = np.argmin(np.abs(data_cut.y.values - y_val))
    
    z_sim = data_cut.z_b.values[-1, iy, :]
    x_sim = data_cut.x.values

    for k in range(4):
        try:
            df = pd.read_csv(f"experimental_results/{name}_exp{k+1}.csv")
            df = df.sort_values(by=df.columns[0])
            df = df.dropna()
            
            x_exp = df.iloc[:, 0].values + 12.09  
            z_exp = df.iloc[:, 1].values
            
            z_sim_interp = np.interp(x_exp, x_sim, z_sim)

            mask_valid = ~np.isnan(z_sim_interp) & ~np.isnan(z_exp)

            z_sim_interp = z_sim_interp[mask_valid]
            z_exp = z_exp[mask_valid]

            rmse = np.sqrt(np.mean((z_sim_interp - z_exp)**2))
            
            mask = z_exp != 0
            mape = np.mean(np.abs((z_sim_interp[mask] - z_exp[mask]) / z_exp[mask])) * 100

            rmse_list.append(rmse)
            mape_list.append(mape)
          

        except FileNotFoundError:
            print(f"No se encontró archivo para {name}")

    if rmse_list:
        all_results.append({
            "name": name,
            "type": "section",
            "RMSE": np.mean(rmse_list),
            "RMSE_std": np.std(rmse_list),
            "MAPE (%)": np.mean(mape_list),
            "MAPE_std (%)": np.std(mape_list)
        })
# save cs
df_all = pd.DataFrame(all_results)

file_path = os.path.join(folder, "metrics_all.csv")
df_all.to_csv(file_path, index=False)

################## Time behavior ##################
## dambreak_h_z_Fr
# define gravity
g = 9.81

# select y-section (modify y_target to change slice)
y_target = 4.6
iy = np.argmin(np.abs(data_cut.y.values - y_target)) 

# create folder for frames
folder = "results/dambreak_frames"
os.makedirs(folder, exist_ok=True)

# compute global limits for fixed axes (optional)
eta_all = data_cut.h.values + data_cut.z_b.values+ data_cut.z.values
zb_all = data_cut.z_b.values

h_all = data_cut.h.values
hu_all = data_cut.hu.values

# velocity field
u_all = np.zeros_like(h_all)
mask = h_all > 1e-6
u_all[mask] = hu_all[mask] / h_all[mask]

# froude field
froude_all = np.full_like(h_all, np.nan)
mask = h_all > 1e-6
froude_all[mask] = np.abs(u_all[mask]) / np.sqrt(g * h_all[mask])

# global min/max
eta_min, eta_max = np.nanmin(eta_all), np.nanmax(eta_all)
zb_min, zb_max = np.nanmin(zb_all), np.nanmax(zb_all)
fr_min, fr_max = np.nanmin(froude_all), np.nanmax(froude_all)

# generate frames
frame_paths = []

for i in range(data_cut.sizes["t"]):
    
    # extract slice at current time
    h_slice = data_cut.h.values[i, iy, :]
    hu_slice = data_cut.hu.values[i, iy, :]
    zb_slice = data_cut.z_b.values[i, iy, :]
    
    # velocity field
    u_slice = np.zeros_like(h_slice)
    mask = h_slice > 1e-6
    u_slice[mask] = hu_slice[mask] / h_slice[mask]

    # froude field
    froude = np.full_like(h_slice, np.nan)
    mask = h_slice > 1e-6
    froude[mask] = np.abs(u_slice[mask]) / np.sqrt(g * h_slice[mask])
        
    # free surface
    eta = h_slice + zb_slice
    
    # create figure
    fig = plt.figure(figsize=(10, 8))
    gs = fig.add_gridspec(3, 1, height_ratios=[1.8, 1, 1])

    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    ax3 = fig.add_subplot(gs[2])

    # free surface plot
    ax1.plot(data_cut.x.values, eta, c="black", linewidth=2)
    ax1.set_ylabel(r"Superficie libre $\eta$ (m)")
    ax1.set_title(f"t = {time_array[i]:.1f} s")
    ax1.set_xlim(0, 27)
    # ax1.set_ylim(eta_min, eta_max)

    # bed elevation
    ax2.plot(data_cut.x.values, zb_slice, c="black", linewidth=2)
    ax2.set_ylabel(r"Altura del lecho $z_b$ (m)")
    ax2.set_xlim(0, 27)
    # ax2.set_ylim(zb_min, zb_max)

    # froude number
    ax3.plot(data_cut.x.values, froude, c="black", linewidth=2)
    ax3.axhline(1.0, linestyle='--', alpha=0.5, c='r', label='Flujo crítico', linewidth=2)
    ax3.set_ylabel(r"Número de Froude")
    ax3.set_xlabel("x (m)")
    ax3.set_xlim(0, 27)
    # ax3.set_ylim(fr_min, fr_max)

    plt.tight_layout()
    
    # save frame
    frame_path = f"{folder}/frame_{i:03d}.png"
    fig.savefig(frame_path, dpi=200)
    plt.close()
    
    frame_paths.append(frame_path)

# build gif
images = [imageio.imread(fp) for fp in frame_paths]
imageio.mimsave("results/dambreak_h_z_Fr.gif", images, fps=6)

## dambreak_2D
# create folder for 2d frames
folder = "results/dambreak_2D_frames"
os.makedirs(folder, exist_ok=True)

# store frame paths
frame_paths = []

# compute global limits for consistent color scale
eta_min = np.nanmin(data_cut.h.values + data_cut.z_b.values+ data_cut.z.values)
eta_max = np.nanmax(data_cut.h.values + data_cut.z_b.values+ data_cut.z.values)

# time loop to generate frames
for i in range(data_cut.sizes["t"]):
    
    # compute free surface at current time
    eta = data_cut.h.values[i] + data_cut.z_b.values[i] + data_cut.z.values[i]
    
    # create figure
    plt.figure(figsize=(8, 3))
    
    # plot 2d field with fixed color limits
    plt.imshow(
        eta,
        origin='upper',
        extent=[data_cut.x.min(), data_cut.x.max(),
                data_cut.y.min(), data_cut.y.max()],
        aspect='auto',
        cmap='jet',
        vmin=eta_min,
        vmax=eta_max
    )
    
    # colorbar and labels
    plt.colorbar(label="Superficie libre $\eta$ (m)")
    plt.title(f"t = {time_array[i]:.1f} s")
    plt.xlabel("$x$ (m)")
    plt.ylabel("$y$ (m)")
    
    plt.tight_layout()
    
    # save frame
    frame_path = f"{folder}/frame_{i:03d}.png"
    plt.savefig(frame_path, dpi=200)
    plt.close()
    
    frame_paths.append(frame_path)

# build gif from frames
images = [imageio.imread(fp) for fp in frame_paths]
imageio.mimsave("results/dambreak_2D.gif", images, fps=5)


