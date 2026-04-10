# import xarray as xr
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt

# # Assuming 'data' is your xarray dataset
# data = xr.open_dataset("90_bend_channel_domain_out.nc")

# g = 9.81
# total_time = 115
# dt = 0.5

# # N_half = data.h.shape[1] // 2
# time_array = np.arange(0, total_time, dt)

# z = data.z_b.isel(t=-1)

# # Section 1
# x_target = 6.34
# ix = np.argmin(np.abs(data.x.values - x_target))
# print(z.y.values[:5])
# print(z.y.values[-5:])

# z_sorted = z.sortby("y")
# profile = z_sorted.isel(x=ix).sel(
#     y=slice(0.445, 0.445 + 0.495 + 2.88)
# )
# print(profile)

# plt.figure(figsize=(10,5))
# plt.plot(profile.y, profile.values)
# plt.ylim(0.0, 0.14)
# plt.xlim(0.0, 3.5)
# plt.yticks([0, 0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.14], [0, 0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.14])
# plt.grid(alpha=0.2)
# plt.savefig("lastz_b_S1.png", dpi=250)
# plt.close()

# __________________________________________

# Section 2:

# y_target = 9.2/2 + 1.45
# iy = np.argmin(np.abs(data.y.values - y_target))

# profile = z.isel(y=iy).sel(x=slice(1.76 + 10.33 + 0.5, 1.76 + 10.33 + 0.5 + 8))

# plt.figure(figsize=(10,5))
# plt.plot(profile.x - (1.76 + 10.33 + 0.5) + 0.5, profile.values)
# plt.ylim(0.0, 0.15)
# plt.xlim(0.0, 8.0)
# plt.yticks([0, 0.05, 0.10, 0.15], [0, 0.05, 0.10, 0.15]) #
# plt.grid(alpha=0.2)
# plt.savefig("lastz_b_S2.png", dpi=250)
# plt.close()

# for i, idt in enumerate(time_array):

#     G = data.G[i].sel(x=slice(10, 24), y=slice(9.2/2+0.5+1.3, 9.2/2-0.5-1.3))
#     h = data.h[i].sel(x=slice(10, 24), y=slice(9.2/2+0.5+1.3, 9.2/2-0.5-1.3))
#     z = data.z[i].sel(x=slice(10, 24), y=slice(9.2/2+0.5+1.3, 9.2/2-0.5-1.3))
#     z_b = data.z_b[i].sel(x=slice(10, 24), y=slice(9.2/2+0.5+1.3, 9.2/2-0.5-1.3))

#     eps = 1e-12
#     u = data.hu[i].sel(x=slice(10, 24),
#                     y=slice(9.2/2+0.5+1.3, 9.2/2-0.5-1.3)) / (h + eps)

#     v = data.hv[i].sel(x=slice(10, 24),
#                     y=slice(9.2/2+0.5+1.3, 9.2/2-0.5-1.3)) / (h + eps)

#     umag = np.sqrt(u**2 + v**2)

#     fig, axs = plt.subplots(2, 2, figsize=(12, 6), constrained_layout=True)

#     fig.suptitle(f"Time: {idt} s")

#     extent = [G.x.min(), G.x.max(), G.y.min(), G.y.max()]

#     # --- Free surface ---
#     im0 = axs[0, 0].imshow(h + z_b + z,
#                         cmap="jet",
#                         extent=extent)
#     axs[0, 0].set_title(r"Free surface $(h + z_b + z) [m]$")
#     plt.colorbar(im0, ax=axs[0, 0])
#     im0.set_clim(2e-2, 0.6)

#     # --- Bed elevation ---
#     im1 = axs[0, 1].imshow(z_b + z,
#                         cmap="jet",
#                         extent=extent)
#     axs[0, 1].set_title(r"Bed elevation $(z_b + z) [m]$")
#     plt.colorbar(im1, ax=axs[0, 1])
#     im1.set_clim(0.1, 0.3)

#     # --- Exner flux / G ---
#     im2 = axs[1, 0].imshow(G,
#                         cmap="jet",
#                         extent=extent)
#     axs[1, 0].set_title(r"Interaction factor $G [s^2/m]$")
#     plt.colorbar(im2, ax=axs[1, 0])
#     im2.set_clim(0.0, 0.002)

#     # --- Velocity magnitude ---
#     im3 = axs[1, 1].imshow(umag,
#                         cmap="jet",
#                         extent=extent)
#     axs[1, 1].set_title(r"Velocity magnitude $|u| [m/s]$")
#     plt.colorbar(im3, ax=axs[1, 1])
#     im3.set_clim(0.0, 4.0)

#     for ax in axs.flat:
#         ax.axis("equal")

#     plt.savefig(f"fields_{i:04d}.png", dpi=250)
#     plt.close()


# z = data.z_b.isel(t=-1)

# y_target = 9.2/2 + 0.70
# iy = np.argmin(np.abs(data.y.values - y_target))

# profile = z.isel(y=iy).sel(x=slice(1.76 + 10.33 + 0.5, 1.76 + 10.33 + 0.5 + 8))

# plt.figure(figsize=(10,5))
# plt.plot(profile.x - (1.76 + 10.33 + 0.5) + 0.5, profile.values)
# plt.ylim(0.0, 0.15)
# plt.xlim(0.0, 8.0)
# plt.yticks([0, 0.05, 0.10, 0.15], [0, 0.05, 0.10, 0.15])
# plt.grid(alpha=0.2)
# plt.savefig("lastz_b_S1.png", dpi=250)
# plt.close()

# y_target = 9.2/2 + 1.45
# iy = np.argmin(np.abs(data.y.values - y_target))

# profile = z.isel(y=iy).sel(x=slice(1.76 + 10.33 + 0.5, 1.76 + 10.33 + 0.5 + 8))

# plt.figure(figsize=(10,5))
# plt.plot(profile.x - (1.76 + 10.33 + 0.5) + 0.5, profile.values)
# plt.ylim(0.0, 0.15)
# plt.xlim(0.0, 8.0)
# plt.yticks([0, 0.05, 0.10, 0.15], [0, 0.05, 0.10, 0.15]) #
# plt.grid(alpha=0.2)
# plt.savefig("lastz_b_S2.png", dpi=250)
# plt.close()

# for i, idt in enumerate(time_array):

#     G = data.G[i].sel(x=slice(10, 24), y=slice(9.2/2+0.5+1.3, 9.2/2-0.5-1.3))
#     h = data.h[i].sel(x=slice(10, 24), y=slice(9.2/2+0.5+1.3, 9.2/2-0.5-1.3))
#     z = data.z[i].sel(x=slice(10, 24), y=slice(9.2/2+0.5+1.3, 9.2/2-0.5-1.3))
#     z_b = data.z_b[i].sel(x=slice(10, 24), y=slice(9.2/2+0.5+1.3, 9.2/2-0.5-1.3))

#     eps = 1e-12
#     u = data.hu[i].sel(x=slice(10, 24),
#                     y=slice(9.2/2+0.5+1.3, 9.2/2-0.5-1.3)) / (h + eps)

#     v = data.hv[i].sel(x=slice(10, 24),
#                     y=slice(9.2/2+0.5+1.3, 9.2/2-0.5-1.3)) / (h + eps)

#     umag = np.sqrt(u**2 + v**2)

#     fig, axs = plt.subplots(2, 2, figsize=(12, 6), constrained_layout=True)

#     fig.suptitle(f"Time: {idt} s")

#     extent = [G.x.min(), G.x.max(), G.y.min(), G.y.max()]

#     # --- Free surface ---
#     im0 = axs[0, 0].imshow(h + z_b + z,
#                         cmap="jet",
#                         extent=extent)
#     axs[0, 0].set_title(r"Free surface $(h + z_b + z) [m]$")
#     plt.colorbar(im0, ax=axs[0, 0])
#     im0.set_clim(2e-2, 0.6)

#     # --- Bed elevation ---
#     im1 = axs[0, 1].imshow(z_b + z,
#                         cmap="jet",
#                         extent=extent)
#     axs[0, 1].set_title(r"Bed elevation $(z_b + z) [m]$")
#     plt.colorbar(im1, ax=axs[0, 1])
#     im1.set_clim(0.1, 0.3)

#     # --- Exner flux / G ---
#     im2 = axs[1, 0].imshow(G,
#                         cmap="jet",
#                         extent=extent)
#     axs[1, 0].set_title(r"Interaction factor $G [s^2/m]$")
#     plt.colorbar(im2, ax=axs[1, 0])
#     im2.set_clim(0.0, 0.002)

#     # --- Velocity magnitude ---
#     im3 = axs[1, 1].imshow(umag,
#                         cmap="jet",
#                         extent=extent)
#     axs[1, 1].set_title(r"Velocity magnitude $|u| [m/s]$")
#     plt.colorbar(im3, ax=axs[1, 1])
#     im3.set_clim(0.0, 4.0)

#     for ax in axs.flat:
#         ax.axis("equal")

#     plt.savefig(f"fields_{i:04d}.png", dpi=250)
#     plt.close()


# _______________________________________________________


# data = xr.open_dataset("90_bend_channel_domain_out.nc")

# g = 9.81
# total_time = 115
# dt = 0.5

# # N_half = data.h.shape[1] // 2
# time_array = np.arange(0, total_time, dt)

# z = data.z_b.isel(t=-1)

# # Section 1
# x_target = 6.34
# ix = np.argmin(np.abs(data.x.values - x_target))
# print(z.y.values[:5])
# print(z.y.values[-5:])

# z_sorted = z.sortby("y")
# profile = z_sorted.isel(x=ix).sel(
#     y=slice(0.445, 0.445 + 0.495 + 2.88)
# )
# print(profile)

# plt.figure(figsize=(10,5))
# plt.plot(profile.y, profile.values)
# plt.ylim(0.0, 0.14)
# plt.xlim(0.0, 3.5)
# plt.yticks([0, 0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.14], [0, 0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.14])
# plt.grid(alpha=0.2)
# plt.savefig("lastz_b_S1.png", dpi=250)
# plt.close()


import xarray as xr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# =========================
# CARGA DE DATOS
# =========================
data = xr.open_dataset("90_bend_channel_domain_out.nc")

total_time = float(data.t.values[-1])
dt = float(data.t.values[1] - data.t.values[0])
time_array = np.arange(0, total_time + dt, dt)

markers = ["o", "s", "x", "+", "^", "d"]
colors = ["red", "orange", "green", "black", "purple", "brown"]

# =========================
# FUNCIONES AUXILIARES
# =========================
def experimental_envelope(exp_dfs, xcol=0, ycol=1, npts=300):
    x_common = np.linspace(
        min(df.iloc[:, xcol].min() for df in exp_dfs),
        max(df.iloc[:, xcol].max() for df in exp_dfs),
        npts
    )

    Y = []
    for df in exp_dfs:
        df = df.sort_values(df.columns[xcol])
        y_interp = np.interp(x_common, df.iloc[:, xcol], df.iloc[:, ycol])
        Y.append(y_interp)

    Y = np.array(Y)
    return x_common, Y.min(axis=0), Y.max(axis=0)

# =========================
# PERFILES DE SECCIÓN (S1, S2)
# =========================
def plot_line_profiles():

    lines = {
        "S1": 6.34,
        "S2": 6.60
    }

    z = data.z_b.isel(t=-1).sortby("y")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    axes = axes.flatten()

    for i, (name, x_target) in enumerate(lines.items()):
        ax = axes[i]

        ix = np.argmin(np.abs(data.x.values - x_target))

        profile = z.isel(x=ix).sel(
            y=slice(0.445, 0.445 + 0.495 + 2.88)
        )

        # =========================
        # EXPERIMENTAL (ENVELOPE)
        # =========================
        expfiles = [
            f for f in os.listdir("experimental_data")
            if name in f and "exp" in f
        ]

        expdfs = []
        for j, file in enumerate(expfiles):
            df = pd.read_csv(os.path.join("experimental_data", file))
            ax.scatter(
                df.iloc[:, 0],
                df.iloc[:, 1],
                marker=markers[j],
                color="black",
                s=10,
                zorder=1,
                label=f"Exp {file.split('exp')[-1][:-4]}"
            )
            expdfs.append(df)

        if expdfs:
            y_common, Ymin, Ymax = experimental_envelope(expdfs)
            ax.fill_between(
                y_common,
                Ymin,
                Ymax,
                color="gray",
                alpha=0.5,
                edgecolor="none",
                label="Experimental envelope",
                zorder=0
            )

        # =========================
        # MODELOS NUMÉRICOS
        # =========================
        files = [
            f for f in os.listdir("experimental_data")
            if name in f and "exp" not in f
        ]

        for j, file in enumerate(files):
            df = pd.read_csv(os.path.join("experimental_data", file))
            ax.plot(
                df.iloc[:, 0],
                df.iloc[:, 1],
                color=colors[j],
                zorder=2,
                label=file[3:-4]
            )

        # =========================
        # PYEXNER
        # =========================
        ax.plot(
            profile.y.values - profile.y.values.min(),
            profile.values,
            color="blue",
            zorder=3,
            label="PyExner"
        )

        ax.set_title(f"{name} (x = {x_target})", fontweight="bold")
        ax.set_xlabel("y [m]")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize="small")

    axes[0].set_ylabel("z [m]")
    fig.tight_layout()
    fig.savefig("sections.png", dpi=250)
    plt.close()

# =========================
# SERIES TEMPORALES (G1–G6)
# =========================
def plot_time_series():

    points = {
        "G1": (2.00, 2.00),
        "G2": (2.74, 0.69),
        "G3": (4.24, 0.69),
        "G4": (5.74, 0.69),
        "G5": (6.48, 1.95),
        "G6": (6.48, 3.45),
    }

    point_indices = {}
    for name, (px, py) in points.items():
        ix = np.argmin(np.abs(data.x.values - px))
        iy = np.argmin(np.abs(data.y.values - py))
        point_indices[name] = (iy, ix)

    eta_vals = {name: [] for name in points}

    for t in range(len(data.t)):
        for name, (iy, ix) in point_indices.items():
            eta_vals[name].append(
                data.z_b.isel(t=t, y=iy, x=ix).values
            )

    fig, ax = plt.subplots(figsize=(8, 4))

    for i, name in enumerate(points):
        ax.plot(
            time_array,
            eta_vals[name],
            color=colors[i],
            label=name,
            zorder=2
        )

        expfiles = [
            f for f in os.listdir("experimental_data")
            if name in f and "exp" in f
        ]

        for j, file in enumerate(expfiles):
            df = pd.read_csv(os.path.join("experimental_data", file))
            ax.scatter(
                df.iloc[:, 0],
                df.iloc[:, 1],
                marker=markers[j],
                color="black",
                s=12,
                zorder=3
            )

    ax.set_xlabel("Time [s]")
    ax.set_ylabel("z [m]")
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=3, fontsize="small")
    fig.tight_layout()
    fig.savefig("time_series.png", dpi=250)
    plt.close()

# =========================
# EJECUCIÓN
# =========================
plot_line_profiles()
plot_time_series()