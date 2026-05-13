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

Cross sections:
S1 -> y = 0.2 m
S2 -> y = 0.7 m
S3 -> y = 1.45 m
"""

# =============================================================================
# OUTPUT DIRECTORIES
# =============================================================================

RESULTS_DIR = "results"
FIELDS_DIR = os.path.join(RESULTS_DIR, "fields")
DAMBREAK_DIR = os.path.join(RESULTS_DIR, "dambreak")
PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIELDS_DIR, exist_ok=True)
os.makedirs(DAMBREAK_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

# =============================================================================
# LOAD DATA
# =============================================================================

data = xr.open_dataset("erodible_channel_out.nc", engine="netcdf4")

print(data)

g = 9.81
total_time = 20.0
dt = 0.25

time_array = np.arange(0, total_time, dt)

markers = ["o", "s", "x", "+"]
colors = ["red", "orange", "green", "black"]


# =============================================================================
# HELPERS
# =============================================================================

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


# =============================================================================
# LINE PROFILES
# =============================================================================

def plot_line_profiles():

    lines = {
        "S1": 0.2,
        "S2": 0.7,
        "S3": 1.45,
    }

    z_b = data.z_b.isel(t=-1)
    z = data.z.isel(t=-1)

    line_indices = {}

    for name, y_coord in lines.items():

        iy = np.argmin(np.abs(data.y.values - y_coord))

        line_indices[name] = iy

    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)

    axes = axes.flatten()

    for i, (name, iy) in enumerate(line_indices.items()):

        ax = axes[i]

        profile = z_b.isel(y=iy).sel(x=slice(0.5, 9.0))

        # ---------------------------------------------------------------------
        # Experimental envelope
        # ---------------------------------------------------------------------

        expfiles = [
            x for x in os.listdir("experimental_results")
            if name in x and "exp" in x and "S" in name
        ]

        expdfs = []

        for j, file in enumerate(expfiles):

            df = pd.read_csv(
                os.path.join("experimental_results", file),
                skipinitialspace=True
            ).sort_values("x")

            ax.scatter(
                df.iloc[:, 0],
                df.iloc[:, 1],
                label=f"Exp {file.split('exp')[-1][:-4]}",
                marker=markers[j % len(markers)],
                color="black",
                s=10,
                zorder=1
            )

            expdfs.append(df)

        if len(expdfs) > 0:

            x_common, Ymin, Ymax = experimental_envelope(expdfs)

            ax.fill_between(
                x_common,
                Ymin,
                Ymax,
                color="gray",
                alpha=0.5,
                edgecolor="none",
                label="Experimental envelope" if i == 0 else None,
                zorder=0
            )

        # ---------------------------------------------------------------------
        # Numerical references
        # ---------------------------------------------------------------------

        files = [
            x for x in os.listdir("experimental_results")
            if name in x and "exp" not in x
        ]

        for j, file in enumerate(files):

            df = pd.read_csv(
                os.path.join("experimental_results", file),
                skipinitialspace=True
            ).sort_values("x")

            ax.plot(
                df.iloc[:, 0],
                df.iloc[:, 1],
                label=f"{file[3:-4]}",
                c=colors[j % len(colors)],
                zorder=2
            )

        # ---------------------------------------------------------------------
        # PyExner
        # ---------------------------------------------------------------------

        ax.plot(
            profile.x,
            profile.values,
            label="PyExner",
            zorder=3,
            color="blue"
        )

        ax.set_title(
            f"Cross-section at {name} (y={lines[name]})",
            fontweight="bold"
        )

        ax.grid(True, alpha=0.3)

        ax.legend(fontsize="small", ncol=2)

        ax.set_xlim(0.0, 8.0)
        ax.set_ylim(-0.05, 0.15)

    plt.tight_layout()

    output_path = os.path.join(PLOTS_DIR, "lines.png")

    fig.savefig(output_path, dpi=250)

    plt.close()

    print(f"Saved: {output_path}")


# =============================================================================
# TIME SERIES
# =============================================================================

def plot_time_series():

    points = {
        "G1": [0.64, -0.50],
        "G2": [0.64, -0.165],
        "G3": [1.94, -0.99],
        "G4": [1.94, -0.33],
    }

    point_indices = {}

    for name, coords in points.items():

        px, py = coords

        idx_x = np.argmin(np.abs(data.x.values - px))
        idx_y = np.argmin(np.abs(data.y.values - py))

        point_indices[name] = (idx_y, idx_x)

    eta_vals = {name: [] for name in points.keys()}

    for time_step in range(len(time_array)):

        for name, (iy, ix) in point_indices.items():

            val = (
                data.z_b.values[time_step, iy, ix]
                + data.z.values[time_step, iy, ix]
                + data.h.values[time_step, iy, ix]
            )

            eta_vals[name].append(val)

    for name in eta_vals:
        eta_vals[name] = np.array(eta_vals[name])

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(12, 8),
        sharey=True,
        sharex=True
    )

    axes = axes.flatten()

    for i, (name, values) in enumerate(eta_vals.items()):

        ax = axes[i]

        files = [
            x for x in os.listdir("experimental_results")
            if name in x
        ]

        for j, file in enumerate(files):

            df = pd.read_csv(
                os.path.join("experimental_results", file),
                skipinitialspace=True
            ).sort_values("t" if "G" in name else "x")

            if "Exp" in file or "exp" in file:

                ax.scatter(
                    df.iloc[:, 0],
                    df.iloc[:, 1],
                    label=f"{file[3:-4]}",
                    marker=".",
                    c=colors[j % len(colors)],
                    zorder=2
                )

            else:

                ax.plot(
                    df.iloc[:, 0],
                    df.iloc[:, 1],
                    label=f"{file[3:-4]}",
                    c=colors[j % len(colors)],
                    zorder=2
                )

        ax.plot(
            time_array,
            values,
            color="tab:blue",
            linewidth=1.5,
            label="PyExner"
        )

        ax.set_title(f"Point {name}", fontweight="bold")

        ax.grid(True, alpha=0.3)

        ax.legend(fontsize="small", ncol=1)

        if i >= 2:
            ax.set_xlabel("Time (s)")

        if i % 2 == 0:
            ax.set_ylabel("Z")

    plt.suptitle(
        "Individual Comparisons of Point Data",
        fontsize=16
    )

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    output_path = os.path.join(PLOTS_DIR, "Gauge_points.png")

    plt.savefig(output_path, dpi=250)

    plt.close()

    print(f"Saved: {output_path}")


# =============================================================================
# FIELD IMAGES
# =============================================================================

def plot_images():

    for i, idt in enumerate(time_array):

        print(f"Timestep: {i}")

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

        fig, axs = plt.subplots(
            2,
            2,
            figsize=(12, 6),
            constrained_layout=True
        )

        fig.suptitle(f"Time: {idt} s")

        extent = [G.x.min(), G.x.max(), G.y.min(), G.y.max()]

        # ---------------------------------------------------------------------
        # Free surface
        # ---------------------------------------------------------------------

        im0 = axs[0, 0].imshow(
            h + z_b + z,
            cmap="jet",
            extent=extent
        )

        axs[0, 0].set_title(r"Free surface $(h + z_b + z) [m]$")

        plt.colorbar(im0, ax=axs[0, 0])

        im0.set_clim(1e-1, 0.6)

        # ---------------------------------------------------------------------
        # Bed elevation
        # ---------------------------------------------------------------------

        im1 = axs[0, 1].imshow(
            z_b + z,
            cmap="jet",
            extent=extent
        )

        axs[0, 1].set_title(r"Bed elevation $(z_b + z) [m]$")

        plt.colorbar(im1, ax=axs[0, 1])

        im1.set_clim(0.0, 0.12)

        # ---------------------------------------------------------------------
        # Interaction factor
        # ---------------------------------------------------------------------

        im2 = axs[1, 0].imshow(
            G,
            cmap="jet",
            extent=extent
        )

        axs[1, 0].set_title(r"Interaction factor $G [s^2/m]$")

        plt.colorbar(im2, ax=axs[1, 0])

        im2.set_clim(0.0, 0.002)

        # ---------------------------------------------------------------------
        # Velocity magnitude
        # ---------------------------------------------------------------------

        im3 = axs[1, 1].imshow(
            umag,
            cmap="jet",
            extent=extent
        )

        axs[1, 1].set_title(r"Velocity magnitude $|u| [m/s]$")

        plt.colorbar(im3, ax=axs[1, 1])

        im3.set_clim(0.0, 4.0)

        for ax in axs.flat:
            ax.axis("equal")

        output_field = os.path.join(
            FIELDS_DIR,
            f"fields_{i:04d}.png"
        )

        plt.savefig(output_field, dpi=250)

        plt.close()

    print(f"Saved field plots to: {FIELDS_DIR}")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":

    plot_line_profiles()

    plot_time_series()

    plot_images()


    print("\nAll results stored in:")
    print(f"  {RESULTS_DIR}/")