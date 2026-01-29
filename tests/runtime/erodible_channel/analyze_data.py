import xarray as xr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Assuming 'data' is your xarray dataset
data = xr.open_dataset("erodible_channel_out.nc")
g = 9.81
total_time = 20.0
dt = 0.25
N_half = data.h.shape[1] // 2
time_array = np.arange(0, total_time, dt)

z = data.z_b.isel(t=-1)

y_target = 9.2/2 + 0.70
iy = np.argmin(np.abs(data.y.values - y_target))

profile = z.isel(y=iy).sel(x=slice(1.76 + 10.33 + 0.5, 1.76 + 10.33 + 0.5 + 8))

plt.figure(figsize=(10,5))
plt.plot(profile.x - (1.76 + 10.33 + 0.5) + 0.5, profile.values)
plt.ylim(0.0, 0.15)
plt.xlim(0.0, 8.0)
plt.yticks([0, 0.05, 0.10, 0.15], [0, 0.05, 0.10, 0.15])
plt.grid(alpha=0.2)
plt.savefig("lastz_b_S1.png", dpi=250)
plt.close()

y_target = 9.2/2 + 1.45
iy = np.argmin(np.abs(data.y.values - y_target))

profile = z.isel(y=iy).sel(x=slice(1.76 + 10.33 + 0.5, 1.76 + 10.33 + 0.5 + 8))

plt.figure(figsize=(10,5))
plt.plot(profile.x - (1.76 + 10.33 + 0.5) + 0.5, profile.values)
plt.ylim(0.0, 0.15)
plt.xlim(0.0, 8.0)
plt.yticks([0, 0.05, 0.10, 0.15], [0, 0.05, 0.10, 0.15]) #
plt.grid(alpha=0.2)
plt.savefig("lastz_b_S2.png", dpi=250)
plt.close()

for i, idt in enumerate(time_array):

    G = data.G[i].sel(x=slice(10, 24), y=slice(9.2/2+0.5+1.3, 9.2/2-0.5-1.3))
    h = data.h[i].sel(x=slice(10, 24), y=slice(9.2/2+0.5+1.3, 9.2/2-0.5-1.3))
    z = data.z[i].sel(x=slice(10, 24), y=slice(9.2/2+0.5+1.3, 9.2/2-0.5-1.3))
    z_b = data.z_b[i].sel(x=slice(10, 24), y=slice(9.2/2+0.5+1.3, 9.2/2-0.5-1.3))

    eps = 1e-12
    u = data.hu[i].sel(x=slice(10, 24),
                    y=slice(9.2/2+0.5+1.3, 9.2/2-0.5-1.3)) / (h + eps)

    v = data.hv[i].sel(x=slice(10, 24),
                    y=slice(9.2/2+0.5+1.3, 9.2/2-0.5-1.3)) / (h + eps)

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
    im0.set_clim(2e-2, 0.6)

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

    plt.savefig(f"fields_{i:04d}.png", dpi=250)
    plt.close()


    """h_slice = data.h[i, N_half]
    zb_slice = data.z_b[i, N_half]
    z_slice = data.z[i, N_half]
    hu_slice = data.hu[i, N_half]

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
    fig.savefig(f"dambreak_{i}.png", dpi=200)
    plt.close()"""

