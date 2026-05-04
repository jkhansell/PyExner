import argparse
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt

def erodible_channel():
    dh = 0.01

    # ==========================================
    # 1. DEFINE COORDINATE BOUNDS
    # Origin (0,0) is at the centerline, exactly where the 10.33m and 15.5m sections meet.
    # The 1m contraction is centered over this origin.
    # ==========================================
    
    # X boundaries
    x_left_tank      = -12.09  # -(10.33 + 1.76)
    x_mid_start      = -10.33  # Start of main channel
    x_contract_start = -0.5    # Contraction overlaps 0.5m upstream
    x_contract_end   = 0.5     # Contraction overlaps 0.5m downstream
    x_right_end      = 15.5    # End of the right channel
    
    # Y boundaries
    y_tank_max     = 4.6             # 9.2 / 2
    y_tank_min     = -y_tank_max
    y_channel_max  = 1.8             # 1.3 + 0.5
    y_channel_min  = -y_channel_max
    y_contract_max = 0.5             # 1.0 / 2
    y_contract_min = -y_contract_max

    # Create coordinate arrays. Adding/subtracting dh/2 ensures clean boundary capture.
    x = np.arange(x_left_tank, x_right_end + dh/2, dh)
    y = np.arange(y_tank_max, y_tank_min - dh/2, -dh)

    X, Y = np.meshgrid(x, y, indexing="xy")

    # ==========================================
    # 2. CREATE GEOMETRY MASKS
    # ==========================================
    mask_left_tank = (X >= x_left_tank) & (X <= x_mid_start) & (Y >= y_tank_min) & (Y <= y_tank_max)
    
    # Upstream channel (stops at the contraction)
    mask_mid_channel = (X > x_mid_start) & (X <= x_contract_start) & (Y >= y_channel_min) & (Y <= y_channel_max)
    
    # The 1m Contraction (centered at x=0)
    mask_contraction = (X > x_contract_start) & (X <= x_contract_end) & (Y >= y_contract_min) & (Y <= y_contract_max)
    
    # Downstream channel (starts after the contraction)
    mask_right_channel = (X > x_contract_end) & (X <= x_right_end) & (Y >= y_channel_min) & (Y <= y_channel_max)

    valid_domain = mask_left_tank | mask_mid_channel | mask_contraction | mask_right_channel
    nanmask = ~valid_domain

    # ==========================================
    # 3. TOPOGRAPHY & SEDIMENT BED
    # ==========================================
    # z: Fixed Concrete Bed. Main channel is z=0. Left tank has a 0.1m step down.
    z = np.zeros_like(X)
    z[mask_left_tank] = -0.1

    # z_b: Erodible Sediment Bed.
    # Diagram shows sediment starting 1.5m BEFORE the junction (x = -1.5)
    # and ending 9.0m AFTER the junction (x = 9.0).
    z_b = np.zeros_like(X)
    
    x_sed_start = -1.5
    x_sed_end = 9.0
    ramp_up_len = 1.5 / 4.0
    ramp_down_len = 1.5 / 6.0
    max_sed_height = 0.085

    # Base sediment mask
    mask_sediment = (X >= x_sed_start) & (X <= x_sed_end) & valid_domain
    
    # Apply ramp up
    mask_ramp_up = mask_sediment & (X < x_sed_start + ramp_up_len)
    z_b[mask_ramp_up] = max_sed_height * (X[mask_ramp_up] - x_sed_start) / ramp_up_len
    
    # Apply plateau
    mask_plateau = mask_sediment & (X >= x_sed_start + ramp_up_len) & (X <= x_sed_end - ramp_down_len)
    z_b[mask_plateau] = max_sed_height
    
    # Apply ramp down
    mask_ramp_down = mask_sediment & (X > x_sed_end - ramp_down_len)
    z_b[mask_ramp_down] = max_sed_height * (x_sed_end - X[mask_ramp_down]) / ramp_down_len

    # ==========================================
    # 4. WATER DEPTH & HYDRODYNAMICS
    # ==========================================
    # Flat Water Surface Elevation (WSE) at 0.57m relative to main channel bed
    WSE = 0.57
    h = WSE - z - z_b
    h = np.where(X <= 0.0, h, 0.0)
    
    # Initialize velocities and roughness
    u = np.zeros_like(h)
    v = np.zeros_like(h)
    n = np.full_like(h, 0.0165)

    # Apply nanmask to cut out the walls
    h[nanmask] = np.nan
    u[nanmask] = np.nan
    v[nanmask] = np.nan
    z[nanmask] = np.nan
    z_b[nanmask] = np.nan
    n[nanmask] = np.nan

    # ==========================================
    # 5. EXPORT
    # ==========================================
    ds = xr.Dataset(
        {
            "h": (["y", "x"], h),
            "hu": (["y", "x"], h * u),
            "hv": (["y", "x"], h * v),
            "z_b": (["y", "x"], z_b),
            "z": (["y", "x"], z),
            "n": (["y", "x"], n),
        },
        coords={"x": x, "y": y},
        attrs={
            "description": "Domain initialized with true origin at junction",
            "x_range": f"[{x_left_tank}, {x_right_end}]",
            "y_range": f"[{y_tank_min}, {y_tank_max}]",
        },
    )

    ds.to_netcdf("erodible_channel.nc", format="NETCDF3_64BIT")

    # Plot to verify
    for var_name, data, cmap in [("h", h, "Blues"), ("z_b", z_b, "YlOrBr"), ("z", z, "gray")]:
        plt.figure(figsize=(12, 4))
        plt.imshow(data, cmap=cmap, extent=[x.min(), x.max(), y.min(), y.max()], origin='upper')
        plt.colorbar(label=var_name)
        plt.axhline(0, color='r', linestyle='--', alpha=0.3) # Y=0 Centerline
        plt.axvline(0, color='r', linestyle='--', alpha=0.3) # X=0 Junction
        plt.title(f"{var_name} Profile")
        plt.xlabel("x (m)")
        plt.ylabel("y (m)")
        plt.savefig(f"{var_name}.png", bbox_inches='tight')
        plt.close()

if __name__ == "__main__":
    erodible_channel()