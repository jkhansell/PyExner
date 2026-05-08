import argparse
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt

def erodible_channel():
    dh = 0.01 

    # ==========================================
    # 1. DEFINE COORDINATE BOUNDS
    # ==========================================
    x_left_tank      = -12.09
    x_mid_start      = -10.33
    x_contract_start = -0.5    
    x_contract_end   = 0.5     
    x_right_end      = 15.5    
    
    y_tank_max     = 4.6             
    y_tank_min     = -y_tank_max
    y_channel_max  = 1.8             
    y_channel_min  = -y_channel_max
    y_contract_max = 0.5             
    y_contract_min = -y_contract_max

    x = np.arange(x_left_tank + dh/2, x_right_end, dh)
    y = np.arange(y_tank_max - dh/2, y_tank_min, -dh)
    X, Y = np.meshgrid(x, y, indexing="xy")

    # ==========================================
    # 2. CREATE GEOMETRY MASKS
    # ==========================================
    mask_left_tank = (X >= x_left_tank) & (X <= x_mid_start) & (Y >= y_tank_min) & (Y <= y_tank_max)
    mask_mid_channel = (X > x_mid_start) & (X <= x_contract_start) & (Y >= y_channel_min) & (Y <= y_channel_max)
    mask_contraction = (X > x_contract_start) & (X <= x_contract_end) & (Y >= y_contract_min) & (Y <= y_contract_max)
    mask_right_channel = (X > x_contract_end) & (X <= x_right_end) & (Y >= y_channel_min) & (Y <= y_channel_max)

    valid_domain = mask_left_tank | mask_mid_channel | mask_contraction | mask_right_channel
    nanmask = ~valid_domain

    # ==========================================
    # 3. TOPOGRAPHY (z) - FIXED CONCRETE GEOMETRY
    # ==========================================
    # Explicit Datum Elevations
    z_edges = 0.155          # Absolute reference
    z_center = 0.0      # Trench depth
    z_tank = -0.10 # Left tank is 0.10m deeper than channel
    target_sed_elevation = 0.085 
    
    # Initialize everything to the edge elevation
    z = np.full_like(X, z_edges)

    # A. Excavate the flat channel bottoms
    mask_main_flow = mask_mid_channel | mask_contraction | mask_right_channel
    z = np.where(mask_main_flow, z_center, z)

    # B. Excavate the Tank
    z = np.where(mask_left_tank, z_tank, z)

    # C. Apply the Trapezoidal Banks (Ramping from z_center up to z_edges)
    bank_width = 0.34
    flat_y_max = 1.46  # 1.8 - 0.34
    y_abs = np.abs(Y)
    
    bank_slope = np.where(
        y_abs > flat_y_max, 
        z_center + (z_edges - z_center) * ((y_abs - flat_y_max) / bank_width), 
        z
    )
    
    mask_trapezoidal = mask_mid_channel | mask_right_channel
    z = np.where(mask_trapezoidal & (y_abs > flat_y_max), bank_slope, z)

    # D. The Solid Concrete Sill
    sill_thickness = 0.05 
    x_sed_end = 9.00
    mask_sill = (X > x_sed_end) & (X <= x_sed_end + sill_thickness) & mask_right_channel
    
    # Sill rises to perfectly match the target sediment surface
    z = np.where(mask_sill, np.maximum(z, target_sed_elevation), z)

    # ==========================================
    # 4. ERODIBLE SEDIMENT BED (z_b)
    # ==========================================
    z_b = np.zeros_like(X)
    
    x_sed_start = -1.00  
    ramp_up_len = 0.20   

    mask_sediment = (X >= x_sed_start) & (X <= x_sed_end) & valid_domain
    
    # Create the target surface elevation array
    target_z = np.copy(z) # Default is bare concrete
    
    # Target ramp up
    mask_ramp_up = mask_sediment & (X < x_sed_start + ramp_up_len)
    progress = (X - x_sed_start) / ramp_up_len
    
    target_z = np.where(
        mask_ramp_up,
        z + (target_sed_elevation - z) * progress,
        target_z
    )
    
    # Target plateau
    mask_plateau = mask_sediment & (X >= x_sed_start + ramp_up_len)
    target_z[mask_plateau] = target_sed_elevation

    # Actual sediment thickness is the Target Surface minus the Concrete Surface
    z_b = np.where(mask_sediment, np.maximum(0.0, target_z - z), 0.0)

    # ==========================================
    # 5. WATER DEPTH & HYDRODYNAMICS
    # ==========================================
    WSE_absolute = 0.47 
    
    h = WSE_absolute - z - z_b
    h = np.where((X <= 0.0) & (h > 0), h, 0.0) 
    
    u = np.zeros_like(h)
    v = np.zeros_like(h)
    n = np.full_like(h, 0.01)

    # Apply masks
    h[nanmask] = np.nan
    u[nanmask] = np.nan
    v[nanmask] = np.nan
    z[nanmask] = np.nan
    z_b[nanmask] = np.nan
    n[nanmask] = np.nan

    # ==========================================
    # 6. EXPORT
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
            "description": "Absolute Datum, sharp corners restored",
            "x_range": f"[{x_left_tank}, {x_right_end}]",
            "y_range": f"[{y_tank_min}, {y_tank_max}]",
        },
    )

    ds.to_netcdf("erodible_channel.nc", format="NETCDF3_64BIT")
    print("Exported erodible_channel.nc successfully!")

    data_ = xr.open_dataset("erodible_channel.nc")

    # Plot to verify
    for var_name, data, cmap in [("h", data_.h, "Blues"), ("z_b", data_.z_b, "YlOrBr"), ("z", data_.z, "gray"), ("n", data_.n, "gray")]:
        plt.figure(figsize=(12, 4))
        plt.imshow(data, cmap=cmap, extent=[x.min(), x.max(), y.min(), y.max()], origin='upper')
        plt.colorbar(label=var_name)
        plt.axhline(0, color='r', linestyle='--', alpha=0.3) # Y=0 Centerline
        plt.axvline(0, color='r', linestyle='--', alpha=0.3) # X=0 Junction
        plt.title(f"{var_name} Profile")
        plt.xlabel("x (m)")
        plt.ylabel("y (m)")
        plt.savefig(f"{var_name}.png", bbox_inches='tight', dpi=250)
        plt.close()

if __name__ == "__main__":
    erodible_channel()