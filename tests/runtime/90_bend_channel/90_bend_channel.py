import argparse
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt

# This is an experimental case therefore its hardcoded
# Taken from https://dx.doi.org/10.2139/ssrn.5269734

def bend_channel():
    Lx = 6.725
    Ly = 3.820
    dh = 0.005

    x_range = [0, Lx]
    y_range = [0, Ly]

    x = np.arange(x_range[0], x_range[1],  dh)
    y = np.arange(y_range[1], y_range[0], -dh)

    X, Y = np.meshgrid(x, y, indexing="xy")

    # X dimensions
    x_0 = 0
    x_1 = 2.39
    x_2 = 3.84
    x_3 = 0.495

    # Y dimensions
    y_r = 2.44 # reservoir
    y_1 = 0.445
    y_2 = 0.495
    y_3 = 2.88

    # X coordinates for masks
    x_1m = x_1
    x_2m = x_1m + x_2
    x_3m = x_2m + x_3

    # Y coordinates for masks
    y_1m = y_1
    y_2m = y_1m + y_2
    y_3m = y_2m + y_3

    # Masks 
    mask1 = (X >= x_0) & (X <= x_1m) & (Y >= y_r)
    mask2 = (X >= x_1m) & (X <= x_2m) & (Y >= y_2m)
    mask3 = (X >= x_1m) & (X <= x_3m) & (Y <= y_1m)

    no_data_mask = mask1 | mask2 | mask3 

    # Where water exists
    h_mask = (X <= x_1m)

    # Where erodible surface exists
    zb_mask = (X >= x_1m) & (X <= x_3m) & (Y >= y_1m) & (Y <= y_3m)

    h = np.where(h_mask, 0.59, 0.0)
    u = np.zeros_like(h)
    v = np.zeros_like(h)
    z_b = np.full_like(h, np.nan)
    z_b[zb_mask] = 0.075
    z = np.where(h_mask, 0.0, 0.33)

    n_p = 0.0165 
    n = n_p*np.ones_like(h)

    h[no_data_mask] = np.nan
    u[no_data_mask] = np.nan
    v[no_data_mask] = np.nan
    z_b[no_data_mask] = np.nan
    z[no_data_mask] = np.nan
    n[no_data_mask] = np.nan

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
            "description": "90 bend channel domain initial condition",
            "x_range": str(x_range),
            "y_range": str(y_range),
        },
    )

    ds.to_netcdf("90_bend_channel_domain.nc", format="NETCDF3_64BIT")

    plt.imshow(h, cmap="jet")
    plt.colorbar()
    plt.savefig("h.png")
    plt.close()
    
    plt.imshow(z_b, cmap="jet")
    plt.colorbar()
    plt.savefig("z_b.png")
    plt.close()

    plt.imshow(z, cmap="jet")
    plt.colorbar()
    plt.savefig("z.png")
    plt.close()


if __name__ == "__main__":
    bend_channel()


    