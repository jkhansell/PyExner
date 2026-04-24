import argparse
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt

# This is an experimental case therefore its hardcoded
# Taken from https://dx.doi.org/10.2139/ssrn.5269734

def L_domain():
    Lx = 6
    Ly = 0.5
    dh = 0.005

    x_range = [0, Lx]
    y_range = [0, Ly]

    x = np.arange(x_range[0], x_range[1],  dh)
    y = np.arange(y_range[1], y_range[0], -dh)

    X, Y = np.meshgrid(x, y, indexing="xy")

    h_mask = (X >= 0) & (X <= 3) 

    h = np.where(h_mask, 0.25, 0.0)
    u = np.zeros_like(h)
    v = np.zeros_like(h)
    z_b = 0.1 * np.ones_like(h)
    z = np.ones_like(h)

    n_p = 0.0185
    n = n_p*np.ones_like(h)

    no_data_mask = (X >= 0) & (X < 4.0) & (Y > 0.25)

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
            "description": "L domain initial condition",
            "x_range": str(x_range),
            "y_range": str(y_range),
        },
    )

    ds.to_netcdf("L_domain.nc", format="NETCDF3_64BIT")


if __name__ == "__main__":
    L_domain()