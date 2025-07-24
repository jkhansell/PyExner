
import xarray as xr
import numpy as np


def build_dambreak(Ll=0, Lr=10, hl=0.005, hr=0.001, x0=5, dh=0.01):
    x_range = [Ll, Lr]
    y_range = [0,10]

    x = np.arange(x_range[0], x_range[1]+dh, dh)
    y = np.arange(y_range[0], y_range[1]+dh, dh)

    X, Y = np.meshgrid(x, y, indexing="xy")

    mask = (X <= x0)

    h = np.where(mask, hl, hr)
    u = np.zeros_like(h)
    v = np.zeros_like(h)
    z = np.ones_like(h)
    n = np.zeros_like(h)

    ds = xr.Dataset(
        {
            "h": (["y", "x"], h),
            "hu": (["y", "x"], h*u),
            "hv": (["y", "x"], h*v),
            "z": (["y", "x"], z),
            "n": (["y", "x"], n),
        },
        coords={
            "x": x,
            "y": y,
        },
        attrs={
            "description": "Initial condition for SWE test case",
            "x_range": str(x_range),
            "y_range": str(y_range),
        }
    )

    ds.to_netcdf("input.nc", format="NETCDF3_64BIT")

if __name__ == "__main__":
    build_dambreak(Ll=0, Lr=30, hl=1, hr=0.2, x0=15)