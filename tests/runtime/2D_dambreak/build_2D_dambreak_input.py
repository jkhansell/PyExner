import sys
import xarray as xr
import numpy as np

def symmetrical_dambreak_exner_2D(Lx, Ly, dh):
    x_range = [-Lx, Lx]
    y_range = [-Ly, Ly]

    x = np.arange(x_range[0], x_range[1]+dh, dh)
    y = np.arange(y_range[0], y_range[1]+dh, dh)
    X, Y = np.meshgrid(x, y, indexing="xy")

    R = min(Lx, Ly) / 4
    mask = X**2 + Y**2 <= R**2

    h = np.where(mask, 1.0, 0.2)
    n = np.zeros_like(h)
    u = np.zeros_like(h)
    v = np.zeros_like(h)
    z = np.ones_like(h)
    z_b = np.ones_like(h)
    
    ds = xr.Dataset(
        {
            "h": (["y", "x"], h.astype(np.float32)),
            "hu": (["y", "x"], (h*u).astype(np.float32)),
            "hv": (["y", "x"], (h*v).astype(np.float32)),
            "z_b": (["y", "x"], z_b.astype(np.float32)),
            "z": (["y", "x"], z.astype(np.float32)),
            "n": (["y", "x"], n.astype(np.float32)),
        },
        coords={
            "x": x.astype(np.float32),
            "y": y.astype(np.float32),
        },
        attrs={
            "description": "2D circular dambreak Initial condition for SWE test case",
            "x_range": str(x_range),
            "y_range": str(y_range),
        }
    )

    ds.to_netcdf("input.nc", format="NETCDF3_64BIT")


if __name__ == "__main__":
    if len(sys.argv) == 3:
        Lx = float(sys.argv[1])
        Ly = Lx
        dh = float(sys.argv[2])
    elif len(sys.argv) == 4:
        Lx = float(sys.argv[1])
        Ly = float(sys.argv[2])
        dh = float(sys.argv[3])
    else:
        print("Usage: python build_2D_dambreak_input.py Lx [Ly] dh")
        sys.exit(1)

    symmetrical_dambreak_exner_2D(Lx, Ly, dh)