import sys
import xarray as xr
import numpy as np

def symmetrical_dambreak_exner_2D(L, dh, G=0.001):
    x_range = [-L, L]
    y_range = [-L, L]

    x = np.arange(x_range[0], x_range[1]+dh, dh)
    y = np.arange(y_range[0], y_range[1]+dh, dh)
    X, Y = np.meshgrid(x, y, indexing="xy")

    mask = X**2 + Y**2 <= (L/4)**2

    h = np.where(mask, 1.0, 0.2)
    n = np.zeros_like(h)
    u = np.zeros_like(h)
    v = np.zeros_like(h)
    z = np.ones_like(h)

    xi = 0.4
    A_g = (1/(1-xi))*G*np.ones_like(h)

    ds = xr.Dataset(
        {
            "h": (["y", "x"], h.astype(np.float32)),
            "hu": (["y", "x"], (h*u).astype(np.float32)),
            "hv": (["y", "x"], (h*v).astype(np.float32)),
            "z": (["y", "x"], z.astype(np.float32)),
            "n": (["y", "x"], n.astype(np.float32)),
            "G": (["y", "x"], A_g.astype(np.float32)) 
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
    
    L = float(sys.argv[1])
    dh = float(sys.argv[2])
    G = float(sys.argv[3])

    symmetrical_dambreak_exner_2D(L, dh, G)