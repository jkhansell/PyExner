import numpy as np
import xarray as xr

def analytical_solution_2D(Ll=0, Lr=7, dh=0.1, t=0, q=3, A_g=0.005, alpha=0.005, beta=0.005, gamma=1, p=3/2, g=9.81):

    x = np.arange(Ll, Lr + dh, dh)
    y = np.arange(0, 1 + dh, dh)

    X, Y = np.meshgrid(x, y, indexing="xy")

    # velocidad cuadrada
    ue2 = ((alpha * X + beta) / A_g)**(1/p)

    # velocidad
    u = np.sqrt(ue2)

    # altura
    h = q / u

    # z_b0
    zb0 = -(u**3 + 2*q*g) / (2*u*g) + gamma

    # evolución en el tiempo
    zb = -alpha * t + zb0

    return h, u, zb, zb0, x, y


def build_analytical_case(Ll=0, Lr=7, dh=0.1):
    x_range = [Ll, Lr]
    y_range = [0, 1]

    x = np.arange(x_range[0], x_range[1] + dh, dh)
    y = np.arange(y_range[0], y_range[1] + dh, dh)

    X, Y = np.meshgrid(x, y, indexing="xy")

    # Condiciones iniciales
    h0, u0, _ , zb0, _ , _ = analytical_solution_2D(t=0, Ll=Ll, Lr=Lr, dh=dh)

    h = h0
    u = u0
    v = np.zeros_like(u)
    z_b = zb0
    z = np.ones_like(h)
    n = np.zeros_like(h)

    ds = xr.Dataset(
        {
            "h": (["y", "x"], h),
            "hu": (["y", "x"], (h * u)),
            "hv": (["y", "x"], (h * v)),
            "z_b": (["y", "x"], z_b),
            "z": (["y", "x"], z),
            "n": (["y", "x"], n),
        },
        coords={
            "x": x.astype(np.float32),
            "y": y.astype(np.float32),
        },
        attrs={
            "description": "Initial condition for SWE test case",
            "x_range": str(x_range),
            "y_range": str(y_range),
        }
    )

    ds.to_netcdf("analytical_case.nc", format="NETCDF3_64BIT")

if __name__ == "__main__":
    build_analytical_case(Ll=0, Lr=7, dh=0.1)