import argparse
import xarray as xr
import numpy as np


def supercritical_dambreak_exner_2D(L, dh):
    x_range = [-L, L]
    y_range = [-L / 5, L / 5]

    x = np.arange(x_range[0], x_range[1] + dh, dh)
    y = np.arange(y_range[0], y_range[1] + dh, dh)
    X, Y = np.meshgrid(x, y, indexing="xy")

    mask = (X <= 5.0) & (X >= -5.0)

    h = np.where(mask, 50.0, 0.2)
    u = np.zeros_like(h)
    v = np.zeros_like(h)

    z = 10.0 * np.ones_like(h)

    n = np.zeros_like(h)
    G = (0.01 / (1.0 - 0.4)) * np.ones_like(h)

    ds = xr.Dataset(
        {
            "h": (["y", "x"], h),
            "hu": (["y", "x"], h * u),
            "hv": (["y", "x"], h * v),
            "z": (["y", "x"], z),
            "G": (["y", "x"], G),
            "n": (["y", "x"], n),
        },
        coords={"x": x, "y": y},
        attrs={
            "description": "Supercritical 2D dambreak Exner initial condition",
            "x_range": str(x_range),
            "y_range": str(y_range),
        },
    )

    ds.to_netcdf("supercritical_dambreak_exner.nc", format="NETCDF3_64BIT")
    print(ds)


def subcritical_dambreak_exner_2D(L, dh):
    x_range = [-L, L]
    y_range = [-L / 5, L / 5]

    x = np.arange(x_range[0], x_range[1] + dh, dh)
    y = np.arange(y_range[0], y_range[1] + dh, dh)
    X, Y = np.meshgrid(x, y, indexing="xy")

    mask = (X >= -0.5) & (X <= 0.5)

    h = np.where(mask, 1.0, 0.2)
    u = np.zeros_like(h)
    v = np.zeros_like(h)

    z = 1.0 * np.ones_like(h)

    n = np.zeros_like(h)
    G = (0.01 / (1.0 - 0.4)) * np.ones_like(h)

    ds = xr.Dataset(
        {
            "h": (["y", "x"], h),
            "hu": (["y", "x"], h * u),
            "hv": (["y", "x"], h * v),
            "z": (["y", "x"], z),
            "G": (["y", "x"], G),
            "n": (["y", "x"], n),
        },
        coords={"x": x, "y": y},
        attrs={
            "description": "Subcritical 2D dambreak Exner initial condition",
            "x_range": str(x_range),
            "y_range": str(y_range),
        },
    )

    ds.to_netcdf("subcritical_dambreak_exner.nc", format="NETCDF3_64BIT")
    print(ds)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate Exner dambreak initial conditions"
    )
    parser.add_argument("--L", type=float, required=True, help="Half-domain length")
    parser.add_argument("--dh", type=float, required=True, help="Grid spacing")
    parser.add_argument(
        "--case",
        choices=["supercritical", "subcritical"],
        required=True,
        help="Dambreak regime",
    )

    args = parser.parse_args()

    if args.case == "supercritical":
        supercritical_dambreak_exner_2D(args.L, args.dh)
    elif args.case == "subcritical":
        subcritical_dambreak_exner_2D(args.L, args.dh)