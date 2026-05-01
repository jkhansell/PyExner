import numpy as np
import xarray as xr
import os
import yaml
import subprocess


def build_analytical_case(dh=0.1, output_name="analytical_case.nc"):

    # ideal case parameters https://doi.org/10.1016/j.advwatres.2021.103931
    A = 0.005           
    alpha = 0.005
    beta = 0.005
    gamma = 1
    q0 = 1 

    x_range = [0, 7]
    y_range = [0, 1]

    x = np.arange(x_range[0], x_range[1] + dh, dh)
    y = np.arange(y_range[1], y_range[0] - dh, -dh)

    X, Y = np.meshgrid(x, y, indexing="xy")

    u_func = lambda x: ((alpha*x + beta)/A)**(1/3)

    q = q0*np.ones_like(X)
    u = u_func(X)
    v = np.zeros_like(u)
    h = q/u
    g = 9.81
    z_b = -(u**3 + 2*g*q)/(2*g*u) + gamma
    n = np.zeros_like(h)
    z = np.zeros_like(h)

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

    ds.to_netcdf(output_name, format="NETCDF3_64BIT")

if __name__ == "__main__":

    build_analytical_case(dh=0.05, output_name='analytical_case.nc')

    # # dh values
    # dh_values = [0.1, 0.05, 0.025, 0.0125, 0.00625]

    # run_script = os.path.abspath("run.py")

    # for i, dh in enumerate(dh_values):
    #     dh_str = str(dh)

    #     folder = f"dh_{dh_str}"
    #     os.makedirs(folder, exist_ok=True)

    #     # names
    #     nc_name = f"analytical_case_{dh_str}.nc"
    #     yaml_name = f"input_{dh_str}.yaml"
    #     out_name = f"analytical_case_{dh_str}_out.nc"

    #     nc_path = os.path.join(folder, nc_name)
    #     yaml_path = os.path.join(folder, yaml_name)

    #     build_analytical_case(dh=dh, output_name=nc_path)  # tu función actual

    #     # load template
    #     with open("input.yaml", "r") as f:
    #         config = yaml.safe_load(f)

    #     # change values
    #     config["input_file"] = nc_name
    #     config["output_file"] = out_name

    #     # save yaml
    #     with open(yaml_path, "w") as f:
    #         yaml.dump(config, f)

    #     # run simulation
    #     subprocess.run(
    #         ["python", run_script, yaml_name],
    #         cwd=folder,
    #         check=True
    #     )

