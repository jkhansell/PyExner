import argparse
import xarray as xr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import yaml
import subprocess

# This is an experimental case therefore its hardcoded
# Taken from https://dx.doi.org/10.2139/ssrn.5269734

def L_domain(h_value, z_value, output_name):
    Lx = 6
    Ly = 0.5
    dh = 0.005

    x_range = [0, Lx]
    y_range = [0, Ly]

    x = np.arange(x_range[0], x_range[1],  dh)
    y = np.arange(y_range[1], y_range[0], -dh)

    X, Y = np.meshgrid(x, y, indexing="xy")

    h_mask = (X >= 0) & (X <= 3) 

    h = np.where(h_mask, h_value, 0.0)
    u = np.zeros_like(h)
    v = np.zeros_like(h)
    z_b = z_value * np.ones_like(h)
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

    ds.to_netcdf(output_name, format="NETCDF3_64BIT")


if __name__ == "__main__":
    
    # exact params (individual run)
    z_value = 0.1
    h_value = 0.25
    d_value = 1.72e-3
    p_value = 0.39
    output_name = "L_domain.nc"
    L_domain(h_value,z_value, output_name)

      # +-
    pm_z = 0.005
    pm_h = 0.005
    pm_d = 0.00003
    pm_p = 0.01
    
    # uncertainty analysis (Monte carlo)
    num_runs = 15
    run_script = os.path.abspath("run.py")

    records = []

    for i in range(num_runs):
        
        # define run-specific parameters distributions
        z_val = round(np.random.uniform(z_value-pm_z, z_value+pm_z), 3)
        h_val = round(np.random.uniform(h_value-pm_h, h_value+pm_h), 3)
        diameter = round(np.random.uniform(d_value - pm_d, d_value + pm_d), 5)
        porosity = round(np.random.uniform(p_value-pm_p, p_value + pm_p), 2)
       
        # create a folder for each run
        folder = f"run_{i+1}"
        os.makedirs(folder, exist_ok=True)

        # save run-specific parameters
        param_path = os.path.join(folder, f"params_{i+1}.txt")

        params = {
            "run": i+1,
            "h_val": h_val,
            "z_val": z_val,
            "diameter": diameter,
            "porosity": porosity
        }

        records.append(params)

        with open(param_path, "w") as f:
            for key, val in params.items():
                f.write(f"{key}: {val}\n")

        # generate initial condition file (.nc)
        nc_path = os.path.join(folder, f"L_domain_{i+1}.nc")
        yaml_name = f"input_{i+1}.yaml"
        L_domain(h_val,z_val, nc_path)

        # load base yaml (template)
        with open("input.yaml", "r") as f:
            config = yaml.safe_load(f)

        # update run-specific parameters
        config["input_file"] = f"L_domain_{i+1}.nc"
        config["output_file"] = f"L_domain_out_{i+1}.nc"
        config["erosion"]["bulk_porosity"] = float(porosity)
        config["erosion"]["sediments"]["sed1"]["diameter"] = float(diameter)

        # save yaml for this run
        yaml_path = os.path.join(folder, f"input_{i+1}.yaml")

        with open(yaml_path, "w") as f:
            yaml.dump(config, f)
        
        # run simulation in run_i folder
        subprocess.run(
            ["python", run_script, yaml_name],
            cwd=folder,
            check=True
        )

    pd.DataFrame(records).to_csv("params_summary.csv", index=False)

        
