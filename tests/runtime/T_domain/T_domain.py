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

def T_domain(h_value,z_value, output_name):
    Lx = 27.59
    Ly = 9.2
    dh = 0.05

    x_range = [0, Lx]
    y_range = [0, Ly]

    x = np.arange(x_range[0], x_range[1],  dh)
    y = np.arange(y_range[1], y_range[0], -dh)

    X, Y = np.meshgrid(x, y, indexing="xy")

    y_1 = 1.3
    y_2 = 1.0 
    y_tot = 9.2
    
    # Y point calculation for masks

    # First bottleneck
    y1_bottle1 = y_tot/2 - y_2/2 - y_1
    y2_bottle1 = y1_bottle1 + 2*y_1 + y_2

    # Second bottleneck coordinates
    y1_bottle2 = y1_bottle1
    y2_bottle2 = y1_bottle2 + y_1

    # third bottleneck coordinates
    y1_bottle3 = y2_bottle2 + y_2
    y2_bottle3 = y1_bottle3 + y_1

    x_1 = 1.76 
    x_2 = 10.33
    x_3 = 15.5 
    x_4 = 1.0

    # X point calculation for masks

    x_tot = x_1 + x_2 + x_3
    
    x1_bottle1 = x_1
    x2_bottle1 = x_1 + x_2 - x_4*0.5

    x1_bottle2 = x2_bottle1
    x2_bottle2 = x1_bottle2 + x_4

    mask1 = (X >= x_1)
    mask2 = (Y <= y1_bottle1) | (Y >= y2_bottle1)

    mask3 = (X >= x1_bottle2) & (X <= x2_bottle2) & (Y >= y1_bottle1) & (Y <= y2_bottle2)
    mask4 = (X >= x1_bottle2) & (X <= x2_bottle2) & (Y >= y1_bottle3) & (Y <= y2_bottle3)

    nanmask = mask1 & mask2 | mask3 | mask4

    mask6 = (X <= x_1 + x_2 - x_4/2)
    maskh = (X <= x_1+x_2) 
    mask5 = (X <= 1.76)
    maskzb1 = (X >= x_1 + x_2 - 1.5) & (X <= x_1 + x_2 + 9.0)
    maskzb2 = (X >= x_1 + x_2 - 1.5) & (X <= x_1 + x_2 - 1.5 + 1.5/4)
    # ramp up from 0 o 0.085 on maskzb2

    z = np.where(mask5, 0.0, 0.1)

    # ramp interval
    x0 = x_1 + x_2 - 1.5
    x1 = x_1 + x_2 - 1.5 + 1.5/4   # ramp length

    # normalized coordinate in [0,1]
    s = (X - x0) / (x1 - x0)
    s = np.clip(s, 0.0, 1.0)

    # right ramp interval
    xR1 = x_1 + x_2 + 9.0        # end of structure
    xR0 = xR1 - 1.5/6            # start of ramp down
    maskzb3 = (X >= xR0) & (X <= xR1)
    # normalized coordinate for ramp down: 1 → 0
    sr = (xR1 - X) / (xR1 - xR0)
    sr = np.clip(sr, 0.0, 1.0)

    z_b = np.where(maskzb1, z_value, 0.0)        # plateau
    z_b = np.where(maskzb2, z_value * s, z_b)    # ramp up
    z_b = np.where(maskzb3, z_value * sr, z_b)   # ramp down

    h = np.where(maskh & ~nanmask, h_value - (z+z_b), 0.0)

    u = np.zeros_like(h)
    v = np.zeros_like(h)

    n_p = 0.0165
    n = n_p*np.ones_like(h)

    h[nanmask] = np.nan
    u[nanmask] = np.nan
    v[nanmask] = np.nan
    z_b[nanmask] = np.nan
    z[nanmask] = np.nan
    n[nanmask] = np.nan

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
            "description": "T domain initial condition",
            "x_range": str(x_range),
            "y_range": str(y_range),
        },
    )

    ds.to_netcdf(output_name, format="NETCDF3_64BIT")

if __name__ == "__main__":
    # exact params (individual run)
    z_value = 0.085
    h_value = 0.57
    d_value = 1.61e-3
    p_value = 0.42
    output_name = "T_domain.nc"
    T_domain(h_value,z_value, output_name)

      # +-
    pm_z = 0.005
    pm_h = 0.005
    pm_d = 0.00003
    pm_p = 0.01
    
    # uncertainty analysis (Monte carlo)
    num_runs = 10
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
        nc_path = os.path.join(folder, f"T_domain_{i+1}.nc")
        yaml_name = f"input_{i+1}.yaml"
        T_domain(h_val,z_val, nc_path)

        # load base yaml (template)
        with open("input.yaml", "r") as f:
            config = yaml.safe_load(f)

        # update run-specific parameters
        config["input_file"] = f"T_domain_{i+1}.nc"
        config["output_file"] = f"T_domain_out_{i+1}.nc"
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