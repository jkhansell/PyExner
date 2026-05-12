import xarray as xr
import numpy as np
import matplotlib.pyplot as plt

data = xr.open_dataset("./strong_scaling/gpus_2/output.nc")
print(data)

for i in range(len(data.z_b)):
    plt.imshow(data.z_b[i])
    plt.savefig(f"h_{i}.png")
    plt.close()