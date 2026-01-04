import xarray as xr
import numpy as np
import matplotlib.pyplot as plt

data = xr.open_dataset("output.nc")
print(data)


for i in range(len(data.h)):
    plt.imshow(data.h[i])
    plt.savefig(f"h_{i}.png")
    plt.close()