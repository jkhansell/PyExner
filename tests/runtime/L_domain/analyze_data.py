import xarray as xr
import numpy as np
import matplotlib.pyplot as plt

data = xr.open_dataset("L_domain_out.nc")
print(data)

for i in range(len(data.h)):
    plt.figure(figsize=(12,3))
    img = plt.imshow(data.h[i], cmap="jet")
    cbar = plt.colorbar(img, orientation='horizontal')
    plt.axis("off")
    plt.savefig(f"h_{i}.png", dpi=200)
    plt.close()

    plt.figure(figsize=(12,3))
    img = plt.imshow(data.hu[i], cmap="jet")
    cbar = plt.colorbar(img, orientation='horizontal')
    plt.axis("off")
    plt.savefig(f"hu_{i}.png", dpi=200)
    plt.close()

    plt.figure(figsize=(12,3))
    img = plt.imshow(data.hv[i], cmap="jet")
    cbar = plt.colorbar(img, orientation='horizontal')
    plt.axis("off")
    plt.savefig(f"hv_{i}.png", dpi=200)
    plt.close()

    plt.figure(figsize=(12,3))
    img = plt.imshow(data.z[i], cmap="jet")
    cbar = plt.colorbar(img, orientation='horizontal')
    plt.axis("off")
    plt.savefig(f"z_{i}.png", dpi=200)
    plt.close()

    plt.figure(figsize=(12,3))
    img = plt.imshow(data.G[i], cmap="jet")
    cbar = plt.colorbar(img, orientation='horizontal')
    plt.savefig(f"G_{i}.png", dpi=200)
    plt.close()