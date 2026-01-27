import argparse
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt

# This is an experimental case therefore its hardcoded
# Taken from https://dx.doi.org/10.2139/ssrn.5269734

def erodible_channel():
    Lx = 27.59
    Ly = 9.2
    dh = 0.005

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
    maskzb2 = (X >= x_1 + x_2 - 1.5) & (X <= x_1 + x_2 - 1.5 + 1.5/5)
    # ramp up from 0 o 0.085 on maskzb2


    z = np.where(mask5, 0.0, 0.1)

    # ramp interval
    x0 = x_1 + x_2 - 1.5
    x1 = x_1 + x_2 - 1.5 + 1.5/5   # ramp length

    # normalized coordinate in [0,1]
    s = (X - x0) / (x1 - x0)
    s = np.clip(s, 0.0, 1.0)

    z_b = np.where(maskzb1, 0.085, 0.0)
    z_b = np.where(maskzb2, 0.085 * s, z_b)

    h = np.where(maskh & ~nanmask, 0.57 - (z+z_b), 0.0)


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
            "description": "L domain initial condition",
            "x_range": str(x_range),
            "y_range": str(y_range),
        },
    )

    ds.to_netcdf("erodible_channel.nc", format="NETCDF3_64BIT")

    plt.imshow(h, cmap="jet")
    plt.colorbar()
    plt.savefig("h.png")
    plt.close()
    
    plt.imshow(z_b, cmap="jet")
    plt.colorbar()
    plt.savefig("z_b.png")
    plt.close()

    plt.imshow(z, cmap="jet")
    plt.colorbar()
    plt.savefig("z.png")
    plt.close()


if __name__ == "__main__":
    erodible_channel()