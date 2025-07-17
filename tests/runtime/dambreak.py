import numpy as np
import matplotlib.pyplot as plt

from PyExner.runtime.driver import run_driver


def build_dambreak(Ll=0, Lr=10, hl=0.005, hr=0.001, x0=5, T=6, dh=0.01):
    x_range = [Ll, Lr]
    y_range = [0,.1]

    x = np.arange(x_range[0], x_range[1]+dh, dh)
    y = np.arange(y_range[0], y_range[1]+dh, dh)

    X, Y = np.meshgrid(x, y, indexing="xy")

    mask = (X <= x0)

    h = np.where(mask, hl, hr)
    u = np.zeros_like(h)
    v = np.zeros_like(h)
    z = 2*np.ones_like(h)
    n = np.zeros_like(h)

    x_range = [0, 10]

    y_range = [0,.1]

    inlet_polygon = [[x_range[0]-dh/2, y_range[0]-dh/2],
                     [x_range[0]+dh/2, y_range[0]-dh/2],
                     [x_range[0]+dh/2, y_range[1]+dh/2],
                     [x_range[0]-dh/2, y_range[1]+dh/2]]
    

    outlet_polygon = [[x_range[1]-dh/2, y_range[0]-dh/2],
                      [x_range[1]+dh/2, y_range[0]-dh/2],
                      [x_range[1]+dh/2, y_range[1]+dh/2],
                      [x_range[1]-dh/2, y_range[1]+dh/2]]
    
    params = {
        "endTime" : T,
        "cfl" : 0.5,
        "flux_scheme": "Roe", 
        "integrator": "Forward Euler", 
        "outFreq" : 1,
        "dh" : dh,
        "initial_conditions": {
            "h_init": h,
            "u_init": u,
            "v_init": v,
            "z_init": z,
            "roughness": n,
        },
        "boundaries": {
            "inlet": {
                "type": "Transmissive",
                "polygon": inlet_polygon,
                #         [h, q, z, qb] 
                "values": [0.0,1.0,0.0,0.0],
                "normal": [-1.0,0.0]
            },
            "outlet": {
                "type": "Transmissive",
                "polygon": outlet_polygon,
                #         [h, q, z, qb] 
                "values": [0.0, 0.0, 0.0, 0.0],
                "normal": [1.0,0.0]
            }
        },
        "X": X,
        "Y": Y
    }

    return params

if __name__ == "__main__":

    params = build_dambreak()
    last_state = run_driver(params)

    h = last_state.h

    plt.plot(h[h.shape[0]//2])
    plt.savefig("h.png")
    plt.close()


