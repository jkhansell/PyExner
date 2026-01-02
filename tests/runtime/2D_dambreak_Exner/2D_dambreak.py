import numpy as np
import matplotlib.pyplot as plt
import functools
import jax
import time 
import xarray as xr

from PyExner import run_driver
from PyExner.utils.constants import g

def build_dambreak(L, dh=0.01, T=6):
    x_range = [-L, L]
    y_range = [-L, L]

    polygon1 = [[x_range[0]-dh/2, y_range[0]-dh/2],
                [x_range[0]+dh/2, y_range[0]-dh/2],
                [x_range[0]+dh/2, y_range[1]+dh/2],
                [x_range[0]-dh/2, y_range[1]+dh/2]]

    polygon2 = [[x_range[1]-dh/2, y_range[0]-dh/2],
                [x_range[1]+dh/2, y_range[0]-dh/2],
                [x_range[1]+dh/2, y_range[1]+dh/2],
                [x_range[1]-dh/2, y_range[1]+dh/2]]

    polygon3 = [[x_range[0]-dh/2, y_range[0]-dh/2],
                [x_range[1]+dh/2, y_range[0]-dh/2],
                [x_range[1]+dh/2, y_range[0]+dh/2],
                [x_range[0]-dh/2, y_range[0]+dh/2]]

    polygon4 = [[x_range[0]-dh/2, y_range[1]-dh/2],
                [x_range[1]+dh/2, y_range[1]-dh/2],
                [x_range[1]+dh/2, y_range[1]+dh/2],
                [x_range[0]-dh/2, y_range[1]+dh/2]]
    
    params = {
        "end_time" : T,
        "out_freq" : T,
        "cfl" : 0.5, 
        "flux_scheme": "Roe Exner", 
        "integrator": "Forward Euler",
        "parNx": 2, 
        "parNy": 2,
        "initial_conditions": "input.nc",
        "boundaries": {
            "outlet1": {
                "type": "Transmissive",
                "polygon": polygon1,
                #         [h, q, z, qb] 
                "values": [0.0,0.0,0.0,0.0],
                "normal": [-1.0,0.0]
            },
            "outlet2": {
                "type": "Transmissive",
                "polygon": polygon2,
                #         [h, q, z, qb] 
                "values": [0.0,0.0,0.0,0.0],
                "normal": [1.0,0.0]
            },
            "outlet3": {
                "type": "Transmissive",
                "polygon": polygon3,
                #         [h, q, z, qb] 
                "values": [0.0,0.0,0.0,0.0],
                "normal": [0.0,-1.0]
            },
            "outlet4": {
                "type": "Transmissive",
                "polygon": polygon4,
                #         [h, q, z, qb] 
                "values": [0.0,0.0,0.0,0.0],
                "normal": [0.0,1.0]
            }
        }
    }

    return params


if __name__ == "__main__":

    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    T = 3
    params = build_dambreak(L=20, dh=0.01, T=T)
    state, coords = run_driver(params)
    
    plt.imshow(state.h)
    plt.colorbar()
    plt.savefig(f"last_h_{rank}.png")
    plt.close()
    
    plt.imshow(state.z)
    plt.colorbar()
    plt.savefig(f"last_z_{rank}.png")
    plt.close()