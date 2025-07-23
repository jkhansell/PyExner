import numpy as np
import matplotlib.pyplot as plt
import functools
from scipy.optimize import fsolve
import jax
import time 

from PyExner import run_driver
from PyExner.utils.constants import g

def solve_polynomial_numerically(hl, hr):
    # Fallback to fsolve if polynomial root finding fails or selects an unphysical one
    def equation_for_cm(cm, g, hlt, hrt):

        cr = np.sqrt(g*hrt)
        cl = np.sqrt(g*hlt)

        term1 = -8 * cr**2 * cm**2 * (cl - cm)**2
        term2 = (cm**2 - cr**2)**2 * (cm**2 + cr**2)
        return term1 + term2

    func_to_solve_cm = functools.partial(equation_for_cm, g=g, hlt=hl, hrt=hr)
    initial_guess_cm = (np.sqrt(g * hl) + np.sqrt(g * hr))
    c_m = fsolve(func_to_solve_cm, initial_guess_cm)
    print("Obtained c_m: {}".format(c_m))

    return c_m[0]


def dambreak_on_wet_no_friction_analytical(t, x, L=10, hl=0.005, hr=0.001, x0=5):
    # Specifically designed to test SWE solver only
    # SWASHES

    cm = solve_polynomial_numerically(hl, hr)

    xat = lambda t: x0 - t*np.sqrt(g*hl)
    xbt = lambda t: x0 + t*(2*np.sqrt(g*hl)-3*cm)
    xct = lambda t: x0 + t*(2*cm**2*(np.sqrt(g*hl)-cm))/(cm**2-g*hr)

    h_1 = lambda t, x: hl
    h_2 = lambda t, x: (4/(9*g))*(np.sqrt(g*hl)-(x - x0)/(2*t))**2
    h_3 = lambda t, x: cm**2/g
    h_4 = lambda t, x: hr

    u_1 = lambda t, x: 0
    u_2 = lambda t, x: (2/3)*((x-x0)/t + np.sqrt(g*hl))
    u_3 = lambda t, x: 2*(np.sqrt(g*hl)-cm)
    u_4 = lambda t, x: 0

    h = np.where(
        x <= xat(t), h_1(t,x), np.where(
            (xat(t) <= x) & (x <= xbt(t)), h_2(t,x), np.where(
                (xbt(t) <= x) & (x <= xct(t)), h_3(t,x), np.where(
                    xct(t) <= x, h_4(t,x), h_4(t,x)
                )
            )
        )
    )

    u = np.where(
        x <= xat(t), u_1(t,x), np.where(
            (xat(t) <= x) & (x <= xbt(t)), u_2(t,x), np.where(
                (xbt(t) <= x) & (x <= xct(t)), u_3(t,x), np.where(
                    xct(t) <= x, u_4(t,x), u_4(t,x)
                )
            )
        )
    )

    return h, u


def build_dambreak(Ll=0, Lr=10, hl=0.005, hr=0.001, x0=5, T=6, dh=0.01):
    x_range = [Ll, Lr]
    y_range = [0,.1]

    x = np.arange(x_range[0]-dh, x_range[1]+dh, dh)
    y = np.arange(y_range[0]-dh, y_range[1]+dh, dh)

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
        "end_time" : T,
        "cfl" : 0.5,
        "flux_scheme": "Roe", 
        "integrator": "Forward Euler", 
        "parallel": True,
        "parNx": 4,
        "parNy": 2, 
        "out_freq" : 1,
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


def pad_with_mask(arr, shard_dims):
    h, w = arr.shape
    shard_h, shard_w = shard_dims

    pad_h = (shard_h - (h % shard_h)) % shard_h
    pad_w = (shard_w - (w % shard_w)) % shard_w

    pad_height = (0, pad_h)
    pad_width = (0, pad_w)

    padded = jnp.pad(arr, (pad_height, pad_width), mode='constant', constant_values=0.0)

    mask = jnp.ones_like(arr, dtype=bool)
    mask = jnp.pad(mask, (pad_height, pad_width), mode='constant', constant_values=False)

    return padded, mask, (pad_height, pad_width)

if __name__ == "__main__":

    T = 6
    params = build_dambreak(T=T, dh=0.01)
    
    a = time.perf_counter()
    last_state = run_driver(params)
    b = time.perf_counter()

    print(f"Elapsed time: {b-a}")

    x = params["X"][params["X"].shape[0]//2]
    h = last_state.h

    print(h.shape)
    h_a, u_a = dambreak_on_wet_no_friction_analytical(T, x)

    plt.plot(x, h_a, linestyle="dashed", linewidth=4, c="black", label="Analytical")
    plt.plot(x, h[h.shape[0]//2], linewidth=2, c="red", label="PyExner")
    plt.legend()
    plt.savefig("h.png", dpi=200)
    plt.close() 

    """import jax
    import jax.numpy as jnp
    from jax.sharding import NamedSharding, PartitionSpec
    import os
    import sys

    proc_id = int(os.environ['SLURM_PROCID'])
    total_procs = int(os.environ['SLURM_NPROCS'])
    visible_devices = [int(gpu) for gpu in os.environ['CUDA_VISIBLE_DEVICES'].split(',')]

    parNx = int(sys.argv[1])
    parNy = int(sys.argv[2])

    jax.distributed.initialize(local_device_ids=visible_devices, num_processes=total_procs, process_id=proc_id)

    print("process id =", jax.process_index())
    print("global devices =", jax.devices())
    print("local devices =", jax.local_devices())

    if parNy*parNx != 1:
        mesh_dims = (parNy, parNx)

        # make device mesh
        mesh = jax.make_mesh(mesh_dims, ('y', 'x')) 

        data, mask, pad_dims = pad_with_mask(params["X"], mesh_dims)

        sharding = NamedSharding(mesh, PartitionSpec('y', 'x'))
        global_X = jax.device_put(data, sharding)
    else:
        global_X = jax.device_put(params["X"]) 
        
    res1 = global_X[:, 1:]
    res2 = global_X[:,:-1]
    res1 = global_X[:, 1:]
    res2 = global_X[:,:-1]

    global_X = global_X.at[:,:-1].set(res1 - res2)
    global_X = global_X.at[:,1:].set(res1 + res2)


    for shard in global_X.addressable_shards:
        print(f"device {shard.device} has local data {shard.data}")

    jax.distributed.shutdown()"""