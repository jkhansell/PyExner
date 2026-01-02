# PyExner/solver/swe.py

import jax
import jax.numpy as jnp
import mpi4jax 
from mpi4py import MPI

from functools import partial

# local imports
from PyExner.state.roe_exner_state import RoeExnerState

from PyExner.solvers.kernels.roe_exner import compute_dt_2D, roe_solve_2D, exner_solve_2D, make_halo_exchange
from PyExner.solvers.registry import SolverConfig, SolverBundle, register_solver_bundle

@jax.jit
def get_mask(state):
    mask = jnp.zeros_like(state.h)
    mask = mask.at[1:-1, 1:-1].set(1)
    return mask

def config_fn_roeexner(state, mpi_handler, boundaries, dx):
    halo_exchange = make_halo_exchange(mpi_handler)

    return SolverConfig(
        mpi_handler = mpi_handler,
        boundaries = boundaries,
        dx = dx, 
        halo_exchange = halo_exchange
    )

def init_fn_roeexner(state: RoeExnerState, config: SolverConfig) -> RoeExnerState:

    h = config.halo_exchange(state.h)
    hu = config.halo_exchange(state.hu)
    hv = config.halo_exchange(state.hv)
    z = config.halo_exchange(state.z)
    n = config.halo_exchange(state.n)
    G = config.halo_exchange(state.G)

    return RoeExnerState(h=h, hu=hu, hv=hv, z=z, n=n, G=G)

@partial(jax.jit, static_argnums=(3,))
def step_fn_roeexner(state: RoeExnerState, time: float, dt: float, config: SolverConfig) -> RoeExnerState:    
    state = roe_solve_2D(state, dt, config.dx)

    # Update halos
    h = config.halo_exchange(state.h)
    hu = config.halo_exchange(state.hu)
    hv = config.halo_exchange(state.hv) 

    state = RoeExnerState(h=h, hu=hu, hv=hv, z=state.z, n=state.n, G=state.G) 

    # Apply boundary conditions (must be pure)
    state = config.boundaries.apply(state, time)
    state = exner_solve_2D(state, dt, config.dx)

    z = config.halo_exchange(state.z)
    final_state = RoeExnerState(h=state.h, hu=state.hu, hv=state.hv, z=z, n=state.n, G=state.G) 

    return final_state

def compute_dt_roeexner(state: RoeExnerState, cfl: float, mask: jax.Array, config: SolverConfig) -> float:
    local_dt = cfl * compute_dt_2D(state, mask, config.dx)
    global_dt = mpi4jax.allreduce(local_dt, op=MPI.MIN, comm=config.mpi_handler.cart_comm)
    return global_dt

@register_solver_bundle("Roe Exner")
def solver_roeexner():
    return SolverBundle(
        name="Roe Exner",
        config=config_fn_roeexner,
        mask_fn=get_mask,
        init_fn=init_fn_roeexner,
        step_fn=step_fn_roeexner,
        compute_dt_fn=compute_dt_roeexner
    )
