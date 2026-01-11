# PyExner/solver/swe.py

import jax
import jax.numpy as jnp
import mpi4jax 
from mpi4py import MPI

from functools import partial

# local imports
from PyExner.state.roe_exner_state import RoeExnerState

from PyExner.solvers.kernels.roe_exner import (
    compute_dt_2D, roe_solve_2D, 
    exner_solve_2D, 
    make_halo_exchange, compute_G,
    momentum_corrections
)
from PyExner.solvers.registry import SolverConfig, SolverBundle, register_solver_bundle

def get_mask(state, dims, b_mask):
    mask = jnp.isnan(state.z)

    y_parts, x_parts = dims

    if y_parts != 1:
        jax.debug.print("here1")
        mask = mask.at[0, :].set(True)
        mask = mask.at[-1, :].set(True)
    
    if x_parts != 1:
        jax.debug.print("here1")
        mask = mask.at[:, 0].set(True)
        mask = mask.at[:, -1].set(True)

    return jnp.stack([mask, b_mask], axis=0)

def config_fn_roeexner(state, mpi_handler, boundaries, dx):
    halo_exchange = make_halo_exchange(mpi_handler)

    return SolverConfig(
        mpi_handler = mpi_handler,
        boundaries = boundaries,
        dx = dx, 
        halo_exchange = halo_exchange
    )

def init_fn_roeexner(state: RoeExnerState, mask, config: SolverConfig) -> RoeExnerState:

    h = config.halo_exchange(state.h)
    hu = config.halo_exchange(state.hu)
    hv = config.halo_exchange(state.hv)
    z = config.halo_exchange(state.z)
    n = config.halo_exchange(state.n)

    state = state.replace(h=h, hu=hu, hv=hv, z=z, n=n)
    
    G = compute_G(state.h, state.hu, state.hv, state.n, state.seds, mask)
    G = config.halo_exchange(G)

    return state.replace(G=G)

#@partial(jax.jit, static_argnums=(4,))
def step_fn_roeexner(state: RoeExnerState, time: float, dt: float, mask, config: SolverConfig) -> RoeExnerState:    
    # Step 1: Solve hydrodynamics
    state = roe_solve_2D(state, dt, config.dx, mask)
    
    # Step 2: Apply boundary conditions FIRST (fills ghost cells)
    state = config.boundaries.apply(state, time)
    
    # Step 3: Momentum corrections (uses boundary-corrected values)
    state = momentum_corrections(state, mask)
    
    # Step 4: Now halo exchange (sends corrected values to neighbors)
    h = config.halo_exchange(state.h)
    hu = config.halo_exchange(state.hu)
    hv = config.halo_exchange(state.hv)
    #z = config.halo_exchange(state.z)
    #n = config.halo_exchange(state.n)
    
    state = state.replace(h=h, hu=hu, hv=hv)

    # Step 5: Compute morphodynamics source term
    G = compute_G(state. h, state.hu, state.hv, state.n, state.seds, mask)    
    G = config.halo_exchange(G) 
    state = state.replace(G=G) 

    # Step 6: Solve Exner (bed evolution)
    state = exner_solve_2D(state, dt, config.dx, mask)
    
    # Step 7: Final halo sync of bed level
    z_final = config.halo_exchange(state.z)

    return state.replace(z=z_final) 


def compute_dt_roeexner(state: RoeExnerState, cfl: float, mask: jax.Array, config: SolverConfig) -> float:
    local_dt = cfl * compute_dt_2D(state, config.dx, mask)
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
