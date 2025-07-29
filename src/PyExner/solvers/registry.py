# PyExner/solver/registry.py

import jax

from PyExner.parallel.mpi_utils import Parallel
from PyExner.domain.boundary_registry import BoundaryManager

from typing import Callable, NamedTuple

class SolverConfig(NamedTuple):
    mpi_handler: Parallel
    boundaries: BoundaryManager
    dx: float
    halo_exchange: Callable[[jax.Array], jax.Array]

class SolverBundle(NamedTuple):
    name: str
    config: Callable
    mask_fn: Callable
    init_fn: Callable
    step_fn: Callable
    compute_dt_fn: Callable

SOLVER_REGISTRY: dict[str, SolverBundle] = {}

def register_solver_bundle(name: str):
    def decorator(bundle: Callable):
        SOLVER_REGISTRY[name] = bundle()
        return bundle()
    return decorator

def create_solver_bundle(name: str):
    bundle = SOLVER_REGISTRY[name]
    if bundle is None:
        raise ValueError(f"Unknown solver scheme '{name}'. Available options: {list(SOLVER_REGISTRY.keys())}")
    return bundle