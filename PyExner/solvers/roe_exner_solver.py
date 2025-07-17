# PyExner/solver/swe.py

import jax.numpy as jnp

# local imports
from PyExner.solvers.base import BaseSolver2D
from PyExner.utils.state.roe_state import RoeExnerState
from PyExner.solvers.kernels import compute_dt, roe_solve_2D, update_state


@register_solver("RoeExner")
class RoeExnerSolver(BaseSolver2D):
    """
    Solver for the 2D Shallow Water Equations (SWE).
    Tracks water height (h), momenta (hu, hv), and bed elevation (z).
    """

    def initialize(self, mesh: Mesh2D, init_fields: SweExnerSolverState):
        """
        Initialize SWE state with keys:
        - 'h': water height
        - 'u': x-velocity
        - 'v': y-velocity
        - 'z': bed elevation
        - 'n': Manning roughness
        """

        self.state = init_fields
        self.fluxes = jnp.zeros_like(self.state.h)
        self.mesh = mesh