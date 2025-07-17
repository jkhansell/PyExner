# PyExner/solver/swe.py

import jax.numpy as jnp

# local imports
from PyExner.solvers.base import BaseSolver2D
from PyExner.state.roe_state import RoeState
from PyExner.solvers.kernels import compute_dt, roe_solve_2D, update_state
from PyExner.solvers.registry import register_solver
from PyExner.domain.mesh import Mesh2D

@register_solver("Roe")
class RoeSolver(BaseSolver2D):
    """
    Solver for the 2D Shallow Water Equations (SWE).
    Tracks water height (h), momenta (hu, hv), and bed elevation (z).
    """

    def initialize(self, init_fields: RoeState):
        """
        Initialize SWE state with keys:
        - 'h': water height
        - 'u': x-velocity
        - 'v': y-velocity
        - 'z': bed elevation
        - 'n': Manning roughness
        """

        self.state = init_fields
        self.fluxes = jnp.zeros((self.mesh.shape[0],self.mesh.shape[1],3))
    
    def step(self, time: float, dt: float):
        fluxes = self._compute_fluxes()
        
        self._update_state(fluxes, dt, self.dx)
        self.apply_boundary_conditions(time)

    def _compute_timestep(self, cfl: float) -> float:
        return cfl*compute_dt(self.state, self.dx)

    def _compute_fluxes(self):
        return roe_solve_2D(self.fluxes, self.state, self.dx) 

    def _update_state(self, fluxes: jnp.ndarray, dt: float, dx: float):
        self.state = update_state(self.state, fluxes, dt, dx)

    def apply_boundary_conditions(self, time: float):
        self.boundaries.apply(self.state, time)

    def get_state(self):
        return self.state