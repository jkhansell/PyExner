# PyExner/solver/swe.py
import jax
import jax.numpy as jnp

# local imports
from PyExner.state.roe_state import RoeState

from PyExner.solvers.base import BaseSolver2D
from PyExner.solvers.kernels import compute_dt, roe_solve_2D
from PyExner.solvers.registry import register_solver

from PyExner.domain.mesh import Mesh2D

@jax.jit
def pad_boundary(arr, pad_dims):
    return jnp.pad(arr, pad_dims, mode='edge')
    
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
        self.state.h = self.state.h.at[self.mesh.mask].set(0.0)
        self.state = self._compute_new_state(dt)
        self.apply_boundary_conditions(time)

    def _compute_timestep(self, cfl: float) -> float:
        return cfl*compute_dt(self.state, self.dx)

    def _compute_new_state(self, dt):
        self.fluxes = jnp.zeros((self.mesh.shape[0],self.mesh.shape[1],3))   
        return roe_solve_2D(self.fluxes, self.state, dt, self.dx)

    def apply_boundary_conditions(self, time: float):
        self.state = self.boundaries.apply(self.state, time)
        

    def get_state(self):
        if hasattr(self.mesh, "mask"):
            state = self.state[self.mesh.mask]
            return state.reshape(self.mesh.unpadded_ny, self.mesh.unpadded_nx)
        else: 
            return self.state