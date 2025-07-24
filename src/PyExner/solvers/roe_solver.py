# PyExner/solver/swe.py
import jax
import jax.numpy as jnp
from functools import partial
import mpi4jax
from mpi4py import MPI

# local imports
from PyExner.state.roe_state import RoeState

from PyExner.solvers.base import BaseSolver2D
from PyExner.solvers.kernels import compute_dt, roe_solve_2D, make_halo_exchange
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

        self.halo_exchange = make_halo_exchange(self.mpi_handler)

        self.state.h = self.halo_exchange(self.state.h)
        self.state.hu = self.halo_exchange(self.state.hu)
        self.state.hv = self.halo_exchange(self.state.hv)
        self.state.z = self.halo_exchange(self.state.z)
        self.state.n = self.halo_exchange(self.state.n)
        
        self.mask = jnp.zeros_like(self.state.h)
        self.mask = self.mask.at[1:-1, 1:-1].set(1)

    def step(self, time: float, dt: float):
        fluxes = jnp.zeros((self.mesh.local_shape[0],self.mesh.local_shape[1],3))

        self.state = roe_solve_2D(fluxes, self.state, dt, self.dx) 

        self.state.h = self.halo_exchange(self.state.h)
        self.state.hu = self.halo_exchange(self.state.hu)
        self.state.hv = self.halo_exchange(self.state.hv)

        self.state = self.boundaries.apply(self.state, time)

    def _compute_timestep(self, cfl: float) -> float:
        local_dt = cfl*compute_dt(self.state, self.mask, self.dx)
        global_dt = mpi4jax.allreduce(x=local_dt, op=MPI.MIN , comm=self.mpi_handler.cart_comm)
        return global_dt

    def get_state(self):
        return self.state

    def get_coords(self): 
        return self.mesh.local_X, self.mesh.local_Y 