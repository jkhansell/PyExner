# PyExner/solver/base.py
import jax.numpy as jnp

from abc import ABC, abstractmethod

# local imports
from PyExner.domain.mesh import Mesh2D
from PyExner.domain.boundary_registry import BoundaryManager
from PyExner.parallel.mpi_utils import Parallel


class BaseSolver2D(ABC):
    """
    Abstract base class for 2D solvers (e.g., SWE, SWE-Exner).
    Provides a common interface and lifecycle.
    """

    def __init__(self, mesh: Mesh2D, boundaries: BoundaryManager, mpi_handler: Parallel):
        self.mesh = mesh
        self.boundaries = boundaries
        self.dx = self.mesh.dh
        self.X = mesh.local_X
        self.Y = mesh.local_Y
        self.time: float = 0.0
        self.mpi_handler = mpi_handler

    @abstractmethod
    def initialize(self, init_fields):
        """
        Initialize the solver state with user-provided fields.
        Must be implemented in child classes.
        """
        pass

    @abstractmethod
    def step(self, dt: float):
        """
        Advance the solver state by one time step.
        """
        pass


    @abstractmethod
    def _compute_timestep(self, cfl):
        """
        Compute numerical fluxes for conserved variables.
        """
        pass

    @abstractmethod
    def get_state(self):
        """
        Return the current simulation state.
        """
        pass