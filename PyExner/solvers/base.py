# PyExner/solver/base.py
import jax.numpy as jnp

from abc import ABC, abstractmethod

# local imports
from PyExner.domain.mesh import Mesh2D
from PyExner.domain.boundary_registry import BoundaryManager

class BaseSolver2D(ABC):
    """
    Abstract base class for 2D solvers (e.g., SWE, SWE-Exner).
    Provides a common interface and lifecycle.
    """

    def __init__(self, mesh: Mesh2D, boundaries: BoundaryManager):
        self.mesh = mesh
        self.boundaries = boundaries
        self.dx = self.mesh.spacing()
        self.X, self.Y = mesh.cell_centers()
        self.time: float = 0.0

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
    def _compute_new_state(self):
        pass

    @abstractmethod
    def apply_boundary_conditions(self):
        """
        Apply boundary conditions
        """
        pass

    @abstractmethod
    def get_state(self):
        """
        Return the current simulation state.
        """
        pass


    def exchange_halos(self):
        """
        Optional override for halo exchange in distributed mode.
        """
        pass
