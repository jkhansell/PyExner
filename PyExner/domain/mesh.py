# PyExner/domain/mesh.py

import jax.numpy as jnp
from typing import Tuple

class Mesh2D:
    """
    A uniform 2D structured mesh for SWE-Exner simulations.
    """

    def __init__(self, params):

        X = params["X"]
        Y = params["Y"]

        self.dx = params["dh"]

        self.nx = X.shape[1]
        self.ny = X.shape[0]

        self.Lx = X[0,-1] - X[0,0]
        self.Ly = Y[0,0] - Y[-1,0]
        
        self.shape = (self.ny, self.nx)
        
        self.X = jnp.array(X)
        self.Y = jnp.array(Y)

    def shape(self) -> Tuple[int, int]:
        """Returns the grid shape (ny, nx)."""
        return self.ny, self.nx

    def cell_centers(self) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Returns X, Y meshgrid of cell center coordinates."""
        return self.X, self.Y

    def spacing(self) -> Tuple[float, float]:
        return self.dx
