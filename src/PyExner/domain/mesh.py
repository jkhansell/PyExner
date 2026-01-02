# PyExner/domain/mesh.py

from dataclasses import dataclass
import jax
import jax.numpy as jnp
import os

from typing import Tuple

@dataclass
class Mesh2D:
    """
    A uniform 2D structured mesh for SWE-Exner simulations.
    """    
    global_Ny: int
    global_Nx: int
    local_Ny: int
    local_Nx: int
    x_offset: int
    y_offset: int
    local_X: jax.Array
    local_Y: jax.Array
    dh: float
    local_shape: Tuple
    global_shape: Tuple