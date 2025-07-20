# PyExner/domain/mesh.py

import jax
import jax.numpy as jnp
import os
from jax.sharding import NamedSharding, PartitionSpec

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


def pad_with_mask(arr, shard_dims):
    h, w = arr.shape
    shard_h, shard_w = shard_dims

    pad_h = (shard_h - (h % shard_h)) % shard_h
    pad_w = (shard_w - (w % shard_w)) % shard_w

    pad_height = (0, pad_h)
    pad_width = (0, pad_w)

    padded = jnp.pad(arr, (pad_height, pad_width), mode='edge')

    mask = jnp.ones_like(arr, dtype=bool)
    mask = jnp.pad(mask, (pad_height, pad_width), mode='constant', constant_values=False)

    return padded, mask, (pad_height, pad_width)

class ParallelMesh2D:
    def __init__(self, params):
        visible_devices = [int(gpu) for gpu in os.environ['CUDA_VISIBLE_DEVICES'].split(',')]

        jax.distributed.initialize(
            local_device_ids=visible_devices
        )

        print("[ParallelMesh2D]: Process ID ", jax.process_index())
        print("[ParallelMesh2D]: Global devices: ", jax.devices())
        print("[ParallelMesh2D]: Local devices: ", jax.devices())

        self.mesh_dims = (params["parNy"], params["parNx"])
        self.unpadded_ny, self.unpadded_nx = params["X"].shape


        # make device mesh
        self.device_mesh = jax.make_mesh(self.mesh_dims, ('y', 'x')) 

        self.X, self.mask, self.pad_dims = pad_with_mask(params["X"], self.mesh_dims)
        self.Y, _, _ = pad_with_mask(params["Y"], self.mesh_dims)

        self.sharding = NamedSharding(self.device_mesh, PartitionSpec('y', 'x'))
        
        self.Lx = self.X[0,-1] - self.X[0 ,0]
        self.Ly = self.Y[0, 0] - self.Y[-1,0]
        
        self.dx = params["dh"]

        self.nx = self.X.shape[1]
        self.ny = self.X.shape[0]

        self.shape = self.X.shape

    def cell_centers(self) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Returns X, Y meshgrid of cell center coordinates."""
        return self.X, self.Y

    def spacing(self) -> Tuple[float, float]:
        return self.dx

