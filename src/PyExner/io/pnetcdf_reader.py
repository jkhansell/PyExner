from mpi4py import MPI
import pnetcdf

import jax
import jax.numpy as jnp

from PyExner.parallel.mpi_utils import Parallel
from PyExner.domain.mesh import Mesh2D

from dataclasses import fields

def extend_with_ghosts(arr, dh):
    left = arr[0] - dh
    right = arr[-1] + dh
    return jnp.concatenate([jnp.array([left]), arr, jnp.array([right])])

class PnetCDFStateIO():
    def __init__(self, file_path: str, mpihandler: Parallel):
        self.file_path = file_path
        self.mpihandler = mpihandler
        self.dataset = None
        self.comm = self.mpihandler.cart_comm
        self.rank = self.mpihandler.rank
        self.size = self.mpihandler.size

    def open(self):
        self.dataset = pnetcdf.File(self.file_path, mode='r', comm=self.comm, info=None)

    def generate_mesh(self):
        self.open()
        global_Ny = len(self.dataset.dimensions['y'])
        global_Nx = len(self.dataset.dimensions['x']) 

        y_parts, x_parts = self.mpihandler.dims
        y_coord, x_coord  = self.mpihandler.coords

        # divide domain
        local_Nx = global_Nx // x_parts
        local_Ny = global_Ny // y_parts

        # remainder handling (if domain not divisible)
        x_offset = x_coord * local_Nx + min(x_coord, global_Nx % x_parts)
        y_offset = y_coord * local_Ny + min(y_coord, global_Ny % y_parts)

        if x_coord < global_Nx % x_parts:
            local_Nx += 1
        if y_coord < global_Ny % y_parts:
            local_Ny += 1

        local_y = jnp.array(self.dataset.variables["y"][y_offset:y_offset+local_Ny]) 
        local_x = jnp.array(self.dataset.variables["x"][x_offset:x_offset+local_Nx])
        
        dh = round(float((local_x[1:] - local_x[:-1])[0]),5)

        if x_parts != 1:
            local_x = extend_with_ghosts(local_x, dh)
            local_Nx += 2

        if y_parts != 1:
            local_y = extend_with_ghosts(local_y, dh)
            local_Ny += 2

        local_X, local_Y = jnp.meshgrid(local_x, local_y, indexing="xy")

        meshdata = {
            "global_Ny": global_Ny,
            "global_Nx": global_Nx,
            "local_Ny": local_Ny,
            "local_Nx": local_Nx,
            "x_offset": x_offset,
            "y_offset": y_offset,
            "local_X": local_X,
            "local_Y": local_Y,
            "dh" : dh,
            "local_shape": (local_Ny, local_Nx),
            "global_shape": (global_Ny,global_Nx)
        }

        self.close()
        return Mesh2D(**meshdata)

    def close(self):
        if self.dataset is not None:
            self.dataset.close()

    def read_state(self, state_instance, mesh):
        """Populate the fields of the given state instance in-place."""
        self.open()

        y_parts, x_parts = self.mpihandler.dims

        for f in fields(state_instance):
            name = f.name
            if name not in self.dataset.variables:
                raise KeyError(f"Variable '{name}' not found in NetCDF file.")
                        
            has_x_halo = x_parts != 1
            has_y_halo = y_parts != 1

            # Remove halos from dataset input if they are expected to be exchanged via MPI
            local_Nx = mesh.local_Nx - 2 if has_x_halo else mesh.local_Nx
            local_Ny = mesh.local_Ny - 2 if has_y_halo else mesh.local_Ny

            data = jnp.array(self.dataset.variables[name][
                mesh.y_offset : mesh.y_offset + local_Ny,
                mesh.x_offset : mesh.x_offset + local_Nx
            ])

            # Reintroduce halo padding for in-process ghost cells
            if has_x_halo:
                data = jnp.pad(data, ((0, 0), (1, 1)), mode="edge")

            if has_y_halo:
                data = jnp.pad(data, ((1, 1), (0, 0)), mode="edge")


            setattr(state_instance, name, data)
        self.close()



