from mpi4py import MPI
import pnetcdf

import jax
import jax.numpy as jnp

import numpy as np

from PyExner.parallel.mpi_utils import Parallel
from PyExner.domain.mesh import Mesh2D

from dataclasses import fields

def extend_with_ghosts(arr, dh):
    left = arr[0] - dh
    right = arr[-1] + dh
    return np.concatenate([np.array([left]), arr, np.array([right])])

class PnetCDFStateIO():
    def __init__(self, in_file_path: str, out_file_path: str, mpihandler: Parallel):
        self.in_file_path = in_file_path
        self.out_file_path = out_file_path
        self.mpihandler = mpihandler
        self.dataset = None
        self.comm = self.mpihandler.cart_comm
        self.rank = self.mpihandler.rank
        self.size = self.mpihandler.size

    def open(self, file_path, mode):
        self.dataset = pnetcdf.File(file_path, mode=mode, comm=self.comm, info=None)

    def close(self):
        if self.dataset is not None:
            self.dataset.close()

    def generate_mesh(self):
        self.open(self.in_file_path, "r")
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

        local_y = np.array(self.dataset.variables["y"][y_offset:y_offset+local_Ny]) 
        local_x = np.array(self.dataset.variables["x"][x_offset:x_offset+local_Nx])
        
        dh = round(float((local_x[1:] - local_x[:-1])[0]),5)

        if x_parts != 1:
            local_x = extend_with_ghosts(local_x, dh)
            local_Nx += 2

        if y_parts != 1:
            local_y = extend_with_ghosts(local_y, dh)
            local_Ny += 2


        local_x = jnp.array(local_x)
        local_y = jnp.array(local_y)

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

    def read_state(self, state_instance, mesh):
        """Populate the fields of the given state instance in-place."""
        self.open(self.in_file_path, "r")
        new_values = {}
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

            new_values[name] = data
        self.close()

        return state_instance.replace(**new_values)

    def write_state(self, state_instance, mesh):
        self.open(self.out_file_path, "w")


        y_parts, x_parts = self.mpihandler.dims

        req_ids = []

        has_x_halo = x_parts != 1
        has_y_halo = y_parts != 1

        # Remove halos from dataset input if they are expected to be exchanged via MPI
        local_Nx = mesh.local_Nx - 2 if has_x_halo else mesh.local_Nx
        local_Ny = mesh.local_Ny - 2 if has_y_halo else mesh.local_Ny

        dim_y = self.dataset.def_dim("y", size=local_Ny)
        dim_x = self.dataset.def_dim("x", size=local_Nx)

        for f in fields(state_instance):
            name = f.name
            var = self.dataset.def_var(varname=name, datatype=pnetcdf.NC_FLOAT, dimensions=("y", "x"))

            req_id = var.iput_var(np.asarray(getattr(state_instance, name)))
            # track the request ID for each write request
            req_ids.append(req_id)
        # wait for nonblocking writes to complete
        
        errs = [None] * len(fields(state_instance))

        self.dataset.enddef() # Exit define mode
        self.dataset.wait_all(len(fields(state_instance)), req_ids, errs)
            
        self.close()