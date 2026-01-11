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
        self.vars = {}
        self.numOut = 0
        
        self.infile = pnetcdf.File(
            filename = self.in_file_path,
            mode = "r",
            format = "NC_64BIT_DATA", 
            comm = self.comm,
            info=None
        )

        self.outfile = pnetcdf.File(
            filename = self.out_file_path,
            mode = "w",
            format = "NC_64BIT_DATA",
            comm = self.comm,
            info=None
        )

    def __del__(self):
        self.outfile.close()

    def generate_mesh(self):
                
        global_Ny = len(self.infile.dimensions['y'])
        global_Nx = len(self.infile.dimensions['x']) 

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

        self.local_y = np.array(self.infile.variables["y"][y_offset:y_offset+local_Ny])
        self.local_x = np.array(self.infile.variables["x"][x_offset:x_offset+local_Nx])
        
        dh = round(float((self.local_x[1:] - self.local_x[:-1])[0]),5)

        if x_parts != 1:
            self.local_x = extend_with_ghosts(self.local_x, dh)
            local_Nx += 2

        if y_parts != 1:
            self.local_y = extend_with_ghosts(self.local_y, dh)
            local_Ny += 2

        local_x = jnp.array(self.local_x) + dh/2
        local_y = jnp.array(self.local_y) - dh/2

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

        return Mesh2D(**meshdata)

    def read_state(self, state_instance, mesh, config):
        """Populate the fields of the given state instance in-place."""
        new_values = {}
        y_parts, x_parts = self.mpihandler.dims
        has_x_halo = x_parts != 1
        has_y_halo = y_parts != 1

        # Remove halos from dataset input if they are expected to be exchanged via MPI
        local_Nx = mesh.local_Nx - 2 if has_x_halo else mesh.local_Nx
        local_Ny = mesh.local_Ny - 2 if has_y_halo else mesh.local_Ny

        for f in fields(state_instance):
            name = f.name

            if name not in self.infile.variables:
                if name == "G":
                    if type(config["erosion"]["grass_factor"]) is str:
                        new_values[name] = jnp.zeros((local_Ny, local_Nx))
                    else:
                        new_values[name] = config["erosion"]["grass_factor"]*jnp.ones((local_Ny, local_Nx))
                    continue        

                if name == "seds":
                    if type(config["erosion"]["grass_factor"]) is str:
                        if config["erosion"]["grass_factor"] == "MPM":
                            theta_c = 0.047
                        else: 
                            # default to MPM formulation -- To be developed 
                            theta_c = 0.047

                        sediments = []
                        for key in config["erosion"]["sediments"].keys():
                            sed_i = [
                                config["erosion"]["sediments"][key]["fraction"],
                                config["erosion"]["sediments"][key]["diameter"],
                                config["erosion"]["sediments"][key]["density"],
                                config["erosion"]["sediments"][key]["deposition_flux"],
                                config["erosion"]["sediments"][key]["erosion_flux"],
                                theta_c,
                                config["erosion"]["bulk_porosity"],
                            ]                                
                            sediments.append(sed_i)

                        new_values[name] = jnp.array(sediments).T
                    else:
                        new_values[name] = jnp.empty(1)
                    
                    continue

                raise KeyError(f"Variable '{name}' not found in NetCDF file.")
                        
            data = jnp.array(self.infile.variables[name][
                mesh.y_offset : mesh.y_offset + local_Ny,
                mesh.x_offset : mesh.x_offset + local_Nx
            ]).astype(jnp.float32)

            # Reintroduce halo padding for in-process ghost cells
            if has_x_halo:
                data = jnp.pad(data, ((0, 0), (1, 1)), mode="edge")

            if has_y_halo:
                data = jnp.pad(data, ((1, 1), (0, 0)), mode="edge")

            new_values[name] = data

        # Also initialize output file 

        dim_t = self.outfile.def_dim("t")
        dim_y = self.outfile.def_dim("y", size=mesh.global_Ny)
        dim_x = self.outfile.def_dim("x", size=mesh.global_Nx)    

        # define coordinates and fields for outfile
        var_y = self.outfile.def_var("y", pnetcdf.NC_DOUBLE, (dim_y))
        var_x = self.outfile.def_var("x", pnetcdf.NC_DOUBLE, (dim_x))

        for f in fields(state_instance):
            if f.name == "seds":
                continue
            self.vars[f.name] = self.outfile.def_var(f.name, pnetcdf.NC_FLOAT, (dim_t, dim_y, dim_x))

        # exit define mode
        self.outfile.enddef()

        # write coordinates to outfile
        start = [mesh.x_offset]
        count = [local_Nx]

        var_x.put_var_all(
            self.infile.variables["x"][mesh.x_offset : mesh.x_offset + local_Nx],
            start = start,
            count = count
        )
        
        start = [mesh.y_offset]
        count = [local_Ny]

        var_y.put_var_all(
            self.infile.variables["y"][mesh.y_offset : mesh.y_offset + local_Ny],
            start = start, 
            count = count    
        )

        # close input file
        self.infile.close()

        return state_instance.replace(**new_values)

    def write_state(self, state_instance, mesh, mask):

        y_parts, x_parts = self.mpihandler.dims

        has_x_halo = x_parts != 1
        has_y_halo = y_parts != 1

        # Remove halos from dataset input if they are expected to be exchanged via MPI
        local_Nx = mesh.local_Nx - 2 if has_x_halo else mesh.local_Nx
        local_Ny = mesh.local_Ny - 2 if has_y_halo else mesh.local_Ny

        start = [self.numOut, mesh.y_offset, mesh.x_offset]
        count = [1, local_Ny, local_Nx]

        state = state_instance.to_host()
        for f in fields(state):
            if f.name == "seds":
                continue
            
            arr = jnp.where(mask, jnp.nan, getattr(state, f.name))
            arr = np.ascontiguousarray(jnp.expand_dims(arr, 0))

            self.vars[f.name].put_var_all(arr.astype(jnp.float32), start=start, count=count)
        
        self.numOut += 1