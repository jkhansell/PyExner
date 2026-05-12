import sys
import numpy as np
import jax.numpy as jnp

from PyExner.domain.mesh import Mesh2D


class MeshBuilderMPI:
    """
    Builds a distributed mesh WITHOUT IO dependency.
    Replaces PnetCDFStateIO.generate_mesh().
    """

    def __init__(self, mpi_handler):
        self.mpi = mpi_handler
        self.comm = mpi_handler.cart_comm
        self.rank = mpi_handler.rank

    def build(self, params):
        # ----------------------------
        # Global domain
        # ----------------------------
        Lx = params["Lx"]
        Ly = params.get("Ly", Lx)
        dh = params["dh"]

        x_global = np.arange(-Lx, Lx + dh, dh)
        y_global = np.arange( Ly, -Ly - dh, -dh)

        global_Nx = len(x_global)
        global_Ny = len(y_global)

        # ----------------------------
        # MPI decomposition
        # ----------------------------
        y_parts, x_parts = self.mpi.dims
        y_coord, x_coord = self.mpi.coords

        local_Nx = global_Nx // x_parts
        local_Ny = global_Ny // y_parts

        x_offset = x_coord * local_Nx + min(x_coord, global_Nx % x_parts)
        y_offset = y_coord * local_Ny + min(y_coord, global_Ny % y_parts)

        if x_coord < global_Nx % x_parts:
            local_Nx += 1
        if y_coord < global_Ny % y_parts:
            local_Ny += 1

        # ----------------------------
        # Local coordinates

        # ----------------------------
        local_x = x_global[x_offset:x_offset + local_Nx]
        local_y = y_global[y_offset:y_offset + local_Ny]

        # ----------------------------
        # Halo handling (same logic you had)
        # ----------------------------
        has_halo = (x_parts * y_parts != 1)

        if has_halo:
            local_x = self._extend_x(local_x, dh)
            local_y = self._extend_y(local_y, dh)

            local_Nx += 2
            local_Ny += 2

        # ----------------------------
        # Cell-centered coordinates
        # ----------------------------
        local_xc = jnp.array(local_x) + dh / 2
        local_yc = jnp.array(local_y) - dh / 2

        local_X, local_Y = jnp.meshgrid(local_xc, local_yc, indexing="xy")

        return Mesh2D(
            global_Ny=global_Ny,
            global_Nx=global_Nx,
            local_Ny=local_Ny,
            local_Nx=local_Nx,
            x_offset=x_offset,
            y_offset=y_offset,
            local_X=local_X,
            local_Y=local_Y,
            dh=dh,
            local_shape=(local_Ny, local_Nx),
            global_shape=(global_Ny, global_Nx),
        )


    # ----------------------------
    # Halo helpers (cleaned)
    # ----------------------------
    @staticmethod
    def _extend_x(arr, dh):
        return np.concatenate(([arr[0] - dh], arr, [arr[-1] + dh]))

    @staticmethod
    def _extend_y(arr, dh):
        return np.concatenate(([arr[0] + dh], arr, [arr[-1] - dh]))

def run_driver(config_path: str):

    import yaml
    from PyExner.parallel.mpi_utils import Parallel
    from PyExner.state.registry import create_empty_state
    from PyExner.solvers.registry import create_solver_bundle
    from PyExner.integrators.registry import create_integrator_bundle

    from PyExner.domain.boundary_registry import BoundaryManager

    # -----------------------------
    # Config
    # -----------------------------
    with open(config_path, "r") as f:
        params = yaml.safe_load(f)

    end_time = params.get("end_time", 1.0)
    out_freq = params.get("out_freq", 1.0)
    cfl = params.get("cfl", 0.5)

    flux_scheme = params.get("flux_scheme", "Roe")
    time_scheme = params.get("integrator", "Forward Euler")

    # -----------------------------
    # MPI
    # -----------------------------
    mpi = Parallel(params)

    # -----------------------------
    # Mesh (NOW MPI-OWNED, NOT IO)
    # -----------------------------
    mesh = MeshBuilderMPI(mpi).build(params)

    # -----------------------------
    # Initial condition
    # -----------------------------
    def ic(mesh):
        X, Y = mesh.local_X, mesh.local_Y
        Lx = params["Lx"]
        Ly = params.get("Ly", Lx)

        R = min(Lx, Ly) / 4
        mask = X**2 + Y**2 <= R**2

        h = jnp.where(mask, 1.0, 0.2)

        return {
            "h": h,
            "hu": jnp.zeros_like(h),
            "hv": jnp.zeros_like(h),
            "z": jnp.ones_like(h),
            "z_b": jnp.ones_like(h),
            "n": jnp.zeros_like(h),
            "G": params["erosion"]["grass_factor"]*jnp.ones_like(h)
        }

    state = create_empty_state(flux_scheme, mesh, mpi.rank)

    for k, v in ic(mesh).items():
        setattr(state, k, v)


    # -----------------------------
    # Boundaries
    # -----------------------------
    boundaries = BoundaryManager(params, mesh.local_X, mesh.local_Y)

    # -----------------------------
    # Solver
    # -----------------------------
    solver = create_solver_bundle(flux_scheme)
    solver_config = solver.config(state, mpi, boundaries, mesh.dh, params)

    # -----------------------------
    # Integrator
    # -----------------------------
    integrator = create_integrator_bundle(time_scheme)
    integrator_config = integrator.config(
        cfl, end_time, out_freq, solver, solver_config
    )

    # -----------------------------
    # Run
    # -----------------------------
    return integrator.run_fn(
        state,
        integrator_config,
        None,
        mesh,
        boundaries.boundary_mask,
    )

if __name__ == "__main__":

    config_path = sys.argv[1]
    run_driver(config_path)
    
