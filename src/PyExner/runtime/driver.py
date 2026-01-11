# PyExner/runtime/driver.py

from PyExner.domain.mesh import Mesh2D
from PyExner.domain.boundary_registry import BoundaryManager

from PyExner.parallel.mpi_utils import Parallel
from PyExner.io.pnetcdf_reader import PnetCDFStateIO

from PyExner.state.registry import create_empty_state
from PyExner.solvers.registry import create_solver_bundle
from PyExner.integrators.registry import create_integrator_bundle

import time
import yaml 


def run_driver(config_path: str):
    """
    Run a simulation given a full parameter dictionary.

    Expected keys in params:
        - x_range: [xmin, xmax]
        - y_range: [ymin, ymax]
        - dh: grid spacing
        - initial_conditions: dict with numpy arrays for h, u, v, z, etc.
        - boundaries: dict describing boundary polygons and types
        - endTime: float, total simulation time
        - outFreq: float, output interval
        - cfl: float, CFL number for timestep control (optional)
    """
    
    # read parameter file

    with open(config_path, "r") as f:
        params = yaml.safe_load(f)
        
    end_time = params.get("end_time", 1)
    out_freq = params.get("out_freq", 1)

    cfl = params.get("cfl", 0.5)
    flux_scheme = params.get("flux_scheme", "Roe")
    time_scheme = params.get("integrator", "Forward Euler")

    # Initialization
    mpi_handler = Parallel(params)
    pnetcdf_io = PnetCDFStateIO(params["input_file"], params["output_file"], mpi_handler)
    mesh = pnetcdf_io.generate_mesh()

    state = create_empty_state(flux_scheme, mesh, mpi_handler.rank)
    state = pnetcdf_io.read_state(state, mesh, params)

    boundaries = BoundaryManager(params, mesh.local_X, mesh.local_Y)

    solver = create_solver_bundle(flux_scheme)
    solver_config = solver.config(state, mpi_handler, boundaries, mesh.dh)

    integrator = create_integrator_bundle(time_scheme)
    integrator_config = integrator.config(cfl, end_time, out_freq, solver, solver_config)

    a = time.perf_counter()

    state = integrator.run_fn(state, integrator_config, pnetcdf_io, mesh, boundaries.boundary_mask)

    b = time.perf_counter()

    if mpi_handler.rank == 0:
        print(f"Elapsed time: {b-a}")

    return state, (mesh.local_X, mesh.local_Y)