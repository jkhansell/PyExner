# PyExner/runtime/driver.py

from PyExner.domain.mesh import Mesh2D
from PyExner.domain.boundary_registry import BoundaryManager

from PyExner.parallel.mpi_utils import Parallel
from PyExner.io.pnetcdf_reader import PnetCDFStateIO

from PyExner.state.registry import create_empty_state
from PyExner.solvers.registry import create_solver
from PyExner.integrators.registry import create_integrator

import time

def run_driver(params: dict):
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
    

    end_time = params.get("end_time", 1)
    out_freq = params.get("out_freq", 1)

    cfl = params.get("cfl", 0.5)
    flux_scheme = params.get("flux_scheme", "Roe")
    time_scheme = params.get("integrator", "Forward Euler")

    mpi_handler = Parallel(params)
    pnetcdf_reader = PnetCDFStateIO(params["initial_conditions"], mpi_handler)
    
    mesh = pnetcdf_reader.generate_mesh()

    state = create_empty_state(flux_scheme, mesh, mpi_handler.rank)

    pnetcdf_reader.read_state(state, mesh)

    boundaries = boundaries = BoundaryManager(params, mesh.local_X, mesh.local_Y)

    # Creates initial conditions from parameters
    solver = create_solver(flux_scheme, mesh, boundaries, mpi_handler)
    solver.initialize(state)

    # Creates integrator from state and solver
    integrator = create_integrator(time_scheme, solver, cfl, end_time, out_freq, mpi_handler)
    
    a = time.perf_counter()
    integrator.run()
    b = time.perf_counter()

    if mpi_handler.rank == 0: 
        print(f"Elapsed Time: {b-a}")

    return solver.get_state(), solver.get_coords()

