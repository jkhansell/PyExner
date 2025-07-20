# PyExner/runtime/driver.py

from PyExner.domain.mesh import Mesh2D, ParallelMesh2D
from PyExner.domain.boundary_registry import BoundaryManager

from PyExner.state.registry import create_state, create_state_from_sharding
from PyExner.solvers.registry import create_solver
from PyExner.integrators.registry import create_integrator

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
    parallel = params.get("parallel", False)

    if parallel == True:
        mesh = ParallelMesh2D(params)

        state = create_state_from_sharding(flux_scheme, params["initial_conditions"], mesh.sharding, mesh.mesh_dims)
        boundaries = BoundaryManager(params, mesh.X, mesh.Y)

    else:
        mesh = Mesh2D(params)

        state = create_state(flux_scheme, params["initial_conditions"])
        boundaries = BoundaryManager(params, mesh.X, mesh.Y)

    # Creates initial conditions from parameters
    solver = create_solver(flux_scheme, boundaries, mesh)
    solver.initialize(state)

    # Creates integrator from state and solver
    integrator = create_integrator(time_scheme, solver, cfl, end_time, out_freq)
    integrator.run()

    if parallel:
        return state.unshard(solver.get_state()) 
    else:
        return solver.get_state()

