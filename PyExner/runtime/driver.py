# PyExner/runtime/driver.py

from PyExner.domain.mesh import Mesh2D
from PyExner.domain.boundary_registry import BoundaryManager

from PyExner.state.registry import create_state
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

    end_time = params.get("endTime", 1)
    cfl = params.get("cfl", 0.5)
    flux_scheme = params.get("flux_scheme", "Roe")
    time_scheme = params.get("integrator", "Forward Euler")

    mesh = Mesh2D(params)

    # Creates initial conditions from parameters
    state = create_state(flux_scheme, params["initial_conditions"])
    
    boundaries = BoundaryManager(params, mesh.X, mesh.Y)

    # Creates initial conditions from parameters
    solver = create_solver(flux_scheme, boundaries, mesh)
    solver.initialize(state)

    # Creates integrator from state and solver
    integrator = create_integrator(time_scheme, solver, cfl, end_time)
    integrator.run()

    return solver.get_state()

