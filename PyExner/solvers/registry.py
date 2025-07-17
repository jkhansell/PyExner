# PyExner/solver/registry.py
import traceback

SOLVER_REGISTRY = {}

def register_solver(name):
    def decorator(cls):
        SOLVER_REGISTRY[name] = cls
        return cls
    return decorator

def create_solver(name: str, boundaries, mesh):
    solver_cls = SOLVER_REGISTRY[name]
    if solver_cls is None:
        raise ValueError(f"Unknown solver scheme '{name}'. Available options: {list(SOLVER_REGISTRY.keys())}")
    return solver_cls(mesh, boundaries)

    