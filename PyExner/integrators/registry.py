# PyExner/integrators/registry.py

INTEGRATOR_REGISTRY = {}

def register_integrator(name):
    def decorator(cls):
        INTEGRATOR_REGISTRY[name] = cls
        return cls
    return decorator

def create_integrator(name: str, solver, cfl, end_time, out_freq):
    integrator_cls = INTEGRATOR_REGISTRY[name]
    if integrator_cls is None:
        raise ValueError(f"Unknown integration scheme '{name}'. Available options: {list(INTEGRATOR_REGISTRY.keys())}")
    return integrator_cls(solver, cfl, end_time, out_freq)