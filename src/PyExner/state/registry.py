# PyExner/state/registry.py
import traceback


STATE_REGISTRY = {}

def register_state(name):
    def decorator(cls):
        STATE_REGISTRY[name] = cls
        return cls
    return decorator

def create_state(name: str, params):
    try:
        state_cls = STATE_REGISTRY[name]
        print(f"[create_state] Creating state: {name}")
        return state_cls.from_params(params)

    except KeyError:
        print(f"[create_state] Unknown state {name}. Available options: {list(STATE_REGISTRY.keys())}")
        traceback.print_stack()
        raise

    except Exception as e:
        print(f"[create_state] Exception occurred while creating state {name}: {e}")
        traceback.print_exc()
        raise

def create_empty_state(name, mesh, rank):
    try:
        state_cls = STATE_REGISTRY[name]
        if rank == 0:
            print(f"[create_state] Creating state: {name}")
        return state_cls.empty(mesh)

    except KeyError:
        if rank == 0:
            print(f"[create_state] Unknown state '{name}'. Available options: {list(STATE_REGISTRY.keys())}")
            traceback.print_stack()
        raise

    except Exception as e:
        if rank == 0:
            print(f"[create_state] Exception occurred while creating state '{name}': {e}")
            traceback.print_exc()
        raise