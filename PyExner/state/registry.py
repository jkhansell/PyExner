# PyExner/state/registry.py
import traceback

from PyExner.domain.mesh import pad_with_mask


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
        print(f"[create_state] Unknown state '{name}'. Available options: {list(STATE_REGISTRY.keys())}")
        traceback.print_stack()
        raise

    except Exception as e:
        print(f"[create_state] Exception occurred while creating state '{name}': {e}")
        traceback.print_exc()
        raise

def create_state_from_sharding(name: str, params, sharding, meshdims):
    try:
        state_cls = STATE_REGISTRY[name]
        print(f"[create_state] Creating state: {name}")

        data = {
            "h_init": pad_with_mask(params["h_init"], meshdims)[0],
            "u_init": pad_with_mask(params["u_init"], meshdims)[0],
            "v_init": pad_with_mask(params["v_init"], meshdims)[0],
            "z_init": pad_with_mask(params["z_init"], meshdims)[0],
            "roughness": pad_with_mask(params["roughness"], meshdims)[0],
        }

        return state_cls.from_sharding(data, sharding)

    except KeyError:
        print(f"[create_state] Unknown state '{name}'. Available options: {list(STATE_REGISTRY.keys())}")
        traceback.print_stack()
        raise

    except Exception as e:
        print(f"[create_state] Exception occurred while creating state '{name}': {e}")
        traceback.print_exc()
        raise