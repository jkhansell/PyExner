import jax
import jax.numpy as jnp
import numpy as np

from dataclasses import dataclass, fields


@dataclass
class BaseState:
    h: jax.Array
    hu: jax.Array
    hv: jax.Array

    @classmethod
    def unshard(cls, state):
        unsharded = {
            f.name: jax.device_get(getattr(state, f.name)) for f in fields(cls)
        }
        return cls(**unsharded)
    
    def __getitem__(self, key):
        cls = self.__class__
        return cls(**{f.name: getattr(self, f.name)[key] for f in fields(cls)})

    def reshape(self, *shape):
        cls = self.__class__
        return cls(
            **{f.name: getattr(self, f.name).reshape(shape) for f in fields(cls)}
        )

    def apply_to_all(self, func, *args, **kwargs):
        # Apply func to all array fields and return new instance
        updated_fields = {}
        for f in fields(self.__class__):
            val = getattr(self, f.name)
            # Only apply if it is a jax.Array (you can customize type checks)
            if isinstance(val, jax.Array):
                updated_fields[f.name] = func(val, *args, **kwargs)
            else:
                updated_fields[f.name] = val
        return self.__class__(**updated_fields)
    
    def to_host(self):
        def _copy(x):
            arr = jax.device_get(x)
            return np.ascontiguousarray(arr)

        return jax.tree_util.tree_map(_copy, self)