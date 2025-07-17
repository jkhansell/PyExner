from dataclasses import dataclass
import jax
import jax.numpy as jnp
from typing import Any

@register_state("RoeExner")
@dataclass
class RoeExnerState:
    h: jnp.ndarray
    hu: jnp.ndarray
    hv: jnp.ndarray
    z: jnp.ndarray
    n: jnp.ndarray

    @classmethod
    def empty(cls, mesh: 'Mesh2D', dtype=jnp.float32) -> "RoeExnerState":
        shape = mesh.shape  # e.g., (nx, ny)
        zeros = jnp.zeros(shape, dtype=dtype)
        ones = jnp.ones(shape, dtype=dtype)
        # Initialize bed elevation z and source term G as needed
        return cls(
            h=zeros,
            hu=zeros,
            hv=zeros,
            z=ones,    # Flat bed default
            n=zeros,
        )

def RoeExner_state_flatten(state: RoeExnerState):
    children = (state.h, state.hu, state.hv, state.z, state.G)
    return children, None

def RoeExner_state_unflatten(aux, children):
    return RoeExnerState(*children)

jax.tree_util.register_pytree_node(
    RoeExnerState, RoeExner_state_flatten, RoeExner_state_unflatten
)
