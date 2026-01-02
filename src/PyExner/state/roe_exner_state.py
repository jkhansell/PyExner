from dataclasses import dataclass, replace as dc_replace
import jax
import jax.numpy as jnp

from PyExner.state.registry import register_state
from PyExner.state.base import BaseState

@register_state("Roe Exner")
@dataclass
class RoeExnerState(BaseState):
    z: jax.Array
    n: jax.Array
    G: jax.Array

    @classmethod
    def empty(cls, mesh: 'Mesh2D', dtype=jnp.float32) -> "RoeExnerState":
        shape = mesh.local_shape  # e.g., (nx, ny)
        zeros = jnp.zeros(shape, dtype=dtype)
        ones = jnp.ones(shape, dtype=dtype)
        # Initialize bed elevation z and source term G as needed
        return cls(
            h=zeros,
            hu=zeros,
            hv=zeros,
            z=ones,    # Flat bed default
            n=zeros,
            G=zeros
        )

    def replace(self, **kwargs):
        return dc_replace(self, **kwargs)

def RoeExner_state_flatten(state: RoeExnerState):
    children = (state.h, state.hu, state.hv, state.z, state.n, state.G)
    return children, None

def RoeExner_state_unflatten(aux, children):
    return RoeExnerState(*children)

jax.tree_util.register_pytree_node(
    RoeExnerState, RoeExner_state_flatten, RoeExner_state_unflatten
)
