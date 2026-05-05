from dataclasses import dataclass
import jax
import jax.numpy as jnp
from typing import Tuple

from PyExner.domain.boundary_registry import register_boundary
from PyExner. state. roe_exner_state import RoeExnerState

@register_boundary("Roe Exner TransmissiveBed")
@dataclass
class RoeExner_TransmissiveBedBoundary:
    mask: jax.Array
    normal: jax.Array                 # [nx, ny]
    interior_indices: Tuple[jax.Array, jax.Array]
    boundary_indices: Tuple[jax.Array, jax.Array]

    def apply(self, state: RoeExnerState, time: float) -> RoeExnerState:
        by, bx = self.boundary_indices
        iy, ix = self.interior_indices
        nx, ny = self.normal

        z_b  = state.z_b[iy, ix]
        # normal / tangential velocity
        
        return state.replace(
            z_b = state.z_b.at[by, bx].set(z_b)
        )


def RoeExner_TransmissiveBedBoundary_flatten(b: RoeExner_TransmissiveBedBoundary):
    children = (b.mask, b.normal, b.interior_indices, b.boundary_indices)
    return children, None

def RoeExner_TransmissiveBedBoundary_unflatten(aux, children):
    return RoeExner_TransmissiveBedBoundary(*children)

jax.tree_util.register_pytree_node(
    RoeExner_TransmissiveBedBoundary,
    RoeExner_TransmissiveBedBoundary_flatten,
    RoeExner_TransmissiveBedBoundary_unflatten
)