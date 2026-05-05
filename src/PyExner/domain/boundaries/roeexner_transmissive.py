from dataclasses import dataclass
import jax
import jax.numpy as jnp
from typing import Tuple

from PyExner.domain.boundary_registry import register_boundary
from PyExner. state. roe_exner_state import RoeExnerState

@register_boundary("Roe Exner Transmissive")
@dataclass
class RoeExner_TransmissiveBoundary:
    mask: jax.Array
    normal: jax.Array                 # [nx, ny]
    interior_indices: Tuple[jax.Array, jax.Array]
    boundary_indices: Tuple[jax.Array, jax.Array]

    def apply(self, state: RoeExnerState, time: float) -> RoeExnerState:
        by, bx = self.boundary_indices
        iy, ix = self.interior_indices
        nx, ny = self.normal

        h  = state.h[iy, ix]
        hu = state.hu[iy, ix]
        hv = state.hv[iy, ix]
        # normal / tangential velocity
        
        return state.replace(
            h  = state.h.at[by, bx].set(h),
            hu = state.hu.at[by, bx].set(hu),
            hv = state.hv.at[by, bx].set(hv),
        )


def RoeExner_TransmissiveBoundary_flatten(b: RoeExner_TransmissiveBoundary):
    children = (b.mask, b.normal, b.interior_indices, b.boundary_indices)
    return children, None

def RoeExner_TransmissiveBoundary_unflatten(aux, children):
    return RoeExner_TransmissiveBoundary(*children)

jax.tree_util.register_pytree_node(
    RoeExner_TransmissiveBoundary,
    RoeExner_TransmissiveBoundary_flatten,
    RoeExner_TransmissiveBoundary_unflatten
)