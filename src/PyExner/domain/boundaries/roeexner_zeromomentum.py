from dataclasses import dataclass
import jax
import jax.numpy as jnp
from typing import Tuple

from PyExner.domain.boundary_registry import register_boundary
from PyExner.state.roe_exner_state import RoeExnerState

@register_boundary("Roe Exner ZeroMomentum")
@dataclass
class RoeExner_ZeroMomentumBoundary:
    mask: jnp.ndarray        # boolean array, True where boundary is
    normal: jnp.ndarray      # [nx, ny], preserved for bookkeeping
    values: jnp.ndarray      # numeric array [h, hu, hv, z]

    def apply(self, state: RoeExnerState, time: float) -> RoeExnerState:
        """
        Sets hu and hv to zero (or values from `self.values`) at the boundary.
        Leaves h, z, G, n untouched.
        """
        hu_val = self.values[1]  # typically 0.0
        hv_val = self.values[2]

        # Use the mask directly
        hu_new = jnp.where(self.mask, jnp.where(jnp.isnan(hu_val), state.hu, hu_val), state.hu)
        hv_new = jnp.where(self.mask, jnp.where(jnp.isnan(hv_val), state.hv, hv_val), state.hv)

        return RoeExnerState(
            h=state.h,
            hu=hu_new,
            hv=hv_new,
            z=state.z,
            n=state.n,
            G=state.G,
            seds=state.seds
        )


# Pytree registration for JAX
def RoeExner_ZeroMomentumBoundary_flatten(b: RoeExner_ZeroMomentumBoundary):
    children = (b.mask, b.interior_indices, b.boundary_indices)
    return children, None

def RoeExner_ZeroMomentumBoundary_unflatten(aux, children):
    return RoeExner_ZeroMomentumBoundary(*children)

jax.tree_util.register_pytree_node(
    RoeExner_ZeroMomentumBoundary,
    RoeExner_ZeroMomentumBoundary_flatten,
    RoeExner_ZeroMomentumBoundary_unflatten
)
