from dataclasses import dataclass
import jax
import jax.numpy as jnp
from typing import Tuple

from PyExner.domain.boundary_registry import register_boundary
from PyExner.state.roe_exner_state import RoeExnerState

@register_boundary("Roe Exner ConstantInflux")
@dataclass
class RoeExner_ConstantInfluxBoundary:
    mask: jnp.ndarray      # boolean mask
    normal: jnp.ndarray    # [nx, ny] normal vector
    values: jnp.ndarray    # [h, q_mag, ...]
    interior_indices: Tuple[jax.Array, jax.Array]
    boundary_indices: Tuple[jax.Array, jax.Array]

    def apply(self, state: RoeExnerState, time: float) -> RoeExnerState:
        """
        Subcritical discharge inlet BC using Riemann invariant.
        """
        g = 9.81

        # 1. DEFINE INDICES AND VALUES FIRST
        by, bx = self.boundary_indices
        iy, ix = self.interior_indices
        
        # Note: Your docstring said values[1], but your code used values[0]. 
        # Ensure you are grabbing the correct index here.
        q_mag = self.values[0] 
        
        # Make sure nx and ny are defined here! 
        # Assuming they are available via self (e.g., self.nx, self.ny)
        nx = self.values[1] 
        ny = self.values[2]

        # Multiply the prescribed magnitude by the normal vector components
        hu_b = q_mag * nx
        hv_b = q_mag * ny

        # 3. APPLY BC TO ALL RELEVANT STATE VARIABLES
        hu_new = state.hu.at[by, bx].set(hu_b)
        hv_new = state.hv.at[by, bx].set(hv_b)

        return state.replace(
            hu=hu_new,
            hv=hv_new,
        )

# JAX Pytree Registration
def RoeExner_ConstantInfluxBoundary_flatten(b: RoeExner_ConstantInfluxBoundary):
    children = (b.mask, b.normal, b.values, b.interior_indices, b.boundary_indices)
    aux_data = None
    return children, aux_data

def RoeExner_ConstantInfluxBoundary_unflatten(aux_data, children):
    return RoeExner_ConstantInfluxBoundary(*children)

jax.tree_util.register_pytree_node(
    RoeExner_ConstantInfluxBoundary,
    RoeExner_ConstantInfluxBoundary_flatten,
    RoeExner_ConstantInfluxBoundary_unflatten
)