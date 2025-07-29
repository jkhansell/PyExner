from dataclasses import dataclass
import jax
import jax.numpy as jnp
from typing import Tuple

from PyExner.domain.boundary_registry import register_boundary
from PyExner.state.roe_state import RoeState

@register_boundary("Roe Transmissive")
@dataclass
class Roe_TransmissiveBoundary:
    mask: jnp.ndarray                            # shape (Ny, Nx), boolean
    normal: jnp.ndarray                          # shape (2,), [nx, ny]
    interior_indices: Tuple[jnp.ndarray, jnp.ndarray]  # (y_interior, x_interior)
    boundary_indices: Tuple[jnp.ndarray, jnp.ndarray]  # (y_boundary, x_boundary)

    def apply(self, state: RoeState, time: float) -> RoeState:
        """
        Applies transmissive boundary conditions:
        - Scalars (`h`, `z`, `n`) are copied from interior cells.
        - Momentums (`hu`, `hv`) have the normal component reflected.
        """
        by, bx = self.boundary_indices
        iy, ix = self.interior_indices
        nx, ny = self.normal

        # Copy scalar fields
        h_new = state.h.at[by, bx].set(state.h[iy, ix])
        z_new = state.z.at[by, bx].set(state.z[iy, ix])
        n_new = state.n.at[by, bx].set(state.n[iy, ix])

        # Copy momentum fields from interior
        hu_int = state.hu[iy, ix]
        hv_int = state.hv[iy, ix]

        # Reflect normal component
        hu_ref = hu_int
        hv_ref = hv_int

        hu_new = state.hu.at[by, bx].set(hu_ref)
        hv_new = state.hv.at[by, bx].set(hv_ref)

        return RoeState(h=h_new, hu=hu_new, hv=hv_new, z=z_new, n=n_new)

def Roe_TransmissiveBoundary_flatten(b: Roe_TransmissiveBoundary):
    children = (b.mask, b.normal, b.interior_indices, b.boundary_indices)
    return children, None

def Roe_TransmissiveBoundary_unflatten(aux, children):
    return Roe_TransmissiveBoundary(*children)

jax.tree_util.register_pytree_node(
    Roe_TransmissiveBoundary,
    Roe_TransmissiveBoundary_flatten,
    Roe_TransmissiveBoundary_unflatten
)