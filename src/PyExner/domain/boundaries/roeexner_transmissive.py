from dataclasses import dataclass
import jax
import jax.numpy as jnp
from typing import Tuple

from PyExner.domain.boundary_registry import register_boundary
from PyExner. state. roe_exner_state import RoeExnerState

@register_boundary("Roe Exner Transmissive")
@dataclass
class RoeExner_TransmissiveBoundary: 
    mask: jax.Array                            # shape (Ny, Nx), boolean
    normal: jax.Array                          # shape (2,), [nx, ny]
    interior_indices: Tuple[jax.Array, jax.Array]  # (y_interior, x_interior)
    boundary_indices: Tuple[jax.Array, jax.Array]  # (y_boundary, x_boundary)

    def apply(self, state: RoeExnerState, time: float) -> RoeExnerState:
        """
        Applies transmissive boundary conditions: 
        - All fields (`h`, `hu`, `hv`, `z`, `n`, `G`) are copied from interior cells. 
        - This allows free outflow with no reflection. 
        """
        by, bx = self.boundary_indices
        iy, ix = self.interior_indices

        # Copy ALL scalar and vector fields from interior
        h_new = state.h.at[by, bx].set(state.h[iy, ix])
        hu_new = state.hu.at[by, bx].set(state.hu[iy, ix])  # FIXED: was state.h
        hv_new = state. hv.at[by, bx].set(state.hv[iy, ix])  # FIXED: was state.h
        z_new = state.z.at[by, bx]. set(state.z[iy, ix])
        n_new = state.n.at[by, bx].set(state.n[iy, ix])
        G_new = state.G.at[by, bx]. set(state.G[iy, ix])

        return RoeExnerState(
            h=h_new, 
            hu=hu_new, 
            hv=hv_new, 
            z=z_new, 
            n=n_new,
            G=G_new,
            seds=state.seds
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