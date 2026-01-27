from dataclasses import dataclass
import jax
import jax.numpy as jnp
from typing import Tuple

from PyExner.domain. boundary_registry import register_boundary
from PyExner.state. roe_exner_state import RoeExnerState

@register_boundary("Roe Exner Reflective")
@dataclass
class RoeExner_ReflectiveBoundary: 
    mask: jnp.ndarray                            # shape (Ny, Nx), boolean
    normal: jnp.ndarray                          # shape (2,), [nx, ny]
    interior_indices:  Tuple[jnp.ndarray, jnp.ndarray]  # (y_interior, x_interior)
    boundary_indices: Tuple[jnp. ndarray, jnp.ndarray]  # (y_boundary, x_boundary)

    def apply(self, state: RoeExnerState, time: float) -> RoeExnerState: 
        """
        Applies reflective (slip) boundary conditions. 
        Normal momentum component is negated; tangential component is preserved.
        """
        by, bx = self.boundary_indices
        iy, ix = self.interior_indices
        nx, ny = self.normal

        # Copy scalar fields
        h_int = state.h[iy, ix]
        z_b_int = state.z_b[iy, ix]
        n_int = state.n[iy, ix]
        G_int = state.G[iy, ix]

        # Extract momentum components from interior
        hu_int = state.hu[iy, ix]
        hv_int = state.hv[iy, ix]

        # Transform to normal-tangential coordinates
        mom_n = hu_int * nx + hv_int * ny      # Normal component
        mom_t = -hu_int * ny + hv_int * nx     # Tangential component

        # Reflect normal component (negate it)
        mom_n_reflected = -mom_n

        # Transform back to (hu, hv) coordinates
        hu_ref = mom_n_reflected * nx - mom_t * ny
        hv_ref = mom_n_reflected * ny + mom_t * nx

        # Apply updates
        h_new = state.h.at[by, bx].set(h_int)
        hu_new = state.hu.at[by, bx].set(hu_ref)
        hv_new = state.hv.at[by, bx].set(hv_ref)
        z_b_new = state.z_b.at[by, bx].set(z_b_int)
        n_new = state.n.at[by, bx].set(n_int)
        G_new = state.G.at[by, bx].set(G_int)

        return state.replace(
            h=h_new, 
            hu=hu_new, 
            hv=hv_new, 
            z_b=z_b_new, 
            n=n_new,
            G=G_new,
            seds=state.seds
        )

def RoeExner_ReflectiveBoundary_flatten(b: RoeExner_ReflectiveBoundary):
    children = (b.mask, b.normal, b.interior_indices, b.boundary_indices)
    return children, None

def RoeExner_ReflectiveBoundary_unflatten(aux, children):
    return RoeExner_ReflectiveBoundary(*children)

jax.tree_util.register_pytree_node(
    RoeExner_ReflectiveBoundary,
    RoeExner_ReflectiveBoundary_flatten,
    RoeExner_ReflectiveBoundary_unflatten
)