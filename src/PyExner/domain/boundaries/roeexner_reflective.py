from dataclasses import dataclass
import jax
import jax.numpy as jnp
from typing import Tuple

from PyExner.domain.boundary_registry import register_boundary
from PyExner.state.roe_exner_state import RoeExnerState

@register_boundary("Roe Exner Reflective")
@dataclass
class RoeExner_ReflectiveBoundary:
    mask: jnp.ndarray                            # shape (Ny, Nx), boolean
    normal: jnp.ndarray                          # shape (2,), [nx, ny]
    interior_indices: Tuple[jnp.ndarray, jnp.ndarray]  # (y_interior, x_interior)
    boundary_indices: Tuple[jnp.ndarray, jnp.ndarray]  # (y_boundary, x_boundary)

    def apply(self, state: RoeExnerState, time: float) -> RoeExnerState:
        by, bx = self.boundary_indices
        iy, ix = self.interior_indices
        nx, ny = self.normal

        h_int = state.h[iy, ix]
        hu_int = state.hu[iy, ix]
        hv_int = state.hv[iy, ix]
        z_int = state.z[iy, ix]

        # Copy momentum fields from interior
        hu_int = state.hu[iy, ix]
        hv_int = state.hv[iy, ix]

        mom_n =  hu_int * nx + hv_int * ny
        mom_t = -hu_int * ny + hv_int * nx 

        mom_n = -mom_n

        hu_ref = mom_n * nx - mom_t * ny
        hv_ref = mom_n * ny + mom_t * nx

        h_new  = state.h.at[by, bx].set(h_int)
        hu_new = state.hu.at[by, bx].set(hu_ref)
        hv_new = state.hv.at[by, bx].set(hv_ref)
        z_new = state.z.at[by, bx].set(z_int)

        return RoeExnerState(h=h_new, hu=hu_new, hv=hv_new, z=z_new, n=state.n, G=state.G, seds=state.seds) 

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