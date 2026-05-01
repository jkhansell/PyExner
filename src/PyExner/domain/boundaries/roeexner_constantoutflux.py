from dataclasses import dataclass
import jax
import jax.numpy as jnp
from typing import Tuple

from PyExner.domain.boundary_registry import register_boundary
from PyExner.state.roe_exner_state import RoeExnerState


@register_boundary("Roe Exner ConstantOutflux")
@dataclass
class RoeExner_ConstantOutfluxBoundary:
    mask: jnp.ndarray
    normal: jnp.ndarray          # [nx, ny]
    values: jnp.ndarray          # [q_out, ...]
    interior_indices: Tuple[jax.Array, jax.Array]
    boundary_indices: Tuple[jax.Array, jax.Array]

    def apply(self, state: RoeExnerState, time: float) -> RoeExnerState:

        g = 9.81

        by, bx = self.boundary_indices
        iy, ix = self.interior_indices

        # prescribed normal discharge
        q_mag = self.values[0]

        nx = self.values[1]
        ny = self.values[2]

        # interior state
        h_i = jnp.maximum(state.h[iy, ix], 1e-8)
        hu_i = state.hu[iy, ix]
        hv_i = state.hv[iy, ix]

        u_i = hu_i / h_i
        v_i = hv_i / h_i

        # normal/tangential interior velocity
        un_i = u_i * nx + v_i * ny
        ut_i = -u_i * ny + v_i * nx

        # outgoing Riemann invariant (subcritical outlet)
        c_i = jnp.sqrt(g * h_i)
        Rp = un_i + 2.0 * c_i
        h0 = h_i

        def body(_, h):
            h = jnp.maximum(h, 1e-8)
            c = jnp.sqrt(g * h)

            f = q_mag / h + 2.0 * c - Rp
            df = -q_mag / (h * h) + g / jnp.maximum(c, 1e-8)

            h_new = h - f / df
            return jnp.maximum(h_new, 1e-8)

        h_b = jax.lax.fori_loop(0, 5, body, h0)

        # reconstruct velocity from flux constraint
        un_b = q_mag / h_b

        hu_b = q_mag*nx
        hv_b = q_mag*ny

        # apply BC only on mask
        h_new  = state.h.at[by, bx].set(h_b)
        hu_new = state.hu.at[by, bx].set(hu_b)
        hv_new = state.hv.at[by, bx].set(hv_b)

        return state.replace(
            h=h_new,
            hu=hu_new,
            hv=hv_new
        )