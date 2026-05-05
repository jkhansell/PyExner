from dataclasses import dataclass
import jax
import jax.numpy as jnp
from typing import Tuple

from PyExner.domain.boundary_registry import register_boundary
from PyExner.state.roe_exner_state import RoeExnerState


@register_boundary("Roe Exner NormalFlowDepth")
@dataclass
class RoeExner_NormalFlowDepthBoundary:
    mask: jnp.ndarray
    normal: jnp.ndarray                  # [nx, ny]
    values: jnp.ndarray                 # [q_target]
    interior_indices: Tuple[jnp.ndarray, jnp.ndarray]
    boundary_indices: Tuple[jnp.ndarray, jnp.ndarray]

    def apply(self, state: RoeExnerState, time: float) -> RoeExnerState:
        """
        Subcritical boundary using prescribed normal discharge + Riemann invariant.

        values[0] = target normal flux q_target = h * u_n
        """
        by, bx = self.boundary_indices
        iy, ix = self.interior_indices

        q     = self.values[0]
        A     = self.values[1]
        alpha = self.values[2]
        beta  = self.values[3]
        gamma = self.values[4]
        x     = self.values[5]

        g = 9.81

        u = jnp.power((alpha * x + beta) / A, 1.0/3.0)
        h = (q / u)

        zb0 = -(u**3 + 2.0 * g * q) / (2.0 * g * jnp.maximum(u, 1e-12)) + gamma
        zb = -alpha * time + zb0

        # interior state
        # bed copy (or extrapolate later)

        return state.replace(
            h=state.h.at[by, bx].set(h), # Analytical head h
            hu=state.hu.at[by, bx].set(state.hu[iy, ix]), # Transmissive Flux
            hv=state.hv.at[by, bx].set(state.hv[iy, ix]), # Transmissive Flux
            z_b=state.z_b.at[by, bx].set(zb) # Analytical bed
        )


# ---------------- pytree registration ----------------

def RoeExner_NormalFlowDepthBoundary_flatten(b):
    children = (
        b.mask,
        b.normal,
        b.values,
        b.interior_indices,
        b.boundary_indices,
    )
    return children, None


def RoeExner_NormalFlowDepthBoundary_unflatten(aux_data, children):
    return RoeExner_NormalFlowDepthBoundary(*children)


jax.tree_util.register_pytree_node(
    RoeExner_NormalFlowDepthBoundary,
    RoeExner_NormalFlowDepthBoundary_flatten,
    RoeExner_NormalFlowDepthBoundary_unflatten
)