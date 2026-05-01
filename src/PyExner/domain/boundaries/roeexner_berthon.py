from dataclasses import dataclass
import jax
from typing import Tuple
import jax.numpy as jnp

from PyExner.domain.boundary_registry import register_boundary
from PyExner.state.roe_exner_state import RoeExnerState


@register_boundary("Roe Exner Berthon")
@dataclass
class RoeExner_BerthonBoundary:
    mask: jnp.ndarray          # boundary mask
    normal: jnp.ndarray
    values: jnp.ndarray        # [q, A, alpha, beta, p, u_cr, C]
    interior_indices: Tuple[jnp.ndarray, jnp.ndarray]
    boundary_indices: Tuple[jnp.ndarray, jnp.ndarray]

    def apply(self, state: RoeExnerState, time: float) -> RoeExnerState:
        """
        Exact analytical inlet BC from:

        Berthon et al.
        'An analytical solution of Shallow Water system coupled to Exner equation'

        values =
            q     : constant discharge
            A     : sediment coefficient
            alpha : linear qb slope
            beta  : qb intercept
            C     : integration constant
            x     : x-coordinate of boundary
        """

        by, bx = self.boundary_indices
        iy, ix = self.interior_indices

        g = 9.81

        q     = self.values[0]
        A     = self.values[1]
        alpha = self.values[2]
        beta  = self.values[3]
        gamma = self.values[4]
        x     = self.values[5]

        # protect against negative argument
        u = jnp.power((alpha * x + beta) / A, 1.0/3.0)

        # bed elevation
        zb0 = -(u**3 + 2.0 * g * q) / (2.0 * g * jnp.maximum(u, 1e-12)) + gamma
        zb = -alpha * time + zb0
        
        return state.replace(
            z_b=state.z_b.at[by, bx].set(zb)
        )


# ---------------- pytree registration ----------------

def RoeExner_BerthonBoundary_flatten(b):
    children = (b.mask, b.normal, b.values)
    aux_data = None
    return children, aux_data


def RoeExner_BerthonBoundary_unflatten(aux_data, children):
    return RoeExner_BerthonBoundary(*children)


jax.tree_util.register_pytree_node(
    RoeExner_BerthonBoundary,
    RoeExner_BerthonBoundary_flatten,
    RoeExner_BerthonBoundary_unflatten
)