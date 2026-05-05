from dataclasses import dataclass
import jax
import jax.numpy as jnp

from PyExner.domain.boundary_registry import register_boundary
from PyExner.state.roe_exner_state import RoeExnerState


@register_boundary("Roe Exner BerthonOutlet")
@dataclass
class RoeExner_BerthonBoundary:
    mask: jnp.ndarray          # boundary mask
    normal: jnp.ndarray
    values: jnp.ndarray        # [q, A, alpha, beta, p, u_cr, C]

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
            p     : exponent
            C     : integration constant
            x     : x-coordinate of boundary
        """

        g = 9.81

        q     = self.values[0]
        A     = self.values[1]
        alpha = self.values[2]
        beta  = self.values[3]
        p     = self.values[4]
        C     = self.values[5]
        x     = self.values[6]

        # protect against negative argument
        arg = jnp.maximum((alpha * x + beta) / A, 0.0)

        # Berthon analytical solution
        ue2 = jnp.power(arg, 1.0 / p)
        u = jnp.sqrt(ue2)

        # hydraulic state
        h = q / jnp.maximum(u, 1e-12)
        hu = q * jnp.ones_like(state.hu)
        hv = jnp.zeros_like(state.hv)

        # bed elevation
        zb0 = -(u**3 + 2.0 * g * q) / (2.0 * g * jnp.maximum(u, 1e-12)) + C
        zb = -alpha * time + zb0

        h_new  = jnp.where(self.mask, h, state.h)
        hu_new = jnp.where(self.mask, hu, state.hu)
        hv_new = jnp.where(self.mask, hv, state.hv)
        zb_new = jnp.where(self.mask, zb, state.z_b)

        return state.replace(
            hu=hu_new,
            hv=hv_new,
            z_b=zb_new,
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