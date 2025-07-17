from dataclasses import dataclass
import jax
import jax.numpy as jnp

from PyExner.state.registry import register_state


@register_state("Roe")
@dataclass
class RoeState:
    h: jnp.ndarray
    hu: jnp.ndarray
    hv: jnp.ndarray
    z: jnp.ndarray
    n: jnp.ndarray

    @classmethod
    def empty(cls, mesh: 'Mesh2D', dtype=jnp.float32) -> "RoeState":
        shape = mesh.shape  # e.g., (nx, ny)
        zeros = jnp.zeros(shape, dtype=dtype)
        ones = jnp.ones(shape, dtype=dtype)
        # Initialize bed elevation z and source term G as needed
        return cls(
            h=zeros,
            hu=zeros,
            hv=zeros,
            z=ones,    # Flat bed default
            n=zeros,
        )

    @classmethod
    def from_params(cls, params: dict):
        # These are arrays.
        h = jnp.array(params.get("h_init"))
        hu = jnp.array(h*params.get("u_init"))
        hv = jnp.array(h*params.get("v_init"))
        z = jnp.array(params.get("z_init"))
        n = jnp.array(params.get("roughness"))

        return cls(h=h, hu=hu, hv=hv, z=z, n=n)

def Roe_state_flatten(state: RoeState):
    children = (state.h, state.hu, state.hv, state.z, state.n)
    return children, None

def Roe_state_unflatten(aux, children):
    return RoeState(*children)

jax.tree_util.register_pytree_node(
    RoeState, Roe_state_flatten, Roe_state_unflatten
)
