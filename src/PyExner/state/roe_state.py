from dataclasses import dataclass
import jax
import jax.numpy as jnp


from PyExner.state.registry import register_state
from PyExner.state.base import BaseState

@register_state("Roe")
@dataclass
class RoeState(BaseState):
    z: jax.Array
    n: jax.Array

    @classmethod
    def empty(cls, mesh: 'Mesh2D', dtype=jnp.float32) -> "RoeState":
        shape = mesh.local_shape  # e.g., (nx, ny)
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
        h = jnp.asarray(params.get("h_init"))
        hu = jnp.asarray(h*params.get("u_init"))
        hv = jnp.asarray(h*params.get("v_init"))
        z = jnp.asarray(params.get("z_init"))
        n = jnp.asarray(params.get("roughness"))

        return cls(h=h, hu=hu, hv=hv, z=z, n=n)
        
    @classmethod
    def from_sharding(cls, params, sharding):
        h  = jax.device_put(jnp.asarray(params.get("h_init")), sharding)
        hu = jax.device_put(jnp.asarray(h*params.get("u_init")), sharding)
        hv = jax.device_put(jnp.asarray(h*params.get("v_init")), sharding)
        z  = jax.device_put(jnp.asarray(params.get("z_init")), sharding)
        n  = jax.device_put(jnp.asarray(params.get("roughness")), sharding)

        return cls(h=h, hu=hu, hv=hv, z=z, n=n)



def Roe_state_flatten(state: RoeState):
    children = (state.h, state.hu, state.hv, state.z, state.n)
    return children, None

def Roe_state_unflatten(aux, children):
    return RoeState(*children)

jax.tree_util.register_pytree_node(
    RoeState, Roe_state_flatten, Roe_state_unflatten
)
