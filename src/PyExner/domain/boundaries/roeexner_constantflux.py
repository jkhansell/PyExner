from dataclasses import dataclass
import jax
import jax.numpy as jnp
from typing import Tuple

from PyExner.domain.boundary_registry import register_boundary
from PyExner.state.roe_exner_state import RoeExnerState

@register_boundary("Roe Exner ConstantFlux")
@dataclass
class RoeExner_ConstantFluxBoundary:
    mask: jnp.ndarray      # array booleano
    normal: jnp.ndarray    # [nx, ny]
    values: jnp.ndarray    # [h, hu, hv, z] 

    def apply(self, state: RoeExnerState, time: float) -> RoeExnerState:
        """
        Aplica un flujo constante (hu, hv) definido en self.values.
        Si el valor en self.values es NaN, se preserva el estado actual.
        """
        
        hu_target = self.values[1]  
        hv_target = self.values[2]

        hu_new = jnp.where(
            self.mask, 
            jnp.where(jnp.isnan(hu_target), state.hu, hu_target), 
            state.hu
        )
        
        hv_new = jnp.where(
            self.mask, 
            jnp.where(jnp.isnan(hv_target), state.hv, hv_target), 
            state.hv
        )

        return RoeExnerState(
            h=state.h,
            hu=hu_new,
            hv=hv_new,
            z=state.z,
            z_b=state.z_b,
            n=state.n,
            n_b=state.n_b,
            G=state.G,
            seds=state.seds
        )

# Registro de Pytree para JAX 
def RoeExner_ConstantFluxBoundary_flatten(b: RoeExner_ConstantFluxBoundary):
    children = (b.mask, b.normal, b.values)
    aux_data = None
    return children, aux_data

def RoeExner_ConstantFluxBoundary_unflatten(aux_data, children):
    return RoeExner_ConstantFluxBoundary(*children)

jax.tree_util.register_pytree_node(
    RoeExner_ConstantFluxBoundary,
    RoeExner_ConstantFluxBoundary_flatten,
    RoeExner_ConstantFluxBoundary_unflatten
)