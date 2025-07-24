import jax
import jax.numpy as jnp


def unshard_state(state: RoeState) -> RoeState:
    return RoeState(
        h=device_get(state.h),
        hu=device_get(state.hu),
        hv=device_get(state.hv),
        z=device_get(state.z),
        n=device_get(state.n),
    )