
import jax
import jax.numpy as jnp
from functools import partial

#local imports
from PyExner.state.roe_state import RoeState
from PyExner.utils.constants import g, DRY_TOL, VEL_TOL

@jax.jit
def compute_masked_dt(state: RoeState, mask: jnp.ndarray, dx: float):
    h = jnp.where(mask, state.h, 0.0)
    hu = jnp.where(mask, state.hu, 0.0)
    hv = jnp.where(mask, state.hv, 0.0)

    u = hu / (h + DRY_TOL)
    v = hv / (h + DRY_TOL)
    c = jnp.sqrt(g * h)

    dt_x = jnp.where(mask, dx / (jnp.abs(u) + c + VEL_TOL), jnp.inf)
    dt_y = jnp.where(mask, dx / (jnp.abs(v) + c + VEL_TOL), jnp.inf)

    dt = jnp.minimum(jnp.min(dt_x), jnp.min(dt_y))
    return dt


@jax.jit
def compute_dt(state: RoeState, dx: float):
    h = jnp.where(state.h > DRY_TOL, state.h, 0.0)
    hv = jnp.where(state.h > DRY_TOL, state.hv, 0.0)
    hu = jnp.where(state.h > DRY_TOL, state.hu, 0.0)

    u = hu / (h + DRY_TOL)
    v = hv / (h + DRY_TOL)
    c = jnp.sqrt(g * h)

    dt_x = jnp.min(dx / (jnp.abs(u) + c + VEL_TOL))
    dt_y = jnp.min(dx / (jnp.abs(v) + c + VEL_TOL))

    dt = jnp.minimum(jnp.min(dt_x), jnp.min(dt_y))
    return dt

@jax.jit
def roe_solver(si: RoeState, sj: RoeState, nx: float, ny: float, dx: float):
    hi, hui, hvi, zi, ni = si.h, si.hu, si.hv, si.z, si.n
    hj, huj, hvj, zj, nj = sj.h, sj.hu, sj.hv, sj.z, sj.n

    hi = jnp.where(hi > DRY_TOL, hi, 0.0)
    hj = jnp.where(hj > DRY_TOL, hj, 0.0)

    sqrt_i = jnp.sqrt(hi)
    sqrt_j = jnp.sqrt(hj)

    # Calculate velocities, handling dry beds
    ui = jnp.where(hi > 0.0, hui/hi, 0.0)
    vi = jnp.where(hi > 0.0, hvi/hi, 0.0)

    uj = jnp.where(hj > 0.0, huj/hj, 0.0)
    vj = jnp.where(hj > 0.0, hvj/hj, 0.0)

    # --- Rotate problem to (normal, tangential) coordinate frame ---
    
    uhati = ui * nx + vi * ny   # Normal velocity component
    vhati = -ui * ny + vi * nx  # Tangential velocity component

    uhatj = uj * nx + vj * ny
    vhatj = -uj * ny + vj * nx

    # --- Roe Averages ---
    # To handle dry/wet transitions robustly, consider the 'h' average carefully.
    # The (sqrt_i + sqrt_j) in the denominator could be zero if both are dry.
    # Add an epsilon to the denominator or use a more robust average for dry states.
    
    htilde = jnp.where(0.5*(hi + hj) > DRY_TOL, 0.5*(hi+hj), 0.0)
    ctilde = jnp.sqrt(g*htilde)

    utilde = (uhati * sqrt_i + uhatj * sqrt_j) / (sqrt_i + sqrt_j)
    vtilde = (vhati * sqrt_i + vhatj * sqrt_j) / (sqrt_i + sqrt_j) 

    # --- Calculate Wave Speeds (Eigenvalues) ---
    lambda_1 = utilde - ctilde
    lambda_2 = utilde
    lambda_3 = utilde + ctilde

    # Entropy correction Harten-Hyman

    # lambda_1
    ei = uhati - jnp.sqrt(g*hi)
    ej = uhatj - jnp.sqrt(g*hj)

    mask_i = (ei < 0.0) & (ej > 0.0)

    lambda_1 = jnp.where(mask_i, ei*(ej-lambda_1)/((ej-ei)+VEL_TOL), lambda_1) # Replace with Lambda hat
    lambda_E1 = jnp.where(mask_i, lambda_1 - ei*(ej-lambda_1)/((ej-ei)+VEL_TOL), 0.0)
    
    # lambda_3
    ei = uhati + jnp.sqrt(g*hi)
    ej = uhatj + jnp.sqrt(g*hj)

    mask_j = (ei < 0.0) & (ej > 0.0)
    
    lambda_3 = jnp.where(mask_j, ej*(lambda_3-ei)/((ej-ei)+VEL_TOL), lambda_3)      # Replace with Lambda hat
    lambda_E3 = jnp.where(mask_j, lambda_3 - ej*(lambda_3-ei)/((ej-ei)+VEL_TOL), 0.0)

    lambda_E2 = jnp.zeros_like(lambda_E3) 
    lambdas_E = jnp.stack([lambda_E1, lambda_E2, lambda_E3], axis=-1)
    # Entropy correction Harten-Hyman

    lambdas = jnp.stack([lambda_1, lambda_2, lambda_3], axis=-1)

    P = jnp.stack([
        jnp.stack([jnp.ones_like(vtilde), jnp.zeros_like(vtilde), jnp.ones_like(vtilde)],axis=-1),
        jnp.stack([lambda_1,             jnp.zeros_like(vtilde),             lambda_3],axis=-1),
        jnp.stack([vtilde,               ctilde,                              vtilde],axis=-1)
    ], axis=-2)

    # Construct the elements of the inner matrix (before factoring out 1/(2c))
    denom = lambda_1-lambda_3 + VEL_TOL
    
    P_inv = jnp.stack([
        jnp.stack([-lambda_3/denom,      jnp.ones_like(utilde)/denom,     jnp.zeros_like(utilde)], axis=-1),
        jnp.stack([-vtilde/ctilde,       jnp.zeros_like(utilde),          jnp.ones_like(utilde)/ctilde], axis=-1),
        jnp.stack([lambda_1/denom,       -jnp.ones_like(utilde)/denom,    jnp.zeros_like(utilde)], axis=-1)
    ], axis=-2)

    # Calculate Inverse of P
    #P_inv = np.linalg.inv(P)

    # --- Difference in Conservative Variables (dU = Uj - Ui) ---
    dh = hj - hi
    # Note: dhu and dhv here are differences in normal and tangential momentum, not global.
    dhu = hj * uhatj - hi * uhati
    dhv = hj * vhatj - hi * vhati

    dU = jnp.stack([dh, dhu, dhv],axis=-1)
        # --- Wave Strengths (alphas) ---
    # alphas = P_inv * dU
    alphas = jnp.einsum("...ji,...i->...j", P_inv, dU)

    # --- Source Terms ---
    # Based on "http://dx.doi.org/10.1016/j.jcp.2010.02.016" 
    
    dz = zj - zi
    di = hi + zi
    dj = hj + zj
    dd = dj - di # Difference in total water depth

    # Topographic source term (pressure gradient due to bed slope)
    # This is a specific well-balanced formulation.
    thrust_a = -g * htilde * dz

    # Alternative thrust calculation for specific dry/wet conditions
    mask1_dz = (dz >= 0) & (di < zj)
    mask2_dz = (dz < 0) & (dj < zi)
    dztilde = jnp.where(mask1_dz, hi, jnp.where(mask2_dz, hj, dz))
    hr = jnp.where(dz >= 0, hi, hj) # Choose upstream or downstream h for reference height
    thrust_b = -g * (hr - 0.5 * jnp.abs(dztilde)) * dztilde

    # Combined thrust using specific conditions
    # This condition aims to correctly handle dry/wet fronts and supercritical/subcritical flow.
    # The condition (dz*dd >= 0.0) is often related to avoiding oscillations when the flow passes
    # over a hump or depression, and (utilde*dz > 0.0) when flow is moving uphill.
    mask_thrust = (dz * dd >= 0.0) & (utilde * dz > 0.0)
    thrust = jnp.where(mask_thrust, jnp.maximum(thrust_a, thrust_b), thrust_b)

    ntilde = 0.5*(ni+nj)
    sf = (ntilde**2*jnp.sqrt(utilde**2+vtilde**2)*utilde)/(jnp.maximum(DRY_TOL, htilde**(4/3)))
    tau = g*htilde*sf*dx

    Tn = jnp.stack([
        jnp.zeros_like(htilde),
        thrust-tau,
        jnp.zeros_like(htilde)
    ], axis=-1)

    betas = jnp.einsum("...ji,...i->...j", P_inv, Tn)

    # Reconstruction of approximate solution
    
    h_i1star = hi + alphas[...,0] - (betas[...,0]/lambdas[...,0])      # 1st intermediate state
    h_j3star = hj - alphas[...,2] + (betas[...,2]/lambdas[...,2])     # 3rd intermediate state

    beta1min = -(hi+alphas[...,0])*jnp.abs(lambdas[...,0])
    beta3min = -(hi-alphas[...,0])*jnp.abs(lambdas[...,2])

    dt = dx / jnp.max(jnp.abs(lambdas), axis=2)

    mask_1 = (h_i1star < 0.0) & (jnp.abs(hi) > VEL_TOL)
    mask_2 = (h_j3star < 0.0) & (jnp.abs(hj) > VEL_TOL)

    dt1star = (dx / 2*lambdas[...,0])*(hi/(hi-h_i1star + VEL_TOL)) 
    dt3star = (dx / 2*lambdas[...,2])*(hj/(hj-h_j3star + VEL_TOL))

    mask = (h_i1star < 0.0) & (h_j3star > 0.0) & (dt1star < dt)  
    betas = betas.at[...,0].set(jnp.where(mask, jnp.where(-beta1min >= beta3min, beta1min, betas[...,0]), betas[...,0]))
    betas = betas.at[...,2].set(jnp.where(mask, -betas[...,0], betas[...,2])) 
    
    mask = (h_i1star > 0.0) & (h_j3star < 0.0) & (dt3star < dt)  
    betas = betas.at[...,2].set(jnp.where(mask, jnp.where(-beta3min >= beta3min, beta1min, betas[...,2]), betas[...,2]))
    betas = betas.at[...,0].set(jnp.where(mask, -betas[...,2], betas[...,0]))

    # Reconstruction of approximate solution
    
    upwP = jnp.zeros_like(lambdas)
    upwM = jnp.zeros_like(lambdas)

    mask_1 = (hi < DRY_TOL) & (h_i1star < 0.0) 
    mask_2 = (hj < DRY_TOL) & (h_j3star < 0.0)

        # --- First loop: Roe with betas ---
    def roe_body(i, upwinds):
        upwP, upwM = upwinds

        _lambda = jnp.expand_dims(lambdas[..., i], axis=-1)
        _alpha = jnp.expand_dims(alphas[..., i], axis=-1)
        _beta = jnp.expand_dims(betas[..., i], axis=-1)
        P_i = P[..., i]  # shape [..., num_vars]

        flux = (_lambda * _alpha - _beta) * P_i

        mask_1x = jnp.expand_dims(mask_1, axis=-1)
        mask_2x = jnp.expand_dims(mask_2, axis=-1)

        flux_pos = jnp.where(mask_2x, 0.0,
                     jnp.where(mask_1x, flux,
                       jnp.where(_lambda > 0.0, flux, 0.0)))
        
        flux_neg = jnp.where(mask_2x, flux,
                     jnp.where(mask_1x, 0.0,
                       jnp.where(_lambda <= 0.0, flux, 0.0)))

        return (upwP + flux_pos, upwM + flux_neg)

    upwP, upwM = jax.lax.fori_loop(0, lambdas.shape[-1], roe_body, (upwP, upwM))

    # --- Second loop: Entropy fix contribution ---
    def entropy_body(i, carry):
        upwP, upwM = carry

        _lambda_E = jnp.expand_dims(lambdas_E[..., i], axis=-1)
        _alpha = jnp.expand_dims(alphas[..., i], axis=-1)
        P_i = P[..., i]

        flux = _lambda_E * _alpha * P_i

        flux_pos = jnp.where(_lambda_E > 0.0, flux, 0.0)
        flux_neg = jnp.where(_lambda_E <= 0.0, flux, 0.0)

        return (upwP + flux_pos, upwM + flux_neg)

    upwP, upwM = jax.lax.fori_loop(0, lambdas_E.shape[-1], entropy_body, (upwP, upwM))

    Tk_inv = jnp.array([[1,  0,   0],
                       [0, nx, -ny],
                       [0, ny,  nx]])

    upwP = jnp.einsum('ji,...i->...j', Tk_inv, upwP)
    upwM = jnp.einsum('ji,...i->...j', Tk_inv, upwM)

    return upwP, upwM

@partial(jax.jit, static_argnums=(2,3))
def roe_solve_2D(fluxes: jnp.ndarray, state: RoeState, pad_height: int, pad_width: int , dt: float, dx: float):

    h, hu, hv, z, n = state.h, state.hu, state.hv, state.z, state.n

    # X-direction interface slices (axis=1)
    s1_x = RoeState(
        h =   h[:,:-1], 
        hu = hu[:,:-1], 
        hv = hv[:,:-1],
        z =   z[:,:-1], 
        n =   n[:,:-1]
    )

    s2_x = RoeState(
        h =   h[:, 1:], 
        hu = hu[:, 1:], 
        hv = hv[:, 1:],
        z =   z[:, 1:], 
        n =   n[:, 1:]
    )

    # Y-direction interface slices (axis=0)
    s1_y = RoeState(
        h =   h[:-1,:], 
        hu = hu[:-1,:], 
        hv = hv[:-1,:],
        z =   z[:-1,:], 
        n =   n[:-1,:]
    )

    s2_y = RoeState(
        h =   h[1:, :], 
        hu = hu[1:, :], 
        hv = hv[1:, :],
        z =   z[1:, :], 
        n =   n[1:, :]
    )

    upwP_x, upwM_x = roe_solver(s1_x, s2_x, 1,  0, dx)
    upwP_y, upwM_y = roe_solver(s1_y, s2_y, 0, -1, dx)

    # upwinding solution

    fluxes = fluxes.at[:,:-1].add(upwM_x) 
    fluxes = fluxes.at[:, 1:].add(upwP_x) 
    fluxes = fluxes.at[:-1,:].add(upwM_y)
    fluxes = fluxes.at[1:, :].add(upwP_y)
    
    dh  = fluxes[..., 0]
    dhu = fluxes[..., 1]
    dhv = fluxes[..., 2]

    h_new  = state.h - dt * dh  / dx
    hu_new = state.hu - dt * dhu / dx
    hv_new = state.hv - dt * dhv / dx

    #Deal with boundary conditions due to padding SPMD

    h_new = h_new.at[-pad_height:,:].set(jnp.expand_dims(h_new[-(pad_height+1),:],axis=0))
    h_new = h_new.at[:,-pad_width:].set(jnp.expand_dims(h_new[:,-(pad_width+1)], axis=1))

    hu_new = hu_new.at[-pad_height:,:].set(jnp.expand_dims(hu_new[-(pad_height+1),:], axis=0))
    hu_new = hu_new.at[:,-pad_width:].set(jnp.expand_dims(hu_new[:,-(pad_width+1)], axis=1))

    hv_new = hv_new.at[-pad_height:,:].set(jnp.expand_dims(hv_new[-(pad_height+1),:], axis=0))
    hv_new = hv_new.at[:,-pad_width:].set(jnp.expand_dims(hv_new[:,-(pad_width+1)], axis=1))

    return RoeState(
        h=h_new,
        hu=hu_new,
        hv=hv_new,
        z=state.z,    # unchanged
        n=state.n     # unchanged
    )

