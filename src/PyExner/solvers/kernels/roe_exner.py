import jax
import jax.numpy as jnp
from functools import partial

import mpi4jax
from mpi4py import MPI

#local imports
from PyExner.state.roe_exner_state import RoeExnerState
from PyExner.utils.constants import g, DRY_TOL, VEL_TOL

import matplotlib.pyplot as plt

def momentum_corrections(state: RoeExnerState, mask):
    h = state.h
    z = state.z

    # Pad to prevent wrap-around at the MPI/Domain edges
    h_pad = jnp.pad(h, 1, mode='edge')
    z_pad = jnp.pad(z, 1, mode='edge')
    mask_pad = jnp.pad(mask[0], 1, mode='edge')

    # Neighbors mapped to your definition:
    # East/West -> Row shifts
    hWest = h_pad[1:-1, :-2]; hEast = h_pad[1:-1, 2:]
    zWest = z_pad[1:-1, :-2]; zEast = z_pad[1:-1, 2:]
    mWest = mask_pad[1:-1, :-2]; mEast = mask_pad[1:-1, 2:]
    
    # North (iy=0) / South (iy=Ny-1) -> Column shifts
    hNorth = h_pad[:-2, 1:-1]; hSouth = h_pad[2:, 1:-1]
    zNorth = z_pad[:-2, 1:-1]; zSouth = z_pad[2:, 1:-1]
    mNorth = mask_pad[:-2, 1:-1]; mSouth = mask_pad[2:, 1:-1]

    eta = h + z

    # Check for obstructions in the North-South direction (along ix)
    # If the neighbor is a wall (mask) or dry bed higher than local water level
    stop_u = (
        ((eta < zEast) & (hEast < DRY_TOL)) | 
        ((eta < zWest) & (hWest < DRY_TOL)) | 
        mEast |  mWest
    )

    # Check for obstructions in the East-West direction (along iy)
    stop_v = (
        ((eta < zNorth) & (hNorth < DRY_TOL)) | 
        ((eta < zSouth) & (hSouth < DRY_TOL)) |
        mNorth | mSouth
    )

    active = (h > DRY_TOL) & ~mask[0] & ~mask[1]
    
    # Enforce u.n = 0 by killing momentum heading into walls or high dry neighbors
    hu_new = jnp.where(active & ~stop_u, state.hu, 0.0)
    hv_new = jnp.where(active & ~stop_v, state.hv, 0.0)

    return RoeExnerState(
        h = state.h, 
        hu = hu_new, 
        hv = hv_new, 
        z = state.z, 
        n = state.n, 
        G = state.G,
        seds=state.seds
    )

def compute_G(h, hu, hv, n, seds, mask):

    h = jnp.maximum(h, DRY_TOL)
    hu = jnp.where(h < DRY_TOL, 0.0, hu)
    hv = jnp.where(h < DRY_TOL, 0.0, hv)

    n  = n
    # (ny,nx)
    u = hu / h
    v = hv / h

    # Lift hydraulics → (1,ny,nx)
    u = u[None, ...]
    v = v[None, ...]
    h = h[None, ...]
    n = n[None, ...]

    # Sediments → (nseds,1,1)    
    frac = seds[0][:, None, None]
    d50    =  seds[1][:, None, None]
    density = seds[2][:, None, None]
    k_d = seds[3][:, None, None]
    k_e = seds[4][:, None, None]
    theta_c = seds[5][:, None, None]
    porosity = seds[6].mean() 

    s = density / 1000

    theta_p = (
        n**2 * (u**2 + v**2)
        / ((s - 1.0) * d50 * jnp.cbrt(h))
    )                            # (nseds,ny,nx)
    theta_p = jnp.maximum(theta_p, 0.0)

    delta_theta = jnp.maximum(theta_p - theta_c, 0.0)

    gamma1 = n**3 * jnp.sqrt(g) / ((s - 1.0) * jnp.sqrt(h))
    gamma2 = 8.0 * jnp.sqrt(delta_theta) / (theta_p**1.5 + 1e-12)
    eta_p = k_e*delta_theta*d50 / (s*k_d)
    gamma3 = s*k_d*eta_p/(k_e*d50)

    G = jnp.sum(frac * gamma1 * gamma2 * gamma3, axis=0)
    G = 1 / (1 -porosity) * G
    G = jnp.where(mask[0], 0.0, G)

    return G 

def _get_theta(_lambda, utilde, ctilde):
    return 3*_lambda**2 - 4*utilde*_lambda + utilde**2 - ctilde**2

def _get_approx_lambda(_lambda, atilde, utilde, ctilde):
    theta = _get_theta(_lambda, utilde, ctilde)

    numerator = _lambda * theta - ctilde**2 * atilde * utilde 
    denominator = theta - ctilde**2*atilde 
    denominator = jnp.where(jnp.abs(denominator) < jnp.abs(ctilde**2*atilde), jnp.sign(denominator)*2*ctilde**2*atilde*0.5, denominator)  

    lambda_ = numerator / denominator

    # Dry-state fix
    lambda_ = jnp.where(atilde == 0.0, 0.0, lambda_)

    return lambda_

def compute_dt(si, sj, nx, ny, dx):
    hi, hui, hvi, zi, gi = si
    hj, huj, hvj, zj, gj = sj
    
    hi = jnp.where(hi > DRY_TOL, hi, DRY_TOL)
    hj = jnp.where(hj > DRY_TOL, hj, DRY_TOL)
    
    sqrt_i = jnp.sqrt(hi)
    sqrt_j = jnp.sqrt(hj)

    ui = jnp.where(hi > 0.0, hui/hi, 0.0)
    vi = jnp.where(hi > 0.0, hvi/hi, 0.0)

    uj = jnp.where(hj > 0.0, huj/hj, 0.0)
    vj = jnp.where(hj > 0.0, hvj/hj, 0.0)

    uhati = ui*nx + vi*ny          # normal 
    vhati = -ui*ny + vi*nx         # tangent
    
    uhatj = uj*nx + vj*ny          # normal
    vhatj = -uj*ny + vj*nx         # tangent

    htilde = 0.5*(hi + hj)
    utilde = (uhati * sqrt_i + uhatj * sqrt_j) / (sqrt_i + sqrt_j)
    vtilde = (vhati * sqrt_i + vhatj * sqrt_j) / (sqrt_i + sqrt_j) 

    ctilde = jnp.sqrt(g*htilde)
    gtilde = 0.5 * (gi + gj)
    atilde = gtilde*(uhati**2 + uhati*uhatj + uhatj**2 + vhati*vhatj)/(jnp.sqrt(hi*hj)) 
    
    # calculate wave speeds 

    lambda_1 = utilde - ctilde
    lambda_3 = utilde + ctilde

    lambda_approx_1 = _get_approx_lambda(lambda_1, atilde, utilde, ctilde)
    lambda_approx_4 = _get_approx_lambda(lambda_3, atilde, utilde, ctilde)

    # Detect degenerate Roe state
    roe_exner_speed = jnp.maximum(jnp.abs(lambda_approx_1), jnp.abs(lambda_approx_4))
    degenerate = roe_exner_speed < VEL_TOL

    dt_exner = dx / roe_exner_speed
    dt_roe = dx / jnp.maximum(jnp.abs(lambda_1), jnp.abs(lambda_3))

    dt = jnp.where(degenerate, dt_roe, dt_exner)

    return dt

#@jax.jit
def compute_dt_2D(state: RoeExnerState, dx, mask):
    h, hu, hv, z, G = state.h, state.hu, state.hv, state.z, state.G

    s1_x = (
        h[:,:-1],
        hu[:,:-1],
        hv[:,:-1],
        z[:,:-1],
        G[:,:-1]
    )

    s2_x = (
        h[:, 1:],
        hu[:, 1:],
        hv[:, 1:],
        z[:, 1:],
        G[:, 1:]
    )

    s1_y = (
        h[:-1,:],
        hu[:-1,:],
        hv[:-1,:],
        z[:-1,:],
        G[:-1,:]
    )

    s2_y = (
        h[1:, :],
        hu[1:, :],
        hv[1:, :],
        z[1:, :],
        G[1:, :]
    )

    nodata_mask_x = mask[0, :, :-1] | mask[0, :, 1:]
    nodata_mask_y = mask[0, :-1, :] | mask[0, 1:, :]

    depth_mask_x = (s1_x[0] >= DRY_TOL) | (s2_x[0] >= DRY_TOL)
    depth_mask_y = (s1_y[0] >= DRY_TOL) | (s2_y[0] >= DRY_TOL)

    active_x = depth_mask_x & ~nodata_mask_x 
    active_y = depth_mask_y & ~nodata_mask_y 

    dt_x = compute_dt(s1_x, s2_x, 1, 0, dx)
    dt_y = compute_dt(s1_y, s2_y, 0,-1, dx)

    dt_x = jnp.where(active_x, dt_x, jnp.inf).min()
    dt_y = jnp.where(active_y, dt_y, jnp.inf).min()

    dt = jnp.minimum(dt_x, dt_y)

    return dt

#@jax.jit
@jax.jit
def roe_solver(si, sj, nx: float, ny: float, dx: float):
    hi, hui, hvi, zi, ni = si
    hj, huj, hvj, zj, nj = sj

    hi = jnp.where(hi > DRY_TOL, hi, DRY_TOL)
    hj = jnp.where(hj > DRY_TOL, hj, DRY_TOL)

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
    
    h_i1star = hi + alphas[...,0] - (betas[...,0]/lambdas[...,0])      # 1st intermediate state
    h_j3star = hj - alphas[...,2] + (betas[...,2]/lambdas[...,2])     # 3rd intermediate state
    
    # Exner formulation requires unmodified source terms to extend up to interaction factors G=0.01
    
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

def roe_solve_2D(state: RoeExnerState, dt: float, dx: float, mask: jax.Array):

    h, hu, hv, z, n = state.h, state.hu, state.hv, state.z, state.n

    s1_x = (
        h[:,:-1],
        hu[:,:-1],
        hv[:,:-1],
        z[:,:-1],
        n[:,:-1]
    )

    s2_x = (
        h[:, 1:],
        hu[:, 1:],
        hv[:, 1:],
        z[:, 1:],
        n[:, 1:]
    )

    s1_y = (
        h[:-1,:],
        hu[:-1,:],
        hv[:-1,:],
        z[:-1,:],
        n[:-1,:]
    )

    s2_y = (
        h[1:, :],
        hu[1:, :],
        hv[1:, :],
        z[1:, :],
        n[1:, :]
    )

    upwP_x, upwM_x = roe_solver(s1_x, s2_x, 1,  0, dx)
    upwP_y, upwM_y = roe_solver(s1_y, s2_y, 0, -1, dx)

    # check masking per edge vs masking per cell later

    nodata_mask_x = mask[0, :, :-1] | mask[0, :, 1:]
    nodata_mask_y = mask[0, :-1, :] | mask[0, 1:, :]

    depth_mask_x = (s1_x[0] >= DRY_TOL) | (s2_x[0] >= DRY_TOL)
    depth_mask_y = (s1_y[0] >= DRY_TOL) | (s2_y[0] >= DRY_TOL)

    active_x = depth_mask_x & ~nodata_mask_x
    active_y = depth_mask_y & ~nodata_mask_y

    upwP_x = jnp.where(active_x[..., None], upwP_x, 0.0)
    upwM_x = jnp.where(active_x[..., None], upwM_x, 0.0)
    upwP_y = jnp.where(active_y[..., None], upwP_y, 0.0)
    upwM_y = jnp.where(active_y[..., None], upwM_y, 0.0)

    # upwinding solution

    fluxes = jnp.zeros((h.shape[0], h.shape[1], 3))

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

    return state.replace(
        h=h_new,
        hu=hu_new,
        hv=hv_new,
    )
   
#@jax.jit
def exner_solver(si, sj,  nx, ny, dx):
    # We will be implementing the Approximately Coupled Solver (ACM) https://doi.org/10.1016/j.advwatres.2021.103931
    hi, hui, hvi, zi, gi = si
    hj, huj, hvj, zj, gj = sj

    hi = jnp.where(hi > DRY_TOL, hi, DRY_TOL)
    hj = jnp.where(hj > DRY_TOL, hj, DRY_TOL)

    sqrt_i = jnp.sqrt(hi)
    sqrt_j = jnp.sqrt(hj)

    # Calculate velocities, handling dry beds
    ui = jnp.where(hi >= DRY_TOL, hui/hi, 0.0)
    vi = jnp.where(hi >= DRY_TOL, hvi/hi, 0.0)

    uj = jnp.where(hj >= DRY_TOL, huj/hj, 0.0)
    vj = jnp.where(hj >= DRY_TOL, hvj/hj, 0.0)

    # --- Rotate problem to (normal, tangential) coordinate frame ---
    uhati = ui * nx + vi * ny   # Normal velocity component
    uhatj = uj * nx + vj * ny

    utilde = (uhati * sqrt_i + uhatj * sqrt_j) / (sqrt_i + sqrt_j)

    umagi = (ui**2+vi**2)
    umagj = (uj**2+vj**2)

    qb_nhati = gi*umagi*uhati
    qb_nhatj = gj*umagj*uhatj

    gtilde = 0.5*(gi+gj)
    dz = zj - zi + 1e-14

    dqbhat = gtilde*umagj*uhatj - gtilde*umagi*uhati

    lambda_4 = jnp.where(jnp.abs(dz) > 1e-4, dqbhat / dz, utilde)

    corrector_i = (gtilde - gi)*umagi*uhati
    corrector_j = (gtilde - gj)*umagj*uhatj

    qbni = qb_nhati + corrector_i
    qbnj = qb_nhatj + corrector_j

    F_exner = jnp.where(lambda_4 >= 0, qbni, qbnj)

    return F_exner

#@jax.jit
def exner_solve_2D(state, dt, dx, mask):
    h, hu, hv, z, G = state.h, state.hu, state.hv, state.z, state.G

    s1_x = (
        h[:,:-1],
        hu[:,:-1],
        hv[:,:-1],
        z[:,:-1],
        G[:,:-1]
    )

    s2_x = (
        h[:, 1:],
        hu[:, 1:],
        hv[:, 1:],
        z[:, 1:],
        G[:, 1:]
    )

    s1_y = (
        h[:-1,:],
        hu[:-1,:],
        hv[:-1,:],
        z[:-1,:],
        G[:-1,:]
    )

    s2_y = (
        h[1:, :],
        hu[1:, :],
        hv[1:, :],
        z[1:, :],
        G[1:, :]
    )

    nodata_mask_x = mask[0, :, :-1] | mask[0, :, 1:]
    nodata_mask_y = mask[0, :-1, :] | mask[0, 1:, :]

    depth_mask_x = (s1_x[0] >= DRY_TOL) | (s2_x[0] >= DRY_TOL)
    depth_mask_y = (s1_y[0] >= DRY_TOL) | (s2_y[0] >= DRY_TOL)

    active_x = depth_mask_x & ~nodata_mask_x
    active_y = depth_mask_y & ~nodata_mask_y

    fluxes = jnp.zeros((state.h.shape[0], state.h.shape[1]))

    # Calculate the upwinded fluxes at the x-interfaces
    F_x = exner_solver(s1_x, s2_x, 1, 0, dx)
    F_y = exner_solver(s1_y, s2_y, 0, -1, dx)
    F_x = jnp.where(active_x, F_x, 0.0)
    F_y = jnp.where(active_y, F_y, 0.0)
    
    fluxes = fluxes.at[:, 1:].add(-F_x)
    fluxes = fluxes.at[:,:-1].add(F_x)
    fluxes = fluxes.at[1:, :].add(-F_y)
    fluxes = fluxes.at[:-1,:].add(F_y)

    z_new = state.z - dt * fluxes / dx

    return state.replace(
        z = z_new 
    )

def make_halo_exchange(mpi_handler):
    neighbors = mpi_handler.neighbors
    comm = mpi_handler.cart_comm

    #@jax.jit
    def halo_exchange(arr):
        send_order = (
            "west",
            "north",
            "east",
            "south",
        )

        # start receiving east, go clockwise
        recv_order = (
            "east",
            "south",
            "west",
            "north",
        )

        overlap_slices_send = dict(
            south=(1, slice(None)),
            west=(slice(None), 1),
            north=(-2, slice(None)),
            east=(slice(None), -2),
        )

        overlap_slices_recv = dict(
            south=(0, slice(None)),
            west=(slice(None), 0),
            north=(-1, slice(None)),
            east=(slice(None), -1),
        )

        for send_dir, recv_dir in zip(send_order, recv_order):
            send_proc = neighbors[send_dir]
            recv_proc = neighbors[recv_dir]

            if send_proc is MPI.PROC_NULL and recv_proc is MPI.PROC_NULL:
                continue

            recv_idx = overlap_slices_recv[recv_dir]
            recv_arr = jnp.empty_like(arr[recv_idx])

            send_idx = overlap_slices_send[send_dir]
            send_arr = arr[send_idx]

            if send_proc is MPI.PROC_NULL:
                recv_arr = mpi4jax.recv(recv_arr, source=recv_proc, comm=comm)
                arr = arr.at[recv_idx].set(recv_arr)
            elif recv_proc is MPI.PROC_NULL:
                mpi4jax.send(send_arr, dest=send_proc, comm=comm)
            else:
                recv_arr = mpi4jax.sendrecv(
                    send_arr,
                    recv_arr,
                    source=recv_proc,
                    dest=send_proc,
                    comm=comm,
                )
                arr = arr.at[recv_idx].set(recv_arr)
            
        return arr

    return halo_exchange