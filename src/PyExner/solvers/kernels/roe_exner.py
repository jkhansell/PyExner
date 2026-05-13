import jax
import jax.numpy as jnp
from functools import partial

import mpi4jax
from mpi4py import MPI

#local imports
from PyExner.state.roe_exner_state import RoeExnerState
from PyExner.utils.constants import g, DRY_TOL, VEL_TOL, SED_TOL

import matplotlib.pyplot as plt

@jax.jit
def momentum_corrections(state, mask):
    h   = state.h
    z   = state.z
    z_b = state.z_b
    m   = mask[0]

    # --- slice definitions ---
    C = jnp.s_[1:-1, 1:-1]
    W = jnp.s_[1:-1, 0:-2]
    E = jnp.s_[1:-1, 2:  ]
    N = jnp.s_[0:-2, 1:-1]
    S = jnp.s_[2:  , 1:-1]

    # --- fields ---
    hc = h[C]
    ztot = z + z_b

    eta = hc + ztot[C]

    hW, hE = h[W], h[E]
    hN, hS = h[N], h[S]

    zW, zE = ztot[W], ztot[E]
    zN, zS = ztot[N], ztot[S]

    mW, mE = m[W], m[E]
    mN, mS = m[N], m[S]

    # --- dry masks ---
    dryW = hW < DRY_TOL
    dryE = hE < DRY_TOL
    dryN = hN < DRY_TOL
    dryS = hS < DRY_TOL

    # --- obstruction logic ---
    stop_u = ((eta < zE) & dryE) | ((eta < zW) & dryW) | mE | mW
    stop_v = ((eta < zN) & dryN) | ((eta < zS) & dryS) | mN | mS

    active = (hc > DRY_TOL) & ~m[C]

    # --- masked momentum (branchless) ---
    hu_new = state.hu[C] * (active & ~stop_u)
    hv_new = state.hv[C] * (active & ~stop_v)

    return state.replace(
        hu=state.hu.at[C].set(hu_new),
        hv=state.hv.at[C].set(hv_new),
    )

@jax.jit
def compute_n(n, z_b, seds):
   
    frac = seds[0][:, None, None]
    d50 = seds[1][:, None, None]
    d50 = jnp.mean(d50 * frac , axis=0)
    
    np = d50**(1/6)/21.1

    return jnp.where(z_b > SED_TOL, np, n)

def compute_n_factory(is_constant: bool):
    if is_constant:
        def _compute_n(n, z_b, seds):
            return n  # Bypass sediment property calculation
        return _compute_n
    else:
        def _compute_n(n, z_b, seds):
            return compute_n(n, z_b, seds)
        return _compute_n

@jax.jit
def compute_G(h, hu, hv, seds, mask):
    h = jnp.maximum(h, DRY_TOL)

    u = hu / h
    v = hv / h
    vel2 = u**2 + v**2

    frac, d50, density, k_d, k_e, theta_c, porosity = seds

    s = density / 1000.0
    np_ = d50**(1/6) / 21.1

    # reshape sediments → (nseds,1,1) only where needed
    frac   = frac[:, None, None]
    d50    = d50[:, None, None]
    s      = s[:, None, None]
    np_    = np_[:, None, None]
    k_d    = k_d[:, None, None]
    k_e    = k_e[:, None, None]
    theta_c = theta_c[:, None, None]

    # --- compute theta_p without lifting h,u,v ---
    theta_p = (np_**2 * vel2) / ((s - 1.0) * d50 * jnp.cbrt(h))
    theta_p = jnp.maximum(theta_p, 0.0)

    delta_theta = jnp.maximum(theta_p - theta_c, 0.0)

    # --- fuse gamma terms ---
    sqrt_h = jnp.sqrt(h)

    gamma1 = np_**3 * jnp.sqrt(g) / ((s - 1.0) * sqrt_h)
    gamma2 = 8.0 * jnp.sqrt(delta_theta) / (theta_p**1.5 + 1e-5)
    gamma3 = delta_theta  # simplified (your algebra cancels)

    # combine early
    contrib = frac * gamma1 * gamma2 * gamma3

    G = jnp.sum(contrib, axis=0)

    G = G / (1.0 - porosity.mean())

    G = jnp.where(mask[0], 0.0, G)
    G = jnp.where(h > DRY_TOL, G, 0.0)

    return G

def compute_G_factory(is_constant: bool):
    """Returns a compute_G function that either passes through state.G
    (constant grass factor read from file) or fully computes it from seds (MPM)."""
    if is_constant:
        def _compute_G(h, hu, hv, seds, mask, state_G):
            G = jnp.where(mask[0], 0.0, state_G)
            G = jnp.where(h > DRY_TOL, G, 0.0)
            return G
        return _compute_G
    else:
        def _compute_G(h, hu, hv, seds, mask, state_G):
            return compute_G(h, hu, hv, seds, mask)
        return _compute_G


    return 3*_lambda**2 - 4*utilde*_lambda + utilde**2 - ctilde**2

def _get_approx_lambda(_lambda, atilde, utilde, ctilde):
    theta = _get_theta(_lambda, utilde, ctilde)

    numerator = _lambda * theta - ctilde**2 * atilde * utilde 
    denominator = theta - ctilde**2*atilde 
    denominator = jnp.where(jnp.abs(denominator) < jnp.abs(ctilde**2*atilde)*0.08, jnp.sign(denominator)*jnp.abs(ctilde**2*atilde)*0.08, denominator)  

    lambda_ = numerator / denominator

    # Dry-state fix
    lambda_ = jnp.where(atilde <= 1e-6, 0.0, lambda_)

    return lambda_

@jax.jit
def _get_lambda(utilde, atilde, ctilde):
    """
    Solves cubic polynomial 
    R_𝜆=-𝜆[(𝑢 - ̃𝜆)^2 - ̃𝑐^2] + ̃𝑐^2 ̃𝑎 (𝜆− ̃𝑢) = 0

    using Cardano-Vieta formulas
    """

    a = -1 
    b = -2*utilde
    c = (1+atilde)*ctilde**2-utilde**2
    d = -ctilde**2*atilde*utilde

    p = (3*a*c - b**2)/(3*a**2)
    q = (2*b**3 - 9*a*b*c + 27*a**2*d)/(27*a**3)

    #delta = q**2 + 4*p**3 / 27

    # get the solutions 

    # Trigonometric solution
    cos_theta = jnp.clip((3*q)/(2*p) * jnp.sqrt(-3/p), -1.0, 1.0)
    theta = jnp.arccos(cos_theta)
    
    sqrt_term = 2 * jnp.sqrt(-p/3)
    shift = -b/(3*a)
    
    # The three roots from trigonometric formula:
    # t_k = 2√(-p/3) cos((θ + 2πk)/3) for k = 0, 1, 2
    #
    # cos((θ + 0)/3)     → always in [0, π/3]     → cos is largest
    # cos((θ + 2π)/3)    → always in [2π/3, π]    → cos is smallest  
    # cos((θ + 4π)/3)    → always in [4π/3, 5π/3] → cos is middle
    
    # So we can directly compute min and max without sorting!
    lambda_max = sqrt_term * jnp.cos(theta/3) + shift                  # γ
    lambda_min = sqrt_term * jnp.cos((theta + 2*jnp.pi)/3) + shift     # α
    
    small = atilde < 1e-5
    lambda_lin_min = utilde - ctilde
    lambda_lin_max = utilde + ctilde

    lambda_max = jnp.where(small, lambda_lin_max, lambda_max)
    lambda_min = jnp.where(small, lambda_lin_min, lambda_min)

    return lambda_min, lambda_max

@jax.jit
def compute_dt(si, sj, nx, ny, dx):
    hi, hui, hvi, zi, gi = si
    hj, huj, hvj, zj, gj = sj
    
    hi = jnp.maximum(hi, DRY_TOL)
    hj = jnp.maximum(hj, DRY_TOL)

    sqrt_i = jnp.sqrt(hi)
    sqrt_j = jnp.sqrt(hj)
    denom = sqrt_i + sqrt_j

    ui = hui / hi
    vi = hvi / hi
    uj = huj / hj
    vj = hvj / hj

    uhati = ui*nx + vi*ny          # normal 
    vhati = -ui*ny + vi*nx         # tangent
    
    uhatj = uj*nx + vj*ny          # normal
    vhatj = -uj*ny + vj*nx         # tangent

    htilde = 0.5*(hi + hj)
    utilde = (uhati * sqrt_i + uhatj * sqrt_j) / denom
    vtilde = (vhati * sqrt_i + vhatj * sqrt_j) / denom 

    ctilde = jnp.sqrt(g*htilde)
    gtilde = 0.5 * (gi + gj)
    atilde = gtilde*(uhati**2 + uhati*uhatj + uhatj**2 + vhati*vhatj)/(sqrt_i*sqrt_j) 
    
    # SWE speeds (cheap)
    lambda_1 = utilde - ctilde
    lambda_3 = utilde + ctilde

    # Exner speeds (expensive)
    lam_min, lam_max = _get_lambda(utilde, atilde, ctilde)

    roe_speed = jnp.maximum(jnp.abs(lam_min), jnp.abs(lam_max))
    swe_speed = jnp.maximum(jnp.abs(lambda_1), jnp.abs(lambda_3))

    dt_exner = dx / roe_speed
    dt_roe   = dx / swe_speed

    return jnp.where(roe_speed < VEL_TOL, dt_roe, dt_exner)

@jax.jit
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

    depth_mask_x = (s1_x[0] > DRY_TOL) | (s2_x[0] > DRY_TOL)
    depth_mask_y = (s1_y[0] > DRY_TOL) | (s2_y[0] > DRY_TOL)

    active_x = depth_mask_x & ~nodata_mask_x 
    active_y = depth_mask_y & ~nodata_mask_y 

    dt_x = compute_dt(s1_x, s2_x, 1, 0, dx)
    dt_y = compute_dt(s1_y, s2_y, 0,-1, dx)

    dt_x = jnp.where(active_x, dt_x, jnp.inf).min()
    dt_y = jnp.where(active_y, dt_y, jnp.inf).min()

    dt = jnp.minimum(dt_x, dt_y)

    return dt

@jax.jit
def roe_solver(si, sj, nx: float, ny: float, dx: float):
    hi, hui, hvi, z_bi, zi, ni = si
    hj, huj, hvj, z_bj, zj, nj = sj

    hi = jnp.maximum(hi, DRY_TOL)
    hj = jnp.maximum(hj, DRY_TOL)

    # Calculate velocities, handling dry beds
    ui = jnp.where(hi > DRY_TOL, hui/hi, 0.0)
    vi = jnp.where(hi > DRY_TOL, hvi/hi, 0.0)

    uj = jnp.where(hj > DRY_TOL, huj/hj, 0.0)
    vj = jnp.where(hj > DRY_TOL, hvj/hj, 0.0)

    # --- Rotate problem to (normal, tangential) coordinate frame ---
    
    uhati = ui * nx + vi * ny   # Normal velocity component
    vhati = -ui * ny + vi * nx  # Tangential velocity component

    uhatj = uj * nx + vj * ny
    vhatj = -uj * ny + vj * nx

    sqrt_i = jnp.sqrt(hi)
    sqrt_j = jnp.sqrt(hj)
    denom_sqrt = sqrt_i + sqrt_j

    # --- Roe Averages ---
    # To handle dry/wet transitions robustly, consider the 'h' average carefully.
    # The (sqrt_i + sqrt_j) in the denominator could be zero if both are dry.
    # Add an epsilon to the denominator or use a more robust average for dry states.
    
    htilde = 0.5*(hi + hj)
    ctilde = jnp.sqrt(g*htilde)

    utilde = (uhati * sqrt_i + uhatj * sqrt_j) / denom_sqrt
    vtilde = (vhati * sqrt_i + vhatj * sqrt_j) / denom_sqrt 

    # --- Calculate Wave Speeds (Eigenvalues) ---
    lambda_1_roe = utilde - ctilde
    lambda_2 = utilde
    lambda_3_roe = utilde + ctilde

    # Entropy correction Harten-Hyman

    ei = uhati - jnp.sqrt(g*hi)
    ej = uhatj - jnp.sqrt(g*hj)
    mask_i = (ei < 0.0) & (ej > 0.0)

    lambda_E1 = jnp.where(mask_i, lambda_1_roe - ei*(ej-lambda_1_roe)/(ej-ei), 0.0)
    lambda_1 = jnp.where(mask_i, ei*(ej-lambda_1_roe)/(ej-ei), lambda_1_roe)

    # lambda_3
    ei = uhati + jnp.sqrt(g*hi)
    ej = uhatj + jnp.sqrt(g*hj)

    mask_j = (ei < 0.0) & (ej > 0.0)
    
    lambda_E3 = jnp.where(mask_j, lambda_3_roe - ej*(lambda_3_roe-ei)/(ej-ei), 0.0)
    lambda_3 = jnp.where(mask_j, ej*(lambda_3_roe-ei)/((ej-ei)+VEL_TOL), lambda_3_roe)
    
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
    # denom = lambda_1 - lambda_3
    # denom = jnp.where(jnp.abs(denom) < VEL_TOL, jnp.sign(denom) * VEL_TOL, denom)
    
    # P_inv = jnp.stack([
    #     jnp.stack([-lambda_3/denom,      jnp.ones_like(utilde)/denom,     jnp.zeros_like(utilde)], axis=-1),
    #     jnp.stack([-vtilde/ctilde,       jnp.zeros_like(utilde),          jnp.ones_like(utilde)/ctilde], axis=-1),
    #     jnp.stack([lambda_1/denom,       -jnp.ones_like(utilde)/denom,    jnp.zeros_like(utilde)], axis=-1)
    # ], axis=-2)

    # Calculate Inverse of P
    #P_inv = np.linalg.inv(P)

    # --- Difference in Conservative Variables (dU = Uj - Ui) ---
    dh = hj - hi
    # Note: dhu and dhv here are differences in normal and tangential momentum, not global.
    dhu = hj * uhatj - hi * uhati
    dhv = hj * vhatj - hi * vhati

    # dU = jnp.stack([dh, dhu, dhv],axis=-1)
    # --- Wave Strengths (alphas) ---
    # alphas = P_inv * dU

    denom = -2.0*ctilde

    alpha_1 = (-lambda_3_roe*dh + dhu) / denom
    alpha_2 = (-vtilde*dh + dhv) / ctilde
    alpha_3 = (lambda_1_roe*dh - dhu) / denom

    alphas = jnp.stack([alpha_1, alpha_2, alpha_3], axis=-1)

    # --- Source Terms ---
    # Based on "http://dx.doi.org/10.1016/j.jcp.2010.02.016" 
    
    # handle sediment bed 

    z_bi = jnp.where(z_bi > SED_TOL, z_bi, SED_TOL*0.01)
    z_bj = jnp.where(z_bj > SED_TOL, z_bj, SED_TOL*0.01)

    dz = (zj+z_bj) - (zi+z_bi)
    di = hi + (zi+z_bi)
    dj = hj + (zj+z_bj)
    dd = dj - di # Difference in total water depth

    # Topographic source term (pressure gradient due to bed slope)
    # This is a specific well-balanced formulation.
    #thrust = -g * htilde * dz
    thrust_a = -g * htilde * dz

    # Alternative thrust calculation for specific dry/wet conditions
    mask1_dz = (dz >= 0) & (di < zj+z_bj)
    mask2_dz = (dz < 0) & (dj < zi+z_bi)
    
    dztilde = jnp.where(mask1_dz, hi, jnp.where(mask2_dz, hj, dz))
    
    hr = jnp.where(dz >= 0, hi, hj) # Choose upstream or downstream h for reference height
    thrust_b = -g * (hr - 0.5 * jnp.abs(dztilde)) * dztilde

    # Combined thrust using specific conditions
    # This condition aims to correctly handle dry/wet fronts and supercritical/subcritical flow.
    # The condition (dz*dd >= 0.0) is often related to avoiding oscillations when the flow passes
    # over a hump or depression, and (utilde*dz > 0.0) when flow is moving uphill.
    mask_thrust = (dz * dd >= 0.0) & (utilde * dz > 0.0)
    thrust = jnp.where(mask_thrust, jnp.maximum(thrust_a, thrust_b), thrust_b)
    
    both_dry = (hi < DRY_TOL) & (hj < DRY_TOL)
    thrust = jnp.where(both_dry, 0.0, thrust)

    # friction

    ntilde = 0.5*(ni+nj)
    sf = (ntilde**2*jnp.sqrt(utilde**2+vtilde**2)*utilde)/(jnp.maximum(DRY_TOL, htilde**(4/3)))
    tau = g*htilde*sf*dx

    #Tn = jnp.stack([
    #    jnp.zeros_like(htilde),
    #    thrust-tau,
    #    jnp.zeros_like(htilde)
    #], axis=-1)

    beta_1 = (thrust - tau) / denom
    beta_2 = jnp.zeros_like(beta_1)

    betas = jnp.stack([beta_1, beta_2, -beta_1], axis=-1)

    h_i1star = hi + alphas[...,0] - jnp.where(jnp.abs(lambdas[...,0]) > VEL_TOL, 
                                            betas[...,0]/lambdas[...,0], 0.0)
    h_j3star = hj - alphas[...,2] + jnp.where(jnp.abs(lambdas[...,2]) > VEL_TOL,
                                            betas[...,2]/lambdas[...,2], 0.0)

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

@jax.jit
def roe_solve_2D(state: RoeExnerState, dt: float, dx: float, mask: jax.Array):

    h, hu, hv, z_b, z, n = state.h, state.hu, state.hv, state.z_b, state.z, state.n_b

    s1_x = (
        h[:,:-1],
        hu[:,:-1],
        hv[:,:-1],
        z_b[:,:-1],
        z[:,:-1],
        n[:,:-1]
    )

    s2_x = (
        h[:, 1:],
        hu[:, 1:],
        hv[:, 1:],
        z_b[:, 1:],
        z[:, 1:],
        n[:, 1:]
    )

    s1_y = (
        h[:-1,:],
        hu[:-1,:],
        hv[:-1,:],
        z_b[:-1,:],
        z[:-1,:],
        n[:-1,:]
    )

    s2_y = (
        h[1:, :],
        hu[1:, :],
        hv[1:, :],
        z_b[1:, :],
        z[1:, :],
        n[1:, :]
    )

    upwP_x, upwM_x = roe_solver(s1_x, s2_x, 1,  0, dx)
    upwP_y, upwM_y = roe_solver(s1_y, s2_y, 0, -1, dx)

    # check masking per edge vs masking per cell later

    nodata_mask_x = mask[0, :, :-1] | mask[0, :, 1:]
    nodata_mask_y = mask[0, :-1, :] | mask[0, 1:, :]
    
    depth_mask_x = (s1_x[0] > DRY_TOL) | (s2_x[0] > DRY_TOL)
    depth_mask_y = (s1_y[0] > DRY_TOL) | (s2_y[0] > DRY_TOL)

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

    h_new  = state.h  - dt * fluxes[..., 0] / dx
    hu_new = state.hu - dt * fluxes[..., 1] / dx
    hv_new = state.hv - dt * fluxes[..., 2] / dx

    return state.replace(
        h=h_new,
        hu=hu_new,
        hv=hv_new,
    )
   
@jax.jit
def exner_solver(si, sj, nx, ny, dx):
    # We will be implementing the Approximately Coupled Solver (ACM) https://doi.org/10.1016/j.advwatres.2021.103931
    hi, hui, hvi, z_bi, zi, gi = si
    hj, huj, hvj, z_bj, zj, gj = sj

    hi = jnp.where(hi > DRY_TOL, hi, DRY_TOL*0.01)
    hj = jnp.where(hj > DRY_TOL, hj, DRY_TOL*0.01)

    z_bi = jnp.where(z_bi > SED_TOL, z_bi, SED_TOL*0.01)
    z_bj = jnp.where(z_bj > SED_TOL, z_bj, SED_TOL*0.01)
    
    sqrt_i = jnp.sqrt(hi)
    sqrt_j = jnp.sqrt(hj)

    # Calculate velocities, handling dry beds
    ui = jnp.where(hi > DRY_TOL, hui/hi, 0.0)
    vi = jnp.where(hi > DRY_TOL, hvi/hi, 0.0)

    uj = jnp.where(hj > DRY_TOL, huj/hj, 0.0)
    vj = jnp.where(hj > DRY_TOL, hvj/hj, 0.0)

    # --- Rotate problem to (normal, tangential) coordinate frame ---
    uhati = ui * nx + vi * ny   # Normal velocity component
    uhatj = uj * nx + vj * ny

    utilde = (uhati * sqrt_i + uhatj * sqrt_j) / (sqrt_i + sqrt_j)

    umagi = (ui**2+vi**2)
    umagj = (uj**2+vj**2)

    qb_nhati = gi*umagi*uhati
    qb_nhatj = gj*umagj*uhatj

    gtilde = 0.5*(gi+gj)
    dz = z_bj - z_bi

    dqbhat = gtilde*umagj*uhatj - gtilde*umagi*uhati

    lambda_4 = jnp.where(jnp.abs(dz) > SED_TOL, dqbhat / dz, utilde)

    corrector_i = (gtilde - gi)*umagi*uhati
    corrector_j = (gtilde - gj)*umagj*uhatj

    qbni = qb_nhati + corrector_i
    qbnj = qb_nhatj + corrector_j

    F_exner = jnp.where(lambda_4 >= 0, qbni, qbnj)

    return F_exner

@jax.jit
def exner_solve_2D(state, dt, dx, mask):
    h, hu, hv, z_b, z, G = state.h, state.hu, state.hv, state.z_b, state.z, state.G

    # 1. Define Slices for Interfaces (Same as Roe Solver)
    s1_x = (h[:,:-1], hu[:,:-1], hv[:,:-1], z_b[:,:-1], z[:,:-1], G[:,:-1])
    s2_x = (h[:, 1:], hu[:, 1:], hv[:, 1:], z_b[:, 1:], z[:, 1:], G[:, 1:])

    s1_y = (h[:-1,:], hu[:-1,:], hv[:-1,:], z_b[:-1,:], z[:-1,:], G[:-1,:])
    s2_y = (h[1:, :], hu[1:, :], hv[1:, :], z_b[1:, :], z[1:, :], G[1:, :])

    # 2. Compute Potential Fluxes
    F_x = exner_solver(s1_x, s2_x, 1, 0, dx)
    F_y = exner_solver(s1_y, s2_y, 0, -1, dx)

    # 3. INTERFACE CORRECTIONS
    # We must stop sediment flux if:
    # a) One side is a wall (mask)
    # b) The water surface (eta) is lower than the neighboring bed (obstruction)
    # c) The cell is effectively dry

    # Masking/Wall logic per interface
    nodata_mask_x = mask[0, :, :-1] | mask[0, :, 1:]
    nodata_mask_y = mask[0, :-1, :] | mask[0, 1:, :]

    # Obstruction logic (Don't allow sediment to climb a dry wall higher than the water)
    # This prevents the "leaking" effect at the edges of the domain
    
    stop_x = nodata_mask_x | ((s1_x[3] < SED_TOL) & (s2_x[3] < SED_TOL))
    stop_y = nodata_mask_y | ((s1_y[3] < SED_TOL) & (s2_y[3] < SED_TOL)) 

    # Dry bed logic: If both cells are dry, sediment flux is zero
    active_x = (s1_x[0] > DRY_TOL) | (s2_x[0] > DRY_TOL) | ~stop_x 
    active_y = (s1_y[0] > DRY_TOL) | (s2_y[0] > DRY_TOL) | ~stop_y

    # Apply corrections
    F_x = jnp.where(active_x, F_x, 0.0)
    F_y = jnp.where(active_y, F_y, 0.0)

    # 4. Final Flux Divergence and Bed Update
    fluxes = jnp.zeros((state.h.shape[0], state.h.shape[1]))

    fluxes = fluxes.at[:, 1:].add(-F_x)
    fluxes = fluxes.at[:,:-1].add(F_x)
    fluxes = fluxes.at[1:, :].add(-F_y)
    fluxes = fluxes.at[:-1,:].add(F_y)

    z_new = state.z_b - dt * fluxes / dx

    return state.replace(z_b = z_new)

def make_halo_exchange(mpi_handler):
    neighbors = mpi_handler.neighbors
    comm = mpi_handler.cart_comm

    @jax.jit
    def halo_exchange(arr):
        send_order = (
            "west",
            "north",
            "east",
            "south",
        )

        # receive from opposite side
        recv_order = (
            "east",
            "south",
            "west",
            "north",
        )

        # =====================================================
        # SEND slices
        # =====================================================

        overlap_slices_send = {

            # send west interior
            "west": (slice(1, -1), 1),

            # send north interior
            "north": (1, slice(1, -1)),

            # send east interior
            "east": (slice(1, -1), -2),

            # send south interior
            "south": (-2, slice(1, -1)),
        }

        # =====================================================
        # RECEIVE ghost slices
        # =====================================================

        overlap_slices_recv = {

            # receive from east neighbor
            "east": (slice(1, -1), -1),

            # receive from south neighbor
            "south": (-1, slice(1, -1)),

            # receive from west neighbor
            "west": (slice(1, -1), 0),

            # receive from north neighbor
            "north": (0, slice(1, -1)),
        }

        for send_dir, recv_dir in zip(send_order, recv_order):
            send_proc = neighbors[send_dir]
            recv_proc = neighbors[recv_dir]

            if send_proc is MPI.PROC_NULL and recv_proc is MPI.PROC_NULL:
                continue

            recv_idx = overlap_slices_recv[recv_dir]
            recv_arr = jnp.empty_like(arr[recv_idx])

            send_idx = overlap_slices_send[send_dir]
            send_arr = arr[send_idx]

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