import jax
import jax.numpy as jnp
from functools import partial

import mpi4jax
from mpi4py import MPI

#local imports
from PyExner.state.roe_exner_state import RoeExnerState
from PyExner.utils.constants import g, DRY_TOL, VEL_TOL, SED_TOL

import matplotlib.pyplot as plt

def momentum_corrections(state: RoeExnerState, mask):
    h = state.h
    z = state.z + state.z_b

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

    active = (h > DRY_TOL) & ~mask[0]
    
    # Enforce u.n = 0 by killing momentum heading into walls or high dry neighbors
    hu_new = jnp.where(active & ~stop_u, state.hu, 0.0)
    hv_new = jnp.where(active & ~stop_v, state.hv, 0.0)

    return state.replace(
        hu = hu_new, 
        hv = hv_new, 
    )

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

def compute_G(h, hu, hv, seds, mask):

    h = jnp.maximum(h, DRY_TOL)

    # (ny,nx)
    u = hu / h
    v = hv / h

    # Lift hydraulics → (1,ny,nx)
    u = u[None, ...]
    v = v[None, ...]
    h = h[None, ...]

    # Sediments → (nseds,1,1)    
    frac = seds[0][:, None, None]
    d50    =  seds[1][:, None, None]
    density = seds[2][:, None, None]
    k_d = seds[3][:, None, None]
    k_e = seds[4][:, None, None]
    theta_c = seds[5][:, None, None]
    porosity = seds[6].mean() 

    np = d50**(1/6)/21.1

    s = density / 1000

    theta_p = (
        np**2 * (u**2 + v**2)
        / ((s - 1.0) * d50 * jnp.cbrt(h))
    )                            # (nseds,ny,nx)
    theta_p = jnp.maximum(theta_p, 0.0)

    delta_theta = jnp.maximum(theta_p - theta_c, 0.0)

    gamma1 = np**3 * jnp.sqrt(g) / ((s - 1.0) * jnp.sqrt(h))
    gamma2 = 8.0 * jnp.sqrt(delta_theta) / (theta_p**1.5 + 1e-5)
    eta_p = k_e*delta_theta*d50 / (s*k_d)
    gamma3 = s*k_d*eta_p/(k_e*d50)

    G = jnp.sum(frac * gamma1 * gamma2 * gamma3, axis=0)
    G = 1 / (1 -porosity) * G
    G = jnp.where(mask[0], 0.0, G)
    G = jnp.where(h[0] > DRY_TOL, G, 0.0)

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

def _get_theta(_lambda, utilde, ctilde):
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
    
    lambda_max = jnp.where(atilde < 1e-5, 0.0, lambda_max)
    lambda_min = jnp.where(atilde < 1e-5, 0.0, lambda_min)

    return lambda_min, lambda_max

def compute_dt(si, sj, nx, ny, dx):
    hi, hui, hvi, zi, gi = si
    hj, huj, hvj, zj, gj = sj
    
    hi = jnp.where(hi > DRY_TOL, hi, DRY_TOL)
    hj = jnp.where(hj > DRY_TOL, hj, DRY_TOL)
    
    sqrt_i = jnp.sqrt(hi)
    sqrt_j = jnp.sqrt(hj)

    ui = jnp.where(hi > DRY_TOL, hui/hi, 0.0)
    vi = jnp.where(hi > DRY_TOL, hvi/hi, 0.0)

    uj = jnp.where(hj > DRY_TOL, huj/hj, 0.0)
    vj = jnp.where(hj > DRY_TOL, hvj/hj, 0.0)

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

    #lambda_approx_1 = _get_approx_lambda(lambda_1, atilde, utilde, ctilde)
    #lambda_approx_4 = _get_approx_lambda(lambda_3, atilde, utilde, ctilde)

    lambda_approx_1, lambda_approx_4 = _get_lambda(utilde, atilde, ctilde)

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

    hi = jnp.where(hi > DRY_TOL, hi, DRY_TOL)
    hj = jnp.where(hj > DRY_TOL, hj, DRY_TOL)

    sqrt_i = jnp.sqrt(hi)
    sqrt_j = jnp.sqrt(hj)

    # Velocities
    ui = jnp.where(hi > DRY_TOL, hui/hi, 0.0)
    vi = jnp.where(hi > DRY_TOL, hvi/hi, 0.0)

    uj = jnp.where(hj > DRY_TOL, huj/hj, 0.0)
    vj = jnp.where(hj > DRY_TOL, hvj/hj, 0.0)

    # Rotate to normal/tangential
    uhati = ui * nx + vi * ny
    vhati = -ui * ny + vi * nx

    uhatj = uj * nx + vj * ny
    vhatj = -uj * ny + vj * nx

    # Roe averages
    htilde = 0.5 * (hi + hj)
    ctilde = jnp.sqrt(g * htilde)
    ctilde = jnp.maximum(ctilde, 1e-8)

    utilde = (uhati * sqrt_i + uhatj * sqrt_j) / (sqrt_i + sqrt_j)
    vtilde = (vhati * sqrt_i + vhatj * sqrt_j) / (sqrt_i + sqrt_j)

    # Eigenvalues
    lambda_1 = utilde - ctilde
    lambda_2 = utilde
    lambda_3 = utilde + ctilde

    lambdas = jnp.stack([lambda_1, lambda_2, lambda_3], axis=-1)

    # Eigenvectors
    P = jnp.stack([
        jnp.stack([jnp.ones_like(vtilde), jnp.zeros_like(vtilde), jnp.ones_like(vtilde)], axis=-1),
        jnp.stack([lambda_1,             jnp.zeros_like(vtilde),  lambda_3], axis=-1),
        jnp.stack([vtilde,               ctilde,                  vtilde], axis=-1)
    ], axis=-2)

    # Inverse
    denom = lambda_1 - lambda_3
    denom = jnp.where(jnp.abs(denom) < VEL_TOL, jnp.sign(denom) * VEL_TOL, denom)

    P_inv = jnp.stack([
        jnp.stack([-lambda_3/denom,  jnp.ones_like(utilde)/denom,  jnp.zeros_like(utilde)], axis=-1),
        jnp.stack([-vtilde/ctilde,   jnp.zeros_like(utilde),       jnp.ones_like(utilde)/ctilde], axis=-1),
        jnp.stack([lambda_1/denom,  -jnp.ones_like(utilde)/denom,  jnp.zeros_like(utilde)], axis=-1)
    ], axis=-2)

    # Conservative jump
    dh  = hj - hi
    dhu = hj * uhatj - hi * uhati
    dhv = hj * vhatj - hi * vhati

    dU = jnp.stack([dh, dhu, dhv], axis=-1)

    # Wave strengths (eq 59)
    alphas = jnp.einsum("...ji,...i->...j", P_inv, dU)

    # =========================
    # SOURCE TERMS (eq 60 EXACT)
    # =========================

    dzb = (z_bj+zj) - (z_bi+zi)

    # Bed slope
    S = -g * htilde * dzb

    # Friction
    ntilde = 0.5 * (ni + nj)
    vel = jnp.sqrt(utilde**2 + vtilde**2)

    Sf = (ntilde**2 * vel * utilde) / jnp.maximum(DRY_TOL, htilde**(4/3))
    T = -g * htilde * Sf

    ST = S + T

    beta1 = -ST / (2 * ctilde)
    beta2 = jnp.zeros_like(beta1)
    beta3 =  ST / (2 * ctilde)

    betas = jnp.stack([beta1, beta2, beta3], axis=-1)

    # =========================
    # FLUX ASSEMBLY
    # =========================

    upwP = jnp.zeros_like(alphas)
    upwM = jnp.zeros_like(alphas)

    def body(i, carry):
        upwP, upwM = carry

        lam = jnp.expand_dims(lambdas[..., i], -1)
        alpha = jnp.expand_dims(alphas[..., i], -1)
        beta  = jnp.expand_dims(betas[..., i], -1)

        P_i = P[..., i]

        flux = (lam * alpha - beta) * P_i

        flux_pos = jnp.where(lam > 0.0, flux, 0.0)
        flux_neg = jnp.where(lam <= 0.0, flux, 0.0)

        return upwP + flux_pos, upwM + flux_neg

    upwP, upwM = jax.lax.fori_loop(0, 3, body, (upwP, upwM))

    # Rotate back
    Tk_inv = jnp.array([[1,  0,   0],
                        [0, nx, -ny],
                        [0, ny,  nx]])

    upwP = jnp.einsum('ji,...i->...j', Tk_inv, upwP)
    upwM = jnp.einsum('ji,...i->...j', Tk_inv, upwM)

    return upwP, upwM

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

    dh  = fluxes[..., 0]
    dhu = fluxes[..., 1]
    dhv = fluxes[..., 2]

    h_new  = state.h - dt * dh / dx
    hu_new = state.hu - dt * dhu / dx
    hv_new = state.hv - dt * dhv / dx

    return state.replace(
        h=h_new,
        hu=hu_new,
        hv=hv_new,
    )
   
#@jax.jit
def exner_solver(si, sj, nx, ny, dx):
    # We will be implementing the Approximately Coupled Solver (ACM) https://doi.org/10.1016/j.advwatres.2021.103931
    hi, hui, hvi, z_bi, zi, gi = si
    hj, huj, hvj, z_bj, zj, gj = sj

    hi = jnp.where(hi > DRY_TOL, hi, DRY_TOL)
    hj = jnp.where(hj > DRY_TOL, hj, DRY_TOL)

    z_bi = jnp.where(z_bi > SED_TOL, z_bi, SED_TOL)
    z_bj = jnp.where(z_bj > SED_TOL, z_bj, SED_TOL)
    
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

#@jax.jit
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