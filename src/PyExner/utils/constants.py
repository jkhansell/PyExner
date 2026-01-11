# utils/tolerances.py

# General machine epsilon for float32 and float64 (can import from jax.numpy if desired)
import jax.numpy as jnp


# ----------------------- Tolerances -----------------------
# Relative and absolute tolerances for solver operations
RTOL_DEFAULT = 1e-6
ATOL_DEFAULT = 1e-9

# Tolerances for convergence checks
RTOL_CONVERGENCE = 1e-8
ATOL_CONVERGENCE = 1e-10

# Tolerances for iterative solvers (if any)
RTOL_ITERATIVE = 1e-5
ATOL_ITERATIVE = 1e-8

# Tolerance for flux or gradient thresholds
FLUX_TOL = 1e-12

# Tolerance for integration methods
TIMESTEP_TOL = 1e-16

# Tolerance for dry domains
DRY_TOL = 1e-4

# Tolerance for slow velocities
VEL_TOL = 1e-12
# ----------------------- Tolerances -----------------------
# ----------------------- Constants -----------------------
g = 9.81
# ----------------------- Constants -----------------------
