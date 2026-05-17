import jax
import jax.numpy as jnp
from jax import vmap
from functools import partial
import traceback

#debugging
import matplotlib.pyplot as plt
from mpi4py import MPI

BOUNDARY_REGISTRY = {}

def register_boundary(name):
    def decorator(cls):
        BOUNDARY_REGISTRY[name] = cls
        return cls
    return decorator

@jax.jit
def point_in_polygon(point, polygon):
    x, y = point
    xi, yi = polygon[:, 0], polygon[:, 1]
    xj, yj = jnp.roll(xi, 1), jnp.roll(yi, 1)

    # 1. Does the horizontal ray cross the Y-bounds of the edge?
    # Using half-open intervals (yi > y) != (yj > y) prevents double-counting vertices.
    # Crucially, if the edge is perfectly horizontal (yi == yj), this evaluates to False.
    crosses_y = (yi > y) != (yj > y)

    # 2. Safe Division
    # We only care about computing test_x if crosses_y is True.
    # If crosses_y is False, the edge might be horizontal (denom=0). 
    # We swap 0s with 1s to prevent JAX from generating silent NaNs during evaluation.
    denom = yj - yi
    safe_denom = jnp.where(crosses_y, denom, 1.0) 

    # 3. Compute X intersection
    test_x = xi + (xj - xi) * (y - yi) / safe_denom

    # 4. Check intersection with a tiny floating-point tolerance
    # We use <= instead of <, and add 1e-8 to catch points sitting exactly on the line
    intersect = crosses_y & (x <= test_x + 1e-8)

    return jnp.sum(intersect) % 2 == 1

@jax.jit
def compute_label_mask(points, polygons):
    """
    Args:
        points: jnp.ndarray of shape (N, 2)
        polygons: jnp.ndarray of shape (M, K, 2), M polygons each with K vertices
    Returns:
        masks: jnp.ndarray of shape (M, N), bool array where masks[i, j] = True
               iff points[j] is inside polygons[i]
    """
    point_mask_fn = jax.vmap(lambda pt, poly: point_in_polygon(pt, poly), in_axes=(0, None))
    polygon_mask_fn = jax.vmap(lambda poly: point_mask_fn(points, poly), in_axes=0)
    
    return polygon_mask_fn(polygons)

def compute_reflective_indices(mask: jnp.ndarray, normal: jnp.ndarray):
    # To get to interior cells, move in direction of -normal
    # We need a shift in [dy, dx]
    dx = -jnp.round(normal[0]).astype(jnp.int32)
    
    # In raster, y-index (row) usually increases DOWNWARDS. 
    # If ny=1 (upward normal), we need to move interior (down), which is +dy.
    # If ny=-1 (downward normal), we need to move interior (up), which is -dy.
    dy = jnp.round(normal[1]).astype(jnp.int32) 
    
    shift = jnp.array([dy, dx])
    
    boundary_cells = jnp.argwhere(mask)
    interior_cells = (boundary_cells + shift).astype(jnp.int32)

    Ny, Nx = mask.shape
    interior_cells = jnp.clip(interior_cells, min=0, max=jnp.array([Ny - 1, Nx - 1]))

    by, bx = boundary_cells[:, 0], boundary_cells[:, 1]
    iy, ix = interior_cells[:, 0], interior_cells[:, 1]

    return (by, bx), (iy, ix)

class BoundaryManager:
    def __init__(self, params, X, Y):
        """
        boundary_specs: dict of {boundary_name: {"type", "polygon", "values", "normal"}}
        X, Y: grid coordinates
        """
        self.X = X
        self.Y = Y
        self.flux_scheme = params["flux_scheme"]
        self.boundary_specs = params["boundaries"]
        self.boundary_handlers = []
        self.names = []
        self._initialize_boundaries()
        
    def _initialize_boundaries(self):
        points = jnp.stack([self.X.ravel(), self.Y.ravel()], axis=-1)

        polygons = []
        boundary_values = []
        normals = []

        for key in self.boundary_specs:
            self.names.append(key)
            values_numeric = jnp.array([
                jnp.nan if str(v).lower() == "nan" else float(v) 
                for v in self.boundary_specs[key]["values"]
            ], dtype=jnp.float32)

            boundary_values.append(jnp.array(values_numeric))
            normals.append(jnp.array(self.boundary_specs[key]["normal"]))
            polygons.append(jnp.array(self.boundary_specs[key]["polygon"]))

        # Pad polygons to uniform vertex count
        max_vertices = max(p.shape[0] for p in polygons)

        polygons_padded = jnp.stack([
            jnp.pad(p, ((0, max_vertices - p.shape[0]), (0, 0))) for p in polygons
        ])

        masks = compute_label_mask(points, polygons_padded).reshape(
            len(polygons), self.X.shape[0], self.X.shape[1]
        )

        self.boundary_mask = masks.any(axis=0)

        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()

        # Instantiate handler objects
        for i, key in enumerate(self.boundary_specs.keys()):
            btype = self.flux_scheme+" "+self.boundary_specs[key]["type"]
            print(f"Registered Boundary: {btype}")
            normal = normals[i]
            mask = masks[i] 

            try:
                handler_cls = BOUNDARY_REGISTRY[btype]
                      
                if "Reflective" in btype or "Transmissive" in btype or "SteepFall" in btype:
                    b_idx, i_idx = compute_reflective_indices(mask, normal)
                    handler = handler_cls(
                        mask=mask,
                        normal=normal,
                        boundary_indices=b_idx,
                        interior_indices=i_idx,
                    )

                elif ("ConstantInflux" in btype or 
                      "ConstantOutflux" in btype or 
                      "NormalFlowDepth" in btype or 
                      "Berthon" in btype
                    ):
                    b_idx, i_idx = compute_reflective_indices(mask, normal)
                    handler = handler_cls(
                        mask=mask,
                        normal=normal,
                        values=boundary_values[i],
                        boundary_indices=b_idx,
                        interior_indices=i_idx,
                    )

                else:
                    handler = handler_cls(
                        mask=mask,
                        normal=normal,
                        values=boundary_values[i]
                    )
                self.boundary_handlers.append(handler)

            except KeyError:
                print(f"[BoundaryManager] Unknown boundary '{btype}'. Available options: {list(BOUNDARY_REGISTRY.keys())}")
                traceback.print_stack()
                raise

            except Exception as e:
                print(f"[BoundaryManager] Exception occurred while creating boundary '{btype}': {e}")
                traceback.print_exc()
                raise
        
    @partial(jax.jit, static_argnums=(0,))
    def apply(self, state, time):
        for handler in self.boundary_handlers:
            state = handler.apply(state, time)
        return state

