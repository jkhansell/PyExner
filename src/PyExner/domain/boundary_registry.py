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

    denom = jnp.where(jnp.abs(yj - yi) < 1e-12, 1e-12, yj - yi)
    test_x = (xj - xi) * (y - yi) / denom + xi
    intersect = ((yi > y) != (yj > y)) & (x < test_x)

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
    shift = -jnp.rint(jnp.flip(normal))
    boundary_cells = jnp.argwhere(mask)
    interior_cells = (boundary_cells + shift).astype(jnp.int32)

    Ny, Nx = mask.shape
    interior_cells = jnp.clip(interior_cells, a_min=0, a_max=jnp.array([Ny - 1, Nx - 1]))

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
            boundary_values.append(jnp.array(self.boundary_specs[key]["values"]))
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

        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()

        # Instantiate handler objects
        for i, key in enumerate(self.boundary_specs.keys()):
            btype = self.flux_scheme+" "+self.boundary_specs[key]["type"]

            try:
                handler_cls = BOUNDARY_REGISTRY[btype]
                if btype == "Roe Reflective" or "Roe Transmissive" :
                    boundary_idx, interior_idx = compute_reflective_indices(masks[i], normals[i])
                    handler = handler_cls(
                        mask=masks[i],
                        normal=normals[i],
                        boundary_indices=boundary_idx,
                        interior_indices=interior_idx,
                    )
                else:
                    handler = handler_cls(
                        mask=masks[i],
                        normal=normals[i],
                        values=boundary_values
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
