from .mesh import Mesh2D
from .boundary_registry import BoundaryManager
from .boundaries import (
    ReflectiveBoundary, 
    TransmissiveBoundary
)

__all__ = ["Mesh2D", "BoundaryManager", "ReflectiveBoundary", "TransmissiveBoundary"]
