# tests/domain/test_mesh.py

import jax.numpy as jnp
from PyExner.domain.mesh import Mesh2D

def test_mesh_shape():
    mesh = Mesh2D(nx=10, ny=5, domain_size=(100.0, 50.0))
    assert mesh.shape() == (5, 10)

def test_cell_centers():
    mesh = Mesh2D(nx=4, ny=2, domain_size=(4.0, 2.0))
    X, Y = mesh.cell_centers()
    assert jnp.allclose(X[0, :], jnp.array([0.5, 1.5, 2.5, 3.5]))
