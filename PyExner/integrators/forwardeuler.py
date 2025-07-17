# PyExner/integrators/euler.py

from PyExner.integrators.base import BaseTimeIntegrator
from PyExner.integrators.registry import register_integrator

@register_integrator("Forward Euler")
class ForwardEulerIntegrator(BaseTimeIntegrator):
    """
    Forward Euler integrator inheriting from BaseTimeIntegrator.
    """

    def step(self):
        self.dt = self.solver._compute_timestep(self.cfl)
        self.solver.step(self.time, self.dt)
        self.time += self.dt
