# PyExner/integrators/euler.py

from PyExner.integrators.base import BaseTimeIntegrator
from PyExner.integrators.registry import register_integrator

@register_integrator("Forward Euler")
class ForwardEulerIntegrator(BaseTimeIntegrator):
    """
    Forward Euler integrator inheriting from BaseTimeIntegrator.
    """

    def step(self, dt, last_dt=False):  

        self.dt = self.solver._compute_timestep(self.cfl)

        next_out_time = (int(self.time / self.out_freq) + 1) * self.out_freq
        next_target = min(next_out_time, self.end_time)
        if self.time + self.dt >= next_target - self.eps:
            self.dt = next_target - self.time

        self.solver.step(self.time, self.dt)

        self.time += self.dt
