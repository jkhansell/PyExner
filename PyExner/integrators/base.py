# PyExner/integrators/base.py

from abc import ABC, abstractmethod
from typing import Callable, List
from PyExner.solvers.base import BaseSolver2D

class BaseTimeIntegrator(ABC):
    def __init__(self, solver: BaseSolver2D, cfl: float, end_time: float):
        self.solver = solver
        self.cfl = cfl
        self.end_time = end_time
        self.time = 0.0
        self.dt = None

        self.callbacks: List[Callable[[BaseSolver2D], None]] = []

    def add_callback(self, fn: Callable[[BaseSolver2D], None]):
        self.callbacks.append(fn)

    @abstractmethod
    def step(self):
        pass

    def run(self):
        while self.time < self.end_time:
            self.step()
            for cb in self.callbacks:
                cb(self.solver)
        return self.solver.get_state()
