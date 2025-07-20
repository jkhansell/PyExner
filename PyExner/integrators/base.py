# PyExner/integrators/base.py

from abc import ABC, abstractmethod
from typing import Callable, List
from PyExner.solvers.base import BaseSolver2D

class BaseTimeIntegrator(ABC):
    def __init__(self, solver: BaseSolver2D, cfl: float, end_time: float, out_freq: float):
        self.solver = solver
        self.cfl = cfl
        self.end_time = end_time
        self.time = 0.0
        self.dt = None
        self.eps = 1e-12
        self.out_freq = out_freq

        self.callbacks: List[Callable[[BaseSolver2D], None]] = []

    def add_callback(self, fn: Callable[[BaseSolver2D], None]):
        self.callbacks.append(fn)

    @abstractmethod
    def step(self):
        pass

    @classmethod
    def get_class_name(cls):
        return cls.__name__

    def run(self):
        self.iters = 0
        while self.time < self.end_time - self.eps:
            self.step(self)
            
            if abs(self.time / self.out_freq - round(self.time / self.out_freq)) < self.eps or self.iters == 0:
                print(f"[{self.get_class_name()}] Iteration: {self.iters}  Time: {self.time:.9f}")
                print(f"[{self.get_class_name()}] Timestep:  {self.dt:.6f}")
            
            self.iters += 1
            
            for cb in self.callbacks:
                cb(self.solver)

        return self.solver.get_state()
