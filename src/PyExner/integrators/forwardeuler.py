# PyExner/integrators/forwardeuler.py

from PyExner.utils.constants import TIMESTEP_TOL

from PyExner.integrators.registry import IntegratorConfig, IntegratorBundle, register_integrator_bundle
from PyExner.solvers.registry import SolverBundle, SolverConfig
from PyExner.state.base import BaseState
from typing import NamedTuple

import jax
import jax.numpy as jnp

def config_fn_forwardeuler(cfl, end_time, out_freq, solver_bundle, solver_config):
    return IntegratorConfig(
        cfl = cfl,
        end_time = end_time, 
        out_freq = out_freq, 
        solver_config = solver_config,
        solver_bundle = solver_bundle
    )


class SimState(NamedTuple):
    time: float
    out_freq: float
    end_time: float
    dt: float
    state: BaseState
    cfl: float

def cond_fn(simstate):
    next_out_time = ((simstate.time / simstate.out_freq).astype(int) + 1) * simstate.out_freq
    next_target = jnp.minimum(next_out_time, simstate.end_time)

    return simstate.time + simstate.dt < next_target - TIMESTEP_TOL

def make_body_fn(solver_bundle, solver_config):

    def body_fn(simstate):
        mask = solver_bundle.mask_fn(simstate.state) 
        new_dt = solver_bundle.compute_dt_fn(simstate.state, simstate.cfl, mask, solver_config)

        new_state = solver_bundle.step_fn(simstate.state, simstate.time, new_dt, solver_config)

        return SimState(
            time = simstate.time + new_dt, 
            out_freq=simstate.out_freq, 
            end_time=simstate.end_time, 
            dt = new_dt,
            state = new_state,
            cfl = simstate.cfl
        )

    return body_fn

def run_fn_forwardeuler(state: BaseState, config: IntegratorConfig, io, mesh) -> BaseState:
    iters = 0
    time = 0.0 
    numOut = 0 
    mask = config.solver_bundle.mask_fn(state) 
    dt = 0.0
    body_fn = make_body_fn(config.solver_bundle, config.solver_config)
    io.write_state(state, mesh)
    
    while time < config.end_time - TIMESTEP_TOL:
        simstate = SimState(
            time=time, 
            out_freq=config.out_freq, 
            end_time=config.end_time, 
            dt = dt,
            state=state,
            cfl = config.cfl
        )

        simstate = jax.lax.while_loop(cond_fn, body_fn, simstate)
        
        # Update state and time to results from lax loop
        time = simstate.time
        state = simstate.state

        dt = config.solver_bundle.compute_dt_fn(state, config.cfl, mask, config.solver_config)

        next_out_time = (int(time / config.out_freq) + 1) * config.out_freq
        next_target = min(next_out_time, config.end_time)
        
        if time + dt >= next_target - TIMESTEP_TOL:
            dt = next_target - time
        
        state = config.solver_bundle.step_fn(state, time, dt, config.solver_config)
        
        time += dt

        if abs(time / config.out_freq - round(time / config.out_freq)) < TIMESTEP_TOL:
            if config.solver_config.mpi_handler.rank == 0:
                print(f"[Forward Euler Integrator] Iteration: {iters}  Time: {time:.6f}")
                print(f"[Forward Euler Integrator] Timestep:  {dt:.6f}")

            io.write_state(state, mesh)
            print(f"[IO Writer] File: {numOut} written. Time: {time:.6f}")
            numOut += 1 
             
        iters += 1

    return state

@register_integrator_bundle("Forward Euler")
def integrator_forwardeuler():
    return IntegratorBundle(
        name="Forward Euler",
        config = config_fn_forwardeuler,
        run_fn = run_fn_forwardeuler
    )