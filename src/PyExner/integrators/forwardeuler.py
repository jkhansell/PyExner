# PyExner/integrators/forwardeuler.py

from PyExner.utils.constants import TIMESTEP_TOL, DRY_TOL
from PyExner.integrators.registry import IntegratorConfig, IntegratorBundle, register_integrator_bundle
from PyExner.solvers.registry import SolverBundle, SolverConfig
from PyExner.state.base import BaseState
from typing import NamedTuple, Dict, Optional

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
    next_out_time = ((simstate.time / simstate.out_freq).astype(jnp.int32) + 1) * simstate.out_freq
    next_target = jnp.minimum(next_out_time, simstate.end_time)
    return simstate.time + simstate.dt < next_target - TIMESTEP_TOL

def make_body_fn(solver_bundle, mask, solver_config):
    def body_fn(simstate):
        new_dt = solver_bundle.compute_dt_fn(simstate.state, simstate.cfl, mask, solver_config)
        new_state = solver_bundle.step_fn(simstate.state, simstate.time, new_dt, mask, solver_config)
        return SimState(
            time = simstate.time + new_dt, 
            out_freq=simstate.out_freq, 
            end_time=simstate.end_time, 
            dt = new_dt,
            state = new_state,
            cfl = simstate.cfl
        )
    return body_fn

def run_fn_forwardeuler(state: BaseState, config: IntegratorConfig, io, mesh, b_mask) -> BaseState:
    iters = 0
    time = 0.0 
    numOut = 0 
    dt = 0.0
    
    # Make masks and body functions
    mask = config.solver_bundle.mask_fn(state, config.solver_config.mpi_handler.dims, b_mask) 
    body_fn = make_body_fn(config.solver_bundle, mask, config.solver_config)
    state = config.solver_bundle.init_fn(state, mask, config.solver_config)
    
    # === CONSERVATION TRACKING INITIALIZATION ===
    dx = config.solver_config.dx
    rank = config.solver_config.mpi_handler.rank
        
    # Write initial condition
    io.write_state(state, mesh, mask[0])
    
    while time < config.end_time - TIMESTEP_TOL:
        simstate = SimState(
            time=time, 
            out_freq=config.out_freq, 
            end_time=config.end_time, 
            dt=dt,
            state=state,
            cfl=config.cfl
        )

        simstate = jax.lax.while_loop(cond_fn, body_fn, simstate)
        
        time = simstate.time
        state = simstate.state
        
        dt = config.solver_bundle.compute_dt_fn(state, config.cfl, mask, config.solver_config)

        next_out_time = (int(time / config.out_freq) + 1) * config.out_freq
        next_target = min(next_out_time, config.end_time)

        if time + dt >= next_target - TIMESTEP_TOL:
            dt = next_target - time

        state = config.solver_bundle.step_fn(state, time, dt, mask, config.solver_config)
        time += dt

        # === OUTPUT AND CONSERVATION CHECK ===
        if abs(time / config.out_freq - round(time / config.out_freq)) < TIMESTEP_TOL:
            if rank == 0:
                print(f"[Forward Euler] Iteration: {iters}  Time: {time:.6f}")
                print(f"[Forward Euler] Timestep:  {dt:.9f}")

            io.write_state(state, mesh, mask[0])
            
            if rank == 0:
                print(f"[IO Writer] File: {numOut} written. Time: {time:.6f}")
                    
            numOut += 1
        
        iters += 1
        
    return state


@register_integrator_bundle("Forward Euler")
def integrator_forwardeuler():
    return IntegratorBundle(
        name="Forward Euler",
        config=config_fn_forwardeuler,
        run_fn=run_fn_forwardeuler
    )