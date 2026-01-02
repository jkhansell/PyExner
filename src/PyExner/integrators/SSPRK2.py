# PyExner/integrators/SSPRK2.py

import jax
import jax.numpy as jnp
from jax.tree_util import tree_map

from typing import NamedTuple

from PyExner.utils.constants import TIMESTEP_TOL
from PyExner.integrators.registry import IntegratorConfig, IntegratorBundle, register_integrator_bundle
from PyExner.solvers.registry import SolverBundle, SolverConfig


def config_fn_SSPRK2(cfl, end_time, out_freq, solver_bundle, solver_config):
    return IntegratorConfig(
        cfl = cfl,
        end_time = end_time, 
        out_freq = out_freq, 
        solver_config = solver_config,
        solver_bundle = solver_bundle
    )


def cond_fn(simstate):
    next_out_time = ((simstate.time / simstate.out_freq).astype(int) + 1) * simstate.out_freq
    next_target = jnp.minimum(next_out_time, simstate.end_time)    

    return simstate.time + simstate.dt < next_target - TIMESTEP_TOL

def make_body_fn(solver_bundle, solver_config):

    def body_fn(simstate):
        mask = solver_bundle.mask_fn(simstate.state) 
        new_dt = solver_bundle.compute_dt_fn(simstate.state, simstate.cfl, mask, solver_config)
        
        state_1 = solver_bundle.step_fn(simstate.state, simstate.time, new_dt, solver_config)
        state_2 = solver_bundle.step_fn(state_1, simstate.time, new_dt, solver_config)
        
        # SSPRK2 combination: u^{n+1} = 0.5 * u^n + 0.5 * u^{(2)}
        new_state = tree_map(lambda a, b: 0.5 * a + 0.5 * b, simstate.state, state_2)
        
        return SimState(
            time = simstate.time + new_dt, 
            out_freq=simstate.out_freq, 
            end_time=simstate.end_time, 
            dt = new_dt,
            state = new_state,
            cfl = simstate.cfl
        )

    return body_fn

def run_fn_SSPRK2(state: BaseState, config: IntegratorConfig) -> BaseState:
    iters = 0
    time = 0.0 
    mask = config.solver_bundle.mask_fn(state) 
    dt = 0.0
    body_fn = make_body_fn(config.solver_bundle, config.solver_config)
    
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
        dt = simstate.dt
        state = simstate.state

        next_out_time = (int(time / config.out_freq) + 1) * config.out_freq
        next_target = min(next_out_time, config.end_time)
        
        dt = config.solver_bundle.compute_dt_fn(state, config.cfl, mask, config.solver_config)

        if time + dt >= next_target - TIMESTEP_TOL:
            dt = next_target - time
        
        state = config.solver_bundle.step_fn(state, time, dt, config.solver_config)

        if config.solver_config.mpi_handler.rank == 0:
            if abs(time / config.out_freq - round(time / config.out_freq)) < TIMESTEP_TOL or iters == 0:
                print(f"[SSPRK2 Integrator] Iteration: {iters}  Time: {time:.9f}")
                print(f"[SSPRK2 Integrator] Timestep:  {dt:.9f}")

        iters += 1
        time += dt

    return state

@register_integrator_bundle("SSPRK2")
def integrator_SSPRK2():
    return IntegratorBundle(
        name="SSPRK2",
        config = config_fn_SSPRK2,
        run_fn = run_fn_SSPRK2
    )