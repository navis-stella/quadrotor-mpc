"""
simulate_dare.py — Stage 2b Closed-Loop Simulation (DARE Terminal Cost)
=======================================================================
Stage 2b:  DARE-based Terminal Cost (Relaxed QIH)

3D Quadrotor NMPC simulation with motor dynamics — the Stage 1 closed loop
(stage1_basic.simulate_basic.simulate: rotor-speed plant, measured Ω → f,
thrust commands held for one sample) with the Stage 2b solver.

Design rationale:
    Stage 2a's QIH terminal set is small; the trajectory reaches it only
    near the end of the maneuver. Stage 2b drops the terminal constraint and
    uses only P_DARE as terminal cost. This retains the infinite-horizon
    cost structure without infeasibility risk from a tight terminal set.

Imports:
    simulate_basic.py            → simulate()  (Stage 1 closed loop, solver passed in)
    ocp_config_dare.py           → create_solver()  (Stage 2b, DARE only)
    plot_utils.py                → plot_states(), plot_inputs(), plot_3d_trajectory()
"""

import os
os.environ.setdefault('OMP_NUM_THREADS', '1')    # before the solver libraries load

import numpy as np

from core.quadrotor_model            import rotor_plant_hover_state, F_MIN, F_MAX
from stage1_basic.simulate_basic     import simulate, X_REF, N_SUB
from stage2_terminal.ocp_config_dare import create_solver
from core.plot_utils                 import plot_states, plot_inputs, plot_3d_trajectory

RESULTS_DIR = os.path.join('results', 'stage2b_dare')


# ─────────────────────────────────────────────────────────────────
# Entry Point
# ─────────────────────────────────────────────────────────────────
if __name__ == '__main__':

    # Stage 1 scenario: hover at the origin → hover at X_REF
    X, U, F_act, _, Ts = simulate(rotor_plant_hover_state(), X_REF, N=20, T_horizon=1.0,
                                  T_sim=5.0, create_solver=create_solver)

    plot_states(X, Ts, X_REF,
                title='Stage 2b — State Trajectory (DARE Terminal Cost)',
                save_path=os.path.join(RESULTS_DIR, 'states.png'))
    plot_inputs(U, Ts,
                title='Stage 2b — Thrust Commands and Rotor Thrust (DARE Terminal Cost)',
                save_path=os.path.join(RESULTS_DIR, 'inputs.png'),
                F_act=F_act, Ts_act=Ts / N_SUB, f_bounds=(F_MIN, F_MAX))
    plot_3d_trajectory(X, X_REF,
                       title='Stage 2b — 3D Flight Path (DARE Terminal Cost)',
                       save_path=os.path.join(RESULTS_DIR, 'trajectory_3d.png'))
