"""
simulate_qih.py — Stage 2a Closed-Loop QIH-NMPC Simulation
==========================================================
Stage 2a:  Complete Quasi-Infinite Horizon NMPC

3D Quadrotor QIH-NMPC simulation with motor dynamics — the Stage 1 closed
loop (stage1_basic.simulate_basic.simulate: rotor-speed plant, measured Ω → f)
with the Stage 2a solver.

Key addition over Stage 1:
    The predicted terminal state x_N of every solve gives
    V_N = Δx_N^T P_set Δx_N, which shows when the terminal set constraint
    is satisfied (V_N <= alpha) or relaxed by its slack (V_N > alpha).

Imports:
    simulate_basic.py           → simulate()  (Stage 1 closed loop, solver passed in)
    ocp_config_qih.py           → QIH-NMPC solver + terminal_set()  (Stage 2a)
    plot_utils.py               → shared plotting functions
"""

import os
os.environ.setdefault('OMP_NUM_THREADS', '1')    # before the solver libraries load

import numpy as np

from core.quadrotor_model           import rotor_plant_hover_state, f_hover, F_MIN, F_MAX, NU
from stage1_basic.simulate_basic    import simulate, X_REF, N_SUB
from stage2_terminal.ocp_config_qih import create_solver, terminal_set
from core.plot_utils                import (plot_states, plot_inputs,
                                            plot_3d_trajectory,
                                            plot_terminal_constraint)

RESULTS_DIR = os.path.join('results', 'stage2a_qih')


# ─────────────────────────────────────────────────────────────────
# Entry Point
# ─────────────────────────────────────────────────────────────────
if __name__ == '__main__':

    # Stage 1 scenario: hover at the origin → hover at X_REF
    X, U, F_act, X_N, Ts = simulate(rotor_plant_hover_state(), X_REF, N=20, T_horizon=1.0,
                                    T_sim=5.0, create_solver=create_solver)

    # ── Terminal set — the same (cached) one the solver used ────
    P_set, alpha_qih = terminal_set()
    xf_ref = np.concatenate([X_REF, np.full(NU, f_hover)])

    # ── Plots ───────────────────────────────────────────────────
    plot_terminal_constraint(X_N, xf_ref, P_set, alpha_qih, Ts,
                             title='Stage 2a — QIH Terminal Constraint Diagnostic',
                             save_path=os.path.join(RESULTS_DIR, 'terminal-constraint_diagnostic.png'))
    plot_states(X, Ts, X_REF,
                title='Stage 2a QIH-NMPC — State Trajectory',
                save_path=os.path.join(RESULTS_DIR, 'states.png'))
    plot_inputs(U, Ts,
                title='Stage 2a QIH-NMPC — Thrust Commands and Rotor Thrust',
                save_path=os.path.join(RESULTS_DIR, 'inputs.png'),
                F_act=F_act, Ts_act=Ts / N_SUB, f_bounds=(F_MIN, F_MAX))
    plot_3d_trajectory(X, X_REF,
                       title='Stage 2a QIH-NMPC — 3D Flight Path',
                       save_path=os.path.join(RESULTS_DIR, 'trajectory_3d.png'))
