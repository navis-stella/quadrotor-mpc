"""
simulate_basic.py — Stage 1 Closed-Loop Simulation (W_e = Q)
============================================================
Stage 1:  Simple Terminal Cost (W_e = Q)

3D Quadrotor NMPC simulation with motor dynamics: the controller sends
thrust COMMANDS, the rotors follow through the ESC speed loop.

    MPC model  [x (13); f (4)]     ḟ = (f_cmd − f)/τ_m          (ocp_config_basic)
    Plant      [x (13); Ω (4)]     Ω̇ = (√(f_cmd/k_f) − Ω)/τ_m,  f = k_f Ω²

Full state feedback: the rotor speeds Ω are measured (ESC rpm telemetry)
and handed to the MPC as thrusts f = k_f Ω².

Imports:
    quadrotor_model.py      → create_rotor_plant() (p = 0 → no disturbance),
                              rotor_plant_hover_state(), rotor_to_thrust_state(),
                              thrust_from_rotor(), quat_normalize()
    ocp_config_basic.py     → create_solver()
    plot_utils.py           → plot_states(), plot_inputs(), plot_3d_trajectory()
"""

import os
os.environ.setdefault('OMP_NUM_THREADS', '1')    # before the solver libraries load

import numpy as np

from core.quadrotor_model          import (create_rotor_plant, rotor_plant_hover_state,
                                           rotor_to_thrust_state, thrust_from_rotor,
                                           quat_normalize, F_MIN, F_MAX, NX, NXM, NU)
from stage1_basic.ocp_config_basic import create_solver
from core.plot_utils               import plot_states, plot_inputs, plot_3d_trajectory

RESULTS_DIR = os.path.join('results', 'stage1_basic')
N_SUB       = 5      # plant substeps per control sample — logs the rotor thrust every 10 ms


# ─────────────────────────────────────────────────────────────────
# Closed-Loop Simulation
# ─────────────────────────────────────────────────────────────────
def simulate(x0:        np.ndarray,
             x_ref:     np.ndarray,
             N:         int   = 20,
             T_horizon: float = 1.0,
             T_sim:     float = 5.0):
    """
    Closed-loop NMPC simulation with AcadosSimSolver as plant.

    At each timestep:
        1. Measure the plant state [x; Ω] → MPC state [x; f = k_f Ω²]
        2. Fix it as initial condition, solve OCP → command sequence
        3. Hold f_cmd*[0] for one sample (ZOH) on the rotor-speed plant
        4. Normalize quaternion to prevent drift
        5. Advance to next step (receding horizon)

    Args:
        x0:         initial plant state (17,)  [x (13); Ω (4)]
        x_ref:      reference body state (13,)
        N:          prediction horizon steps
        T_horizon:  prediction horizon duration [s]
        T_sim:      total simulation time [s]

    Returns:
        X:      MPC-state trajectory [x; f]   (n_sim+1, 17)
        U:      thrust commands f_cmd          (n_sim,   4)
        F_act:  rotor thrust k_f Ω² every Ts/N_SUB  (n_sim·N_SUB+1, 4)
        Ts:     sample time [s]
    """
    Ts    = T_horizon / N
    n_sim = int(T_sim / Ts)

    print('Compiling acados solver — takes ~30s on first run...')
    solver = create_solver(x_ref, N, T_horizon)
    print('Solver ready.')

    print('Creating plant simulator...')
    plant = create_rotor_plant(Ts / N_SUB)          # p = 0: no disturbance in Stage 1
    print('Plant ready.\n')

    X     = np.zeros((n_sim + 1, NXM))
    U     = np.zeros((n_sim,     NU))
    F_act = np.zeros((n_sim * N_SUB + 1, NU))
    x_plant  = x0.copy()
    X[0]     = rotor_to_thrust_state(x_plant)
    F_act[0] = X[0, NX:]

    for k in range(n_sim):
        # set current state as initial condition
        solver.set(0, 'lbx', X[k])
        solver.set(0, 'ubx', X[k])

        # solve OCP
        status = solver.solve()
        if status not in [0, 2]:
            print(f'  [step {k:3d}] solver status {status} — stopping.')
            X, U, F_act = X[:k+1], U[:k], F_act[:k * N_SUB + 1]
            break

        # extract first optimal command
        U[k] = solver.get(0, 'u')

        # simulate plant one sample (N_SUB substeps, command held)
        for j in range(N_SUB):
            plant.set('x', x_plant)
            plant.set('u', U[k])
            plant.solve()
            x_plant = quat_normalize(plant.get('x'))
            F_act[k * N_SUB + j + 1] = thrust_from_rotor(x_plant[NX:])
        X[k+1] = rotor_to_thrust_state(x_plant)

        if k % 20 == 0:
            print(f'  step {k:3d}/{n_sim}  |  '
                  f'px={X[k,0]:+.3f}  py={X[k,1]:+.3f}  pz={X[k,2]:+.3f}  |  '
                  f'qw={X[k,6]:+.4f}  qx={X[k,7]:+.4f}  qy={X[k,8]:+.4f}  qz={X[k,9]:+.4f}')

    print('\nSimulation complete.')
    return X, U, F_act, Ts


# ─────────────────────────────────────────────────────────────────
# Entry Point
# ─────────────────────────────────────────────────────────────────
if __name__ == '__main__':

    # reference: hover at (1.0, 0.5, 1.5), level attitude, zero velocity
    x_ref = np.array([
        1.0, 0.5, 1.5,        # px, py, pz
        0.0, 0.0, 0.0,        # vx, vy, vz
        1.0, 0.0, 0.0, 0.0,   # qw, qx, qy, qz   (identity = level)
        0.0, 0.0, 0.0         # p, q, r
    ])

    # initial state: hovering at the origin — level, stationary, rotors at hover speed
    x0 = rotor_plant_hover_state()

    X, U, F_act, Ts = simulate(x0, x_ref, N=20, T_horizon=1.0, T_sim=5.0)

    plot_states(X, Ts, x_ref,
                title='Stage 1 — State Trajectory (Simple Terminal Cost)',
                save_path=os.path.join(RESULTS_DIR, 'states.png'))
    plot_inputs(U, Ts,
                title='Stage 1 — Thrust Commands and Rotor Thrust (Simple Terminal Cost)',
                save_path=os.path.join(RESULTS_DIR, 'inputs.png'),
                F_act=F_act, Ts_act=Ts / N_SUB, f_bounds=(F_MIN, F_MAX))
    plot_3d_trajectory(X, x_ref,
                       title='Stage 1 — 3D Flight Path (Simple Terminal Cost)',
                       save_path=os.path.join(RESULTS_DIR, 'trajectory_3d.png'))
