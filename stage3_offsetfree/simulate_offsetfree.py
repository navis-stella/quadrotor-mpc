"""
simulate_offsetfree.py — Stage 3 Closed-Loop Offset-Free NMPC Simulation
========================================================================
Stage 3:  Offset-Free NMPC with EKF Disturbance Estimation (motor dynamics)

Demonstrates the complete offset-free architecture:
    1. EKF estimates augmented state  z = [x; f; d]
    2. Steady-state target calculator computes (x_s, f_s) from d̂
    3. d̂ is injected into the MPC prediction model as a parameter
    4. MPC tracks the equilibrium (x_s, f_s) instead of (x_ref, f_hover)
    5. Rotor-speed plant is simulated with the TRUE disturbance

Plant and feedback as in Stage 1: plant state [x; Ω], measured Ω handed to
the EKF and the MPC as thrusts f = k_f Ω².

Validation scenario:
    Constant disturbance from t=0:
        d_fx = +0.5 N     (wind force in x, world frame)
        d_fy = -0.3 N     (wind force in y, world frame)
        d_fz = -2.943 N   (30% mass error)
    Stage 4 (closed_loop) reuses these values for comparability.

    Phase 1 (0 ≤ t < T_ACTIVATE):
        EKF runs and converges d̂, but offset-free is OFF.
        Standard MPC with zero disturbance assumption.
        → Steady-state position offset visible.

    Phase 2 (t ≥ T_ACTIVATE):
        Offset-free activated: d̂ injected into MPC, target calculator
        updates reference to (x_s, f_s).
        → Position converges back to reference.

Imports:
    quadrotor_model.py          → create_rotor_plant(), rotor_plant_hover_state(),
                                   rotor_to_thrust_state(), thrust_from_rotor()
    ocp_config_offsetfree.py    → create_solver(), set_disturbance_param(), set_reference()
    ekf_aug.py                  → AugEKF, get_default_ekf_tuning()
    ss_target.py                → compute_ss_target(), print_ss_target()
    plot_utils.py               → shared figures (t_switch marks the activation)
"""

import os
os.environ.setdefault('OMP_NUM_THREADS', '1')    # before the solver libraries load

import numpy as np

from core.quadrotor_model import (
    create_rotor_plant, rotor_plant_hover_state, rotor_to_thrust_state,
    thrust_from_rotor, quat_normalize,
    f_hover, F_MIN, F_MAX, NX, NXM, NU, ND,
)
from stage3_offsetfree.ocp_config_offsetfree import create_solver, set_disturbance_param, set_reference
from stage3_offsetfree.ekf_aug import AugEKF, get_default_ekf_tuning
from stage1_basic.simulate_basic import N_SUB
from core.ss_target import compute_ss_target, print_ss_target
from core.plot_utils import plot_states, plot_inputs, plot_disturbance, plot_3d_trajectory

RESULTS_DIR = os.path.join('results', 'stage3_offsetfree')

# Reference: hover at (2, 1, 3), level
X_REF = np.array([
    2.0, 1.0, 3.0,            # px, py, pz
    0.0, 0.0, 0.0,            # vx, vy, vz
    1.0, 0.0, 0.0, 0.0,       # qw, qx, qy, qz
    0.0, 0.0, 0.0,            # p, q, r
])


# ─────────────────────────────────────────────────────────────────
# Disturbance Scenario
# ─────────────────────────────────────────────────────────────────
def get_disturbance(t: float) -> np.ndarray:
    """
    True disturbance at time t: constant wind + mass error from t = 0.
    This lets the standard MPC stabilize first (with residual offset),
    then the offset-free mechanism activates to eliminate the error.

    Returns:
        d_true: (6,) = [d_fx, d_fy, d_fz, d_tx, d_ty, d_tz]
    """
    d = np.zeros(ND)
    d[0] = 0.5          # d_fx =  0.5 N    wind force in x (world frame)
    d[1] = -0.3         # d_fy = -0.3 N    wind force in y (world frame)
    d[2] = -0.981 * 3   # d_fz = -2.943 N  30% mass error (Δm·g)
    return d

# Offset-free activation time
T_ACTIVATE = 5.0   # [s] offset-free MPC activates at this time (two equal 5 s phases)


# ─────────────────────────────────────────────────────────────────
# Closed-Loop Simulation
# ─────────────────────────────────────────────────────────────────
def simulate(x0:        np.ndarray,
             x_ref:     np.ndarray,
             N:         int   = 20,
             T_horizon: float = 1.0,
             T_sim:     float = 10.0):
    """
    Closed-loop offset-free NMPC simulation.

        Phase 1 (t < T_ACTIVATE):  d̂ is estimated but NOT used:
            p = 0, reference (x_ref, f_hover)       → steady-state offset
        Phase 2 (t ≥ T_ACTIVATE):  p = d̂, reference (x_s, f_s)
            from the target calculator                → offset removed

    Args:
        x0:     initial plant state (17,)  [x (13); Ω (4)]
        x_ref:  reference body state (13,)

    Returns:
        X:      MPC-state trajectory [x; f]     (n_sim+1, 17)
        U:      thrust commands f_cmd            (n_sim,   4)
        F_act:  rotor thrust every Ts/N_SUB      (n_sim·N_SUB+1, 4)
        D_hat:  estimated disturbance            (n_sim,   6)
        D_true: true disturbance                 (n_sim,   6)
        X_s:    equilibrium targets (body)       (n_sim,  13)
        Ts:     sample time [s]
    """
    Ts    = T_horizon / N
    n_sim = int(T_sim / Ts)

    print('Compiling acados solver (thrust-lag disturbance model)...')
    solver = create_solver(x_ref, N, T_horizon)
    print('Solver ready.')

    print('Creating rotor plant simulator...')
    plant = create_rotor_plant(Ts / N_SUB)
    print('Plant ready.')

    Q_x, Q_d, R_ekf = get_default_ekf_tuning()
    ekf = AugEKF(Ts=Ts, Q_x=Q_x, Q_d=Q_d, R=R_ekf, x0=rotor_to_thrust_state(x0))
    print('EKF initialized.\n')

    X      = np.zeros((n_sim + 1, NXM))
    U      = np.zeros((n_sim,     NU))
    F_act  = np.zeros((n_sim * N_SUB + 1, NU))
    D_hat  = np.zeros((n_sim,     ND))
    D_true = np.zeros((n_sim,     ND))
    X_s    = np.zeros((n_sim,     NX))
    x_plant  = x0.copy()
    X[0]     = rotor_to_thrust_state(x_plant)
    F_act[0] = X[0, NX:]
    f_s = np.full(NU, f_hover)

    for k in range(n_sim):
        t = k * Ts

        # ── 1. Measurement: [x; f = k_f Ω²] (full state) ───────
        y = X[k].copy()

        # ── 2. EKF update (always runs — builds d̂ even when inactive)
        ekf.update(y)
        d_hat = ekf.get_disturbance()
        D_hat[k] = d_hat

        # ── 3. Offset-free activation switch ───────────────────
        offset_free_active = (t >= T_ACTIVATE)
        if offset_free_active:
            x_s, f_s = compute_ss_target(x_ref, d_hat, f_min=F_MIN)
            set_disturbance_param(solver, d_hat, N)
        else:
            x_s, f_s = x_ref.copy(), np.full(NU, f_hover)
            set_disturbance_param(solver, np.zeros(ND), N)
        set_reference(solver, x_s, f_s, N)
        X_s[k] = x_s

        # ── 4. Solve MPC from the measured state ───────────────
        solver.set(0, 'lbx', X[k])
        solver.set(0, 'ubx', X[k])
        status = solver.solve()
        if status not in [0, 2]:
            print(f'  [step {k:3d}] solver status {status} — stopping.')
            X, U, F_act = X[:k+1], U[:k], F_act[:k * N_SUB + 1]
            D_hat, D_true, X_s = D_hat[:k], D_true[:k], X_s[:k]
            break
        U[k] = solver.get(0, 'u')

        # ── 5. EKF predict with the applied command ────────────
        ekf.predict(U[k])

        # ── 6. Plant step with the TRUE disturbance ────────────
        d_true = get_disturbance(t)
        D_true[k] = d_true
        for j in range(N_SUB):
            plant.set('x', x_plant)
            plant.set('u', U[k])
            plant.set('p', d_true)
            plant.solve()
            x_plant = quat_normalize(plant.get('x'))
            F_act[k * N_SUB + j + 1] = thrust_from_rotor(x_plant[NX:])
        X[k+1] = rotor_to_thrust_state(x_plant)

        if k % 20 == 0:
            mode = "OFFSET-FREE" if offset_free_active else "STANDARD"
            print(f'  step {k:3d}/{n_sim}  t={t:5.2f}s  [{mode}]  |  '
                  f'px={X[k,0]:+.3f}  pz={X[k,2]:+.3f}  |  '
                  f'd̂_fx={d_hat[0]:+.4f}  d̂_fz={d_hat[2]:+.4f}')

    if np.any(D_hat[-1] != 0):
        print_ss_target(x_ref, X_s[-1], f_s, D_hat[-1])

    print('\nSimulation complete.')
    return X, U, F_act, D_hat, D_true, X_s, Ts


# ─────────────────────────────────────────────────────────────────
# Summary Statistics
# ─────────────────────────────────────────────────────────────────
def print_summary(X, U, D_hat, D_true, Ts, x_ref, t_dist=0.0):
    """Print convergence diagnostics."""
    n = X.shape[0] - 1
    n_last = int(1.0 / Ts)

    pos_err = np.mean(np.abs(X[-n_last:, :3] - x_ref[:3]), axis=0)
    d_err   = np.mean(np.abs(D_hat[-n_last:] - D_true[-n_last:]), axis=0)

    idx_dist  = int(t_dist / Ts)
    d_fx_true = D_true[idx_dist, 0]
    if d_fx_true != 0:
        d_fx_err  = np.abs(D_hat[idx_dist:, 0] - d_fx_true)
        converged = np.where(d_fx_err < 0.05 * abs(d_fx_true))[0]
        t_conv = converged[0] * Ts if len(converged) > 0 else float('inf')
    else:
        t_conv = 0.0

    print('\n' + '='*60)
    print('STAGE 3 — OFFSET-FREE NMPC SUMMARY (motor dynamics)')
    print('='*60)
    print(f'  Simulation time:     {n*Ts:.1f} s')
    print(f'  Disturbance onset:   t = {t_dist} s')
    print(f'  Offset-free active:  t = {T_ACTIVATE} s')
    print(f'\n  Steady-state position error (last 1s):')
    for i, ax in enumerate('xyz'):
        print(f'    Δp{ax} = {pos_err[i]:.5f} m')

    i0, i1 = int((T_ACTIVATE - 1.0) / Ts), int(T_ACTIVATE / Ts)    # last second of phase 1
    pos_off = np.mean(X[i0:i1, :3] - x_ref[:3], axis=0)
    print(f'\n  Phase-1 position offset (t = {T_ACTIVATE-1:.0f}–{T_ACTIVATE:.0f} s, standard MPC):')
    for i, ax in enumerate('xyz'):
        print(f'    Δp{ax} = {pos_off[i]:+.4f} m')

    print(f'\n  Disturbance estimation error (last 1s):')
    for i, name in enumerate(['d_fx', 'd_fy', 'd_fz', 'd_τx', 'd_τy', 'd_τz']):
        print(f'    Δ{name} = {d_err[i]:.6f}')
    print(f'\n  d̂_fx 95% convergence time: {t_conv:.2f} s after onset')

    T_tot = U.sum(axis=1)
    print(f'\n  Thrust command: f_cmd in [{U.min():.3f}, {U.max():.3f}] N, '
          f'total peak {T_tot.max():.2f} N, final f_cmd {U[-1].round(4)}')
    print('='*60)


# ─────────────────────────────────────────────────────────────────
# Entry Point
# ─────────────────────────────────────────────────────────────────
if __name__ == '__main__':

    # initial state: hovering at the origin — level, rotors at hover speed
    x0 = rotor_plant_hover_state()

    X, U, F_act, D_hat, D_true, X_s, Ts = simulate(x0, X_REF, N=20, T_horizon=1.0, T_sim=10.0)

    marker = dict(t_switch=T_ACTIVATE, switch_label=f'offset-free ON (t = {T_ACTIVATE:g} s)')
    plot_states(X, Ts, X_REF, X_s=X_s, **marker,
                title='Stage 3 — Offset-Free NMPC: State Trajectory',
                save_path=os.path.join(RESULTS_DIR, 'states.png'))
    plot_inputs(U, Ts, **marker,
                title='Stage 3 — Offset-Free NMPC: Thrust Commands and Rotor Thrust',
                save_path=os.path.join(RESULTS_DIR, 'inputs.png'),
                F_act=F_act, Ts_act=Ts / N_SUB, f_bounds=(F_MIN, F_MAX))
    plot_disturbance(D_hat, D_true, Ts, **marker,
                     title='Stage 3 — EKF Disturbance Estimation',
                     save_path=os.path.join(RESULTS_DIR, 'disturbance.png'))
    plot_3d_trajectory(X, X_REF,
                       title='Stage 3 Offset-Free NMPC — 3D Flight Path',
                       save_path=os.path.join(RESULTS_DIR, 'trajectory_3d.png'))

    print_summary(X, U, D_hat, D_true, Ts, X_REF)
