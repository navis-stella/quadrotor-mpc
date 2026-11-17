"""
simulate_trajectory.py — Stage 6 Closed-Loop Trajectory-Tracking NMPC
=====================================================================
The quadrotor flies the helix of reference_generate.py — take-off hop, spiral-
out, constant-rate climb (R = 2 m, 5 m climb over 3 turns, 2 m/s) — with
the Stage 4b measurement and estimator:

    plant        rotor-speed plant [x; Ω], no disturbance  (D_TRUE = 0, see there)
    measurement  y = [p; q; ω] with mocap noise  (sensor_simulator; f not measured)
    estimator    certified Lyapunov MHE  ẑ = [x̂; f̂; d̂]   (state_est_mhe,
                 motor model of Stage 4b; needs data/mhe_params.npz)
    controller   trajectory-tracking NMPC on the thrust-lag model (ocp_config_trajectory)

    The MHE certificate holds for any attitude, 4·F_MIN ≤ Σf ≤ 4·F_MAX and
    |ωᵢ| ≤ OMEGA_MAX; the helix needs |ω_r| ≤ 0.55 rad/s, and the MPC's soft
    |ωᵢ| ≤ OMEGA_MPC_MOTOR = 0.85·OMEGA_MAX keeps the plant inside the envelope.

Loop (Ts = 0.05 s, MPC horizon 1 s, N = 20):

    y_k             = sensor(x_k)
    ẑ_k             = MHE(y_k, u_{k−1})   (prediction form: uses y up to k−1)
    (X_r, U_r)_k    = flatness( helix(t_k + i·Ts), d̂_k ),  i = 0..N
                      X_r = [x_r; f_r], U_r = f_r + τ_m·ḟ_r  (motor_horizon)
    u_k             = MPC(x̂_k, X_r, U_r, d̂_k)           (shifted warm start)
    x_{k+1}         = plant(x_k, u_k, d_true)           (5 substeps of Ts/5)

    d̂ (≈ 0 here, but still estimated) enters the flatness map too: the
    reference attitude and thrust are those that fly the helix UNDER the
    estimated disturbance, so the reference stays a trajectory of the
    prediction model f(·, ·, d̂).

--nominal: d_true = 0, x̂ = [x; k_f Ω²] (full state feedback), d̂ = 0 — the
tracking MPC alone, as a check of the controller without estimation.

Imports:
    reference_generate.py     → HelixReference (p, v, a, jerk, snap, yaw)
    reference_flatness.py     → motor_horizon  (→ [x_r; f_r], f_cmd,r under d̂)
    ocp_config_trajectory.py  → create_solver() + runtime helpers
    state_est_mhe.py          → MovingHorizonEstimator  (Stage 4b, motor model)
    stage4 closed_loop.py     → OMEGA_MPC_MOTOR  (Stage 4 value)

Run:  python -m stage6_trajectory.simulate_trajectory [--nominal]
      → results/stage6_trajectory/ (nominal: results/stage6_trajectory/nominal/)
"""

import os
os.environ.setdefault('OMP_NUM_THREADS', '1')    # before the solver libraries load

import argparse
import time
import numpy as np

from core.quadrotor_model                      import (create_rotor_plant, rotor_to_thrust_state,
                                                       thrust_to_rotor_state, thrust_from_rotor,
                                                       quat_normalize, project_thrust_lag_measurement,
                                                       OMEGA_MAX, F_MIN, F_MAX, NX, NXM, NU, ND, NY)
from stage6_trajectory.reference_generate      import HelixReference
from stage6_trajectory.reference_flatness      import motor_horizon
import stage6_trajectory.ocp_config_trajectory as traj
from stage4_estimation.closed_loop             import OMEGA_MPC_MOTOR
from stage1_basic.simulate_basic               import N_SUB
from core.sensor_simulator                     import SensorSimulator, MOCAP_NOISE_STD
from stage4_estimation.state_est_mhe           import MovingHorizonEstimator
from core.plot_utils                           import (plot_states, plot_inputs, plot_3d_trajectory,
                                                       plot_tracking_error, plot_disturbance,
                                                       plot_solve_times)

RESULTS_DIR = os.path.join('results', 'stage6_trajectory')

# ─────────────────────────────────────────────────────────────────
# Scenario
# ─────────────────────────────────────────────────────────────────
N         = 20
T_HORIZON = 1.0
TS        = T_HORIZON / N            # 0.05 s — plant, sensor, MHE and MPC
REF       = HelixReference()         # default helix (reference_generate.HelixParams)
T_SIM     = REF.duration             # ≈ 22.9 s: hop + ramp + 3 turns
N_SIM     = int(np.ceil(T_SIM / TS))

# No disturbance. With the certified weights the MHE needs ≈ 2 s to learn a
# constant d from its prior d = 0 (a Stage 4b property: the arrival weight on
# d is about one position sample) — the 30 % mass error of Stages 3–4 made the
# quadrotor sag 0.43 m, the wind alone drift 0.2 m during take-off. That is an
# estimator start-up effect, not part of trajectory tracking. The d̂ path
# (prediction model + flatness map) stays in place for a non-zero D_TRUE.
D_TRUE = np.array([0, 0, 0.0, 0.0, 0.0, 0.0])   # [N, N, N, N·m, N·m, N·m]

P = REF.params
T_CRUISE = P.T_hop + P.T_ramp        # constant-radius helix from here on
PHASES   = [(P.T_hop, 'spiral'), (T_CRUISE, 'cruise')]


# ─────────────────────────────────────────────────────────────────
# Closed Loop
# ─────────────────────────────────────────────────────────────────
def simulate(nominal: bool = False, sensor_seed: int = 42) -> dict:
    """
    Returns dict with X (n+1, 17) = [x; f = k_f Ω²], X_hat (n, 17), D_hat (n, 6),
    D_true (n, 6), X_r (n, 17), U_r (n, 4), U (n, 4), F_act (n·N_SUB+1, 4)
    rotor thrust every Ts/N_SUB, status (n,), T_est (n,), T_mpc (n,) [s],
    Y (n, 10) noisy measurements y_k = [p; q; ω] (None for --nominal).

    X_r[k], U_r[k] = node 0 of the reference handed to the MPC at t_k (under
    d̂_k). T_mpc covers flatness map + reference setting + solve. A failed
    solve (status ∉ {0, 2}) is counted; its first command is still applied,
    clipped to the motor limits (u_r if not finite).
    """
    X_r0, U_r0 = motor_horizon(REF, 0.0, N, TS)
    x0 = X_r0[0]                                    # at rest at the origin, rotors at hover
    x_plant = thrust_to_rotor_state(x0)             # plant state [x; Ω]
    print('Compiling acados NMPC solver (trajectory tracking, motor model)...')
    solver = traj.create_solver(N, T_HORIZON, omega_max=OMEGA_MPC_MOTOR)
    plant  = create_rotor_plant(TS / N_SUB)

    if not nominal:
        sensor = SensorSimulator(noise_std=MOCAP_NOISE_STD, seed=sensor_seed)
        y      = sensor.measure(x0)
        print('Building MHE (certificate data/mhe_params.npz)...')
        est    = MovingHorizonEstimator(Ts=TS, z0=project_thrust_lag_measurement(y))

    traj.init_guess(solver, X_r0, U_r0, N)

    X      = np.zeros((N_SIM + 1, NXM)); X[0] = x0
    X_hat  = np.zeros((N_SIM, NXM))
    D_hat  = np.zeros((N_SIM, ND))
    D_true = np.zeros((N_SIM, ND))
    X_r    = np.zeros((N_SIM, NXM))
    U_r    = np.zeros((N_SIM, NU))
    U      = np.zeros((N_SIM, NU))
    F_act  = np.zeros((N_SIM * N_SUB + 1, NU)); F_act[0] = x0[NX:]
    status = np.zeros(N_SIM, dtype=int)
    T_est  = np.full(N_SIM, np.nan)
    T_mpc  = np.zeros(N_SIM)
    Y      = None if nominal else np.zeros((N_SIM, NY))

    for k in range(N_SIM):
        t = k * TS

        # 1. State + disturbance estimate
        if nominal:
            x_hat, d_hat = X[k].copy(), np.zeros(ND)
        else:
            if k > 0:
                y = sensor.measure(X[k])
            Y[k] = y                                         # y₀ was taken above (prior)
            t0 = time.perf_counter()
            z_hat = est.step(y, U[k - 1] if k > 0 else None)
            T_est[k] = time.perf_counter() - t0
            x_hat = quat_normalize(z_hat[:NXM].copy())      # MPC wants unit q
            d_hat = z_hat[NXM:]
        X_hat[k], D_hat[k] = x_hat, d_hat

        # 2. Reference over the horizon (flatness under d̂) + NMPC
        t0 = time.perf_counter()
        Xr_h, Ur_h = motor_horizon(REF, t, N, TS, d_hat)
        if k > 0:
            traj.shift_warm_start(solver, Xr_h, Ur_h, N)
        traj.set_trajectory(solver, Xr_h, Ur_h, d_hat, N)
        traj.set_initial_state(solver, x_hat)
        status[k] = solver.solve()
        T_mpc[k]  = time.perf_counter() - t0

        u = solver.get(0, 'u')
        if not np.all(np.isfinite(u)):
            u = Ur_h[0]
        U[k] = np.clip(u, F_MIN, F_MAX)
        X_r[k], U_r[k] = Xr_h[0], Ur_h[0]

        # 3. Plant with the TRUE disturbance (N_SUB substeps, command held)
        d_true    = np.zeros(ND) if nominal else D_TRUE
        D_true[k] = d_true
        for j in range(N_SUB):
            plant.set('x', x_plant)
            plant.set('u', U[k])
            plant.set('p', d_true)
            plant.solve()
            x_plant = quat_normalize(plant.get('x'))
            F_act[k * N_SUB + j + 1] = thrust_from_rotor(x_plant[NX:])
        X[k + 1] = rotor_to_thrust_state(x_plant)

        if k % int(round(2.0 / TS)) == 0:
            print(f'  t={t:5.2f}s  |  ‖p − p_r‖ = '
                  f'{np.linalg.norm(X[k, :3] - X_r[k, :3]):.3f} m  |  '
                  f'd̂_fx = {d_hat[0]:+.3f} (true {d_true[0]:+.3f})  |  status {status[k]}')

    return dict(X=X, X_hat=X_hat, D_hat=D_hat, D_true=D_true, X_r=X_r,
                U_r=U_r, U=U, F_act=F_act, status=status, T_est=T_est, T_mpc=T_mpc, Y=Y,
                info='' if nominal else est.info())


# ─────────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────────
def tracking_errors(res: dict):
    """‖p − p_r‖ [m] and attitude error angle [°] at t_k, k = 0..n−1."""
    X, X_r = res['X'][:-1], res['X_r']
    e_p = np.linalg.norm(X[:, :3] - X_r[:, :3], axis=1)
    dot = np.abs(np.sum(X[:, 6:10] * X_r[:, 6:10], axis=1))
    e_q = np.rad2deg(2.0 * np.arccos(np.clip(dot, 0.0, 1.0)))
    return e_p, e_q


def summarize(res: dict, label: str) -> None:
    e_p, e_q = tracking_errors(res)
    t      = np.arange(len(e_p)) * TS
    cruise = t >= T_CRUISE + 2.0          # skip the first 2 s after the ramp
    fail   = ~np.isin(res['status'], [0, 2])
    U, T_mpc = res['U'], res['T_mpc'] * 1e3
    sat    = np.mean(np.any((U <= F_MIN + 1e-6) | (U >= F_MAX - 1e-6), axis=1))
    rms    = lambda e: np.sqrt(np.mean(e ** 2))

    print('\n' + '═' * 66)
    print(f'  Stage 6 — Trajectory-tracking NMPC ({label})')
    print('═' * 66)
    print(f'    failed OCP calls          : {fail.sum()}/{len(fail)}')
    print(f'    ‖p − p_r‖  RMS / max (all): {1e3 * rms(e_p):7.1f} / {1e3 * e_p.max():7.1f} mm')
    print(f'    ‖p − p_r‖  RMS / max (t ≥ {T_CRUISE + 2:.0f} s): '
          f'{1e3 * rms(e_p[cruise]):7.1f} / {1e3 * e_p[cruise].max():7.1f} mm')
    print(f'    attitude err RMS (t ≥ {T_CRUISE + 2:.0f} s): {rms(e_q[cruise]):.2f}°')
    print(f'    max |ω|                   : {np.abs(res["X"][:, 10:13]).max():.3f} rad/s'
          f'   (soft bound {OMEGA_MPC_MOTOR:.2f}, envelope {OMEGA_MAX})')
    print(f'    thrust command range      : [{U.min():.3f}, {U.max():.3f}] N'
          f'   (limits [{F_MIN:.3f}, {F_MAX:.3f}]), saturated {sat:.1%} of samples')
    f_err = np.abs(res['X'][:-1, NX:] - res['X_r'][:, NX:])
    print(f'    |f − f_r| mean (t ≥ {T_CRUISE + 2:.0f} s)   : {f_err[cruise].mean():.4f} N')
    if not np.all(np.isnan(res['T_est'])):
        f_hat_err = np.abs(res['X_hat'][:, NX:] - res['X'][:-1, NX:])[cruise].mean()
        print(f'    |f̂ − f| mean (t ≥ {T_CRUISE + 2:.0f} s)   : {f_hat_err:.4f} N   (thrust not measured)')
        d_err = np.abs(res['D_hat'][cruise] - res['D_true'][cruise]).mean(axis=0)
        print(f'    |d̂ − d| mean (t ≥ {T_CRUISE + 2:.0f} s)  : force {np.round(d_err[:3], 4)} N,'
              f' torque {np.round(d_err[3:], 5)} N·m')
        T_est = res['T_est'] * 1e3
        print(f'    MHE time mean / p99 / max : {np.nanmean(T_est):.2f} / '
              f'{np.nanpercentile(T_est, 99):.2f} / {np.nanmax(T_est):.2f} ms, '
              f'over budget in {np.sum(T_est + T_mpc > 1e3 * TS)} steps (MHE + MPC), '
              f'slowest at t = {np.nanargmax(T_est) * TS:.2f} s')
        print(f'    {res["info"]}')
    print(f'    MPC time mean / p99 / max : {T_mpc.mean():.2f} / {np.percentile(T_mpc, 99):.2f}'
          f' / {T_mpc.max():.2f} ms   (flatness + solve, budget {1e3 * TS:.0f} ms)')


# ─────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='Stage 6 trajectory-tracking NMPC')
    ap.add_argument('--nominal', action='store_true',
                    help='no disturbance, full state feedback (tracking MPC alone)')
    args = ap.parse_args()

    label = 'nominal, full state' if args.nominal else 'MHE, noisy y'
    out   = os.path.join(RESULTS_DIR, 'nominal') if args.nominal else RESULTS_DIR
    res   = simulate(nominal=args.nominal)
    summarize(res, label)

    title = 'Stage 6 — Trajectory tracking' + (' (nominal)' if args.nominal else '')
    plot_tracking_error(res['X'], res['X_r'], TS,
                        f'{title}: tracking error', os.path.join(out, 'tracking_error.png'),
                        t_marks=PHASES)
    plot_states(res['X'], TS, res['X_r'],
                f'{title}: states vs reference x_r', os.path.join(out, 'states.png'),
                Y=res['Y'])
    plot_inputs(res['U'], TS, f'{title}: thrust commands, rotor thrust (dashed: u_r)',
                os.path.join(out, 'inputs.png'), U_ref=res['U_r'],
                F_act=res['F_act'], Ts_act=TS / N_SUB, f_bounds=(F_MIN, F_MAX))
    plot_3d_trajectory(res['X'], res['X_r'],
                       f'{title}: 3D flight path', os.path.join(out, 'trajectory_3d.png'),
                       P_meas=None if res['Y'] is None else res['Y'][:, :3])
    if not args.nominal:
        plot_disturbance(res['D_hat'], res['D_true'], TS,
                         f'{title}: disturbance estimate', os.path.join(out, 'disturbance.png'))
        plot_solve_times(res['T_est'], res['T_mpc'], TS, TS, 'MHE',
                         f'{title}: computation time per call', os.path.join(out, 'solve_time.png'))
