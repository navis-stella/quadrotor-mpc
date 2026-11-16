"""
simulate_smpc.py — Stage 9 Closed-Loop Stochastic MPC on the Nonlinear Plant
============================================================================
Regulation to a hover target that lies BEHIND a wall, under a Gaussian
disturbance, with the stochastic MPC of ocp_config_smpc:

    OCP       z₀ = x(t),  G z_k ≤ θ_k,  H v_k ≤ η_k     (deterministic equivalent)
    applied   u  = f_hover + v₀*(x(t))
    plant     x⁺ = F(x, u, d),  d ~ N(0, Σ_d) i.i.d.     (nonlinear, RK4)

Scenario: start 1.5 m in front of the wall at rest. The target is 0.1 m
behind the wall, so the cost pushes the drone against the chance constraint
for the whole run — the stationary phase is where the constraint is tested.

A chance constraint is a statement about a probability, so one run cannot
check it: N_MC independent closed loops (seeded) give, for every sample t,
the frequency of G_j x(t) > g_j over the realizations.

Checked:
    violation frequency ≤ β_j      every row, every sample (binomial 3σ band);
                                   reported: worst sample and stationary mean
    0 ≤ f_i ≤ F_MAX                hard — the first input is deterministic
    slack = 0                      how often the soft state rows were needed
Expected (Kouvaritakis & Cannon 2016, ch. 7): with z₀ = x(t) the closed-loop
frequency stays BELOW β — the constraint of step k = 1 alone would give
exactly β, the later steps (larger σ_k) push the plan further back.

Imports:
    quadrotor_model.py      → create_disturbance_plant(), quat_normalize()
    compute_tube_params.py  → reduce_state()  (deviation coordinates)
    compute_smpc_params.py  → load_smpc_params(), backoff_factor(), row_std()
    ocp_config_smpc.py      → create_solver(), solve(), shifted_plan(), warm_start()
    plot_utils.py           → plot_smpc_chance, plot_smpc_prediction, plot_states,
                               plot_inputs, plot_3d_trajectory

Run:  python -m stage9_smpc.compute_smpc_params   (once)
      python -m stage9_smpc.simulate_smpc         → results/stage9_smpc/
"""

import os
os.environ.setdefault('OMP_NUM_THREADS', '1')    # before the solver libraries load

import time
import numpy as np

from core.quadrotor_model import (create_disturbance_plant, quat_normalize,
                                  f_hover, F_MAX, NX, NU, ND)
from stage8_tube.compute_tube_params import NE, reduce_state
from stage9_smpc.compute_smpc_params import load_smpc_params, backoff_factor, row_std
import stage9_smpc.ocp_config_smpc as smpc
from core.plot_utils import (plot_smpc_chance, plot_smpc_prediction,
                             plot_states, plot_inputs, plot_3d_trajectory)

RESULTS_DIR = os.path.join('results', 'stage9_smpc')

# ─────────────────────────────────────────────────────────────────
# Scenario
# ─────────────────────────────────────────────────────────────────
T_SIM    = 10.0
P_REF    = np.array([2.0, 1.0, 3.0])            # target hover point (behind the wall)
P0       = P_REF + np.array([-1.6, 0.6, -0.4])  # 1.5 m in front of the wall
N_MC     = 300                                  # closed-loop realizations
SEED     = 0                                    # run i uses the seed SEED + i
T_SETTLE = 4.0                                  # [s] stationary phase starts here
T_PRED   = (0.0, 1.0, 2.0, 5.0)                 # [s] predictions kept from run 0 (figure)


# ─────────────────────────────────────────────────────────────────
# Closed loop
# ─────────────────────────────────────────────────────────────────
def run_once(solver, plant, par: dict, seed: int, keep_pred: bool = False) -> dict:
    """One closed-loop realization; the disturbance sequence is fixed by the seed."""
    Ts, N = par['Ts'], int(par['N'])
    A, B, K = par['A'], par['B'], par['K']
    n_sim = int(round(T_SIM / Ts))
    rng   = np.random.default_rng(seed)
    D     = rng.standard_normal((n_sim, ND)) * par['d_sigma']
    k_pred = {int(round(t / Ts)) for t in T_PRED} if keep_pred else set()

    x0 = np.zeros(NX); x0[0:3] = P0; x0[6] = 1.0
    X   = np.zeros((n_sim + 1, NX)); X[0] = x0
    X_e = np.zeros((n_sim + 1, NE)); X_e[0] = reduce_state(x0, P_REF)
    U   = np.zeros((n_sim, NU))
    status  = np.zeros(n_sim, dtype=int)
    slack   = np.zeros(n_sim)
    T_solve = np.zeros(n_sim)
    preds   = []
    Z_pl = V_pl = None

    for k in range(n_sim):
        x_e = X_e[k]
        if Z_pl is not None:
            Z_pl, V_pl = smpc.shifted_plan(Z_pl, V_pl, A, B, K)
            smpc.warm_start(solver, Z_pl, V_pl)
        t0 = time.perf_counter()
        status[k], Z_new, V_new, slack[k] = smpc.solve(solver, x_e, N)
        T_solve[k] = time.perf_counter() - t0

        if status[k] == 0:
            Z_pl, V_pl = Z_new, V_new
            u_d = V_pl[0]
        elif Z_pl is not None:                   # shifted plan + error feedback, inside the box
            u_d = np.clip(V_pl[0] + K @ (x_e - Z_pl[0]), -f_hover, F_MAX - f_hover)
        else:
            raise RuntimeError(f'OCP failed at t = 0 (status {status[k]})')
        if k in k_pred:
            preds.append((k * Ts, Z_pl.copy()))

        U[k] = f_hover + u_d
        plant.set('x', X[k]); plant.set('u', U[k]); plant.set('p', D[k])
        plant.solve()
        X[k + 1]   = quat_normalize(plant.get('x'))
        X_e[k + 1] = reduce_state(X[k + 1], P_REF)

    return dict(X=X, X_e=X_e, U=U, D=D, status=status, slack=slack,
                T_solve=T_solve, preds=preds)


def simulate() -> dict:
    """Run 0 in full (figures) + N_MC realizations for the violation statistics."""
    par    = load_smpc_params()
    plant  = create_disturbance_plant(par['Ts'])
    solver = smpc.create_solver(par)

    runs = [run_once(solver, plant, par, SEED + i, keep_pred=(i == 0)) for i in range(N_MC)]
    return dict(par=par, Ts=par['Ts'], run0=runs[0],
                X_e=np.array([r['X_e'] for r in runs]),          # (N_MC, n+1, 12)
                U=np.array([r['U'] for r in runs]),              # (N_MC, n, 4)
                status=np.array([r['status'] for r in runs]),
                slack=np.array([r['slack'] for r in runs]),
                T_solve=np.array([r['T_solve'] for r in runs]))


def violation_frequency(res: dict):
    """(freq (n+1, n_g), Y (N_MC, n+1, n_g)):  Y = G x,  freq = share of runs with Y > g."""
    par = res['par']
    Y   = res['X_e'] @ par['G'].T
    return np.mean(Y > par['g'], axis=0), Y


def summarize(res: dict) -> None:
    par, Ts = res['par'], res['Ts']
    freq, Y = violation_frequency(res)
    t       = Ts * np.arange(freq.shape[0])
    tail    = t >= T_SETTLE
    beta    = par['beta_x']
    band    = beta + 3.0 * np.sqrt(beta * (1.0 - beta) / N_MC)   # 3σ of a frequency estimate
    names   = ['wall  px', 'tilt +qx', 'tilt −qx', 'tilt +qy', 'tilt −qy']
    U       = res['U']
    p_tail  = res['X_e'][:, tail, 0]
    n_solve = res['status'].size

    print('\n' + '═' * 70)
    print(f'  Stage 9 — Stochastic MPC, nonlinear plant, d ~ N(0, Σ_d), {N_MC} runs')
    print('═' * 70)
    print(f'    OCP status ≠ 0        : {(res["status"] != 0).sum()}/{n_solve}')
    print(f'    soft rows used        : {(res["slack"] > 1e-6).sum()}/{n_solve} solves'
          f'   (largest slack {res["slack"].max():.2e})')
    print(f'    violation frequency   : {"row":9s} {"β":>5s} {"worst sample":>13s}'
          f' {"mean t ≥ " + f"{T_SETTLE:.0f} s":>13s}   (3σ band of β: {band[0]:.3f})')
    for j, name in enumerate(names):
        ok = '✓' if freq[1:, j].max() <= band[j] else '✗'
        print(f'    {"":22s}: {name:9s} {beta[j]:5.2f} {freq[1:, j].max():13.4f}'
              f' {freq[tail, j].mean():13.4f}   {ok}')
    print(f'    motors                : [{U.min():.3f}, {U.max():.3f}] N'
          f'   box [0, {F_MAX:.3f}]   {"✓" if U.min() >= 0 and U.max() <= F_MAX else "✗"}')
    print(f'    wall distance t ≥ {T_SETTLE:.0f} s : mean {par["g"][0] - p_tail.mean():.4f} m,'
          f' std {p_tail.std():.4f} m'
          f'   (prediction: κσ_∞ = {backoff_factor(beta[0], par["method"]) * par["sig_x_inf"][0]:.4f},'
          f' σ_∞ = {par["sig_x_inf"][0]:.4f})')
    print(f'    solve time mean/max   : {1e3 * res["T_solve"].mean():.2f} / '
          f'{1e3 * res["T_solve"].max():.2f} ms   (budget {1e3 * Ts:.0f} ms)')


# ─────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    res = simulate()
    summarize(res)

    par, Ts, r0 = res['par'], res['Ts'], res['run0']
    freq, Y = violation_frequency(res)
    kappa   = backoff_factor(par['beta_x'], par['method'])
    x_ref   = np.zeros(NX); x_ref[0:3] = P_REF; x_ref[6] = 1.0

    rows = [(0, 'p_x − p_ref,x [m]  (wall, zoom)', (par['g'][0] - 0.30, par['g'][0] + 0.05)),
            (3, 'q_y [-]  (tilt, upper)', None), (4, '−q_y [-]  (tilt, lower)', None)]
    plot_smpc_chance([dict(Y=Y[:, :, j], g=par['g'][j], beta=par['beta_x'][j],
                           freq=freq[:, j], label=lab, ylim=ylim,
                           theta_inf=par['g'][j] - kappa[j] * par['sig_x_inf'][j])
                      for j, lab, ylim in rows], Ts,
                     f'Stage 9 — Stochastic MPC: {N_MC} closed-loop realizations '
                     'against the chance constraints',
                     os.path.join(RESULTS_DIR, 'chance_constraints.png'))
    plot_smpc_prediction(r0['X_e'][:, 0], [(t0, Z[:, 0]) for t0, Z in r0['preds']],
                         par['sig_x'][:, 0], par['g'][0], Ts,
                         {'gaussian': backoff_factor(par['beta_x'][0], 'gaussian'),
                          'cantelli': backoff_factor(par['beta_x'][0], 'cantelli')},
                         par['method'],
                         'Stage 9 — Stochastic MPC: predicted mean, back-off κσ_k and the wall',
                         os.path.join(RESULTS_DIR, 'prediction_backoff.png'))
    plot_states(r0['X'], Ts, x_ref,
                'Stage 9 — Stochastic MPC: states (nonlinear plant, one realization)',
                os.path.join(RESULTS_DIR, 'states.png'))
    plot_inputs(r0['U'], Ts,
                'Stage 9 — Stochastic MPC: motor thrusts (one realization)',
                os.path.join(RESULTS_DIR, 'inputs.png'))
    plot_3d_trajectory(r0['X'], x_ref,
                       'Stage 9 — Stochastic MPC: 3D trajectory (one realization)',
                       os.path.join(RESULTS_DIR, 'trajectory_3d.png'))
