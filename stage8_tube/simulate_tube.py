"""
simulate_tube.py — Stage 8 Closed-Loop Tube MPC on the Nonlinear Plant
======================================================================
Regulation to a hover target under a bounded, time-varying disturbance,
with the tube controller of ocp_config_tube:

    nominal   z⁺ = A z + B v*(z),   z₀ = x₀           (linear model, no w)
    applied   u  = f_hover + v*(z) + K (x − z)         (ancillary feedback)
    plant     x⁺ = F(x, u, d),  d ∈ D                  (nonlinear, RK4)

Scenario: start 1.4 m from the target at rest; d piecewise constant, a
random VERTEX of D every 0.5 s (seeded) — the persistent worst-case sign
pattern, not a gentle average.

Checked (the guarantee of compute_tube_params):
    e = x − z ∈ S            exact: gauge γ(e) ≤ 1 by LP, every GAUGE_STRIDE-th
                             sample; and at every sample along the directions
                             the tightening uses (12 axes, 4 wrench rows of K)
    x ∈ X,  ν ∈ U            state box, wrench box (hence motors in (0, F_MAX))
    w ∈ W                    realized one-step model error inside the sampled box
Expected (Mayne et al. 2005): z → 0 (nominal MPC), so x → S around the target.

Imports:
    quadrotor_model.py      → create_disturbance_plant(), quat_normalize(), MIXER
    compute_tube_params.py  → load_tube_params(), reduce_state(), tube_gauge()
    ocp_config_tube.py      → create_solver(), solve_nominal(), shifted_candidate()
    plot_utils.py           → plot_tube_states/inputs/certificate, plot_states,
                               plot_inputs, plot_3d_trajectory

Run:  python -m stage8_tube.compute_tube_params   (once)
      python -m stage8_tube.simulate_tube         → results/stage8_tube/
"""

import os
os.environ.setdefault('OMP_NUM_THREADS', '1')    # before the solver libraries load

import time
import numpy as np

from core.quadrotor_model import (create_disturbance_plant, quat_normalize, MIXER,
                                  f_hover, NX, NU, ND)
from stage8_tube.compute_tube_params import NE, NAMES, load_tube_params, reduce_state, tube_gauge
import stage8_tube.ocp_config_tube as tube
from core.plot_utils import (plot_tube_states, plot_tube_inputs, plot_tube_certificate,
                             plot_states, plot_inputs, plot_3d_trajectory)

RESULTS_DIR = os.path.join('results', 'stage8_tube')

# ─────────────────────────────────────────────────────────────────
# Scenario
# ─────────────────────────────────────────────────────────────────
N       = 60                                # 3 s horizon — slow nominal (tilt ≤ 2.8°)
T_SIM   = 15.0
P_REF   = np.array([2.0, 1.0, 3.0])         # target hover point
P0      = P_REF + np.array([-1.0, 0.8, -0.5])
T_HOLD  = 0.5                               # [s] d is redrawn every T_HOLD
D_SEED  = 1
GAUGE_STRIDE = 5                            # exact e ∈ S test (LP) every 5th sample


def disturbance_sequence(n: int, Ts: float, d_bound: np.ndarray, seed: int = D_SEED):
    """(n, 6) piecewise-constant d: a random vertex of D, held for T_HOLD."""
    rng   = np.random.default_rng(seed)
    hold  = int(round(T_HOLD / Ts))
    signs = rng.choice([-1.0, 1.0], size=(n // hold + 1, ND))
    return np.repeat(signs, hold, axis=0)[:n] * d_bound


# ─────────────────────────────────────────────────────────────────
# Closed loop
# ─────────────────────────────────────────────────────────────────
def simulate() -> dict:
    par   = load_tube_params()
    Ts    = par['Ts']
    A, B, K, K_f = par['A'], par['B'], par['K'], par['K_f']
    n_sim = int(round(T_SIM / Ts))

    plant  = create_disturbance_plant(Ts)
    solver = tube.create_solver(par, N)

    x0 = np.zeros(NX); x0[0:3] = P0; x0[6] = 1.0
    D  = disturbance_sequence(n_sim, Ts, par['d_bound'])

    X    = np.zeros((n_sim + 1, NX)); X[0] = x0
    X_e  = np.zeros((n_sim + 1, NE))
    Z    = np.zeros((n_sim + 1, NE))
    U    = np.zeros((n_sim, NU))
    Nu   = np.zeros((n_sim, NU))
    Nu_n = np.zeros((n_sim, NU))
    Wr   = np.zeros((n_sim, NE))
    status   = np.zeros(n_sim, dtype=int)
    fallback = np.zeros(n_sim, dtype=bool)
    T_solve  = np.zeros(n_sim)

    x_e = reduce_state(x0, P_REF)
    z   = x_e.copy()                             # z₀ = x₀  →  e₀ = 0
    assert np.all(np.abs(z[par['idx_x']]) <= par['x_bar']), \
        'x₀ ∉ X̄ — the nominal trajectory must start inside the tightened set'
    X_e[0], Z[0] = x_e, z
    Z_pl = V_pl = None

    for k in range(n_sim):
        if Z_pl is not None:
            tube.warm_start(solver, *tube.shifted_candidate(Z_pl, V_pl, A, B, K_f))
        t0 = time.perf_counter()
        status[k], Z_new, V_new = tube.solve_nominal(solver, z, N)
        T_solve[k] = time.perf_counter() - t0

        if status[k] == 0:
            Z_pl, V_pl = Z_new, V_new
        elif Z_pl is not None:                   # keep the guarantee: shifted candidate
            Z_pl, V_pl = tube.shifted_candidate(Z_pl, V_pl, A, B, K_f)
            fallback[k] = True
        else:
            raise RuntimeError(f'nominal OCP infeasible at t = 0 (status {status[k]}) — '
                               'x₀ outside the feasible region; lengthen N')

        v   = V_pl[0]
        u_d = tube.ancillary_input(v, x_e, z, K)
        u   = f_hover + u_d

        plant.set('x', X[k]); plant.set('u', u); plant.set('p', D[k])
        plant.solve()
        X[k + 1] = quat_normalize(plant.get('x'))

        x_e_next = reduce_state(X[k + 1], P_REF)
        Wr[k]    = x_e_next - A @ x_e - B @ u_d  # realized one-step model error
        U[k], Nu[k], Nu_n[k] = u, MIXER @ u_d, MIXER @ v

        z, x_e = Z_pl[1], x_e_next
        X_e[k + 1], Z[k + 1] = x_e, z

    k_g   = np.arange(0, n_sim + 1, GAUGE_STRIDE)
    gauge = tube_gauge((X_e - Z)[k_g], par)      # exact membership of e in S
    return dict(par=par, Ts=Ts, X=X, X_e=X_e, Z=Z, U=U, Nu=Nu, Nu_nom=Nu_n, W=Wr, D=D,
                status=status, fallback=fallback, T_solve=T_solve,
                t_gauge=k_g * Ts, gauge=gauge)


def summarize(res: dict) -> None:
    par   = res['par']
    E     = res['X_e'] - res['Z']
    e_sh  = np.abs(E) / par['h_x']
    ke_sh = np.abs(E @ (MIXER @ par['K']).T) / par['h_nu']
    x_sh  = np.abs(res['X_e'][:, par['idx_x']]) / par['x_bound']
    z_sh  = np.abs(res['Z'][:, par['idx_x']]) / par['x_bar']
    u_sh  = np.abs(res['Nu']) / par['u_bound']
    w_sh  = np.abs(res['W']) / par['w_bar']
    p_err = np.linalg.norm(res['X_e'][:, 0:3], axis=1)
    t_tail = res['Ts'] * np.arange(len(p_err)) >= T_SIM - 5.0
    ok = lambda m: '✓' if m <= 1.0 + 1e-9 else '✗'

    print('\n' + '═' * 66)
    print('  Stage 8 — Tube MPC, nonlinear plant, d ∈ D (random vertices)')
    print('═' * 66)
    print(f'    OCP status ≠ 0        : {(res["status"] != 0).sum()}/{len(res["status"])}'
          f'   (shifted candidate applied {res["fallback"].sum()}×)')
    print(f'    gauge γ(e) of S (LP)  : {res["gauge"].max():.3f}  {ok(res["gauge"].max())}'
          f'   e = x − z ∈ S, exact, every {GAUGE_STRIDE}th sample')
    print(f'    max |e_j| / h_S(e_j)  : {e_sh.max():.3f}  {ok(e_sh.max())}'
          f'   (worst: {NAMES[int(np.argmax(e_sh.max(axis=0)))]})   along the 12 axes')
    print(f'    max |(MKe)_i| / h_KS  : {ke_sh.max():.3f}  {ok(ke_sh.max())}   along the 4 wrench rows')
    print(f'    max |x_j| / X_j       : {x_sh.max():.3f}  {ok(x_sh.max())}   x ∈ X')
    print(f'    max |z_j| / X̄_j       : {z_sh.max():.3f}  {ok(z_sh.max())}   z ∈ X̄')
    print(f'    max |ν_i| / U_i       : {u_sh.max():.3f}  {ok(u_sh.max())}   ν ∈ U')
    print(f'    max |w_j| / w̄_j       : {w_sh.max():.3f}  {ok(w_sh.max())}'
          f'   (worst: {NAMES[int(np.argmax(w_sh.max(axis=0)))]})   w ∈ W')
    print(f'    motors                : [{res["U"].min():.3f}, {res["U"].max():.3f}] N')
    print(f'    ‖p − p_ref‖ last 5 s  : max {p_err[t_tail].max():.3f} m   '
          f'(tube: ±{par["h_x"][0]:.3f}, ±{par["h_x"][1]:.3f}, ±{par["h_x"][2]:.3f} m)')
    print(f'    ‖z‖ at the end        : {np.linalg.norm(res["Z"][-1]):.2e}')
    print(f'    solve time mean/max   : {1e3 * res["T_solve"].mean():.2f} / '
          f'{1e3 * res["T_solve"].max():.2f} ms   (budget {1e3 * res["Ts"]:.0f} ms)')


# ─────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    res = simulate()
    summarize(res)

    par, Ts = res['par'], res['Ts']
    x_ref = np.zeros(NX); x_ref[0:3] = P_REF; x_ref[6] = 1.0
    E = res['X_e'] - res['Z']

    plot_tube_states(res['X_e'], res['Z'], par['h_x'], Ts, par['idx_x'],
                     par['x_bound'], par['x_bar'],
                     'Stage 8 — Tube MPC: state x, nominal z and tube z ⊕ S',
                     os.path.join(RESULTS_DIR, 'tube_states.png'))
    plot_tube_inputs(res['Nu'], res['Nu_nom'], par['h_nu'], Ts,
                     par['u_bound'], par['nu_bar'],
                     'Stage 8 — Tube MPC: wrench ν = M(u − f_hover), nominal M v and K·S band',
                     os.path.join(RESULTS_DIR, 'tube_inputs.png'))
    plot_tube_certificate(np.abs(E) / par['h_x'], np.abs(res['W']) / par['w_bar'],
                          res['D'] / par['d_bound'], Ts,
                          'Stage 8 — Tube MPC: the inequalities behind the guarantee',
                          os.path.join(RESULTS_DIR, 'tube_certificate.png'),
                          gauge=(res['t_gauge'], res['gauge']))
    plot_states(res['X'], Ts, x_ref,
                'Stage 8 — Tube MPC: states (nonlinear plant)',
                os.path.join(RESULTS_DIR, 'states.png'))
    plot_inputs(res['U'], Ts,
                'Stage 8 — Tube MPC: motor thrusts',
                os.path.join(RESULTS_DIR, 'inputs.png'))
    plot_3d_trajectory(res['X'], x_ref,
                       'Stage 8 — Tube MPC: 3D trajectory',
                       os.path.join(RESULTS_DIR, 'trajectory_3d.png'))
