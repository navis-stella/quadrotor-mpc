"""
compute_smpc_params.py — Stage 9 Offline Stochastic-MPC Parameters
==================================================================
Everything the stochastic MPC needs offline (Kouvaritakis & Cannon, "Model
Predictive Control: Classical, Robust and Stochastic", Springer 2016, ch. 6–7;
Farina, Giulioni & Scattolini, "Stochastic linear MPC with chance constraints
— a review", J. Process Control 2016; Hewing & Zeilinger, IEEE CDC 2018):

    1. linear model      x⁺ = A x + B u + w               (hover, ZOH, 12 states)
    2. disturbance       w = G_d d,  d ~ N(0, Σ_d) i.i.d.  → Σ_w = G_d Σ_d G_dᵀ
    3. error feedback    K, P    (DARE of the MPC cost: u = v + K e, V_f = ‖z‖²_P)
    4. error covariance  Σ_e(k+1) = A_K Σ_e(k) A_Kᵀ + Σ_w,   Σ_e(0) = 0
    5. tightening        θ_{j,k} = g_j − κ_j √(G_j Σ_e(k) G_jᵀ)         (states)
                         η_{j,k} = h_j − κ_j √(H_j K Σ_e(k) Kᵀ H_jᵀ)    (inputs)

Coordinates (the ones of Stage 8, compute_tube_params): deviation from the
target hover point,  x = [p − p_ref, v, qx qy qz, ω]  (12),  u = f − f_hover.

Why this turns the chance constraint into a linear one:
    Split x = z + e with z = E{x} and u = v + K e. Then z⁺ = A z + B v carries
    no randomness and e⁺ = A_K e + w, e(0) = 0, does not depend on x(t) or on
    the decision v — its covariance is the recursion 4, known before any
    measurement. For Gaussian w, G_j e(k) ~ N(0, G_j Σ_e(k) G_jᵀ), so

        Pr{ G_j x(k) ≤ g_j } ≥ 1 − β_j   ⇔   G_j z(k) ≤ g_j − Φ⁻¹(1 − β_j) σ_{j,k}

    exactly (κ = Φ⁻¹(1 − β), the standard-normal quantile). The
    distribution-free alternative is the Cantelli bound κ = √((1 − β)/β),
    valid for ANY zero-mean w with covariance Σ_w, at the price of a 2.6×
    larger back-off at β = 0.05 (METHOD, printed side by side).

    Σ_e(0) = 0: the first input u(t) = v(t|t) is deterministic, so the motor
    limits hold as HARD constraints in closed loop; only the predicted inputs
    k ≥ 1 are tightened.

What the model leaves out:
    The plant is the nonlinear quadrotor. Σ_w is the covariance of d pushed
    through G_d = ∂F/∂d at hover (the plant's own RK4 step) — the
    linearization error of Stage 8 is NOT in it: it is neither zero-mean nor
    independent of x. The tilt chance constraint keeps the plant where that
    error is small, and self_test / the Monte Carlo of simulate_smpc measure
    on the nonlinear plant what the linear theory promises.

Sets — the one definition (OCP, simulation and plots import these):
    X: G x ≤ g  — a wall in front of the target, p_x − p_ref,x ≤ WALL_X, and a
       tilt box |qx|, |qy| ≤ Q_TILT; each row its own β_j.
    U: H u ≤ h  — the motor box 0 ≤ f_i ≤ F_MAX.

Writes data/smpc_params.npz (SMPC_PARAMS_PATH), read by load_smpc_params().

Run:  python -m stage9_smpc.compute_smpc_params            → data/smpc_params.npz
      python -m stage9_smpc.compute_smpc_params --check    self_test of the saved parameters
"""

import os

import numpy as np
import casadi as ca
from scipy.linalg import solve_discrete_are
from scipy.stats import norm

from core.quadrotor_3d_model import f_hover, F_MAX, NU, ND
from stage8_tube.compute_tube_params import NE, NAMES, TS, linear_model, residual_function


# ─────────────────────────────────────────────────────────────────
# Disturbance  d ~ N(0, diag(D_SIGMA²)), drawn anew at every sample
# ─────────────────────────────────────────────────────────────────
D_SIGMA = np.array([0.5, 0.5, 0.5,          # [N]    d_f (world) — gusts, ≈ 5 % of the weight
                    5e-3, 5e-3, 5e-3])      # [N·m]  d_τ (body)

# ─────────────────────────────────────────────────────────────────
# Chance constraints  Pr{ G_j x ≤ g_j } ≥ 1 − β_j
# ─────────────────────────────────────────────────────────────────
WALL_X  = -0.10               # [m] wall at p_ref,x + WALL_X: the target lies BEHIND it,
                              #     so the constraint is active in steady state
Q_TILT  = 0.15                # [-] |qx|, |qy|  (≈ 17° roll / pitch)
BETA_X  = 0.05                # allowed violation probability, wall
BETA_Q  = 0.05                # allowed violation probability, each tilt row
BETA_U  = 0.05                # allowed violation probability, each motor limit (k ≥ 1)

METHOD  = 'gaussian'          # 'gaussian' (Φ⁻¹) | 'cantelli' (distribution-free)
N_HORIZON = 40                # 2 s — the tightening is computed per prediction step

# ─────────────────────────────────────────────────────────────────
# Cost  (Bryson's rule: one unit of cost per scale)
# ─────────────────────────────────────────────────────────────────
Q_SCALE = np.array([0.5] * 3 +              # [m]     position
                   [1.0] * 3 +              # [m/s]   velocity
                   [0.1] * 3 +              # [-]     qx, qy, qz
                   [1.0] * 3)               # [rad/s] body rates
R_SCALE = 1.0                               # [N]     motor thrust deviation

SMPC_PARAMS_PATH = os.path.join('data', 'smpc_params.npz')


# ─────────────────────────────────────────────────────────────────
# Sets
# ─────────────────────────────────────────────────────────────────
def state_constraints():
    """
    (G (5, 12), g (5,), β (5,), names) in deviation coordinates:
    the wall row first, then ±qx, ±qy.
    """
    G = np.zeros((5, NE))
    G[0, 0] = 1.0                                       #  p_x ≤ WALL_X
    G[1, 6], G[2, 6] = 1.0, -1.0                        # ±qx ≤ Q_TILT
    G[3, 7], G[4, 7] = 1.0, -1.0                        # ±qy ≤ Q_TILT
    g    = np.array([WALL_X] + [Q_TILT] * 4)
    beta = np.array([BETA_X] + [BETA_Q] * 4)
    return G, g, beta, ['px ≤ wall', 'qx ≤', '−qx ≤', 'qy ≤', '−qy ≤']


def input_constraints():
    """(H (8, 4), h (8,), β (8,)): f_i − f_hover ≤ F_MAX − f_hover, −(f_i − f_hover) ≤ f_hover."""
    H = np.vstack([np.eye(NU), -np.eye(NU)])
    h = np.concatenate([np.full(NU, F_MAX - f_hover), np.full(NU, f_hover)])
    return H, h, np.full(2 * NU, BETA_U)


# ─────────────────────────────────────────────────────────────────
# 2. Disturbance covariance
# ─────────────────────────────────────────────────────────────────
def disturbance_gain(A: np.ndarray, B: np.ndarray, Ts: float = TS) -> np.ndarray:
    """
    G_d = ∂x⁺/∂d at hover (12 × 6) — Jacobian of the plant's RK4 step
    (compute_tube_params.residual_function), so w = G_d d is the first-order
    effect of d on the very integrator the simulation uses.
    """
    w_fun = residual_function(A, B, Ts)
    d     = ca.SX.sym('d', ND)
    J     = ca.Function('J', [d], [ca.jacobian(w_fun(ca.DM.zeros(NE), ca.DM.zeros(NU), d), d)])
    return np.array(J(np.zeros(ND)))


# ─────────────────────────────────────────────────────────────────
# 3.–4. Feedback gain + error covariance
# ─────────────────────────────────────────────────────────────────
def lqr(A, B, Q, R):
    """DARE (P, K) of the MPC cost: K is the error feedback, P the terminal weight."""
    P = solve_discrete_are(A, B, Q, R)
    K = -np.linalg.solve(R + B.T @ P @ B, B.T @ P @ A)
    return P, K


def error_covariance(A_K: np.ndarray, Sigma_w: np.ndarray, N: int) -> np.ndarray:
    """Σ_e(k), k = 0..N  (N+1, 12, 12):  Σ_e(0) = 0,  Σ_e(k+1) = A_K Σ_e(k) A_Kᵀ + Σ_w."""
    S = np.zeros((N + 1, NE, NE))
    for k in range(N):
        S[k + 1] = A_K @ S[k] @ A_K.T + Sigma_w
    return S


# ─────────────────────────────────────────────────────────────────
# 5. Tightening
# ─────────────────────────────────────────────────────────────────
def backoff_factor(beta, method: str = METHOD) -> np.ndarray:
    """
    κ with Pr{ y ≥ κ σ } ≤ β for a zero-mean scalar y of standard deviation σ:
        gaussian   κ = Φ⁻¹(1 − β)        exact for Gaussian w
        cantelli   κ = √((1 − β)/β)      any distribution (one-sided Chebyshev)
    """
    beta = np.asarray(beta, dtype=float)
    if method == 'gaussian':
        return norm.ppf(1.0 - beta)
    if method == 'cantelli':
        return np.sqrt((1.0 - beta) / beta)
    raise ValueError(f'unknown method {method!r}')


def row_std(C: np.ndarray, Sigma_e: np.ndarray) -> np.ndarray:
    """σ_{j,k} = √(C_j Σ_e(k) C_jᵀ)  →  (len(Σ_e), rows of C)."""
    return np.sqrt(np.maximum(np.einsum('ja,kab,jb->kj', C, Sigma_e, C), 0.0))


# ─────────────────────────────────────────────────────────────────
# Pipeline + I/O
# ─────────────────────────────────────────────────────────────────
def compute_smpc_params(N: int = N_HORIZON, method: str = METHOD, verbose: bool = True) -> dict:
    A, B    = linear_model(TS)
    G_d     = disturbance_gain(A, B)
    Sigma_w = G_d @ np.diag(D_SIGMA**2) @ G_d.T

    Q_cost, R_cost = np.diag(1.0 / Q_SCALE**2), np.eye(NU) / R_SCALE**2
    P, K    = lqr(A, B, Q_cost, R_cost)
    A_K     = A + B @ K
    Sigma_e = error_covariance(A_K, Sigma_w, N)

    G, g, beta_x, names = state_constraints()
    H, h, beta_u        = input_constraints()
    sig_x = row_std(G, Sigma_e)                       # (N+1, 5)
    sig_u = row_std(H @ K, Sigma_e[:N])               # (N, 8)
    theta = g - backoff_factor(beta_x, method) * sig_x
    eta   = h - backoff_factor(beta_u, method) * sig_u
    assert np.all(eta > 0), 'tightened motor box is empty — K too aggressive for Σ_w'
    assert np.all(theta[:, 1:] > 0), 'tightened tilt box is empty'

    # stationary covariance (k → ∞): how far the horizon is from it
    S_inf = Sigma_e[-1]
    for _ in range(5000):
        S_inf = A_K @ S_inf @ A_K.T + Sigma_w
    sig_inf = row_std(G, S_inf[None])[0]

    par = dict(Ts=TS, N=N, A=A, B=B, G_d=G_d, d_sigma=D_SIGMA, Sigma_w=Sigma_w,
               K=K, P=P, Q_cost=Q_cost, R_cost=R_cost,
               rho_K=np.max(np.abs(np.linalg.eigvals(A_K))), Sigma_e=Sigma_e,
               G=G, g=g, beta_x=beta_x, theta=theta, sig_x=sig_x, sig_x_inf=sig_inf,
               H=H, h=h, beta_u=beta_u, eta=eta, sig_u=sig_u, method=method)

    if verbose:
        k_g, k_c = backoff_factor(BETA_X, 'gaussian'), backoff_factor(BETA_X, 'cantelli')
        print('─── Stage 9 — stochastic MPC offline parameters ───────────────────')
        print(f'  d ~ N(0, σ²)       : σ_f = {D_SIGMA[0]} N, σ_τ = {D_SIGMA[3]} N·m, i.i.d. per sample')
        print(f'  w = G_d d, std     : ' + ', '.join(
            f'{NAMES[j]} {np.sqrt(Sigma_w[j, j]):.1e}' for j in (0, 3, 6, 9)))
        print(f'  error feedback K   : DARE of the MPC cost,  ρ(A + BK) = {par["rho_K"]:.4f}')
        print(f'  back-off factor κ  : gaussian {k_g:.3f}, cantelli {k_c:.3f}'
              f'   (β = {BETA_X})   → using {method}')
        print(f'  {"k":>4s} {"σ(px) [m]":>10s} {"θ wall [m]":>11s} {"σ(qx)":>8s} {"θ tilt":>8s}'
              f' {"σ(u₁) [N]":>10s} {"η upper [N]":>12s} {"η lower [N]":>12s}')
        for k in sorted({0, 1, 2, 5, 10, 20, N}):
            u_part = (f'{sig_u[k, 0]:10.4f} {eta[k, 0]:12.4f} {eta[k, NU]:12.4f}' if k < N
                      else f'{"—":>10s} {"—":>12s} {"—":>12s}')
            print(f'  {k:4d} {sig_x[k, 0]:10.4f} {theta[k, 0]:11.4f} {sig_x[k, 1]:8.4f}'
                  f' {theta[k, 1]:8.4f} ' + u_part)
        print(f'  {"∞":>4s} {sig_inf[0]:10.4f} {g[0] - k_g * sig_inf[0]:11.4f} {sig_inf[1]:8.4f}'
              f' {g[1] - k_g * sig_inf[1]:8.4f}')
        print(f'  wall g = {g[0]} m, tilt g = {g[1]};  motor box h = +{h[0]:.3f} / −{h[NU]:.3f} N')
        print('────────────────────────────────────────────────────────────────────')
    return par


def save_smpc_params(par: dict, path: str = SMPC_PARAMS_PATH) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez(path, **par)
    print(f'Saved: {path}')


def load_smpc_params(path: str = SMPC_PARAMS_PATH) -> dict:
    """Read the SMPC parameters; scalars come back as Python numbers / str."""
    if not os.path.exists(path):
        raise FileNotFoundError(f'{path} missing — run compute_smpc_params.py first')
    with np.load(path) as f:
        return {k: (f[k].item() if f[k].ndim == 0 else f[k]) for k in f.files}


# ─────────────────────────────────────────────────────────────────
# Self-test  (python -m stage9_smpc.compute_smpc_params --check)
# ─────────────────────────────────────────────────────────────────
def self_test(par: dict, n: int = 20_000, seed: int = 3) -> None:
    """
    The open-loop prediction the tightening rests on, against the NONLINEAR
    plant: n realizations of the error system started at hover (z = 0, v = 0,
    so x = e and u = f_hover + K e), d ~ N(0, Σ_d) every sample.

      1. E{e(k)} ≈ 0 and std of G_j e(k), H_j K e(k) ≈ √(· Σ_e(k) ·ᵀ)
      2. coverage  Pr{ G_j e(k) ≤ κ_j σ_{j,k} } ≈ 1 − β_j  (what θ promises)
    """
    rng   = np.random.default_rng(seed)
    A, B, K, N = par['A'], par['B'], par['K'], int(par['N'])
    G, H  = par['G'], par['H']
    w_map = residual_function(A, B, par['Ts']).map(n)
    kap_x = backoff_factor(par['beta_x'], par['method'])

    E     = np.zeros((n, NE))
    mean  = np.zeros(N + 1)
    r_x   = np.ones((N + 1, G.shape[0]))             # sample std / predicted std
    r_u   = np.ones((N, H.shape[0]))
    cov   = np.ones((N + 1, G.shape[0]))
    for k in range(N + 1):
        if k > 0:
            r_x[k] = (E @ G.T).std(axis=0) / par['sig_x'][k]
            cov[k] = np.mean(E @ G.T <= kap_x * par['sig_x'][k], axis=0)
            mean[k] = np.max(np.abs(E.mean(axis=0)) / np.sqrt(np.diag(par['Sigma_e'][k])))
        if k == N:
            break
        U = E @ K.T
        if k > 0:
            r_u[k] = (U @ H.T).std(axis=0) / par['sig_u'][k]
        D = rng.standard_normal((n, ND)) * par['d_sigma']
        E = E @ A.T + U @ B.T + np.array(w_map(E.T, U.T, D.T)).T

    # 3σ band of a sample std ratio / of a binomial coverage estimate
    tol_std = 3.0 / np.sqrt(2.0 * n) + 0.02          # + 2 %: linearization error
    tol_cov = 3.0 * np.sqrt(par['beta_x'] * (1.0 - par['beta_x']) / n) + 0.005
    ok_std  = max(np.abs(r_x - 1).max(), np.abs(r_u - 1).max()) < tol_std
    ok_cov  = np.all(np.abs(cov[1:] - (1.0 - par['beta_x'])) < tol_cov)

    print('─── self-test (nonlinear plant, open-loop error system) ───────────')
    print(f'  realizations               : {n}, {N} steps, d ~ N(0, Σ_d) i.i.d.')
    print(f'  max |E{{e_j}}| / σ_j         : {mean.max():.3f}')
    print(f'  sample std / √(GΣ_eGᵀ)     : [{r_x[1:].min():.3f}, {r_x[1:].max():.3f}]'
          f'   inputs [{r_u[1:].min():.3f}, {r_u[1:].max():.3f}]   {"✓" if ok_std else "✗"}')
    print(f'  coverage Pr{{G_j e ≤ κσ}}    : [{cov[1:].min():.4f}, {cov[1:].max():.4f}]'
          f'   target {1.0 - par["beta_x"][0]:.2f}   {"✓" if ok_cov else "✗"}'
          + ('' if par['method'] == 'gaussian' else '   (cantelli: ≥ target)'))
    if par['method'] == 'gaussian':
        assert ok_std and ok_cov, 'error covariance / coverage do not match the nonlinear plant'
    else:
        assert ok_std and np.all(cov[1:] >= 1.0 - par['beta_x'] - tol_cov)


if __name__ == '__main__':
    import sys
    if '--check' in sys.argv:
        self_test(load_smpc_params())
    else:
        save_smpc_params(compute_smpc_params())
