"""
compute_qih_params.py — Stage 2a Offline QIH-NMPC Parameters
============================================================
Stage 2a:  Complete Quasi-Infinite Horizon NMPC

Computes the terminal cost P, terminal region radius alpha, stability
margin kappa, and auxiliary LQR gain K for QIH-NMPC on the motor-dynamics
model (state [x (13); f (4)], input f_cmd).

Theory (Chen & Allgöwer, 1998 — continuous-time formulation):
    1. Auxiliary controller:  kappa_f(x) = K x — any K with A + BK Hurwitz;
                              here K = -R_aux^{-1} B^T P_care from the CARE
                              with R_aux = R_AUX_SCALE · R (see below)
    2. Stability margin:     0 < kappa < -max Re(eig(A_K))
    3. Terminal cost P:      (A_K + kappa I)^T P + P (A_K + kappa I) = -Q*,
                             Q* = Q + K^T R K
    4. Terminal set:         Omega_alpha = { x : x^T P x <= alpha }
       where alpha is the largest value such that, on Omega_alpha,
         (a) the auxiliary inputs respect the input bounds, and
         (b) the terminal cost decreases at least as fast as the stage cost
             accumulates, along the NONLINEAR dynamics:
                 dV/dt = 2 x^T P f(x, Kx)  <=  -x^T Q* x          (decrease)

    With (a) and (b), V(x) = x^T P x bounds the cost of the infinite tail and
    Omega_alpha is positively invariant — the ingredients of the QIH proof.

How (b) is certified — the direct decrease condition:
    Chen & Allgöwer bound the nonlinear residual phi(x) = f(x, Kx) - A_K x by
    a Lipschitz constant, ‖phi(x)‖ <= L ‖x‖ with L <= kappa λmin(P)/‖P‖.
    That sufficient condition puts meters, rad/s and newtons into one
    Euclidean norm and divides by the condition number of P, so it is far
    more conservative than needed (it gave alpha ≈ 1e-4 on the 13-state
    model). Here the inequality the proof uses is checked itself:

        m(x) = (dV/dt + x^T Q* x) / (x^T P x)  <=  0     on Omega_alpha.

    m → -2·kappa at the origin (the kappa margin); the nonlinearity pushes it
    up away from the origin. The maximum of m over the ellipsoid is
    estimated by seeded sampling (interior + boundary) followed by a local
    maximization (SLSQP, exact CasADi gradients) from the worst samples, and
    alpha is the largest value (bisection) with max m <= 0. A local maximum
    is still a lower bound of the true maximum — an empirical certificate,
    but on the exact condition instead of a bound of a bound.

The auxiliary gain is a design choice:
    K is never applied online; it only certifies the terminal ingredients.
    The proof needs P from the Lyapunov equation with the COST's
    Q* = Q + K^T R K, not that K be the LQR gain of the cost. With the OCP's
    small command weight R the LQR gain is very aggressive and saturates the
    motors a few millimetres away from hover — then the input bounds, not the
    nonlinearity, set alpha. A gentler K (R_aux = s·R, s > 1) leaves more
    input headroom. select_aux_gain() sweeps s and picks the largest terminal
    set (ellipsoid volume ∝ alpha^{n/2} / √det P); R_AUX_SCALE holds the result.

Quaternion handling:
    qw is uncontrollable at hover — removed, all computations run on the 16
    states [p, v, qx, qy, qz, ω, f]. When a sampled state is evaluated on the
    nonlinear model, qw = √(1 − ‖q_v‖²) keeps the quaternion on the unit
    sphere. P (16×16) is embedded back into 17×17 with a small qw weight.

Self-test (check_terminal_set): the auxiliary controller is simulated on the
nonlinear model from points on the boundary of Omega_alpha; V must stay
inside alpha and decrease at least as fast as the stage cost accumulates.

Reproducibility: all sampling uses a seeded generator, so alpha is the same
on every run — the solver and the diagnostic plot see one terminal set.

Used by:  ocp_config_qih.py  (Stage 2a solver)
"""

import numpy as np
import casadi as ca
from scipy.linalg import (solve_continuous_are, solve_continuous_lyapunov,
                          inv, cholesky)
from scipy.optimize import minimize

from core.quadrotor_model import (get_thrust_lag_hover_linearization, create_thrust_lag_model,
                                  f_hover, F_MIN, F_MAX, NX, NXM, NU)


# ─────────────────────────────────────────────────────────────────
# Indices for quaternion reduction
# ─────────────────────────────────────────────────────────────────
# Full state:    [px,py,pz, vx,vy,vz, qw, qx,qy,qz, p,q,r, f1..f4]  (17)
# Reduced state: [px,py,pz, vx,vy,vz,     qx,qy,qz, p,q,r, f1..f4]  (16)
IDX_KEEP = [i for i in range(NXM) if i != 6]     # skip qw (index 6)
N_RED    = len(IDX_KEEP)                         # 16
IDX_QV   = [6, 7, 8]                             # qx, qy, qz inside the reduced state

R_AUX_SCALE = 10.0    # from select_aux_gain() (python -m stage2_terminal.compute_qih_params):
                      # largest terminal set over s ∈ {1, 10, …, 1e5}; R_aux = R_AUX_SCALE · R


# ─────────────────────────────────────────────────────────────────
# Decrease margin m(x) as a CasADi function
# ─────────────────────────────────────────────────────────────────
def _decrease_margin_function(K: np.ndarray, P: np.ndarray, Q_star: np.ndarray) -> ca.Function:
    """
    m(x_red) = (dV/dt + x^T Q* x) / (x^T P x) along the nonlinear thrust-lag
    model under the auxiliary controller f_cmd = f_hover + K x_red.

    x_red holds deviations from hover: position, velocity, q_v, ω and
    f − f_hover; qw is reconstructed on the unit sphere.
    """
    model = create_thrust_lag_model()
    f_nl  = ca.Function('f_nl', [model.x, model.u], [model.f_expl_expr])

    xr  = ca.SX.sym('x_red', N_RED)
    qv  = xr[IDX_QV]
    qw  = ca.sqrt(ca.fmax(1.0 - ca.dot(qv, qv), 0.0))
    x   = ca.vertcat(xr[0:6], qw, qv, xr[9:12], xr[12:16] + f_hover)
    u   = f_hover + ca.mtimes(ca.DM(K), xr)
    f_red = f_nl(x, u)[IDX_KEEP]

    Pd, Qd = ca.DM(P), ca.DM(Q_star)
    V    = ca.mtimes([xr.T, Pd, xr])
    Vdot = 2 * ca.mtimes([xr.T, Pd, f_red])
    m    = (Vdot + ca.mtimes([xr.T, Qd, xr])) / V
    return ca.Function('m', [xr], [m, ca.gradient(m, xr)])


def _max_margin(m_fun: ca.Function, Z: np.ndarray, L_inv_T: np.ndarray,
                P: np.ndarray, alpha: float, n_local: int = 5):
    """
    Estimated max of m over {x : x^T P x <= alpha}: the sampled maximum over
    x = √α · L^{-T} z (z in the unit ball), refined by SLSQP from the
    n_local worst samples. Returns (max m, maximizer).
    """
    X  = np.sqrt(alpha) * Z @ L_inv_T.T                    # (n, 16)
    mv = np.array(m_fun.map(X.shape[0])(X.T)[0]).ravel()
    best_val = mv.max()
    best_x   = X[np.argmax(mv)]

    obj = lambda x: (-float(m_fun(x)[0]), -np.array(m_fun(x)[1]).ravel())
    con = {'type': 'ineq', 'fun': lambda x: alpha - x @ P @ x, 'jac': lambda x: -2 * P @ x}
    for i in np.argsort(mv)[-n_local:]:
        res = minimize(obj, X[i], jac=True, method='SLSQP', constraints=[con],
                       options={'maxiter': 200, 'ftol': 1e-12})
        if res.success and res.x @ P @ res.x <= alpha * (1 + 1e-9) and -res.fun > best_val:
            best_val, best_x = -res.fun, res.x
    return best_val, best_x


# ─────────────────────────────────────────────────────────────────
# Offline Computation
# ─────────────────────────────────────────────────────────────────
def compute_qih_offline_parameters(Q:              np.ndarray,
                                   R:              np.ndarray,
                                   qw_weight:      float,
                                   r_aux_scale:    float = None,
                                   kappa_fraction: float = 0.5,
                                   n_samples:      int   = 4000,
                                   n_bisect:       int   = 30,
                                   seed:           int   = 0):
    """
    Compute all offline QIH-NMPC parameters.

    Args:
        Q:          (17×17) stage weight on [x; f]  (blkdiag(Q_body, Q_f))
        R:          (4×4)   stage weight on f_cmd
        qw_weight:  terminal weight put back on qw (not part of the theory;
                    keeps the Gauss-Newton Hessian regular)
        r_aux_scale:     R_aux = r_aux_scale · R for the auxiliary gain
                         (None → R_AUX_SCALE)
        kappa_fraction:  fraction of the stability margin used as kappa (0 < · < 1)
        n_samples:       samples of the ellipsoid per alpha candidate (half on the boundary)
        n_bisect:        bisection steps on log(alpha)
        seed:            RNG seed of the sampling

    Returns:
        P_full:  (17×17) terminal cost matrix (16×16 Lyapunov P + qw weight)
        alpha:   terminal region radius  (x^T P x <= alpha, x without qw)
        kappa:   exponential decay rate
        K_red:   (4×16)  auxiliary LQR gain on the reduced state
    """
    # ── Step 1: Continuous-time linearization at hover ──────────
    A_c, B_c = get_thrust_lag_hover_linearization()

    # ── Step 2: Remove qw → 16-state reduced system ────────────
    A_red = A_c[np.ix_(IDX_KEEP, IDX_KEEP)]      # (16×16)
    B_red = B_c[IDX_KEEP, :]                     # (16×4)
    Q_red = Q[np.ix_(IDX_KEEP, IDX_KEEP)]        # (16×16)

    # ── Step 3: CARE → auxiliary gain K ─────────────────────────
    #   CARE:  A^T P_care + P_care A - P_care B R_aux^{-1} B^T P_care + Q = 0
    #   Gain:  K = -R_aux^{-1} B^T P_care
    if r_aux_scale is None:
        r_aux_scale = R_AUX_SCALE
    R_aux  = r_aux_scale * R
    P_care = solve_continuous_are(A_red, B_red, Q_red, R_aux)
    K_red  = -inv(R_aux) @ B_red.T @ P_care       # (4×16)

    # ── Step 4: Verify closed-loop stability ────────────────────
    A_K = A_red + B_red @ K_red
    eigvals_AK   = np.linalg.eigvals(A_K)
    max_real_eig = np.max(np.real(eigvals_AK))
    assert max_real_eig < 0, f"A_K not Hurwitz! max Re(eig) = {max_real_eig:.6f}"

    # ── Step 5: Choose kappa ────────────────────────────────────
    #   0 < kappa < -max Re(eig(A_K)); larger → more decay margin per unit
    #   of V, but a larger P (smaller alpha for the same input headroom).
    stability_margin = abs(max_real_eig)
    kappa = kappa_fraction * stability_margin

    print(f'─── QIH offline parameters (thrust-lag model, 16 states, R_aux = {r_aux_scale:g}·R) ───')
    print(f'  A_K: slowest eigenvalue Re = {max_real_eig:.4f},  fastest Re = {np.min(np.real(eigvals_AK)):.1f}')
    print(f'  kappa              = {kappa:.4f}  ({kappa_fraction:.0%} of the margin {stability_margin:.4f})')

    # ── Step 6: Modified Lyapunov equation → terminal cost P ────
    #   (A_K + kappa I)^T P + P (A_K + kappa I) = -Q*
    #   scipy solves A X + X A^H = Q_rhs → pass A = (A_K + kappa I)^T
    Q_star = Q_red + K_red.T @ R @ K_red
    P_red  = solve_continuous_lyapunov((A_K + kappa * np.eye(N_RED)).T, -Q_star)
    P_red  = 0.5 * (P_red + P_red.T)
    assert np.all(np.linalg.eigvalsh(P_red) > 0), 'P_red not positive definite'

    # ── Step 7: Alpha from input constraints ────────────────────
    #   f_cmd = f_hover + K x ∈ [F_MIN, F_MAX]. The ellipsoid is symmetric, so
    #   max_{x^T P x <= α} |K_i x| = √(α K_i P^{-1} K_i^T) must fit the tighter side.
    du_bound  = min(F_MAX - f_hover, f_hover - F_MIN)
    P_red_inv = inv(P_red)
    alpha_input = min(du_bound**2 / (K_red[i] @ P_red_inv @ K_red[i]) for i in range(NU))
    print(f'  alpha (input limit)        = {alpha_input:.5g}')

    # ── Step 8: Alpha from the decrease condition ───────────────
    m_fun   = _decrease_margin_function(K_red, P_red, Q_star)
    L_inv_T = inv(cholesky(P_red, lower=True)).T           # x = L^{-T} z maps the unit ball onto x^T P x <= 1
    rng = np.random.default_rng(seed)
    Z   = rng.standard_normal((n_samples, N_RED))
    Z  /= np.linalg.norm(Z, axis=1, keepdims=True)
    Z[n_samples // 2:] *= rng.random((n_samples - n_samples // 2, 1)) ** (1.0 / N_RED)   # half interior

    m_in, _ = _max_margin(m_fun, Z, L_inv_T, P_red, alpha_input)
    if m_in <= 0.0:
        alpha, limit = alpha_input, 'input bounds'
    else:
        lo, hi = np.log(alpha_input) - 30.0, np.log(alpha_input)       # e^-30 ≈ 1e-13: surely fine
        assert _max_margin(m_fun, Z, L_inv_T, P_red, np.exp(lo))[0] <= 0.0
        for _ in range(n_bisect):
            mid = 0.5 * (lo + hi)
            if _max_margin(m_fun, Z, L_inv_T, P_red, np.exp(mid))[0] <= 0.0:
                lo = mid
            else:
                hi = mid
        alpha, limit = np.exp(lo), 'decrease condition (nonlinearity)'
    m_alpha, x_worst = _max_margin(m_fun, Z, L_inv_T, P_red, alpha)
    print(f'  max m at alpha_input       = {m_in:+.4f}   (m ≤ 0 required; m → −2κ = {-2*kappa:.4f} at 0)')
    print(f'  alpha                      = {alpha:.5g}   limited by: {limit}')
    print(f'  max m on Omega_alpha       = {m_alpha:+.2e}')

    # ── Step 9: Embed P_red (16×16) → P_full (17×17) ───────────
    P_full = np.zeros((NXM, NXM))
    P_full[np.ix_(IDX_KEEP, IDX_KEEP)] = P_red
    P_full[6, 6] = qw_weight                       # qw: small weight, not from Lyapunov

    # ── Size of Omega_alpha along single coordinates ────────────
    #   largest |x_i| on the ellipsoid: √(α (P^{-1})_ii)
    half_width = np.sqrt(alpha * np.diag(P_red_inv))
    names = ['px', 'py', 'pz', 'vx', 'vy', 'vz', 'qx', 'qy', 'qz', 'p', 'q', 'r', 'f1', 'f2', 'f3', 'f4']
    print('  Omega_alpha half-widths (one coordinate at a time):')
    print('    ' + '  '.join(f'{n}={w:.3g}' for n, w in zip(names, half_width)))
    print('─' * 60)

    return P_full, alpha, kappa, K_red


def log_volume(P_full: np.ndarray, alpha: float) -> float:
    """log of the volume of {x : x^T P x <= alpha} (16-dim), up to a constant."""
    P = P_full[np.ix_(IDX_KEEP, IDX_KEEP)]
    return 0.5 * N_RED * np.log(alpha) - 0.5 * np.linalg.slogdet(P)[1]


def select_aux_gain(Q: np.ndarray, R: np.ndarray, qw_weight: float,
                    scales=(1, 10, 100, 1e3, 1e4, 1e5)) -> float:
    """
    Sweep R_aux = s·R and return the s with the largest terminal set
    (log volume). Prints one line per s: alpha, what limits it, the
    position half-width and the volume.
    """
    rows = []
    for s_ in scales:
        P, a, k, K = compute_qih_offline_parameters(Q, R, qw_weight, r_aux_scale=s_)
        Pr = P[np.ix_(IDX_KEEP, IDX_KEEP)]
        rows.append((s_, a, k, np.sqrt(a * inv(Pr)[0, 0]), log_volume(P, a)))
    print('─── Auxiliary gain sweep: R_aux = s · R ─────────────────────')
    print('       s |    alpha   |  kappa |  px half-width | log volume')
    for s_, a, k, w, v in rows:
        print(f'  {s_:6g} | {a:10.4g} | {k:6.3f} | {1e3*w:10.2f} mm | {v:10.2f}')
    best = max(rows, key=lambda r: r[4])[0]
    print(f'  → largest terminal set at s = {best:g}')
    return best


# ─────────────────────────────────────────────────────────────────
# Self-test: the auxiliary controller inside Omega_alpha
# ─────────────────────────────────────────────────────────────────
def check_terminal_set(Q: np.ndarray, R: np.ndarray, P_full: np.ndarray, alpha: float,
                       K_red: np.ndarray, n_starts: int = 40, T: float = 3.0,
                       dt: float = 1e-3, seed: int = 1) -> bool:
    """
    Simulate ẋ = f(x, f_hover + K x) on the nonlinear model (RK4, step dt) from
    n_starts random points ON the boundary x^T P x = alpha. Checks along every
    trajectory:
        (i)   f_cmd stays inside [F_MIN, F_MAX]
        (ii)  V(t) ≤ alpha                         (positive invariance)
        (iii) V(t) + ∫ x^T Q* x dτ  is nonincreasing   (decrease condition,
              integrated form: V bounds the cost of the remaining tail)
    """
    P   = P_full[np.ix_(IDX_KEEP, IDX_KEEP)]
    Qs  = Q[np.ix_(IDX_KEEP, IDX_KEEP)] + K_red.T @ R @ K_red
    model = create_thrust_lag_model()
    f_nl  = ca.Function('f_nl', [model.x, model.u], [model.f_expl_expr])

    def to_full(xr):
        qv = xr[6:9]
        return np.concatenate([xr[0:6], [np.sqrt(max(1.0 - qv @ qv, 0.0))], qv,
                               xr[9:12], xr[12:16] + f_hover])

    def rhs(xr):
        return np.array(f_nl(to_full(xr), f_hover + K_red @ xr)).ravel()[IDX_KEEP]

    rng = np.random.default_rng(seed)
    L_inv_T = inv(cholesky(P, lower=True)).T
    worst_V, worst_inc, worst_u = 0.0, -np.inf, (np.inf, -np.inf)
    n = int(round(T / dt))
    for _ in range(n_starts):
        z = rng.standard_normal(N_RED)
        x = np.sqrt(alpha) * L_inv_T @ (z / np.linalg.norm(z))
        W = x @ P @ x                                     # V + accumulated stage cost
        W0 = W
        for _ in range(n):
            u = f_hover + K_red @ x
            worst_u = (min(worst_u[0], u.min()), max(worst_u[1], u.max()))
            k1 = rhs(x); k2 = rhs(x + dt/2*k1); k3 = rhs(x + dt/2*k2); k4 = rhs(x + dt*k3)
            x_new = x + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
            W_new = x_new @ P @ x_new + (W - x @ P @ x) + dt/2 * (x @ Qs @ x + x_new @ Qs @ x_new)
            worst_inc = max(worst_inc, (W_new - W) / W0)
            worst_V   = max(worst_V, x_new @ P @ x_new / alpha)
            x, W = x_new, W_new
    ok = (worst_u[0] >= F_MIN - 1e-9 and worst_u[1] <= F_MAX + 1e-9
          and worst_V <= 1.0 + 1e-6 and worst_inc <= 1e-9)
    print(f'─── Terminal-set check ({n_starts} starts on ∂Omega_alpha, {T:g} s, nonlinear model) ───')
    print(f'  f_cmd range                = [{worst_u[0]:.3f}, {worst_u[1]:.3f}] N   (bounds [{F_MIN:.3f}, {F_MAX:.3f}])')
    print(f'  max V(t) / alpha           = {worst_V:.6f}   (≤ 1: invariant)')
    print(f'  max step of V + ∫x^TQ*x    = {worst_inc:+.2e} · V(0)   (≤ 0: decrease condition)')
    print(f'  {"✓" if ok else "✗"} terminal set {"certified" if ok else "check FAILED"} on these trajectories')
    return ok


# ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from scipy.linalg import block_diag
    from stage1_basic.ocp_config_basic import Q as Q_BODY, Q_F, R_CMD
    Ts  = 0.05
    Q17 = block_diag(Q_BODY, Q_F)
    s_best = select_aux_gain(Q17, R_CMD, qw_weight=Ts * Q_BODY[6, 6])
    if s_best != R_AUX_SCALE:
        print(f'  NOTE: R_AUX_SCALE = {R_AUX_SCALE} in this file; set it to {s_best:g}.')
    P, alpha, kappa, K = compute_qih_offline_parameters(Q17, R_CMD, qw_weight=Ts * Q_BODY[6, 6],
                                                        r_aux_scale=s_best)
    check_terminal_set(Q17, R_CMD, P, alpha, K)
