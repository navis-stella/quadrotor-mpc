"""
terminal_cost.py — All Stages DARE Terminal Cost
================================================
The reduced-order DARE terminal cost P_lqr at hover — the one implementation,
used by Stage 2b (W_e = P_lqr), Stage 3–4 (offset-free NMPC) and Stage 6
(trajectory tracking, on the 12 error coordinates).

  compute_dare_terminal_cost             13-state model, thrusts as input
                                         (stages not yet on the motor dynamics)
  compute_thrust_lag_dare_terminal_cost  17-state thrust-lag model, f_cmd as
                                         input, scaled consistently with
                                         acados' cost_scaling (see there)

It lives in core/ because Stage 2b would otherwise import it from Stage 3 —
a dependency pointing backwards along the stage order.
"""

import numpy as np
from scipy.linalg import solve_discrete_are
from scipy.signal import cont2discrete

from core.quadrotor_model import (get_hover_linearization, get_thrust_lag_hover_linearization,
                                  NX, NXM, NU)


# ─────────────────────────────────────────────────────────────────
# DARE Terminal Cost — Reduced-Order for Quaternion
# ─────────────────────────────────────────────────────────────────
def compute_dare_terminal_cost(Q: np.ndarray,
                               R: np.ndarray,
                               Ts: float) -> np.ndarray:
    """
    Compute P_lqr via reduced-order DARE.

    Removes qw (index 6), solves DARE on 12 states, embeds back.
    The disturbance does NOT enter the DARE — the terminal cost
    penalizes deviation from the reference, and d̂ handles the offset
    through the prediction model, not the terminal cost.

    Returns:
        P_lqr: (13×13) terminal cost matrix
    """
    A_c, B_c = get_hover_linearization()

    idx_keep = [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12]
    A_c_red = A_c[np.ix_(idx_keep, idx_keep)]
    B_c_red = B_c[idx_keep, :]
    Q_red   = Q[np.ix_(idx_keep, idx_keep)]

    n_red = len(idx_keep)
    sys_d = cont2discrete(
        (A_c_red, B_c_red, np.eye(n_red), np.zeros((n_red, NU))),
        Ts, method='zoh'
    )
    A_d_red, B_d_red = sys_d[0], sys_d[1]

    P_red = solve_discrete_are(A_d_red, B_d_red, Q_red, R)

    eigvals_red = np.linalg.eigvalsh(P_red)
    assert np.all(eigvals_red > 0), \
        f'P_red not positive definite! eigvals={eigvals_red}'

    P_lqr = np.zeros((NX, NX))
    P_lqr[np.ix_(idx_keep, idx_keep)] = P_red
    P_lqr[6, 6] = Q[6, 6]

    print('─── DARE Terminal Cost (reduced-order) ────────────')
    print(f'  Sample time Ts     = {Ts:.4f} s')
    print(f'  P_lqr diagonal     = {np.diag(P_lqr)}')
    print(f'  Ratio P/Q diag     = {np.diag(P_lqr) / np.diag(Q)}')
    print('───────────────────────────────────────────────────')

    return P_lqr


# ─────────────────────────────────────────────────────────────────
# DARE Terminal Cost — Thrust-Lag Model (motor dynamics)
# ─────────────────────────────────────────────────────────────────
IDX_KEEP_TL = [i for i in range(NXM) if i != 6]     # 16 states: all but qw


def compute_thrust_lag_dare_terminal_cost(Q:  np.ndarray,
                                          R:  np.ndarray,
                                          Ts: float) -> np.ndarray:
    """
    P_lqr for the thrust-lag model [x (13); f (4)] with input f_cmd — the
    terminal cost of every OCP on the motor-dynamics model.

    Args:
        Q:   (17×17) stage weight on [x; f]  (blkdiag(Q_body, Q_f))
        R:   (4×4)   stage weight on f_cmd
        Ts:  shooting interval [s]

    COST SCALING. acados multiplies every stage cost by its time step
    (default cost_scaling = [Ts, …, Ts, 1]), so the OCP's stage cost is
    Σ_k Ts·ℓ(x_k, u_k) — the Riemann sum of ∫ ℓ dt — while the terminal cost
    is not scaled. The cost-to-go that matches this sum is the DARE solution
    for the weights (Ts·Q, Ts·R), i.e. Ts times the DARE solution for (Q, R)
    (the DARE is homogeneous in the weights). Without the factor the terminal
    cost would be 1/Ts = 20× the tail it stands for.

    Same reduction as compute_dare_terminal_cost: qw (index 6) is an
    uncontrollable marginal mode at hover, so the DARE runs on the other 16
    states and qw gets the small weight Ts·Q[6,6] back (one stage interval's
    worth), which keeps the Gauss-Newton Hessian regular.

    Returns:
        P_lqr: (17×17)
    """
    A_c, B_c = get_thrust_lag_hover_linearization()
    keep = IDX_KEEP_TL
    n    = len(keep)
    sys_d = cont2discrete((A_c[np.ix_(keep, keep)], B_c[keep, :],
                           np.eye(n), np.zeros((n, NU))), Ts, method='zoh')
    A_d, B_d = sys_d[0], sys_d[1]

    P_red = solve_discrete_are(A_d, B_d, Ts * Q[np.ix_(keep, keep)], Ts * R)
    P_red = 0.5 * (P_red + P_red.T)
    assert np.all(np.linalg.eigvalsh(P_red) > 0), 'P_red not positive definite'

    P = np.zeros((NXM, NXM))
    P[np.ix_(keep, keep)] = P_red
    P[6, 6] = Ts * Q[6, 6]

    print('─── DARE Terminal Cost (thrust-lag model, Ts-scaled) ─')
    print(f'  Sample time Ts     = {Ts:.4f} s')
    print(f'  P diagonal         = {np.array2string(np.diag(P), precision=3)}')
    print(f'  P / (Ts·Q) diag    = {np.array2string(np.diag(P) / (Ts * np.diag(Q)), precision=2)}')
    print('───────────────────────────────────────────────────')
    return P
