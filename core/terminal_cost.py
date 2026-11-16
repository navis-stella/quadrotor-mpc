"""
terminal_cost.py — All Stages DARE Terminal Cost
================================================
The reduced-order DARE terminal cost P_lqr at hover — the one implementation,
used by Stage 2b (W_e = P_lqr), Stage 3–4 (offset-free NMPC) and Stage 6
(trajectory tracking, on the 12 error coordinates).

It lives in core/ because Stage 2b would otherwise import it from Stage 3 —
a dependency pointing backwards along the stage order.
"""

import numpy as np
from scipy.linalg import solve_discrete_are
from scipy.signal import cont2discrete

from core.quadrotor_3d_model import get_hover_linearization, NX, NU


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
