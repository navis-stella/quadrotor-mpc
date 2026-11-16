"""
observability_check.py — Stage 4a Local Observability at Hover (Motor Dynamics)
==============================================================================
Verify that the linearized augmented pair (F, H) at hover is observable.

    z = [x(13); f(4); d(6)]                    augmented thrust-lag state (23)
    F = ∂f_aug/∂z |_(z_hover, u_hover)          (23 × 23)
    H = selection of y = [p; q; ω]             (10 × 23, constant)
                                                v, f, d unmeasured

If (F, H) is observable, the estimator can (in principle) recover the full
augmented state from the measurement stream. This is the local,
single-point sanity check behind both:

    EKF (Stage 4a):  observability ⇒ linearized covariance update converges
    MHE (Stage 4b):  needs the stronger, non-local statement — i-iIOSS on
                     the whole operating envelope — which detectability_check.py
                     certifies via LMIs (and whose weights the MHE uses).

Second case for comparison: the thrusts measured as well (ESC rpm
telemetry, y = [p; q; ω; f], 14) — not assumed in Stage 4.

Numerics: the thrust lag puts −1/τ_m = −33 s⁻¹ into F, so the unscaled
powers F^k reach ~10³³ at k = 22 and a relative rank tolerance would
discard every physical direction. Observability is invariant to scaling
F by a nonzero constant (same Krylov space), so O is built from F·T_S —
the dynamics over one sample. The PBH test, rank [λI − F; H] = n for every
eigenvalue λ of F, is evaluated as an independent cross-check.
"""

import numpy as np

from core.quadrotor_model import (
    get_thrust_lag_augmented_dynamics,
    thrust_lag_augmented_hover_point,
    IDX_Y, IDX_Y_TL, NZM,
)

T_S = 0.05     # [s] time scale of F in the observability matrix (one sample)


# ─────────────────────────────────────────────────────────────────
# State labels for diagnostic output
# ─────────────────────────────────────────────────────────────────
STATE_NAMES = [
    'px',   'py',   'pz',
    'vx',   'vy',   'vz',
    'qw',   'qx',   'qy',   'qz',
    'p',    'q',    'r',
    'f1',   'f2',   'f3',   'f4',
    'd_fx', 'd_fy', 'd_fz',
    'd_tx', 'd_ty', 'd_tz',
]


# ─────────────────────────────────────────────────────────────────
# Observability matrix
# ─────────────────────────────────────────────────────────────────
def build_observability_matrix(F: np.ndarray, H: np.ndarray) -> np.ndarray:
    """
    Standard Kalman observability matrix:
        O = [ H; HF; HF^2; ...; HF^{n-1} ]

    Shape: (n * ny, n) where n = F.shape[0].
    Rank(O) = n  ⇔  (F, H) is observable.
    """
    n = F.shape[0]
    ny = H.shape[0]
    O = np.zeros((n * ny, n))
    HFk = H.copy()
    for k in range(n):
        O[k*ny:(k+1)*ny, :] = HFk
        HFk = HFk @ F
    return O


# ─────────────────────────────────────────────────────────────────
# Observability check
# ─────────────────────────────────────────────────────────────────
def check_observability(measure_thrust: bool = False, verbose: bool = True) -> dict:
    """
    Test full observability of the augmented pair (F, H) at hover.

    Args:
        measure_thrust: False → y = [p; q; ω]     (Stage 4 sensor, 10)
                        True  → y = [p; q; ω; f]  (with rpm telemetry, 14)

    Returns:
        result: dict with keys
            'rank'         — numerical rank of O
            'n'            — full rank target (= NZM)
            'is_observable'— bool
            'sigmas'       — singular values of O
            'cond'         — σ_max / σ_min
            'unobs_basis'  — null-space basis (columns), empty if observable
    """
    z_hover, u_hover = thrust_lag_augmented_hover_point()

    _, F_func = get_thrust_lag_augmented_dynamics()
    H = np.eye(NZM)[IDX_Y_TL if measure_thrust else IDX_Y]

    F = np.array(F_func(z_hover, u_hover))
    O = build_observability_matrix(F * T_S, H)        # scaled: see the header

    # PBH cross-check: rank [λI − F; H] for every distinct eigenvalue of F
    pbh = {}
    for lam in np.unique(np.round(np.linalg.eigvals(F), 9)):
        M = np.vstack([lam * np.eye(F.shape[0]) - F, H])
        pbh[complex(lam)] = int(np.linalg.matrix_rank(M))

    # SVD-based rank with a scale-aware tolerance
    _, sigmas, Vt = np.linalg.svd(O)
    tol = max(O.shape) * sigmas[0] * np.finfo(float).eps
    rank = int(np.sum(sigmas > tol))
    n = F.shape[0]

    unobs_basis = np.zeros((n, 0))
    if rank < n:
        unobs_basis = Vt[-(n - rank):].T   # columns span null(O)

    result = {
        'rank':          rank,
        'n':             n,
        'is_observable': rank == n,
        'sigmas':        sigmas,
        'cond':          sigmas[0] / sigmas[-1],
        'unobs_basis':   unobs_basis,
        'pbh':           pbh,
    }
    if verbose:
        label = ('y = [p; q; ω; f]  (thrust measured, comparison)' if measure_thrust
                 else 'y = [p; q; ω]  (Stage 4 sensor)')
        _print_observability_report(F, H, O, sigmas, tol, rank, n, unobs_basis, label)
        print("  PBH test, rank [λI − F; H] per eigenvalue λ of F:")
        for lam, r in pbh.items():
            print(f"    λ = {lam.real:+9.3f}{lam.imag:+.3f}j   rank {r} / {n}"
                  f"{'' if r == n else '   ✗'}")
        print("═" * 68)
    return result


def _print_observability_report(F, H, O, sigmas, tol, rank, n, unobs_basis, label):
    print("═" * 68)
    print(f"  Stage 4a — Observability check at hover, {label}")
    print("═" * 68)
    print(f"  F shape       : {F.shape}")
    print(f"  H shape       : {H.shape}")
    print(f"  O shape       : {O.shape}   (should be {n*H.shape[0]} × {n}), built from F·{T_S} s")
    print()

    print("  Singular values of O (largest to smallest):")
    for i, sv in enumerate(sigmas):
        marker = "   ← below tol" if sv <= tol else ""
        print(f"    σ[{i:2d}] = {sv:.4e}{marker}")

    print()
    print(f"  Rank tolerance : {tol:.4e}")
    print(f"  Rank(O)        : {rank}  /  n = {n}")
    print(f"  Condition      : {sigmas[0] / sigmas[-1]:.3e}")
    print()

    if rank == n:
        print("  ✓ (F, H) is OBSERVABLE at hover.")
        print("    → the EKF can recover z = [x; f; d] from y (locally).")
    else:
        print(f"  ✗ (F, H) is NOT observable — {n - rank} unobservable direction(s).")
        print()
        print("  Unobservable subspace (columns of null(O), |v| > 0.01 shown):")
        for j in range(unobs_basis.shape[1]):
            print(f"    Direction {j+1}:")
            for i, name in enumerate(STATE_NAMES):
                v = unobs_basis[i, j]
                if abs(v) > 0.01:
                    print(f"      {name:6s}  {v:+.4f}")
    print()


# ─────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    check_observability(measure_thrust=False)
    print()
    check_observability(measure_thrust=True)
