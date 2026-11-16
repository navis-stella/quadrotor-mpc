"""
compute_tube_params.py — Stage 8 Offline Tube-MPC Parameters
============================================================
Everything the tube MPC needs offline (Mayne, Seron & Raković, Automatica
2005; Rawlings, Mayne & Diehl, "Model Predictive Control", 2nd ed., §3.5),
computed for the NONLINEAR plant regulated around a hover point:

    1. linear model      x⁺ = A x + B u                    (hover, ZOH, 12 states)
    2. disturbance set   W ∋ w = F(x, u, d) − A x − B u    (box, sampled bound)
    3. ancillary gain    K   (LQR, weights chosen to make the tube small)
    4. tube cross-section S ⊇ S_K(∞) = ⊕ᵢ A_Kⁱ W          (RPI, support function only)
    5. tightening        X̄ = X ⊖ S,  Ū = U ⊖ K S
    6. terminal set      X_f = { z : zᵀ P_f z ≤ α_f } ⊂ X̄  (DARE of the nominal cost)

Coordinates — deviation from the target hover point (p_ref, v = 0, q = 1, ω = 0):

    x = [p − p_ref, v, qx qy qz, ω]   (12, qw dropped — as in the DARE cost)
    u = f − f_hover                     (4 motor thrust deviations)
    ν = MIXER u = [δT, τx, τy, τz]      (the wrench — U is a box on ν)

Why the guarantee covers the nonlinear plant:
    w is NOT only the force/torque disturbance d. It is everything the linear
    model misses in one plant step — d, the thrust-tilt coupling δT·q, the
    gyroscopic terms ω×Iω, the kinematic tilt-rate → qz coupling, RK4 vs ZOH,
    the quaternion normalization. Bounding w over X × U × D gives W. The tube
    MPC keeps x ∈ X and u ∈ U at every step, so the bound stays valid along
    the closed loop — no circular argument. The bound is sampled (all
    vertices + seeded Monte Carlo + local maximization, times W_MARGIN), the
    same kind of certificate as the Lipschitz check of Stage 2a: a searched
    maximum, not a proof. d is taken constant over each sample.

    w does not depend on p or v (asserted), so X needs no position bound and
    the target can be any hover point.

    The controller itself is purely the LINEAR tube MPC: the nonlinearity
    only enters through the size of W.

    Consequence, visible in the printout: at these set sizes the linearization
    error is LARGER than the effect of d. It grows quadratically with the size
    of X × U, the d part does not — that trade-off fixes the set sizes below.

The tube (Rawlings, Mayne & Diehl §3.5): nominal z⁺ = A z + B v started at
z₀ = x₀, ancillary law u = v + K(x − z). The error e = x − z obeys
e⁺ = A_K e + w, e₀ = 0, so e_k ∈ F_k = ⊕_{i<k} A_Kⁱ W ⊂ S_K(∞) for every k.
The tightening only needs the SUPPORT FUNCTION of S_K(∞), never the set
itself — which matters in 12 dimensions, where no explicit polytope is
computable and the explicit RPI shapes are too large for X at these sets:
an ellipsoid (LMI, K free) does not exist with K e limited to 50 % of U and
is ≥ 1.4× the state bounds with 70–90 %; a box in the eigen-coordinates of
A_K is 1.3–4.6× the mRPI set along the axes (largest share 1.75). With
W a box, h_W(c) = Σ |cᵢ| w̄ᵢ and

    h_S(c) = (1 − α)⁻¹ Σ_{i<s} h_W((A_Kⁱ)ᵀ c),    A_Kˢ W ⊆ α W

is the support function of the outer approximation F(α, s) ⊇ S_K(∞) of
Raković, Kerrigan, Kouramas & Mayne (IEEE TAC 2005) — itself RPI.

The ancillary gain K is an LQR gain on (A, B·MIXER⁻¹) whose weights are
picked from a small grid: the one whose tube occupies the smallest share
of any state bound, wrench bound or POS_SCALE (position). The nominal MPC
has its own cost (nominal_cost, Bryson on X̄ and Ū) — the two gains have
different jobs: K keeps e small, the MPC moves z to the target.

Writes data/tube_params.npz (TUBE_PARAMS_PATH), read by load_tube_params().

Run:  python -m stage8_tube.compute_tube_params            → data/tube_params.npz
      python -m stage8_tube.compute_tube_params --check    self_test of the saved parameters
"""

import os
import itertools

import numpy as np
import casadi as ca
import scipy.sparse as sp
from scipy.signal import cont2discrete
from scipy.linalg import solve_discrete_are
from scipy.optimize import linprog

from core.quadrotor_3d_model import (create_disturbance_model, create_disturbance_plant,
                                     get_hover_linearization, quat_normalize,
                                     MIXER, f_hover, F_MAX, NX, NU, ND)
from core.integrators import rk4_step          # the plant's ERK4, one step


# ─────────────────────────────────────────────────────────────────
# Coordinates
# ─────────────────────────────────────────────────────────────────
IDX_KEEP = np.array([0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12])   # skip qw (index 6)
NE       = len(IDX_KEEP)                                          # 12
NAMES    = ['px', 'py', 'pz', 'vx', 'vy', 'vz', 'qx', 'qy', 'qz', 'p', 'q', 'r']

TS = 0.05                     # [s] sample time of the tube MPC and the plant

# ─────────────────────────────────────────────────────────────────
# Sets  (the one definition — OCP, simulation and plots import these)
# ─────────────────────────────────────────────────────────────────
#   X: |x_j| ≤ X_BOUND[j] on the constrained coordinates IDX_X (reduced
#      indices). Position is free: w does not depend on it.
#   U: |ν_i| ≤ U_BOUND[i], ν = MIXER u — a box on the WRENCH deviation, not
#      on the motors. One full motor step changes ω by ~2 rad/s within a
#      single sample and w grows with δT·tilt, so the motor box [0, F_MAX]
#      would make W larger than X. The wrench box lies strictly inside the
#      motor box (asserted in check_wrench_box).
#   D: |d_j| ≤ D_BOUND[j] — time-varying force/torque disturbance.
#
#   Sizes from a sweep over (tilt, rate, δT) boxes — the smallest largest
#   tube share; tilt ≈ 5.7°, so this is regulation AROUND the target.
V_MAX   = 1.0                 # [m/s]   linear velocity, each axis
Q_TILT  = 0.05                # [-]     |qx|, |qy|   (≈ 5.7° roll / pitch)
Q_YAW   = 0.05                # [-]     |qz|         (≈ 5.7° yaw)
W_XY    = 0.5                 # [rad/s] roll / pitch rate
W_Z     = 0.2                 # [rad/s] yaw rate

IDX_X   = np.arange(3, NE)    # v, qv, ω  (reduced indices 3..11)
X_BOUND = np.array([V_MAX] * 3 + [Q_TILT, Q_TILT, Q_YAW] + [W_XY, W_XY, W_Z])

U_BOUND = np.array([0.6,                    # [N]    |δT|  total thrust deviation
                    0.05, 0.05,             # [N·m]  |τx|, |τy|
                    0.012])                 # [N·m]  |τz|  (c_tau small → weak yaw)
M_INV   = np.linalg.inv(MIXER)              # wrench → motor thrusts

D_BOUND = np.array([0.05, 0.05, 0.05,       # [N]    d_f (world) — light wind
                    5e-4, 5e-4, 5e-4])      # [N·m]  d_τ (body)  — CoG offset ~0.5 mm

W_MARGIN  = 1.1               # safety factor on the sampled bound of w
POS_SCALE = 0.5               # [m] reference size of the position tube (K selection)
ALPHA_RPI = 0.01              # A_Kˢ W ⊆ α W  → F(α, s) ⊇ S_K(∞)

POS_NOM   = 1.0               # [m] position scale of the nominal cost (Bryson)

# ancillary-gain weight grid (scaled by the bounds, see select_gain)
K_GRID = dict(r=[0.03, 0.1, 0.3], q_att=[1.0, 3.0, 10.0], q_pos=[0.1, 0.3, 1.0])

TUBE_PARAMS_PATH = os.path.join('data', 'tube_params.npz')


# ─────────────────────────────────────────────────────────────────
# Coordinate maps
# ─────────────────────────────────────────────────────────────────
def reduce_state(x: np.ndarray, p_ref: np.ndarray) -> np.ndarray:
    """13-state (qw > 0) → 12-dim deviation [p − p_ref, v, qv, ω]."""
    e = x[IDX_KEEP].copy()
    e[0:3] -= p_ref
    return e


def embed_state(e: np.ndarray, p_ref: np.ndarray) -> np.ndarray:
    """12-dim deviation → 13-state, qw = √(1 − ‖qv‖²)."""
    x = np.zeros(NX)
    x[IDX_KEEP] = e
    x[0:3] += p_ref
    x[6] = np.sqrt(max(1.0 - e[6:9] @ e[6:9], 0.0))
    return x


# ─────────────────────────────────────────────────────────────────
# 1. Linear model
# ─────────────────────────────────────────────────────────────────
def linear_model(Ts: float = TS):
    """(A, B) of the hover linearization, qw removed, ZOH — same as the DARE cost."""
    A_c, B_c = get_hover_linearization()
    A_c = A_c[np.ix_(IDX_KEEP, IDX_KEEP)]
    B_c = B_c[IDX_KEEP, :]
    A, B, *_ = cont2discrete((A_c, B_c, np.eye(NE), np.zeros((NE, NU))), Ts, method='zoh')
    return A, B


def check_wrench_box() -> np.ndarray:
    """The wrench box must map into the motor box: f_hover + M⁻¹ν ∈ (0, F_MAX)."""
    nu = np.array(list(itertools.product([-1.0, 1.0], repeat=NU))) * U_BOUND
    f  = f_hover + nu @ M_INV.T
    assert f.min() > 0.0 and f.max() < F_MAX, f'wrench box leaves [0, F_MAX]: {f.min()}..{f.max()}'
    return np.array([f.min(), f.max()])


# ─────────────────────────────────────────────────────────────────
# 2. Disturbance set W
# ─────────────────────────────────────────────────────────────────
def residual_function(A: np.ndarray, B: np.ndarray, Ts: float = TS) -> ca.Function:
    """
    w(e, u, d) = reduce(normalize(F(embed(e), f_hover + u, d))) − A e − B u

    F = one RK4 step of the disturbance model (the plant), followed by the
    quaternion normalization the simulation applies after every step.
    Evaluated at p_ref = 0 (w is translation invariant).
    """
    model  = create_disturbance_model()
    x_next = rk4_step(model, Ts)
    F      = ca.Function('F', [model.x, model.u, model.p], [x_next])

    e = ca.SX.sym('e', NE)
    u = ca.SX.sym('u', NU)
    d = ca.SX.sym('d', ND)
    qw = ca.sqrt(1.0 - ca.sumsqr(e[6:9]))
    xn = F(ca.vertcat(e[0:6], qw, e[6:12]), f_hover + u, d)
    qn = xn[6:10] / ca.norm_2(xn[6:10])
    en = ca.vertcat(xn[0:6], qn[1:4], xn[10:13])
    w  = en - ca.mtimes(ca.DM(A), e) - ca.mtimes(ca.DM(B), u)
    return ca.Function('w_res', [e, u, d], [w])


def sample_w_bound(A, B, n_random: int = 200_000, seed: int = 0, Ts: float = TS):
    """
    Componentwise bound of |w| over X × U × D. w does not depend on p or v
    (asserted), so the search runs over y = (qv, ω, ν, d) ∈ R¹⁶ at p = v = 0:

        1. all 2¹⁶ vertices of the y box
        2. uniform samples of the y box (seeded)
        3. local maximization of ±w_j (IPOPT, box-constrained) started from
           the best vertex and the best random sample of every component —
           catches a maximum in the interior of a face that 1–2 miss

    Returns:
        w_sampled  (12,) largest |wᵢ| found   (W_MARGIN NOT applied)
        w_d        (12,) the part a linear model would see: max |w| at e = u = 0
        info       dict  sample counts; vertex, random and local maxima
    """
    w_fun  = residual_function(A, B, Ts)
    box_hi = np.concatenate([X_BOUND[3:9], U_BOUND, D_BOUND])            # 16
    rng    = np.random.default_rng(seed)

    def eval_y(Y, PV=None):
        """w for rows y = (qv, ω, ν, d); PV (n, 6) optional position / velocity."""
        E = np.zeros((Y.shape[0], NE))
        E[:, 6:12] = Y[:, 0:6]
        if PV is not None:
            E[:, 0:6] = PV
        return np.array(w_fun.map(Y.shape[0])(E.T, (Y[:, 6:10] @ M_INV.T).T, Y[:, 10:16].T)).T

    # ── 1. vertices, 2. uniform samples ─────────────────────────
    Y_v = np.array(list(itertools.product([-1.0, 1.0], repeat=len(box_hi)))) * box_hi
    Y_r = rng.uniform(-box_hi, box_hi, (n_random, len(box_hi)))
    S_v, S_r = eval_y(Y_v), eval_y(Y_r)                    # signed
    W_v, W_r = np.abs(S_v).max(axis=0), np.abs(S_r).max(axis=0)

    # ── w must not depend on p or v (position free, v only bounded) ──
    n_s = 200
    PV  = np.hstack([rng.uniform(-5.0, 5.0, (n_s, 3)), rng.uniform(-V_MAX, V_MAX, (n_s, 3))])
    assert np.max(np.abs(eval_y(Y_r[:n_s], PV) - S_r[:n_s])) < 1e-9, \
        'w depends on p or v — the search over (qv, ω, ν, d) alone is not enough'

    # ── 3. local maximization from the best samples ─────────────
    y    = ca.SX.sym('y', len(box_hi))
    w_y  = w_fun(ca.vertcat(ca.SX.zeros(6), y[0:6]), ca.mtimes(ca.DM(M_INV), y[6:10]), y[10:16])
    w_of = ca.Function('w_of_y', [y], [w_y])
    opts = {'ipopt.print_level': 0, 'ipopt.sb': 'yes', 'print_time': 0, 'ipopt.tol': 1e-10}
    W_l  = np.zeros(NE)
    for j in range(NE):
        for sgn in (1.0, -1.0):
            nlp = ca.nlpsol('w_max', 'ipopt', {'x': y, 'f': -sgn * w_y[j] / W_v[j]}, opts)
            starts = (0.98 * Y_v[np.argmax(sgn * S_v[:, j])],      # just inside the best vertex
                      Y_r[np.argmax(sgn * S_r[:, j])])
            for y0 in starts:
                y_opt  = nlp(x0=y0, lbx=-box_hi, ubx=box_hi)['x']
                W_l[j] = max(W_l[j], sgn * float(w_of(y_opt)[j]))

    # ── the d part alone (e = u = 0, vertices of D) ─────────────
    Y_d = np.zeros((2**ND, len(box_hi)))
    Y_d[:, 10:16] = np.array(list(itertools.product([-1.0, 1.0], repeat=ND))) * D_BOUND
    W_d = np.abs(eval_y(Y_d)).max(axis=0)

    return np.maximum.reduce([W_v, W_r, W_l]), W_d, dict(
        n_vertex=len(Y_v), n_random=n_random, w_vertex=W_v, w_random=W_r, w_local=W_l)


# ─────────────────────────────────────────────────────────────────
# 3.–4. Ancillary gain + tube support function
# ─────────────────────────────────────────────────────────────────
def scale_rows() -> np.ndarray:
    """Per reduced coordinate: the size the tube is measured against."""
    s = np.zeros(NE)
    s[0:3]   = POS_SCALE
    s[IDX_X] = X_BOUND
    return s


def lqr_gain(A, B, Q, R_wrench):
    """LQR on the wrench input ν (B_ν = B M⁻¹); returned as motor gain K = M⁻¹ K_ν."""
    B_nu = B @ M_INV
    P    = solve_discrete_are(A, B_nu, Q, R_wrench)
    K_nu = -np.linalg.solve(R_wrench + B_nu.T @ P @ B_nu, B_nu.T @ P @ A)
    return M_INV @ K_nu


def rpi_horizon(A_K, w_bar, alpha: float = ALPHA_RPI, s_max: int = 5000) -> int:
    """Smallest s with A_Kˢ W ⊆ α W — for the box W: |A_Kˢ| w̄ ≤ α w̄ (row-wise)."""
    Ak = np.eye(NE)
    for s in range(1, s_max + 1):
        Ak = A_K @ Ak
        if np.all(np.abs(Ak) @ w_bar <= alpha * w_bar):
            return s
    raise RuntimeError(f'A_K^s W ⊄ αW for s ≤ {s_max} — A + BK too slow')


def tube_support(A_K, w_bar, C: np.ndarray, s: int, alpha: float = ALPHA_RPI) -> np.ndarray:
    """
    h_S(c) for every row c of C, S = F(α, s) = (1 − α)⁻¹ ⊕_{i<s} A_Kⁱ W:

        h_S(c) = (1 − α)⁻¹ Σ_{i<s} Σ_j |(cᵀ A_Kⁱ)_j| w̄_j
    """
    h  = np.zeros(C.shape[0])
    CA = C.copy()
    for _ in range(s):
        h  += np.abs(CA) @ w_bar
        CA  = CA @ A_K
    return h / (1.0 - alpha)


def tube_gauge(E: np.ndarray, par: dict) -> np.ndarray:
    """
    γ(e) = min { t : e ∈ t·S } for every row of E — the exact membership test
    e ∈ S ⇔ γ(e) ≤ 1 (the support function along the 12 axes is only a
    necessary condition: S is far from a box, a point at 20 % of every axis
    bound can lie outside it).

    S = F(α, s) is a zonotope, S = { G ξ : ‖ξ‖∞ ≤ 1 } with the 12·s generators
    G = (1 − α)⁻¹ [ diag(w̄), A_K diag(w̄), …, A_Kˢ⁻¹ diag(w̄) ], so γ is the LP

        min t   s.t.  G ξ = e,   −t ≤ ξ_i ≤ t        (≈ 0.1 s per point)
    """
    A_K, w_bar = par['A'] + par['B'] @ par['K'], par['w_bar']
    G, Ak = [], np.eye(NE)
    for _ in range(int(par['s_rpi'])):
        G.append(Ak * w_bar[None, :])
        Ak = A_K @ Ak
    G  = np.hstack(G) / (1.0 - par['alpha_rpi'])
    ng = G.shape[1]

    I, one = sp.identity(ng, format='csr'), sp.csr_matrix(np.ones((ng, 1)))
    A_ub = sp.vstack([sp.hstack([I, -one]), sp.hstack([-I, -one])]).tocsr()
    A_eq = sp.csr_matrix(np.hstack([G, np.zeros((NE, 1))]))
    c    = np.zeros(ng + 1); c[-1] = 1.0
    bnd  = [(None, None)] * ng + [(0.0, None)]
    return np.array([linprog(c, A_ub=A_ub, b_ub=np.zeros(2 * ng), A_eq=A_eq, b_eq=e,
                             bounds=bnd, method='highs').fun for e in np.atleast_2d(E)])


def tube_shares(K, A_K, w_bar, s: int):
    """(h_x (12,), h_ν (4,), largest share) — h along the axes and the wrench rows."""
    h_x  = tube_support(A_K, w_bar, np.eye(NE), s)
    h_nu = tube_support(A_K, w_bar, MIXER @ K, s)
    share = max(np.max(h_x / scale_rows()), np.max(h_nu / U_BOUND))
    return h_x, h_nu, share


def select_gain(A, B, w_bar):
    """
    LQR weights from K_GRID, normalized by the bounds (Bryson-style):

        Q = diag(q_pos, q_pos, q_pos, 0.3·1₃, q_att·1₃, 0.1·1₃) / s²,   R = r / Ū²

    The grid point whose tube has the smallest largest share wins.
    """
    s_row = scale_rows()
    best = None
    for r, q_att, q_pos in itertools.product(K_GRID['r'], K_GRID['q_att'], K_GRID['q_pos']):
        q = np.array([q_pos] * 3 + [0.3] * 3 + [q_att] * 3 + [0.1] * 3)
        K = lqr_gain(A, B, np.diag(q / s_row**2), np.diag(r / U_BOUND**2))
        A_K = A + B @ K
        s = rpi_horizon(A_K, w_bar)
        _, _, share = tube_shares(K, A_K, w_bar, s)
        if best is None or share < best['share']:
            best = dict(K=K, share=share, s=s, weights=(r, q_att, q_pos))
    return best


# ─────────────────────────────────────────────────────────────────
# 6. Terminal set of the nominal MPC
# ─────────────────────────────────────────────────────────────────
def nominal_cost(x_bar, nu_bar):
    """
    Stage weights of the nominal MPC — Bryson's rule on the TIGHTENED sets:

        Q = diag(1/POS_NOM² ·1₃, 1/x̄_j²),     R = Mᵀ diag(1/ν̄²) M  (wrench)

    Why not the Stage 2b weights (position 80, tilt 120): their DARE gain is
    so aggressive in position that the terminal ellipsoid has to shrink to
    ~4 mm to keep K_f z inside the 2.8° tilt of X̄. Normalized weights give a
    terminal set of ~0.3 m, reachable within the horizon.
    """
    q = np.zeros(NE)
    q[0:3]   = 1.0 / POS_NOM**2
    q[IDX_X] = 1.0 / x_bar**2
    return np.diag(q), MIXER.T @ np.diag(1.0 / nu_bar**2) @ MIXER


def terminal_set(A, B, x_bar, nu_bar, Q_cost, R_cost):
    """
    DARE (P_f, K_f) of the nominal cost and the largest α_f with
    {zᵀP_f z ≤ α_f} ⊂ X̄ and MIXER K_f z ∈ Ū. The nominal system has no
    disturbance, so V_f decreases by ℓ(z, K_f z) and the set is invariant.
    """
    P_f = solve_discrete_are(A, B, Q_cost, R_cost)
    K_f = -np.linalg.solve(R_cost + B.T @ P_f @ B, B.T @ P_f @ A)
    P_inv = np.linalg.inv(P_f)
    MK  = MIXER @ K_f
    a_x = x_bar**2  / np.diag(P_inv)[IDX_X]
    a_u = nu_bar**2 / np.einsum('ij,jk,ik->i', MK, P_inv, MK)
    return P_f, K_f, float(min(a_x.min(), a_u.min()))


# ─────────────────────────────────────────────────────────────────
# Pipeline + I/O
# ─────────────────────────────────────────────────────────────────
def compute_tube_params(seed: int = 0, verbose: bool = True) -> dict:
    f_range = check_wrench_box()
    A, B = linear_model(TS)
    w_samp, w_d, info = sample_w_bound(A, B, seed=seed)
    w_bar = W_MARGIN * w_samp

    gain = select_gain(A, B, w_bar)
    K, s = gain['K'], gain['s']
    A_K  = A + B @ K
    h_x, h_nu, share = tube_shares(K, A_K, w_bar, s)
    x_bar  = X_BOUND - h_x[IDX_X]
    nu_bar = U_BOUND - h_nu
    assert np.all(x_bar > 0) and np.all(nu_bar > 0), 'tightened sets are empty'
    Q_cost, R_cost = nominal_cost(x_bar, nu_bar)
    P_f, K_f, alpha_f = terminal_set(A, B, x_bar, nu_bar, Q_cost, R_cost)
    f_semi = np.sqrt(alpha_f * np.diag(np.linalg.inv(P_f)))

    par = dict(Ts=TS, A=A, B=B, w_bar=w_bar, w_sampled=w_samp, w_d=w_d,
               K=K, rho_K=np.max(np.abs(np.linalg.eigvals(A_K))), s_rpi=s,
               alpha_rpi=ALPHA_RPI, share=share, k_weights=np.array(gain['weights']),
               idx_x=IDX_X, x_bound=X_BOUND, x_bar=x_bar, h_x=h_x,
               u_bound=U_BOUND, nu_bar=nu_bar, h_nu=h_nu, d_bound=D_BOUND,
               P_f=P_f, K_f=K_f, alpha_f=alpha_f, Q_cost=Q_cost, R_cost=R_cost)

    if verbose:
        print('─── Stage 8 — tube MPC offline parameters ─────────────────────────')
        print(f'  wrench box ±ν      : {U_BOUND}   → motors in [{f_range[0]:.2f}, {f_range[1]:.2f}] N')
        print(f'  w samples          : {info["n_vertex"]} vertices + {info["n_random"]} random'
              f' + local NLP,  W_MARGIN = {W_MARGIN}')
        print(f'  local / vertex max : up to {np.max(info["w_local"] / info["w_vertex"]):.5f}'
              f'   random / vertex: {np.min(info["w_random"] / info["w_vertex"]):.2f}'
              f' … {np.max(info["w_random"] / info["w_vertex"]):.2f}')
        print(f'  {"":3s} {"w from d":>9s} {"w sampled":>9s} {"vertex":>9s} {"random":>9s}'
              f' {"h_S":>8s} {"bound":>7s} {"X̄":>8s} {"share":>6s}')
        for j in range(NE):
            k = np.where(IDX_X == j)[0]
            b, bt, sh = (X_BOUND[k[0]], x_bar[k[0]], h_x[j] / X_BOUND[k[0]]) if len(k) else (None, None, None)
            print(f'  {NAMES[j]:3s} {w_d[j]:9.2e} {w_samp[j]:9.2e} {info["w_vertex"][j]:9.2e} '
                  f'{info["w_random"][j]:9.2e} {h_x[j]:8.4f} '
                  + (f'{b:7.3f} {bt:8.4f} {sh:6.2f}' if b is not None else '   free        —      —'))
        print(f'  ancillary K        : LQR weights (r, q_att, q_pos) = {gain["weights"]}'
              f'   ρ(A + BK) = {par["rho_K"]:.4f}')
        print(f'  RPI outer approx.  : F(α = {ALPHA_RPI}, s = {s})')
        print(f'  wrench h_KS        : {np.round(h_nu, 4)}   → Ū = ±{np.round(nu_bar, 4)}'
              f'   shares {np.round(h_nu / U_BOUND, 2)}')
        print(f'  largest tube share : {share:.3f}')
        print(f'  terminal α_f       : {alpha_f:.4f}   semi-axes p {np.round(f_semi[0:3], 3)} m,'
              f' v {np.round(f_semi[3:6], 3)} m/s')
        print('────────────────────────────────────────────────────────────────────')
    return par


def save_tube_params(par: dict, path: str = TUBE_PARAMS_PATH) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez(path, **par)
    print(f'Saved: {path}')


def load_tube_params(path: str = TUBE_PARAMS_PATH) -> dict:
    """Read the tube parameters; scalars come back as Python numbers."""
    if not os.path.exists(path):
        raise FileNotFoundError(f'{path} missing — run compute_tube_params.py first')
    with np.load(path) as f:
        return {k: (f[k].item() if f[k].ndim == 0 else f[k]) for k in f.files}


# ─────────────────────────────────────────────────────────────────
# Self-test  (python -m stage8_tube.compute_tube_params --check)
# ─────────────────────────────────────────────────────────────────
def self_test(par: dict, n: int = 300, seed: int = 3) -> None:
    """
    Three checks of the offline ingredients against independent computations:

      1. w(e, u, d) of residual_function = the acados plant step (the W bound
         is about the plant, not about a CasADi copy of it)
      2. h_S(c) is attained: the sequence w_i = w̄ ∘ sign((A_Kⁱ)ᵀ c) drives
         cᵀe_s to exactly (1 − α)·h_S(c) — the support function is tight
      3. X_f: on its boundary |z_j| ≤ x̄_j, |M K_f z| ≤ ν̄ and
         V_f(z⁺) − V_f(z) = −ℓ(z, K_f z) — inside X̄ and invariant
    """
    rng  = np.random.default_rng(seed)
    A, B, K, w_bar = par['A'], par['B'], par['K'], par['w_bar']
    A_K, s, alpha  = A + B @ K, int(par['s_rpi']), par['alpha_rpi']

    # ── 1. residual vs plant ────────────────────────────────────
    w_fun, plant, gap = residual_function(A, B, par['Ts']), create_disturbance_plant(par['Ts']), 0.0
    for _ in range(n):
        e = np.concatenate([rng.uniform(-2.0, 2.0, 3), rng.uniform(-X_BOUND, X_BOUND)])
        u = M_INV @ rng.uniform(-U_BOUND, U_BOUND)
        d = rng.uniform(-D_BOUND, D_BOUND)
        plant.set('x', embed_state(e, np.zeros(3))); plant.set('u', f_hover + u); plant.set('p', d)
        plant.solve()
        w_plant = reduce_state(quat_normalize(plant.get('x')), np.zeros(3)) - A @ e - B @ u
        gap = max(gap, np.max(np.abs(w_plant - np.array(w_fun(e, u, d)).ravel())))
    assert gap < 1e-12, f'residual_function differs from the plant: {gap:.2e}'

    # ── 2. support function attained by the worst-case sequence ──
    C, ratio = np.vstack([np.eye(NE), MIXER @ K]), []
    h = tube_support(A_K, w_bar, C, s)
    for c, h_c in zip(C, h):
        signs, cA = [], c.copy()
        for _ in range(s):
            signs.append(np.sign(cA)); cA = cA @ A_K
        e = np.zeros(NE)
        for sg in reversed(signs):                 # the w that gets A_Kⁱ is applied i steps earlier
            e = A_K @ e + w_bar * sg
        ratio.append(c @ e / ((1.0 - alpha) * h_c))
    assert np.allclose(ratio, 1.0, atol=1e-9), f'support function not attained: {ratio}'

    # ── 3. terminal set ─────────────────────────────────────────
    P_f, K_f, a_f = par['P_f'], par['K_f'], par['alpha_f']
    L  = np.linalg.cholesky(np.linalg.inv(P_f))
    Zb = rng.standard_normal((20_000, NE))
    Zb = np.sqrt(a_f) * (Zb / np.linalg.norm(Zb, axis=1)[:, None]) @ L.T        # zᵀP_f z = α_f
    Zn = Zb @ (A + B @ K_f).T
    quad = lambda Z, M: np.einsum('ij,jk,ik->i', Z, M, Z)
    x_sh = np.max(np.abs(Zb[:, IDX_X]) / par['x_bar'])
    u_sh = np.max(np.abs(Zb @ (MIXER @ K_f).T) / par['nu_bar'])
    dec  = np.max(np.abs(quad(Zn, P_f) - quad(Zb, P_f)
                         + quad(Zb, par['Q_cost'] + K_f.T @ par['R_cost'] @ K_f)))
    assert x_sh <= 1.0 and u_sh <= 1.0 and dec < 1e-9 * a_f, (x_sh, u_sh, dec)

    print('─── self-test ─────────────────────────────────────────────────────')
    print(f'  residual vs acados plant   : max difference {gap:.1e}   ({n} samples)')
    print(f'  h_S attained (16 rows)     : cᵀe_s / ((1−α) h_S) ∈ [{min(ratio):.6f}, {max(ratio):.6f}]')
    print(f'  X_f boundary (20000 pts)   : max |z|/x̄ = {x_sh:.3f}, max |M K_f z|/ν̄ = {u_sh:.3f},'
          f'  |ΔV_f + ℓ| ≤ {dec:.1e}')


if __name__ == '__main__':
    import sys
    if '--check' in sys.argv:
        self_test(load_tube_params())
    else:
        save_tube_params(compute_tube_params())
