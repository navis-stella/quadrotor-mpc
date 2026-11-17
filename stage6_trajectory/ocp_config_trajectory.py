"""
ocp_config_trajectory.py — Stage 6 Trajectory-Tracking NMPC
===========================================================
NMPC that tracks a time-varying reference (x_r(t), u_r(t)). The reference
comes from a smooth flat-output trajectory (reference_generate.py: helix,
derivatives up to snap) mapped through differential flatness
(reference_flatness.py), so every pair (x_r, u_r) satisfies the prediction
model exactly — the stage cost vanishes ON the reference and the MPC only
has to correct deviations, not fight an inconsistent target:

    min_u  Σ_{i<N} ‖ [e(x_i, x_r,i); u_i − u_r,i] ‖²_W  +  ‖ e(x_N, x_r,N) ‖²_P
    s.t.   x_{i+1} = f(x_i, u_i, d̂),   x_0 = x̂(t)
           F_MIN ≤ u_i ≤ F_MAX,   |ω_i| ≤ ω_max (soft, optional)

Motor-dynamics model: x = [body (13); f (4)], u = f_cmd,
ḟ = (f_cmd − f)/τ_m. The reference comes from
reference_flatness.motor_horizon: x_r = [x_r,body; f_r], u_r = f_r + τ_m·ḟ_r,
an exact trajectory of the thrust-lag model. Stage 7 adds its obstacle
constraints to exactly this OCP (create_ocp).

    reference at node i = trajectory at t + i·Ts   (shifts with the clock)

State error e(x, x_r) (16):
    [ p − p_r,  v − v_r,  vec(q ⊗ q_r⁻¹),  ω − ω_r,  f − f_r ]

    The attitude error is the vector part of the error quaternion
    q ⊗ q_r⁻¹ (quat_error_casadi, the same one the MHE cost uses), i.e.
    sin(θ/2)·n for a rotation by θ about the world-frame axis n. Unlike
    the component difference q − q_r of Stages 1–5 it is exact for any
    reference attitude, needs no qw weight, and ‖vec‖² is invariant under
    q → −q (double cover), so no hemisphere bookkeeping is needed.

Disturbance (Stage 3–4 offset-free mechanism, unchanged):
    d̂ (EKF / MHE) enters the prediction model as a parameter AND the
    flatness map (u_r, attitude tilt under d̂), so the reference stays a
    trajectory of f(·, ·, d̂). d̂ = 0 gives the nominal tracking MPC.

Terminal cost:
    W_e = P_lqr, the reduced-order DARE at hover, restricted to the error
    coordinates — at hover vec(q ⊗ q_r⁻¹) = q_vec, exactly the reduced DARE
    coordinates: the Ts-scaled 16-state DARE of Stage 2b
    (compute_thrust_lag_dare_terminal_cost). It is
    the LQR cost-to-go of the error dynamics linearized at hover, a good
    approximation as long as the reference stays near hover (helix: tilt
    ≤ 18°, |ω_r| ≤ 0.55 rad/s). A certified terminal cost for a generic
    reference must depend on the reference itself, see J. Köhler,
    M. A. Müller, F. Allgöwer, "A nonlinear model predictive control
    framework using reference generic terminal ingredients," IEEE TAC 65(8),
    2020, https://doi.org/10.1109/TAC.2019.2949350 .

acados mapping (same pattern as Stage 5: the residual IS the error, yref = 0):
    cost      NONLINEAR_LS, y = [state_error(x, x_r); u − u_r],  yref = 0
              (the error quaternion needs q_r inside the expression, so the
               whole reference is passed as a parameter, not via yref)
    p_i       [d̂ (6); x_r,i (17); u_r,i (4)]   (27 per node;
              u_r,N unused — no terminal input)
    RTI       one SQP iteration per sample; the previous solution is
              SHIFTED one node (shift_warm_start) because the reference
              moves with time — an unshifted guess lags by one sample.
"""

import numpy as np
import casadi as ca
from scipy.linalg import block_diag
from acados_template import AcadosOcp, AcadosOcpSolver

from core.quadrotor_model import (create_thrust_lag_disturbance_model,
                                  quat_error_casadi, f_hover, F_MIN, F_MAX, NX, NXM, NU, ND)
from core.terminal_cost import compute_thrust_lag_dare_terminal_cost
from stage1_basic.ocp_config_basic import Q, Q_F, R_CMD
from stage3_offsetfree.ocp_config_offsetfree import add_soft_omega_box
from core.codegen import build_path

# ─────────────────────────────────────────────────────────────────
# Tracking Design
# ─────────────────────────────────────────────────────────────────
# The Stage 1 weights on [x; f] and f_cmd (identical to Stages 1–5)
Q_MOTOR = block_diag(Q, Q_F)                   # 17×17 (qw slot unused here)

IDX_E = np.array([0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16])   # x slots of e
NE    = len(IDX_E)             # 16 — error coordinates (qw dropped)


# ─────────────────────────────────────────────────────────────────
# State Error
# ─────────────────────────────────────────────────────────────────
def state_error(x, x_r):
    """
    e(x, x_r) = [p − p_r, v − v_r, vec(q ⊗ q_r⁻¹), ω − ω_r]  (12).

    vec(q ⊗ q_r⁻¹) = sin(θ/2)·n ≈ ½·(small rotation vector, world frame);
    zero iff q and q_r are the same rotation (either sign).
    """
    return ca.vertcat(x[0:6] - x_r[0:6],
                      quat_error_casadi(x[6:10], x_r[6:10])[1:4],
                      x[10:13] - x_r[10:13],
                      x[NX:] - x_r[NX:])


# ─────────────────────────────────────────────────────────────────
# Solver Creator
# ─────────────────────────────────────────────────────────────────
def create_ocp(N:         int   = 20,
               T_horizon: float = 1.0,
               omega_max: float = None) -> AcadosOcp:
    """
    The trajectory-tracking OCP, not yet compiled.

    Split from create_solver so that a later stage can add constraints to
    exactly this problem (Stage 7: obstacle constraints, ocp_config_obstacle)
    instead of copying it.

    Args:
        N, T_horizon  horizon steps and duration [s]; Ts = T_horizon / N
                      must equal the sample time of the reference grid
        omega_max     optional soft |ωᵢ| bound [rad/s] (add_soft_omega_box);
                      None = no rate limit
    """
    Ts = T_horizon / N
    acados_model = create_thrust_lag_disturbance_model()
    acados_model.name = 'quadrotor_trajectory'
    nx, idx_e, ne, n_rk4 = NXM, IDX_E, NE, 2              # τ_m < Ts
    W_x   = Q_MOTOR[np.ix_(idx_e, idx_e)]
    R_u   = R_CMD
    P_lqr = compute_thrust_lag_dare_terminal_cost(Q_MOTOR, R_CMD, Ts)
    u_min = F_MIN

    d   = acados_model.p                              # d̂ (6), as in Stage 3–4
    x_r = ca.SX.sym('x_r', nx)                        # reference state at the node
    u_r = ca.SX.sym('u_r', NU)                        # reference input at the node
    acados_model.p = ca.vertcat(d, x_r, u_r)

    ocp = AcadosOcp()
    ocp.model = acados_model
    x_hover = np.zeros(nx); x_hover[6] = 1.0; x_hover[NX:] = f_hover
    ocp.parameter_values = pack_param(np.zeros(ND), x_hover, np.full(NU, f_hover))

    # ── Stage cost ‖[e; u − u_r]‖²_W ───────────────────────────
    ocp.cost.cost_type    = 'NONLINEAR_LS'
    ocp.model.cost_y_expr = ca.vertcat(state_error(acados_model.x, x_r), acados_model.u - u_r)
    ocp.cost.W            = block_diag(W_x, R_u)
    ocp.cost.yref         = np.zeros(ne + NU)

    # ── Terminal cost ‖e_N‖²_P  (hover DARE, error coordinates) ─
    ocp.cost.cost_type_e    = 'NONLINEAR_LS'
    ocp.model.cost_y_expr_e = state_error(acados_model.x, x_r)
    ocp.cost.W_e            = P_lqr[np.ix_(idx_e, idx_e)]
    ocp.cost.yref_e         = np.zeros(ne)

    # ── Constraints ─────────────────────────────────────────────
    ocp.constraints.x0 = np.zeros(nx)                             # runtime

    ocp.constraints.lbu   = np.full(NU, u_min)
    ocp.constraints.ubu   = np.full(NU, F_MAX)
    ocp.constraints.idxbu = np.arange(NU)

    if omega_max is not None:
        add_soft_omega_box(ocp, omega_max)

    # ── Solver Options ───────────────────────────────────────────
    ocp.solver_options.N_horizon       = N
    ocp.solver_options.tf              = T_horizon
    ocp.solver_options.nlp_solver_type = 'SQP_RTI'
    ocp.solver_options.integrator_type = 'ERK'
    ocp.solver_options.sim_method_num_stages = 4
    ocp.solver_options.sim_method_num_steps  = n_rk4
    ocp.solver_options.qp_solver       = 'PARTIAL_CONDENSING_HPIPM'
    ocp.solver_options.hessian_approx  = 'GAUSS_NEWTON'
    ocp.solver_options.print_level     = 0

    ocp.code_export_directory = build_path('c_generated_code_trajectory')
    return ocp


def create_solver(N:         int   = 20,
                  T_horizon: float = 1.0,
                  omega_max: float = None) -> AcadosOcpSolver:
    """
    Build the trajectory-tracking NMPC solver (create_ocp, compiled).

    Returns:
        AcadosOcpSolver — feed it with set_trajectory + set_initial_state
    """
    ocp = create_ocp(N, T_horizon, omega_max)
    return AcadosOcpSolver(ocp, json_file=build_path('acados_ocp_trajectory.json'),
                           verbose=False)


# ─────────────────────────────────────────────────────────────────
# Runtime Helpers
# ─────────────────────────────────────────────────────────────────
def pack_param(d_hat: np.ndarray, x_r: np.ndarray, u_r: np.ndarray) -> np.ndarray:
    """Parameter of one node: [d̂ (6); x_r (17); u_r (4)]  (27)."""
    return np.concatenate([d_hat, x_r, u_r])


def set_trajectory(solver: AcadosOcpSolver,
                   X_r:    np.ndarray,
                   U_r:    np.ndarray,
                   d_hat:  np.ndarray,
                   N:      int) -> None:
    """
    Reference over the horizon, node i ↔ time t + i·Ts.

        X_r  (N+1, 17)  flatness states [x_r; f_r] (reference_flatness.motor_horizon,
                        under d̂)
        U_r  (N+1, 4)   reference commands (U_r[N] unused — no terminal input)
        d̂    (6,)       disturbance estimate, same at every node
    """
    for i in range(N + 1):
        solver.set(i, 'p', pack_param(d_hat, X_r[i], U_r[i]))


def set_initial_state(solver: AcadosOcpSolver, x: np.ndarray) -> None:
    """Fix x_0 = x̂(t)."""
    solver.set(0, 'lbx', x)
    solver.set(0, 'ubx', x)


def init_guess(solver: AcadosOcpSolver,
               X_r:    np.ndarray,
               U_r:    np.ndarray,
               N:      int) -> None:
    """First call: the reference itself — a trajectory of the model (flatness)."""
    for i in range(N + 1):
        solver.set(i, 'x', X_r[i])
    for i in range(N):
        solver.set(i, 'u', U_r[i])


def shift_warm_start(solver: AcadosOcpSolver,
                     X_r:    np.ndarray,
                     U_r:    np.ndarray,
                     N:      int) -> None:
    """
    Shift the previous solution one node forward (it was computed for the
    grid t − Ts + i·Ts) and append the new reference at the end:

        x_i ← x_{i+1},  u_i ← u_{i+1}  (i < N−1),   u_{N−1} ← u_r,N−1,  x_N ← x_r,N
    """
    X = [solver.get(i, 'x') for i in range(N + 1)]
    U = [solver.get(i, 'u') for i in range(N)]
    for i in range(N - 1):
        solver.set(i, 'x', X[i + 1])
        solver.set(i, 'u', U[i + 1])
    solver.set(N - 1, 'x', X[N])
    solver.set(N - 1, 'u', U_r[N - 1])
    solver.set(N,     'x', X_r[N])
