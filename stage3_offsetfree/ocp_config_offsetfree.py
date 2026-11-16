"""
ocp_config_offsetfree.py — Stage 3 Offset-Free NMPC Solver (Motor Dynamics)
===========================================================================
Creates the acados OCP solver on the disturbance-parameterized thrust-lag
model: state [x (13); f (4)], input f_cmd (4), parameter p = d̂ (6).

Key difference from Stage 2b (ocp_config_dare.py):
    The prediction model includes disturbance forces/torques as runtime
    parameters. At each MPC step, the estimate d̂ (EKF) is injected into
    every shooting node, so the MPC predicts the future trajectory
    accounting for the estimated disturbance.

    This is what eliminates steady-state offset:
        Standard MPC:     min || f(x,u) - ref ||   → ignores disturbance → offset
        Offset-free MPC:  min || f(x,u,d̂) - ref || → compensates d̂ → zero offset

Architecture:
    Model:     create_thrust_lag_disturbance_model()  →  model.p = d (6 params)
    Terminal:  P from the Ts-scaled 16-state DARE (same as Stage 2b)
    Cost:      NONLINEAR_LS ||[x; f; f_cmd] - [x_s; f_s; f_s]||²_W   (Stage 1 weights)
    Solver:    SQP_RTI + ERK4 (2 steps per interval, τ_m < Ts)

Runtime usage:
    x_s, f_s = compute_ss_target(x_ref, d_hat, f_min=F_MIN)
    set_disturbance_param(ocp_solver, d_hat, N)
    set_reference        (ocp_solver, x_s, f_s, N)
    ocp_solver.set(0, 'lbx', x_meas)       # [x; f] (17)
    ocp_solver.set(0, 'ubx', x_meas)
    ocp_solver.solve()
"""

import numpy as np
import casadi as ca
from scipy.linalg import block_diag
from acados_template import AcadosOcp, AcadosOcpSolver

from core.quadrotor_model import (
    create_thrust_lag_disturbance_model,
    f_hover, F_MIN, F_MAX, NXM, NU, ND,
)
from core.terminal_cost import compute_thrust_lag_dare_terminal_cost
from core.codegen import build_path
from stage1_basic.ocp_config_basic import Q, Q_F, R_CMD


# ─────────────────────────────────────────────────────────────────
# Optional state constraint: body-rate box (soft)
# ─────────────────────────────────────────────────────────────────
def add_soft_omega_box(ocp: AcadosOcp, omega_max: float,
                       z_l1: float = 1e3, z_l2: float = 1e2):
    """
    |p|, |q|, |r| ≤ omega_max on nodes 1..N (node 0 is fixed by x0).
    Works for the 13-state and the 17-state model (ω sits at x[10:13] in both).

    Why: the MHE's i-iIOSS certificate only holds on |ωᵢ| ≤ ω_max
    (detectability_check.py). Without this box the MPC commands up to
    ~5.6 rad/s in the take-off transient — the true state leaves the
    certified set and the guarantee is void.

    Soft (L1 + L2 slack penalty): the box is a performance/validity limit,
    not a physical one, and a hard box could make the QP infeasible when
    an estimate starts slightly outside it. z_l1 is large enough to act as
    an exact penalty, i.e. the box holds whenever it can be held.
    """
    idx = np.array([10, 11, 12])
    lb  = np.full(3, -omega_max)
    ub  = np.full(3, +omega_max)
    ns  = len(idx)

    ocp.constraints.idxbx    = idx
    ocp.constraints.lbx      = lb
    ocp.constraints.ubx      = ub
    ocp.constraints.idxsbx   = np.arange(ns)
    ocp.cost.zl = np.full(ns, z_l1);   ocp.cost.zu = np.full(ns, z_l1)
    ocp.cost.Zl = np.full(ns, z_l2);   ocp.cost.Zu = np.full(ns, z_l2)

    ocp.constraints.idxbx_e  = idx
    ocp.constraints.lbx_e    = lb
    ocp.constraints.ubx_e    = ub
    ocp.constraints.idxsbx_e = np.arange(ns)
    ocp.cost.zl_e = np.full(ns, z_l1); ocp.cost.zu_e = np.full(ns, z_l1)
    ocp.cost.Zl_e = np.full(ns, z_l2); ocp.cost.Zu_e = np.full(ns, z_l2)


# ─────────────────────────────────────────────────────────────────
# Solver Creator
# ─────────────────────────────────────────────────────────────────
def create_solver(x_ref:     np.ndarray,
                  N:         int   = 20,
                  T_horizon: float = 1.0,
                  omega_max: float = None) -> AcadosOcpSolver:
    """
    Build the acados offset-free NMPC solver on the motor-dynamics model.

    Args:
        x_ref:      reference body state (13,)
        N:          prediction horizon steps
        T_horizon:  prediction horizon duration [s]
        omega_max:  optional soft |ωᵢ| bound [rad/s] (add_soft_omega_box);
                    None = no rate limit (the Stage 3 formulation)

    Cost structure (NONLINEAR_LS), weights of Stage 1:
        Stage:    || [x; f; f_cmd] - [x_s; f_s; f_s] ||²_W      (21-dim)
        Terminal: || [x_N; f_N] - [x_s; f_s] ||²_{P_dare}       (17-dim)

    Constraints:
        F_MIN <= f_cmd,i <= F_MAX

    Parameters:
        p = d̂ = [d_fx, d_fy, d_fz, d_tx, d_ty, d_tz] at every node k = 0..N
    """
    model = create_thrust_lag_disturbance_model()
    ocp   = AcadosOcp()
    ocp.model = model

    Ts    = T_horizon / N
    f_ref = np.full(NU, f_hover)

    # ── Cost ────────────────────────────────────────────────────
    ocp.cost.cost_type   = 'NONLINEAR_LS'
    ocp.cost.cost_type_e = 'NONLINEAR_LS'

    ocp.model.cost_y_expr   = ca.vertcat(model.x, model.u)   # (21,)
    ocp.model.cost_y_expr_e = model.x                        # (17,)

    # Terminal cost: DARE at level hover with Ts-scaled weights (Stage 2b).
    # d does not enter the DARE; at a tilted equilibrium the hover P is a
    # small mismatch for tilts of a few degrees.
    P_dare = compute_thrust_lag_dare_terminal_cost(block_diag(Q, Q_F), R_CMD, Ts)

    ocp.cost.W   = block_diag(Q, Q_F, R_CMD)
    ocp.cost.W_e = P_dare

    ocp.cost.yref   = np.concatenate([x_ref, f_ref, f_ref])   # (21,)  replaced at runtime
    ocp.cost.yref_e = np.concatenate([x_ref, f_ref])          # (17,)

    # ── Constraints ─────────────────────────────────────────────
    ocp.constraints.x0 = np.zeros(NXM)

    ocp.constraints.lbu   = np.full(NU, F_MIN)
    ocp.constraints.ubu   = np.full(NU, F_MAX)
    ocp.constraints.idxbu = np.arange(NU)

    if omega_max is not None:
        add_soft_omega_box(ocp, omega_max)

    # ── Parameter default (zero disturbance) ────────────────────
    ocp.parameter_values = np.zeros(ND)

    # ── Solver options ──────────────────────────────────────────
    ocp.solver_options.N_horizon       = N
    ocp.solver_options.tf              = T_horizon
    ocp.solver_options.nlp_solver_type = 'SQP_RTI'
    ocp.solver_options.integrator_type = 'ERK'
    ocp.solver_options.sim_method_num_stages = 4       # ERK4
    ocp.solver_options.sim_method_num_steps  = 2       # τ_m < Ts (see ocp_config_basic)
    ocp.solver_options.qp_solver       = 'PARTIAL_CONDENSING_HPIPM'
    ocp.solver_options.hessian_approx  = 'GAUSS_NEWTON'
    ocp.solver_options.print_level     = 0

    ocp.code_export_directory = build_path('c_generated_code')
    return AcadosOcpSolver(ocp, json_file=build_path(f'{ocp.model.name}_ocp.json'))


# ─────────────────────────────────────────────────────────────────
# Helper: inject d̂ into all shooting nodes
# ─────────────────────────────────────────────────────────────────
def set_disturbance_param(ocp_solver: AcadosOcpSolver,
                          d_hat: np.ndarray,
                          N: int):
    """
    Set the disturbance estimate d̂ (6,) at every shooting node 0..N,
    once per MPC step before solve() (constant-disturbance assumption).
    """
    for k in range(N + 1):
        ocp_solver.set(k, 'p', d_hat)


# ─────────────────────────────────────────────────────────────────
# Helper: update MPC reference to the equilibrium (x_s, f_s)
# ─────────────────────────────────────────────────────────────────
def set_reference(ocp_solver: AcadosOcpSolver,
                  x_s: np.ndarray,
                  f_s: np.ndarray,
                  N: int):
    """
    Move the reference at all nodes to the equilibrium under d̂.

    On the thrust-lag model the equilibrium thrust appears twice: as the
    thrust STATE f_s and as the command f_cmd = f_s (ḟ = 0 at rest).

    Args:
        x_s:  (13,) equilibrium body state from the target calculator
        f_s:  (4,)  equilibrium motor thrusts
    """
    yref   = np.concatenate([x_s, f_s, f_s])   # (21,)
    yref_e = np.concatenate([x_s, f_s])        # (17,)

    for k in range(N):
        ocp_solver.set(k, 'yref', yref)
    ocp_solver.set(N, 'yref', yref_e)
