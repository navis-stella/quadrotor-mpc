"""
ocp_config_dare.py — Stage 2b DARE Terminal Cost NMPC Solver
============================================================
Creates and returns the acados OCP solver.

Stage 1:   W_e = blkdiag(Q, Q_f)   (simple terminal cost)
Stage 2a:  W_e = P_lyap            (QIH — terminal cost + terminal set constraint)
Stage 2b:  W_e = P_lqr             (DARE-based terminal cost, no terminal constraint)  ← THIS FILE

Prediction model, stage cost, bounds and solver are Stage 1's (motor-dynamics
model [x; f], input f_cmd, weights imported from ocp_config_basic); only the
terminal weight changes.

Design rationale — why Stage 2b relaxes QIH:
    The QIH terminal set Omega_alpha (Stage 2a) is small: the trajectory
    reaches it only near the end of the maneuver. Dropping the terminal set
    constraint and using P_DARE as terminal cost retains the essential
    property (the infinite tail is approximated by a quadratic) without the
    infeasibility risk of a tight constraint. This is the standard
    practical choice for drone NMPC.

DARE with quaternion and motor states (compute_thrust_lag_dare_terminal_cost):
    qw is uncontrollable in the hover linearization (row/col 6 of A_c are
    zero), so the DARE runs on the other 16 states [p, v, qx..qz, ω, f] and
    qw gets a small weight back. The weights are scaled by Ts, matching
    acados' cost scaling of the stage costs — P_lqr is the cost-to-go of the
    OCP's own (Ts-weighted) stage-cost sum.

    The implementation lives in core/terminal_cost.py — later stages reuse
    this exact terminal cost, so there is one copy and the stages cannot
    drift apart.

Quaternion cost design:
    For small angles: phi ≈ 2*qx,  theta ≈ 2*qy,  psi ≈ 2*qz
    So quaternion weights ≈ 4× Euler angle weights for equivalent behavior.
    qw is only lightly penalized — it stays near 1 naturally via unit norm.
"""

import numpy as np
import casadi as ca
from scipy.linalg import block_diag
from acados_template import AcadosOcp, AcadosOcpSolver

from core.quadrotor_model          import create_thrust_lag_model, f_hover, F_MIN, F_MAX, NXM, NU
from core.terminal_cost            import compute_thrust_lag_dare_terminal_cost
from core.codegen                  import build_path
from stage1_basic.ocp_config_basic import Q, Q_F, R_CMD


# ─────────────────────────────────────────────────────────────────
# Solver Creator
# ─────────────────────────────────────────────────────────────────
def create_solver(x_ref:     np.ndarray,
                  N:         int   = 20,
                  T_horizon: float = 1.0) -> AcadosOcpSolver:
    """
    Build and compile the acados NMPC solver with the DARE terminal cost.

    Args:
        x_ref:      reference body state (13,)
                     [px,py,pz, vx,vy,vz, qw,qx,qy,qz, p,q,r]
        N:          prediction horizon (number of steps)
        T_horizon:  prediction horizon duration [s]

    Cost structure (NONLINEAR_LS):
        Stage:    || [x; f; f_cmd] - [x_ref; f_hover; f_hover] ||^2_W   (21-dim)
        Terminal: || [x_N; f_N] - [x_ref; f_hover] ||^2_{P_lqr}         (17-dim, DARE)

    No terminal set constraint — this is the relaxed Stage 2b design.
    See ocp_config_qih.py (Stage 2a) for the full QIH version.

    Constraints:
        F_MIN <= f_cmd,i <= F_MAX   for each motor

    Solver:  SQP_RTI + ERK4 (2 steps per shooting interval, for the motor lag)

    Returns:
        AcadosOcpSolver ready to be called in the simulation loop
    """
    model = create_thrust_lag_model()
    ocp   = AcadosOcp()
    ocp.model = model

    Ts    = T_horizon / N                                  # sample time
    f_ref = np.full(NU, f_hover)                           # hover thrust = hover command

    # ── Cost ────────────────────────────────────────────────────
    ocp.cost.cost_type   = 'NONLINEAR_LS'
    ocp.cost.cost_type_e = 'NONLINEAR_LS'

    ocp.model.cost_y_expr   = ca.vertcat(model.x, model.u)  # stage:    [x; f; f_cmd] (21-dim)
    ocp.model.cost_y_expr_e = model.x                       # terminal: [x; f]        (17-dim)

    # ── Terminal Cost: DARE-based P_lqr (Stage 2b) ─────────────
    P_lqr = compute_thrust_lag_dare_terminal_cost(block_diag(Q, Q_F), R_CMD, Ts)

    ocp.cost.W   = block_diag(Q, Q_F, R_CMD)  # stage weight    (21×21) — Stage 1's
    ocp.cost.W_e = P_lqr                      # terminal weight (17×17) ← DARE

    # references
    ocp.cost.yref   = np.concatenate([x_ref, f_ref, f_ref])   # stage    (21,)
    ocp.cost.yref_e = np.concatenate([x_ref, f_ref])          # terminal (17,)

    # ── Constraints ─────────────────────────────────────────────
    ocp.constraints.x0 = np.zeros(NXM)                 # updated at runtime

    ocp.constraints.lbu   = np.full(NU, F_MIN)
    ocp.constraints.ubu   = np.full(NU, F_MAX)
    ocp.constraints.idxbu = np.arange(NU)

    # ── Solver Options ───────────────────────────────────────────
    ocp.solver_options.N_horizon       = N
    ocp.solver_options.tf              = T_horizon
    ocp.solver_options.nlp_solver_type = 'SQP_RTI'
    ocp.solver_options.integrator_type = 'ERK'
    ocp.solver_options.sim_method_num_stages = 4
    ocp.solver_options.sim_method_num_steps  = 2       # τ_m < Ts (see ocp_config_basic)
    ocp.solver_options.qp_solver       = 'PARTIAL_CONDENSING_HPIPM'
    ocp.solver_options.hessian_approx  = 'GAUSS_NEWTON'
    ocp.solver_options.print_level     = 0

    ocp.code_export_directory = build_path('c_generated_code')
    return AcadosOcpSolver(ocp, json_file=build_path(f'{ocp.model.name}_ocp.json'))
