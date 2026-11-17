"""
ocp_config_tracking.py — Stage 5 Tracking MPC with Artificial Reference
=======================================================================
Nonlinear MPC for tracking with an artificial reference, after
D. Limon, A. Ferramosca, I. Alvarado, T. Alamo, "Nonlinear MPC for
tracking piece-wise constant reference signals," IEEE TAC 63(11), 2018,
https://doi.org/10.1109/TAC.2018.2798803 :

    min_{u, y_a}  Σ L(x(k) − x_a, u(k) − u_a)  +  V_o(y_a − r)
    s.t.          x(k+1) = f(x(k), u(k)),   x(0) = x(t)
                  (x(k), u(k)) ∈ Z
                  x(N) = x_a                         ← artificial, not r!
                  (x_a, u_a) = g(y_a),   y_a ∈ Y_s

Quadrotor choices (motor-dynamics model: state [x; f], input f_cmd):
    y_a = p_a ∈ R³              hover position (yaw fixed at 0 for now)
    g(p_a)                      steady_state_map: hover at p_a, f_a = u_a = f_hover
    Z                           F_MIN ≤ f_cmd,i ≤ F_MAX,  ROOM_LB ≤ p ≤ ROOM_UB (walls)
    Y_s = [Y_LB, Y_UB]          room shrunk by MARGIN → compact, convex,
                                and its steady states are STRICTLY inside Z
    V_o = ‖p_a − r‖²_T          positive definite, strictly convex
                                → unique optimal reachable reference y_r*

acados mapping:
    p_a is a decision variable shared by all nodes → state augmentation
    x̃ = [x; f; p_a] with ṗ_a = 0 (create_tracking_model). Only [x; f] is
    fixed at node 0 (idxbx_0 = 0..16), p_a is free there and copied to every node.

        L        NONLINEAR_LS stage cost on [x − x_a(p_a); f − f_a; f_cmd − u_a], yref = 0
        V_o      NONLINEAR_LS terminal cost on p_a − r  (r = runtime parameter)
        x_N=x_a  LINEAR constraint  C_e x̃_N = 0  (x_a is linear in p_a)
        Y_s      box bounds on the p_a states (nodes 1..N−1; p_a is constant)

    r appears ONLY in the cost (as parameter) — never in a constraint.
    This is why any reference sequence keeps the OCP recursively feasible.

    Terminal equality on 16 of 17 states (IDX_TERM_EQ, qw left out — see
    quadrotor_model.py; including it would violate LICQ). The thrust states
    are included: x_N is a steady state only with f_N = f_hover.

Cost weighting note:
    acados scales stage costs by the step length Ts (cost_scaling default),
    the terminal cost by 1. So the effective weight ratio between L and V_o
    is Ts·Q : T_OFFSET. T_OFFSET is tuned with this in mind.
"""

import numpy as np
import casadi as ca
from scipy.linalg import block_diag
from acados_template import AcadosOcp, AcadosOcpSolver

from core.quadrotor_model import (create_tracking_model, steady_state_map,
                                  F_MIN, F_MAX, NXM, NU, NA, NXA, IDX_TERM_EQ)
from core.codegen import build_path
from stage1_basic.ocp_config_basic import Q, Q_F, R_CMD

# ─────────────────────────────────────────────────────────────────
# Tracking Design
# ─────────────────────────────────────────────────────────────────
# Stage cost: the Stage 1 weights on [x; f; f_cmd] (identical to Stages 1–4)
W_STAGE = block_diag(Q, Q_F, R_CMD)

ROOM_LB = np.array([-8.5, -8.5, -3.5])     # [m]  position state constraint,
ROOM_UB = np.array([ 8.5,  8.5,  3.5])     # [m]  part of Z (walls)

MARGIN = 0.5                          # [m]  Y_s = room shrunk by MARGIN
Y_LB   = ROOM_LB + MARGIN             # strictly feasible steady states
Y_UB   = ROOM_UB - MARGIN

T_OFFSET = np.diag([50., 50., 50.])   # V_o weight  (see note above)


# ─────────────────────────────────────────────────────────────────
# Solver Creator
# ─────────────────────────────────────────────────────────────────
def create_solver(N:          int   = 20,
                  T_horizon:  float = 1.0,
                  T_off:      np.ndarray = T_OFFSET,
                  nlp_solver: str   = 'SQP_RTI',
                  tag:        str   = '') -> AcadosOcpSolver:
    """
    Build the tracking-MPC solver.

    Args:
        N, T_horizon  horizon steps and duration [s]
        T_off         offset-cost weight T in V_o = ‖p_a − r‖²_T  (3×3)
        nlp_solver    'SQP_RTI' (closed loop) or 'SQP' (iterated to convergence)
        tag           suffix for model name / export folder

    Returns:
        AcadosOcpSolver   — set r with set_reference, read p_a* with
                            get_artificial_reference
    """
    model = create_tracking_model()
    model.name = f'quadrotor_tracking{tag}'
    r = ca.SX.sym('r', NA)
    model.p = r                                       # external reference

    x, p_a   = model.x[:NXM], model.x[NXM:]           # x = [body (13); thrusts (4)]
    x_a, u_a = steady_state_map(p_a)                  # g(p_a): hover at p_a

    ocp = AcadosOcp()
    ocp.model = model
    ocp.parameter_values = np.zeros(NA)

    # ── Stage cost L(x − x_a, u − u_a) ─────────────────────────
    ocp.cost.cost_type    = 'NONLINEAR_LS'
    ocp.model.cost_y_expr = ca.vertcat(x - x_a, model.u - u_a)   # [body; f; f_cmd]
    ocp.cost.W            = W_STAGE
    ocp.cost.yref         = np.zeros(NXM + NU)

    # ── Offset cost V_o(p_a − r)  (terminal; p_a is constant anyway)
    ocp.cost.cost_type_e    = 'NONLINEAR_LS'
    ocp.model.cost_y_expr_e = p_a - r
    ocp.cost.W_e            = T_off
    ocp.cost.yref_e         = np.zeros(NA)

    # ── Constraints ─────────────────────────────────────────────
    # initial condition on [x; f] only — p_a stays free at node 0
    ocp.constraints.idxbx_0  = np.arange(NXM)
    ocp.constraints.lbx_0    = np.zeros(NXM)                         # runtime
    ocp.constraints.ubx_0    = np.zeros(NXM)
    ocp.constraints.idxbxe_0 = np.arange(NXM)

    ocp.constraints.lbu   = np.full(NU, F_MIN)
    ocp.constraints.ubu   = np.full(NU, F_MAX)
    ocp.constraints.idxbu = np.arange(NU)

    # room (part of Z)  +  Y_s on the artificial reference
    ocp.constraints.idxbx = np.concatenate([np.arange(3), NXM + np.arange(NA)])
    ocp.constraints.lbx   = np.concatenate([ROOM_LB, Y_LB])
    ocp.constraints.ubx   = np.concatenate([ROOM_UB, Y_UB])

    # terminal equality  x_N[IDX_TERM_EQ] − x_a(p_a)[IDX_TERM_EQ] = 0
    C_e = np.zeros((len(IDX_TERM_EQ), NXA))
    C_e[np.arange(len(IDX_TERM_EQ)), IDX_TERM_EQ] = 1.0
    C_e[0:3, NXM:NXM + NA] = -np.eye(NA)              # p_N − p_a = 0
    ocp.constraints.C_e  = C_e                        # other rows: v, q_vec, ω = 0, f = f_hover
    g_e = np.zeros(len(IDX_TERM_EQ))
    g_e[-NU:] = u_a.full().ravel()                    # f_N = f_hover  (x_a's thrust part)
    ocp.constraints.lg_e = g_e
    ocp.constraints.ug_e = g_e

    # ── Solver Options ───────────────────────────────────────────
    ocp.solver_options.N_horizon       = N
    ocp.solver_options.tf              = T_horizon
    ocp.solver_options.nlp_solver_type = nlp_solver
    ocp.solver_options.integrator_type = 'ERK'
    ocp.solver_options.sim_method_num_stages = 4
    ocp.solver_options.sim_method_num_steps  = 2       # τ_m < Ts (see ocp_config_basic)
    ocp.solver_options.qp_solver       = 'PARTIAL_CONDENSING_HPIPM'
    ocp.solver_options.hessian_approx  = 'GAUSS_NEWTON'
    ocp.solver_options.print_level     = 0
    if nlp_solver == 'SQP':
        ocp.solver_options.nlp_solver_max_iter = 200

    ocp.code_export_directory = build_path(f'c_generated_code_tracking{tag}')
    return AcadosOcpSolver(ocp, json_file=build_path(f'acados_ocp_tracking{tag}.json'),
                           verbose=False)


# ─────────────────────────────────────────────────────────────────
# Runtime Helpers
# ─────────────────────────────────────────────────────────────────
def set_reference(solver: AcadosOcpSolver, r: np.ndarray, N: int) -> None:
    """Set the external reference r — a cost parameter at every node."""
    for k in range(N + 1):
        solver.set(k, 'p', r)


def set_initial_state(solver: AcadosOcpSolver, x: np.ndarray) -> None:
    """Fix [x; f](0|t) = measured [x; f]; the p_a part of node 0 stays free."""
    solver.set(0, 'lbx', x)
    solver.set(0, 'ubx', x)


def get_artificial_reference(solver: AcadosOcpSolver) -> np.ndarray:
    """Optimal artificial reference p_a*(t) (identical at every node)."""
    return solver.get(0, 'x')[NXM:]


def init_guess(solver: AcadosOcpSolver, x0: np.ndarray, N: int) -> None:
    """
    Warm start at rest with p_a = current position: hovering where you are
    is a feasible steady state, so the first OCP is trivially feasible.
    x0 is the measured [x; f] (17,).
    """
    xa0 = np.concatenate([x0, x0[:3]])
    _, u_a0 = steady_state_map(x0[:3])
    for k in range(N + 1):
        solver.set(k, 'x', xa0)
    for k in range(N):
        solver.set(k, 'u', u_a0)


def optimal_reachable_reference(r: np.ndarray, T_off: np.ndarray = T_OFFSET):
    """
    y_r* = argmin_{y ∈ Y_s} ‖y − r‖²_T.  For a box Y_s and DIAGONAL T this
    is the componentwise clip of r onto the box (the weighted projection).
    Used as the theoretical endpoint in plots: the closed loop converges to
    this optimal reachable steady state (Limon et al., 2018).
    """
    assert np.allclose(T_off, np.diag(np.diag(T_off))), 'needs diagonal T'
    return np.clip(r, Y_LB, Y_UB)
