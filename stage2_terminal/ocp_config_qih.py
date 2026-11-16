"""
ocp_config_qih.py — Stage 2a QIH-NMPC Solver
============================================
Stage 2a:  Complete Quasi-Infinite Horizon NMPC

Implements QIH-NMPC on the motor-dynamics model (state [x; f], input f_cmd)
with:
    - Terminal cost W_e = P_lyap   (from the modified Lyapunov equation)
    - Terminal constraint:  Δx_N^T P Δx_N <= alpha   (Δx without qw)
      implemented as a SOFT constraint to avoid hard infeasibility.

The terminal constraint enforces x_N ∈ Omega_alpha, the positively
invariant region in which the auxiliary controller K stabilizes the system
without violating input bounds, with the terminal cost decreasing at least
as fast as the stage cost accumulates. Softening it (L1 + L2 slack
penalties) keeps the stability guarantee whenever the hard problem is
feasible (exact penalty), and returns a solution instead of failing when
it is not.

Cost structure (NONLINEAR_LS), stage cost and bounds as in Stage 1:
    Stage:     || [x; f; f_cmd] - [x_ref; f_hover; f_hover] ||^2_W    (21-dim)
    Terminal:  || [x_N; f_N] - [x_ref; f_hover] ||^2_{P_lyap}         (17-dim)
    + soft penalty on terminal set violation

Constraints:
    Input:     F_MIN <= f_cmd,i <= F_MAX          (hard, box on u)
    Terminal:  Δx_N^T P_set Δx_N <= alpha          (soft)

Offline parameters (P, alpha, kappa, K) come from get_qih_params(): computed
once per process and shared with simulate_qih.py's diagnostic plot, so the
plotted terminal set is exactly the one the solver enforces.

Extends:  Stage 1 (W_e = blkdiag(Q, Q_f))
See also: Stage 2b (ocp_config_dare.py) — relaxed, P_DARE only
"""

from functools import lru_cache

import numpy as np
import casadi as ca
from scipy.linalg import block_diag
from acados_template import AcadosOcp, AcadosOcpSolver

from core.quadrotor_model               import create_thrust_lag_model, f_hover, F_MIN, F_MAX, NXM, NU
from core.codegen                       import build_path
from stage1_basic.ocp_config_basic      import Q, Q_F, R_CMD
from stage2_terminal.compute_qih_params import compute_qih_offline_parameters

TS_NOMINAL = 0.05     # [s] T/N of the Stage 1–2 scenario; sets the small qw terminal weight


# ─────────────────────────────────────────────────────────────────
# Offline QIH Parameters
# ─────────────────────────────────────────────────────────────────
@lru_cache(maxsize=1)
def get_qih_params():
    """
    (P_qih, alpha_qih, kappa, K_red) for the Stage 1 weights.

    Cached: the CARE / Lyapunov / decrease-condition pipeline runs once per
    process, however often the solver and the plots ask for it.
    """
    return compute_qih_offline_parameters(block_diag(Q, Q_F), R_CMD,
                                          qw_weight=TS_NOMINAL * Q[6, 6])


def terminal_set():
    """
    (P_set, alpha): the terminal region Omega_alpha = {Δx : Δx^T P_set Δx <= alpha}
    on the 17-state model. P_set is P_qih without the qw entry — qw is not
    part of the reduced system the region is certified on (with ‖q‖ = 1 it
    follows from q_v).
    """
    P_qih, alpha, _, _ = get_qih_params()
    P_set = P_qih.copy()
    P_set[6, 6] = 0.0
    return P_set, alpha


# ─────────────────────────────────────────────────────────────────
# Solver Creator
# ─────────────────────────────────────────────────────────────────
def create_solver(x_ref:     np.ndarray,
                  N:         int   = 20,
                  T_horizon: float = 1.0) -> AcadosOcpSolver:
    """
    Build and compile the acados QIH-NMPC solver.

    Args:
        x_ref:      reference body state (13,)
        N:          prediction horizon steps
        T_horizon:  prediction horizon duration [s]

    Returns:
        AcadosOcpSolver ready for the simulation loop
    """
    model = create_thrust_lag_model()
    ocp   = AcadosOcp()
    ocp.model = model

    f_ref  = np.full(NU, f_hover)
    xf_ref = np.concatenate([x_ref, f_ref])            # (17,) terminal reference

    # ── QIH Offline Parameters ──────────────────────────────────
    P_qih, _, _, _   = get_qih_params()
    P_set, alpha_qih = terminal_set()

    # ── Cost: NONLINEAR_LS ──────────────────────────────────────
    ocp.cost.cost_type   = 'NONLINEAR_LS'
    ocp.cost.cost_type_e = 'NONLINEAR_LS'

    ocp.model.cost_y_expr   = ca.vertcat(model.x, model.u)   # (21,)
    ocp.model.cost_y_expr_e = model.x                        # (17,)

    ocp.cost.W   = block_diag(Q, Q_F, R_CMD)   # (21×21) stage — Stage 1's
    ocp.cost.W_e = P_qih                       # (17×17) terminal — Lyapunov P

    ocp.cost.yref   = np.concatenate([x_ref, f_ref, f_ref])   # (21,)
    ocp.cost.yref_e = xf_ref                                  # (17,)

    # ── Input Constraints (hard) ────────────────────────────────
    ocp.constraints.x0    = np.zeros(NXM)              # set at runtime
    ocp.constraints.lbu   = np.full(NU, F_MIN)
    ocp.constraints.ubu   = np.full(NU, F_MAX)
    ocp.constraints.idxbu = np.arange(NU)

    # ── Terminal Constraint: Δx_N^T P_set Δx_N <= alpha (soft) ──
    #
    #   Nonlinear terminal constraint h_e(x_N) = Δx_N^T P_set Δx_N,
    #   Δx_N = x_N − [x_ref; f_hover].  Bounds:  0 <= h_e <= alpha
    #
    #   P_set and x_ref are baked in as numbers at compile time; a runtime
    #   reference change would need them as acados parameters.
    #
    x_dev = model.x - xf_ref                                 # CasADi symbolic (17,)
    ocp.model.con_h_expr_e = ca.mtimes([x_dev.T, ca.DM(P_set), x_dev])

    ocp.constraints.lh_e = np.array([0.0])
    ocp.constraints.uh_e = np.array([alpha_qih])

    # ── Soft Constraint on Terminal Set ─────────────────────────
    #
    #   Slack s >= 0 relaxes the upper bound:  h_e <= alpha + s
    #   Penalty added to cost:  zu * s  +  Zu * s^2
    #
    #   zu (L1): exact penalty — the softened problem has the hard
    #            problem's solution whenever zu exceeds the multiplier.
    #   Zu (L2): smooths the penalty for the QP solver.
    #   Only the upper bound is softened (h_e >= 0 always holds).
    #
    ocp.constraints.idxsh_e = np.array([0])

    ocp.cost.zl_e = np.array([0.0])
    ocp.cost.Zl_e = np.array([0.0])
    ocp.cost.zu_e = np.array([1e3])            # L1 penalty on upper violation
    ocp.cost.Zu_e = np.array([1e5])            # L2 penalty on upper violation

    # ── Solver Options ──────────────────────────────────────────
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
    return AcadosOcpSolver(ocp, json_file=build_path(f'{ocp.model.name}_qih_ocp.json'))
