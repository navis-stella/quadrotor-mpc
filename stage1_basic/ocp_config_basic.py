"""
ocp_config_basic.py — Stage 1 Basic NMPC Solver (W_e = Q)
=========================================================
Creates and returns the acados OCP solver.

Stage 1:  W_e = Q   (simple terminal cost, no terminal constraint)

Prediction model: the motor-dynamics model (quadrotor_3d_model.
create_motor_model, d = 0) — 17 states [x (13); f (4)], input the thrust
COMMAND f_cmd, rotor thrusts following with ḟ = (f_cmd − f)/τ_m.
The terminal cost is simply Q — no DARE or Lyapunov computation.
This is the simplest possible NMPC formulation: it works, but
provides no infinite-horizon cost guarantee at the prediction tail.
"""

import numpy as np
import casadi as ca
from scipy.linalg import block_diag
from acados_template import AcadosOcp, AcadosOcpSolver

from core.quadrotor_3d_model import create_motor_model, f_hover, F_MIN, F_MAX, NX, NXM, NU
from core.codegen import build_path


# ─────────────────────────────────────────────────────────────────
# Weights
# ─────────────────────────────────────────────────────────────────
Q = np.diag([
    80.,  80.,  120.,           # px, py, pz          position: high
    10.,  10.,   15.,           # vx, vy, vz          velocity: moderate
    10., 120.,  120.,  80.,     # qw, qx, qy, qz     attitude: scaled from Euler
     1.,   1.,    1.            # p,  q,  r           angular rate: low
])
Q_F   = np.diag([0.1, 0.1, 0.1, 0.1])    # f1..f4      thrust ACTING on the body (the old R)
R_CMD = np.diag([0.01] * NU)             # f_cmd1..4   command: light regularization only —
                                         #             what matters is the thrust it produces


# ─────────────────────────────────────────────────────────────────
# Solver Creator
# ─────────────────────────────────────────────────────────────────
def create_solver(x_ref:     np.ndarray,
                  N:         int   = 20,
                  T_horizon: float = 1.0) -> AcadosOcpSolver:
    """
    Build and compile the acados NMPC solver for 3D quadrotor (quaternion)
    with motor dynamics.

    Args:
        x_ref:      reference body state (13,)
                     [px,py,pz, vx,vy,vz, qw,qx,qy,qz, p,q,r]
                     (the thrust and command references are f_hover)
        N:          prediction horizon (number of steps)
        T_horizon:  prediction horizon duration [s]

    Cost structure (NONLINEAR_LS):
        Stage:    || [x; f; f_cmd] - [x_ref; f_hover; f_hover] ||^2_W   (21-dim)
        Terminal: || [x_N; f_N] - [x_ref; f_hover] ||^2_{W_e}           (17-dim)

        The old thrust weight R now sits on the thrust STATES f — the thrust
        that acts — and the command f_cmd only gets a small weight R_CMD (it
        keeps the Gauss-Newton Hessian positive definite in the input).

    Constraints:
        F_MIN <= f_cmd,i <= F_MAX   for each motor  (idle speed, full throttle)
        The thrust states need no bound: ḟ = (f_cmd − f)/τ_m keeps f between
        its start value and the command, i.e. inside the same envelope.

    Solver:  SQP_RTI + ERK4 (2 steps per shooting interval, for the motor lag)

    Returns:
        AcadosOcpSolver ready to be called in the simulation loop
    """
    model = create_motor_model()
    ocp   = AcadosOcp()
    ocp.model = model

    f_ref = np.full(NU, f_hover)                       # hover thrust = hover command

    # ── Cost ────────────────────────────────────────────────────
    ocp.cost.cost_type   = 'NONLINEAR_LS'
    ocp.cost.cost_type_e = 'NONLINEAR_LS'

    # cost expressions: what we penalize
    ocp.model.cost_y_expr   = ca.vertcat(model.x, model.u)  # stage:    [x; f; f_cmd] (21-dim)
    ocp.model.cost_y_expr_e = model.x                       # terminal: [x; f]        (17-dim)

    ocp.cost.W   = block_diag(Q, Q_F, R_CMD)    # stage weight    (21×21)
    ocp.cost.W_e = block_diag(Q, Q_F)           # terminal weight (17×17) — simple, no DARE

    # references
    ocp.cost.yref   = np.concatenate([x_ref, f_ref, f_ref])   # stage    (21,)
    ocp.cost.yref_e = np.concatenate([x_ref, f_ref])          # terminal (17,)

    # ── Constraints ─────────────────────────────────────────────
    ocp.constraints.x0 = np.zeros(NXM)                 # updated at runtime

    ocp.constraints.lbu   = np.full(NU, F_MIN)         # idle: ESCs keep the props spinning
    ocp.constraints.ubu   = np.full(NU, F_MAX)         # full throttle (shared envelope)
    ocp.constraints.idxbu = np.arange(NU)

    # ── Solver Options ───────────────────────────────────────────
    ocp.solver_options.N_horizon       = N
    ocp.solver_options.tf              = T_horizon
    ocp.solver_options.nlp_solver_type = 'SQP_RTI'
    ocp.solver_options.integrator_type = 'ERK'
    ocp.solver_options.sim_method_num_stages = 4
    ocp.solver_options.sim_method_num_steps  = 2       # τ_m = 30 ms < Ts = 50 ms: one RK4 step
                                                       # leaves 0.27 of a thrust step after Ts
                                                       # (exact e^(−Ts/τ_m) = 0.19); two give 0.191
    ocp.solver_options.qp_solver      = 'PARTIAL_CONDENSING_HPIPM'
    ocp.solver_options.hessian_approx  = 'GAUSS_NEWTON'
    ocp.solver_options.print_level     = 0

    ocp.code_export_directory = build_path('c_generated_code')
    return AcadosOcpSolver(ocp, json_file=build_path(f'{ocp.model.name}_ocp.json'))
