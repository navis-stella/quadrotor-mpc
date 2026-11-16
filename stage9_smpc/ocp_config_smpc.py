"""
ocp_config_smpc.py — Stage 9 Stochastic MPC (Deterministic Equivalent)
======================================================================
Linear stochastic MPC with chance constraints, rewritten as an ordinary QP
(Kouvaritakis & Cannon, "Model Predictive Control: Classical, Robust and
Stochastic", 2016, ch. 6–7; Farina, Giulioni & Scattolini, J. Process
Control 2016):

    OCP        min  Σ_{k<N} ‖z_k‖²_Q + ‖v_k‖²_R  +  ‖z_N‖²_P
               s.t. z_{k+1} = A z_k + B v_k,     z_0 = x(t)   (measured state)
                    G z_k ≤ θ_k                   k = 1..N     (chance constraints)
                    H v_k ≤ η_k                   k = 0..N−1   (η_0 = h: hard)
    applied    u(t) = f_hover + v_0*

Sets, gain and the tightened right-hand sides θ_k, η_k come from
compute_smpc_params (data/smpc_params.npz); z, x are 12-dim deviations from
the target hover point, v motor-thrust deviations.

The difference to the tube MPC of Stage 8 is the initial condition: z_0 = x(t)
at EVERY step, so the predicted error starts at e(t|t) = 0 and θ_k, η_k depend
only on the prediction step k — the feedback gain K appears in the OCP only
through them (the predicted input is v_k + K e_k, the applied one is v_0).
The cost is the expected cost up to a constant: E{‖x‖²_Q + ‖u‖²_R} =
‖z‖²_Q + ‖v‖²_R + tr((Q + KᵀRK) Σ_e), and the trace does not depend on v.

Feasibility: w is Gaussian, i.e. unbounded — no tightening can make z_0 = x(t)
feasible for every realization (recursive feasibility needs a bounded support,
Lorenzen et al., IEEE TAC 2017, or an interpolated initial state, Köhler &
Zeilinger, IEEE L-CSS 2022). The state rows are therefore SOFT with an exact
L1 penalty: the slack is zero whenever the tightened problem is feasible and
the OCP stays solvable when a rare disturbance makes it infeasible. The input
rows stay hard.

acados details:
    DISCRETE integrator, cost_scaling = 1 on every node (as in Stage 8).
    One general-constraint block  [G 0; 0 H] [z; v] ≤ [θ_k; η_k]  with the
    right-hand side set per node (`set_tightening`); lg = −ACADOS_INFTY.
    Node 0 carries the state rows too — there z_0 is fixed, so they are
    switched off with a large bound (BIG).
    LINEAR_LS + linear constraints → a QP; SQP converges in one or two steps.
"""

import numpy as np
import casadi as ca
from scipy.linalg import block_diag
from acados_template import AcadosOcp, AcadosOcpSolver, AcadosModel, ACADOS_INFTY

from core.quadrotor_model import NU
from stage8_tube.compute_tube_params import NE
from stage9_smpc.compute_smpc_params import load_smpc_params
from core.codegen import build_path

Z_L1 = 1e4          # exact-penalty weight of the state-row slacks (≫ multipliers)
Z_L2 = 1e0          # small quadratic part: unique slack, well-conditioned QP
BIG  = 1e6          # 'no constraint' for the state rows at node 0 (< ACADOS_INFTY)


# ─────────────────────────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────────────────────────
def create_nominal_linear_model(A: np.ndarray, B: np.ndarray) -> AcadosModel:
    """Discrete-time nominal (mean) model z⁺ = A z + B v (12 states, 4 inputs)."""
    z = ca.SX.sym('z', NE)
    v = ca.SX.sym('v', NU)
    model               = AcadosModel()
    model.name          = 'quadrotor_smpc'
    model.x             = z
    model.u             = v
    model.disc_dyn_expr = ca.mtimes(ca.DM(A), z) + ca.mtimes(ca.DM(B), v)
    return model


# ─────────────────────────────────────────────────────────────────
# Solver
# ─────────────────────────────────────────────────────────────────
def create_solver(par: dict = None) -> AcadosOcpSolver:
    """
    Stochastic-MPC solver (see the header). par = load_smpc_params() by
    default; the horizon is par['N'] — the tightening was computed for it.
    """
    par = load_smpc_params() if par is None else par
    Ts, N = par['Ts'], int(par['N'])
    G, H  = par['G'], par['H']
    n_g, n_h = G.shape[0], H.shape[0]

    ocp = AcadosOcp()
    ocp.model = create_nominal_linear_model(par['A'], par['B'])

    # ── cost:  y = [z; v] → 0,   terminal ‖z_N‖²_P ───────────────
    ny = NE + NU
    Vx = np.zeros((ny, NE)); Vx[:NE, :] = np.eye(NE)
    Vu = np.zeros((ny, NU)); Vu[NE:, :] = np.eye(NU)
    W  = block_diag(par['Q_cost'], par['R_cost'])
    ocp.cost.cost_type_0 = ocp.cost.cost_type = 'LINEAR_LS'
    ocp.cost.cost_type_e = 'LINEAR_LS'
    ocp.cost.Vx_0, ocp.cost.Vu_0, ocp.cost.W_0, ocp.cost.yref_0 = Vx, Vu, W, np.zeros(ny)
    ocp.cost.Vx,   ocp.cost.Vu,   ocp.cost.W,   ocp.cost.yref   = Vx, Vu, W, np.zeros(ny)
    ocp.cost.Vx_e, ocp.cost.W_e, ocp.cost.yref_e = np.eye(NE), par['P'], np.zeros(NE)

    # ── constraints ─────────────────────────────────────────────
    ocp.constraints.x0 = np.zeros(NE)                          # z_0 = x(t), set per solve

    #   [G 0; 0 H] [z; v] ≤ [θ_k; η_k]   — right-hand sides per node below
    ocp.constraints.C  = np.vstack([G, np.zeros((n_h, NE))])
    ocp.constraints.D  = np.vstack([np.zeros((n_g, NU)), H])
    ocp.constraints.lg = np.full(n_g + n_h, -ACADOS_INFTY)
    ocp.constraints.ug = np.concatenate([par['g'], par['h']])

    ocp.constraints.C_e  = G                                   # G z_N ≤ θ_N
    ocp.constraints.lg_e = np.full(n_g, -ACADOS_INFTY)
    ocp.constraints.ug_e = par['theta'][N]

    # ── soft state rows (exact penalty), hard input rows ────────
    ocp.constraints.idxsg   = np.arange(n_g)
    ocp.constraints.idxsg_e = np.arange(n_g)
    ocp.cost.zl = ocp.cost.zu = np.full(n_g, Z_L1)
    ocp.cost.Zl = ocp.cost.Zu = np.full(n_g, Z_L2)
    ocp.cost.zl_e = ocp.cost.zu_e = np.full(n_g, Z_L1)
    ocp.cost.Zl_e = ocp.cost.Zu_e = np.full(n_g, Z_L2)

    # ── solver ──────────────────────────────────────────────────
    so = ocp.solver_options
    so.N_horizon       = N
    so.tf              = N * Ts
    so.integrator_type = 'DISCRETE'
    so.nlp_solver_type = 'SQP'
    so.hessian_approx  = 'GAUSS_NEWTON'                        # exact: the OCP is a QP
    so.qp_solver       = 'PARTIAL_CONDENSING_HPIPM'
    so.nlp_solver_max_iter = 20
    so.tol             = 1e-8
    so.cost_scaling    = np.ones(N + 1)
    so.print_level     = 0

    ocp.code_export_directory = build_path('c_generated_code_smpc')
    solver = AcadosOcpSolver(ocp, json_file=build_path('acados_ocp_smpc.json'), verbose=False)
    set_tightening(solver, par)
    return solver


def set_tightening(solver: AcadosOcpSolver, par: dict) -> None:
    """
    Write θ_k, η_k into the nodes 0..N−1 (the terminal θ_N is set at build).
    Node 0: state rows off (z_0 is the measurement), input rows η_0 = h.
    """
    n_g = par['G'].shape[0]
    for k in range(int(par['N'])):
        theta_k = par['theta'][k] if k > 0 else np.full(n_g, BIG)
        solver.constraints_set(k, 'ug', np.concatenate([theta_k, par['eta'][k]]))


# ─────────────────────────────────────────────────────────────────
# Runtime helpers
# ─────────────────────────────────────────────────────────────────
def solve(solver: AcadosOcpSolver, x: np.ndarray, N: int):
    """
    Solve for the measured state x (12-dim deviation).
    Returns (status, Z (N+1, 12), V (N, 4), largest state-row slack).
    """
    solver.set(0, 'lbx', x)
    solver.set(0, 'ubx', x)
    status = solver.solve()
    Z = np.array([solver.get(i, 'x') for i in range(N + 1)])
    V = np.array([solver.get(i, 'u') for i in range(N)])
    slack = max(np.max(solver.get(i, 'su')) for i in range(1, N + 1))
    return status, Z, V, slack


def shifted_plan(Z: np.ndarray, V: np.ndarray, A: np.ndarray, B: np.ndarray,
                 K: np.ndarray):
    """Drop node 0, append v = K z_N — warm start, and the plan used if a solve fails."""
    v_N = K @ Z[-1]
    return np.vstack([Z[1:], A @ Z[-1] + B @ v_N]), np.vstack([V[1:], v_N])


def warm_start(solver: AcadosOcpSolver, Z: np.ndarray, V: np.ndarray) -> None:
    """Initialize every node with a (shifted) trajectory."""
    for i in range(V.shape[0]):
        solver.set(i, 'x', Z[i])
        solver.set(i, 'u', V[i])
    solver.set(V.shape[0], 'x', Z[-1])
