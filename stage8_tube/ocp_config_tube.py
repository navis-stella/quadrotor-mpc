"""
ocp_config_tube.py — Stage 8 Nominal MPC of the Tube Controller
===============================================================
Tube MPC = nominal MPC on the linear model + ancillary feedback on the error
(Mayne, Seron & Raković, Automatica 2005; Rawlings, Mayne & Diehl §3.5):

    nominal OCP   min  Σ_{i<N} ‖z_i‖²_Q + ‖v_i‖²_R  +  ‖z_N‖²_{P_f}
                  s.t. z_{i+1} = A z_i + B v_i,      z_0 = z (nominal state)
                       z_i ∈ X̄ = X ⊖ S              i = 1..N−1  (box on v, qv, ω)
                       MIXER v_i ∈ Ū = U ⊖ KS        i = 0..N−1  (wrench box)
                       z_Nᵀ P_f z_N ≤ α_f            X_f ⊂ X̄ (DARE ellipsoid)
    applied       u = f_hover + v₀* + K (x − z),    z ← z₁*

Coordinates and every set come from compute_tube_params (data/tube_params.npz):
z, x are 12-dim deviations from the target hover point, v, u motor-thrust
deviations. The OCP is a QP plus ONE convex quadratic constraint (the
terminal ellipsoid), solved by SQP with the exact Hessian — the constraint
Hessian 2·μ·P_f is what makes it converge in a few iterations.

The measured state x never enters the OCP: the nominal trajectory is an
exact, disturbance-free prediction of itself, so recursive feasibility is the
nominal argument (shift + v = K_f z), and x stays in z ⊕ S because the
ancillary law keeps e = x − z in the RPI set S whatever w ∈ W does. If a solve
does not converge, the controller applies exactly that shifted candidate
(`shifted_candidate`), so the guarantee never depends on the solver.

acados details:
    DISCRETE integrator, disc_dyn_expr = A z + B v   (the very (A, B) of W)
    cost_scaling = 1 on every node — acados' default scales stage costs by the
    step length, which would break V_f(z⁺) − V_f(z) ≤ −ℓ(z, K_f z).
"""

import numpy as np
import casadi as ca
from scipy.linalg import block_diag
from acados_template import AcadosOcp, AcadosOcpSolver, AcadosModel, ACADOS_INFTY

from core.quadrotor_3d_model import MIXER, NU
from stage8_tube.compute_tube_params import NE, load_tube_params
from core.codegen import build_path


# ─────────────────────────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────────────────────────
def create_nominal_linear_model(A: np.ndarray, B: np.ndarray) -> AcadosModel:
    """Discrete-time nominal model z⁺ = A z + B v (12 states, 4 inputs)."""
    z = ca.SX.sym('z', NE)
    v = ca.SX.sym('v', NU)
    model               = AcadosModel()
    model.name          = 'quadrotor_3d_tube'
    model.x             = z
    model.u             = v
    model.disc_dyn_expr = ca.mtimes(ca.DM(A), z) + ca.mtimes(ca.DM(B), v)
    return model


# ─────────────────────────────────────────────────────────────────
# Solver
# ─────────────────────────────────────────────────────────────────
def create_solver(par: dict = None, N: int = 60) -> AcadosOcpSolver:
    """
    Nominal tube-MPC solver (see the header). par = load_tube_params() by default.
    """
    par = load_tube_params() if par is None else par
    Ts  = par['Ts']

    ocp = AcadosOcp()
    ocp.model = create_nominal_linear_model(par['A'], par['B'])

    # ── cost:  y = [z; v] → 0,   terminal ‖z_N‖²_{P_f} ─────────────
    ny = NE + NU
    Vx = np.zeros((ny, NE)); Vx[:NE, :] = np.eye(NE)
    Vu = np.zeros((ny, NU)); Vu[NE:, :] = np.eye(NU)
    W  = block_diag(par['Q_cost'], par['R_cost'])
    ocp.cost.cost_type_0 = ocp.cost.cost_type = 'LINEAR_LS'
    ocp.cost.cost_type_e = 'LINEAR_LS'
    ocp.cost.Vx_0, ocp.cost.Vu_0, ocp.cost.W_0, ocp.cost.yref_0 = Vx, Vu, W, np.zeros(ny)
    ocp.cost.Vx,   ocp.cost.Vu,   ocp.cost.W,   ocp.cost.yref   = Vx, Vu, W, np.zeros(ny)
    ocp.cost.Vx_e, ocp.cost.W_e, ocp.cost.yref_e = np.eye(NE), par['P_f'], np.zeros(NE)

    # ── constraints ─────────────────────────────────────────────
    ocp.constraints.x0 = np.zeros(NE)                          # z_0 = nominal state

    idx_x = np.asarray(par['idx_x'], dtype=int)
    ocp.constraints.idxbx = idx_x                              # z_i ∈ X̄, nodes 1..N−1
    ocp.constraints.lbx   = -par['x_bar']
    ocp.constraints.ubx   =  par['x_bar']

    ocp.constraints.C  = np.zeros((NU, NE))                    # MIXER v_i ∈ Ū
    ocp.constraints.D  = MIXER
    ocp.constraints.lg = -par['nu_bar']
    ocp.constraints.ug =  par['nu_bar']

    # z_N ∈ X_f, one-sided: a lower bound 0 would be active with zero gradient
    # at the cold start z = 0, and its multiplier would enter the exact
    # Hessian as −μ·2P_f — non-convex QP, HPIPM stalls (MINSTEP).
    z = ocp.model.x
    ocp.model.con_h_expr_e = ca.mtimes([z.T, ca.DM(par['P_f']), z])
    ocp.constraints.lh_e   = np.array([-ACADOS_INFTY])
    ocp.constraints.uh_e   = np.array([par['alpha_f']])

    # ── solver ──────────────────────────────────────────────────
    so = ocp.solver_options
    so.N_horizon       = N
    so.tf              = N * Ts
    so.integrator_type = 'DISCRETE'
    so.nlp_solver_type = 'SQP'
    so.hessian_approx  = 'EXACT'
    so.qp_solver       = 'PARTIAL_CONDENSING_HPIPM'
    so.nlp_solver_max_iter = 50
    so.tol             = 1e-8
    so.cost_scaling    = np.ones(N + 1)
    so.print_level     = 0

    ocp.code_export_directory = build_path('c_generated_code_tube')
    return AcadosOcpSolver(ocp, json_file=build_path('acados_ocp_tube.json'), verbose=False)


# ─────────────────────────────────────────────────────────────────
# Runtime helpers
# ─────────────────────────────────────────────────────────────────
def solve_nominal(solver: AcadosOcpSolver, z: np.ndarray, N: int):
    """
    Solve for the nominal state z. Returns (status, Z (N+1, 12), V (N, 4)).
    """
    solver.set(0, 'lbx', z)
    solver.set(0, 'ubx', z)
    status = solver.solve()
    Z = np.array([solver.get(i, 'x') for i in range(N + 1)])
    V = np.array([solver.get(i, 'u') for i in range(N)])
    return status, Z, V


def shifted_candidate(Z: np.ndarray, V: np.ndarray, A: np.ndarray, B: np.ndarray,
                      K_f: np.ndarray):
    """
    The recursive-feasibility candidate: drop node 0, append v = K_f z_N.
    Feasible for the next OCP whenever (Z, V) was feasible (X_f invariant
    under K_f, K_f X_f ⊂ Ū, X_f ⊂ X̄).
    """
    v_N = K_f @ Z[-1]
    Z_c = np.vstack([Z[1:], A @ Z[-1] + B @ v_N])
    V_c = np.vstack([V[1:], v_N])
    return Z_c, V_c


def warm_start(solver: AcadosOcpSolver, Z: np.ndarray, V: np.ndarray) -> None:
    """Initialize every node with a (shifted) trajectory."""
    for i in range(V.shape[0]):
        solver.set(i, 'x', Z[i])
        solver.set(i, 'u', V[i])
    solver.set(V.shape[0], 'x', Z[-1])


def ancillary_input(v: np.ndarray, x: np.ndarray, z: np.ndarray, K: np.ndarray) -> np.ndarray:
    """Motor thrust deviation u = v + K (x − z)."""
    return v + K @ (x - z)
