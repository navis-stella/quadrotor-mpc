"""
quadrotor_model.py — All Stages Full 3D Quadrotor Model (Quaternion)
====================================================================
Single source of the physics for every stage. Contains:
  - Physical parameters, dimensions, motor mixer, operating envelope
  - Quaternion utilities                (numpy + CasADi)
  - Nominal model                       (Stage 1–2 MPC prediction model)
  - Disturbance model                   (Stage 3–4 MPC prediction model,
                                          plant of every stage)
  - Augmented model z = [x; d]          (EKF: CasADi functions,
                                          MHE: acados model with process noise)
  - Measurement model y = [p; q; ω]     (sensor, EKF, MHE, observability)
  - Hover point + linearization         (QIH / DARE terminal cost, observability)
  - AcadosSimSolver plant               (one plant for every stage)
  - Tracking model x̃ = [x; p_a]          (Stage 5 tracking MPC: artificial
                                          reference as constant extra state)
  - Motor dynamics                      (thrust COMMAND as input: MPC model
                                          [x; f] with ḟ = (f_cmd − f)/τ_m,
                                          plant [x; Ω] with rotor-speed loop
                                          and f = k_f Ω²; check: core/motor_check)

State (13):
    x = [px, py, pz, vx, vy, vz, qw, qx, qy, qz, p, q, r]

    Position:            px, py, pz        [m]      world frame
    Linear velocity:     vx, vy, vz        [m/s]    world frame
    Quaternion:          qw, qx, qy, qz    [-]      scalar-first, body-to-world
    Angular velocity:    p, q, r           [rad/s]  body frame (roll, pitch, yaw rate)

    Note on naming: 'q' as pitch rate (aerospace standard) is separate from
    'qw, qx, qy, qz' quaternion components. Context makes them unambiguous.

Input (4):
    u = [f1, f2, f3, f4]   individual motor thrusts [N]

Motor layout (+ configuration, top view, z-up):

          M1 (front)
           |
    M4 ----+---- M2 (right)
    (left) |
          M3 (back)

    M1 at [ L,  0, 0]   front     spin CW  → reactive tau_z < 0
    M2 at [ 0, -L, 0]   right     spin CCW → reactive tau_z > 0
    M3 at [-L,  0, 0]   back      spin CW  → reactive tau_z < 0
    M4 at [ 0,  L, 0]   left      spin CCW → reactive tau_z > 0

Quaternion convention:
    q = [qw, qx, qy, qz]   scalar-first
    ||q|| = 1              unit norm constraint
    Hover reference: q = [1, 0, 0, 0] (identity rotation)
    q and -q represent the same rotation (double cover)

    Rotation matrix R(q) transforms body → world:
        v_world = R(q) @ v_body

Disturbance (Stage 3 onward):
    d = [d_fx, d_fy, d_fz, d_tx, d_ty, d_tz]

    d_fx, d_fy, d_fz   [N]     constant force disturbance (world frame)
                                 covers: wind, mass error (Δm·g in z)
    d_tx, d_ty, d_tz   [N·m]   constant torque disturbance (body frame)
                                 covers: CoG offset (parasitic torques)

    Where d lives in each component:
        MPC prediction  create_disturbance_model()   p = d̂        (parameter)
        Plant           create_disturbance_plant()   p = d_true   (parameter)
        EKF             get_augmented_dynamics()     z = [x; d], ḋ = 0
        MHE             create_mhe_model()           z = [x; d], ż = f_aug + w

Measurement (Stage 4):
    y = [px, py, pz, qw, qx, qy, qz, p, q, r]   (NY = 10)
    → position, quaternion, angular velocity directly measured
    → velocity v and disturbance d inferred by the estimator
    h(z) is LINEAR ⇒ H = ∂h/∂z is a constant selection matrix (IDX_Y).
"""

import numpy as np
import casadi as ca
from acados_template import AcadosModel, AcadosSim, AcadosSimSolver

from core.codegen import build_path

# ─────────────────────────────────────────────────────────────────
# Physical Parameters
# ─────────────────────────────────────────────────────────────────
m     = 1.0       # [kg]     total mass
g     = 9.81      # [m/s²]   gravity
L     = 0.175     # [m]      arm length (center to motor)
Ixx   = 0.01      # [kg·m²]  moment of inertia about x (roll)
Iyy   = 0.01      # [kg·m²]  moment of inertia about y (pitch)
Izz   = 0.02      # [kg·m²]  moment of inertia about z (yaw)
c_tau = 0.0036    # [m]      thrust-to-reactive-torque coefficient

f_hover = m * g / 4.0    # [N] hover thrust per motor ≈ 2.45 N

# ─────────────────────────────────────────────────────────────────
# Dimensions
# ─────────────────────────────────────────────────────────────────
NX = 13        # 3 pos + 3 vel + 4 quat + 3 angular vel
NU = 4         # motor thrusts
ND = 6         # 3 force (world) + 3 torque (body)
NZ = NX + ND   # 19 — augmented state z = [x; d]   (EKF and MHE)
NY = 10        # 3 pos + 4 quat + 3 angular vel  —  y = [p; q; ω]
NW = NZ        # 19 — MHE process noise, additive on every derivative of z
NM = 4         # motor states (thrusts f in the MPC model, rotor speeds Ω in the plant)
NXM = NX + NM  # 17 — motor-dynamics state [x; f]  (plant: [x; Ω])

IDX_Y = np.array([0, 1, 2, 6, 7, 8, 9, 10, 11, 12])   # y = z[IDX_Y]

# ─────────────────────────────────────────────────────────────────
# Motor Mixer and Operating Envelope
# ─────────────────────────────────────────────────────────────────
#   [T_total]     [  1       1       1       1    ] [f1]
#   [tau_x  ]  =  [  0      -L       0       L    ] [f2]
#   [tau_y  ]     [ -L       0       L       0    ] [f3]
#   [tau_z  ]     [-c_tau   c_tau  -c_tau   c_tau ] [f4]
#
#   Used symbolically by the dynamics and inverted by ss_target.
MIXER = np.array([
    [ 1.0,    1.0,     1.0,    1.0   ],
    [ 0.0,   -L,       0.0,    L     ],
    [-L,      0.0,     L,      0.0   ],
    [-c_tau,  c_tau,  -c_tau,  c_tau ],
])

F_MAX     = 3.0 * f_hover   # [N]     motor thrust limit (MPC bound, ss_target clip)
OMEGA_MAX = 2.0             # [rad/s] certified body-rate envelope of the MHE
                            #         (detectability_check); the MPC keeps
                            #         |ωᵢ| ≤ 0.9·OMEGA_MAX so the plant stays in it

# ─────────────────────────────────────────────────────────────────
# Rotor / Motor Parameters  (motor-dynamics models, see the end of the file)
# ─────────────────────────────────────────────────────────────────
#   Chain on a real drone:  f_cmd → ω_cmd = √(f_cmd / k_f) → ESC speed loop → Ω → f = k_f Ω²
#   The speed loop is modeled as a first-order lag Ω̇ = (ω_cmd − Ω) / τ_m.
#   Ω (capital) = rotor speed, to keep it apart from the body rates ω = [p, q, r].
TAU_MOTOR         = 0.03                         # [s]       closed speed-loop time constant
OMEGA_ROTOR_MAX   = 1300.0                       # [rad/s]   full throttle (≈ 12 400 rpm)
OMEGA_ROTOR_MIN   = 0.1 * OMEGA_ROTOR_MAX        # [rad/s]   idle — ESCs never stop the props in flight
K_F               = F_MAX / OMEGA_ROTOR_MAX**2   # [N·s²]    thrust coefficient: k_f Ω_max² = F_MAX
K_Q               = c_tau * K_F                  # [N·m·s²]  drag-torque coefficient: k_q Ω² = c_tau·f
F_MIN             = K_F * OMEGA_ROTOR_MIN**2     # [N]       idle thrust = F_MAX / 100 (input lower bound)
OMEGA_ROTOR_HOVER = np.sqrt(f_hover / K_F)       # [rad/s]   ≈ 751 rad/s (≈ 7 200 rpm)


# ─────────────────────────────────────────────────────────────────
# Quaternion Utilities  (numpy)
# ─────────────────────────────────────────────────────────────────
def quat_to_euler(q: np.ndarray) -> np.ndarray:
    """
    Convert quaternion(s) to Euler angles (ZYX convention).

    Args:
        q: (4,) or (N, 4) array  [qw, qx, qy, qz]

    Returns:
        euler: (3,) or (N, 3) array  [roll, pitch, yaw] in radians
    """
    if q.ndim == 1:
        q = q.reshape(1, -1)
        squeeze = True
    else:
        squeeze = False

    qw, qx, qy, qz = q[:, 0], q[:, 1], q[:, 2], q[:, 3]

    # Roll (φ)
    sinr_cosp = 2.0 * (qw * qx + qy * qz)
    cosr_cosp = 1.0 - 2.0 * (qx**2 + qy**2)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    # Pitch (θ)
    sinp = 2.0 * (qw * qy - qz * qx)
    sinp = np.clip(sinp, -1.0, 1.0)
    pitch = np.arcsin(sinp)

    # Yaw (ψ)
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy**2 + qz**2)
    yaw = np.arctan2(siny_cosp, cosy_cosp)

    euler = np.column_stack([roll, pitch, yaw])
    return euler.squeeze() if squeeze else euler


def quat_normalize(x: np.ndarray) -> np.ndarray:
    """
    Normalize the quaternion slots x[6:10] of a state (13) or augmented
    state (19) in place and return x. Call after each integration step to
    prevent drift from ||q|| = 1.

    Also enforces qw > 0 convention to avoid the double-cover ambiguity
    (q and -q represent the same rotation — we pick the qw > 0 hemisphere).
    """
    q_norm = np.linalg.norm(x[6:10])
    if q_norm > 1e-12:
        x[6:10] /= q_norm
    if x[6] < 0:
        x[6:10] *= -1
    return x


# ─────────────────────────────────────────────────────────────────
# Quaternion Utilities  (CasADi — used by the MHE cost)
# ─────────────────────────────────────────────────────────────────
def quat_multiply_casadi(q1, q2):
    """
    Hamilton product  q1 ⊗ q2  for scalar-first quaternions (CasADi SX/MX).
    Convention: q = [qw, qx, qy, qz].
    """
    w = q1[0]*q2[0] - q1[1]*q2[1] - q1[2]*q2[2] - q1[3]*q2[3]
    x = q1[0]*q2[1] + q1[1]*q2[0] + q1[2]*q2[3] - q1[3]*q2[2]
    y = q1[0]*q2[2] - q1[1]*q2[3] + q1[2]*q2[0] + q1[3]*q2[1]
    z = q1[0]*q2[3] + q1[1]*q2[2] - q1[2]*q2[1] + q1[3]*q2[0]
    return ca.vertcat(w, x, y, z)


def quat_error_casadi(q_hat, q_ref):
    """
    Error quaternion  q_err = q_hat ⊗ q_ref⁻¹  (CasADi symbolic).

    For unit inputs, q_err → ±[1, 0, 0, 0] when q_hat and q_ref represent
    the same rotation (double cover of SO(3)). The MHE cost combines this
    with  fabs(q_err) − [1, 0, 0, 0]  so the term vanishes for both signs
    and stays smooth.
    """
    q_ref_inv = ca.vertcat(q_ref[0], -q_ref[1], -q_ref[2], -q_ref[3])
    return quat_multiply_casadi(q_hat, q_ref_inv)


# ─────────────────────────────────────────────────────────────────
# Core Symbolic Dynamics  (shared by every model variant)
# ─────────────────────────────────────────────────────────────────
def _build_core_symbols():
    """
    Shared CasADi symbols: state x (13), input u (4) and the thrust /
    torque wrench [T_total, tau_x, tau_y, tau_z] = MIXER @ u.
    """
    x = ca.SX.sym('x', NX)
    u = ca.SX.sym('u', NU)
    wrench = ca.mtimes(ca.DM(MIXER), u)
    return {'x': x, 'u': u, 'T_total': wrench[0], 'tau': wrench[1:4]}


def _build_f_expl(s, d=None):
    """
    Explicit ODE  ẋ = f(x, u, d)  from the core symbols.

    d = None gives the nominal drift (d = 0); otherwise d is a 6-vector
    (symbolic parameter or state) injected as:

        d_fx, d_fy, d_fz  →  translational dynamics (world frame)
            v̇ += d_f / m
        d_tx, d_ty, d_tz  →  rotational dynamics (body frame)
            ω̇ += I⁻¹ d_τ
    """
    x, T_total, tau = s['x'], s['T_total'], s['tau']
    vx, vy, vz     = x[3], x[4], x[5]
    qw, qx, qy, qz = x[6], x[7], x[8], x[9]
    p,  q,  r      = x[10], x[11], x[12]
    if d is None:
        d = ca.SX.zeros(ND)

    # ── Subsystem 1: Translational dynamics (world frame) ──────
    #    Only the third column of R(q) is needed for thrust direction.
    dvx = T_total / m * 2 * (qx*qz + qw*qy)            + d[0] / m
    dvy = T_total / m * 2 * (qy*qz - qw*qx)            + d[1] / m
    dvz = T_total / m * (1 - 2*(qx**2 + qy**2)) - g    + d[2] / m

    # ── Subsystem 2: Quaternion kinematics ─────────────────────
    #    q_dot = 0.5 * q ⊗ [0, p, q, r]    (no disturbance here)
    dqw = 0.5 * (-qx*p - qy*q - qz*r)
    dqx = 0.5 * ( qw*p + qy*r - qz*q)
    dqy = 0.5 * ( qw*q - qx*r + qz*p)
    dqz = 0.5 * ( qw*r + qx*q - qy*p)

    # ── Subsystem 3: Rotational dynamics (Euler's equations) ───
    dp_dt = (Iyy - Izz) / Ixx * q * r + tau[0] / Ixx     + d[3] / Ixx
    dq_dt = (Izz - Ixx) / Iyy * p * r + tau[1] / Iyy     + d[4] / Iyy
    dr_dt = (Ixx - Iyy) / Izz * p * q + tau[2] / Izz     + d[5] / Izz

    return ca.vertcat(
        vx, vy, vz,                   # position kinematics
        dvx, dvy, dvz,                # translational dynamics
        dqw, dqx, dqy, dqz,           # quaternion kinematics
        dp_dt, dq_dt, dr_dt           # rotational dynamics
    )


def _build_augmented():
    """
    Augmented dynamics shared by the EKF and the MHE:

        z = [x(13); d(6)],     ż = f_aug(z, u) = [ f(x, u, d) ; 0 ]

    ḋ = 0 is the constant-disturbance assumption. Both estimators
    discretize exactly this expression — the EKF by RK4 + Euler-linearized
    covariance, the MHE by acados ERK with process noise w added.
    """
    s = _build_core_symbols()
    d = ca.SX.sym('d', ND)
    z = ca.vertcat(s['x'], d)
    z_dot = ca.vertcat(_build_f_expl(s, d), ca.SX.zeros(ND))
    return z, s['u'], z_dot


# ═════════════════════════════════════════════════════════════════
# Nominal Model  (Stage 1–2 MPC prediction model)
# ═════════════════════════════════════════════════════════════════
def create_nominal_model() -> AcadosModel:
    """
    Nominal quadrotor dynamics ẋ = f(x, u), d = 0, no parameters.

    Used by the Stage 1–2 OCPs (ocp_config_basic / _qih / _dare) and by the
    QIH Lipschitz check (compute_qih_params). Same expression as the
    disturbance model with d = 0, so the plant of those stages is simply
    create_disturbance_plant() with p left at zero — no model mismatch.
    """
    s = _build_core_symbols()

    model             = AcadosModel()
    model.name        = 'quadrotor'
    model.x           = s['x']
    model.u           = s['u']
    model.xdot        = ca.SX.sym('xdot', NX)
    model.f_expl_expr = _build_f_expl(s)

    return model


# ═════════════════════════════════════════════════════════════════
# Disturbance as Runtime Parameter  (MPC prediction model + plant)
# ═════════════════════════════════════════════════════════════════
def create_disturbance_model() -> AcadosModel:
    """
    Quadrotor dynamics with disturbance forces/torques as runtime parameters.

    Used by:
      - MPC solver:  set p = d̂  (estimate from EKF / MHE) at each shooting node
      - Plant sim:   set p = d_true (known external disturbance for testing)

    The state dimension stays NX = 13 — disturbances are NOT optimized.

    Parameter vector:
        p = [d_fx, d_fy, d_fz, d_tx, d_ty, d_tz]
    """
    s = _build_core_symbols()
    d = ca.SX.sym('d', ND)

    model             = AcadosModel()
    model.name        = 'quadrotor_disturb'
    model.x           = s['x']
    model.u           = s['u']
    model.p           = d
    model.xdot        = ca.SX.sym('xdot', NX)
    model.f_expl_expr = _build_f_expl(s, d)

    return model


# ═════════════════════════════════════════════════════════════════
# Augmented Model  (Stage 4 estimators)
# ═════════════════════════════════════════════════════════════════
def get_augmented_dynamics():
    """
    Augmented dynamics as CasADi functions for the EKF.

    Returns:
        f_aug:  CasADi Function  ż = f_aug(z, u)        (19,)
        F_func: CasADi Function  ∂f_aug/∂z at (z, u)    (19×19)

    Usage in EKF:
        z_dot_val = f_aug(z_hat, u)                    # prediction
        F_val     = F_func(z_hat, u)                   # Jacobian for covariance
    """
    z, u, z_dot = _build_augmented()
    f_aug  = ca.Function('f_aug', [z, u], [z_dot], ['z', 'u'], ['z_dot'])
    F_func = ca.Function('F_aug', [z, u], [ca.jacobian(z_dot, z)],
                         ['z', 'u'], ['F'])
    return f_aug, F_func


def create_mhe_model() -> AcadosModel:
    """
    Augmented dynamics as an acados model for the Lyapunov MHE.

        ż = f_aug(z, u_rotor) + w,     z ∈ R^19,  w ∈ R^19 (NW)

    The MHE optimizes over z₀ and the process noise w, so the acados roles
    are swapped with respect to the MPC:

        model.x = z = [x(13); d(6)]
        model.u = w         process noise — the MHE's decision variable
        model.p = u_rotor   known rotor thrusts, set from the control history
                            (ocp_config_mhe augments model.p further with the
                             discount exponent, node flags, prior and y_meas)

    w_d makes d a random walk: the MHE pays 2‖w_d‖²_Q for every change of d̂,
    and a constant d costs nothing once it has been learned.

    Why this is certifiable (detectability_check): d enters f LINEARLY with
    constant coefficients (d_f/m in v̇, I⁻¹d_τ in ω̇), so ∂f/∂z depends only on
    (q, ω, T_total) — the vertex reduction of the LMI applies.
    """
    z, u, z_dot = _build_augmented()
    w = ca.SX.sym('w', NW)

    model             = AcadosModel()
    model.name        = 'quadrotor_mhe'
    model.x           = z
    model.u           = w              # MHE 'input' = process noise
    model.p           = u              # rotor thrusts as known parameter
    model.xdot        = ca.SX.sym('zdot', NZ)
    model.f_expl_expr = z_dot + w

    return model


# ═════════════════════════════════════════════════════════════════
# Measurement Model  (Stage 4)
# ═════════════════════════════════════════════════════════════════
def measurement_expr(z_sym):
    """
    y = h(z) = [p; q; ω] as a CasADi expression — the selection z[IDX_Y].

    Works for the 13-dim state and the 19-dim augmented state (only the
    first 13 entries are read; d is not measured). Used inside the MHE cost
    and the detectability LMI.
    """
    return ca.vertcat(*[z_sym[int(i)] for i in IDX_Y])


def get_measurement_function():
    """
    Measurement function for the augmented state z (19).

    Stage 4 assumption: position p, quaternion q, and angular velocity ω
    are directly measured (idealized outer pose source, mocap-equivalent).
    Linear velocity v and disturbances d are unmeasured — inferred through
    the model's kinematic and dynamic coupling.

    Returns:
        h_func:  CasADi Function  y = h(z),  z (19,) → y (10,)
        H_z:     (10, 19) constant Jacobian ∂h/∂z  (EKF gain, observability)
    """
    z = ca.SX.sym('z', NZ)
    h_func = ca.Function('h_meas', [z], [measurement_expr(z)], ['z'], ['y'])
    H_z = np.eye(NZ)[IDX_Y]
    return h_func, H_z


def project_measurement(y: np.ndarray) -> np.ndarray:
    """
    Best guess of z from a single measurement — the common initial prior
    z̄₀ of both estimators:

        p, q, ω ← measured,    v ← 0,    d ← 0
    """
    z = np.zeros(NZ)
    z[IDX_Y] = y
    return z


# ─────────────────────────────────────────────────────────────────
# Hover Point + Linearization
# ─────────────────────────────────────────────────────────────────
def hover_point():
    """(z_hover (19), u_hover (4)) — rest at the origin, level, d = 0."""
    z_hover = np.zeros(NZ)
    z_hover[6] = 1.0                    # qw = 1 (identity quaternion)
    return z_hover, np.full(NU, f_hover)


def get_hover_linearization():
    """
    Continuous-time linearization of the nominal dynamics (d = 0) at hover,
    differentiated from the same CasADi model the solvers use.

    Key couplings at hover (quaternion version):
        qy → acceleration in x    (A[3,8]  =  2g)
        qx → acceleration in -y   (A[4,7]  = -2g)
    The factor 2 (vs g for Euler angles) comes from R(q): small-angle
    φ ≈ 2 qx, θ ≈ 2 qy.

    Returns:
        A_c:  (13×13) state Jacobian at hover
        B_c:  (13×4)  input Jacobian at hover
    """
    s = _build_core_symbols()
    f = _build_f_expl(s)
    jac = ca.Function('jac_hover', [s['x'], s['u']],
                      [ca.jacobian(f, s['x']), ca.jacobian(f, s['u'])])
    z_hover, u_hover = hover_point()
    A_c, B_c = jac(z_hover[:NX], u_hover)
    return A_c.full(), B_c.full()


# ─────────────────────────────────────────────────────────────────
# AcadosSim Plant
# ─────────────────────────────────────────────────────────────────
def create_disturbance_plant(Ts: float) -> AcadosSimSolver:
    """
    AcadosSimSolver plant WITH disturbance: one call advances by Ts.

    Usage:
        plant.set('x', x_current)
        plant.set('u', u_current)
        plant.set('p', d_true)          # ← set true disturbance
        plant.solve()
        x_next = quat_normalize(plant.get('x'))

    The disturbance is integrated continuously within each step,
    not applied as a discrete impulse — physically correct.

    Stages 1–2 never call plant.set('p', ...): p stays at its default
    zero and this is the nominal plant of create_nominal_model().
    """
    sim = AcadosSim()
    sim.model = create_disturbance_model()
    sim.solver_options.T = Ts
    sim.solver_options.integrator_type = 'ERK'
    sim.solver_options.num_stages = 4
    sim.parameter_values = np.zeros(ND)     # dimension must be known
    sim.code_export_directory = build_path('c_generated_code')
    return AcadosSimSolver(sim, json_file=build_path(f'{sim.model.name}_sim.json'))


# ═════════════════════════════════════════════════════════════════
# Tracking Model  (Stage 5 — artificial reference)
# ═════════════════════════════════════════════════════════════════
NA  = 3          # artificial reference y_a = p_a (hover position; yaw fixed at 0)
NXA = NX + NA    # 16 — tracking-MPC state x̃ = [x; p_a]

# Terminal equality x_N = x_a(p_a) is imposed on these 12 entries of x.
# qw is left out: the dynamics keep ‖q‖ = 1, so with qx = qy = qz = 0
# fixed, qw = 1 follows. Constraining all 4 quaternion entries would give
# 4 equality rows with only rank 3 → LICQ fails and HPIPM struggles.
IDX_TERM_EQ = np.array([0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12])


def steady_state_map(p_a: np.ndarray):
    """
    Steady-state mapping g(y_a) of the tracking MPC (parametrizes the
    steady states by y_a, as in Limon et al., IEEE TAC 2018), nominal d = 0:

        y_a = p_a  ↦  x_a = [p_a, v = 0, q = [1,0,0,0], ω = 0],   u_a = f_hover·1

    Every hover point is a steady state; f_hover = F_MAX / 3 lies strictly
    inside [0, F_MAX], so (x_a, u_a) is a STRICTLY feasible steady state.
    g is affine in p_a → Lipschitz with L_g = 1.

    One definition for both uses: a NumPy p_a gives arrays (simulation,
    plots), a CasADi p_a gives expressions (the tracking-MPC cost), so the
    OCP and the simulation can never disagree on g.

    Returns:
        x_a (13,), u_a (4,)    — ndarray, or ca.SX/MX/DM for a CasADi p_a
    """
    if isinstance(p_a, (ca.SX, ca.MX)):
        x_a = ca.vertcat(p_a, ca.DM.zeros(3), ca.DM([1, 0, 0, 0]), ca.DM.zeros(3))
        return x_a, ca.DM(np.full(NU, f_hover))
    x_a = np.zeros(NX)
    x_a[0:3] = p_a
    x_a[6]   = 1.0
    return x_a, np.full(NU, f_hover)


def create_tracking_model() -> AcadosModel:
    """
    Nominal dynamics augmented with the artificial reference as a state:

        x̃ = [x (13); p_a (3)],     ẋ = f(x, u),     ṗ_a = 0

    acados has no 'global' decision variable shared by all shooting nodes.
    Making p_a a state with zero dynamics gives exactly that: its value at
    node 0 is free (only x is fixed by the initial condition) and the
    dynamics copy it unchanged to every node — one decision variable p_a.

    The reference r is NOT part of the model; the OCP adds it as parameter
    (it only enters the offset cost V_o, never a constraint).
    """
    s   = _build_core_symbols()
    p_a = ca.SX.sym('p_a', NA)

    model             = AcadosModel()
    model.name        = 'quadrotor_tracking'
    model.x           = ca.vertcat(s['x'], p_a)
    model.u           = s['u']
    model.xdot        = ca.SX.sym('xdot', NXA)
    model.f_expl_expr = ca.vertcat(_build_f_expl(s), ca.SX.zeros(NA))

    return model


# ═════════════════════════════════════════════════════════════════
# Motor Dynamics  (thrust command as input, rotor lag as states)
# ═════════════════════════════════════════════════════════════════
# The models above take the thrusts f as the input: the rotors would follow
# a command instantly. On a real drone the controller sends a thrust command
# f_cmd, the ESC tracks the matching rotor speed, and the thrust follows
# with the speed-loop lag. Two views of that lag:
#
#   Plant  (create_rotor_plant)   rotor speeds Ω as states — the physics:
#       ω_cmd = √(clip(f_cmd, F_MIN, F_MAX) / k_f)      inverse thrust map
#       Ω̇     = (ω_cmd − Ω) / τ_m                       closed speed loop
#       f     = k_f Ω²                                  thrust on the body
#
#   MPC model  (create_thrust_lag_model)   thrusts f as states — what the MPC sees:
#       ḟ = (f_cmd − f) / τ_m
#
# Both agree to first order: ḟ = 2 k_f Ω Ω̇ = (f_cmd − f) · 2Ω / (ω_cmd + Ω),
# and the factor 2Ω / (ω_cmd + Ω) is 1 for small steps. Large steps make
# the plant slower when spinning up (ω_cmd > Ω: factor < 1) and faster when
# spinning down (factor > 1) — a real, small plant–model mismatch, as on
# hardware (core/motor_check: t63 = 35 ms to full throttle, 18 ms to idle,
# 30 ms for small steps).
#
# Full state feedback measures Ω (ESC rpm telemetry) and hands the MPC
# f = k_f Ω²  (rotor_to_thrust_state).

def thrust_from_rotor(Omega):
    """f = k_f Ω²  (NumPy or CasADi)."""
    return K_F * Omega**2


def rotor_from_thrust(f):
    """Ω = √(f / k_f) for f ≥ 0  (NumPy)."""
    return np.sqrt(np.maximum(f, 0.0) / K_F)


def rotor_to_thrust_state(x_plant: np.ndarray) -> np.ndarray:
    """[x (13); Ω (4)]  →  [x (13); f (4)]  — rotor speeds to thrusts."""
    x_model = np.array(x_plant, dtype=float, copy=True)
    x_model[NX:NXM] = thrust_from_rotor(x_plant[NX:NXM])
    return x_model


def thrust_to_rotor_state(x_model: np.ndarray) -> np.ndarray:
    """[x (13); f (4)]  →  [x (13); Ω (4)]  — e.g. to start the plant at a model state."""
    x_plant = np.array(x_model, dtype=float, copy=True)
    x_plant[NX:NXM] = rotor_from_thrust(x_model[NX:NXM])
    return x_plant


def _build_thrust_lag_symbols():
    """
    Core symbols of the motor-dynamics MPC model: state [x (13); f (4)],
    input u = f_cmd (4). The wrench comes from the thrust STATES f, so
    _build_f_expl (which reads only x[0:13], T_total and tau) is reused as is.
    """
    x  = ca.SX.sym('x', NXM)
    u  = ca.SX.sym('f_cmd', NU)
    wrench = ca.mtimes(ca.DM(MIXER), x[NX:NXM])
    return {'x': x, 'u': u, 'T_total': wrench[0], 'tau': wrench[1:4]}


def _build_thrust_lag_f_expl(s, d=None):
    """[ẋ; ḟ] = [f(x, f, d); (f_cmd − f) / τ_m]."""
    f_state = s['x'][NX:NXM]
    return ca.vertcat(_build_f_expl(s, d), (s['u'] - f_state) / TAU_MOTOR)


def create_thrust_lag_model() -> AcadosModel:
    """
    Motor-dynamics MPC model, nominal (d = 0), no parameters:

        state  [x (13); f (4)] = 17,    input  f_cmd (4)

    Counterpart of create_nominal_model(). Input bounds for the OCP:
    F_MIN ≤ f_cmd ≤ F_MAX (idle speed > 0, full throttle).
    """
    s = _build_thrust_lag_symbols()

    model             = AcadosModel()
    model.name        = 'quadrotor_thrust_lag'
    model.x           = s['x']
    model.u           = s['u']
    model.xdot        = ca.SX.sym('xdot', NXM)
    model.f_expl_expr = _build_thrust_lag_f_expl(s)

    return model


def create_thrust_lag_disturbance_model() -> AcadosModel:
    """
    Motor-dynamics MPC model with the disturbance d (6) as runtime parameter
    p — counterpart of create_disturbance_model(). The disturbance acts on
    the body (forces, torques), not on the motors.
    """
    s = _build_thrust_lag_symbols()
    d = ca.SX.sym('d', ND)

    model             = AcadosModel()
    model.name        = 'quadrotor_thrust_lag_disturb'
    model.x           = s['x']
    model.u           = s['u']
    model.p           = d
    model.xdot        = ca.SX.sym('xdot', NXM)
    model.f_expl_expr = _build_thrust_lag_f_expl(s, d)

    return model


def thrust_lag_hover_point():
    """(x_hover (17), u_hover (4)) of the MPC model: level hover, f = f_cmd = f_hover."""
    x_hover = np.zeros(NXM)
    x_hover[6]      = 1.0
    x_hover[NX:NXM] = f_hover
    return x_hover, np.full(NU, f_hover)


def get_thrust_lag_hover_linearization():
    """
    Continuous-time linearization of the motor MPC model (d = 0) at hover.

    Structure:   A = [A_body  B_body ]     B = [   0    ]
                     [  0    −I/τ_m  ]         [ I/τ_m  ]
    (A_body, B_body = get_hover_linearization(): the old input matrix now
    couples the thrust STATES into the body; the input reaches the body only
    through the lag.)

    Returns:
        A_c (17×17), B_c (17×4)
    """
    s = _build_thrust_lag_symbols()
    f = _build_thrust_lag_f_expl(s)
    jac = ca.Function('jac_motor_hover', [s['x'], s['u']],
                      [ca.jacobian(f, s['x']), ca.jacobian(f, s['u'])])
    x_hover, u_hover = thrust_lag_hover_point()
    A_c, B_c = jac(x_hover, u_hover)
    return A_c.full(), B_c.full()


def _build_rotor_plant_model() -> AcadosModel:
    """Plant ODE on [x (13); Ω (4)] with f_cmd as input and p = d (see section header)."""
    x     = ca.SX.sym('x', NXM)
    u     = ca.SX.sym('f_cmd', NU)
    d     = ca.SX.sym('d', ND)
    Omega = x[NX:NXM]

    f_cmd_sat = ca.fmin(ca.fmax(u, F_MIN), F_MAX)           # ESC / motor envelope
    omega_cmd = ca.sqrt(f_cmd_sat / K_F)                     # inverse thrust map
    wrench    = ca.mtimes(ca.DM(MIXER), thrust_from_rotor(Omega))
    s = {'x': x, 'u': u, 'T_total': wrench[0], 'tau': wrench[1:4]}

    model             = AcadosModel()
    model.name        = 'quadrotor_rotor_plant'
    model.x           = x
    model.u           = u
    model.p           = d
    model.xdot        = ca.SX.sym('xdot', NXM)
    model.f_expl_expr = ca.vertcat(_build_f_expl(s, d),
                                   (omega_cmd - Omega) / TAU_MOTOR)
    return model


def create_rotor_plant(Ts: float) -> AcadosSimSolver:
    """
    AcadosSimSolver plant with motor dynamics: one call advances by Ts.

        plant.set('x', x_plant)          # [x (13); Ω (4)]  rotor speeds in rad/s
        plant.set('u', f_cmd)            # thrust command (saturated to [F_MIN, F_MAX] inside)
        plant.set('p', d_true)           # disturbance, zero by default
        plant.solve()
        x_plant = quat_normalize(plant.get('x'))

    Counterpart of create_disturbance_plant(); the controller's input is now
    a COMMAND, the thrust on the body lags behind it. 4 RK4 substeps per call
    resolve τ_m = 30 ms at Ts = 50 ms.
    """
    sim = AcadosSim()
    sim.model = _build_rotor_plant_model()
    sim.solver_options.T = Ts
    sim.solver_options.integrator_type = 'ERK'
    sim.solver_options.num_stages = 4
    sim.solver_options.num_steps  = 4
    sim.parameter_values = np.zeros(ND)
    sim.code_export_directory = build_path('c_generated_code')
    return AcadosSimSolver(sim, json_file=build_path(f'{sim.model.name}_sim.json'))


def rotor_plant_hover_state() -> np.ndarray:
    """Plant state [x (13); Ω (4)] at level hover: Ω = OMEGA_ROTOR_HOVER."""
    x = np.zeros(NXM)
    x[6]      = 1.0
    x[NX:NXM] = OMEGA_ROTOR_HOVER
    return x
