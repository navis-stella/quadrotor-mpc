"""
ekf_aug.py — Stage 3 Augmented EKF (Full-State Measurement, Motor Dynamics)
===========================================================================
Estimates the augmented state z = [x(13); f(4); d(6)] ∈ R^23 where:
    x = rigid-body state                        (measured)
    f = rotor thrusts f = k_f Ω²                 (measured: ESC rpm telemetry)
    d = [d_fx, d_fy, d_fz, d_tx, d_ty, d_tz]     constant disturbance

Prediction model: the MPC's thrust-lag model with d as extra states,
    [ẋ; ḟ] = [f(x, f, d); (f_cmd − f)/τ_m],     ḋ = 0.

Measurement model:
    y = H @ z = [x; f]      (full state access, H = [I_17 | 0_17×6])

Since [x; f] is directly measured, the EKF innovation primarily corrects
the disturbance estimate d̂ via the cross-covariance P_xd.
This is the mechanism that gives offset-free MPC its integral action.

Algorithm (discrete EKF on the integrated model):
    Predict:  z⁺ = Φ(z, u)  — RK4 over Ts in N_RK4 substeps (τ_m < Ts)
              F_d = ∂Φ/∂z   — exact Jacobian of that discrete map (CasADi)
              P⁺  = F_d P F_dᵀ + Q_aug·Ts
    Update:   K = P H^T (H P H^T + R)^{-1}
              z_hat += K (y - H z_hat)
              P = (I - K H) P (I - K H)^T + K R K^T   (Joseph form)

Why the exact discrete Jacobian: the 13-state version used F_d ≈ I + F_c·Ts.
The thrust lag has the eigenvalue −1/τ_m = −33 s⁻¹, so the Euler factor
1 − Ts/τ_m = −0.67 would flip the sign of the thrust covariance every step,
while the true factor is e^{−Ts/τ_m} = 0.19. ∂Φ/∂z is the linearization of
exactly the map the state prediction uses, so it carries the correct decay.

Tuning guide:
    Q_d (ND):    disturbance process noise — controls adaptation speed
    Q_x (NXM):   state process noise — covers model imperfections; on f it
                 covers the lag mismatch of the rotor plant (factor
                 2Ω/(ω_cmd+Ω) ≠ 1 for large steps, see quadrotor_model)
    R   (NXM):   measurement noise covariance (clean full-state access)

Relation to Stage 4a (state_est_ekf.py):
    Stage 4a measures only y = [p; q; ω] and has not been migrated to the
    motor model yet (13-state augmented dynamics, Euler covariance step).
"""

import numpy as np
import casadi as ca

from core.quadrotor_model import (
    NXM, NU, ND,
    create_thrust_lag_disturbance_model, thrust_lag_hover_point,
    quat_normalize,
)

NZM    = NXM + ND   # 23 — augmented state [x; f; d]
N_RK4  = 2          # RK4 substeps per Ts, as sim_method_num_steps of the OCP


# ─────────────────────────────────────────────────────────────────
# Discrete augmented dynamics  Φ(z, u)  and  ∂Φ/∂z
# ─────────────────────────────────────────────────────────────────
def get_augmented_step(Ts: float, n_steps: int = N_RK4):
    """
    CasADi functions of one prediction step of the augmented model:

        Phi(z, u)  = [RK4ⁿ(x, u; d); d]      (23,)
        F_d(z, u)  = ∂Phi/∂z                  (23×23)

    The disturbance enters the thrust-lag model as the parameter p; here it
    is the last ND entries of z and stays constant over the step.
    """
    model = create_thrust_lag_disturbance_model()
    f = ca.Function('f_tl', [model.x, model.u, model.p], [model.f_expl_expr])

    z = ca.SX.sym('z', NZM)
    u = ca.SX.sym('u', NU)
    x, d = z[:NXM], z[NXM:]

    h = Ts / n_steps
    for _ in range(n_steps):
        k1 = f(x,              u, d)
        k2 = f(x + h / 2 * k1, u, d)
        k3 = f(x + h / 2 * k2, u, d)
        k4 = f(x + h * k3,     u, d)
        x = x + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

    z_next = ca.vertcat(x, d)
    Phi = ca.Function('Phi_aug', [z, u], [z_next], ['z', 'u'], ['z_next'])
    F_d = ca.Function('F_d_aug', [z, u], [ca.jacobian(z_next, z)], ['z', 'u'], ['F_d'])
    return Phi, F_d


class AugEKF:
    """
    Extended Kalman Filter on the augmented motor-dynamics model.

    State: z = [x(13); f(4); d(6)] ∈ R^23
    Measurement: y = [x; f] ∈ R^17 (full state access)

    Usage in closed-loop:
        ekf = AugEKF(Ts=0.05, Q_x=..., Q_d=..., R=...)

        for each MPC step:
            ekf.update(y)                   # 1. correct with the measurement
            x_hat = ekf.get_state()         # 2. (17,)
            d_hat = ekf.get_disturbance()   #    (6,)
            ...                             # 3. solve MPC with d_hat
            ekf.predict(u_applied)          # 4. propagate with the command
    """

    def __init__(self, Ts: float,
                 Q_x: np.ndarray, Q_d: np.ndarray, R: np.ndarray,
                 x0: np.ndarray = None, d0: np.ndarray = None,
                 P0: np.ndarray = None):
        """
        Args:
            Ts:   sample time [s]
            Q_x:  (NXM,) or (NXM,NXM) state process noise
            Q_d:  (ND,) or (ND,ND) disturbance process noise
            R:    (NXM,) or (NXM,NXM) measurement noise covariance
            x0:   (NXM,) initial state estimate (default: hover)
            d0:   (ND,) initial disturbance estimate (default: zeros)
            P0:   (NZM,NZM) initial covariance (default: block-diagonal)
        """
        self.Ts = Ts

        Q_x = np.diag(Q_x) if np.ndim(Q_x) == 1 else Q_x
        Q_d = np.diag(Q_d) if np.ndim(Q_d) == 1 else Q_d
        R   = np.diag(R)   if np.ndim(R)   == 1 else R
        self.Q_aug = np.block([
            [Q_x,                 np.zeros((NXM, ND))],
            [np.zeros((ND, NXM)), Q_d                ],
        ])
        self.R = R

        # ── Measurement matrix H = [I_17 | 0_17×6] ────────────
        self.H = np.hstack([np.eye(NXM), np.zeros((NXM, ND))])

        # ── Initial augmented state ────────────────────────────
        if x0 is None:
            x0, _ = thrust_lag_hover_point()
        if d0 is None:
            d0 = np.zeros(ND)
        self.z_hat = np.concatenate([x0, d0])

        # ── Initial covariance: state known, disturbance unknown ─
        if P0 is not None:
            self.P = P0.copy()
        else:
            self.P = np.block([
                [0.01 * np.eye(NXM),  np.zeros((NXM, ND))],
                [np.zeros((ND, NXM)), 1.0 * np.eye(ND)   ],
            ])

        # ── CasADi discrete step and its Jacobian ──────────────
        self.Phi, self.F_d = get_augmented_step(Ts)

    # ─────────────────────────────────────────────────────────────
    # Prediction step
    # ─────────────────────────────────────────────────────────────
    def predict(self, u: np.ndarray):
        """
        Propagate state and covariance over one Ts with the applied command u.

        State:      z(k+1|k) = Φ(z(k|k), u(k))
        Covariance: P(k+1|k) = F_d P(k|k) F_d^T + Q_aug·Ts,  F_d = ∂Φ/∂z at z(k|k)
        """
        F_d = np.array(self.F_d(self.z_hat, u))
        self.z_hat = np.array(self.Phi(self.z_hat, u)).flatten()
        quat_normalize(self.z_hat)

        self.P = F_d @ self.P @ F_d.T + self.Q_aug * self.Ts
        self.P = 0.5 * (self.P + self.P.T)

    # ─────────────────────────────────────────────────────────────
    # Update (correction) step
    # ─────────────────────────────────────────────────────────────
    def update(self, y: np.ndarray):
        """
        Correct the estimate with the measurement y = [x; f] (17,).

        Even though [x; f] is measured directly, the Kalman gain has nonzero
        rows for d (the last 6) because P_xd ≠ 0. This is how d̂ adapts.
        """
        H, P, R = self.H, self.P, self.R

        # Quaternion double cover: flip the measured q into the estimate's hemisphere
        if y[6] * self.z_hat[6] < 0:
            y = y.copy()
            y[6:10] *= -1
        e = y - H @ self.z_hat

        S = H @ P @ H.T + R
        K = P @ H.T @ np.linalg.inv(S)

        self.z_hat = self.z_hat + K @ e
        quat_normalize(self.z_hat)

        I_KH = np.eye(NZM) - K @ H
        self.P = I_KH @ P @ I_KH.T + K @ R @ K.T
        self.P = 0.5 * (self.P + self.P.T)

    # ─────────────────────────────────────────────────────────────
    # Accessors
    # ─────────────────────────────────────────────────────────────
    def get_state(self) -> np.ndarray:
        """Estimated MPC state [x̂; f̂] (17,)."""
        return self.z_hat[:NXM].copy()

    def get_disturbance(self) -> np.ndarray:
        """Estimated disturbance d̂ (6,)."""
        return self.z_hat[NXM:].copy()

    def get_covariance(self) -> np.ndarray:
        """Augmented covariance P (23×23)."""
        return self.P.copy()


# ─────────────────────────────────────────────────────────────────
# Default tuning
# ─────────────────────────────────────────────────────────────────
def get_default_ekf_tuning():
    """
    Default Q_x, Q_d, R for the motor-dynamics EKF.

    Body states as in the 13-state version. Thrust states:
        Q_f = 0.01 N²/s  — the rotor plant's lag differs from ḟ = (f_cmd − f)/τ_m
                           for large steps; this lets the measured f correct f̂
                           instead of pushing the mismatch into d̂
        R_f = 0.001 N²   — rpm telemetry, f = k_f Ω²
    """
    Q_x = np.array([
        0.001, 0.001, 0.001,           # position [m²/s]
        0.01,  0.01,  0.01,            # velocity [m²/s³]
        0.001, 0.001, 0.001, 0.001,    # quaternion [-]
        0.01,  0.01,  0.01,            # angular rates [rad²/s³]
        0.01,  0.01,  0.01,  0.01,     # thrusts [N²/s]
    ])
    Q_d = np.array([
        0.1,  0.1,  0.1,               # force disturbance [N²/s]
        0.01, 0.01, 0.01,              # torque disturbance [N²m²/s]
    ])
    R = np.array([
        0.001, 0.001, 0.001,           # position [m²]
        0.01,  0.01,  0.01,            # velocity [m²/s²]
        0.001, 0.001, 0.001, 0.001,    # quaternion [-]
        0.01,  0.01,  0.01,            # angular rates [rad²/s²]
        0.001, 0.001, 0.001, 0.001,    # thrusts [N²]
    ])
    return Q_x, Q_d, R
