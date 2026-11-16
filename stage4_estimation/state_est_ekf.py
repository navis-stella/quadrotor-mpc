"""
state_est_ekf.py — Stage 4a Augmented Extended Kalman Filter (Motor Dynamics)
=============================================================================
EKF for the 3D quadrotor with thrust lag and augmented disturbance state.

    State:        z = [x(13); f(4); d(6)]  ∈ R^23  (augmented thrust-lag model)
    Measurement:  y = [p; q; ω]            ∈ R^10  (position, quaternion,
                                                     angular velocity)
    Unmeasured:   linear velocity v, rotor thrusts f, disturbance d

    The thrusts are predicted from the commands through the lag model and
    corrected through their effect on v̇ (total thrust) and ω̇ (differential
    thrust) — the same paths that reveal d. They are told apart from d by
    their time behavior: f relaxes with −1/τ_m to the command, d is constant.

Model sources (quadrotor_model):
    - Discrete augmented step + exact Jacobian  (get_thrust_lag_augmented_step),
      the same prediction as the Stage 3 EKF
    - Measurement selection                      (IDX_Y, linear → constant H)

Estimator interface (identical for state_est_mhe.MovingHorizonEstimator):
    est = ExtendedKalmanFilter(Ts, z0, ...)
    z_hat_k = est.step(y_k, u_prev)     u_prev = u_{k-1}, None at k = 0
    → the closed loop in closed_loop.py does not know which one it runs.

Tuning is continuous-time where it can be (Q), so changing the sample
time does not silently change how fast d̂ may move.
"""

import numpy as np

from core.quadrotor_model import (
    get_thrust_lag_augmented_step,
    quat_normalize,
    NXM, NZM, NY, IDX_Y,
)

N_RK4 = 2          # RK4 substeps per Ts, as the OCP integrator (τ_m < Ts)


# ═════════════════════════════════════════════════════════════════
# Default Tuning
# ═════════════════════════════════════════════════════════════════
def default_ekf_Q(Ts: float) -> np.ndarray:
    """
    Process noise covariance per sample, Q_d = Q_c · Ts   (23×23).

    Q_c is a spectral density [unit²/s]: small on measured/well-modeled
    states, larger on the unmeasured velocity (model error accumulates) and
    on the disturbance channels (Q_c,d sets how fast d̂ is allowed to move).

    Q_c,d rationale (mocap-equivalent noise):
        Q_c,f = 0.2 N²/s    →  σ ≈ 0.45 N per √s of random-walk drift.
                               Gives ~1 s force-disturbance tracking transient;
                               the residual jitter on d̂_fx, d̂_fy under noise
                               is the observability signature discussed in
                               docs/stage4a_ekf.md §2.2. Lower it for smoother
                               estimates at the cost of slower adaptation to a
                               step change in disturbance.
        Q_c,τ = 2e-3 N²m²/s →  torque disturbances are well observable
                               through ω, so this can stay modest without
                               hurting responsiveness.
    Thrust states (unmeasured):
        Q_c,thr = 1e-2 N²/s →  covers the rotor plant's lag mismatch for large
                               steps (spin-up slower, spin-down faster than
                               ḟ = (f_cmd − f)/τ_m); without it f̂ would follow
                               the model lag of the commands exactly.
    """
    Q_c = np.zeros((NZM, NZM))
    Q_c[0:3,   0:3]   = 2e-5 * np.eye(3)     # pos    (measured)
    Q_c[3:6,   3:6]   = 2e-3 * np.eye(3)     # vel    (unmeasured, model error)
    Q_c[6:10,  6:10]  = 2e-5 * np.eye(4)     # quat   (measured)
    Q_c[10:13, 10:13] = 2e-5 * np.eye(3)     # ω      (measured)
    Q_c[13:17, 13:17] = 1e-2 * np.eye(4)     # thrust (unmeasured, lag mismatch)
    Q_c[17:20, 17:20] = 2e-1 * np.eye(3)     # d_force  — N
    Q_c[20:23, 20:23] = 2e-3 * np.eye(3)     # d_torque — N·m
    return Q_c * Ts


def default_ekf_R(noise_std: dict = None) -> np.ndarray:
    """
    Measurement noise covariance R (10×10), matched to the sensor.

        noise_std = sensor σ dict (sensor_simulator.MOCAP_NOISE_STD)
                    → R = diag(σ²) per channel — the Kalman gain then
                      reflects the actual signal-to-noise ratio
        noise_std = None  (noise-free sensor, open-loop checks)
                    → R = 1e-6 · I — small regularization so S = HPHᵀ + R
                      stays invertible for exact measurements
    """
    if noise_std is None:
        return 1e-6 * np.eye(NY)

    R = np.zeros((NY, NY))
    R[0:3, 0:3]   = noise_std['pos']**2   * np.eye(3)    # pos    [m]
    R[3:7, 3:7]   = noise_std['quat']**2  * np.eye(4)    # quat   [-]
    R[7:10, 7:10] = noise_std['omega']**2 * np.eye(3)    # omega  [rad/s]
    return R


def default_ekf_P0() -> np.ndarray:
    """
    Initial covariance P0 (23×23).

    Measured states start with small P (will snap in one update anyway).
    Unmeasured states (v, d) start with large P — we don't know them.
    The thrusts start at f_hover (project_thrust_lag_measurement) with
    ±0.3 N: the drone is known to hover, the exact rotor state is not.
    """
    P0 = np.zeros((NZM, NZM))
    P0[0:3,   0:3]   = 1e-2 * np.eye(3)     # pos:    ±0.1 m   (measured)
    P0[3:6,   3:6]   = 1.0  * np.eye(3)     # vel:    ±1 m/s  (unmeasured)
    P0[6:10,  6:10]  = 1e-2 * np.eye(4)     # quat:   small   (measured)
    P0[10:13, 10:13] = 1e-2 * np.eye(3)     # omega:  small   (measured)
    P0[13:17, 13:17] = 0.1  * np.eye(4)     # thrust: ±0.3 N  (unmeasured, hover prior)
    P0[17:20, 17:20] = 1.0  * np.eye(3)     # d_f:    ±1 N    (unmeasured)
    P0[20:23, 20:23] = 0.1  * np.eye(3)     # d_τ:    ±0.3 Nm (unmeasured)
    return P0


# ═════════════════════════════════════════════════════════════════
# Extended Kalman Filter
# ═════════════════════════════════════════════════════════════════
class ExtendedKalmanFilter:
    """
    Augmented EKF on the thrust-lag model with force + torque disturbances.

    Discretization:
        State:      RK4 over Ts (N_RK4 substeps) of the augmented model
        Covariance: exact Jacobian of that discrete map, F_d = ∂Φ/∂z
                    (the Euler form I + F_c·Ts is wrong for the thrust lag:
                     1 − Ts/τ_m = −0.67 instead of e^{−Ts/τ_m} = 0.19)
        Update:     Joseph form (numerically robust to non-symmetric P)

    Quaternion handling:
        - z[6:10] renormalized after prediction and after update
        - qw > 0 hemisphere enforced (consistent with plant convention)
        - Innovation flips measurement sign if y_q and hat_q lie in
          opposite hemispheres (SO(3) double-cover fix)
    """

    name = 'EKF'

    def __init__(self,
                 Ts: float,
                 z0: np.ndarray,
                 P0: np.ndarray = None,
                 Q:  np.ndarray = None,
                 R:  np.ndarray = None):
        self.Ts = Ts

        # ── Dynamics + measurement from the model file ─────────
        self.Phi, self.F_d = get_thrust_lag_augmented_step(Ts, N_RK4)
        self.H = np.eye(NZM)[IDX_Y]                    # (10×23) constant selection

        # ── Tuning (defaults if not supplied) ──────────────────
        self.Q = Q  if Q  is not None else default_ekf_Q(Ts)
        self.R = R  if R  is not None else default_ekf_R(None)
        self.P = P0 if P0 is not None else default_ekf_P0()

        # ── State ──────────────────────────────────────────────
        self.z = quat_normalize(np.asarray(z0, dtype=float).copy())

        # ── Shape checks ───────────────────────────────────────
        assert self.z.shape == (NZM,),      f'z0 shape {self.z.shape}'
        assert self.P.shape == (NZM, NZM),  f'P0 shape {self.P.shape}'
        assert self.Q.shape == (NZM, NZM),  f'Q shape  {self.Q.shape}'
        assert self.R.shape == (NY, NY),    f'R shape  {self.R.shape}'

    # ── Estimator interface ─────────────────────────────────────
    def step(self, y: np.ndarray, u_prev: np.ndarray = None) -> np.ndarray:
        """
        One sample: predict with u_{k-1} (skipped at k = 0), update with y_k.
        Returns the posterior ẑ_k (23,).
        """
        if u_prev is not None:
            self.predict(u_prev)
        self.update(y)
        return self.z.copy()

    def info(self) -> str:
        """One-line description for the closed-loop summary."""
        return (f'EKF: RK4 mean ({N_RK4} substeps), exact discrete Jacobian, '
                f'Ts = {self.Ts} s')

    # ── State access ────────────────────────────────────────────
    @property
    def x_hat(self) -> np.ndarray:
        """Estimated MPC state [x̂; f̂] (17,)."""
        return self.z[:NXM].copy()

    @property
    def d_hat(self) -> np.ndarray:
        """Estimated disturbance vector (6,) — for MPC parameter update."""
        return self.z[NXM:].copy()

    # ── Predict step ────────────────────────────────────────────
    def predict(self, u: np.ndarray) -> None:
        """
        Time update:
            z ← Φ(z, u)
            P ← F_d P F_dᵀ + Q,   F_d = ∂Φ/∂z at the prior z
        """
        F_d = np.array(self.F_d(self.z, u))
        self.z = quat_normalize(np.array(self.Phi(self.z, u)).flatten())
        self.P = F_d @ self.P @ F_d.T + self.Q

    # ── Update step ─────────────────────────────────────────────
    def update(self, y: np.ndarray) -> None:
        """
        Measurement update (Joseph form):
            S = H P Hᵀ + R
            K = P Hᵀ S⁻¹
            z ← z + K (y - H z)
            P ← (I - K H) P (I - K H)ᵀ + K R Kᵀ
        """
        H = self.H

        # Quaternion double-cover fix: if y_q and hat_q are in opposite
        # hemispheres, flip y_q so the innovation is small.
        y_used = y.copy()
        if np.dot(y[3:7], self.z[6:10]) < 0:
            y_used[3:7] = -y[3:7]

        innov = y_used - H @ self.z

        # Kalman gain via linear solve (more stable than explicit inverse)
        S = H @ self.P @ H.T + self.R
        K = np.linalg.solve(S.T, (self.P @ H.T).T).T

        self.z = quat_normalize(self.z + K @ innov)

        I_KH = np.eye(NZM) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ self.R @ K.T
