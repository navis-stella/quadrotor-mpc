"""
motor_check.py — All Stages Motor Dynamics Check
================================================
Checks the motor-dynamics models of quadrotor_model before any stage
uses them:

    1. Hover is an equilibrium of the MPC model [x; f] and of the plant [x; Ω].
    2. Linearization structure: A = [A_body B_body; 0 −I/τ], B = [0; I/τ],
       with (A_body, B_body) exactly the old 13-state hover linearization.
    3. Thrust step responses (all four motors, so the attitude stays level):
       small step → plant and lag model agree, f reaches 63 % at t ≈ τ_m;
       full throttle / cut to idle → the plant saturates at F_MAX / F_MIN and
       shows the nonlinear speed-loop lag (slower spin-up, faster spin-down
       than the linear thrust lag).

Run:  python -m core.motor_check     → results/motor_dynamics/step_response.png
"""

import numpy as np
import casadi as ca

from core.quadrotor_model import (
    create_thrust_lag_model, _build_rotor_plant_model, create_rotor_plant,
    thrust_lag_hover_point, rotor_plant_hover_state, get_thrust_lag_hover_linearization,
    get_hover_linearization, thrust_from_rotor, quat_normalize,
    TAU_MOTOR, F_MIN, F_MAX, K_F, OMEGA_ROTOR_MIN, OMEGA_ROTOR_MAX,
    OMEGA_ROTOR_HOVER, f_hover, NX, NXM, NU, ND,
)
from core.plot_utils import plot_motor_steps

TS_CHECK = 0.002          # [s]  fine sampling of the step responses
T_STEP   = 0.20           # [s]  ≈ 6.7 τ_m
OUT      = 'results/motor_dynamics/step_response.png'


def _ok(flag: bool) -> str:
    return '✓' if flag else '✗'


# ─────────────────────────────────────────────────────────────────
# 1. Hover equilibrium
# ─────────────────────────────────────────────────────────────────
def check_hover() -> bool:
    mdl = create_thrust_lag_model()
    f_m = ca.Function('f_m', [mdl.x, mdl.u], [mdl.f_expl_expr])
    x_h, u_h = thrust_lag_hover_point()
    r_model = np.abs(f_m(x_h, u_h).full()).max()

    plt_mdl = _build_rotor_plant_model()
    f_p = ca.Function('f_p', [plt_mdl.x, plt_mdl.u, plt_mdl.p], [plt_mdl.f_expl_expr])
    r_plant = np.abs(f_p(rotor_plant_hover_state(), u_h, np.zeros(ND)).full()).max()

    ok = r_model < 1e-12 and r_plant < 1e-9
    print('─── 1. Hover equilibrium ──────────────────────────')
    print(f'  max |ẋ| MPC model [x; f] = {r_model:.2e}')
    print(f'  max |ẋ| plant     [x; Ω] = {r_plant:.2e}   (Ω_hover = {OMEGA_ROTOR_HOVER:.1f} rad/s)')
    print(f'  {_ok(ok)} hover is an equilibrium of both')
    return ok


# ─────────────────────────────────────────────────────────────────
# 2. Linearization structure
# ─────────────────────────────────────────────────────────────────
def check_linearization() -> bool:
    A, B = get_thrust_lag_hover_linearization()
    A_b, B_b = get_hover_linearization()
    I = np.eye(NU)
    errs = {
        'A[x, x] = A_body':     np.abs(A[:NX, :NX] - A_b).max(),
        'A[x, f] = B_body':     np.abs(A[:NX, NX:] - B_b).max(),
        'A[f, x] = 0':          np.abs(A[NX:, :NX]).max(),
        'A[f, f] = −I/τ':       np.abs(A[NX:, NX:] + I / TAU_MOTOR).max(),
        'B[x]    = 0':          np.abs(B[:NX]).max(),
        'B[f]    = I/τ':        np.abs(B[NX:] - I / TAU_MOTOR).max(),
    }
    ok = max(errs.values()) < 1e-12
    print('─── 2. Linearization at hover ─────────────────────')
    for k, v in errs.items():
        print(f'  {k:<20s} max err = {v:.1e}')
    print(f'  {_ok(ok)} input reaches the body only through the lag')
    return ok


# ─────────────────────────────────────────────────────────────────
# 3. Step responses
# ─────────────────────────────────────────────────────────────────
def step_response(plant, f_cmd: float):
    """Plant thrust k_f Ω(t) and MPC-model thrust after a step hover → f_cmd."""
    n = int(round(T_STEP / TS_CHECK)) + 1
    t = np.arange(n) * TS_CHECK
    x = rotor_plant_hover_state()
    f_plant = np.empty(n)
    f_plant[0] = f_hover
    for k in range(1, n):
        plant.set('x', x)
        plant.set('u', np.full(NU, f_cmd))
        plant.solve()
        x = quat_normalize(plant.get('x'))
        f_plant[k] = thrust_from_rotor(x[NX])
    f_model = f_cmd + (f_hover - f_cmd) * np.exp(-t / TAU_MOTOR)   # exact solution of ḟ = (f_cmd − f)/τ
    return t, f_plant, f_model


def t63(t, f):
    """Time at which f has covered 1 − e⁻¹ of its total change (linear interpolation)."""
    y = (f - f[0]) / (f[-1] - f[0])
    k = np.argmax(y >= 1.0 - np.exp(-1.0))
    return t[k - 1] + (1.0 - np.exp(-1.0) - y[k - 1]) / (y[k] - y[k - 1]) * (t[k] - t[k - 1])


def check_steps() -> bool:
    plant = create_rotor_plant(TS_CHECK)
    specs = [('small step +10 %',     1.1 * f_hover),
             ('full throttle',        F_MAX),
             ('cut to idle (f_cmd=0)', 0.0)]
    cases, ok = [], True
    print('─── 3. Thrust step responses (from hover) ─────────')
    print(f'  τ_m = {TAU_MOTOR*1e3:.0f} ms,  Ω ∈ [{OMEGA_ROTOR_MIN:.0f}, {OMEGA_ROTOR_MAX:.0f}] rad/s,'
          f'  f ∈ [{F_MIN:.3f}, {F_MAX:.3f}] N,  k_f = {K_F:.3e} N·s²')
    for label, f_cmd in specs:
        t, f_p, f_m = step_response(plant, f_cmd)
        f_target = np.clip(f_cmd, F_MIN, F_MAX)
        if f_cmd < F_MIN:            # the model has no clip: compare it with the saturated command
            f_m = f_target + (f_hover - f_target) * np.exp(-t / TAU_MOTOR)
        dev = np.abs(f_p - f_m).max() / abs(f_target - f_hover)
        t63_p = t63(t, f_p)
        final_ok = abs(f_p[-1] - f_target) < 0.01 * abs(f_target - f_hover)
        print(f'  {label:<22s} f_∞ = {f_p[-1]:.3f} N (target {f_target:.3f})   '
              f't63 plant = {t63_p*1e3:5.1f} ms   max |plant − model| = {100*dev:4.1f} % of step')
        ok &= final_ok
        if label.startswith('small'):
            ok &= abs(t63_p - TAU_MOTOR) < 0.03 * TAU_MOTOR and dev < 0.03
        cases.append({'label': label, 'f_cmd': f_cmd, 'f_plant': f_p, 'f_model': f_m})
    print(f'  {_ok(ok)} small step matches the lag model; large steps saturate at the envelope')

    plot_motor_steps(t, cases, TAU_MOTOR, F_MIN, F_MAX,
                     'Motor dynamics: thrust step responses from hover (all four motors)', OUT)
    return ok


if __name__ == '__main__':
    results = [check_hover(), check_linearization(), check_steps()]
    print('═' * 52)
    print(f'  {_ok(all(results))} motor dynamics check {"passed" if all(results) else "FAILED"}')
