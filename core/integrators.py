"""
integrators.py — All Stages Symbolic RK4 Step
=============================================
F(x, u, p) = RK4 over Ts of an acados model, in n_steps equal substeps —
the same integrator the OCPs use (ERK, sim_method_num_stages = 4,
sim_method_num_steps = n_steps; 2 on the thrust-lag model, τ_m < Ts).

Used by Stage 7b (DCBF rows h(F(x, u)) on the predicted next state) and by
Stage 8 (the residual w = F(x, u, d) − Ax − Bu of the plant's step).
"""

import casadi as ca


def rk4_step(model, Ts: float, n_steps: int = 1):
    """
    F(x, u, p) = RK4 of model.f_expl_expr over Ts in n_steps substeps
    (CasADi expression in model.x, model.u, model.p).
    """
    f = ca.Function('f', [model.x, model.u, model.p], [model.f_expl_expr])
    x, u, p = model.x, model.u, model.p
    h = Ts / n_steps
    for _ in range(n_steps):
        k1 = f(x,              u, p)
        k2 = f(x + h / 2 * k1, u, p)
        k3 = f(x + h / 2 * k2, u, p)
        k4 = f(x + h * k3,     u, p)
        x  = x + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return x
