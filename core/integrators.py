"""
integrators.py — All Stages Symbolic RK4 Step
=============================================
F(x, u, p) = one explicit RK4 step of an acados model — the same integrator
the OCPs use (ERK, sim_method_num_stages = 4, one step per interval).

Used by Stage 7b (DCBF rows h(F(x, u)) on the predicted next state) and by
Stage 8 (the residual w = F(x, u, d) − Ax − Bu of the plant's step).
"""

import casadi as ca


def rk4_step(model, Ts: float):
    """
    F(x, u, p) = one RK4 step of model.f_expl_expr over Ts (CasADi expression
    in model.x, model.u, model.p).
    """
    f = ca.Function('f', [model.x, model.u, model.p], [model.f_expl_expr])
    x, u, p = model.x, model.u, model.p
    k1 = f(x,               u, p)
    k2 = f(x + Ts / 2 * k1, u, p)
    k3 = f(x + Ts / 2 * k2, u, p)
    k4 = f(x + Ts * k3,     u, p)
    return x + Ts / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
