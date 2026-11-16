# Stage 1 — Basic NMPC (Terminal Cost $W_e = Q$)

| | |
|---|---|
| Git tag | `stage1` (13-state model, thrusts as direct input; tag `rigid-body` = all stages before the motor dynamics) |
| Files | `stage1_basic/ocp_config_basic.py`, `stage1_basic/simulate_basic.py` (+ shared `core/quadrotor_model.py`, `core/plot_utils.py`) |
| Terminal treatment | terminal cost $W_e = Q$, no terminal constraint |
| Results | `results/stage1_basic/` |

## 1. Motivation

Stage 1 is the baseline of the project: a standard NMPC with the simplest possible terminal
treatment. All later stages run on the same plant and the same cost structure, so differences
can be attributed to the one ingredient each stage changes. Stage 1 answers two questions:
does the toolchain (CasADi model → acados OCP → `SQP_RTI` closed loop) work on the quadrotor
with motor dynamics, and what does an NMPC *without* stability-certifying terminal ingredients deliver?

## 2. Formulation

### 2.1 Starting point: zero-terminal-constraint MPC

The textbook route to a provably stable MPC scheme is the zero-terminal-constraint formulation.
At each time $t$, given the measured state $x(t)$ (written in error coordinates, so the
reference is the origin), solve

```math
\min_{\bar{u}(\cdot;t)} \; J\bigl(x(t),\, \bar{u}(\cdot;t)\bigr) = \int_{t}^{t+T} L\bigl(\bar{x}(\tau;t),\, \bar{u}(\tau;t)\bigr)\, d\tau
```

subject to

```math
\begin{aligned}
\dot{\bar{x}} &= f(\bar{x},\, \bar{u}) && \text{(system dynamics)}
\\ \bar{x}(t;\,t) &= x(t) && \text{(initial condition)}
\\ \bar{u}(\tau;\,t) &\in \mathcal{U}, \quad \bar{x}(\tau;\,t) \in \mathcal{X} && \forall\,\tau \in [t,\, t+T]
\\ \bar{x}(t+T;\,t) &= 0 && \text{(zero-terminal constraint)}
\end{aligned}
```

Forcing the predicted terminal state exactly onto the origin makes the cost-to-go beyond
$t+T$ exactly zero. The optimal value $`J^*(x)`$ then serves as a Lyapunov function that strictly
decreases along the closed loop, which gives asymptotic stability.

### 2.2 Why it is too tight

The equality $`\bar{x}(t+T;\,t) = 0`$ is a hard constraint on all 17 terminal states. The solver
must find admissible thrust commands that steer the full nonlinear model *exactly* to the
reference within $T$ seconds. With box-constrained thrusts and the short horizons needed for
real-time `SQP_RTI`, the feasible set shrinks to a small neighborhood of hover. Larger
setpoint changes, disturbances, or re-initialization then produce infeasible QPs, which is
unacceptable for an online controller. This motivates relaxing the constraint into a
**terminal cost** (this stage) or a **terminal cost + terminal set** (Stage 2a).

### 2.3 Stage 1 relaxation: terminal cost only

Stage 1 drops the terminal constraint and penalizes the terminal state with the stage-cost
weight:

```math
J\bigl(x(t),\, \bar{u}(\cdot;t)\bigr) = \int_{t}^{t+T} L(\bar{x},\, \bar{u})\, d\tau \;+\; \bar{x}(t+T;\,t)^\top Q\, \bar{x}(t+T;\,t)
```

with no constraint on $`\bar{x}(t+T;\,t)`$ beyond the regular bounds.

**What this gives up.** $Q$ is chosen for tracking performance, not derived from the plant, so
$\bar{x}^\top Q \bar{x}$ is not a certified control Lyapunov function. No argument in this
formulation proves that $`J^*`$ decreases along the closed loop, and nothing tells us *how long*
the horizon must be for the closed loop to be stable.

**What it keeps.** In the nominal case the reference is a true equilibrium of the plant
($x_{\text{ref}}$ at hover with $u_{\text{ref}} = u_{\text{hover}}$), where the cost is zero. The
reference is therefore a fixed point of the closed loop and there is no structural offset.
Results on unconstrained-terminal MPC (e.g. Jadbabaie & Hauser 2005; Grüne 2012) show that
such schemes are asymptotically stable for a *sufficiently long* horizon; the horizon bound is
just not available from this formulation. In practice Stage 1 is stable for reasonable
horizons, but stability is observed rather than certified.

**What it improves.** With no terminal equality, feasibility only depends on the input and
state bounds, so the OCP stays feasible even for large setpoint changes.

> A steady-state offset is a different phenomenon: it appears when the prediction model and
> the plant disagree (wind, mass error). That failure mode is treated in Stage 3.

## 3. Implementation

| Item | Setting |
|---|---|
| Model | 17 states $[p, v, q, \omega, f]$ (body + rotor thrusts), 4 inputs $`f_{\text{cmd}} = [f_{\text{cmd},1} \dots f_{\text{cmd},4}]`$, $`\dot f = (f_{\text{cmd}} - f)/\tau_m`$ with $\tau_m = 30$ ms (`create_thrust_lag_model`), see [README](../README.md#quadrotor-model) |
| Integrator | ERK4 (acados), 2 steps per interval because $\tau_m < T_s$ |
| Cost | `NONLINEAR_LS`, $`y = [x;\, f;\, f_{\text{cmd}}]`$, $`W = \mathrm{blkdiag}(Q, Q_f, R_{\text{cmd}})`$, $`y_{\text{ref}} = [x_{\text{ref}};\, f_{\text{hover}};\, f_{\text{hover}}]`$ |
| Terminal cost | $`W_e = \mathrm{blkdiag}(Q, Q_f)`$, $`y_e = [x;\, f]`$ |
| Input bounds | $`f_{\min} \le f_{\text{cmd},i} \le f_{\max} = 3 f_{\text{hover}} \approx 7.36`$ N, with idle thrust $`f_{\min} = k_f \Omega_{\min}^2 \approx 0.074`$ N |
| Solver | `SQP_RTI` (one QP per sample), Gauss-Newton Hessian, `PARTIAL_CONDENSING_HPIPM` |
| Horizon | $N = 20$ steps, $T = 1.0$ s, so $T_s = T/N = 0.05$ s |
| Weights $Q$ | position $(80, 80, 120)$, velocity $(10, 10, 15)$, quaternion $(q_w, q_x, q_y, q_z) = (10, 120, 120, 80)$, body rates $(1, 1, 1)$ |
| Weights $Q_f$, $`R_{\text{cmd}}`$ | $0.1$ per rotor thrust (the former input weight), $0.01$ per command |

The quaternion weights are scaled from Euler-angle weights: since $\phi \approx 2 q_x$ for small
angles, a quaternion weight of $4 w$ matches an Euler weight $w$ (here roll/pitch $30$, yaw
$20$). $q_w$ is only lightly weighted because it is not independent of the vector part.

The thrust weight sits on the thrust **states** $f$, the thrust that acts on the body. The
command only gets a small weight, which keeps the Gauss-Newton Hessian positive definite in the
input. The thrust states need no bounds: the lag keeps $f$ between its start value and the
command, so inside $`[f_{\min}, f_{\max}]`$.

The plant (`create_rotor_plant`) models the chain of a real drone with the rotor speeds
$\Omega$ as states: the command becomes a speed command
$`\omega_{\text{cmd}} = \sqrt{f_{\text{cmd}}/k_f}`$, the ESC speed loop follows with
$`\dot\Omega = (\omega_{\text{cmd}} - \Omega)/\tau_m`$, and the thrust is $`f = k_f \Omega^2`$.
To first order this is the MPC's thrust lag. For large steps the rotors spin up slower and spin
down faster than the linear lag (`core/motor_check.py`): a small, realistic plant–model
mismatch. The rotor speeds are measured (ESC rpm telemetry) and passed to the MPC as
$`f = k_f \Omega^2`$. The plant runs 5 substeps per sample, so the figures show the thrust
between samples. After every plant step the quaternion is re-normalized (`quat_normalize`) to
prevent norm drift.

Run it:

```bash
./run.sh stage1_basic/simulate_basic.py   # → results/stage1_basic/
```

The 13-state version (thrusts as direct input) from its tag:

```bash
git checkout stage1
./run.sh simulate_basic.py        # → results/stage1_basic/
```

## 4. Scenario

Hover-to-hover step from the origin to $`p_{\text{ref}} = (1.0,\; 0.5,\; 1.5)`$ m with level
attitude and zero velocity, $T_{\text{sim}} = 5$ s. The drone starts in hover with the rotors at
hover speed ($\Omega \approx 751$ rad/s). No disturbance, full state measured; the only
plant–model mismatch is the rotor nonlinearity described above.

## 5. Results

### 5.1 Flight path

<p align="center">
  <img src="../results/stage1_basic/trajectory_3d.png" alt="3D flight path" width="60%">
</p>

The drone first climbs almost vertically and then flies a nearly straight line to the target.
The vertical start comes from $v_z$ building up before the horizontal velocities (it peaks
at $\approx 2.9$ m/s at $t \approx 0.3$ s). After that the path is straight because the
horizontal velocities stay proportional to the displacement: the peaks
$v_x \approx 1.35$ m/s and $v_y \approx 0.72$ m/s, both at $t \approx 0.5$ s, have roughly the
$2:1$ ratio of $\Delta p_x : \Delta p_y = 1.0 : 0.5$.

### 5.2 Inputs

![Input trajectory](../results/stage1_basic/inputs.png)

Thin steps are the commands $f_{\text{cmd}}$, solid lines the thrust the rotors actually
produce ($k_f \Omega^2$, logged every 10 ms). The controller uses the full actuation range:
at $t = 0$ all four commands sit at $f_{\max} \approx 7.36$ N, but the rotors need about one
time constant to get there, so the total thrust peaks at $\approx 27.2$ N at $t = 0.15$ s
instead of jumping to $3mg \approx 29.4$ N. Up to $t = 0.2$ s the commands split between
$f_{\max}$ and lower values to generate the torques that tilt the airframe. Between
$t \approx 0.25$ s and $0.4$ s up to two commands sit at the idle bound $f_{\min}$ to brake the
upward velocity; the rotors spin down with the lag, and the total thrust bottoms out at
$\approx 1.1$ N at $t = 0.45$ s, after the commands have already risen again. From there thrust
recovers and settles at $f_{\text{hover}} \approx 2.45$ N ($T_{\text{total}} = mg$) by
$t \approx 2$ s. Both input bounds are active during the transient, which is exactly the regime
where MPC differs from an unconstrained LQR.

Because the MPC predicts the lag, it **leads** the thrust it wants: the commands move ahead of
the rotor thrust and go past the value the thrust should reach (down to $f_{\min}$ while the
rotors are still spinning down, back up before they arrive). A controller that ignored the lag
would ask for the thrust it wants and get it about 30 ms late, every sample.

The aggressiveness follows from the weighting. Position errors are weighted $80\text{–}120$ while
thrust deviations cost only $Q_f = 0.1$, so the optimizer treats thrust as almost free and goes
straight to the bounds. A larger $Q_f$ would give a smoother, slower transient.

### 5.3 States

![State trajectory](../results/stage1_basic/states.png)

**Position** is within 1 cm of the reference from $t = 1.9$ s on, with no overshoot in any
axis, and stays there for the rest of the run with no visible offset (final error
$\approx 10^{-6}$ m).

**Attitude** swings in both directions: pitch $+25° \to -15°$, roll $-14° \to +8°$. This is
not overshoot but the required maneuver: a quadrotor can only accelerate horizontally by
tilting, so it tilts toward the target to accelerate and tilts the other way to decelerate.
The large amplitudes reflect the aggressive acceleration chosen by the weights, with body
rates up to $q \approx 5.0$ rad/s and $p \approx -2.3$ rad/s. Yaw is not commanded but shows a
small transient ($\pm 2.5°$) caused by the yaw torque $\tau_z = c_\tau(-f_1+f_2-f_3+f_4)$ from
unequal motor thrusts during the saturated phase; it returns to zero by $t \approx 2.5$ s.

Compared with the 13-state version (tag `stage1`), the maneuver is slightly gentler (pitch
$+25°$ instead of $+27°$, $v_z$ peak $2.9$ instead of $3.1$ m/s) because the thrust cannot
jump, and it settles in the same time.

**Settling.** All 17 states settle to their references within $\approx 2\text{–}2.5$ s. This is
longer than the prediction horizon $T = 1.0$ s: at $t = 0$ the predicted trajectory ends far
from the target, and the terminal cost $W_e = Q$ is only a rough guess of the remaining
cost-to-go. The loop still converges, but this is the situation in which a certified terminal
cost matters most (Stage 2).

## 6. Takeaways

The terminal-cost-only NMPC works well in this nominal run: fast, constraint-respecting, and
convergent with no visible offset. What the run cannot show is *why* it converges. With
$W_e = Q$ there is no certificate, so a shorter horizon, a larger setpoint, or retuned weights
could break stability without any warning from the formulation.

Stage 2 adds the missing theory on the same scenario:

- **Stage 2a — Quasi-Infinite Horizon:** terminal cost $P_{\text{lyap}}$ plus terminal set
  $\Omega_\alpha$, giving asymptotic stability and recursive feasibility.
- **Stage 2b — DARE terminal cost:** terminal cost $P_{\text{lqr}}$ without a terminal set,
  giving a control Lyapunov function as terminal cost.

| | Terminal treatment | Guarantee |
|---|---|---|
| Zero-terminal-constraint MPC | hard equality $\bar{x}(t+T) = 0$ | asymptotic stability, often infeasible |
| **Stage 1 (this doc)** | terminal cost $W_e = Q$ | none certified; stable for sufficiently long horizon |
| Stage 2a — Quasi-infinite horizon | $P_{\text{lyap}}$ + terminal set $\Omega_\alpha$ | asymptotic stability + recursive feasibility |
| Stage 2b — DARE terminal cost | $P_{\text{lqr}}$, no terminal set | CLF terminal cost, asymptotic stability for sufficiently long horizon |

Because the nominal trajectories of Stages 1, 2a and 2b look nearly identical, Stage 2 compares
the methods through their certificates rather than through state plots →
[Stage 2](stage2_terminal_cost.md).

## References

- A. Jadbabaie, J. Hauser, "On the stability of receding horizon control with a general terminal cost," *IEEE TAC*, 2005.
- L. Grüne, "NMPC without terminal constraints," *IFAC NMPC Conference*, 2012.
- H. Chen, F. Allgöwer, "A quasi-infinite horizon nonlinear model predictive control scheme with guaranteed stability," *Automatica*, 1998.
