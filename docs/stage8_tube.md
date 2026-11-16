# Stage 8 — Robust Tube MPC around a Hover Point

Stages 1–7 give no guarantee once the model is wrong: the constraints hold for the *prediction*,
and the true state can still leave them. This stage implements **tube-based robust MPC** for a
linear system with a bounded additive disturbance (Mayne, Seron & Raković, 2005; Rawlings, Mayne &
Diehl, §3.5). The controller regulates the quadrotor to a hover point, and the plant is still the
**nonlinear** quadrotor. The linear tube MPC "sees" it as

```math
x^+ = A x + B u + w, \qquad w \in \mathbb W,
```

where $w$ contains everything the linear model gets wrong: the force/torque disturbance $d$ **and**
the linearization error. The tube then guarantees that the state and input constraints hold for the
real nonlinear plant at every sample, for every disturbance in the set $\mathbb D$. The one caveat is
that $\mathbb W$ is a sampled bound ([§2](#2-from-the-nonlinear-quadrotor-to-x--ax--bu--w)).

The controller is the **linear** tube MPC, unchanged: linear model, linear error feedback, a tube
of constant cross-section. The nonlinearity enters in one place only, the size of $\mathbb W$. No
nonlinear robust MPC theory is used.

| | |
|---|---|
| Prediction model | hover linearization, $\dot q_w$ row removed (12 states), ZOH at $T_s = 0.05$ s |
| Plant | `create_disturbance_plant()` — nonlinear, ERK4, $d(t) \in \mathbb D$ time-varying |
| Feedback | full state |
| Horizon | $N = 60$, $T = 3$ s |
| Solver | acados `SQP`, exact Hessian, `PARTIAL_CONDENSING_HPIPM`, `DISCRETE` dynamics |
| Offline | `compute_tube_params.py` → `data/tube_params.npz` |
| Entry point | `./run.sh simulate_tube.py` → `results/stage8_tube/` |

---

## 1. Tube MPC in one page

For $x^+ = Ax + Bu + w$, $w \in \mathbb W$, with constraints $x \in \mathbb X$, $u \in \mathbb U$, the
controller has two parts:

```math
\begin{aligned}
\text{nominal system} \quad & z^+ = A z + B v, \qquad z_0 = x_0
\\ \text{ancillary law} \quad & u = v + K (x - z) .
\end{aligned}
```

The error $e = x - z$ obeys $e^+ = (A + BK) e + w = A_K e + w$, and $e_0 = 0$. So for every
disturbance sequence

```math
e_k \in F_k = \bigoplus_{i=0}^{k-1} A_K^i \mathbb W \;\subset\; S_K(\infty) = \bigoplus_{i=0}^{\infty} A_K^i \mathbb W ,
```

the **minimal robust positively invariant (mRPI) set**. The true state lies in a tube of fixed
cross-section around the nominal trajectory, $x_k \in z_k \oplus S$. If the *nominal* system
satisfies the tightened constraints

```math
z \in \bar{\mathbb X} = \mathbb X \ominus S, \qquad v \in \bar{\mathbb U} = \mathbb U \ominus K S ,
```

then $x \in \mathbb X$ and $u \in \mathbb U$ for every $w$. The nominal system has no disturbance,
so an ordinary MPC on it, with terminal ingredients for $(\bar{\mathbb X}, \bar{\mathbb U})$, is
recursively feasible and drives $z \to 0$. Hence $x \to S$.

The disturbance is handled in two places. Offline, the tube $S$ is computed once and the constraints
are shrunk by it. Online, the feedback $K(x - z)$ keeps the state inside the tube, and the MPC only
plans the nominal trajectory.

## 2. From the nonlinear quadrotor to x⁺ = Ax + Bu + w

### Coordinates

The deviation from the target hover point $(p_{\text{ref}}, v = 0, q = 1, \omega = 0)$:

```math
x = \big[\, p - p_{\text{ref}},\; v,\; q_x, q_y, q_z,\; \omega \,\big] \in \mathbb R^{12}, \qquad
u = f - f_{\text{hover}} \in \mathbb R^4 .
```

$q_w$ is dropped, as in the DARE terminal cost of Stage 2b. It is uncontrollable in the
linearization and follows from $\lVert q \rVert = 1$. $(A, B)$ is the ZOH discretization of
`get_hover_linearization()`.

### The disturbance set is a model-error set

Define $w$ as the difference between one plant step and the linear prediction:

```math
w(x, u, d) = F(x, u, d) - A x - B u ,
```

with $F$ = one RK4 step of the nonlinear model followed by the quaternion normalization (exactly
what the simulation does). $w$ collects

- the disturbance $d$ (force in $\dot v$, torque in $\dot\omega$),
- the thrust-tilt coupling $\delta T \cdot q$ (thrust changes and tilt multiply),
- the gyroscopic terms $\omega \times I \omega$,
- the kinematic coupling of roll/pitch rates and tilt into $q_z$,
- RK4 versus ZOH, and the normalization.

$\mathbb W$ is the componentwise bound $\lvert w_j \rvert \le \bar w_j$ over $\mathbb X \times \mathbb U
\times \mathbb D$. The tube MPC keeps $x \in \mathbb X$ and $u \in \mathbb U$, so the bound holds
along the closed loop. This is not circular: the sets are fixed first, and the guarantee keeps the
state inside them.

The maximum is **searched, not proven**, in three steps:

1. all $2^{16}$ vertices of the $(q_v, \omega, \nu, d)$ box,
2. $2 \cdot 10^5$ uniform samples of the box,
3. a local maximization of each $\pm w_j$ over the box (IPOPT), started from the best vertex and
   the best random sample.

The result is multiplied by a margin of 1.1. In every component the random maximum stays 0–21 %
below the vertex maximum, as expected for a residual made of products of state and input. The local
maximization finds values at most 0.015 % above the best vertex, so the vertices are not exactly
the maximizers, and the margin covers the difference many times over. It is the same kind of
certificate as the Lipschitz check of Stage 2a.

Two assumptions go into it:

- $w$ does not depend on $p$ or $v$ (asserted numerically), so $\mathbb X$ needs no position bound
  and the search runs over 16 instead of 22 variables.
- $d$ is constant over each sample.

The function that is bounded is the plant itself: it agrees with the acados plant step to
$10^{-15}$ (`compute_tube_params.py --check`).

### The sets

| Set | Bound | |
|---|---|---|
| $\mathbb X$ | $`\lvert v_i \rvert \le 1`$ m/s, $`\lvert q_x \rvert, \lvert q_y \rvert \le 0.05`$ (5.7°), $`\lvert q_z \rvert \le 0.05`$, $`\lvert p \rvert, \lvert q \rvert \le 0.5`$ rad/s, $`\lvert r \rvert \le 0.2`$ rad/s | position free |
| $\mathbb U$ | $`\lvert \delta T \rvert \le 0.6`$ N, $`\lvert \tau_x \rvert, \lvert \tau_y \rvert \le 0.05`$ N·m, $`\lvert \tau_z \rvert \le 0.012`$ N·m | box on the **wrench** $\nu = M u$ |
| $\mathbb D$ | $`\lvert d_{f,i} \rvert \le 0.05`$ N, $`\lvert d_{\tau,i} \rvert \le 5 \cdot 10^{-4}`$ N·m | light wind, CoG offset ≈ 0.5 mm |

**Why a wrench box.** With the motor box $0 \le f_i \le F_{\max}$, one sample with a full motor
step changes $\omega$ by about 2 rad/s, and $\delta T$ up to 20 N multiplies the tilt. The resulting
$\bar w$ is larger than $\mathbb X$ itself, and no tube fits. Bounding the wrench
$\nu = M u = [\delta T, \tau_x, \tau_y, \tau_z]$ (with $M$ = `MIXER`) directly limits both effects.
The box lies inside the motor box: $f_i \in [1.33, 3.58]$ N.

**Why so small.** The linearization error grows *quadratically* with the size of
$\mathbb X \times \mathbb U$, the effect of $d$ does not. A larger tilt box gives more authority
against $d$ but a larger $\bar w$. The sets come from a sweep over tilt, rate and $\delta T$ boxes
that minimized the largest tube share ([§3](#3-the-tube-cross-section-s)). This is regulation *around*
the target, not aggressive flight.

| | $p$ | $v$ | $q_{x,y}$ | $q_z$ | $\omega_{x,y}$ | $r$ |
|---|---|---|---|---|---|---|
| $\bar w$ from $d$ only | $6.3 \cdot 10^{-5}$ | $2.5 \cdot 10^{-3}$ | $3.1 \cdot 10^{-5}$ | $1.6 \cdot 10^{-5}$ | $2.5 \cdot 10^{-3}$ | $1.3 \cdot 10^{-3}$ |
| $\bar w$ sampled | $2.2 \cdot 10^{-4}$ | $9.3 \cdot 10^{-3}$ | $1.2 \cdot 10^{-3}$ | $1.6 \cdot 10^{-3}$ | $9.3 \cdot 10^{-3}$ | $1.3 \cdot 10^{-3}$ |
| ratio | 3.5 | 3.7 | 40 | 103 | 3.7 | 1.0 |

The **linearization error dominates**: 3.5–3.7 times the effect of $d$ in $p, v, \omega$, and 40–100
times in the attitude. Only the yaw rate is pure $d$ (yaw torque is linear, and the gyroscopic
term $`(I_{xx} - I_{yy})\, p\, q`$ vanishes for $I_{xx} = I_{yy}$).

## 3. The tube cross-section S

### Only the support function is needed

The tightening $\mathbb X \ominus S$ of a box needs, for each row $c$, only the **support
function** $h_S(c) = \max_{e \in S} c^\top e$:

```math
\mathbb X \ominus S = \{ x : c_j^\top x \le b_j - h_S(c_j) \}.
```

The tube never has to be represented as a set. That matters here: an explicit polytope of the mRPI
set in 12 dimensions (Minkowski sums of a 12-dimensional box) is not computable in practice. For a
box $\mathbb W$, $h_{\mathbb W}(c) = \sum_j \lvert c_j \rvert \bar w_j$, and the support function of
the Minkowski sum is the sum of the support functions. The outer approximation of Raković,
Kerrigan, Kouramas & Mayne (2005),

```math
S = F(\alpha, s) = (1 - \alpha)^{-1} \bigoplus_{i=0}^{s-1} A_K^i \mathbb W \;\supseteq\; S_K(\infty),
\qquad A_K^s \mathbb W \subseteq \alpha \mathbb W ,
```

is itself RPI and has the closed-form support function

```math
h_S(c) = (1 - \alpha)^{-1} \sum_{i=0}^{s-1} \sum_j \big\lvert (c^\top A_K^i)_j \big\rvert \, \bar w_j .
```

For a box, $A_K^s \mathbb W \subseteq \alpha \mathbb W$ holds iff
$`\lvert A_K^s \rvert \bar w \le \alpha \bar w`$ row by row. With $\alpha = 0.01$ this gives $s = 131$.

The formula is tight: the disturbance sequence $`w_i = \bar w \circ \mathrm{sign}((A_K^i)^\top c)`$
drives $c^\top e$ to exactly $(1 - \alpha)\, h_S(c)$ after $s$ steps (checked by `--check` for the
12 axes and the 4 wrench rows).

$S$ is a zonotope with $12 s = 1572$ generators and is far from a box: a point at 20 % of every axis
bound $h_S(e_j)$ can lie outside it. Membership is therefore tested exactly, by the gauge

```math
\gamma(e) = \min \{\, t : e \in t\, S \,\}, \qquad e \in S \iff \gamma(e) \le 1 ,
```

a linear program in the generator coefficients (`tube_gauge`).

### Explicit tube shapes were too conservative

Before settling on the support function, explicit RPI sets were tried at the final sets. Mayne et
al. (2005) need one for the variant with a free initial nominal state ([§9](#9-observations-and-limitations)):

- an **ellipsoid** $`\{ e^\top P e \le 1 \}`$ with $K$ and $P$ from one LMI (S-procedure over the box
  $\mathbb W$, feedback $K e$ limited to a share of $\mathbb U$, contraction rate $\lambda$ from
  0.8 to 0.99). With a 50 % share the LMI is infeasible for every $\lambda$. With 70 % and 90 % an
  ellipsoid exists but is 2.4 and 1.4 times the state bounds, so it does not fit into $\mathbb X$;
- a **modal box** (intervals and discs in the eigen-coordinates of $A_K$, RPI in closed form):
  1.3 to 4.6 times the mRPI set along the axes, 1.75 times the bounds.

The mRPI support function is the tightest choice for a given $K$ and $\mathbb W$, and it was the only
one that left room for the nominal trajectory.

### The ancillary gain K

$K$ is an LQR gain on the wrench input, $(A, B M^{-1})$, with weights normalized by the bounds:
$`Q = \mathrm{diag}(q_{\text{pos}} \mathbf 1_3,\; 0.3 \cdot \mathbf 1_3,\; q_{\text{att}} \mathbf 1_3,\; 0.1 \cdot \mathbf 1_3) / s^2`$
and $R = r / \bar\nu^2$, where $s$ holds the state bounds (0.5 m for position). A 27-point grid over
$(r, q_{\text{att}}, q_{\text{pos}})$ is searched. The point whose tube occupies the smallest share of
any bound wins: $(0.1, 1, 1)$, with $\rho(A_K) = 0.950$.

$K$ and the nominal-MPC cost have different jobs. $K$ keeps $e$ small against every
$w \in \mathbb W$. The MPC moves $z$ to the target.

### Result: tube and tightened sets

| | $v_{x,y}$ | $v_z$ | $q_{x,y}$ | $q_z$ | $\omega_{x,y}$ | $r$ | $\delta T$ | $\tau_{x,y}$ | $\tau_z$ |
|---|---|---|---|---|---|---|---|---|---|
| bound | 1.0 | 1.0 | 0.05 | 0.05 | 0.5 | 0.2 | 0.6 | 0.05 | 0.012 |
| $h_S$ (tube) | 0.246 | 0.107 | 0.0257 | 0.0154 | 0.192 | 0.081 | 0.299 | 0.017 | 0.0062 |
| share | 0.25 | 0.11 | **0.51** | 0.31 | 0.38 | 0.40 | 0.50 | 0.34 | **0.52** |
| tightened | 0.754 | 0.893 | 0.0243 | 0.0346 | 0.309 | 0.119 | 0.301 | 0.033 | 0.0058 |

The position cross-section is $\pm 0.20$ m horizontally and $\pm 0.06$ m vertically. The tube takes up
to 52 % of a bound, mostly because of the linearization error. With $\bar w$ from $d$ alone, the
largest share would be 0.13.

## 4. Nominal MPC and terminal ingredients

```math
\begin{aligned}
\min_{v_0, \dots, v_{N-1}} \quad & \sum_{i=0}^{N-1} \big( \lVert z_i \rVert_Q^2 + \lVert v_i \rVert_R^2 \big) + \lVert z_N \rVert_{P_f}^2
\\ \text{s.t.} \quad & z_{i+1} = A z_i + B v_i, \qquad z_0 = z_k \quad \text{(the nominal state)}
\\ & z_i \in \bar{\mathbb X}, \quad i = 1, \dots, N-1, \qquad M v_i \in \bar{\mathbb U}, \quad i = 0, \dots, N-1
\\ & z_N^\top P_f z_N \le \alpha_f
\end{aligned}
```

- **Terminal set.** $P_f$ and $K_f$ come from the DARE of the nominal cost, so
  $V_f(z^+) - V_f(z) = -\ell(z, K_f z)$ under $v = K_f z$ and every sublevel set is invariant.
  $\alpha_f$ is the largest level with the ellipsoid inside $\bar{\mathbb X}$ and
  $M K_f z \in \bar{\mathbb U}$, in closed form (the support function of an ellipsoid). Result:
  $\alpha_f = 2.53$, semi-axes 0.28 m in position and 0.19 m/s in velocity.
- **Cost.** Bryson's rule on the *tightened* sets: $`Q = \mathrm{diag}(1/(1\,\text{m})^2,\; 1/\bar x_j^2)`$,
  $`R = M^\top \mathrm{diag}(1/\bar\nu^2)\, M`$. The Stage 2b weights (position 80, tilt 120) were
  tried first. Their DARE gain is so aggressive in position that the terminal ellipsoid shrinks to
  3.7 mm to keep $K_f z$ within the 2.8° of tilt left in $\bar{\mathbb X}$. The horizon then has to
  do all the work: with $N = 40$, a start only 0.8 m away was already infeasible.
- **Horizon.** The tightened tilt allows about 0.5 m/s² of lateral acceleration, so 1.4 m takes a
  few seconds. With $N = 40$ (2 s) the start is infeasible, so $N = 60$ (3 s) is used.
- **Recursive feasibility** is the nominal argument. The nominal model is exact for $z$, so the
  shifted solution with $v_N = K_f z_N$ is feasible at the next sample. If a solve does not
  converge, the controller applies exactly this candidate (`shifted_candidate`), and the
  guarantee does not depend on the solver. In the run it was never needed. A test with two of
  every three solves forced to fail (199 of 300) kept every inequality of [§7](#7-results).

## 5. Mapping to acados

| Theory | acados |
|---|---|
| $z^+ = A z + B v$ | `integrator_type = 'DISCRETE'`, `disc_dyn_expr` — the very $(A, B)$ of $\mathbb W$ |
| $\ell$, $V_f$ | `LINEAR_LS`, $y = [z; v]$, `yref = 0`; terminal `W_e = P_f` |
| no step-length scaling | `cost_scaling = 1` on every node — the default ($T_s$ on the stage cost) would break $V_f(z^+) - V_f(z) \le -\ell$ |
| $z_0 = z_k$ | `constraints.x0` |
| $z_i \in \bar{\mathbb X}$ | `idxbx`, `lbx = −x̄`, `ubx = x̄` |
| $M v_i \in \bar{\mathbb U}$ | general linear constraints `C = 0`, `D = MIXER`, `lg/ug = ±ν̄` |
| $z_N \in \mathbb X_f$ | `con_h_expr_e` $= z^\top P_f z$, `uh_e = α_f`, `lh_e = −ACADOS_INFTY` |
| convex QCQP | `SQP` with `hessian_approx = 'EXACT'` (the constraint Hessian $2\mu P_f$), 8 iterations from a cold start, warm-started from the shifted plan afterwards |

The first OCP was also solved with an independent conic solver (CVXPY + Clarabel): same cost to 8
digits, trajectories within $4 \cdot 10^{-6}$. So the acados problem is the one written in [§4](#4-nominal-mpc-and-terminal-ingredients),
including the unit cost scaling and the constraint at node 0.

**One-sided terminal bound.** With `lh_e = 0`, the lower bound $z^\top P_f z \ge 0$ is active at the
cold start $z = 0$ with zero gradient. Its multiplier enters the exact Hessian as $-2\mu P_f$, the
QP becomes non-convex and HPIPM stops with `MINSTEP` in the second SQP iteration. The fix is
`lh_e = −ACADOS_INFTY`.

## 6. Scenario

```
z_{k+1} = A z_k + B v*(z_k),         z_0 = x_0            nominal MPC, linear, no w
u_k     = f_hover + v*(z_k) + K (x_k − z_k)               ancillary feedback
x_{k+1} = plant(x_k, u_k, d_k)                            nonlinear, d_k ∈ D
```

- Target $p_{\text{ref}} = (2, 1, 3)$ m, start $p_{\text{ref}} + (-1.0, 0.8, -0.5)$ m (1.37 m away),
  at rest.
- $d$ is piecewise constant: a random **vertex** of $\mathbb D$ every 0.5 s (seeded). That is the
  persistent worst-case sign pattern, not a gentle average.
- 15 s, 300 samples.

## 7. Results

All figures are in `results/stage8_tube/`; the numbers are printed by `summarize()`.

| Check | Value | Guarantee |
|---|---|---|
| failed OCP solves / shifted candidate used | 0 / 0 of 300 | |
| gauge $\max \gamma(e)$, every 5th sample | 0.23 | $\le 1$: $e \in S$ (exact) |
| $`\max \lvert e_j \rvert / h_S(e_j)`$, 12 axes | 0.23 (worst $p_z$) | $\le 1$: what $\mathbb X \ominus S$ uses |
| $`\max \lvert (M K e)_i \rvert / h_{KS,i}`$, 4 wrench rows | 0.23 | $\le 1$: what $\mathbb U \ominus K S$ uses |
| $`\max \lvert x_j \rvert / \mathbb X_j`$ | 0.47 | $\le 1$: $x \in \mathbb X$ |
| $`\max \lvert z_j \rvert / \bar{\mathbb X}_j`$ | 0.94 | $\le 1$: $z \in \bar{\mathbb X}$ |
| $`\max \lvert \nu_i \rvert / \mathbb U_i`$ | 0.66 | $\le 1$: $u \in \mathbb U$ |
| $`\max \lvert w_j \rvert / \bar w_j`$ | 0.91 (worst $r$) | $\le 1$: realized $w \in \mathbb W$ |
| motor thrusts | $[2.38, 2.61]$ N | |
| $`\lVert p - p_{\text{ref}} \rVert`$, last 5 s | ≤ 23 mm | tube $\pm 0.20 / 0.20 / 0.06$ m |
| MPC time mean / max | 2–3 / 11–31 ms (wall-clock on WSL, varies between runs) | budget 50 ms |

### 7.1 State, nominal and tube

![tube states](../results/stage8_tube/tube_states.png)

- The nominal (green, dashed) reaches the target in about 6 s. Its pitch comes close to the
  tightened tilt bound $\bar{\mathbb X}$ (orange) at $t \approx 0.4$ s. That is the 0.94 in the
  table: the MPC uses the room it has.
- The true state (blue) stays inside the green band $z \pm h_S$ everywhere, and far inside it: the
  largest error is 23 % of the cross-section. After the transient, $x$ keeps moving around the
  target because $d$ jumps between vertices, and the motion stays a fraction of the tube.
- The band crosses the orange line in the attitude rows at $t \approx 0.5$ s. That is expected:
  $z \in \bar{\mathbb X}$ only guarantees $z \pm h_S \subset \mathbb X$ (red), not
  $z \pm h_S \subset \bar{\mathbb X}$.

### 7.2 Wrench

![tube inputs](../results/stage8_tube/tube_inputs.png)

At $t = 0$ the error is zero, so the applied wrench equals the nominal one, which sits on
$\bar{\mathbb U}$ for $\tau_y$. Afterwards the feedback $K e$ makes up all of the input: the nominal
is $\approx 0$ and the applied wrench follows the disturbance. The saw-tooth in $\delta T$ is the
ancillary feedback answering each new vertex of $d_{f,z}$ every 0.5 s.

### 7.3 The inequalities behind the guarantee

![tube certificate](../results/stage8_tube/tube_certificate.png)

- **Left**: the error against the tube. The curves are the shares along the axes, the black dots
  the exact gauge $\gamma(e)$. The gauge lies above the axis shares, as it must, and stays below
  0.24: $e \in S$ with a factor of 4 to spare.
- **Middle**: the realized one-step model error. The yaw-rate component sits at $0.91 = 1/1.1$.
  Its $\bar w$ is pure $d$, and $d$ is always at a vertex, so only the margin separates it from the
  bound. The other components stay at 25 % ($p, v$) and 2 % ($q_v$) of $\bar w$: the sampled bound
  is the worst case over all of $\mathbb X \times \mathbb U$, and this flight stays near hover.
- **Right**: the disturbance sign pattern.

### 7.4 Flight path, states, motor thrusts

<p align="center">
  <img src="../results/stage8_tube/trajectory_3d.png" alt="3D flight path" width="60%">
</p>

![states](../results/stage8_tube/states.png)

![inputs](../results/stage8_tube/inputs.png)

## 8. Where the conservatism comes from

The guaranteed tube is about four times wider than the error ever was. Three bounds stack up:

1. **$\mathbb W$ is a box over all of $\mathbb X \times \mathbb U$.** The linearization error is
   state-dependent: zero at hover, largest at the corners. The box charges every sample with the
   corner value, which is 4–50× the realized value here (7.3).
2. **Every component at its worst, independently, at every step.** $h_S$ sums the worst case over
   all disturbance directions and all past samples. A realized $w$ is correlated and far from that.
3. **The worst case needs an adversary.** Random vertices of $\mathbb D$ excite the error, but not
   with the sign pattern that would be resonant for $A_K$.

That is the price of a guarantee for every admissible disturbance sequence. Item 1 is specific to
using a linear tube for a nonlinear plant. Tubes that scale with the state (nonlinear or
"homothetic" tubes) or a state-dependent bound on $w$ would shrink it.

## 9. Observations and limitations

- **The nominal does not see the state.** In this form (Rawlings, Mayne & Diehl, §3.5) $z$ evolves
  on its own from $z_0 = x_0$. If $x$ drifts inside the tube, the MPC does not re-plan; the
  ancillary law alone corrects it. Mayne et al. (2005) make $z_0$ a decision variable with the
  constraint $x - z_0 \in S$, which re-centres the tube at every step. That needs an **explicit**
  representation of $S$. The ellipsoid and the modal box were too large here ([§3](#3-the-tube-cross-section-s)), and the
  polytopic $F(\alpha, s)$ would need $12 s \approx 1600$ auxiliary variables in the OCP.
- **Searched, not proven.** $\mathbb W$ is the largest value found by $2.66 \cdot 10^5$ samples and
  a local maximization, times 1.1. A global maximum of a non-convex function is not certified this
  way, as in Stage 2a.
- **A small envelope.** Tilt ≤ 5.7°, body rates ≤ 0.5 rad/s, $\lvert \delta T \rvert \le 0.6$ N. Larger
  sets make the linearization error grow faster than the authority, and no tube fits. Robust MPC on
  the linear model is a method for regulation around an operating point.
- **Full state feedback.** With an estimator, $x$ is not known, and the estimation error is a
  second tube around the estimate (output-feedback tube MPC; Mayne, Raković, Findeisen & Allgöwer,
  2006).
- **No offset-free action.** A constant $d$ leaves a steady error inside the tube. Stage 3's
  disturbance estimate would remove it, but then $d - \hat d$ is the disturbance the tube has to
  cover.

## Files

| File | Role |
|---|---|
| `compute_tube_params.py` | sets $\mathbb X, \mathbb U, \mathbb D$; `linear_model()`, `sample_w_bound()`, `select_gain()`, `tube_support()` ($h_S$ of $F(\alpha, s)$), `tube_gauge()` (exact $e \in S$), `nominal_cost()`, `terminal_set()`; `save/load_tube_params()`; `reduce_state()` / `embed_state()`; `self_test()` (`--check`) |
| `ocp_config_tube.py` | nominal OCP: `create_solver()`, `solve_nominal()`, `shifted_candidate()`, `warm_start()`, `ancillary_input()` |
| `simulate_tube.py` | entry point: scenario, closed loop, `summarize()`, figures → `results/stage8_tube/` |
| `data/tube_params.npz` | $A, B, \bar w, K, h_S, \bar{\mathbb X}, \bar{\mathbb U}, P_f, \alpha_f$, nominal cost |
| `plot_utils.py` | `plot_tube_states()`, `plot_tube_inputs()`, `plot_tube_certificate()` (new) |

```bash
./run.sh compute_tube_params.py             # → data/tube_params.npz (≈ 12 s)
./run.sh compute_tube_params.py --check     # self-test: plant, support function, terminal set
./run.sh simulate_tube.py                   # → results/stage8_tube/
QUAD_PLOTS=none ./run.sh simulate_tube.py   # save only, no windows
```

## References

- D. Q. Mayne, M. M. Seron, S. V. Raković, "Robust model predictive control of constrained linear
  systems with bounded disturbances," *Automatica* 41(2), 2005.
- J. B. Rawlings, D. Q. Mayne, M. M. Diehl, *Model Predictive Control: Theory, Computation, and
  Design*, 2nd ed., Nob Hill Publishing, 2017.
- S. V. Raković, E. C. Kerrigan, K. I. Kouramas, D. Q. Mayne, "Invariant approximations of the
  minimal robust positively invariant set," *IEEE Transactions on Automatic Control* 50(3), 2005.
- I. Kolmanovsky, E. G. Gilbert, "Theory and computation of disturbance invariant sets for
  discrete-time linear systems," *Mathematical Problems in Engineering* 4(4), 1998.
- D. Q. Mayne, S. V. Raković, R. Findeisen, F. Allgöwer, "Robust output feedback model predictive
  control of constrained linear systems," *Automatica* 42(7), 2006.
- A. E. Bryson, Y.-C. Ho, *Applied Optimal Control*, Hemisphere, 1975.
