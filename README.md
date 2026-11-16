# 3D Quadrotor NMPC

Nonlinear model predictive control for a 3D quadrotor in Python, using the
[acados](https://docs.acados.org/) solver and the [CasADi](https://web.casadi.org/) symbolic
framework. The project works through MPC theory stage by stage, from terminal ingredients and
stability to offset-free control, state estimation, tracking of piece-wise constant
references and of time-varying trajectories, obstacle avoidance and robust tube MPC. Every stage runs on the same quadrotor
model and changes one ingredient at a time, so the effect of each method can be compared
directly.

Each stage has its own document with the theory, the implementation, the scenario and the
results with plots. This README gives the overview.

## Quadrotor model

| Property | Value |
|----------|-------|
| Rigid body (13) | position, velocity (world frame), unit quaternion (scalar-first), body rates |
| Input (4) | thrust **commands** to the four ESCs, $`f_{\min} \le f_{\text{cmd},i} \le f_{\max} = 3 f_{\text{hover}} \approx 7.36`$ N, idle thrust $f_{\min} \approx 0.074$ N |
| Plant (17) | rigid body + rotor speeds $\Omega$: inverse thrust map $`\omega_{\text{cmd}} = \sqrt{f_{\text{cmd}}/k_f}`$, ESC speed loop $`\dot\Omega = (\omega_{\text{cmd}} - \Omega)/\tau_m`$ with $\tau_m = 30$ ms, thrust $`f = k_f \Omega^2`$ |
| MPC prediction model (17) | rigid body + rotor thrusts $f$ with the lag $`\dot f = (f_{\text{cmd}} - f)/\tau_m`$ (equal to the plant to first order); rotor speeds measured, passed as $`f = k_f \Omega^2`$ |
| Migration to the motor dynamics | done: Stages 1–2. Stages 3–9 still use the 13-state model with the thrusts as direct input; the 13-state version of all stages is tag `rigid-body` |
| Layout | `+` configuration, hover thrust $f_{\text{hover}} = mg/4 \approx 2.45$ N |
| Disturbance (Stages 3–4) | constant force (world frame) and torque (body frame), $d \in \mathbb{R}^6$ |
| Reference trajectory (Stage 6) | analytic helix, mapped by differential flatness to a full state and input reference $(x_r(t), u_r(t))$ |
| Tracking model (Stage 5) | state augmented by an artificial reference position, $`\tilde x = [x;\, p_a] \in \mathbb{R}^{16}`$, $\dot p_a = 0$ |
| Integration | ERK4 (acados) |

Equations of motion, motor dynamics, disturbance model and quaternion conventions →
[docs/quadrotor_model.md](docs/quadrotor_model.md)

## Stages

| Tag | Stage | Method | Document |
|-----|-------|--------|----------|
| `stage1` | 1 | **Basic NMPC** — terminal cost $W_e = Q$, no terminal constraint | [stage1_basic_nmpc.md](docs/stage1_basic_nmpc.md) |
| `stage2` | 2a | **Quasi-infinite horizon** — terminal cost $P_{\text{lyap}}$ + soft terminal set $\Omega_\alpha$ | [stage2_terminal_cost.md](docs/stage2_terminal_cost.md) |
| `stage2` | 2b | **DARE terminal cost** — $P_{\text{lqr}}$ from the reduced-order DARE, no terminal set | [stage2_terminal_cost.md](docs/stage2_terminal_cost.md) |
| `stage3` | 3 | **Offset-free NMPC** — disturbance model + EKF + analytical steady-state target calculator | [stage3_offset_free.md](docs/stage3_offset_free.md) |
| `stage4` | 4a | **Partial measurement, augmented EKF** — only $`y = [p;\, q;\, \omega]`$ measured, with noise | [stage4a_ekf.md](docs/stage4a_ekf.md) |
| `stage4` | 4b | **Lyapunov MHE** — weights and horizon from an LMI detectability certificate; comparison with the EKF | [stage4b_mhe.md](docs/stage4b_mhe.md) |
| `stage5` | 5 | **Tracking NMPC with artificial reference** — piece-wise constant references, recursively feasible for any reference, converges to the closest admissible steady state (nominal plant, full state feedback) | [stage5_tracking.md](docs/stage5_tracking.md) |
| `stage6` | 6 | **Trajectory-tracking NMPC** — helix reference, flatness feedforward $(x_r, u_r)$, time-indexed reference over the horizon, quaternion-error cost; noisy $`y = [p;\, q;\, \omega]`$ with the Stage 4b MHE | [stage6_trajectory.md](docs/stage6_trajectory.md) |
| `stage7` | 7a | **Obstacle avoidance with state constraints** — vertical pillars on the Stage 6 helix as soft squared-distance constraints $h_j(p) \ge 0$ (exact penalty), warm start projected onto the free space (nominal plant, full state feedback) | [stage7a_obst_dist.md](docs/stage7a_obst_dist.md) |
| `stage7` | 7b | **Discrete-time control barrier function** — same scene, constraint $`h_j(x_{i+1}) \ge (1-\gamma)\, h_j(x_i)`$ with the next state as an RK4 step inside the constraint; $\gamma$ sweep | [stage7b_obst_dcbf.md](docs/stage7b_obst_dcbf.md) |
| `stage8` | 8 | **Tube-based robust MPC** — nominal MPC on the hover linearization + ancillary LQR feedback; the disturbance set $\mathbb W$ bounds $d$ *and* the linearization error of the nonlinear plant (sampled), constraints tightened by the support function of the mRPI set (nonlinear plant, full state feedback) | [stage8_tube.md](docs/stage8_tube.md) |
| `stage9` | 9 | **Stochastic MPC with chance constraints** — Gaussian disturbance, deterministic equivalent on the hover linearization: error covariance propagated offline, state and input constraints tightened by $\Phi^{-1}(1-\beta)\,\sigma_k$ per prediction step (Cantelli factor as distribution-free option), $z_0 = x(t)$, soft state rows; Monte Carlo check on the nonlinear plant | [stage9_smpc.md](docs/stage9_smpc.md) |

## Key results

| Stage | Result |
|-------|--------|
| 1 | Hover-to-hover step with motor dynamics converges in about 2 s with no visible offset; the MPC leads the 30 ms rotor lag with its commands. $W_e = Q$ gives no stability certificate. |
| 2a | Certified by the decrease condition itself (sampling + local maximization) with a tuned auxiliary gain, the terminal set is still small: about ±7 mm in position, ±1° tilt. The trajectory enters it after 1.05 s, so the guarantee covers only the final approach. |
| 2b | Same nominal trajectory as Stage 1, now with the LQR cost-to-go as terminal cost (DARE weights scaled by $T_s$ to match acados' cost scaling); used as the terminal cost from Stage 3 on. |
| 3 | Under wind and a 30% mass error, standard NMPC settles 58 / −36 / −79 mm off target; offset-free NMPC removes the offset to below 0.05 mm. |
| 4a | With only $`[p;\, q;\, \omega]`$ measured, the EKF keeps tracking unbiased; velocity estimation error 53 mm/s, force-disturbance error 0.14 N, 0.7 ms per step. |
| 4b | The certified MHE ($T = 3$ s $> T_{\min} = 2$ s) estimates velocity $5\times$ and the disturbance $18\times$ more accurately than the EKF, at 14 ms per step (budget 50 ms). |
| 5 | A reference sequence with 6 m jumps, the last one outside the room ($x = 12$ m), is flown on the nominal plant. The artificial reference $p_a^\ast$ runs ahead of the quadrotor and stops at the boundary of the steady-state set; the position settles at the closest admissible point $y_r^\ast = (8, 6, 2)$ m. The flight is aggressive (tilt up to ≈ 65°, body rates up to ≈ 6.6 rad/s, motors saturated at the jumps). |
| 6 | A 22.9 s climbing helix (2 m/s, tilt up to 17.5°) is flown with only noisy $`[p;\, q;\, \omega]`$ measured and the certified MHE in the loop: 10 mm RMS position error and 0.65° attitude error, 0 failed solves, MPC ≈ 3 ms per call. With full state feedback the same controller tracks to below 0.2 mm. |
| 7a | Three pillars stand on the helix, crossed 9 times in 3 turns. The quadrotor leaves the reference about one horizon (1 s) before each pillar, passes it on the near side 0.35 m off the reference and keeps exactly the 5 cm margin (47.6 mm between samples); 0 failed solves, slack at solver tolerance, MPC ≈ 5 ms per call. |
| 7b | The DCBF with $\gamma = 0.2$ starts the avoidance 0.25 s earlier and keeps 85–111 mm instead of 50 mm from the pillars, with smaller step changes of the body rates. The $\gamma$ sweep shows the trade-off: a smaller $\gamma$ avoids earlier and farther, but below $\gamma \approx 0.2$ it binds for all pillars at once, the rate bound is hit in up to 36 % of the samples and the tilt reaches 40°. |
| 8 | Regulation to a hover point 1.4 m away under a time-varying disturbance at the vertices of $\mathbb D$, on the nonlinear plant. Every guaranteed inequality holds at all 300 samples: error inside the tube (at most 23 % of its cross-section), state and wrench inside $\mathbb X$, $\mathbb U$, realized model error inside $\mathbb W$; 0 failed solves, MPC ≈ 2 ms per call. The linearization error is 3.5–100× larger than the effect of $d$ and sets the tube width (±0.20 m horizontally), which limits the method to a small envelope (tilt ≤ 5.7°). |
| 9 | Regulation towards a hover point 0.1 m behind a wall under Gaussian gusts ($\sigma_f = 0.5$ N), 300 closed-loop runs on the nonlinear plant. Open-loop prediction verified on the nonlinear plant (coverage 0.946–0.954 for the target 0.95). Closed loop: wall violated in at most 0.33 % of the runs at any sample ($\beta = 5$ %), tilt never, motor limits hard; 0 failed solves in 60 000, the soft state rows were needed in 20. The drone hovers 50 mm from the wall; re-planning from the measured state makes the closed loop far more cautious than $\beta$ allows. |

## File structure

```
quadrotor_3d_nmpc/
├── core/                         # Shared by all stages
│   ├── quadrotor_model.py        #   CasADi dynamics (nominal, disturbance-parameterized,
│   │                             #   augmented [x;d], MHE model), hover linearization,
│   │                             #   AcadosSimSolver plant, quaternion utilities,
│   │                             #   measurement model h(z), tracking model
│   │                             #   [x; p_a] and steady-state map g(p_a); motor
│   │                             #   dynamics: thrust-lag MPC model [x; f], rotor plant [x; Ω]
│   ├── motor_check.py            #   checks of the motor models (equilibrium, linearization, steps)
│   ├── plot_utils.py             #   all figures
│   ├── sensor_simulator.py       #   y = [p; q; ω] + Gaussian noise (Stages 4, 6)
│   ├── ss_target.py              #   analytical steady-state target (Stages 3–4)
│   ├── terminal_cost.py          #   reduced-order DARE terminal cost P_lqr (Stages 2b, 3–4, 6)
│   ├── integrators.py            #   symbolic RK4 step of an acados model (Stages 7b, 8)
│   └── codegen.py                #   build/ location of the generated solver code
│
├── stage1_basic/                 # Stage 1   — W_e = Q
│   ├── ocp_config_basic.py
│   └── simulate_basic.py         #             entry point
├── stage2_terminal/              # Stage 2   — terminal ingredients
│   ├── compute_qih_params.py     #   2a: offline CARE / Lyapunov P, α (decrease condition), gain sweep
│   ├── ocp_config_qih.py         #   2a: P_lyap + soft terminal set
│   ├── simulate_qih.py           #   2a: entry point (+ terminal-set diagnostic)
│   ├── ocp_config_dare.py        #   2b: P_lqr terminal cost
│   └── simulate_dare.py          #   2b: entry point
├── stage3_offsetfree/            # Stage 3   — offset-free NMPC, full state feedback
│   ├── ocp_config_offsetfree.py  #             d̂ as runtime parameter, DARE terminal cost,
│   │                             #             soft |ω| box (reused in Stage 4)
│   ├── ekf_aug.py                #             augmented EKF, full-state measurement
│   └── simulate_offsetfree.py    #             entry point (offset-free switched on at t = 4 s)
├── stage4_estimation/            # Stage 4   — partial noisy measurement, EKF (4a) / MHE (4b)
│   ├── observability_check.py    #   observability of (F, H) at hover
│   ├── state_est_ekf.py          #   4a: augmented EKF
│   ├── detectability_check.py    #   4b: LMI detectability certificate → data/mhe_params.npz
│   ├── ocp_config_mhe.py         #   4b: acados MHE OCP + certificate I/O
│   ├── state_est_mhe.py          #   4b: MHE runtime wrapper (buffers, prior, warm start)
│   ├── closed_loop_sim_config.py #   the one closed loop both estimators run in
│   ├── simulate_ekf.py           #   4a: entry point
│   ├── simulate_mhe.py           #   4b: entry point
│   └── simulate_compare.py       #   EKF vs MHE from the saved runs
├── stage5_tracking/              # Stage 5   — tracking MPC with artificial reference
│   ├── ocp_config_tracking.py    #             p_a as extra state, x_N = x_a, offset cost V_o
│   └── simulate_tracking.py      #             entry point (reference jumps at t = 0, 5, 10 s)
├── stage6_trajectory/            # Stage 6   — helix tracking
│   ├── reference_generate.py     #             helix reference (p … snap, yaw), smooth up to snap
│   ├── reference_flatness.py     #             differential flatness: (p, ψ, derivatives) → (x_r, u_r)
│   ├── ocp_config_trajectory.py  #             per-node reference, quaternion state error, warm start
│   └── simulate_trajectory.py    #             entry point (MHE run; --nominal verification)
├── stage7_obstacle/              # Stage 7   — pillars on the helix
│   ├── ocp_config_obstacle.py    #             Stage 6 OCP + soft pillar constraints: distance form
│   │                             #             h_j(p) ≥ 0 (7a) or DCBF (7b), projected warm start
│   ├── obstacle_sim_config.py    #             shared scene + closed loop
│   ├── simulate_obst_dist.py     #   7a: entry point, distance constraint
│   └── simulate_obst_dcbf.py     #   7b: entry point, DCBF (--sweep over γ)
├── stage8_tube/                  # Stage 8   — robust tube MPC
│   ├── compute_tube_params.py    #             offline: linear model, sampled W, ancillary K,
│   │                             #             mRPI support function, tightening, terminal set
│   ├── ocp_config_tube.py        #             nominal MPC (linear, tightened sets, terminal ellipsoid)
│   └── simulate_tube.py          #             entry point: tube MPC on the nonlinear plant
├── stage9_smpc/                  # Stage 9   — stochastic MPC
│   ├── compute_smpc_params.py    #             offline: Σ_w, feedback K, error covariance Σ_e(k),
│   │                             #             tightened bounds θ_k, η_k (Gaussian / Cantelli)
│   ├── ocp_config_smpc.py        #             deterministic-equivalent QP (per-node bounds, soft state rows)
│   └── simulate_smpc.py          #             entry point: 300 closed loops on the nonlinear plant
│
├── data/                         # Offline results read at run time
│   ├── mhe_params.npz            #   Stage 4b certificate (P, Q, R, λ, T_min, envelope)
│   ├── tube_params.npz           #   Stage 8 tube parameters
│   └── smpc_params.npz           #   Stage 9 SMPC parameters
├── run.sh                        # WSL bridge launcher (Windows Git Bash → WSL2)
├── docs/
│   ├── quadrotor_model.md
│   ├── stage1_basic_nmpc.md
│   ├── stage2_terminal_cost.md
│   ├── stage3_offset_free.md
│   ├── stage4a_ekf.md
│   ├── stage4b_mhe.md
│   ├── stage5_tracking.md
│   ├── stage6_trajectory.md
│   ├── stage7a_obst_dist.md
│   ├── stage7b_obst_dcbf.md
│   ├── stage8_tube.md
│   └── stage9_smpc.md
├── results/                      # Figures per stage (tracked)
│   ├── stage1_basic/
│   ├── stage2a_qih/
│   ├── stage2b_dare/
│   ├── stage3_offsetfree/
│   ├── stage4a_ekf/              # figures + sim_data.npz
│   ├── stage4b_mhe/              # figures + sim_data.npz
│   ├── stage4_compare/
│   ├── stage5_tracking/
│   ├── stage6_trajectory/        # MHE run; nominal/ = verification run
│   ├── stage7a_obst_dist/
│   ├── stage7b_obst_dcbf/
│   ├── stage8_tube/
│   └── stage9_smpc/
└── build/                        # acados generated code + solver JSON (gitignored)
```

Each stage is a Python package; code used by several stages lives in `core/`. A later stage
imports what it builds on directly from the earlier stage (e.g. Stage 7 extends the Stage 6 OCP,
Stage 9 reuses the Stage 8 coordinates), so the imports show the dependencies between stages.
Shared code only grows backward-compatibly: new stages add functions or optional keyword
arguments, so earlier stages keep running unchanged. The tags `stage1` … `stage9` are snapshots
of the code as it was when each stage was finished (flat layout, all files in the root). OCP
configuration is kept separate from the simulation loop, so methods can be swapped and compared
side by side.

## Toolchain

- **acados** (Python interface, `acados_template`) — OCP solver for the NMPC and the MHE
- **CasADi** 3.7.2 — symbolic dynamics and automatic differentiation
- **NumPy** 2.x, **SciPy** — offline computations (CARE, DARE, Lyapunov equations)
- **CVXPY** + **MOSEK** — SDP for the Stage 4b certificate (only `stage4_estimation/detectability_check.py`)
- **WSL2 / Ubuntu** — acados C libraries and Python venv (`~/acados_env`)
- **VS Code** on Windows for editing; execution routed through the WSL bridge

## Running

The project files live on the Windows drive; acados runs in WSL2. `run.sh` handles the hop:

```bash
# From Windows Git Bash, in the project directory (main branch)
./run.sh stage1_basic/simulate_basic.py                 # Stage 1              → results/stage1_basic/
./run.sh stage2_terminal/simulate_qih.py                # Stage 2a             → results/stage2a_qih/
./run.sh stage2_terminal/simulate_dare.py               # Stage 2b             → results/stage2b_dare/
./run.sh stage3_offsetfree/simulate_offsetfree.py       # Stage 3              → results/stage3_offsetfree/
./run.sh stage4_estimation/simulate_ekf.py              # Stage 4a closed loop → results/stage4a_ekf/
./run.sh stage4_estimation/simulate_mhe.py              # Stage 4b closed loop → results/stage4b_mhe/
./run.sh stage4_estimation/simulate_compare.py          # EKF vs MHE           → results/stage4_compare/
./run.sh stage5_tracking/simulate_tracking.py           # Stage 5              → results/stage5_tracking/
./run.sh stage6_trajectory/simulate_trajectory.py       # Stage 6 (MHE)        → results/stage6_trajectory/
./run.sh stage6_trajectory/simulate_trajectory.py --nominal  # Stage 6 check   → results/stage6_trajectory/nominal/
./run.sh stage7_obstacle/simulate_obst_dist.py          # Stage 7a             → results/stage7a_obst_dist/
./run.sh stage7_obstacle/simulate_obst_dcbf.py [--sweep]     # Stage 7b        → results/stage7b_obst_dcbf/
./run.sh stage8_tube/compute_tube_params.py             # Stage 8 offline      → data/tube_params.npz
./run.sh stage8_tube/simulate_tube.py                   # Stage 8              → results/stage8_tube/
./run.sh stage9_smpc/compute_smpc_params.py             # Stage 9 offline      → data/smpc_params.npz
./run.sh stage9_smpc/simulate_smpc.py                   # Stage 9 (300 runs)   → results/stage9_smpc/

./run.sh stage4_estimation/observability_check.py       # observability at hover
./run.sh stage4_estimation/detectability_check.py       # re-derive the MHE certificate (needs cvxpy + MOSEK)

# Scripts run as modules from the repo root (they import each other by package name);
# run.sh turns stageN_x/script.py into -m stageN_x.script.

# The code as it was at the end of a stage: check out its tag, e.g.
git checkout stage3 && ./run.sh simulate_offsetfree.py   # back with: git checkout main

# Quick environment check
./run.sh -c "import acados_template; print('ok')"
```

Or directly inside WSL:

```bash
source ~/acados_env/bin/activate
cd /path/to/quadrotor_3d_nmpc        # the repo root (on a Windows drive: /mnt/<drive>/...)
python -m stage4_estimation.simulate_ekf
```

The first solver build compiles C code (about 30 s) into `build/`, which is regenerated on later
runs and is gitignored.
