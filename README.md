# Pineapple Arm

Control, identification and calibration for the 6-actuator pineapple arm, driven over
Unitree DDS at 200 Hz.

The repo is organised as a **pipeline**: identify the dynamics, verify and calibrate the
gravity/torque model, then use it for end-effector pose and trajectory tracking. Every
stage has an **offline mode that needs no hardware** — run it before the hardware
command, every time.

```
 1. sysid/            2. verify_gravity     3. arm_drag         4. pineapple_arm      5. arm_tvlqr
    joint dynamics  ->   g(q) vs measured ->  float & evaluate ->  EE pose (IK)     ->   EE trajectory
    armature/                N_tau, S           tau_cmd_scale        point-to-point       TVLQR tracking
    friction/damping                                |                                        ^
         |                                          v                                        |
         +--> pineapple_arm.xml     model/tau_cmd_scale.json --> arm_ff.motor_tau ------------+
                                    model/gravity_calib.json --> arm_ff.gravity_torque
```

## Hardware

| | |
|---|---|
| Joints (motor order `j0..j5`) | `arm_joint`, `arm_base_joint`, `upper_arm_joint`, `fore_arm_joint`, `5dof_joint`, `gripper_case_joint` |
| Motors | **DM4340** on j0–j2 (±27 N·m), **DM4310** on j3–j5 (±7 N·m) |
| Control rate | 200 Hz (`dt = 0.005`) |
| Transport | `unitree_sdk2py`, `rt/lowcmd` / `rt/lowstate` |
| Low-level law | `tau_applied = A·tau_sent + S·kp·(q_des−q) + S·kd·(dq_des−dq)` — see [Hardware quirks](#hardware-quirks) for why `A` and `S` are **not** 1 |

Motor index, `arm_ik` joint order and the URDF joint order are **one-to-one, same sign,
no offset**, so a solver's output goes straight into `motor_cmd[i].q`.

Joint limits (rad), from `pineapple_arm.xml`:

```
low  = [-1.5708, 0.0,    -3.1416, -1.5708, -1.5708, -1.5708]
high = [ 1.5708, 3.1416,  0.0,     1.7453,  1.5708,  1.5708]
```

> Note `arm_base` is bounded **below** by 0 and `upper_arm` **above** by 0 — the
> all-zeros home pose sits exactly on a limit for both. See quirk 2.

## Environment

Everything runs in the **`mujoco-learning`** conda env — the only one with pinocchio
3.4.0, `unitree_sdk2py` and mujoco 3.10 together. The system `python3` has no pinocchio.

```bash
PY=~/anaconda3/envs/mujoco-learning/bin/python     # used throughout this README
# equivalently:  conda run -n mujoco-learning python
$PY -c "import pinocchio, mujoco, unitree_sdk2py; print('ok')"
```

`sysid/collect_data.py` is the one exception: it needs only `unitree_sdk2py` + numpy (no
MuJoCo, no pinocchio) so it can run in the arm's own runtime.

## Repo map

| file | role | |
|---|---|---|
| `arm_ik.py` | 6-DOF pose IK (pinocchio, damped least squares) | lib |
| `arm_ff.py` | gravity/friction feedforward **and** `motor_tau()`, the joint→command torque conversion | run (prints a demo) |
| `ee_traj.py` | EE path → `q_ref`/`dq_ref`/`ddq_ref`/`tau_ref` via IK + RNEA | lib |
| `pineapple_arm.py` | main DDS controller: PD + feedforward, REPL with point-to-point EE pose | run |
| `pineapple_arm_vis.py` | browser (Viser) 3D view + EE-drag teleop | run |
| `arm_tvlqr.py` | EE **trajectory** tracking with time-varying LQR | run |
| `verify_gravity.py` | measure real gravity torque, fit per-joint scale factors, calibrate | run |
| `arm_drag.py` | float the arm (`kp=0`) so it can be hand-dragged, and score the compensation | run |
| `compare_ee_tracking.py` | MuJoCo A/B of feedforward on vs off | run |
| `test_tvlqr.py` | unit tests for the trajectory/LQR stack (imports `arm_tvlqr`, needs `arm_tvlqr_test.py`) | run |
| `test_frames.py` | guards that `robot.urdf` and the MJCF describe the same robot in the same frame | run |
| `sysid/` | joint-dynamics identification — see [sysid/README.md](sysid/README.md) | |
| `model/` | `robot.urdf` (arm mounted at z=0.192735, matching the MJCF) + calibration overlays | data |
| `data/` | collected logs and generated plots | data |

---

## Stage 1 — System identification

**Produces** per-joint `armature`, `frictionloss`, `damping` → a drop-in
`pineapple_arm_identified.xml`, so the MuJoCo model matches the hardware.

```bash
cd sysid
$PY -m unittest test_sysid_pipeline          # offline
$PY sim2sim_selftest.py                      # offline: inject known params, check recovery
$PY check_excitation.py                      # offline: size amplitudes to the torque/velocity caps
python collect_data.py eth0                  # ON THE ROBOT -> data/arm_chirp_<ts>.npz
$PY check_excitation.py --analyze data/arm_chirp_<ts>.npz    # verify realized peaks
$PY sysid_fit.py --data data/arm_chirp_<ts>.npz --out results/run1
```

Use the result **only** if the fit emits `pineapple_arm_identified.xml` and does *not*
write `INVALID_FIT.txt`.

📖 **Full detail — method, parameter bounds, excitation sizing, diagnosing an
armature gap — is in [sysid/README.md](sysid/README.md).** Not repeated here.

## Stage 2 — Gravity verification and calibration

**Produces** `data/grav_<ts>.npz`, and from it the per-joint factors that the rest of the
pipeline depends on: `N_tau` (reported-torque scale) and `S` (gain-path inflation).

At a settled static pose the measured `tau_est` *is* the joint gravity torque, plus
Coulomb friction whose sign follows the last direction of motion. The sweep approaches
each pose **from below and from above** and averages, cancelling that friction and
leaving ground-truth `g_real(q)` to compare against `arm_ff.gravity_torque(q)`.

```bash
$PY verify_gravity.py --selftest                 # offline: fit recovery, overlay round-trip, gate
$PY verify_gravity.py --dry-run                  # offline: preview the pose sweep
python verify_gravity.py eth0                    # ON THE ROBOT (comp OFF) -> data/grav_<ts>.npz
```

Then analyse offline:

| command | what it gives you | writes |
|---|---|---|
| `--fit-scale <npz>` | **the useful one**: per-joint `N_tau`, stiffness `S`, `S/pi` → gear ratio, with R² and a confidence verdict | `<npz>_scale.png` |
| `--plot-raw <npz>` | only what was logged — no model, no pinocchio; works on an aborted log | `<npz>_raw.png` |
| `--analyze <npz>` | per-joint error report + per-link mass calibration | `<npz>_verify.png`, `<npz>_calib.yaml` |
| `--analyze <npz> --apply` | writes `model/gravity_calib.json` — **only** from a fit that passes the adequacy gate | overlay |
| `--compare A.npz B.npz` | two logs vs the shared model and each other (e.g. sim vs real) | `<A>_vs_<B>_compare.png` |

`--fit-scale` is what the arm actually runs on today. `--analyze`'s mass calibration is
currently **rejected on every log collected** — see [Current state](#current-state).

## Stage 3 — Drag evaluation (gravity compensation, scored)

**Produces** `data/drag_<ts>.npz` + per-joint metrics, and is where
`model/tau_cmd_scale.json` came from.

Stages 1–2 judge gravity compensation *indirectly*. Here the position loop is switched
**off** (`kp = 0`) and the arm is held up by feedforward torque alone: if `tau_ff` truly
cancels gravity the arm floats, and any model error shows immediately as **drift**.
Drift is measured by the **encoder**, a channel independent of the torque sensor — which
matters, because the torque sensor's own scale is one of the things under test.

```bash
$PY arm_drag.py --show-model                 # offline: resolved per-joint model, with reasons
$PY arm_drag.py --show-model --no-ident      # offline: pure-URDF baseline
$PY arm_drag.py --selftest                   # offline: 56 checks
python arm_drag.py eth0                      # ON THE ROBOT
```

REPL:

| command | effect |
|---|---|
| `engage` / `hold` | ramp `kp`→0 and float / restore the position hold in place |
| `trim [<joint> <s>]` | live per-joint gravity scale; bare `trim` prints the table |
| `model [<npz>\|urdf]` | reload the identified fit live (auto-holds first, resets `trim`) |
| `catch [dq <v>\|drift <v>\|on\|off]` | retune or disarm the fall-catch without restarting |
| `fric`, `mon` | friction compensation / 2 Hz live readout |
| `rec [name]` / `stop` | capture a session → npz + metrics + `<npz>_drag.png` |
| `home`, `help`, `exit` | |

`stop` reports, per joint: **`drift_g`** (hands-off drift projected on the pull
direction — *the sign is the verdict*), residual torque, drag effort, and a suggested
trim. With the calibration in place `trim` should sit at **1.0**.

Safety: the fall-catch requires sustained motion **in the gravity-losing direction**
before restoring `kp`, so ordinary dragging does not trip it; a hard watchdog
(`|dq| > 10` rad/s, torque, position) sits behind it.

## Stage 4 — EE pose tracking (point-to-point)

**Produces** motion to a commanded 6-DOF end-effector pose, solved once by IK then
PD-servoed with gravity + friction feedforward.

```bash
python pineapple_arm.py eth0                 # ON THE ROBOT (default iface "lo")
```

| command | effect |
|---|---|
| `pose x y z [roll pitch yaw]` | IK to an EE pose (metres / radians; rpy optional, default 0 0 0) |
| `return` / `move` | zero pose / a demo joint pose |
| `grav [on\|off]`, `fric [on\|off]` | toggle gravity / friction feedforward |
| `exit` | shut down |

IK refuses to command when it has not converged, so an unreachable target moves nothing.

### Browser view + drag teleop

```bash
$PY pineapple_arm_vis.py --self-check        # offline: headless logic check
$PY pineapple_arm_vis.py --sim               # no hardware: the arm follows IK
$PY pineapple_arm_vis.py eth0                # ON THE ROBOT -> http://127.0.0.1:8080
```

Kinematics only (no `mj_step`). Drag a 6-DOF gizmo and **release** to commit — nothing is
commanded mid-drag. Safety model: binds to **localhost** unless you pass
`--host 0.0.0.0 --allow-remote`; teleop starts OFF and must additionally be **armed by
typing `ARM` at the terminal**; startup fails closed if no `rt/lowstate` arrives.

**Motion is planned, not slewed** (2026-08-04). A release calls the same
`arm_smooth_move.plan_to_poses` as Stage 4′ and the 200 Hz publisher plays out
`(q_ref, dq_ref)`. It replaces a constant-velocity slew of the commanded position,
recomputed in the 50 Hz GUI loop and published with `dq_des = 0`. On CENTER→HOME (0.8 rad),
through the `S`-inflated hardware law:

| | old slew, `dq_des=0` | planned, `dq_des=dq_ref` |
|---|---|---|
| peak \|τ\| `arm_base` @ 1 rad/s | 15.66 N·m | **2.34 N·m** |
| peak \|τ\| `arm_base` @ 2 rad/s | **27.0 N·m — pinned at the motor limit, watchdog trips** | **2.38 N·m**, no trip |
| peak commanded acceleration | **200 rad/s²** (cap is 4) | **1.9 rad/s²** |

Two reasons it was that bad: `dq_des = 0` fights the motion with `kd_eff` of 18.8 / 12.6
(~19 N·m for the whole move), and a 0.02 rad step held for four 200 Hz ticks is an
acceleration impulse. The planned profile barely changes with the speed slider because the
**jerk** cap binds first — which is why raising it is now safe, and why the slider is the
planner's velocity cap rather than a slew rate.

Re-planning mid-move is **C²**: `joint_traj.quintic_from_state` seeds the new profile from
the current reference, so interrupting a move continues its velocity and acceleration
instead of stepping them. A reversal takes ~2× longer than a fresh move (braking is bounded
too), so the planned duration is shown in the status panel. Any trajectory failing
`joint_traj.validate` is refused before it is commanded, and the previous motion continues.

## Stage 5 — EE trajectory tracking (moving reference)

**Produces** tracking of a *moving* EE trajectory, stabilised by a time-varying LQR
designed offline along the reference.

`ee_traj.py` turns an EE path into a full joint reference:

```
x_ref(t) --IK--> q_ref --J⁻¹--> dq_ref --diff--> ddq_ref --RNEA--> tau_ref
```

`tau_ref` **already includes gravity** — a controller using it must not also add
`arm_ff.gravity_torque`, or gravity is applied twice.

```bash
# offline: is the reference even feasible? (IK, joint limits, dq/tau caps) -> ee_tvlqr.png
$PY arm_tvlqr.py --dry-run --shape line --p0 0.205 0 0.503 --p1 0.205 0 0.603

# offline: closed-loop comparison against a deliberately wrong plant (pd / ff / lqr)
$PY arm_tvlqr.py --simulate --shape circle --center 0.205 0 0.523 --radius 0.05

# ON THE ROBOT
python arm_tvlqr.py eth0 --shape line --p0 0.205 0 0.503 --p1 0.205 0 0.603 --duration 4
```

Shapes: `hold`, `line` (`--p0/--p1`), `circle` (`--center/--radius/--axis/--turns`),
`waypoints` (`--points x y z x y z ...`). Weights via `--q-pos/--q-vel/--r-tau/--w-ee/--w-rot`;
`--joint-space` drops the task-space cost. Tracking aborts past `--track-abort` rad.

`compare_ee_tracking.py` answers the narrower question — how much does feedforward buy? —
by running the same reference twice in MuJoCo with only `tau_ff` differing:

```bash
$PY compare_ee_tracking.py                   # -> ee_tracking_compare.png
```

---

## Calibration overlays (`model/`)

`robot.urdf` plus two **optional** JSON overlays that every controller picks up at import.
Both are fail-safe — missing, corrupt, wrong joint order or out-of-band values are ignored
with a warning, falling back to uncorrected behaviour — and both are reverted by simply
**deleting the file**.

| file | status | effect |
|---|---|---|
| `tau_cmd_scale.json` | **present** | per-joint commanded-torque scale — measured `0.20 / 0.30 / 1.065` on `arm_base` / `upper_arm` / `fore_arm`, `1.0` on the three gravity-free joints — applied by `arm_ff.motor_tau()` |
| `gravity_calib.json` | **absent by design** | per-link mass scales for `arm_ff.gravity_torque`; would be written by `verify_gravity.py --analyze --apply` |

### The domain rule

`arm_ff.gravity_torque()`, `friction_torque()` and `feedforward()` return **physical joint
torques** and are deliberately left unscaled — `verify_gravity.py` and `test_tvlqr.py`
compare hardware measurements against them, so scaling them would corrupt the
identification. Only `arm_ff.motor_tau()` converts to the command domain:

```python
tau_sent = arm_ff.motor_tau(arm_ff.gravity_torque(q))   # what goes in motor_cmd.tau
```

Every controller that writes `motor_cmd[i].tau` must funnel through it.

## Hardware quirks

Read this before commanding torque. All four were found the hard way.

**1. There are THREE separate scale errors. Do not assume one applies to another channel.**

| channel | error | where it is handled |
|---|---|---|
| commanded torque | too strong by **5.0× on `arm_base`, 3.3× on `upper_arm`**, 0.94× on `fore_arm` — **measured per joint** | `model/tau_cmd_scale.json` → `arm_ff.motor_tau()` |
| gain path (`kp`/`kd`) | `x S = N·pi` (≈18.8 on `arm_base`, 12.5 on `upper_arm`) | `arm_drag` divides `kd` by `S`; `arm_tvlqr` cancels the PD with the effective gains |
| reported torque | `x N_tau` — **varies with firmware**: 6.06 / 3.93 raw, 1.52 / 0.98 with `TAU_MAX=112` applied | reported by `--fit-scale`; **not** applied to commands |

The commanded-torque numbers are **measurements, not a formula** — resist the temptation
to derive them. The gain path gives `S = N·pi`, so the stray `pi` alone would predict
`N/S = 1/pi = 0.318` wherever it is present. `upper_arm` (0.30) and `fore_arm` (1.065 vs
1.0) fit that; **`arm_base` does not** — it needs 0.20, a further ~1.6× below the
prediction, despite being the same motor type as `upper_arm`. So the cause is
per-**joint**, not per motor type, and an earlier `1/pi`-everywhere version of this table
was wrong on `arm_base` by 60%.

Two consequences:

- **`arm_joint` (j0), `5dof_joint` and `gripper_case_joint` are left at 1.0 because they
  carry no gravity and dragging cannot measure them** — that is an absence of data, not a
  statement that they are correct. Gravity compensation is unaffected (their `g` is ~0),
  but `arm_tvlqr`'s `tau_ref` includes *inertial* terms on them that will be off by an
  unknown factor. Measuring them needs an excitation that loads them, not a drag session.
- Re-run `--fit-scale` and **re-tune with `arm_drag.py`** after *any* `damiao.cpp` change.
  `S` has been stable across every hardware log, but `N_tau` moves with firmware, and the
  command scale is not derivable from either.

**2. The all-zeros home pose IS a joint limit** for `arm_base` (low) and `upper_arm`
(high). Any "keep a margin from the stop" logic reads the robot's normal resting pose as
a violation — that produced ±6 N·m at rest and a fall-catch within 20 ms. Measure
penetration from the *actual* limit, gate cushions on outward velocity, and clip
commanded positions to the exact limits.

**3. The effective `kp` is enormous** (502–752 N·m/rad after the `S` inflation), so
feedforward errors are invisible in normal use: 14 N·m of over-torque shows up as
0.028 rad of offset. `pineapple_arm.py` and `arm_tvlqr.py` over-commanded ~3× on j0–j2
for weeks without a symptom. Only `arm_drag.py`, which sets `kp = 0`, exposes them.

**4. `data/grav_*.npz` mixes SIMULATION and HARDWARE logs**, and nothing in the filename
or metadata distinguishes them. The tell is `S`: **≈1.0 means simulation**, **≈`N·pi`
(≈18.8 on `arm_base`, ≈12.5 on `upper_arm`) means hardware**. Half of the current logs
are sim, and one drag session has already been run against a sim-derived model by
accident. Both `verify_gravity` and `arm_drag` default to the *newest* log, so pass
`--log` explicitly whenever it matters. The same `S = 1` that marks a sim log is why the
MuJoCo DDS bridge disagrees with `--simulate` — see
[Sim-to-sim](#sim-to-sim---sim--simulate-vs-the-unitree_mujoco-dds-bridge). To classify what
you have:

```bash
$PY -c "
import numpy as np, glob, os, verify_gravity as VG
for p in sorted(glob.glob('data/grav_2026*.npz')):
    d = {r['j']: r for r in VG.compute_joint_scales(VG.load_log(p))['rows']}
    S = d[1]['S'] if 1 in d else float('nan')
    print(f'{os.path.basename(p):32s} S={S:8.3f}  ' + ('SIM' if S < 2 else 'HARDWARE'))"
```

## Sim-to-sim: `--sim`/`--simulate` vs the `unitree_mujoco` DDS bridge

There are **two** offline ways to run this arm, and they disagree by ~70× on EE error. That
is expected, and this section is why. Same plan, one `--pose 0.205 0 0.523` move, nominal
masses:

| | EE pos RMS | max joint err |
|---|---|---|
| `arm_smooth_move.py --sim --mass-scale 1.0` | **0.61 mm** | 6.1 mrad |
| the same published commands, replayed through the bridge's law in MuJoCo | **43.41 mm** | 175.7 mrad |

The first row reproduces exactly:

```bash
$PY arm_smooth_move.py --sim --pose 0.205 0 0.523 --mass-scale 1.0   # -> EE position RMS 0.61 mm
```

Note `--mass-scale` defaults to **1.15**, a deliberate plant perturbation, which alone takes
that 0.61 mm to 2.19 mm. Always state the mass scale when quoting a `--sim` number.

**The gap is not a MuJoCo-vs-pinocchio physics difference. It is two constants.**

The bridge (`pineapple_mujoco/simulate_python/unitree_sdk2py_bridge.py`, `LowCmdHandler`)
applies the low-cmd **verbatim**:

```python
mj_data.ctrl[i] = (msg.motor_cmd[i].tau
    + msg.motor_cmd[i].kp * (msg.motor_cmd[i].q  - sensordata[i])
    + msg.motor_cmd[i].kd * (msg.motor_cmd[i].dq - sensordata[i + num_motor]))
```

That is `A = 1, S = 1, N_tau = 1` — a **nominal** motor interface. But this robot's measured
command path has none of those (quirk 1), and the `--simulate` rollouts model it:

```python
# arm_tvlqr_test.simulate / arm_smooth_move.simulate
applied = sent / arm_ff.TAU_CMD_SCALE + S*kp*(q_ref - q) + S*kd*(dq_ref - dq)
#         A = 1/TAU_CMD_SCALE = [1, 5.0, 3.33, 0.94, 1, 1]
#         S = GAIN_INFLATION  = [1, 18.81, 12.56, 1, 1, 1]
```

Every controller publishes `arm_ff.motor_tau(tau)`, i.e. **pre-divided by `A`** so the real
drive's over-drive cancels it. The bridge has no over-drive, so that division is not undone:

| at `q = [0, 0.8, −0.8, 0.3, 0, 0]` | j1 `arm_base` | j2 `upper_arm` |
|---|---|---|
| `g(q)` needed | −0.614 | −7.004 |
| `motor_cmd.tau` published | −0.123 | −2.101 |
| arrives on hardware (`×A`) | −0.614 ✓ | −7.004 ✓ |
| **arrives in the bridge (`×1`)** | **−0.123** | **−2.101** |
| gravity deficit | −0.49 N·m | **−4.90 N·m** |
| static sag at `S·kp = 502` (hardware) | 0 rad | 0 rad |
| static sag at `kp = 40` (bridge) | −0.012 rad | **−0.123 rad = 7.0°** |

0.123 rad on `upper_arm` is ~38 mm at the EE — the whole gap. Toggling each factor in a
MuJoCo rollout of the bridge's law confirms the split (same move, nominal masses):

```
bridge as-is (motor_tau ON, S=1)         EE RMS  43.41 mm   <- what you get today
motor_tau OFF (raw joint gravity), S=1   EE RMS   4.29 mm   <- ~90% of the gap is A
motor_tau ON,  kp/kd pre-inflated by S   EE RMS   3.44 mm
motor_tau OFF, kp/kd pre-inflated by S   EE RMS   0.66 mm   <- lands on --sim's 0.61 mm
  ...+ contacts disabled                 EE RMS   0.66 mm
  ...+ frictionloss & damping zeroed      EE RMS   0.62 mm   <- MJCF passives are <0.1 mm
```

The last row converging on `--sim`'s 0.61 mm is the point: once `A` and `S` match, the two
sims agree to ~0.01 mm and **every remaining MJCF-vs-URDF difference is worth less than
0.1 mm** on this move. For comparison, the pinocchio rollout with only `S` removed
(`--sim --mass-scale 1.0 --no-gain-inflation`) gives 4.40 mm against the 4.29 mm above —
the residual 0.11 mm is the MJCF's damping/friction/armature.

**Neither sim is wrong.** `--simulate` predicts *this* robot; the bridge predicts a *nominal*
one. Pick deliberately, and never compare their numbers directly.

### To make a bridge run comparable to `--sim`

Cancel both constants at the source:

```bash
mv model/tau_cmd_scale.json /tmp/            # motor_tau -> identity (fail-safe by design)
$PY arm_smooth_move.py lo --pose 0.205 0 0.523 --no-gain-inflation
```

> ⚠️ **A configuration set up this way must never be pointed at hardware.** Without
> `tau_cmd_scale.json` every feedforward is 5× / 3.3× too strong on `arm_base` / `upper_arm`
> — quirk 1 and quirk 3, which is exactly the failure `arm_drag.py` was built to catch. Put
> the file back before any `eth0` run, and confirm with `$PY arm_drag.py --show-model`.

### Second-order differences (not the cause, but they govern stability)

| | `--simulate` (pinocchio / URDF) | bridge (MuJoCo / MJCF) |
|---|---|---|
| joint damping | 0 — `robot.urdf` has no `<dynamics>` | 0.02, integrated implicitly |
| frictionloss | 0 | 0.1, solved as a constraint (real stiction) |
| armature | 0 | 0.004 — **277 % of `gripper_case`'s 0.00144 kg·m²**; the `2I/dt` discrete-`kd` limit is 2.18 with it, 0.576 without, and `KD[5] = 0.5` is 87 % of the bare limit |
| DOF | 6, fingers welded (`ee_traj.build_arm_model`) | 8, finger slides free |
| contacts | none | enabled; at rest `upper_arm`↔`6dof` carry 10.91 N each and `arm_base` hangs on its stop at 2.197 N·m, so the resting pose is **not** zeros |
| latency | none, synchronous | ZOH `ctrl` + `BQueue(10)` + a 200 Hz `lowstate` timer phase-independent of `mj_step`, all under one GIL across 6 threads |
| real time | none, numpy loop | wall-clock paced, and `step_start` is re-read each iteration so overruns are **never** compensated — sim time silently falls behind |
| torque clamp | `arm_ff.TAU_LIMIT = [10,27,27,10,7,7]` | `ctrlrange = [27,27,27,7,7,7]` |
| `tau_est` | exactly the applied torque | `jointactuatorfrc`, so `N_tau = 1` vs 6.06 / 3.93 on hardware — every `SAFETY_TAU` trip means something different |

Link masses are identical (8.6 kg) and MuJoCo's `qfrc_bias` matches
`arm_ff.gravity_torque` to 1.7e-6 N·m, so **`g(q)` itself is not a source of gap** —
`test_frames.py` pins that.

### Traps

- **`fore_arm` (j3) is clamped at ±7 N·m by MuJoCo but ±10 by `arm_ff`.** Its reference
  already runs at `1.58 / 9.0 N·m`, so the bridge silently truncates commands the offline
  run accepted. The `# [27,27,27,7,7,7]` comment in `arm_tvlqr.py` is stale.
- **`import arm_tvlqr` resolves to two different interface models.** `arm_tvlqr.py` has no
  `GAIN_INFLATION`, no `TAU_CMD_SCALE` and no `motor_tau`, so its `simulate()` *is*
  byte-for-byte the bridge's law — which is why it agrees with the bridge (0.10 vs 0.71 mm)
  while `arm_tvlqr_test.py` does not (5.91 vs 105.56 mm). **`arm_tvlqr_test.py` is the
  authoritative implementation**: `test_tvlqr.py` does `import arm_tvlqr as L` but uses
  `L.TRACK_KP`, `L.GAIN_INFLATION`, `L.clamp_applied`, `L.torque_headroom`,
  `L.effective_gains` — none of which exist in `arm_tvlqr.py`. All 65 tests pass against
  `arm_tvlqr_test.py` and 29 error out against `arm_tvlqr.py`. Renaming is pending work.
- `arm_convex_mpc.py` sets `GAIN_INFLATION = np.ones(6)` deliberately, making it a *third*
  interface model — it agrees with the bridge on gains but not on torque.
- **`arm_tvlqr.py` insets `q_des` by 0.05 rad from the limits.** `arm_base`'s low limit and
  `upper_arm`'s high limit are exactly 0.0 and the bridge rests at
  `[0, −0.0006, −0.0169, …]`, so a hold-at-zero becomes a 33–50 mrad step at `t=0`. That is
  quirk 2 again; `arm_smooth_move.py` and `joint_traj.clip_to_limits` use the exact limits.
- Bridge config must be `ROBOT = "pineapple_arm"`, `DOMAIN_ID = 1`, `INTERFACE = "lo"`, and
  it must be launched **from `simulate_python/`** because `ROBOT_SCENE` is relative. Setting
  `ROBOT` back to `pineapple_v2_arm` silently switches it to the `unitree_hg` IDL, which
  nothing in this repo can talk to.

### One frame, as of 2026-08-04

`model/robot.urdf` used to mount the arm at `z = 0.12` while the MJCF used `0.192735`, so
the MuJoCo EE sat a constant **+72.735 mm** above the pinocchio EE that IK and `ee_traj`
plan in — 72.7 mm of frame error against tracking numbers of 4–15 mm. The URDF now carries
`0.192735` and the two frames are identical; `test_frames.py` fails if they ever diverge
again. Every hard-coded EE `z` moved by `+0.072735` to keep the same physical motion
(`0.43 → 0.503`, `0.45 → 0.523`, `0.53 → 0.603`), so **older notes, plots and logged EE
targets are in the old frame** — subtract 72.735 mm to compare. Gravity is unaffected
(translation invariance), so no calibration was re-run.

## Testing

Offline-first. **Everything below runs with no hardware and no DDS** — run the check for
a stage before its hardware command.

| command | covers |
|---|---|
| `$PY -m unittest test_frames` | 7 tests: URDF↔MJCF frame agreement, gravity translation-invariance, IK target shift |
| `$PY -m unittest test_tvlqr` | 65 tests: EE paths, IK reference, LQR/Riccati, gain validation, closed loop. **Written against `arm_tvlqr_test.py`** — 29 error against `arm_tvlqr.py`, see [Sim-to-sim traps](#traps) |
| `$PY -m unittest test_smooth_move` | 77 tests: jerk-limited planning, moving-start profile + splice continuity, limit handling, command guards |
| `$PY -m unittest test_convex_mpc` | 29 tests: box-QP, MPC rollout, torque margins |
| `cd sysid && $PY -m unittest test_sysid_pipeline` | 20 tests: log validation, resampling, MjSpec modifiers |
| `$PY arm_drag.py --selftest` | 56 checks: model resolution, engage/catch, limit barrier, command guards, metrics |
| `$PY verify_gravity.py --selftest` | fit recovery, overlay round-trip, fail-safe, collection watchdog, calibration gate |
| `$PY sysid/sim2sim_selftest.py` | injects known parameters, checks the pipeline recovers them (~1%) |
| `$PY pineapple_arm_vis.py --self-check` | headless teleop logic: commit gating, planned motion, publish-clock pacing, C² re-plan splice, watchdog |
| `$PY arm_tvlqr.py --dry-run` | reference feasibility (IK, limits, dq/tau caps) |
| `$PY arm_tvlqr.py --simulate` | closed-loop pd/ff/lqr comparison against a perturbed plant — **currently exits 1 by design**, see [Current state](#current-state). Models the hardware interface, so it does **not** predict the DDS bridge: [Sim-to-sim](#sim-to-sim---sim--simulate-vs-the-unitree_mujoco-dds-bridge) |
| `$PY arm_smooth_move.py --sim` | jerk-limited move against the S-inflated plant — same caveat as above |
| `$PY compare_ee_tracking.py` | feedforward on vs off, in MuJoCo |
| `$PY arm_drag.py --show-model` | which per-joint factors are live, and why |

### On the robot

- Clear the workspace; keep a hand near the forearm.
- **Ctrl-C is safe.** Every DDS controller here has a guaranteed `safe_return` that ramps
  the arm down, and a debounced watchdog on torque / velocity / position / state
  staleness.
- `arm_drag.py` is the one mode with **no position hold** — the arm is held by
  feedforward alone and *will* fall if the model is wrong. Engage slowly.

## Current state

Honest status as of 2026-08-04:

- **`pineapple_arm_vis.py`'s teleop is jerk-limited.** It plans with
  `arm_smooth_move.plan_to_poses` and publishes a real `dq_ref` instead of slewing with
  `dq_des = 0`; peak `arm_base` torque on a home move drops 15.7 → 2.3 N·m and the 2 rad/s
  case no longer pins the motor limit. New: `joint_traj.quintic_from_state` (moving-start
  quintic, so mid-move re-planning is C²) and `plan_to_poses(rot=, dq_start=, ddq_start=)`.
  **`pineapple_arm.py`'s own REPL still uses the old linear blend** — it was left alone.
- **The URDF↔MJCF frame mismatch is fixed.** `model/robot.urdf` now mounts the arm at
  `z = 0.192735` like the MJCF, so IK, `ee_traj` and MuJoCo finally share one frame, and
  `test_frames.py` (7 tests) keeps them there. Every hard-coded EE `z` moved `+0.072735`;
  gravity, `tau_cmd_scale.json` and all `grav_*.npz` are unaffected. **Pre-existing plots,
  notes and logged EE targets are still in the old frame.**
- **The `--simulate`-vs-DDS-bridge gap is understood and documented, not fixed** — see
  [Sim-to-sim](#sim-to-sim---sim--simulate-vs-the-unitree_mujoco-dds-bridge). The bridge
  applies `A = S = 1`; the offline rollouts model the measured `A` and `S`. Both are kept
  as-is on purpose. Emulating `A`/`S` in the bridge is unstarted work.
- **`test_tvlqr.py` errors 29 of 65 tests against `arm_tvlqr.py`** and passes all 65 against
  `arm_tvlqr_test.py`, which is the authoritative implementation. The module rename is
  unstarted.
- **`sysid/results/latest` is an `INVALID_FIT`** — the distal armature/damping estimates
  hit their bounds. The marker file is intentional; re-collect before producing a
  replacement model.
- **`model/gravity_calib.json` is absent on purpose.** The per-link mass fit is rejected
  on every log collected so far (`upper_arm` scale −1.46, `5dof` 0.001). Don't chase it:
  the residual is a per-joint **scale** error, not a mass error, so a mass-space fit can
  only explain it with unphysical masses.
- **`arm_tvlqr` needs an LQR weight retune, and `--simulate` exits 1 to say so.** Its
  weights were tuned assuming an effective `kp` of 40, but it is really 502–752, so
  cancelling that hardware PD gives up *position* accuracy. On the default line:

  | | plain PD | PD + feedforward | TVLQR |
  |---|---|---|---|
  | EE position RMS | 15.32 mm | **4.48 mm** | 5.08 mm |
  | EE orientation RMS | 5.78° | 1.51° | **0.96°** |

  So TVLQR is 1.58× better on orientation but 0.88× on position, and `--simulate` judges
  on position alone — hence the nonzero exit. It is not broken. Retuning `Q_POS`/`W_EE`
  against the true gains, or publishing `kp`/`kd` divided by `S`, is the fix.
- **`arm_drag`'s `creep` catch** can fire on a continuous, no-pause sweep of more than
  1.5 rad in the gravity direction. `catch drift <larger>` or `catch off` covers it.
- **No `.gitignore`**, and `data/`, `model/` and several root scripts are untracked.
  Generated artifacts (`*.png`, `*_calib.yaml`, `MUJOCO_LOG.TXT`, `__pycache__/`) are
  currently mixed in with source.
