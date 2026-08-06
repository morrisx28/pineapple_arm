# Pineapple Arm — System Identification (MuJoCo `sysid`)

Identify the **per-joint dynamics** of the 6-actuator pineapple arm — `armature`
(reflected rotor inertia), `frictionloss` (Coulomb/dry friction) and `damping`
(viscous friction) — so the MuJoCo model matches the real hardware. In
`pineapple_arm.xml` these are currently shared placeholders
(`armature=0.004 damping=0.02 frictionloss=0.1`); this pipeline replaces them
with fitted, per-joint values.

Method: DeepMind's official **`mujoco.sysid`** toolbox — box-constrained
nonlinear least squares over a threaded `mujoco.rollout`, with **multiple
shooting** (short re-initialised segments) so the open-loop rollout stays close
to the data. Data is collected on the real arm the way `pineapple_arm.py` drives
it (PD position control), then the measured joint torque is **replayed** through
the model and the parameters are tuned to reproduce the measured motion.

## Files

| file | what it does |
|------|--------------|
| `sysid_common.py`     | shared config: model path, joint/actuator order (= Unitree motor index 0..5), parameter bounds, log validation/resampling, `MjSpec` modifiers, identified-XML writer |
| `collect_data.py`     | **on the real arm** — runs smooth endpoint-tapered per-joint excitation, atomically logs `q/dq/tau` + state/command timestamps, and aborts on stale state or live torque/velocity limits. Standalone (no MuJoCo). |
| `check_excitation.py` | no hardware — sizes the per-joint amplitudes to the torque/velocity caps (bridge-accurate clamped-PD sim), reports peak torque/velocity/saturation, and runs an observability fit so you can confirm armature is identifiable. Run it whenever `CENTER`, the gains, or the model change. |
| `sysid_fit.py`        | validate and fit the 18 parameters (6 joints × {armature, frictionloss, damping}); refuses unsafe/legacy data and does not emit a drop-in XML when optimization fails or parameters hit bounds |
| `sim2sim_selftest.py` | no hardware — inject known parameters, synthesise data, verify the pipeline recovers them |

## Environment

Runs in the **`mujoco-learning`** conda env. `mujoco.sysid` needs mujoco ≥ 3.5
(this env was upgraded to 3.10) plus a few reporting deps:

```bash
conda run -n mujoco-learning pip install -U mujoco tabulate plotly   # matplotlib/scipy already present
conda run -n mujoco-learning python -c "import mujoco.sysid"          # should succeed
```

`collect_data.py` runs on the robot and only needs `unitree_sdk2py` + numpy
(no MuJoCo), so it can run in the arm's existing runtime environment.

## Workflow

**1. Validate the pipeline offline (recommended first):**
```bash
cd sysid
conda run -n mujoco-learning python -m unittest -v test_sysid_pipeline.py
conda run -n mujoco-learning python sim2sim_selftest.py          # expect: PASS, ~1% error
conda run -n mujoco-learning python sim2sim_selftest.py --drive pd
```

**2. Size/verify the excitation, then collect on the real arm:**
```bash
conda run -n mujoco-learning python check_excitation.py          # sizes AMP to the caps; verifies 0% saturation
conda run -n mujoco-learning python collect_data.py --dry-run    # writes data/excitation_preview.png
python collect_data.py eth0                                      # on the robot (DDS iface); Ctrl-C is safe
# -> writes data/arm_chirp_<timestamp>.npz
```
The excitation is **sequential (one joint at a time)**. Every moving segment has
a raised-cosine endpoint taper, so it returns to `CENTER` with zero velocity
instead of producing the old one-step torque impulse. The collector commands
zero desired velocity, matching `pineapple_arm.py`, and defaults to its current
`kp=[20,25,25,20,20,20]`, `kd=[0.5,0.1,0.1,0.1,0.1,0.1]` (override with
`--kp`/`--kd`; the actual values are logged).

The trajectory is sized so realized peaks stay under the
per-joint caps `|tau| ≤ [10,18,18,10,10,10] Nm` (the gravity-loaded shoulder/elbow get 18 Nm; the
DM-4310 joints j4/j5/j6 are further bounded by their ±7 motor ctrlrange), `|dq| ≤ [20,10,10,20,20,20]
rad/s` — below the ±27/±7
`ctrlrange`. This matters: torque saturation destroys the acceleration content that armature ID
depends on. Paste any updated `AMP`/`F1` printed by `check_excitation.py` into `collect_data.py`.

**2b. VERIFY the collected data actually respected the caps** (the offline sizer is synchronous and
under-predicts the async-DDS simulator, so always check the real data):
```bash
conda run -n mujoco-learning python check_excitation.py --analyze data/arm_chirp_<timestamp>.npz
# reports realized per-joint |tau|/|dq|/saturation/tracking vs caps; flags TAU!/DQ!/TRACK!
```
If anything is flagged, lower that joint's `AMP` (and `F1` for a resonating
roll joint) and re-collect before fitting. The live collector also aborts after
three consecutive over-limit states (or one severe hit/stale-state event) and
saves a partial log marked `complete=false`; the fitter rejects partial logs.

**3. Fit the parameters:**
```bash
conda run -n mujoco-learning python sysid_fit.py --data data/arm_chirp_<timestamp>.npz --out results/run1
# Async DDS data is auto-cleaned first: --resample {auto,on,off} (default auto) drops the sim's
# duplicate full-state frames + resamples measured signals linearly and commands
# by zero-order hold, using separate real state-arrival and command-send timestamps.
# Under tight torque caps the gravity-loaded joints are weakly excited -> their armature/damping
# can drift "confidently wrong". Guard against it:
#   --freeze armature        hold armature at nominal (don't let it absorb model error)
#   --reg 0.05               Tikhonov pull toward nominal on unconstrained directions
#   --max-rmse 0.05          max per-joint rollout RMSE [rad] for the fit to be VALID
# -> prints identified armature/frictionloss/damping ± CI
# -> results/run1/pineapple_arm_identified.xml  (drop-in replacement, visuals intact)
```

> **Interval caveat.** With `--reg 0` the printed `±` values are ordinary
> measurement-only 95% confidence intervals. With `--reg > 0` the Tikhonov rows are part
> of the residual/Jacobian, so the values are **regularized (MAP-style) intervals shrunk
> by the prior** — narrower than the data alone justifies. The console table, and the
> `kind` field in `confidence.pkl`, label which one you are looking at.

`sysid_fit.py` validates required keys, six-column shapes, finite values, exact
joint ordering, collection completeness, excitation caps, saturation, and
tracking before optimization. It refuses legacy duplicate logs with fabricated
uniform timestamps because they cannot be repaired. After optimization it
withholds the deployment XML if the optimizer failed, an estimate is at a
bound, or a 95% confidence interval is unavailable/crosses a parameter bound.
Diagnostic overrides
(`--allow-unsafe-data`, `--allow-legacy-time`, `--allow-invalid-result`) are
explicit and should never be used to produce a deployment model.

**4. Use it only when the fit exits successfully, emits
`pineapple_arm_identified.xml`, and does not create `INVALID_FIT.txt`.** Point
your sim at that XML, or copy its identified `<joint ...>` attributes back into
`pineapple_arm.xml`.

> `results/latest` currently contains an older rejected fit whose distal
> armature/damping values hit their upper bounds. Its `INVALID_FIT.txt` is
> intentional; re-collect before producing a replacement.

## Parameters & bounds

18 parameters (6 joints × {armature, frictionloss, damping} — the 5 arm joints + the gripper
wrist-roll `gripper_case_joint`; the 2 unactuated gripper fingers are welded in `fresh_spec`);
bounds/nominals in
`sysid_common.PARAM_BOUNDS`. They are broad on purpose — tighten them once the
DM-4310 / DM-4340 datasheet rotor-inertia / friction figures are known. Restrict
the set with e.g. `--attrs armature frictionloss`.

## Notes

- **Drive mode.** `--drive torque` (default) replays the measured joint torque
  and is the most sensitive/robust identifier. `--drive pd` replays the position
  command through PD position actuators using the logged gains, zero desired
  velocity, and the original ±27/±7 actuator-force clamps; use it as a
  cross-check if the motor torque estimate looks biased. A poorly-identified
  parameter shows up as a wide confidence interval or a value pinned to a bound.
- **Link inertials are trusted** (from CAD in the XML); any error there is
  absorbed into the friction/armature estimates. A later extension could
  co-identify a mass/inertia scale (see `mujoco.sysid.body_inertia_param`).
- **Multiple shooting** (`--seg`, default 100 steps) is what makes open-loop
  torque replay well-conditioned; too long a segment lets the rollout drift.
  Diagnostic plots use the same requested segment length.
- **The offline sizer under-predicts the async-DDS simulator.** `check_excitation.py`
  sizes amplitudes with a *synchronous* clamped-PD rollout; the real
  `unitree_mujoco` bridge adds ~1-step ZOH delay and runs hotter (a roll joint at
  too high `F1` resonated to 51 rad/s and its coupling pushed held joints over
  their caps). So the sizer targets only ~80 % of the caps for headroom, keeps
  roll-joint `F1` low, and you MUST re-verify realized peaks with `--analyze` on
  collected data — never trust the sizer's numbers as final.
- **Gravity eats the torque budget on the pitch joints (hard-cap regime).** Under
  the `[10,18,18,10,10,10]` Nm caps, `upper_arm` spends ~7 Nm just holding itself up (arm+gripper), so
  <3 Nm remains for excitation → it barely moves → its armature/damping are only
  weakly identifiable and will show a wide CI (or, without a guard, drift
  "confidently wrong"). Mitigations within the caps: `--reg`/`--freeze armature`
  on those joints, longer records, and choosing an excitation `CENTER` pose that
  lowers the gravity load. **Recommended next step:** add **gravity feed-forward**
  (`tau_ff = qfrc_bias`) so the PD budget frees up for dynamics — it respects the
  hard caps (compensation, not relaxation) and would substantially improve the
  proximal joints. Not yet implemented.

## Status of the excitation (hard-cap regime)

- The current `AMP`/`F1` in `collect_data.py` are sized by the global iterative
  sizer so the **synchronous** full trajectory sits at ~80 % of the caps with 0
  saturation: `AMP=[0.403,0.220,0.346,0.695,0.600,0.600]`,
  `F1=[1.8,1.2,1.2,1.6,1.3,1.3]`. Distal amplitudes also have conservative
  async-run ceilings rather than being allowed to grow to their position limits.
  Realized peaks in the async simulator **must be confirmed** with
  `check_excitation.py --analyze <npz>` after collection (the sizer under-predicts).
- Earlier synchronous studies (before the caps were fixed as a hard limit) showed
  that with rich, non-saturating motion armature recovers to ~1–20 % (best on the
  light distal joints), versus **100–190 %** when the excitation saturated. Under
  the `[10,18,18,10,10,10]` caps the gravity-loaded pitch joints reach less of that
  richness, so expect their armature/damping to stay uncertain — guard with
  `--reg`/`--freeze` and treat their CIs as the honest measure of confidence.

## Diagnosing an identified-vs-truth gap (esp. armature)

armature only shows up through joint **acceleration**, so it is the first
parameter to go bad when the data is uninformative. If a sim-to-sim check shows
a gap, check, in order:

1. **Torque saturation.** If the excitation pins `|tau|` at the `ctrlrange`
   (±27 / ±7 here), those intervals carry no parameter information. Run
   `check_excitation.py` — it must report **0 % saturation** and peaks within the
   caps. This was the original cause of a 2–3× armature gap on the proximal joints
   (the one unsaturated distal joint identified fine at ~1×).
2. **Observability of small values.** armature ≈ 0.004 kg·m² is tiny next to the
   CAD link inertias, so even clean data leaves a wide CI — check the CI, and note
   that a sim-to-sim test where "ground truth" equals the fit's initial guess
   cannot distinguish "recovered" from "never moved." Inject **large, distinct**
   per-joint values (as `sim2sim_selftest.py` / `check_excitation.py` do).
3. **Generation vs fit mismatch.** dt and integrator must match between the data
   generator and the fit (the fit uses dt = median sample interval + the XML's
   Euler integrator). MuJoCo integrates joint damping implicitly, so a mismatch
   biases damping/armature.
4. **Async sampling / timing (the confirmed cause of the `unitree_mujoco` gap).**
   `collect_data.py` samples on its own wall-clock 200 Hz loop, unsynchronized with
   the sim's independent step + state-publish threads. So consecutive logged samples
   span a **variable number of physics steps** — measured on real sim data: **~12.6 %
   are exact-duplicate frames** (the sim didn't advance) and ~12 % span two steps —
   while the open-loop torque-replay fit assumes exactly one `dt` step per sample.
   That injects a velocity-proportional error the optimizer dumps into armature and
   damping (worst on the fast roll joints `5dof`/`gripper_case`, which rail at their
   bounds). dt/integrator/torque are otherwise consistent, so this — not a config
   mismatch — is the >20 % gap. Fix (implemented):
   - `collect_data.py` now atomically snapshots each DDS message and logs the
     **real state-arrival timestamp**, actual command-send timestamp, and
     `motor_state.tick`, instead of a fabricated `arange×dt` grid.
   - `sysid_fit.py --resample {auto,on,off}` (default `auto`) drops duplicate frames
     only when the full measured state is unchanged (or `tick` did not advance);
     it linearly resamples measurements and zero-order-holds commands onto a
     uniform grid before the fixed-timestep rollout. It rejects logs that need
     this cleanup but lack proven real timestamps. **Re-collect** first — old
     `.npz` files stored only the fabricated grid and cannot be salvaged.
   - This is **exact for the real arm** (its state genuinely arrives at the logged
     times). For the async **sim** it removes the duplicate-frame error but cannot
     recover *skipped* steps (the sim publishes no true sim-time and the wall clock is
     ~uniform), so the sim gap shrinks but may not vanish. For a rigorous, exact
     sim-to-sim check prefer a synchronous in-process rollout (`sim2sim_selftest.py`).
