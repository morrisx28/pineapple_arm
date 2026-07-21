# Pineapple Arm — System Identification (MuJoCo `sysid`)

Identify the **per-joint dynamics** of the 5-DOF pineapple arm — `armature`
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
| `sysid_common.py`     | shared config: model path, joint/actuator order (= Unitree motor index 0..4), parameter bounds, `MjSpec` modifiers, data IO, identified-XML writer |
| `collect_data.py`     | **on the real arm** — runs a per-joint excitation (chirp + slow triangle) sized so no joint saturates its torque limit, and logs `q/dq/tau` + commands to `.npz`. Standalone (no MuJoCo). |
| `check_excitation.py` | no hardware — sizes the per-joint amplitudes to the torque/velocity caps (bridge-accurate clamped-PD sim), reports peak torque/velocity/saturation, and runs an observability fit so you can confirm armature is identifiable. Run it whenever `CENTER`, the gains, or the model change. |
| `sysid_fit.py`        | fit the 15 parameters from a logged `.npz`; prints a table + 95% confidence intervals, writes results and a drop-in `pineapple_arm_identified.xml` + plots |
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
conda run -n mujoco-learning python sim2sim_selftest.py          # expect: PASS, ~1% error
```

**2. Size/verify the excitation, then collect on the real arm:**
```bash
conda run -n mujoco-learning python check_excitation.py          # sizes AMP to the caps; verifies 0% saturation
conda run -n mujoco-learning python collect_data.py --dry-run    # writes data/excitation_preview.png
python collect_data.py eth0                                      # on the robot (DDS iface); Ctrl-C is safe
# -> writes data/arm_chirp_<timestamp>.npz
```
The excitation is **sequential (one joint at a time)** and sized so realized peaks stay under the
per-joint caps `|tau| ≤ [10,10,10,10,10] Nm` (j4/j5 further bounded by their ±7 motor ctrlrange),
`|dq| ≤ [20,10,10,20,20] rad/s` — below the ±27/±7
`ctrlrange`. This matters: torque saturation destroys the acceleration content that armature ID
depends on. Paste any updated `AMP`/`F1` printed by `check_excitation.py` into `collect_data.py`.

**2b. VERIFY the collected data actually respected the caps** (the offline sizer is synchronous and
under-predicts the async-DDS simulator, so always check the real data):
```bash
conda run -n mujoco-learning python check_excitation.py --analyze data/arm_chirp_<timestamp>.npz
# reports realized per-joint |tau|/|dq|/saturation/tracking vs caps; flags TAU!/DQ!/TRACK!
```
If anything is flagged over-cap, lower that joint's `AMP` (and `F1` for a resonating roll joint) and
re-collect before fitting.

**3. Fit the parameters:**
```bash
conda run -n mujoco-learning python sysid_fit.py --data data/arm_chirp_<timestamp>.npz --out results/run1
# Under tight torque caps the gravity-loaded joints are weakly excited -> their armature/damping
# can drift "confidently wrong". Guard against it:
#   --freeze armature        hold armature at nominal (don't let it absorb model error)
#   --reg 0.05               Tikhonov pull toward nominal on unconstrained directions
# -> prints identified armature/frictionloss/damping ± CI
# -> results/run1/pineapple_arm_identified.xml  (drop-in replacement, visuals intact)
```

**4. Use it:** point your sim at `results/run1/pineapple_arm_identified.xml`, or
copy the identified `<joint ...>` attributes back into `pineapple_arm.xml`.

## Parameters & bounds

15 parameters (5 joints × {armature, frictionloss, damping}); bounds/nominals in
`sysid_common.PARAM_BOUNDS`. They are broad on purpose — tighten them once the
DM-4310 / DM-4340 datasheet rotor-inertia / friction figures are known. Restrict
the set with e.g. `--attrs armature frictionloss`.

## Notes

- **Drive mode.** `--drive torque` (default) replays the measured joint torque
  and is the most sensitive/robust identifier. `--drive pd` replays the position
  command through PD position actuators (needs the same kp/kd); use it as a
  cross-check if the motor torque estimate looks biased. A poorly-identified
  parameter shows up as a wide confidence interval or a value pinned to a bound.
- **Link inertials are trusted** (from CAD in the XML); any error there is
  absorbed into the friction/armature estimates. A later extension could
  co-identify a mass/inertia scale (see `mujoco.sysid.body_inertia_param`).
- **Multiple shooting** (`--seg`, default 100 steps) is what makes open-loop
  torque replay well-conditioned; too long a segment lets the rollout drift.
- **The offline sizer under-predicts the async-DDS simulator.** `check_excitation.py`
  sizes amplitudes with a *synchronous* clamped-PD rollout; the real
  `unitree_mujoco` bridge adds ~1-step ZOH delay and runs hotter (a roll joint at
  too high `F1` resonated to 51 rad/s and its coupling pushed held joints over
  their caps). So the sizer targets only ~80 % of the caps for headroom, keeps
  roll-joint `F1` low, and you MUST re-verify realized peaks with `--analyze` on
  collected data — never trust the sizer's numbers as final.
- **Gravity eats the torque budget on the pitch joints (hard-cap regime).** Under
  the `[10,10,10,10,10]` Nm caps, `upper_arm` spends ~7 Nm just holding itself up, so
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
  saturation. Realized peaks in the async simulator **must be confirmed** with
  `check_excitation.py --analyze <npz>` after collection (the sizer under-predicts).
- Earlier synchronous studies (before the caps were fixed as a hard limit) showed
  that with rich, non-saturating motion armature recovers to ~1–20 % (best on the
  light distal joints), versus **100–190 %** when the excitation saturated. Under
  the tight `[5,10,10,5,5]` caps the gravity-loaded pitch joints can't reach that
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
4. **Timing.** `collect_data.py` writes a synthetic `arange×dt` time grid; if the
   real DDS logging jitters, sub-sample torque↔state misalignment corrupts the
   acceleration signal (armature/damping) while leaving Coulomb friction intact.
   For a rigorous sim-to-sim check, prefer a synchronous in-process rollout
   (`sim2sim_selftest.py`) over the async DDS path.
