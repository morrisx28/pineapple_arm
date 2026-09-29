# Pineapple Arm

Move the arm through end-effector (EE) poses using **gravity compensation + PD
control** and smooth trajectories at 200 Hz.

**Setup → Verify gravity → Tune compensation → Preview trajectory → Run**

## 1. Setup

Run commands from this repository:

```bash
conda activate mujoco-learning
python -c "import numpy, pinocchio, scipy, matplotlib, unitree_sdk2py; print('OK')"
```

The robot model is `model/robot.urdf`. Replace `eth0` below with your robot's
network interface. Keep the workspace clear and run only one controller at a time.

## 2. Verify gravity for each joint

Preview the measurement poses, collect a hardware log, then inspect the scales:

```bash
python verify_gravity.py --dry-run
python verify_gravity.py eth0 --out data/gravity_hardware.npz
python verify_gravity.py --fit-scale data/gravity_hardware.npz
```

Collection moves the arm with gravity compensation **off**, held by PD.
Review each joint's fit confidence. `N_tau` describes reported torque and `S`
describes PD stiffness; neither is the command scale to save in the next step.
Joints with little gravity loading cannot be calibrated by this method.

## 3. Tune and save compensation

Load the hardware log and start the drag tool:

```bash
python arm_drag.py --show-model --log data/gravity_hardware.npz
python arm_drag.py eth0 --log data/gravity_hardware.npz
```

At the drag prompt:

| Command | What to do |
|---|---|
| `trim` | Inspect each joint's scale and trim |
| `trim 1 0.2` | Example: set joint 1's trim to 0.2; choose values for your arm |
| `engage` | Reduce position stiffness and float the supported arm |
| `hold` | Restore position hold |
| `rec gravity_check` / `stop` | Record and report drift |
| `exit` | Shut down before starting another controller |

**Before `engage`:** the drag tool does not load `tau_cmd_scale.json`.
To start from a previously verified calibration, set each joint's
`trim = saved tau_cmd_scale / displayed g_scale`.

Tune one loaded joint at a time until hands-off drift is small across several
poses. Keep the fall catch enabled. Then:

1. Run `trim` to display the final values.
2. Copy the six **`applied = g_scale × trim`** values into the `tau_cmd_scale`
   array in [`model/tau_cmd_scale.json`](model/tau_cmd_scale.json), preserving
   its joint order and updating the measurement notes.
3. Exit the drag tool and restart controllers to load the saved calibration.

Saving is manual: `trim` and `stop` do not update the JSON file. Values of 1.0
on unmeasured joints are placeholders. Recheck calibration after firmware or
mechanical changes.

## 4. Preview a smooth trajectory

Positions are x/y/z in **metres**, in the robot model's world frame. Add
`--rpy R P Y` for orientation in **radians**; the default is identity orientation.

```bash
# Plan and check limits; writes smooth_move.png. No hardware motion.
python arm_smooth_move.py --dry-run --pose 0.205 0 0.523 \
    --dq-max 1.0 --ddq-max 2.0 --jerk-max 10.0

# Check tracking in simulation. No hardware motion.
python arm_smooth_move.py --sim --pose 0.205 0 0.523 --mass-scale 1.0 \
    --dq-max 1.0 --ddq-max 2.0 --jerk-max 10.0
```

The three limits control speed, acceleration, and jerk. Offline planning starts
at zero joints; use `--from-q Q0 Q1 Q2 Q3 Q4 Q5` to preview another starting pose.
Simulation checks the model, not your hardware calibration.

## 5. Run on the arm

Use the same target and limits you previewed:

```bash
python arm_smooth_move.py eth0 --pose 0.205 0 0.523 \
    --dq-max 1.0 --ddq-max 2.0 --jerk-max 10.0
```

The controller plans from the measured joints, follows smooth position and
velocity references with gravity compensation + PD, then attempts to return
home and release torque. **It does not hold the final pose after exiting.**

For several EE waypoints, replace `--pose` with `--via`:

```bash
python arm_smooth_move.py --dry-run \
    --via 0.205 0 0.503  0.205 0 0.523  0.205 0 0.543 \
    --dq-max 1.0 --ddq-max 2.0 --jerk-max 10.0
```

After reviewing the plan, replace `--dry-run` with `eth0` to execute it.
Motion is smooth in joint space; the EE path between waypoints may curve.
There is no obstacle avoidance.

<details>
<summary>Optional: browser control and offline tests</summary>

The browser viewer requires MuJoCo, Viser, and the sibling `pineapple_mujoco`
model/assets. Use `--scene /path/to/scene.xml` for another location.

```bash
python pineapple_arm_vis.py --sim   # Kinematic preview
python pineapple_arm_vis.py eth0    # Hardware control
```

Open `http://127.0.0.1:8080`. Hardware teleoperation requires terminal arming
(`ARM`); release the EE gizmo to plan a smooth move.

Run offline checks with:

```bash
python -m unittest test_ee_traj test_smooth_move
python verify_gravity.py --selftest
python arm_drag.py --selftest
```

For additional options, run the relevant script with `--help`.

</details>
