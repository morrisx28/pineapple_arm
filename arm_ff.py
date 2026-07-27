"""Gravity + friction feedforward torque for the pineapple arm.

Written into the Unitree low-cmd slot ``motor_cmd[i].tau``, whose control law is
``tau_applied = tau_ff + kp*(q_des - q) + kd*(dq_des - dq)``, so the PD term is left
handling only tracking error (less sag, easier hand-guiding).

At nominal masses gravity matches the sysid MuJoCo model exactly (q=0:
[0, 2.2, -6.97, -1.57, 0, 0] N*m). If ``model/gravity_calib.json`` exists (from
``verify_gravity.py --apply``) link masses are scaled by the identified factors;
delete that file to revert to nominal.

Joint/motor order (i=0..5): arm_joint, arm_base_joint, upper_arm_joint,
fore_arm_joint, 5dof_joint, gripper_case_joint.
"""

import json
import os

import numpy as np
import pinocchio

import arm_ik

NUM_ARM_DOF = arm_ik.NUM_ARM_DOF

# Dedicated dynamics model so arm_ik's IK model stays kinematically pristine.
CALIB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "model", "gravity_calib.json")
GRAV_MODEL = pinocchio.buildModelFromUrdf(arm_ik.URDF_FILENAME)
GRAV_DATA = GRAV_MODEL.createData()

# Arm links are pinocchio body indices 1..NUM_ARM_DOF (0 = universe).
_ARM_JOINT_NAMES = ["arm_joint", "arm_base_joint", "upper_arm_joint",
                    "fore_arm_joint", "5dof_joint", "gripper_case_joint"]
MASS_SCALE_MIN, MASS_SCALE_MAX = 0.5, 1.5


def load_gravity_calibration(path=CALIB_PATH):
    """Per-link mass scales (6,) from the calibration overlay, or ``None``.

    Fail-safe: missing/malformed/wrong-order/out-of-range yields ``None`` so gravity
    comp falls back to nominal masses rather than crashing the hardware controller.
    """
    if not os.path.isfile(path):
        return None
    try:
        with open(path) as f:
            data = json.load(f)
        names = list(data["joint_names"])
        scales = np.asarray(data["mass_scale"], dtype=float)
    except Exception as e:
        print(f"[arm_ff] WARNING: ignoring gravity calibration {path}: {e}")
        return None
    if names != _ARM_JOINT_NAMES:
        print(f"[arm_ff] WARNING: gravity calibration joint order mismatch; ignoring {path}")
        return None
    if scales.shape != (NUM_ARM_DOF,) or not np.all(np.isfinite(scales)):
        print(f"[arm_ff] WARNING: gravity calibration needs {NUM_ARM_DOF} finite scales; "
              f"ignoring {path}")
        return None
    if np.any(scales < MASS_SCALE_MIN) or np.any(scales > MASS_SCALE_MAX):
        print(f"[arm_ff] WARNING: gravity calibration scales outside "
              f"[{MASS_SCALE_MIN}, {MASS_SCALE_MAX}]; ignoring {path}")
        return None
    return scales


def _apply_mass_scales(model, scales):
    """Scale each arm link's inertia uniformly by its mass scale (CoM fixed)."""
    for b, s in enumerate(scales):
        inertia = model.inertias[b + 1]
        model.inertias[b + 1] = pinocchio.Inertia(
            inertia.mass * float(s), inertia.lever, inertia.inertia * float(s)
        )


_GRAV_SCALES = load_gravity_calibration()
if _GRAV_SCALES is not None:
    _apply_mass_scales(GRAV_MODEL, _GRAV_SCALES)
    print("[arm_ff] gravity compensation using calibrated link mass scales: "
          + np.array2string(np.round(_GRAV_SCALES, 3)))

# Motor limits [N*m]: DM-4340 (j0-2) +-27, DM-4310 (j3-5) +-7. tau_ff is clamped
# below these so tau_ff + PD cannot saturate the drive.
TAU_LIMIT = np.array([27.0, 27.0, 27.0, 7.0, 7.0, 7.0])
FF_CLAMP_FRAC = 0.8  # leaves headroom for the PD term

# Coulomb friction [N*m]. Deliberately CONSERVATIVE: the sysid-identified
# frictionloss is unreliable for the distal joints (5dof / gripper_case came back
# pinned at their bounds from cap-violating data). Do NOT paste params_x_hat.yaml
# here blindly -- only update from a trustworthy fit, and reject bound-stuck values.
FRICTIONLOSS = np.array([0.10, 0.10, 0.10, 0.10, 0.10, 0.10])
FRICTION_SCALE = 0.6       # < 1: deliberately under-compensate to avoid limit cycles
FRICTION_VEL_EPS = 0.05    # rad/s: tanh smoothing width so it does not chatter at rest


def gravity_torque(q_arm):
    """Gravity joint torques g(q) [N*m], (6,), from the (optionally calibrated)
    GRAV_MODEL. Falls back to RNEA at zero vel/accel on older pinocchio."""
    q = arm_ik.mj_arm_to_pin(q_arm)
    model, data = GRAV_MODEL, GRAV_DATA
    if hasattr(pinocchio, "computeGeneralizedGravity"):
        g = pinocchio.computeGeneralizedGravity(model, data, q)
    else:
        zero = np.zeros(model.nv)
        g = pinocchio.rnea(model, data, q, zero, zero)
    return np.asarray(g[:NUM_ARM_DOF], dtype=float)


def friction_torque(qvel, frictionloss=None, scale=None, v_eps=None):
    """Coulomb friction compensation [N*m], (6,): ``scale*fl*tanh(dq/v_eps)``.

    tanh rather than sign() so it does not chatter around zero velocity.
    """
    fl = FRICTIONLOSS if frictionloss is None else np.asarray(frictionloss, float)
    s = FRICTION_SCALE if scale is None else float(scale)
    ve = FRICTION_VEL_EPS if v_eps is None else float(v_eps)
    dq = np.asarray(qvel, float)[:NUM_ARM_DOF]
    return s * fl[:NUM_ARM_DOF] * np.tanh(dq / ve)


def feedforward(q_arm, qvel, gravity=True, friction=True, clamp=True):
    """Total feedforward torque [N*m], shape (6,): gravity comp + friction comp."""
    tau = np.zeros(NUM_ARM_DOF)
    if gravity:
        tau += gravity_torque(q_arm)
    if friction:
        tau += friction_torque(qvel)
    if clamp:
        lim = FF_CLAMP_FRAC * TAU_LIMIT
        tau = np.clip(tau, -lim, lim)
    return tau


if __name__ == "__main__":
    for q in (np.zeros(6), np.array([0.0, 0.8, -0.8, 0.3, 0.0, 0.0])):
        print("q =", np.round(q, 3))
        print("  gravity          :", np.round(gravity_torque(q), 3))
        print("  friction (dq=0.5):", np.round(friction_torque(np.full(6, 0.5)), 4))
        print("  feedforward      :", np.round(feedforward(q, np.full(6, 0.5)), 3))
