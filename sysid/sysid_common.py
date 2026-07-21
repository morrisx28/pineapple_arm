"""Shared configuration and helpers for pineapple-arm system identification.

This module centralises everything the data-collection, self-test and fitting
scripts need so they stay consistent:

  * the arm's MuJoCo model path, joint / actuator ordering (which matches the
    Unitree ``motor_state[i]`` / ``motor_cmd[i]`` index ``i`` used by
    ``pineapple_arm.py``),
  * the per-joint dynamics parameters we identify (armature, frictionloss,
    damping) with their bounds and how to write them into an ``MjSpec``,
  * builders for the ``mujoco.sysid`` ``ParameterDict`` and for a fresh,
    fit-ready ``MjSpec`` (torque-driven or PD-driven), and
  * ``.npz`` logging helpers shared by collection and fitting.

The identification method follows DeepMind's ``mujoco.sysid`` toolbox
(box-constrained nonlinear least squares over a threaded ``mujoco.rollout``);
see ``sysid_fit.py``.
"""

from __future__ import annotations

import pathlib
from typing import Sequence

import numpy as np

import mujoco
from mujoco import sysid


# --------------------------------------------------------------------------- #
# Model + ordering
# --------------------------------------------------------------------------- #

# Absolute path to the arm model (fixed base, no floor -> ideal for arm sysid).
ARM_XML = pathlib.Path(
    "/home/csl/pineapple/pineapple_mujoco/pineapple_robots/pineapple_arm/pineapple_arm.xml"
)

# Joint order == actuator order == Unitree motor index 0..4 in pineapple_arm.py.
JOINTS = [
    "arm_joint",        # motor 0  (DM-4340, ctrl +-27 Nm)
    "arm_base_joint",   # motor 1  (DM-4340)
    "upper_arm_joint",  # motor 2  (DM-4340)
    "fore_arm_joint",   # motor 3  (DM-4310, ctrl +-7 Nm)
    "5dof_joint",       # motor 4  (DM-4310)
]
ACTUATORS = ["arm_1_dof", "arm_2_dof", "arm_3_dof", "arm_4_dof", "arm_5_dof"]
NUM_MOTORS = len(JOINTS)

# Sampling period used on the real arm (pineapple_arm.py runs the low-cmd loop
# at 200 Hz) and, by default, for the identification rollout.
DT_DEFAULT = 0.005

# PD gains used by the real controller (pineapple_arm.py Controller.kps / kds).
# Needed for the optional PD-replay drive mode and logged during collection.
DEFAULT_KP = np.array([20.0, 25.0, 25.0, 20.0, 20.0])
DEFAULT_KD = np.array([0.1, 0.5, 0.5, 0.1, 0.1])


# --------------------------------------------------------------------------- #
# Parameters to identify: (attr, nominal, lower, upper)
# --------------------------------------------------------------------------- #
# Bounds are deliberately broad; tighten them once the DM-4310/DM-4340 datasheet
# rotor-inertia / friction figures (or prior leg-tuned values) are known.
PARAM_BOUNDS = {
    # reflected rotor inertia [kg m^2]; XML nominal is 0.004
    "armature": (0.004, 1.0e-4, 0.5),
    # Coulomb / dry friction torque [N m]; XML nominal is 0.1
    "frictionloss": (0.1, 0.0, 3.0),
    # viscous friction [N m s/rad]; XML nominal is 0.02
    "damping": (0.02, 0.0, 5.0),
}
ALL_ATTRS = ("armature", "frictionloss", "damping")


# --------------------------------------------------------------------------- #
# Writing a scalar joint-dynamics parameter into an MjSpec
# --------------------------------------------------------------------------- #
# In mujoco 3.10 MjsJoint, ``armature`` and ``frictionloss`` are scalars while
# ``damping`` is exposed as a length-3 array (only element [0] is used for a
# 1-DoF hinge). ``set_joint_dyn`` hides that asymmetry.
def set_joint_dyn(spec: mujoco.MjSpec, jname: str, attr: str, value: float) -> None:
    """Set ``attr`` (armature|frictionloss|damping) of joint ``jname`` in spec."""
    joint = spec.joint(jname)
    value = float(np.asarray(value).ravel()[0])
    if attr == "damping":
        vec = np.zeros(3)
        vec[0] = value
        joint.damping = vec
    else:
        setattr(joint, attr, value)


def make_joint_modifier(attr: str, jname: str):
    """Return a ``mujoco.sysid`` modifier callback ``(spec, param) -> None``."""

    def modifier(spec: mujoco.MjSpec, param: sysid.Parameter) -> None:
        set_joint_dyn(spec, jname, attr, np.asarray(param.value).ravel()[0])

    return modifier


def build_param_dict(
    attrs: Sequence[str] = ALL_ATTRS,
    nominal: dict[str, float] | None = None,
    per_joint_nominal: dict[str, np.ndarray] | None = None,
    freeze_attrs: Sequence[str] = (),
) -> sysid.ParameterDict:
    """Build the ParameterDict of joint-dynamics parameters to identify.

    Args:
      attrs: which parameters to identify per joint (subset of ALL_ATTRS).
      nominal: optional ``{attr: value}`` override of the initial guess (applied
        to every joint).
      per_joint_nominal: optional ``{attr: array(NUM_MOTORS)}`` override giving a
        distinct initial guess per joint (takes precedence over ``nominal``).
      freeze_attrs: attrs to hold FIXED at their nominal (``frozen=True``) instead
        of optimizing -- use for parameters that the excitation can't observe
        (e.g. armature on the heavy proximal joints under tight torque caps), so
        they can't drift "confidently wrong" while absorbing model error.

    Names are ``"<joint>/<attr>"`` (e.g. ``"arm_joint/armature"``).
    """
    params = sysid.ParameterDict()
    for attr in attrs:
        nom0, lo, hi = PARAM_BOUNDS[attr]
        for j, jname in enumerate(JOINTS):
            nom = nom0
            if nominal and attr in nominal:
                nom = float(nominal[attr])
            if per_joint_nominal and attr in per_joint_nominal:
                nom = float(per_joint_nominal[attr][j])
            params.add(
                sysid.Parameter(
                    f"{jname}/{attr}",
                    nominal=nom,
                    min_value=lo,
                    max_value=hi,
                    frozen=(attr in freeze_attrs),
                    modifier=make_joint_modifier(attr, jname),
                )
            )
    return params


# --------------------------------------------------------------------------- #
# Building a fit-ready MjSpec
# --------------------------------------------------------------------------- #
def make_position_actuators(
    spec: mujoco.MjSpec, kp: np.ndarray, kd: np.ndarray
) -> None:
    """Convert the arm's torque ``<motor>`` actuators into PD position actuators.

    After this, actuator force = ``kp*(ctrl - q) - kd*qvel`` (i.e. a PD law with
    zero desired velocity), so feeding the logged position command ``q_cmd`` as
    ``ctrl`` reproduces the real controller (the ``kd*dq_cmd`` feed-forward term
    is neglected). Used by the optional ``drive="pd"`` mode.
    """
    for i, aname in enumerate(ACTUATORS):
        act = spec.actuator(aname)
        act.gaintype = mujoco.mjtGain.mjGAIN_FIXED
        act.biastype = mujoco.mjtBias.mjBIAS_AFFINE
        act.gainprm[0] = kp[i]
        act.biasprm[0] = 0.0
        act.biasprm[1] = -kp[i]
        act.biasprm[2] = -kd[i]


def fresh_spec(
    drive: str = "torque",
    dt: float | None = DT_DEFAULT,
    kp: np.ndarray | None = None,
    kd: np.ndarray | None = None,
    disable_contacts: bool = True,
    remove_visual: bool = True,
    integrator: str | None = None,
    gravity: bool = True,
) -> mujoco.MjSpec:
    """Load a fresh ``MjSpec`` of the arm configured for identification.

    Args:
      drive: ``"torque"`` (feed measured joint torque to the motors, default) or
        ``"pd"`` (convert to PD position actuators, feed the position command).
      dt: rollout timestep; ``None`` keeps the model default.
      kp, kd: PD gains for ``drive="pd"`` (default DEFAULT_KP/KD).
      disable_contacts: disable contact dynamics (arm sysid has no useful
        contacts and it speeds up the rollout).
      remove_visual: strip visual-only geometry/assets to speed compilation.
      integrator: optional override, one of "euler", "implicit", "implicitfast",
        "rk4"; ``None`` keeps the XML setting.
      gravity: if False, zero out gravity (used by the sim-to-sim self-test to
        isolate parameter recovery).
    """
    spec = mujoco.MjSpec.from_file(ARM_XML.as_posix())

    if drive == "pd":
        make_position_actuators(
            spec,
            DEFAULT_KP if kp is None else np.asarray(kp),
            DEFAULT_KD if kd is None else np.asarray(kd),
        )
    elif drive != "torque":
        raise ValueError(f"drive must be 'torque' or 'pd', got {drive!r}")

    if dt is not None:
        spec.option.timestep = float(dt)
    if not gravity:
        spec.option.gravity = np.zeros(3)
    if disable_contacts:
        spec.option.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_CONTACT)
    if integrator is not None:
        spec.option.integrator = {
            "euler": mujoco.mjtIntegrator.mjINT_EULER,
            "implicit": mujoco.mjtIntegrator.mjINT_IMPLICIT,
            "implicitfast": mujoco.mjtIntegrator.mjINT_IMPLICITFAST,
            "rk4": mujoco.mjtIntegrator.mjINT_RK4,
        }[integrator]

    if remove_visual:
        spec = sysid.remove_visuals(spec)
    return spec


# --------------------------------------------------------------------------- #
# Packing measured data into mujoco.sysid TimeSeries / ModelSequences
# --------------------------------------------------------------------------- #
def obs_names() -> list[tuple[str, sysid.SignalType]]:
    """Observation signals: joint positions then joint velocities (state-based).

    State observations are resolved by joint name inside the residual, so they
    are robust to sensor ordering (unlike sensor-name observations).
    """
    return [(j, sysid.SignalType.MjStateQPos) for j in JOINTS] + [
        (j, sysid.SignalType.MjStateQVel) for j in JOINTS
    ]


def pack_measured(
    t: np.ndarray, q: np.ndarray, dq: np.ndarray, model: mujoco.MjModel
) -> sysid.TimeSeries:
    """Measured joint pos+vel -> observation TimeSeries (columns [q(5), dq(5)])."""
    data = np.hstack([np.asarray(q), np.asarray(dq)])
    return sysid.TimeSeries.from_names(t, data, model, names=obs_names())


def pack_control(
    t: np.ndarray, ctrl: np.ndarray, model: mujoco.MjModel
) -> sysid.TimeSeries:
    """Control TimeSeries: measured torque (drive=torque) or q_cmd (drive=pd)."""
    return sysid.TimeSeries.from_control_names(t, np.asarray(ctrl), model)


def make_model_sequences(
    spec: mujoco.MjSpec,
    t: np.ndarray,
    ctrl: np.ndarray,
    q: np.ndarray,
    dq: np.ndarray,
    name: str = "pineapple_arm",
    sequence_name: str = "excite",
    seg_steps: int | None = 100,
) -> sysid.ModelSequences:
    """Assemble a ModelSequences for the residual pipeline.

    With ``seg_steps`` set, the trajectory is split into consecutive
    non-overlapping segments of that many steps, each re-initialised from the
    corresponding *measured* state (multiple shooting). This is important for
    open-loop torque replay: it stops the wrong-parameter model from drifting
    arbitrarily far from the data over a long rollout, keeping the residual a
    well-conditioned, sensitive function of the parameters. Pass ``None`` for a
    single full-length sequence.
    """
    model = spec.compile()
    t = np.asarray(t); ctrl = np.asarray(ctrl)
    q = np.asarray(q); dq = np.asarray(dq)
    n = len(t)
    seg = n if seg_steps is None else int(seg_steps)

    names, states, controls, measures = [], [], [], []
    start, c = 0, 0
    while start + 2 <= n:
        end = min(start + seg, n)
        if end - start < 2:
            break
        sl = slice(start, end)
        # Local (zero-based) times: each segment's rollout restarts sim time at
        # 0, so measured timestamps must too, else the residual's time window is
        # empty.
        t_local = t[sl] - t[start]
        controls.append(pack_control(t_local, ctrl[sl], model))
        measures.append(pack_measured(t_local, q[sl], dq[sl], model))
        states.append(sysid.create_initial_state(model, q[start], dq[start]))
        names.append(f"{sequence_name}_{c}")
        start += seg
        c += 1

    return sysid.ModelSequences(
        name, spec, names, states, controls, measures,
        allow_missing_sensors=True,
    )


# --------------------------------------------------------------------------- #
# Dataset IO (.npz)  -- shared by collect_data.py and sysid_fit.py
# --------------------------------------------------------------------------- #
# Columns of every array are in JOINTS / motor order (index 0..4).
LOG_KEYS = ("t", "q", "dq", "tau", "q_cmd", "dq_cmd", "tau_ff", "kp", "kd")


def save_log(path: str | pathlib.Path, **arrays) -> None:
    """Save a logged trajectory to ``.npz``. Stores JOINTS for provenance."""
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {k: np.asarray(v) for k, v in arrays.items()}
    payload["joint_names"] = np.array(JOINTS)
    np.savez(path, **payload)


def load_log(path: str | pathlib.Path) -> dict[str, np.ndarray]:
    """Load a logged trajectory ``.npz`` into a plain dict of arrays."""
    with np.load(path, allow_pickle=True) as npz:
        return {k: npz[k] for k in npz.files}


def save_identified_xml(opt_params: sysid.ParameterDict, out_path: str | pathlib.Path) -> None:
    """Write a drop-in identified model.

    Loads the *original* arm XML unchanged (visuals, contacts, timestep all
    preserved), applies only the identified joint-dynamics parameters, and
    writes it out -- a clean replacement for ``pineapple_arm.xml``.
    """
    spec = mujoco.MjSpec.from_file(ARM_XML.as_posix())
    # Apply parameters by parsing "<joint>/<attr>" names directly, so this works
    # for any ParameterDict -- including one reloaded from disk, whose (lambda)
    # modifiers are not restored by yaml.
    for name, param in opt_params.items():
        if param.frozen:
            continue
        jname, attr = name.rsplit("/", 1)
        set_joint_dyn(spec, jname, attr, float(np.asarray(param.value).ravel()[0]))
    # Absolute meshdir so the written file loads from any directory (the source
    # XML uses a relative "meshes/").
    spec.meshdir = (ARM_XML.parent / "meshes").as_posix()
    spec.compile()  # validate before writing
    pathlib.Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    spec.to_file(pathlib.Path(out_path).as_posix())
