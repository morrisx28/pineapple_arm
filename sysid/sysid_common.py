"""Shared model, ordering, parameter bounds, and log helpers for arm system ID."""

from __future__ import annotations

import os
import pathlib
from typing import Sequence

import numpy as np

import mujoco
from mujoco import sysid

# Fixed-base arm model (no floor). The env override keeps this portable.
_DEFAULT_ARM_XML = (
    pathlib.Path(__file__).resolve().parents[2]
    / "pineapple_mujoco/pineapple_robots/pineapple_arm/pineapple_arm.xml"
)
ARM_XML = pathlib.Path(os.environ.get("PINEAPPLE_ARM_XML", _DEFAULT_ARM_XML)).expanduser()

# Joint, actuator, and motor order are identical. Weld finger slides to match six-column
# hardware data.
JOINTS = [
    "arm_joint",          # motor 0  (DM-4340, ctrl +-27 Nm)
    "arm_base_joint",     # motor 1  (DM-4340)
    "upper_arm_joint",    # motor 2  (DM-4340)
    "fore_arm_joint",     # motor 3  (DM-4310, ctrl +-7 Nm)
    "5dof_joint",         # motor 4  (DM-4310)
    "gripper_case_joint", # motor 5  (DM-4310, wrist roll)
]
ACTUATORS = ["arm_1_dof", "arm_2_dof", "arm_3_dof", "arm_4_dof", "arm_5_dof", "arm_6_dof"]
NUM_MOTORS = len(JOINTS)

# Unactuated gripper-finger slide joints -- welded (removed) for identification.
GRIPPER_FINGER_JOINTS = ["gripper_left_joint", "gripper_right_joint"]

DT_DEFAULT = 0.005

# Controller gains used for PD replay and stored at collection.
DEFAULT_KP = np.array([20.0, 40.0, 40.0, 20.0, 20.0, 20.0])
DEFAULT_KD = np.array([0.5, 1.0, 1.0, 0.5, 0.5, 0.5])


# Broad bounds avoid assuming unavailable motor datasheet values.
PARAM_BOUNDS = {
    # reflected rotor inertia [kg m^2]; XML nominal is 0.004
    "armature": (0.004, 1.0e-4, 0.5),
    # Coulomb / dry friction torque [N m]; XML nominal is 0.1
    "frictionloss": (0.1, 0.0, 3.0),
    # viscous friction [N m s/rad]; XML nominal is 0.02
    "damping": (0.02, 0.0, 5.0),
}
ALL_ATTRS = ("armature", "frictionloss", "damping")


# MuJoCo stores hinge damping in element 0 of a length-three vector, unlike the scalar
# armature and friction fields.
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
    """ParameterDict of the joint-dynamics parameters to identify.

    Names are ``"<joint>/<attr>"``. ``per_joint_nominal`` overrides ``nominal``.
    ``freeze_attrs`` holds an attr FIXED at nominal -- use it for parameters the
    excitation cannot observe, so they cannot drift "confidently wrong" while
    absorbing model error.
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


def make_position_actuators(
    spec: mujoco.MjSpec, kp: np.ndarray, kd: np.ndarray
) -> None:
    """Convert the arm's torque ``<motor>`` actuators into PD position actuators.

    After this, actuator force = ``kp*(ctrl - q) - kd*qvel`` (i.e. a PD law with
    zero desired velocity), so feeding the logged position command ``q_cmd`` as
    ``ctrl`` reproduces the collector and ``pineapple_arm.py``. Used by the
    optional ``drive="pd"`` mode.
    """
    for i, aname in enumerate(ACTUATORS):
        act = spec.actuator(aname)
        # A motor's ctrlrange is a TORQUE limit. Once ctrl means position, that
        # limit must move to forcerange or PD replay applies unbounded torque and
        # stops matching the collector/bridge.
        torque_range = np.asarray(act.ctrlrange, dtype=float).copy()
        joint_range = np.asarray(spec.joint(JOINTS[i]).range, dtype=float).copy()
        act.gaintype = mujoco.mjtGain.mjGAIN_FIXED
        act.biastype = mujoco.mjtBias.mjBIAS_AFFINE
        act.gainprm[0] = kp[i]
        act.biasprm[0] = 0.0
        act.biasprm[1] = -kp[i]
        act.biasprm[2] = -kd[i]
        act.forcerange = torque_range
        act.forcelimited = mujoco.mjtLimited.mjLIMITED_TRUE
        act.ctrlrange = joint_range
        act.ctrllimited = mujoco.mjtLimited.mjLIMITED_TRUE


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
    """Fresh ``MjSpec`` of the arm configured for identification.

    ``drive``: "torque" feeds measured joint torque to the motors; "pd" converts to
    PD position actuators fed the position command. ``gravity=False`` is used by the
    self-test to isolate parameter recovery.
    """
    if not ARM_XML.is_file():
        raise FileNotFoundError(
            f"arm XML not found at {ARM_XML}; set PINEAPPLE_ARM_XML to its path"
        )
    spec = mujoco.MjSpec.from_file(ARM_XML.as_posix())

    # Weld the unactuated fingers: deleting the joints leaves the bodies rigidly
    # attached (mass retained) so the model is 6-DOF. Safe: no tendons/equalities
    # couple them.
    for jname in GRIPPER_FINGER_JOINTS:
        j = spec.joint(jname)
        if j is not None:
            spec.delete(j)

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


def obs_names() -> list[tuple[str, sysid.SignalType]]:
    """Observation signals: joint positions then velocities.

    State-based (resolved by joint NAME in the residual), so unlike sensor-name
    observations this is robust to sensor ordering.
    """
    return [(j, sysid.SignalType.MjStateQPos) for j in JOINTS] + [
        (j, sysid.SignalType.MjStateQVel) for j in JOINTS
    ]


def pack_measured(
    t: np.ndarray, q: np.ndarray, dq: np.ndarray, model: mujoco.MjModel
) -> sysid.TimeSeries:
    """Measured joint pos+vel -> observation TimeSeries (columns [q(6), dq(6)])."""
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
    """ModelSequences for the residual pipeline.

    ``seg_steps`` splits the trajectory into segments each re-initialised from the
    MEASURED state (multiple shooting). Essential for open-loop torque replay: it
    stops a wrong-parameter model drifting arbitrarily far from the data, keeping
    the residual well-conditioned and sensitive. ``None`` = one full-length sequence.
    """
    model = spec.compile()
    t = np.asarray(t); ctrl = np.asarray(ctrl)
    q = np.asarray(q); dq = np.asarray(dq)
    n = len(t)
    seg = n if seg_steps is None else int(seg_steps)
    if n < 2:
        raise ValueError("make_model_sequences requires at least 2 samples")
    if seg < 2:
        raise ValueError(f"seg_steps must be >=2 or None, got {seg_steps}")
    expected = (n, NUM_MOTORS)
    for label, arr in (("ctrl", ctrl), ("q", q), ("dq", dq)):
        if arr.shape != expected:
            raise ValueError(f"{label} shape must be {expected}, got {arr.shape}")

    names, states, controls, measures = [], [], [], []
    start, c = 0, 0
    while start + 2 <= n:
        end = min(start + seg, n)
        if end - start < 2:
            break
        sl = slice(start, end)
        # Zero-based times: each segment's rollout restarts sim time at 0, so the
        # measured stamps must too or the residual's time window is empty.
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


# Columns of every per-joint array are in JOINTS / motor order (index 0..5).
LOG_KEYS = (
    "t", "t_cmd", "q", "dq", "tau", "q_cmd", "dq_cmd", "tau_ff",
    "tick", "kp", "kd",
)
LOG_META_KEYS = ("schema_version", "time_source", "complete", "abort_reason")


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


def _scalar_string(value) -> str:
    arr = np.asarray(value)
    return str(arr.item()) if arr.size == 1 else ""


def has_real_timestamps(log: dict) -> bool:
    """Whether a log has credible state-arrival timestamps for resampling.

    Schema-v2 names the clock explicitly; the fallback accepts the previous
    collector revision (``tick`` + visibly non-uniform stamps). Legacy logs with
    fabricated uniform ``arange*dt`` timestamps are REJECTED -- they cannot be repaired.
    """
    if _scalar_string(log.get("time_source", "")) == "state_arrival_perf_counter":
        return True
    if "tick" not in log or "t" not in log:
        return False
    t = np.asarray(log["t"], dtype=float).ravel()
    d = np.diff(t)
    return d.size > 0 and float(np.std(d)) > 1e-7


def validate_log(log: dict, drive: str = "torque", min_samples: int = 3) -> None:
    """Validate a collector log before any interpolation/model construction."""
    required = {"t", "q", "dq", "tau", "joint_names"}
    if drive == "pd":
        required |= {"q_cmd", "kp", "kd"}
    missing = sorted(required - set(log))
    if missing:
        raise ValueError(f"log is missing required keys: {missing}")

    names = [str(x) for x in np.asarray(log["joint_names"]).ravel()]
    if names != JOINTS:
        raise ValueError(
            "joint_names/order mismatch: "
            f"expected {JOINTS}, got {names}; refusing to silently remap columns"
        )

    t = np.asarray(log["t"], dtype=float)
    if t.ndim != 1 or len(t) < min_samples:
        raise ValueError(f"t must be 1-D with at least {min_samples} samples")
    if not np.all(np.isfinite(t)):
        raise ValueError("t contains NaN/Inf")
    n = len(t)

    per_joint = ("q", "dq", "tau", "q_cmd", "dq_cmd", "tau_ff")
    for key in per_joint:
        if key not in log:
            continue
        arr = np.asarray(log[key])
        if arr.shape != (n, NUM_MOTORS):
            raise ValueError(
                f"{key} shape must be {(n, NUM_MOTORS)}, got {arr.shape}"
            )
        if not np.all(np.isfinite(arr.astype(float))):
            raise ValueError(f"{key} contains NaN/Inf")

    for key in ("kp", "kd"):
        if key not in log:
            continue
        arr = np.asarray(log[key])
        if arr.shape not in ((NUM_MOTORS,), (n, NUM_MOTORS)):
            raise ValueError(
                f"{key} shape must be {(NUM_MOTORS,)} or {(n, NUM_MOTORS)}, "
                f"got {arr.shape}"
            )
        if not np.all(np.isfinite(arr.astype(float))):
            raise ValueError(f"{key} contains NaN/Inf")

    if "t_cmd" in log:
        t_cmd = np.asarray(log["t_cmd"], dtype=float)
        if t_cmd.shape != (n,) or not np.all(np.isfinite(t_cmd)):
            raise ValueError(f"t_cmd must be finite with shape {(n,)}")
        if np.any(np.diff(t_cmd) <= 0):
            raise ValueError("t_cmd must be strictly increasing")

    if "tick" in log and np.asarray(log["tick"]).shape != (n,):
        raise ValueError(f"tick must have shape {(n,)}")

    if "complete" in log and not bool(np.asarray(log["complete"]).item()):
        reason = _scalar_string(log.get("abort_reason", "unknown safety abort"))
        raise ValueError(f"collector marked this log incomplete: {reason}")


def dedup_mask(
    q: np.ndarray,
    t: np.ndarray,
    dq: np.ndarray | None = None,
    tau: np.ndarray | None = None,
    tick: np.ndarray | None = None,
) -> np.ndarray:
    """Keep monotonic, genuinely new state frames.

    A usable ``tick`` is authoritative. Otherwise a frame counts as duplicate only
    when q, dq AND tau are all unchanged -- so legitimate quantized-position samples
    whose velocity/torque moved are not discarded.
    """
    q = np.asarray(q, dtype=float)
    t = np.asarray(t, dtype=float).ravel()
    n = len(t)
    if q.shape[0] != n:
        raise ValueError("q/t length mismatch")
    dq_arr = None if dq is None else np.asarray(dq, dtype=float)
    tau_arr = None if tau is None else np.asarray(tau, dtype=float)
    tick_arr = None if tick is None else np.asarray(tick).ravel()
    use_tick = tick_arr is not None and tick_arr.shape == (n,) and np.unique(tick_arr).size > 1

    keep = np.zeros(n, dtype=bool)
    last = -1
    for i in range(n):
        if last >= 0 and t[i] <= t[last]:
            continue
        if last >= 0 and use_tick:
            duplicate = tick_arr[i] == tick_arr[last]
        elif last >= 0:
            duplicate = np.array_equal(q[i], q[last])
            if dq_arr is not None:
                duplicate = duplicate and np.array_equal(dq_arr[i], dq_arr[last])
            if tau_arr is not None:
                duplicate = duplicate and np.array_equal(tau_arr[i], tau_arr[last])
        else:
            duplicate = False
        if not duplicate:
            keep[i] = True
            last = i
    return keep


def resample_log(log: dict, dt: float = DT_DEFAULT) -> dict:
    """Clean + resample a logged trajectory onto a uniform ``dt`` grid (new dict).

    DDS collection is not synchronized with the arm/sim's publish thread, giving
    duplicate frames (~12% on the async sim) and jittered timing. The fit assumes
    exactly one ``dt`` step per sample, so raw logs BIAS armature/damping. Drops
    duplicates, interpolates measured q/dq/tau, zero-order-holds commands/gains,
    and honours separate state-arrival vs command-send stamps.

    Exact for the real arm. On the async sim it removes the duplicate bias but
    cannot recover SKIPPED physics steps (no true sim-time is published).
    """
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError(f"resample dt must be positive, got {dt}")
    validate_log(log)
    t_raw = np.asarray(log["t"], dtype=float).ravel()
    keep = dedup_mask(
        log["q"], t_raw,
        dq=log.get("dq"), tau=log.get("tau"), tick=log.get("tick"),
    )
    t_state = t_raw[keep]
    if t_state.size < 2:
        raise ValueError("resample_log: <2 usable samples after dedup")
    t_u = np.arange(t_state[0], t_state[-1] + 1e-9, dt)
    out: dict = {}
    for key in (
        "joint_names", "schema_version", "time_source", "complete", "abort_reason"
    ):
        if key in log:
            out[key] = np.asarray(log[key]).copy()

    for key in ("q", "dq", "tau"):
        if key not in log:
            continue
        arr = np.asarray(log[key], dtype=float)[keep]
        out[key] = np.column_stack(
            [np.interp(t_u, t_state, arr[:, j]) for j in range(arr.shape[1])]
        )

    # Commands use their own SEND timestamps: pairing them with state-arrival
    # times creates a systematic PD-replay phase error.
    t_ctrl = np.asarray(log.get("t_cmd", t_raw), dtype=float).ravel()
    if np.any(np.diff(t_ctrl) <= 0):
        raise ValueError("control timestamps must be strictly increasing")
    ctrl_idx = np.searchsorted(t_ctrl, t_u, side="right") - 1
    ctrl_idx = np.clip(ctrl_idx, 0, len(t_ctrl) - 1)
    for key in ("q_cmd", "dq_cmd", "tau_ff", "kp", "kd"):
        if key in log:
            arr = np.asarray(log[key], dtype=float)
            if arr.ndim == 1:
                arr = np.tile(arr, (len(t_ctrl), 1))
            out[key] = arr[ctrl_idx]

    if "tick" in log:
        state_idx = np.searchsorted(t_state, t_u, side="right") - 1
        state_idx = np.clip(state_idx, 0, len(t_state) - 1)
        out["tick"] = np.asarray(log["tick"])[keep][state_idx]
    out["t"] = t_u - t_u[0]
    out["t_cmd"] = out["t"].copy()
    return out


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
