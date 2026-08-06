"""Build EE trajectories and dynamically consistent joint references.

    EE pose x_ref(t)  --IK-->   q_ref(t)
                      --J^-1--> dq_ref(t)   (from the 6-D EE twist)
                      --diff--> ddq_ref(t)
                      --RNEA--> tau_ref(t)

``tau_ref`` includes gravity and must not be combined with another gravity term. The
pipeline is offline-only and follows ``arm_ik`` joint order.
"""

from __future__ import annotations

import numpy as np
import pinocchio

import arm_ik

NUM_ARM_DOF = arm_ik.NUM_ARM_DOF  # 6

# Joint limits [rad] from the MJCF.
JOINT_LOW = np.array([-1.5708, 0.0, -3.1416, -1.5708, -1.5708, -1.5708])
JOINT_HIGH = np.array([1.5708, 3.1416, 0.0, 1.7453, 1.5708, 1.5708])
JOINT_MARGIN = 0.05

# Leave feedback headroom when validating a reference.
DQ_REF_MAX = np.full(NUM_ARM_DOF, 2.0)                       # rad/s
TAU_REF_MAX = np.array([27.0, 27.0, 27.0, 7.0, 7.0, 7.0])    # N*m (motor limits)
TAU_REF_FRAC = 0.7   # leave headroom for the feedback correction

_GRIPPER_FINGER_JOINTS = ("gripper_left_joint", "gripper_right_joint")


def build_arm_model(apply_calibration: bool = True):
    """6-DOF pinocchio model with the unactuated gripper fingers LOCKED.

    ``arm_ik.IK_MODEL`` is nq=nv=8 (6 arm + 2 finger slides). Locking keeps the finger
    mass but removes the DOFs so the dynamics are square and match the actuated
    joints (same reason ``sysid_common.fresh_spec`` welds them in MuJoCo).
    Applies ``model/gravity_calib.json`` mass scales so this agrees with the gravity
    compensation the hardware is actually running.
    """
    full = arm_ik.IK_MODEL
    locked = [full.getJointId(n) for n in _GRIPPER_FINGER_JOINTS
              if full.existJointName(n)]
    model = (pinocchio.buildReducedModel(full, locked, pinocchio.neutral(full))
             if locked else full.copy())
    if apply_calibration:
        try:
            import arm_ff
            scales = arm_ff.load_gravity_calibration()
            if scales is not None:
                arm_ff._apply_mass_scales(model, scales)
        except Exception as e:  # Calibration is optional; nominal dynamics remain valid.
            print(f"[traj] (gravity calibration not applied: {e})")
    return model


def quintic_scaling(t, duration):
    """Smooth 0->1 scaling, zero velocity AND acceleration at both ends -> (s,ds,dds).

    Zero end-accel matters: a trapezoid/cosine leaves an accel step at t=0, which
    becomes a torque jump in ``tau_ref`` and an avoidable tracking transient.
    """
    t = np.clip(np.asarray(t, float) / max(duration, 1e-9), 0.0, 1.0)
    s = 10 * t**3 - 15 * t**4 + 6 * t**5
    ds = (30 * t**2 - 60 * t**3 + 30 * t**4) / max(duration, 1e-9)
    dds = (60 * t - 180 * t**2 + 120 * t**3) / max(duration, 1e-9) ** 2
    return s, ds, dds


def _unit(v, fallback):
    v = np.asarray(v, float)
    n = np.linalg.norm(v)
    return fallback if n < 1e-9 else v / n


def ee_hold(pose, duration, dt):
    """Hold one EE position (regulation / stiffness test)."""
    n = max(2, int(round(duration / dt)))
    t = np.arange(n) * dt
    p = np.tile(np.asarray(pose, float), (n, 1))
    return t, p, np.zeros_like(p)


def ee_line(p0, p1, duration, dt):
    """Straight EE line p0 -> p1 with quintic time scaling."""
    n = max(2, int(round(duration / dt)))
    t = np.arange(n) * dt
    s, ds, _ = quintic_scaling(t, duration)
    p0 = np.asarray(p0, float); p1 = np.asarray(p1, float)
    d = p1 - p0
    return t, p0 + s[:, None] * d, ds[:, None] * d


def ee_circle(center, radius, duration, dt, axis=(0, 0, 1), turns=1.0):
    """Circular EE path around ``center`` in the plane normal to ``axis``."""
    n = max(2, int(round(duration / dt)))
    t = np.arange(n) * dt
    s, ds, _ = quintic_scaling(t, duration)          # ramp the ANGLE, not the speed
    theta = 2 * np.pi * turns * s
    dtheta = 2 * np.pi * turns * ds
    k = _unit(axis, np.array([0.0, 0.0, 1.0]))
    # Construct an orthonormal basis for the circle plane.
    seed = np.array([1.0, 0.0, 0.0]) if abs(k[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = _unit(np.cross(k, seed), np.array([1.0, 0.0, 0.0]))
    v = np.cross(k, u)
    c = np.asarray(center, float)
    p = c + radius * (np.cos(theta)[:, None] * u + np.sin(theta)[:, None] * v)
    dp = radius * (-np.sin(theta) * dtheta)[:, None] * u \
        + radius * (np.cos(theta) * dtheta)[:, None] * v
    return t, p, dp


def ee_waypoints(points, duration, dt):
    """Piecewise-linear path through ``points``, quintic-scaled over arc length.

    One global scaling over the whole path: smooth start/stop, no pausing at the
    intermediate waypoints.
    """
    pts = np.asarray(points, float)
    if pts.ndim != 2 or pts.shape[1] != 3 or len(pts) < 2:
        raise ValueError("waypoints must be an (N>=2, 3) array of EE positions")
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    total = float(seg.sum())
    if total < 1e-9:
        return ee_hold(pts[0], duration, dt)
    knots = np.concatenate([[0.0], np.cumsum(seg)]) / total   # arc length in [0,1]

    n = max(2, int(round(duration / dt)))
    t = np.arange(n) * dt
    s, ds, _ = quintic_scaling(t, duration)
    p = np.empty((n, 3))
    dp = np.empty((n, 3))
    for i in range(3):
        p[:, i] = np.interp(s, knots, pts[:, i])
    # d/dt p = (dp/ds) * ds, with dp/ds piecewise constant per segment.
    idx = np.clip(np.searchsorted(knots, s, side="right") - 1, 0, len(seg) - 1)
    dpds = (pts[1:] - pts[:-1]) / np.maximum(np.diff(knots), 1e-12)[:, None]
    dp = dpds[idx] * ds[:, None]
    return t, p, dp


def build_ee_path(shape, dt, duration, **kw):
    """Dispatch to a named EE path. Returns (t, p_ee (N,3), v_ee (N,3))."""
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError(f"dt must be finite and positive, got {dt}")
    if not np.isfinite(duration) or duration <= 0:
        raise ValueError(f"duration must be finite and positive, got {duration}")
    if shape == "hold":
        return ee_hold(kw["p0"], duration, dt)
    if shape == "line":
        return ee_line(kw["p0"], kw["p1"], duration, dt)
    if shape == "circle":
        return ee_circle(kw["center"], kw["radius"], duration, dt,
                         axis=kw.get("axis", (0, 0, 1)), turns=kw.get("turns", 1.0))
    if shape == "waypoints":
        return ee_waypoints(kw["points"], duration, dt)
    raise ValueError(f"unknown EE path shape: {shape}")


class ReferenceError(RuntimeError):
    """Raised when the requested EE trajectory cannot be executed by this arm."""


def _frame_jacobian(model, data, q, joint_id):
    """Translational (3x6) and full (6x6) Jacobian of the EE joint, world-aligned.

    Uses ``computeJointJacobians`` (plural) + ``getJointJacobian(...,
    LOCAL_WORLD_ALIGNED)``. Do NOT hand-rotate the LOCAL Jacobian by
    ``data.oMi[joint_id].rotation``: the singular ``computeJointJacobian`` does not
    refresh ``oMi``, so that rotation can be left over from a previous configuration
    and silently corrupts the result (observed error up to 0.1 in the Jacobian).
    """
    q = np.asarray(q, float)
    pinocchio.computeJointJacobians(model, data, q)
    Jw = pinocchio.getJointJacobian(model, data, joint_id,
                                    pinocchio.ReferenceFrame.LOCAL_WORLD_ALIGNED)
    Jw = np.asarray(Jw)
    return Jw[:3], Jw


def make_joint_reference(t, p_ee, v_ee, rpy=None, model=None, seed=None,
                         damping=1e-8):
    """EE path -> (q_ref, dq_ref, ddq_ref, tau_ref), all (N,6).

    ``q_ref`` is warm-started IK (each sample seeds the next, keeping the branch
    continuous); ``ddq_ref`` central-differences ``dq_ref``; ``tau_ref`` is RNEA.

    ``dq_ref`` is solved from the FULL 6-D twist ``[v_ee; omega=0]``, not the
    translational Jacobian alone: ``arm_ik.solve_ik`` always solves a 6-D pose task,
    so the joints must also move to HOLD orientation. Translation-only ``J^+ v``
    ignores that -- measured on a 60 mm line it disagreed with ``d/dt q_ref`` by
    0.215 rad/s and let the EE spin at 0.163 rad/s. The 6-D solve reproduces the
    twist to ~1e-8 and agrees with ``d/dt q_ref`` to 0.006 rad/s, so q_ref and dq_ref
    stay mutually consistent -- which matters because ``tau_ref`` uses both.

    Raises ``ReferenceError`` if IK fails anywhere: tracking an unconverged solution
    would drive the arm toward an unreachable pose.
    """
    model = build_arm_model() if model is None else model
    data = model.createData()
    joint_id = min(arm_ik.JOINT_ID, model.njoints - 1)
    n = len(t)
    R_des = arm_ik.rpy_to_matrix(*rpy) if rpy is not None else None

    q_ref = np.zeros((n, NUM_ARM_DOF))
    dq_ref = np.zeros((n, NUM_ARM_DOF))
    q_seed = np.zeros(NUM_ARM_DOF) if seed is None else np.asarray(seed, float).copy()
    failures = []
    for k in range(n):
        q_sol, ok = arm_ik.solve_ik(q_seed, p_ee[k], R_des)
        if not ok:
            failures.append(k)
        q_seed = np.asarray(q_sol, float)
        q_ref[k] = q_seed
        # Translate at v_ee AND hold orientation (omega=0), matching the IK's task.
        _, Jf = _frame_jacobian(model, data, q_seed, joint_id)
        twist = np.concatenate([v_ee[k], np.zeros(3)])
        dq_ref[k] = np.linalg.solve(Jf.T @ Jf + damping * np.eye(NUM_ARM_DOF),
                                    Jf.T @ twist)

    if failures:
        raise ReferenceError(
            f"IK did not converge at {len(failures)}/{n} samples "
            f"(first at t={t[failures[0]]:.3f}s, target={np.round(p_ee[failures[0]],3)}); "
            "the EE path leaves the reachable workspace"
        )

    dt = float(np.median(np.diff(t))) if n > 1 else 1.0
    ddq_ref = np.gradient(dq_ref, dt, axis=0, edge_order=1)

    tau_ref = np.zeros((n, NUM_ARM_DOF))
    for k in range(n):
        tau_ref[k] = pinocchio.rnea(model, data, q_ref[k], dq_ref[k], ddq_ref[k])
    return q_ref, dq_ref, ddq_ref, tau_ref


def ee_positions_of(q_ref, model=None):
    """Forward kinematics: EE position for each row of ``q_ref``. (N,3)"""
    model = build_arm_model() if model is None else model
    data = model.createData()
    joint_id = min(arm_ik.JOINT_ID, model.njoints - 1)
    out = np.zeros((len(q_ref), 3))
    for k, q in enumerate(q_ref):
        pinocchio.forwardKinematics(model, data, np.asarray(q, float))
        out[k] = data.oMi[joint_id].translation
    return out


def ee_rotations_of(q_ref, model=None):
    """Forward kinematics: EE rotation matrix for each row of ``q_ref``. (N,3,3)"""
    model = build_arm_model() if model is None else model
    data = model.createData()
    joint_id = min(arm_ik.JOINT_ID, model.njoints - 1)
    out = np.zeros((len(q_ref), 3, 3))
    for k, q in enumerate(q_ref):
        pinocchio.forwardKinematics(model, data, np.asarray(q, float))
        out[k] = data.oMi[joint_id].rotation
    return out


def orientation_error(q, q_ref, model=None):
    """EE orientation error [rad] per sample: geodesic angle of R_actual^T R_ref."""
    Ra = ee_rotations_of(q, model)
    Rr = ee_rotations_of(q_ref, model)
    n = min(len(Ra), len(Rr))
    ang = np.zeros(n)
    for k in range(n):
        c = (np.trace(Ra[k].T @ Rr[k]) - 1.0) / 2.0
        ang[k] = np.arccos(np.clip(c, -1.0, 1.0))
    return ang


def validate_reference(q_ref, dq_ref, tau_ref, p_ee=None, model=None,
                       ee_tol=5e-3) -> list[str]:
    """Fail-closed feasibility check of a joint reference -> list of problems.

    Finiteness, joint limits (with margin), velocity/torque caps, and -- given
    ``p_ee`` -- that FK(q_ref) reproduces the requested path (catches a
    converged-but-wrong IK branch).
    """
    issues: list[str] = []
    for label, arr in (("q_ref", q_ref), ("dq_ref", dq_ref), ("tau_ref", tau_ref)):
        if not np.all(np.isfinite(arr)):
            issues.append(f"{label} contains NaN/Inf")
    if issues:
        return issues

    lo, hi = JOINT_LOW + JOINT_MARGIN, JOINT_HIGH - JOINT_MARGIN
    for j in range(NUM_ARM_DOF):
        name = arm_ik.IK_MODEL.names[j + 1]
        if np.any(q_ref[:, j] < lo[j]) or np.any(q_ref[:, j] > hi[j]):
            worst = q_ref[np.argmax(np.abs(q_ref[:, j] - np.clip(q_ref[:, j], lo[j], hi[j]))), j]
            issues.append(f"{name}: q_ref reaches {worst:+.3f} rad, outside "
                          f"[{lo[j]:+.3f}, {hi[j]:+.3f}]")
        pk_dq = float(np.max(np.abs(dq_ref[:, j])))
        if pk_dq > DQ_REF_MAX[j]:
            issues.append(f"{name}: |dq_ref|max={pk_dq:.2f} > {DQ_REF_MAX[j]:.2f} rad/s "
                          "(slow the trajectory down)")
        pk_tau = float(np.max(np.abs(tau_ref[:, j])))
        cap = TAU_REF_FRAC * TAU_REF_MAX[j]
        if pk_tau > cap:
            issues.append(f"{name}: |tau_ref|max={pk_tau:.2f} > {cap:.2f} Nm "
                          f"({TAU_REF_FRAC:.0%} of the {TAU_REF_MAX[j]:.0f} Nm motor "
                          "limit; no headroom left for feedback)")

    if p_ee is not None:
        err = np.linalg.norm(ee_positions_of(q_ref, model) - p_ee, axis=1)
        if float(err.max()) > ee_tol:
            issues.append(f"IK solution does not reproduce the EE path: max FK error "
                          f"{err.max()*1000:.1f} mm > {ee_tol*1000:.1f} mm")
    return issues
