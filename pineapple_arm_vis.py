"""Browser visualization and commit-on-release EE teleoperation.

The scene mirrors measured joint state with kinematics only. Releasing the 6-DOF gizmo
commits one IK target and starts a validated jerk-limited trajectory; dragging never
streams intermediate commands. Mid-motion replanning preserves reference velocity and
acceleration, and playback runs on the 200 Hz publish clock rather than GUI time.

Real-arm teleoperation starts disabled, requires terminal arming, permits one owning
client, and binds to localhost unless remote access is explicitly allowed. Startup and
the live watchdog fail closed on missing or stale state and safety-limit violations.
Disconnect, re-home, or teleop disable clears pending work and aborts playback.

The URDF-to-MJCF frame offset is measured at startup even though the current models agree,
so cross-repository frame drift remains detectable.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import socket
import sys
import threading
import time
import types

import numpy as np
import mujoco

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import arm_ik  # pinocchio IK (reuses model/robot.urdf); needs the real pinocchio
import arm_ff  # torque limits for the live safety watchdog
import arm_smooth_move as SM  # plan_to_poses: the validated jerk-limited planner
import ee_traj                # ReferenceError
import joint_traj as J        # caps, clip_to_limits
import pinocchio

EE_BODY = "gripper_case_link"
NUM_ARM = arm_ik.NUM_ARM_DOF          # 6
CENTER = np.array([0.0, 0.8, -0.8, 0.3, 0.0, 0.0])
HOME_POSE = np.zeros(NUM_ARM)
MAX_JOINT_SPEED = 1.0                 # rad/s velocity CAP handed to the planner
FPS = 50.0

# Reference playback advances on the publisher clock, never GUI wall time.
PUBLISH_DT = SM.DT                    # 0.005 s, 200 Hz
# A constant sample count prevents render stalls from jumping the reference.
STEPS_PER_TICK = max(1, int(round((1.0 / FPS) / PUBLISH_DT)))   # 4

# Reuse the dynamics model for each planner torque check.
PLANT_MODEL = ee_traj.build_arm_model()

# Live safety limits (same basis as verify_gravity.py / collect_data.py).
STATE_TIMEOUT = 0.2                   # s; older measured state => stop commanding
SAFETY_TAU = 0.90 * arm_ff.TAU_LIMIT  # [27,27,27,7,7,7] * 0.9
DQ_LIMIT = np.full(NUM_ARM, 6.0)      # rad/s
TRIP_SAMPLES = 3                      # consecutive over-limit states before tripping

_SCENE_CANDIDATES = (
    os.path.join(_HERE, "model", "scene.xml"),
    # Support sibling and nested repository layouts.
    os.path.join(_HERE, "..", "pineapple_mujoco", "pineapple_robots",
                 "pineapple_arm", "scene.xml"),
    os.path.join(_HERE, "..", "pineapple", "pineapple_mujoco", "pineapple_robots",
                 "pineapple_arm", "scene.xml"),
)


def resolve_scene(explicit=None):
    """Locate scene.xml: --scene, then $PINEAPPLE_SCENE_XML, then repo-relative.

    An EXPLICIT source is authoritative: if set but missing we raise rather than
    quietly falling back -- silently rendering a different model is worse than failing.
    """
    for src, cand in (("--scene", explicit),
                      ("$PINEAPPLE_SCENE_XML", os.environ.get("PINEAPPLE_SCENE_XML"))):
        if cand:
            path = os.path.abspath(os.path.expanduser(cand))
            if not os.path.isfile(path):
                raise FileNotFoundError(f"{src} points at a missing file: {path}")
            return path
    tried = []
    for cand in _SCENE_CANDIDATES:
        path = os.path.abspath(os.path.expanduser(cand))
        if os.path.isfile(path):
            return path
        tried.append(path)
    raise FileNotFoundError(
        "scene.xml not found. Pass --scene, set PINEAPPLE_SCENE_XML, or place it at a "
        "repo-relative path. Tried:\n  " + "\n  ".join(tried))


def build_model(scene=None):
    """Compile scene.xml (arm + floor). Returns (model, ee_bid). No mocap body --
    the draggable target is a Viser transform-control, not a MuJoCo body."""
    path = resolve_scene(scene)
    model = mujoco.MjModel.from_xml_path(path)
    ee_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, EE_BODY)
    if ee_bid < 0:
        raise RuntimeError(f"EE body '{EE_BODY}' not found in {path}")
    return model, ee_bid


def extract_visual_meshes(model):
    """Visual mesh geoms -> (geom_id, verts, faces, color, opacity).

    Vertices are in the geom's local frame: MuJoCo bakes mesh_pos/mesh_quat into
    mesh_vert at compile time, so the per-frame world pose is exactly
    (geom_xpos, geom_xmat) -- verified against the renderer.
    """
    out = []
    for g in range(model.ngeom):
        if model.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH:
            continue  # skip floor plane + group-3 collision primitives
        m = int(model.geom_dataid[g])
        vb, vn = int(model.mesh_vertadr[m]), int(model.mesh_vertnum[m])
        fb, fn = int(model.mesh_faceadr[m]), int(model.mesh_facenum[m])
        verts = model.mesh_vert[vb:vb + vn].reshape(-1, 3).astype(np.float32)
        faces = model.mesh_face[fb:fb + fn].reshape(-1, 3).astype(np.int32)
        rgba = model.geom_rgba[g]
        color = tuple(int(np.clip(c, 0, 1) * 255) for c in rgba[:3])
        out.append((g, verts, faces, color, float(rgba[3])))
    return out


def frame_offset(model, ee_bid):
    """Constant MJCF-EE minus pinocchio-EE position. Orientation matches.

    Expected to be ~0 now that model/robot.urdf mounts the arm at the MJCF's 0.192735.
    Kept as a live measurement rather than a constant so a future divergence between the
    two model files shows up here instead of silently biasing every IK target.
    """
    data = mujoco.MjData(model)
    data.qpos[:NUM_ARM] = CENTER
    mujoco.mj_forward(model, data)
    mj_ee = data.xpos[ee_bid].copy()
    qp = arm_ik.mj_arm_to_pin(CENTER)
    pinocchio.forwardKinematics(arm_ik.IK_MODEL, arm_ik.IK_DATA, qp)
    pin_ee = np.array(arm_ik.IK_DATA.oMi[arm_ik.JOINT_ID].translation)
    return mj_ee - pin_ee


def quat_to_mat(q):
    R = np.zeros(9)
    mujoco.mju_quat2Mat(R, np.asarray(q, dtype=float))
    return R.reshape(3, 3)


def mat_to_wxyz(mat_flat):
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.asarray(mat_flat, dtype=float))
    return q


def set_arm(model, data, q6):
    data.qpos[:NUM_ARM] = q6
    data.qpos[NUM_ARM:] = 0.0            # gripper fingers held open (viz only)
    mujoco.mj_forward(model, data)


class TrajectorySlot:
    """A planned trajectory plus a playback index, swapped atomically.

    Module level and backend-agnostic on purpose. This is the code that decides what gets
    published to the motors, it is shared by the real controller and the kinematic sim, and
    the real one lives on a background thread -- so it is the last thing that should be
    buried in a closure where no test can reach it. ``--self-check`` exercises it directly.

    The lock covers the (trajectory, index) PAIR. Updating them separately would let one
    200 Hz tick read a freshly loaded trajectory at the previous trajectory's index.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._traj = None          # (q, dq, ddq), each (N,6), or None when idle
        self._k = 0
        self._latched = None       # (q, dq, ddq) held after a plan ends or is aborted

    def load(self, q, dq, ddq):
        """Install a trajectory and restart playback at sample 0."""
        q, dq, ddq = (np.asarray(v, float) for v in (q, dq, ddq))
        if not (len(q) == len(dq) == len(ddq)) or len(q) == 0:
            raise ValueError(f"trajectory arrays must be non-empty and equal length, "
                             f"got {len(q)}, {len(dq)}, {len(ddq)}")
        with self._lock:
            self._traj = (q, dq, ddq)
            self._k = 0

    def abort(self):
        """Drop the trajectory and latch the reference where it currently is, at rest."""
        with self._lock:
            if self._traj is not None:
                q, _, _ = self._traj
                self._latched = (q[min(self._k, len(q) - 1)].copy(),
                                 np.zeros(NUM_ARM), np.zeros(NUM_ARM))
            self._traj = None
            self._k = 0

    def active(self):
        with self._lock:
            return self._traj is not None

    def reference(self, fallback=None):
        """Current reference (q, dq, ddq) without consuming it.

        Deliberately the REFERENCE and never a measurement: ``quintic_from_state`` needs a
        start velocity inside the velocity cap, which a validated trajectory guarantees and
        a measured dq -- noisy, and unbounded during a fault -- does not.
        """
        with self._lock:
            if self._traj is not None:
                q, dq, ddq = self._traj
                k = min(self._k, len(q) - 1)
                return q[k].copy(), dq[k].copy(), ddq[k].copy()
            if self._latched is not None:
                return tuple(v.copy() for v in self._latched)
        if fallback is None:
            return np.zeros(NUM_ARM), np.zeros(NUM_ARM), np.zeros(NUM_ARM)
        return np.asarray(fallback, float).copy(), np.zeros(NUM_ARM), np.zeros(NUM_ARM)

    def advance(self, n=1, fallback=None):
        """Consume ``n`` samples and return the (q, dq) to publish now.

        On reaching the end the endpoint is latched at rest and playback stops. It keeps
        being returned rather than going silent: dropping the command would leave the drive
        holding its last value with nothing watching it.
        """
        with self._lock:
            if self._traj is not None:
                q, dq, _ = self._traj
                k = min(self._k, len(q) - 1)
                out = (q[k].copy(), dq[k].copy())
                self._k = k + int(n)
                if self._k >= len(q):
                    self._latched = (q[-1].copy(), np.zeros(NUM_ARM), np.zeros(NUM_ARM))
                    self._traj = None
                    self._k = 0
                return out
            if self._latched is not None:
                return self._latched[0].copy(), self._latched[1].copy()
        if fallback is None:
            return np.zeros(NUM_ARM), np.zeros(NUM_ARM)
        return np.asarray(fallback, float).copy(), np.zeros(NUM_ARM)


class RealBackend:
    """Wraps pineapple_arm.Controller: mirrors measured q, streams joint targets."""

    def __init__(self, iface, timeout=5.0):
        from unitree_sdk2py.core.channel import ChannelFactoryInitialize
        import pineapple_arm

        class _StampedController(pineapple_arm.Controller):
            """Controller that timestamps state AND plays back a planned trajectory.

            Two additions over the base class, both by subclassing so pineapple_arm.py
            stays untouched (its REPL still wants the plain linear blend):

            1. ``t_state``: the base class keeps no timestamp, so a DDS dropout is
               undetectable.
            2. Trajectory playback. The base ``moveToPose`` publishes ``dq = 0.0``
               unconditionally, which -- with the S-inflated effective kd of 18.8 on
               arm_base and 12.6 on upper_arm -- applies ~19 Nm AGAINST any commanded
               motion for its whole duration. Here each 200 Hz tick emits the planned
               ``(q_ref[k], dq_ref[k])`` instead, so the velocity term helps rather than
               fights. See arm_smooth_move.py's defect 3.
            """

            def __init__(self):
                super().__init__()
                self.t_state = 0.0
                self.slot = TrajectorySlot()

            def LowStateMessageHandler(self, msg):
                super().LowStateMessageHandler(msg)
                self.t_state = time.perf_counter()

            def _hold(self):
                """Fallback reference when nothing is loaded: the primed target pose."""
                return np.asarray(self.target_dof_pos, float)[:NUM_ARM]

            def moveToPose(self):
                """Publish one tick. Overrides the base linear blend entirely.

                Reached once per 200 Hz iteration of the base class's LowCmdWrite loop
                while ``mode == 'pose'``, so ONE call is exactly one publish period --
                which is why the reference index cannot be influenced by GUI wall time.
                ``tau`` is zeroed here exactly as the base does, because
                ``apply_feedforward`` overwrites it immediately after.
                """
                q_ref, dq_ref = self.slot.advance(1, fallback=self._hold())
                # np.clip does not remove NaN, so finiteness is a separate check -- and it
                # must run BEFORE the write, not after (arm_smooth_move._write).
                if not (np.all(np.isfinite(q_ref)) and np.all(np.isfinite(dq_ref))):
                    print(f"[vis] refusing to publish non-finite reference "
                          f"q={q_ref} dq={dq_ref}; holding")
                    return
                q_ref = J.clip_to_limits(q_ref)   # exact limits: home IS a limit for j1/j2
                self.target_dof_pos = np.asarray(q_ref, dtype=np.float32)
                self.target_dof_vel = np.asarray(dq_ref, dtype=np.float32)
                for i in range(NUM_ARM):
                    m = self.low_cmd.motor_cmd[i]
                    m.q = float(q_ref[i])
                    m.dq = float(dq_ref[i])
                    m.kp = float(self.kps[i])
                    m.kd = float(self.kds[i])
                    m.tau = 0.0

        ChannelFactoryInitialize(1, iface)
        self.c = _StampedController()
        self.c.Init()
        t0 = time.perf_counter()
        while self.c.low_state is None and time.perf_counter() - t0 < timeout:
            time.sleep(0.02)
        # Never enter pose mode before receiving a real state; the zero buffer is not a pose.
        if self.c.low_state is None:
            self.c.ShutDown()
            raise RuntimeError(
                f"no rt/lowstate within {timeout:.0f}s on iface '{iface}'. Refusing to "
                "start: without a measurement the arm would be commanded to zero at "
                "full gain. Check the arm/sim is up and the DDS interface is right.")
        # Prime the hold target from the first real measurement.
        self.c.setTargetPose(self.measured())
        self.c.mode = "pose"

    def measured(self):
        return np.asarray(self.c.qpos, dtype=float).copy()

    def state_age(self):
        return time.perf_counter() - self.c.t_state

    def torque(self):
        return np.asarray(self.c.qtau, dtype=float).copy()

    def velocity(self):
        return np.asarray(self.c.qvel, dtype=float).copy()

    def command_traj(self, q, dq, ddq):
        self.c.slot.load(q, dq, ddq)

    def reference(self):
        return self.c.slot.reference(fallback=self.c._hold())

    def abort_traj(self):
        self.c.slot.abort()

    def active_traj(self):
        return self.c.slot.active()

    def advance(self):
        """No-op: the controller's own 200 Hz thread advances the reference index."""

    def shutdown(self):
        self.c.ShutDown()


class SimBackend:
    """Kinematic backend advancing a fixed sample count per GUI tick."""

    def __init__(self):
        self.q = CENTER.copy()
        self.slot = TrajectorySlot()

    def measured(self):
        return self.q.copy()

    def command_traj(self, q, dq, ddq):
        self.slot.load(q, dq, ddq)

    def abort_traj(self):
        self.slot.abort()

    def active_traj(self):
        return self.slot.active()

    def reference(self):
        return self.slot.reference(fallback=self.q)

    def advance(self):
        """Consume STEPS_PER_TICK reference samples, mirroring one GUI tick of 200 Hz."""
        if not self.slot.active():
            return
        self.slot.advance(STEPS_PER_TICK, fallback=self.q)
        # Read the post-advance index; ``advance`` returns the sample it consumed.
        self.q = J.clip_to_limits(self.slot.reference(fallback=self.q)[0]).copy()

    # A kinematic simulation is always fresh and unloaded.
    def state_age(self):
        return 0.0

    def torque(self):
        return np.zeros(NUM_ARM)

    def velocity(self):
        return np.zeros(NUM_ARM)

    def shutdown(self):
        pass


class SafetyMonitor:
    """Live fail-closed checks on the measured state.

    Trips on stale state IMMEDIATELY (a dropout means we would be commanding from
    stale measurements), and on over-torque/velocity only after ``trip_samples``
    consecutive samples so one noisy frame does not disarm teleop.
    """

    def __init__(self, trip_samples=TRIP_SAMPLES, state_timeout=STATE_TIMEOUT):
        self.trip_samples = max(1, int(trip_samples))
        self.state_timeout = float(state_timeout)
        self._trip = np.zeros(NUM_ARM, dtype=int)
        self.reason = ""

    def check(self, backend):
        """Return "" when safe, else a reason string (latched until reset())."""
        if self.reason:
            return self.reason
        age = backend.state_age()
        if not np.isfinite(age) or age > self.state_timeout:
            self.reason = f"stale state: {age:.2f}s > {self.state_timeout:.2f}s"
            return self.reason
        tau, dq = backend.torque(), backend.velocity()
        if not (np.all(np.isfinite(tau)) and np.all(np.isfinite(dq))):
            self.reason = "non-finite measured state"
            return self.reason
        over = (np.abs(tau) > SAFETY_TAU) | (np.abs(dq) > DQ_LIMIT)
        self._trip = np.where(over, self._trip + 1, 0)
        bad = np.flatnonzero(self._trip >= self.trip_samples)
        if bad.size:
            j = int(bad[0])
            self.reason = (f"joint {j} over limit: |tau|={abs(tau[j]):.1f} "
                           f"|dq|={abs(dq[j]):.1f} ({self.trip_samples} consec)")
        return self.reason

    def reset(self):
        self._trip[:] = 0
        self.reason = ""


class Marker:
    """Track ordered drag phases and expose one valid commit per release.

    Async callbacks preserve phase order while IK remains on the control thread. A
    synthesized end event from a disconnected client is discarded rather than treated as
    a release, and only the current teleop owner may commit.
    """

    def __init__(self, handle):
        self.h = handle
        self._dragging = False
        self._commit = False
        self._commit_client = None    # client_id that produced the pending commit
        self._drag_client = None      # client_id currently holding the gizmo

        if hasattr(handle, "on_update"):
            @handle.on_update
            async def _(ev):
                cid = getattr(ev, "client_id", None)
                phase = getattr(ev, "phase", None)
                if phase == "start":
                    self._dragging = True
                    self._drag_client = cid
                elif phase == "end":
                    self._dragging = False
                    self._commit = True
                    self._commit_client = cid
                    self._drag_client = None
                elif phase is None:
                    # Older Viser versions expose no phase information.
                    self._commit = True
                    self._commit_client = cid

    @property
    def dragging(self) -> bool:
        return self._dragging

    def request_commit(self, client_id=None):
        """Commit as if the user had just released; used by the headless self-check."""
        self._commit = True
        self._commit_client = client_id

    def take_commit(self, owner=None, connected=None) -> bool:
        """True exactly once per release; clears the pending-commit flag.

        DISCARDS (not defers) a commit from a client that is no longer connected --
        viser's synthesized end-on-disconnect -- or that is not the teleop owner.
        ``owner``/``connected`` of None skip the respective check (sim/self-check).
        """
        pending, self._commit = self._commit, False
        cid, self._commit_client = self._commit_client, None
        if not pending:
            return False
        if connected is not None and cid is not None and cid not in connected:
            print(f"[vis] ignoring commit from disconnected client {cid} "
                  "(viser synthesizes an 'end' on disconnect)")
            return False
        if owner is not None and cid is not None and cid != owner:
            print(f"[vis] ignoring commit from non-owner client {cid} (owner={owner})")
            return False
        return True

    def get(self):
        return np.asarray(self.h.position, dtype=float), np.asarray(self.h.wxyz, dtype=float)

    def set(self, pos, wxyz):
        self.h.position = tuple(float(v) for v in pos)
        self.h.wxyz = tuple(float(v) for v in wxyz)


def plan_joint_move(backend, q_target, dq_max=MAX_JOINT_SPEED):
    """Plan a jerk-limited JOINT-space move from the current reference. -> (t,q,dq,ddq)

    For targets that are already joint vectors -- "return arm to home" -- so no IK runs.
    Same profile and same ``validate`` gate as the EE path; raises the same way.
    """
    q_ref, dq_ref, ddq_ref = backend.reference()
    q_target = J.clip_to_limits(np.asarray(q_target, float)[:NUM_ARM])
    caps = {"dq_max": np.full(NUM_ARM, float(dq_max))}
    t, q, dq, ddq = J.quintic_from_state(q_ref, dq_ref, ddq_ref, q_target,
                                         PUBLISH_DT, **caps)
    issues = J.validate(t, q, dq, ddq, model=PLANT_MODEL, warn=False, **caps)
    if issues:
        raise ee_traj.ReferenceError(
            "planned joint move is not executable:\n  - " + "\n  - ".join(issues))
    return t, q, dq, ddq


def plan_from_reference(backend, target_pos, target_rot, dq_max=MAX_JOINT_SPEED):
    """Plan a jerk-limited move to an EE pose from the CURRENT reference state.

    -> (t, q, dq, ddq). Raises ``ee_traj.ReferenceError`` if IK fails or the profile is
    infeasible, and ``ValueError`` on a bad start state -- both fail closed, so the caller
    keeps whatever it was already executing.

    Seeded from ``backend.reference()`` and not from the measurement, so re-planning
    mid-move is C^2: the new profile continues the old one's velocity and acceleration
    instead of stepping. That is the whole reason ``joint_traj.quintic_from_state`` exists.
    """
    q_ref, dq_ref, ddq_ref = backend.reference()
    caps = {"dq_max": np.full(NUM_ARM, float(dq_max))}
    return SM.plan_to_poses(
        q_ref, [np.asarray(target_pos, float)], rot=target_rot,
        dt=PUBLISH_DT, caps=caps, model=PLANT_MODEL,
        dq_start=dq_ref, ddq_start=ddq_ref,
        warn=False,   # a re-planning loop must not print on every release
    )


def teleop_step(model, data, ee_bid, marker, offset, backend,
                teleop_on, dq_max, goal=None, owner=None, connected=None):
    """Process one GUI control step and return reference, IK status, EE pose, goal, note.

    ``ok`` is tri-state: ``None`` means no solve occurred. Only an owned, connected release
    creates a goal. Each release plans once, while the publisher clock executes the
    trajectory independently of GUI timing. ``dq_max`` is the planner velocity cap.
    """
    meas = backend.measured()
    set_arm(model, data, meas)                       # render measured arm; updates EE xpos/xquat
    ee_pos = data.xpos[ee_bid].copy()
    ee_quat = data.xquat[ee_bid].copy()
    q_ref = backend.reference()[0]

    if not teleop_on:
        marker.set(ee_pos, ee_quat)                  # gizmo sticks to EE (no jump on enable)
        marker.take_commit()                         # drop a release that happened while off
        backend.abort_traj()                         # an in-flight plan must not outlive teleop
        return q_ref, None, ee_pos, ee_quat, None, ""   # stale goal must not survive re-enable

    ok = None                                        # None = no solve happened this tick
    note = ""
    if marker.take_commit(owner=owner, connected=connected):
        m_pos, m_wxyz = marker.get()
        target_pos, target_rot = m_pos - offset, quat_to_mat(m_wxyz)
        try:
            t, q, dq, ddq = plan_from_reference(backend, target_pos, target_rot, dq_max)
            goal = q[-1].copy()
            backend.command_traj(q, dq, ddq)
            ok = True
            # Surface the duration: a mid-move reversal legitimately takes ~2x longer than
            # a fresh move (braking is bounded too), and without this it reads as lag.
            note = f"planned {t[-1]:.2f} s move"
        except (ee_traj.ReferenceError, ValueError) as e:
            # Fail closed on BOTH: unreachable/infeasible keeps the previous trajectory
            # running rather than commanding something unvalidated. Surfaced to the GUI.
            ok = False
            note = str(e).splitlines()[0]
            print(f"[vis] plan refused: {note}")

    backend.advance()
    return backend.reference()[0], ok, ee_pos, ee_quat, goal, note


def build_scene(server, meshes):
    """Add floor grid, lights and one Viser mesh per visual geom. Returns a dict
    {geom_id: mesh_handle} to be re-posed each frame."""
    server.scene.set_up_direction("+z")
    server.scene.add_grid("/floor", width=2.0, height=2.0, plane="xy",
                          cell_size=0.1, section_size=0.5)
    server.scene.add_light_ambient("/amb", intensity=0.6)
    server.scene.add_light_directional("/sun", intensity=1.2, position=(1.0, 1.0, 2.0))
    handles = {}
    for g, verts, faces, color, opacity in meshes:
        handles[g] = server.scene.add_mesh_simple(
            f"/arm/geom_{g}", verts, faces, color=color,
            opacity=(None if opacity >= 0.999 else opacity),
            flat_shading=False, side="double",
        )
    return handles


def render(server, data, handles):
    """Pose every arm mesh handle from the current forward-kinematics geom frames."""
    with server.atomic():
        for g, h in handles.items():
            h.position = tuple(float(v) for v in data.geom_xpos[g])
            h.wxyz = tuple(float(v) for v in mat_to_wxyz(data.geom_xmat[g]))


def home_refusal(armed, teleop_on, owner, client_id):
    """Why a "return arm to home" click must be refused, or "" to allow it.

    Home commands real motion, so it is held to the SAME rules as gizmo teleop rather
    than bypassing them -- otherwise an un-armed or view-only browser could move the
    arm. Module-level (not a closure) so the self-check can exercise it.
    """
    if not armed:
        return "home refused: session is not armed"
    if not teleop_on:
        return "home refused: enable teleop first"
    if owner is not None and client_id != owner:
        return f"home refused: client {client_id} does not own teleop"
    return ""


def lan_url(port):
    """Best-effort LAN URL for remote browsers (falls back to localhost)."""
    ip = "127.0.0.1"
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))       # no packet sent; just picks the primary iface
        ip = s.getsockname()[0]
        s.close()
    except OSError:
        pass
    return f"http://{ip}:{port}"


def run_server(backend, host, port, share, is_real, scene=None, armed=True):
    import viser
    model, ee_bid = build_model(scene)
    data = mujoco.MjData(model)
    meshes = extract_visual_meshes(model)
    offset = frame_offset(model, ee_bid)
    print(f"[vis] rendering {len(meshes)} visual mesh geoms; "
          f"MJCF<-IK frame offset = {np.round(offset, 4)} m")

    server = viser.ViserServer(host=host, port=port)
    handles = build_scene(server, meshes)

    # EE-target gizmo, initialized exactly on the current EE (no jump on enable)
    set_arm(model, data, backend.measured())
    ee0_pos = data.xpos[ee_bid].copy()
    ee0_quat = data.xquat[ee_bid].copy()
    tc = server.scene.add_transform_controls(
        "/ee_target", scale=0.15, line_width=2.5,
        position=tuple(ee0_pos), wxyz=tuple(ee0_quat),
    )
    marker = Marker(tc)
    render(server, data, handles)

    # Only ONE client may command the arm, and only while connected: the marker is a
    # single server-side object, so otherwise two browsers interleave drag phases and
    # one client's release commits the other's marker pose.
    ctl = {"owner": None, "connected": set(), "alarm": "", "disarm": False}

    @server.on_client_connect
    def _(client):
        ctl["connected"].add(client.client_id)

    @server.on_client_disconnect
    def _(client):
        ctl["connected"].discard(client.client_id)
        if client.client_id == ctl["owner"]:
            # Owner vanished, possibly mid-drag: viser synthesizes an "end", which
            # take_commit() drops because the client is no longer connected.
            ctl["owner"] = None
            ctl["disarm"] = True
            print(f"[vis] teleop owner {client.client_id} disconnected -> disarmed")

    # GUI
    with server.gui.add_folder("Teleop"):
        cb_teleop = server.gui.add_checkbox(
            "Enable teleop", False, disabled=not armed,
            hint=("armed at the terminal" if armed else
                  "NOT ARMED: restart and confirm at the terminal to enable"))
        btn_home = server.gui.add_button("Re-home marker to EE")
        btn_arm_home = server.gui.add_button(
            "Return arm to home [0,0,0,0,0,0]", disabled=not armed,
            hint=("moves the ARM (not the marker); needs teleop enabled and owned"
                  if armed else
                  "NOT ARMED: restart and confirm at the terminal to enable"))
        # The planner's VELOCITY CAP, not a slew rate: acceleration and jerk are bounded
        # by joint_traj's caps regardless, so raising this no longer risks a torque spike.
        sl_speed = server.gui.add_slider("Max joint speed [rad/s]", 0.1,
                                         float(J.DQ_MAX.min()), 0.1, MAX_JOINT_SPEED)

    @cb_teleop.on_update
    def _(ev):
        if cb_teleop.value:
            ctl["owner"] = getattr(ev, "client_id", None)
            print(f"[vis] teleop ENABLED by client {ctl['owner']}")
        else:
            ctl["owner"] = None
    cb_grav = cb_fric = None
    if is_real:
        with server.gui.add_folder("Compensation (real)"):
            cb_grav = server.gui.add_checkbox("Gravity comp", backend.c.use_gravity_comp)
            cb_fric = server.gui.add_checkbox("Friction comp", backend.c.use_friction_comp)
    with server.gui.add_folder("Status"):
        txt = server.gui.add_text("state", "", multiline=True, disabled=True)

    rehome = {"flag": False}
    home = {"flag": False, "active": False, "note": ""}

    @btn_home.on_click
    def _(_ev):
        rehome["flag"] = True

    @btn_arm_home.on_click
    def _(ev):
        # Only sets a flag; the motion runs on the control-loop thread. Gating lives
        # in home_refusal() so it is unit-testable.
        home["note"] = home_refusal(armed, bool(cb_teleop.value), ctl["owner"],
                                    getattr(ev, "client_id", None))
        if home["note"]:
            print(f"[vis] {home['note']}")
        else:
            home["flag"] = True

    # Advertising the LAN address while bound to localhost is a dead link.
    url = (f"http://127.0.0.1:{port}"
           if host in ("127.0.0.1", "localhost", "::1") else lan_url(port))
    print(f"\n[vis] Viser web view:  {url}   (open in any browser)")
    if share:
        try:
            print(f"[vis] public share URL: {server.request_share_url()}")
        except Exception as e:  # noqa: BLE001
            print(f"[vis] share URL unavailable: {e}")
    print("[vis] GUI: tick 'Enable teleop' to command the arm (starts OFF). Ctrl-C to quit.\n")

    q_ref = backend.reference()[0]
    goal = None                      # joint target of the ACTIVE plan; set on release
    ik_ok = None                     # LATCHED last real IK result (None = none yet)
    plan_note = ""                   # last planner message (duration, or a refusal)
    monitor = SafetyMonitor()
    period = 1.0 / FPS
    try:
        while True:
            now = time.perf_counter()

            # Fail closed: stop commanding rather than drive from stale measurements.
            alarm = monitor.check(backend)
            if alarm and not ctl["disarm"]:
                ctl["disarm"] = True
                print(f"[vis] SAFETY: {alarm} -> teleop disarmed, commands stopped")

            if ctl["disarm"]:
                cb_teleop.value = False
                ctl["owner"] = None
                goal = None
                marker.take_commit()
                backend.abort_traj()     # an alarm must stop an in-flight plan, not
                home["active"] = False   # merely stop issuing new ones
                ctl["disarm"] = False

            if home["flag"]:
                home["flag"] = False
                # Plan the home move exactly like a dragged one -- jerk-limited, validated,
                # continuous from whatever the arm is currently doing. This is the upgrade
                # over the old constant-velocity slew toward zeros.
                try:
                    _, hq, hdq, hddq = plan_joint_move(
                        backend, HOME_POSE, float(sl_speed.value))
                    backend.command_traj(hq, hdq, hddq)
                    goal = hq[-1].copy()
                    home["active"] = True
                    ik_ok = None
                    marker.take_commit()   # a queued release must not fight the move
                    print(f"[vis] returning arm to home {HOME_POSE.tolist()}")
                except (ee_traj.ReferenceError, ValueError) as e:
                    home["note"] = f"home refused: {str(e).splitlines()[0]}"
                    print(f"[vis] {home['note']}")

            if rehome["flag"]:
                set_arm(model, data, backend.measured())
                marker.set(data.xpos[ee_bid].copy(), data.xquat[ee_bid].copy())
                backend.abort_traj()  # snapping the marker must not leave a plan running
                goal = None           # ... and must not re-trigger motion
                ik_ok = None          # clear a latched failure
                plan_note = ""
                marker.take_commit()
                monitor.reset()
                rehome["flag"] = False

            if is_real:
                backend.c.use_gravity_comp = bool(cb_grav.value)
                backend.c.use_friction_comp = bool(cb_fric.value)

            teleop_on = bool(cb_teleop.value) and not alarm
            q_ref, ok, ee_pos, _, goal, note = teleop_step(
                model, data, ee_bid, marker, offset, backend,
                teleop_on, float(sl_speed.value), goal,
                owner=ctl["owner"], connected=ctl["connected"])
            if ok is not None:       # only a real solve updates the latch
                ik_ok = ok
            if note:
                plan_note = note

            if home["active"]:
                if goal is None or not teleop_on:
                    home["active"] = False        # cancelled (teleop off / disarmed)
                else:
                    # Park the gizmo on the EE so it follows the arm home instead of
                    # being left behind where the last drag ended.
                    marker.set(ee_pos, data.xquat[ee_bid].copy())
                    marker.take_commit()
                    if (np.max(np.abs(goal - q_ref)) < 1e-3
                            and np.max(np.abs(backend.measured() - goal)) < 5e-3):
                        home["active"] = False
                        print("[vis] arm reached home pose")
            render(server, data, handles)

            m_pos, _ = marker.get()
            if alarm:
                motion = f"SAFETY STOP -- {alarm}"
            elif not armed:
                motion = "NOT ARMED (confirm at the terminal to enable)"
            elif not cb_teleop.value:
                motion = "teleop off"
            elif home["active"]:
                motion = "HOMING -> [0,0,0,0,0,0]"
            elif marker.dragging:
                motion = "DRAGGING -- arm held, releases on drop"
            elif goal is None:
                motion = "idle (drag the gizmo, then let go)"
            elif np.max(np.abs(goal - q_ref)) > 1e-3:
                motion = "executing planned trajectory"
            else:
                motion = "at target"
            ik_txt = ("-" if ik_ok is None
                      else "ok" if ik_ok else "NO SOLUTION (latched)")
            txt.value = (
                f"teleop: {'ON' if cb_teleop.value else 'off'}   [{motion}]\n"
                f"owner: {ctl['owner']}  clients: {sorted(ctl['connected'])}\n"
                f"state age: {backend.state_age()*1000:.0f} ms\n"
                f"IK: {ik_txt}\n"
                f"EE  (render): {np.round(ee_pos, 3).tolist()}\n"
                f"marker(render): {np.round(m_pos, 3).tolist()}\n"
                f"IK target(base): {np.round(m_pos - offset, 3).tolist()}\n"
                f"q meas: {np.round(backend.measured(), 3).tolist()}\n"
                f"q ref : {np.round(q_ref, 3).tolist()}\n"
                f"dq ref: {np.round(backend.reference()[1], 3).tolist()}\n"
                f"q goal: {'-' if goal is None else np.round(goal, 3).tolist()}"
                + (f"\nplan: {plan_note}" if plan_note else "")
                + (f"\n{home['note']}" if home["note"] else "")
            )
            time.sleep(max(0.0, period - (time.perf_counter() - now)))
    except KeyboardInterrupt:
        print("\n[vis] shutting down.")
    finally:
        backend.shutdown()
        server.stop()


def self_check(port, scene=None):
    """Headless: start a Viser server, build the scene from MuJoCo meshes, compute
    the frame offset, simulate a drag on the gizmo, and slew a sim arm to it via IK.
    No browser needed."""
    import viser
    model, ee_bid = build_model(scene)
    data = mujoco.MjData(model)
    meshes = extract_visual_meshes(model)
    offset = frame_offset(model, ee_bid)
    print(f"scene: {resolve_scene(scene)}")
    print(f"visual mesh geoms: {len(meshes)}  ee_bid={ee_bid}")
    print(f"frame offset (MJCF-IK) = {np.round(offset, 5)} m  (expect ~[0,0,0])")
    assert len(meshes) > 10, "too few visual meshes extracted"
    # The URDF and the MJCF must agree. This used to expect 0.0727 on z, which was the
    # symptom of model/robot.urdf mounting the arm at 0.12 against the MJCF's 0.192735.
    # A nonzero offset here now means the two model files have drifted apart again.
    assert np.allclose(offset, 0, atol=1e-6), \
        (f"URDF/MJCF frame mismatch: offset {offset} m. model/robot.urdf's arm_joint "
         f"origin must match the MJCF arm_base pos (z=0.192735).")

    server = viser.ViserServer(host="127.0.0.1", port=port)
    handles = build_scene(server, meshes)
    tc = server.scene.add_transform_controls("/ee_target", scale=0.15)
    marker = Marker(tc)

    backend = SimBackend()
    set_arm(model, data, backend.measured())
    render(server, data, handles)
    # every mesh handle got a finite pose
    for g, h in handles.items():
        assert np.all(np.isfinite(np.asarray(h.position))), f"geom {g} bad pos"
        assert np.all(np.isfinite(np.asarray(h.wxyz))), f"geom {g} bad quat"

    ee0 = data.xpos[ee_bid].copy()
    marker.set(ee0 + np.array([0.06, 0.0, 0.04]), data.xquat[ee_bid].copy())  # simulate drag

    goal = None

    # Drive the REAL viser drag callbacks (not the private flags) so this covers the
    # actual start/update/end path the browser uses.
    def fire(phase, client_id=1):
        ev = types.SimpleNamespace(phase=phase, target=tc, client=None,
                                   client_id=client_id)

        async def _run():
            for cb in tc._impl_aux.update_cb:
                res = cb(ev)
                if asyncio.iscoroutine(res):
                    await res
        asyncio.run(_run())

    def step(goal, teleop_on=True, dq_max=1.0, owner=None, connected=None):
        """teleop_step with the self-check's defaults. -> (q_ref, ok, goal, note)"""
        q_ref, ok, _, _, goal, note = teleop_step(
            model, data, ee_bid, marker, offset, backend, teleop_on, dq_max,
            goal, owner=owner, connected=connected)
        return q_ref, ok, goal, note

    # Dragging must not command the arm before release.
    fire("start")
    fire("update")
    assert marker.dragging, "drag-start event did not register"
    for _ in range(20):
        _, _, goal, _ = step(goal)
    assert np.allclose(backend.measured(), CENTER), \
        f"arm moved while the gizmo was still being dragged: {backend.measured()}"
    assert goal is None, "a target was committed mid-drag"
    print("held during drag: OK (no command issued)")

    # Release commits once; derive the tick budget from the cap-dependent plan duration.
    fire("end")
    assert not marker.dragging, "drag-end event did not clear the dragging flag"
    q_ref, ok_first, goal, note = step(goal)
    assert ok_first, f"the release did not produce a plan: {note}"
    assert goal is not None, "release did not commit a goal"
    print(f"plan on release: OK ({note})")
    dq_seen = []
    for _ in range(400):                       # generous ceiling; asserted to finish early
        q_ref, _, goal, _ = step(goal)
        dq_seen.append(float(np.max(np.abs(backend.reference()[1]))))
        if np.max(np.abs(goal - q_ref)) < 1e-6 and not backend.active_traj():
            break
    else:
        raise AssertionError("planned trajectory did not finish within 400 ticks")
    set_arm(model, data, backend.measured())
    render(server, data, handles)
    ee_final = data.xpos[ee_bid].copy()
    tgt, _ = marker.get()
    err = float(np.linalg.norm(ee_final - tgt))
    print(f"target marker (render) = {np.round(tgt,3)}")
    print(f"reached EE     (render) = {np.round(ee_final,3)}   ‖err‖ = {err*1000:.1f} mm")
    print(f"final q = {np.round(backend.measured(),3)}")
    assert np.all(np.isfinite(backend.measured())), "non-finite q"
    assert err < 0.02, f"EE did not reach the dragged target (err {err*1000:.1f} mm)"
    # The whole point of the rewrite: a NONZERO velocity reference is published while
    # moving. dq_des = 0 is what applied ~19 Nm against the motion on arm_base.
    assert max(dq_seen) > 1e-3, \
        f"no velocity reference was ever published (max |dq_ref| = {max(dq_seen):.2e})"
    assert dq_seen[-1] < 1e-9, f"trajectory ended with |dq_ref| = {dq_seen[-1]:.2e}, not rest"
    print(f"velocity reference: OK (peak |dq_ref| = {max(dq_seen):.3f} rad/s, ends at rest)")

    # Viser synthesizes drag-end on disconnect; discard that abandoned commit.
    connected = {1}
    q_before = backend.measured().copy()
    marker.set(ee0 + np.array([-0.05, 0.0, -0.03]), data.xquat[ee_bid].copy())
    fire("start", client_id=1)
    fire("update", client_id=1)
    connected.discard(1)                      # browser goes away mid-drag
    fire("end", client_id=1)                  # <- viser's synthetic end
    goal_d = None
    for _ in range(20):
        _, _, goal_d, _ = step(goal_d, owner=1, connected=connected)
    assert goal_d is None, "synthetic end from a DISCONNECTED client was committed"
    assert np.allclose(backend.measured(), q_before), "arm moved on a disconnect"
    print("disconnect mid-drag: OK (synthetic end ignored, arm did not move)")

    # Ignore releases from non-owning clients.
    connected = {1, 2}
    fire("start", client_id=2)
    fire("end", client_id=2)
    goal_n = None
    for _ in range(5):
        _, _, goal_n, _ = step(goal_n, owner=1, connected=connected)
    assert goal_n is None, "a non-owner client committed a target"
    print("non-owner release: OK (ignored)")

    # Each GUI tick consumes a fixed number of publisher-clock samples, regardless of stalls.
    marker.set(ee0 + np.array([0.04, 0.0, 0.02]), data.xquat[ee_bid].copy())
    fire("start"); fire("end")
    _, ok_s, goal_s, _ = step(None)
    assert ok_s, "setup for the publish-clock check did not plan"
    n_before = backend.slot._k
    for _ in range(3):
        step(goal_s)
    advanced = backend.slot._k - n_before
    assert advanced == 3 * STEPS_PER_TICK, \
        f"3 ticks advanced the reference by {advanced} samples, expected {3*STEPS_PER_TICK}"
    print(f"publish-clock pacing: OK ({STEPS_PER_TICK} samples/tick, independent of GUI dt)")

    # A mid-move release must re-plan C^1-continuously rather than step the velocity --
    # what joint_traj.quintic_from_state buys, and it had zero coverage before.
    #
    # The invariant lives at sample 0 of the NEW trajectory, which must equal the reference
    # the plan was seeded from. Comparing reference() before against reference() after is
    # WRONG: one teleop_step also advances STEPS_PER_TICK samples, so the two differ by a
    # tick of legitimate evolution (~2e-2 rad/s) even when the splice is exact.
    q_pre, dq_pre, ddq_pre = (v.copy() for v in backend.reference())
    assert np.max(np.abs(dq_pre)) > 1e-3, "the re-plan test did not actually happen mid-move"
    marker.set(ee0 + np.array([-0.03, 0.02, 0.05]), data.xquat[ee_bid].copy())
    fire("start"); fire("end")
    _, ok_m, goal_m, note_m = step(goal_s)
    assert ok_m, f"mid-move re-plan was refused: {note_m}"
    assert backend.active_traj(), "the re-plan produced no trajectory"
    q_new, dq_new, ddq_new = (a[0] for a in backend.slot._traj)
    for name, was, now in (("q", q_pre, q_new), ("dq", dq_pre, dq_new),
                           ("ddq", ddq_pre, ddq_new)):
        jump = float(np.max(np.abs(now - was)))
        assert jump < 1e-9, f"re-plan stepped {name} by {jump:.3e} at the splice (not C^2)"
    print(f"mid-move re-plan: OK (q/dq/ddq continuous at the splice, "
          f"|dq| was {np.max(np.abs(dq_pre)):.3f} rad/s)")
    backend.abort_traj()

    # No solve returns ``None`` so callers can retain the previous status.
    _, ok_none, _, _ = step(None)
    assert ok_none is None, "teleop_step reported a solve result without solving"
    print("IK status latching: OK (no phantom 'ok' between solves)")

    # Test TrajectorySlot independently because it feeds the publisher thread.
    slot = TrajectorySlot()
    hold = np.full(NUM_ARM, 0.25)
    q_s, dq_s = slot.advance(1, fallback=hold)
    np.testing.assert_allclose(q_s, hold)                 # idle -> the fallback, at rest
    np.testing.assert_allclose(dq_s, 0.0)
    qs = np.linspace(0.0, 1.0, 10)[:, None] * np.ones(NUM_ARM)
    dqs = np.full((10, NUM_ARM), 0.5)
    slot.load(qs, dqs, np.zeros((10, NUM_ARM)))
    assert slot.active(), "load() did not activate the slot"
    seen = [slot.advance(1)[0][0] for _ in range(10)]
    np.testing.assert_allclose(seen, qs[:, 0])            # every sample, in order, once
    assert not slot.active(), "slot still active after consuming every sample"
    # Past the end it latches the ENDPOINT at rest and keeps returning it, rather than
    # going silent (which would leave the drive holding a value nothing is watching).
    for _ in range(3):
        q_e, dq_e = slot.advance(1)
        np.testing.assert_allclose(q_e, qs[-1])
        np.testing.assert_allclose(dq_e, 0.0)
    # abort() mid-playback latches where it IS, not where it was going.
    slot.load(qs, dqs, np.zeros((10, NUM_ARM)))
    slot.advance(4)
    mid = slot.reference()[0].copy()
    slot.abort()
    assert not slot.active(), "abort() left the slot active"
    np.testing.assert_allclose(slot.reference()[0], mid)
    np.testing.assert_allclose(slot.reference()[1], 0.0)
    for bad in (np.zeros((0, NUM_ARM)), np.zeros((3, NUM_ARM))):
        try:
            slot.load(bad, dqs, np.zeros((10, NUM_ARM)))
        except ValueError:
            pass
        else:
            raise AssertionError(f"load() accepted mismatched arrays {bad.shape}")
    print("TrajectorySlot: OK (ordered playback, endpoint latch, abort, input guards)")

    # The safety monitor fails closed on stale state or over-limit torque.
    class _Stale(SimBackend):
        def state_age(self):
            return 9.9

    class _Hot(SimBackend):
        def torque(self):
            return np.full(NUM_ARM, 99.0)

    assert SafetyMonitor().check(SimBackend()) == "", "monitor tripped on a good state"
    assert "stale" in SafetyMonitor().check(_Stale()), "stale state not caught"
    mon, hot = SafetyMonitor(), _Hot()
    trips = [mon.check(hot) for _ in range(TRIP_SAMPLES)]
    assert trips[0] == "" and trips[-1] != "", f"debounced torque trip failed: {trips}"
    print("safety monitor: OK (stale + debounced over-torque both trip)")

    # Home uses the same validated planner and velocity cap as dragged targets.
    backend.q = CENTER.copy()
    backend.abort_traj()
    backend.slot._latched = None
    t_h, q_h, dq_h, ddq_h = plan_joint_move(backend, HOME_POSE, 1.0)
    assert not J.validate(t_h, q_h, dq_h, ddq_h, model=PLANT_MODEL, warn=False), \
        "the home trajectory does not pass its own validation"
    backend.command_traj(q_h, dq_h, ddq_h)
    goal_h = q_h[-1].copy()
    steps = []
    budget = int(np.ceil(t_h[-1] / (1.0 / FPS))) + 5   # from the PLAN, not a magic number
    for _ in range(budget):
        prev = backend.reference()[0].copy()
        _, _, goal_h, _ = step(goal_h)
        steps.append(float(np.max(np.abs(backend.reference()[0] - prev))))
    reached = float(np.max(np.abs(backend.measured() - HOME_POSE)))
    cap = float(J.DQ_MAX.max()) * STEPS_PER_TICK * PUBLISH_DT
    assert reached < 1e-3, f"arm did not reach home (max err {reached:.4f} rad)"
    assert max(steps) <= cap + 1e-9, \
        f"home advance {max(steps):.4f} rad/tick exceeded DQ_MAX*{STEPS_PER_TICK}*dt = {cap:.4f}"
    pk_h = J.peaks(dq_h, ddq_h, PUBLISH_DT)
    print(f"home pose: OK (T={t_h[-1]:.2f}s, reached zeros, max advance "
          f"{max(steps):.4f} <= {cap:.4f} rad/tick)")
    print(f"home profile peaks: dq={pk_h['dq'].max():.2f} ddq={pk_h['ddq'].max():.2f} "
          f"jerk={pk_h['jerk'].max():.2f} (caps {J.DQ_MAX.max():.0f}/"
          f"{J.DDQ_MAX.max():.0f}/{J.JERK_MAX.max():.0f})")

    # The click gating: armed + teleop on + owner, all required before any motion.
    assert home_refusal(False, True, 1, 1).startswith("home refused: session")
    assert home_refusal(True, False, 1, 1).startswith("home refused: enable")
    assert home_refusal(True, True, 1, 2).startswith("home refused: client 2")
    assert home_refusal(True, True, 1, 1) == ""      # owner, armed, teleop on
    assert home_refusal(True, True, None, 7) == ""   # no owner recorded yet
    print("home gating: OK (un-armed / teleop-off / non-owner all refused)")

    # Teleop off must ABORT an in-flight trajectory, not merely stop issuing new ones --
    # with playback in a background thread, "stop commanding" is no longer automatic.
    backend.q = CENTER.copy()
    backend.slot._latched = None
    t_o, q_o, dq_o, ddq_o = plan_joint_move(backend, HOME_POSE, 1.0)
    backend.command_traj(q_o, dq_o, ddq_o)
    assert backend.active_traj(), "setup failed: no trajectory loaded"
    _, _, g_off, _ = step(HOME_POSE.copy(), teleop_on=False)
    assert g_off is None, "a home goal survived teleop being off"
    assert not backend.active_traj(), "an in-flight trajectory survived teleop being off"
    assert np.allclose(backend.measured(), CENTER), "arm moved home with teleop off"
    print("teleop off: OK (goal dropped, in-flight trajectory aborted, arm did not move)")

    server.stop()
    print("SELF-CHECK PASS")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("iface", nargs="?", default="lo", help="DDS network interface (default lo)")
    ap.add_argument("--sim", action="store_true",
                    help="no hardware: the arm follows the IK solution directly (kinematic preview)")
    ap.add_argument("--self-check", action="store_true", help="headless logic check, then exit")
    # Localhost by DEFAULT: this commands a real arm and has no transport security.
    ap.add_argument("--host", default="127.0.0.1",
                    help="Viser bind host (default 127.0.0.1 = localhost only; "
                         "use 0.0.0.0 to expose on the LAN -- see --allow-remote)")
    ap.add_argument("--port", type=int, default=8080, help="Viser HTTP port (default 8080)")
    ap.add_argument("--share", action="store_true",
                    help="request a public share URL (tunnel). Refused when teleop is "
                         "armed -- a public tunnel must not be able to drive the arm.")
    ap.add_argument("--allow-remote", action="store_true",
                    help="acknowledge binding to a non-localhost host")
    ap.add_argument("--scene", default=None,
                    help="path to scene.xml (else $PINEAPPLE_SCENE_XML, else repo-relative)")
    ap.add_argument("--no-teleop", action="store_true",
                    help="view-only: never arm teleop, no terminal prompt")
    args = ap.parse_args()

    remote = args.host not in ("127.0.0.1", "localhost", "::1")
    if remote and not args.allow_remote:
        ap.error(f"--host {args.host} exposes arm teleop beyond this machine; "
                 "pass --allow-remote to confirm")

    if args.self_check:
        return self_check(args.port, scene=args.scene)

    if args.sim:
        print("[vis] SIM mode -- no DDS; the rendered arm follows the IK solution.")
        run_server(SimBackend(), args.host, args.port, args.share, is_real=False,
                   scene=args.scene, armed=not args.no_teleop)
        return 0

    # Enabling teleop from a browser must require shell access on THIS machine.
    armed = False
    if not args.no_teleop:
        print("WARNING: this drives the REAL arm. Keep the workspace clear.")
        if remote:
            print(f"WARNING: bound to {args.host} -- anyone who can reach this port "
                  "and arms teleop can move the arm.")
        armed = input("Type ARM to allow teleop this session (anything else = "
                      "view-only): ").strip().upper() == "ARM"
    print(f"[vis] teleop capability: {'ARMED' if armed else 'view-only'}")
    if args.share and armed:
        print("[vis] ERROR: --share opens a PUBLIC tunnel; refusing while teleop is "
              "armed. Re-run with --no-teleop to share a view-only session.")
        return 2
    run_server(RealBackend(args.iface), args.host, args.port, args.share,
               is_real=True, scene=args.scene, armed=armed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
