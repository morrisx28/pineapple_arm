"""Web-based (Viser) visualization + EE-drag teleop for the pineapple arm.

A browser 3D view (headless-friendly, no local display) that:
  1. mirrors the REAL arm -- renders measured `rt/lowstate` joint state posed by
     forward kinematics each frame (KINEMATICS ONLY -- no `mj_step`, no physics);
  2. shows a draggable 6-DOF "EE target" gizmo. Drag and RELEASE: on release the
     pose is solved by IK once and the arm slews there (slew-rate limited).

Commit-on-release: while you HOLD the gizmo the arm is not commanded at all. Only
letting go commits a target. This stops the drag streaming a chase of intermediate
poses to the robot, and solves IK once per gesture instead of every frame.

Access control: this commands a real arm and has no transport security, so it binds
to LOCALHOST by default. Exposing it needs an explicit `--host 0.0.0.0 --allow-remote`.
On the real arm, teleop must additionally be ARMED by typing ARM at the terminal
(i.e. it requires shell access on this machine); otherwise the session is view-only.
Exactly one client owns teleop -- the one that ticked the checkbox -- and only that
client's gizmo releases command the arm. `--share` (public tunnel) is refused while
teleop is armed.

Env: the `mujoco-learning` conda python (mujoco + pinocchio + unitree_sdk2py + viser).

    conda run -n mujoco-learning python pineapple_arm_vis.py            # real robot (DDS iface "lo")
    conda run -n mujoco-learning python pineapple_arm_vis.py eth0       # real robot on eth0
    conda run -n mujoco-learning python pineapple_arm_vis.py --sim      # no hardware: arm follows IK
    conda run -n mujoco-learning python pineapple_arm_vis.py --self-check   # headless logic check

Browser GUI: "Enable teleop" (starts OFF), "Re-home marker to EE", "Max joint speed".

Safety: startup FAILS CLOSED -- no `rt/lowstate` means the process refuses to run
rather than commanding a zero pose at full gain. A live watchdog stops commanding and
disarms teleop on a DDS dropout (stale state) or sustained over-torque/over-velocity.
Teleop is OFF at startup; the gizmo starts exactly on the EE (no jump when you
enable); nothing is commanded until you release the gizmo, and never mid-drag; the
per-tick slew step is capped by the control dt so a loop stall cannot become one big
jump; IK non-convergence keeps the previous target and the failure stays LATCHED in
the status panel; toggling teleop off, re-homing, or an owner disconnect clears the
pending target; gravity+friction feedforward stays active through the Controller.

Disconnect note: viser 1.0.30 does NOT simply drop the drag "end" when a browser goes
away mid-drag -- `SceneApi._drop_active_drags_for_client` synthesizes one from the last
observed pose. Committing that would move the arm to a half-dragged pose on every
closed tab, so each pending commit carries its client id and is discarded unless that
client is still connected and owns teleop.

Frame note: the IK URDF and the render MJCF mount the arm at different heights (a
constant ~[0,0,0.073] m z offset, same orientation). It is measured at startup and
applied so the gizmo lines up with the render and the IK target stays correct.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import socket
import sys
import time
import types

import numpy as np
import mujoco

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import arm_ik  # pinocchio IK (reuses model/robot.urdf); needs the real pinocchio
import arm_ff  # torque limits for the live safety watchdog
import pinocchio

EE_BODY = "gripper_case_link"
NUM_ARM = arm_ik.NUM_ARM_DOF          # 6
CENTER = np.array([0.0, 0.8, -0.8, 0.3, 0.0, 0.0])
MAX_JOINT_SPEED = 1.0                 # rad/s slew cap on the commanded target (safety)
FPS = 50.0

# Live safety limits (same basis as verify_gravity.py / collect_data.py).
STATE_TIMEOUT = 0.2                   # s; older measured state => stop commanding
SAFETY_TAU = 0.90 * arm_ff.TAU_LIMIT  # [27,27,27,7,7,7] * 0.9
DQ_LIMIT = np.full(NUM_ARM, 6.0)      # rad/s
TRIP_SAMPLES = 3                      # consecutive over-limit states before tripping

# Resolved at runtime -- never hard-code one machine's path.
_SCENE_CANDIDATES = (
    os.path.join(_HERE, "model", "scene.xml"),
    os.path.join(_HERE, "..", "pineapple_mujoco", "pineapple_robots",
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
    """Constant MJCF-EE minus pinocchio-EE position: the two models mount the arm at
    different heights. Orientation matches."""
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


class RealBackend:
    """Wraps pineapple_arm.Controller: mirrors measured q, streams joint targets."""

    def __init__(self, iface, timeout=5.0):
        from unitree_sdk2py.core.channel import ChannelFactoryInitialize
        import pineapple_arm

        class _StampedController(pineapple_arm.Controller):
            """Controller that records WHEN each state arrived: the base class keeps
            no timestamp, so a DDS dropout is undetectable. Subclassing keeps
            pineapple_arm.py untouched."""

            def __init__(self):
                super().__init__()
                self.t_state = 0.0

            def LowStateMessageHandler(self, msg):
                super().LowStateMessageHandler(msg)
                self.t_state = time.perf_counter()

        ChannelFactoryInitialize(1, iface)
        self.c = _StampedController()
        self.c.Init()
        t0 = time.perf_counter()   # wait for the first measured state
        while self.c.low_state is None and time.perf_counter() - t0 < timeout:
            time.sleep(0.02)
        # FAIL CLOSED: warning here and arming pose mode with measured()==zeros and
        # step=transition_steps (skipping the ramp) published q=0 at full kp -- a
        # full-gain slam to zero from wherever the arm physically was.
        if self.c.low_state is None:
            self.c.ShutDown()
            raise RuntimeError(
                f"no rt/lowstate within {timeout:.0f}s on iface '{iface}'. Refusing to "
                "start: without a measurement the arm would be commanded to zero at "
                "full gain. Check the arm/sim is up and the DDS interface is right.")
        # Only now, with a REAL measurement: primes target_dof_pos at the current
        # pose, so no motion is commanded.
        self.c.setTargetPose(self.measured())
        self.c.mode = "pose"
        self.c.step = self.c.transition_steps

    def measured(self):
        return np.asarray(self.c.qpos, dtype=float).copy()

    def state_age(self):
        return time.perf_counter() - self.c.t_state

    def torque(self):
        return np.asarray(self.c.qtau, dtype=float).copy()

    def velocity(self):
        return np.asarray(self.c.qvel, dtype=float).copy()

    def command(self, q6):
        self.c.setTargetPose(q6)

    def shutdown(self):
        self.c.ShutDown()


class SimBackend:
    """No hardware: the 'measured' state is just the last commanded (kinematic)."""

    def __init__(self):
        self.q = CENTER.copy()

    def measured(self):
        return self.q.copy()

    def command(self, q6):
        self.q = np.asarray(q6, dtype=float).copy()

    # The watchdog interface: a kinematic sim is always fresh and unloaded.
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
    """Draggable EE-target gizmo. get()/set() its world pose as (pos, wxyz).

    Also tracks whether the user is CURRENTLY dragging. Viser's ``on_update`` fires
    with ``event.phase`` in {"start", "update", "end"} over a gesture, so the control
    loop can tell "still dragging" from "let go" instead of guessing from pose
    changes. While a drag is in progress the arm is not commanded; the new target is
    committed once, on release (see ``teleop_step``).

    The handler is ``async def`` deliberately: viser runs plain ``def`` callbacks in a
    threadpool where, per its docs, phases "may run out of order" -- an "end" before
    its "start" would leave ``dragging`` stuck True and the arm unresponsive. Async
    callbacks are awaited in order, and only flip flags so the IK solve stays on the
    control-loop thread.

    DISCONNECT IS NOT A RELEASE. viser 1.0.30 does not drop the "end" event when a
    browser goes away mid-drag -- ``SceneApi._drop_active_drags_for_client``
    SYNTHESIZES one "using the most recently observed client-reported positions".
    Committing that would move the arm to a half-dragged pose on every closed tab, so
    each pending commit records its client and is only consumed if that client is
    STILL CONNECTED and owns teleop. The disconnect bookkeeping and the synthetic end
    both run on viser's event loop while the control loop is a separate thread, so the
    disconnect is already recorded by the next <=20 ms tick -- no race.
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
                    # Older viser: no phase info, so every change is a commit
                    # (restores the pre-drag-detection streaming behaviour).
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


def teleop_step(model, data, ee_bid, marker, offset, backend, desired, dt,
                teleop_on, max_speed, goal=None, owner=None, connected=None):
    """One control step. Returns (desired, ok, ee_pos, ee_quat, goal).

    ``ok`` is TRI-STATE: True/False when an IK solve ran this tick, ``None`` when
    none did, so the caller can LATCH the last real result instead of letting a
    failure flash for one 20 ms frame and then read "ok" again.

    The arm is commanded only toward a committed ``goal``, committed only on RELEASE,
    so dragging never streams intermediate poses. ``owner``/``connected`` gate whose
    release counts (see ``Marker.take_commit``). IK is solved once per release, which
    also keeps the joint branch stable while the arm moves.
    """
    meas = backend.measured()
    set_arm(model, data, meas)                       # render measured arm; updates EE xpos/xquat
    ee_pos = data.xpos[ee_bid].copy()
    ee_quat = data.xquat[ee_bid].copy()

    if not teleop_on:
        marker.set(ee_pos, ee_quat)                  # gizmo sticks to EE (no jump on enable)
        marker.take_commit()                         # drop a release that happened while off
        return desired, None, ee_pos, ee_quat, None  # stale goal must not survive re-enable

    ok = None                                        # None = no solve happened this tick
    if marker.take_commit(owner=owner, connected=connected):
        m_pos, m_wxyz = marker.get()
        q_sol, ok = arm_ik.solve_ik(meas, m_pos - offset, quat_to_mat(m_wxyz))
        if ok:
            goal = np.asarray(q_sol, dtype=float)
        # On IK failure keep the previous goal: never chase an unconverged solution.

    if marker.dragging or goal is None:
        return desired, ok, ee_pos, ee_quat, goal    # hold: issue no new command

    # Cap the control dt. `dt` is wall time since the last tick and is unbounded
    # above: after a rendering/network stall it would allow max_speed*dt in ONE step
    # (a 1 s stall => 1.0 rad instead of 0.02), defeating the slew limit.
    step = max_speed * min(dt, 1.0 / FPS)
    desired = desired + np.clip(goal - desired, -step, step)
    backend.command(desired)
    return desired, ok, ee_pos, ee_quat, goal


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

    # ---- GUI ------------------------------------------------------------- #
    with server.gui.add_folder("Teleop"):
        cb_teleop = server.gui.add_checkbox(
            "Enable teleop", False, disabled=not armed,
            hint=("armed at the terminal" if armed else
                  "NOT ARMED: restart and confirm at the terminal to enable"))
        btn_home = server.gui.add_button("Re-home marker to EE")
        sl_speed = server.gui.add_slider("Max joint speed [rad/s]", 0.1, 3.0, 0.1,
                                         MAX_JOINT_SPEED)

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

    @btn_home.on_click
    def _(_ev):
        rehome["flag"] = True

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

    desired = backend.measured()
    goal = None                      # committed joint target; set on gizmo release
    ik_ok = None                     # LATCHED last real IK result (None = none yet)
    monitor = SafetyMonitor()
    period = 1.0 / FPS
    last = time.perf_counter()
    try:
        while True:
            now = time.perf_counter()
            dt = max(now - last, 1e-3)
            last = now

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
                ctl["disarm"] = False

            if rehome["flag"]:
                set_arm(model, data, backend.measured())
                marker.set(data.xpos[ee_bid].copy(), data.xquat[ee_bid].copy())
                desired = backend.measured()
                goal = None          # snapping the marker must not re-trigger motion
                ik_ok = None         # clear a latched failure
                marker.take_commit()
                monitor.reset()
                rehome["flag"] = False

            if is_real:
                backend.c.use_gravity_comp = bool(cb_grav.value)
                backend.c.use_friction_comp = bool(cb_fric.value)

            teleop_on = bool(cb_teleop.value) and not alarm
            desired, ok, ee_pos, _, goal = teleop_step(
                model, data, ee_bid, marker, offset, backend, desired, dt,
                teleop_on, float(sl_speed.value), goal,
                owner=ctl["owner"], connected=ctl["connected"])
            if ok is not None:       # only a real solve updates the latch
                ik_ok = ok
            render(server, data, handles)

            m_pos, _ = marker.get()
            if alarm:
                motion = f"SAFETY STOP -- {alarm}"
            elif not armed:
                motion = "NOT ARMED (confirm at the terminal to enable)"
            elif not cb_teleop.value:
                motion = "teleop off"
            elif marker.dragging:
                motion = "DRAGGING -- arm held, releases on drop"
            elif goal is None:
                motion = "idle (drag the gizmo, then let go)"
            elif np.max(np.abs(goal - desired)) > 1e-3:
                motion = "slewing to released target"
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
                f"q cmd : {np.round(desired, 3).tolist()}\n"
                f"q goal: {'-' if goal is None else np.round(goal, 3).tolist()}"
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
    print(f"frame offset (MJCF-IK) = {np.round(offset, 5)} m  (expect ~[0,0,0.073])")
    assert len(meshes) > 10, "too few visual meshes extracted"
    assert np.allclose(offset[:2], 0, atol=1e-3) and abs(offset[2] - 0.0727) < 5e-3, \
        f"unexpected frame offset {offset}"

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

    desired = backend.measured()
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

    # 1) While the gizmo is still held, the arm must NOT be commanded.
    fire("start")
    fire("update")
    assert marker.dragging, "drag-start event did not register"
    for _ in range(20):
        desired, _, _, _, goal = teleop_step(model, data, ee_bid, marker, offset,
                                             backend, desired, 0.02, teleop_on=True,
                                             max_speed=1.0, goal=goal)
    assert np.allclose(backend.measured(), CENTER), \
        f"arm moved while the gizmo was still being dragged: {backend.measured()}"
    assert goal is None, "a target was committed mid-drag"
    print("held during drag: OK (no command issued)")

    # 2) Release -> commit exactly once -> slew to the target.
    fire("end")
    assert not marker.dragging, "drag-end event did not clear the dragging flag"
    ok_any = False
    for _ in range(200):  # ~4 s of slewing at 50 Hz
        desired, ok, _, _, goal = teleop_step(model, data, ee_bid, marker, offset,
                                              backend, desired, 0.02, teleop_on=True,
                                              max_speed=1.0, goal=goal)
        ok_any = ok_any or ok
    set_arm(model, data, backend.measured())
    render(server, data, handles)
    ee_final = data.xpos[ee_bid].copy()
    tgt, _ = marker.get()
    err = float(np.linalg.norm(ee_final - tgt))
    print(f"IK converged at least once: {ok_any}")
    print(f"target marker (render) = {np.round(tgt,3)}")
    print(f"reached EE     (render) = {np.round(ee_final,3)}   ‖err‖ = {err*1000:.1f} mm")
    print(f"final q = {np.round(backend.measured(),3)}")
    assert ok_any, "IK never converged"
    assert np.all(np.isfinite(backend.measured())), "non-finite q"
    assert err < 0.02, f"EE did not reach the dragged target (err {err*1000:.1f} mm)"

    # 3) P1b: viser SYNTHESIZES an "end" when a client disconnects mid-drag. A commit
    #    from a client that is gone must be dropped, or closing a tab moves the arm.
    connected = {1}
    q_before = backend.measured().copy()
    marker.set(ee0 + np.array([-0.05, 0.0, -0.03]), data.xquat[ee_bid].copy())
    fire("start", client_id=1)
    fire("update", client_id=1)
    connected.discard(1)                      # browser goes away mid-drag
    fire("end", client_id=1)                  # <- viser's synthetic end
    goal_d = None
    for _ in range(20):
        _, _, _, _, goal_d = teleop_step(model, data, ee_bid, marker, offset, backend,
                                         backend.measured(), 0.02, teleop_on=True,
                                         max_speed=1.0, goal=goal_d,
                                         owner=1, connected=connected)
    assert goal_d is None, "synthetic end from a DISCONNECTED client was committed"
    assert np.allclose(backend.measured(), q_before), "arm moved on a disconnect"
    print("disconnect mid-drag: OK (synthetic end ignored, arm did not move)")

    # 4) P1c: a release from a client that is not the owner must be ignored.
    connected = {1, 2}
    fire("start", client_id=2)
    fire("end", client_id=2)
    goal_n = None
    for _ in range(5):
        _, _, _, _, goal_n = teleop_step(model, data, ee_bid, marker, offset, backend,
                                         backend.measured(), 0.02, teleop_on=True,
                                         max_speed=1.0, goal=goal_n,
                                         owner=1, connected=connected)
    assert goal_n is None, "a non-owner client committed a target"
    print("non-owner release: OK (ignored)")

    # 5) P2a: a long loop stall must not become one huge slew step.
    far = backend.measured() + 1.0
    d0 = backend.measured().copy()
    d1, *_ = teleop_step(model, data, ee_bid, marker, offset, backend, d0,
                         5.0,                      # 5 s "stall"
                         teleop_on=True, max_speed=1.0, goal=far,
                         owner=None, connected=None)
    step = float(np.max(np.abs(d1 - d0)))
    assert step <= 1.0 / FPS + 1e-9, f"slew step {step:.3f} rad exceeded the dt cap"
    print(f"stall dt cap: OK (step {step:.4f} rad <= {1.0/FPS:.4f})")

    # 6) P3: with no solve this tick, ok is None so the caller can latch the last one.
    _, ok_none, _, _, _ = teleop_step(model, data, ee_bid, marker, offset, backend,
                                      backend.measured(), 0.02, teleop_on=True,
                                      max_speed=1.0, goal=None,
                                      owner=None, connected=None)
    assert ok_none is None, "teleop_step reported a solve result without solving"
    print("IK status latching: OK (no phantom 'ok' between solves)")

    # 7) P1a: the safety monitor fails closed on stale state / over-limit torque.
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
