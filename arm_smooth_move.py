"""Jerk-limited EE pose-to-pose motion through the hardware position servo.

IK is solved at the requested poses, then a validated joint trajectory is planned once
from a state snapshot. The controller publishes both position and velocity references at
200 Hz with gravity and optional friction feedforward.
"""

from __future__ import annotations

import argparse
import sys
import threading
import time
import traceback

import numpy as np

import arm_ff
import arm_ik
import ee_traj
import joint_traj as J

NUM_MOTORS = arm_ik.NUM_ARM_DOF  # 6
DT = 0.005                       # 200 Hz, matching pineapple_arm.py and the hardware rate

# This path relies on, rather than cancels, the hardware PD and its gain inflation.
KP = np.array([20.0, 40.0, 40.0, 20.0, 20.0, 20.0])
KD = np.array([0.5, 1.0, 1.0, 0.5, 0.5, 0.5])

# ``--sim`` must include measured gain inflation to represent hardware stiffness.
# Entries at 1.0 are unmeasured, not confirmed unity gains.
GAIN_INFLATION = np.array([1.0, 18.81, 12.56, 1.0, 1.0, 1.0])

# Debounce noisy soft violations; abort severe violations immediately.
SAFETY_TAU = 0.90 * arm_ff.TAU_LIMIT
SAFETY_DQ = np.full(NUM_MOTORS, 3.0)      # rad/s, well above the 2.0 rad/s reference cap
SAFETY_SEVERE = 1.10

# Home lies on two joint limits, and encoder offset can read slightly beyond them. Measure
# actual penetration instead of creating an inset that would command motion at rest.
LIMIT_PENETRATION = 0.05                  # rad
TRIP_SAMPLES = 3
STATE_TIMEOUT = 0.1                       # s without rt/lowstate -> abort
TRACK_ABORT = 0.35                        # rad of per-joint tracking error -> abort
SAFE_RETURN_S = 3.0


# Planning (no DDS or hardware)

def plan_to_poses(q_start, poses, rpy=None, dt=DT, duration=None, caps=None,
                  model=None, rot=None, dq_start=None, ddq_start=None, warn=True):
    """Return a validated jerk-limited joint trajectory through EE poses.

    IK is warm-started between poses. ``rot`` is a 3x3 alternative to ``rpy`` and the two
    are mutually exclusive. A single pose produces a point-to-point quintic; multiple
    poses produce a C2 spline.

    ``dq_start`` and ``ddq_start`` enable a moving start for a single target; they are
    rejected with multiple poses rather than silently discarding motion. ``warn`` controls
    start-limit diagnostics. IK or feasibility failure raises ``ee_traj.ReferenceError``.
    """
    caps = {} if caps is None else caps
    q_start = np.asarray(q_start, float)[:NUM_MOTORS]
    if rpy is not None and rot is not None:
        raise ValueError("pass rpy or rot, not both")
    R_des = np.asarray(rot, float) if rot is not None else (
        arm_ik.rpy_to_matrix(*rpy) if rpy is not None else None)
    if R_des is not None and R_des.shape != (3, 3):
        raise ValueError(f"rot must be 3x3, got {R_des.shape}")
    poses = list(poses)   # Materialize generators before counting and iteration.
    moving_start = dq_start is not None or ddq_start is not None
    if moving_start and len(poses) != 1:
        raise ValueError("dq_start/ddq_start require exactly one pose "
                         "(no moving-start spline through vias)")

    vias = [q_start]
    seed = q_start
    for i, p in enumerate(poses):
        q_sol, ok = arm_ik.solve_ik(seed, p, R_des)
        if not ok:
            raise ee_traj.ReferenceError(
                f"IK did not converge for pose {i} = {np.round(p, 4).tolist()}"
                f"{'' if rpy is None else f' rpy={np.round(rpy, 4).tolist()}'}; "
                "the target is outside the reachable workspace"
            )
        seed = np.asarray(q_sol, float)
        vias.append(seed)

    vias = np.asarray(vias, float)
    if moving_start:
        zero = np.zeros(NUM_MOTORS)
        t, q, dq, ddq = J.quintic_from_state(
            vias[0],
            zero if dq_start is None else dq_start,
            zero if ddq_start is None else ddq_start,
            vias[1], dt, duration=duration, **caps)
    elif len(vias) == 2:
        t, q, dq, ddq = J.quintic_point_to_point(vias[0], vias[1], dt,
                                                 duration=duration, **caps)
    else:
        t, q, dq, ddq = J.quintic_spline_via(vias, dt, duration=duration, **caps)

    issues = J.validate(t, q, dq, ddq, model=model, warn=warn, **caps)
    if issues:
        raise ee_traj.ReferenceError(
            "planned trajectory is not executable:\n  - " + "\n  - ".join(issues))
    return t, q, dq, ddq


# Hardware controller

class TrackingAbort(RuntimeError):
    """Raised by the watchdog or the tracking-error guard; always triggers safe_return."""


class SmoothMoveController:
    """DDS position servo that follows a precomputed jerk-limited joint trajectory."""

    def __init__(self, kp=KP, kd=KD, dt=DT, use_gravity=True, use_friction=False,
                 state_timeout=STATE_TIMEOUT, track_abort=TRACK_ABORT):
        # unitree_sdk2py is imported here, not at module scope, so --dry-run / --sim /
        # --compare-linear run in an env without the SDK.
        from unitree_sdk2py.core.channel import ChannelPublisher, ChannelSubscriber
        from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowCmd_
        from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowCmd_, LowState_
        from unitree_sdk2py.utils.crc import CRC

        self.kp = np.asarray(kp, float)
        self.kd = np.asarray(kd, float)
        for name, g in (("kp", self.kp), ("kd", self.kd)):
            # Validated BEFORE the publisher exists: arm_tvlqr once published nan to the
            # motors because a bad --kp reached the wire.
            if g.shape != (NUM_MOTORS,) or not np.all(np.isfinite(g)) or np.any(g < 0):
                raise ValueError(f"{name} must be {NUM_MOTORS} finite non-negative "
                                 f"values, got {g}")
        self.dt = float(dt)
        self.use_gravity = bool(use_gravity)
        self.use_friction = bool(use_friction)
        self.state_timeout = float(state_timeout)
        self.track_abort = float(track_abort)

        self.low_cmd = unitree_go_msg_dds__LowCmd_()
        self._init_low_cmd()
        self.crc = CRC()

        self._lock = threading.Lock()
        self._qpos = np.zeros(NUM_MOTORS)
        self._qvel = np.zeros(NUM_MOTORS)
        self._qtau = np.zeros(NUM_MOTORS)
        self._stamp = None            # None until the first rt/lowstate arrives
        self._trip = np.zeros(NUM_MOTORS, dtype=int)
        self._published = False       # gates safe_return; see there for why

        self.lowcmd_publisher = ChannelPublisher("rt/lowcmd", LowCmd_)
        self.lowcmd_publisher.Init()
        self.lowstate_subscriber = ChannelSubscriber("rt/lowstate", LowState_)
        self.lowstate_subscriber.Init(self._on_low_state, 10)

    # -- plumbing ---------------------------------------------------------- #

    def _init_low_cmd(self):
        self.low_cmd.head[0] = 0xFE
        self.low_cmd.head[1] = 0xEF
        self.low_cmd.level_flag = 0xFF
        self.low_cmd.gpio = 0
        for i in range(NUM_MOTORS):
            m = self.low_cmd.motor_cmd[i]
            m.mode = 0x01   # PMSM
            m.q = 0.0
            m.dq = 0.0
            m.kp = 0.0
            m.kd = 0.0
            m.tau = 0.0

    def _on_low_state(self, msg):
        with self._lock:
            for i in range(NUM_MOTORS):
                self._qpos[i] = msg.motor_state[i].q
                self._qvel[i] = msg.motor_state[i].dq
                self._qtau[i] = msg.motor_state[i].tau_est
            self._stamp = time.perf_counter()

    def snapshot(self):
        """Thread-safe copy of the latest measured state -> (q, dq, tau, stamp)."""
        with self._lock:
            return (self._qpos.copy(), self._qvel.copy(), self._qtau.copy(), self._stamp)

    def wait_for_state(self, timeout=2.0):
        """Block until rt/lowstate arrives. Fails CLOSED -- returns False on timeout."""
        deadline = time.perf_counter() + float(timeout)
        while time.perf_counter() < deadline:
            if self.snapshot()[3] is not None:
                return True
            time.sleep(0.01)
        return False

    def _write(self, q_des, dq_des, tau):
        """Publish one command. Refuses non-finite values and clips to the exact limits."""
        for name, v in (("q_des", q_des), ("dq_des", dq_des), ("tau", tau)):
            # np.clip does NOT remove NaN, so this check cannot be folded into the clip.
            if not np.all(np.isfinite(v)):
                raise TrackingAbort(f"refusing to publish non-finite {name}: {v}")
        # EXACT limits, no inset margin: the all-zeros resting pose IS arm_base's low and
        # upper_arm's high limit, so an inset would command a step off the stop at t=0 --
        # a jerk injected by the safety code (README quirk 2).
        q_des = J.clip_to_limits(q_des)
        for i in range(NUM_MOTORS):
            m = self.low_cmd.motor_cmd[i]
            m.q = float(q_des[i])
            m.dq = float(dq_des[i])
            m.kp = float(self.kp[i])
            m.kd = float(self.kd[i])
            m.tau = float(tau[i])
        self.low_cmd.crc = self.crc.Crc(self.low_cmd)
        self.lowcmd_publisher.Write(self.low_cmd)
        self._published = True

    def feedforward(self, q, dq):
        """Gravity + friction feedforward, in the COMMAND domain.

        Routed through ``arm_ff.motor_tau``, which is mandatory: a raw joint torque
        over-drives arm_base by 5x and upper_arm by 3.3x (measured per joint,
        model/tau_cmd_scale.json). This does NOT use an RNEA reference torque, so gravity is
        applied exactly once.
        """
        if not (self.use_gravity or self.use_friction):
            return np.zeros(NUM_MOTORS)
        return arm_ff.motor_tau(arm_ff.feedforward(
            q, dq, gravity=self.use_gravity, friction=self.use_friction))

    def _watchdog(self, q, dq, tau, stamp, now):
        if stamp is None or (now - stamp) > self.state_timeout:
            age = float("inf") if stamp is None else (now - stamp)
            raise TrackingAbort(f"rt/lowstate is stale ({age*1e3:.0f} ms > "
                                f"{self.state_timeout*1e3:.0f} ms)")
        soft = (np.abs(tau) > SAFETY_TAU) | (np.abs(dq) > SAFETY_DQ) \
            | (q < J.JOINT_LOW - LIMIT_PENETRATION) \
            | (q > J.JOINT_HIGH + LIMIT_PENETRATION)
        severe = (np.abs(tau) > SAFETY_SEVERE * SAFETY_TAU) \
            | (np.abs(dq) > SAFETY_SEVERE * SAFETY_DQ)
        self._trip = np.where(soft, self._trip + 1, 0)
        for i in range(NUM_MOTORS):
            name = arm_ik.IK_MODEL.names[i + 1]
            if severe[i]:
                raise TrackingAbort(
                    f"{name}: severe limit breach -- tau={tau[i]:+.2f} Nm, "
                    f"dq={dq[i]:+.2f} rad/s, q={q[i]:+.3f} rad")
            if self._trip[i] >= TRIP_SAMPLES:
                raise TrackingAbort(
                    f"{name}: limit exceeded on {TRIP_SAMPLES} consecutive samples -- "
                    f"tau={tau[i]:+.2f} Nm, dq={dq[i]:+.2f} rad/s, q={q[i]:+.3f} rad")

    # -- execution --------------------------------------------------------- #

    def follow(self, t, q_ref, dq_ref, watchdog=True):
        """Stream a planned trajectory at 1/dt Hz. -> log dict.

        Paced against the TRAJECTORY clock, not per-iteration, so the motion cannot drift
        in time the way ``pineapple_arm.py``'s accumulating per-tick sleep does.

        No entry ramp: the trajectory starts at the measured q by construction, so its own
        zero-velocity, zero-acceleration start IS the gentle entry.
        """
        n = len(t)
        log = {k: np.zeros((n, NUM_MOTORS)) for k in
               ("q", "dq", "tau_meas", "tau_cmd", "q_ref", "dq_ref")}
        log["q_ref"][:] = q_ref
        log["dq_ref"][:] = dq_ref
        log["complete"] = False
        log["abort_reason"] = None
        log["overrun_ms"] = 0.0

        t0 = time.perf_counter()
        try:
            for k in range(n):
                q, dq, tau_meas, stamp = self.snapshot()
                now = time.perf_counter()
                if watchdog:
                    self._watchdog(q, dq, tau_meas, stamp, now)
                    err = np.abs(q - q_ref[k])
                    if np.any(err > self.track_abort):
                        j = int(np.argmax(err))
                        raise TrackingAbort(
                            f"{arm_ik.IK_MODEL.names[j+1]}: tracking error "
                            f"{err[j]:.3f} rad > {self.track_abort:.3f} rad "
                            f"at t={t[k]:.3f}s")

                tau_cmd = self.feedforward(q, dq)
                self._write(q_ref[k], dq_ref[k], tau_cmd)

                log["q"][k] = q
                log["dq"][k] = dq
                log["tau_meas"][k] = tau_meas
                log["tau_cmd"][k] = tau_cmd

                sleep = (k + 1) * self.dt - (time.perf_counter() - t0)
                if sleep > 0:
                    time.sleep(sleep)
                else:
                    log["overrun_ms"] = max(log["overrun_ms"], -sleep * 1e3)
                    if k % 200 == 0:
                        print(f"[smooth] WARNING: loop overran by {-sleep*1e3:.1f} ms")
            log["complete"] = True
        except TrackingAbort as e:
            log["abort_reason"] = str(e)
            print(f"[smooth] ABORT: {e}")
        return log

    def safe_return(self, duration=SAFE_RETURN_S):
        """Ramp gently from wherever the arm is to the zero pose, then release torque.

        Runs with the watchdog DISABLED: this is the recovery path, so a limit breach must
        not be able to block the very motion that resolves it.

        No-ops if this controller never published a command. It is called from a blanket
        ``finally``, which also fires when planning failed before any motion -- and moving
        the arm to zeros as a side effect of a failed IK solve would be a surprise. An
        unreachable target must move nothing, as in ``pineapple_arm.py``.
        """
        q, dq, _, stamp = self.snapshot()
        if stamp is None or not self._published:
            return
        try:
            t, q_ref, dq_ref, _ = J.quintic_point_to_point(
                q, J.clip_to_limits(np.zeros(NUM_MOTORS)), self.dt,
                duration=float(duration))
        except ValueError as e:
            print(f"[smooth] safe_return could not plan a ramp ({e}); holding position")
            return
        print(f"[smooth] safe_return: ramping to zero over {t[-1]:.1f} s")
        self.follow(t, q_ref, dq_ref, watchdog=False)
        for i in range(NUM_MOTORS):
            m = self.low_cmd.motor_cmd[i]
            m.kp = 0.0
            m.kd = 0.0
            m.tau = 0.0
        self.low_cmd.crc = self.crc.Crc(self.low_cmd)
        self.lowcmd_publisher.Write(self.low_cmd)


# Reporting and offline modes

def report_tracking(log):
    """Print measured-vs-reference tracking, including EE error."""
    q, q_ref = log["q"], log["q_ref"]
    err = q - q_ref
    print(f"\n{'joint':<20}{'RMS err':>12}{'max err':>12}{'|tau|meas':>12}")
    for j in range(NUM_MOTORS):
        print(f"{arm_ik.IK_MODEL.names[j+1]:<20}"
              f"{np.sqrt(np.mean(err[:, j]**2))*1e3:9.2f} mrad"
              f"{np.max(np.abs(err[:, j]))*1e3:9.2f} mrad"
              f"{np.max(np.abs(log['tau_meas'][:, j])):10.2f} Nm")
    model = ee_traj.build_arm_model()
    p, p_ref = ee_traj.ee_positions_of(q, model), ee_traj.ee_positions_of(q_ref, model)
    d = np.linalg.norm(p - p_ref, axis=1)
    ang = ee_traj.orientation_error(q, q_ref, model)
    print(f"\nEE position  RMS {np.sqrt(np.mean(d**2))*1e3:6.2f} mm   "
          f"max {d.max()*1e3:6.2f} mm")
    print(f"EE orientation RMS {np.degrees(np.sqrt(np.mean(ang**2))):5.2f} deg  "
          f"max {np.degrees(ang.max()):5.2f} deg")
    if log["overrun_ms"] > 0:
        print(f"worst loop overrun {log['overrun_ms']:.1f} ms")


def simulate(t, q_ref, dq_ref, kp=KP, kd=KD, model=None, mass_scale=1.0,
             gain_inflation=GAIN_INFLATION):
    """Closed-loop rollout against the pinocchio plant, reproducing the hardware law.

    ``tau_applied = tau_ff + S*kp*(q_ref - q) + S*kd*(dq_ref - dq)`` with semi-implicit
    Euler. The ``S`` factor is what makes this predictive: with ``S = 1`` the rollout is
    ~13x softer than the robot on the two loaded joints and reports tracking error that the
    hardware will not have. ``mass_scale != 1`` perturbs the plant so the result is not just
    the model tracking itself.
    """
    import pinocchio

    S = np.broadcast_to(np.asarray(gain_inflation, float), (NUM_MOTORS,))
    kp = np.asarray(kp, float) * S
    kd = np.asarray(kd, float) * S

    model = ee_traj.build_arm_model() if model is None else model
    plant = model.copy()
    if mass_scale != 1.0:
        for b in range(1, plant.njoints):
            inertia = plant.inertias[b]
            plant.inertias[b] = pinocchio.Inertia(
                inertia.mass * mass_scale, inertia.lever, inertia.inertia * mass_scale)
    data = plant.createData()

    dt = float(t[1] - t[0])
    n = len(t)
    q = np.zeros((n, NUM_MOTORS))
    dq = np.zeros((n, NUM_MOTORS))
    tau_ap = np.zeros((n, NUM_MOTORS))
    qk, vk = q_ref[0].copy(), dq_ref[0].copy()
    for k in range(n):
        q[k], dq[k] = qk, vk
        # Feedforward in the JOINT domain here, not the command domain: the plant is
        # physics, so it receives physical torque. motor_tau's scaling belongs only on the
        # wire (see arm_ff's domain rule).
        tau_ff = arm_ff.feedforward(qk, vk, gravity=True, friction=False)
        applied = tau_ff + kp * (q_ref[k] - qk) + kd * (dq_ref[k] - vk)
        applied = np.clip(applied, -arm_ff.TAU_LIMIT, arm_ff.TAU_LIMIT)
        tau_ap[k] = applied
        vk = vk + dt * pinocchio.aba(plant, data, qk, vk, applied)
        qk = qk + dt * vk
    return {"q": q, "dq": dq, "tau_meas": tau_ap, "tau_cmd": tau_ap,
            "q_ref": q_ref, "dq_ref": dq_ref, "complete": True,
            "abort_reason": None, "overrun_ms": 0.0}


def _plot(t, q, dq, ddq, path, linear=None):
    """Write a q/dq/ddq/jerk figure; optionally overlay the linear blend it replaces."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[smooth] (plot skipped: {e})")
        return
    dt = float(t[1] - t[0])
    jerk = np.gradient(ddq, dt, axis=0, edge_order=1)
    fig, ax = plt.subplots(4, 1, figsize=(10, 11), sharex=True)
    names = [arm_ik.IK_MODEL.names[j + 1] for j in range(NUM_MOTORS)]
    for j in range(NUM_MOTORS):
        ax[0].plot(t, q[:, j], label=names[j])
        ax[1].plot(t, dq[:, j])
        ax[2].plot(t, ddq[:, j])
        ax[3].plot(t, jerk[:, j])
    if linear is not None:
        tl, ql, dql, ddql = linear
        jl = np.gradient(ddql, float(tl[1] - tl[0]), axis=0, edge_order=1)
        for j in range(NUM_MOTORS):
            kw = dict(color="0.6", lw=0.8, ls="--")
            ax[0].plot(tl, ql[:, j], **kw)
            ax[1].plot(tl, dql[:, j], **kw)
            ax[2].plot(tl, ddql[:, j], **kw)
            ax[3].plot(tl, jl[:, j], **kw)
        ax[0].plot([], [], color="0.6", lw=0.8, ls="--",
                   label="linear blend (pineapple_arm.py)")
    for a, lab in zip(ax, ["q [rad]", "dq [rad/s]", "ddq [rad/s^2]", "jerk [rad/s^3]"]):
        a.set_ylabel(lab)
        a.grid(alpha=0.3)
    ax[0].legend(fontsize=7, ncol=3)
    ax[3].set_xlabel("t [s]")
    ax[0].set_title("jerk-limited quintic" + ("" if linear is None else " vs linear blend"))
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    print(f"[smooth] wrote {path}")


# CLI

def _parse_args(argv=None):
    ap = argparse.ArgumentParser(
        description="Jerk-limited point-to-point EE motion for the pineapple arm.")
    ap.add_argument("net", nargs="?", default=None,
                    help="DDS interface (e.g. eth0). Omit with --dry-run/--sim/"
                         "--compare-linear; 'lo' is the default when running on the robot.")
    ap.add_argument("--pose", nargs=3, type=float, metavar=("X", "Y", "Z"),
                    help="single EE target position [m]")
    ap.add_argument("--via", nargs="+", type=float, metavar="X Y Z ...",
                    help="EE positions to spline through, as a flat x y z x y z ... list")
    ap.add_argument("--rpy", nargs=3, type=float, default=None,
                    metavar=("R", "P", "Y"), help="EE orientation [rad]; default identity")
    ap.add_argument("--from-q", nargs=NUM_MOTORS, type=float, default=None,
                    help="start configuration for offline modes (default: zeros)")
    ap.add_argument("--dt", type=float, default=DT)
    ap.add_argument("--duration", type=float, default=None,
                    help="force a duration [s]; default is the shortest meeting the caps")
    ap.add_argument("--dq-max", type=float, default=None, help="rad/s (default 2.0)")
    ap.add_argument("--ddq-max", type=float, default=None, help="rad/s^2 (default 4.0)")
    ap.add_argument("--jerk-max", type=float, default=None, help="rad/s^3 (default 20.0)")
    ap.add_argument("--friction", action="store_true",
                    help="add friction feedforward (off by default, as in "
                         "pineapple_arm.py; gravity is always on)")
    ap.add_argument("--dry-run", action="store_true",
                    help="plan, validate and plot; no DDS, no motion")
    ap.add_argument("--sim", action="store_true",
                    help="closed-loop rollout against a perturbed pinocchio plant")
    ap.add_argument("--compare-linear", action="store_true",
                    help="overlay pineapple_arm.py's linear blend -- the anti-jerk claim")
    ap.add_argument("--mass-scale", type=float, default=1.15,
                    help="plant mass perturbation for --sim (default 1.15)")
    ap.add_argument("--no-gain-inflation", action="store_true",
                    help="--sim only: use the COMMANDED gains instead of the S-inflated "
                         "effective ones. Makes the rollout ~13x softer than the robot; "
                         "for isolating the profile from the servo, not for prediction.")
    ap.add_argument("--plot", default="smooth_move.png")
    ap.add_argument("--yes", action="store_true", help="skip the on-robot confirmation")
    args = ap.parse_args(argv)

    for name in ("dt", "duration", "dq_max", "ddq_max", "jerk_max", "mass_scale"):
        v = getattr(args, name)
        # Finiteness first: NaN passes ordinary comparisons silently.
        if v is not None and (not np.isfinite(v) or v <= 0):
            ap.error(f"--{name.replace('_', '-')} must be finite and positive, got {v}")
    if args.via is not None and len(args.via) % 3 != 0:
        ap.error(f"--via needs a multiple of 3 values (x y z ...), got {len(args.via)}")
    if args.pose is None and args.via is None:
        ap.error("give a target: --pose X Y Z, or --via X Y Z X Y Z ...")
    if args.pose is not None and args.via is not None:
        ap.error("--pose and --via are mutually exclusive")
    return args


def _poses_of(args):
    if args.pose is not None:
        return [np.asarray(args.pose, float)]
    v = np.asarray(args.via, float).reshape(-1, 3)
    return [row for row in v]


def _caps_of(args):
    caps = {}
    for key, val in (("dq_max", args.dq_max), ("ddq_max", args.ddq_max),
                     ("jerk_max", args.jerk_max)):
        if val is not None:
            caps[key] = np.full(NUM_MOTORS, float(val))
    return caps


def main(argv=None):
    args = _parse_args(argv)
    poses = _poses_of(args)
    caps = _caps_of(args)
    offline = args.dry_run or args.sim or args.compare_linear
    model = ee_traj.build_arm_model()

    if offline:
        # NOT clipped to limits: the point of --from-q is to reproduce a real measured
        # state, and the measured resting pose reads slightly OUTSIDE arm_base's limit.
        # Clipping would erase exactly the condition the offline modes need to test.
        q_start = np.zeros(NUM_MOTORS) if args.from_q is None \
            else np.asarray(args.from_q, float)
    else:
        q_start = None   # taken from the robot below

    controller = None
    try:
        if not offline:
            from unitree_sdk2py.core.channel import ChannelFactoryInitialize
            print("WARNING: ensure there are no obstacles around the robot.")
            if not args.yes:
                input("Press Enter to continue...")
            ChannelFactoryInitialize(1, args.net or "lo")
            controller = SmoothMoveController(dt=args.dt, use_friction=args.friction)
            if not controller.wait_for_state():
                print("[smooth] no rt/lowstate received -- refusing to command. "
                      "Is the arm powered and the interface correct?")
                return 2
            q_start = controller.snapshot()[0]
            print(f"[smooth] measured start q = {np.round(q_start, 4).tolist()}")

        t, q, dq, ddq = plan_to_poses(q_start, poses, rpy=args.rpy, dt=args.dt,
                                      duration=args.duration, caps=caps, model=model)
        eff = {"dq_max": J.DQ_MAX, "ddq_max": J.DDQ_MAX, "jerk_max": J.JERK_MAX, **caps}
        print(f"\ncaps: dq_max={eff['dq_max'][0]:.2f} rad/s  "
              f"ddq_max={eff['ddq_max'][0]:.2f} rad/s^2  "
              f"jerk_max={eff['jerk_max'][0]:.2f} rad/s^3")
        print("(ddq_max and jerk_max have no measured basis -- they are conservative "
              "knobs; the binding gate is the torque column)")
        print(J.format_report(t, q, dq, ddq, model=model, **caps))

        if args.compare_linear:
            lin = J.linear_blend(q[0], q[-1], args.dt, duration=3.0)
            gp = J.peaks(dq, ddq, args.dt)
            lp = J.peaks(lin[2], lin[3], args.dt)
            print("\npineapple_arm.py's linear blend over the same move (3.0 s):")
            print(f"{'joint':<20}{'|ddq| quintic':>16}{'|ddq| linear':>16}"
                  f"{'|jerk| quintic':>17}{'|jerk| linear':>16}")
            for j in range(NUM_MOTORS):
                print(f"{arm_ik.IK_MODEL.names[j+1]:<20}"
                      f"{gp['ddq'][j]:14.2f}  {lp['ddq'][j]:14.2f}  "
                      f"{gp['jerk'][j]:15.1f}  {lp['jerk'][j]:14.1f}")
            _plot(t, q, dq, ddq, args.plot, linear=lin)
            return 0

        if args.dry_run:
            _plot(t, q, dq, ddq, args.plot)
            return 0

        if args.sim:
            S = np.ones(NUM_MOTORS) if args.no_gain_inflation else GAIN_INFLATION
            log = simulate(t, q, dq, model=model, mass_scale=args.mass_scale,
                           gain_inflation=S)
            print(f"\n--- closed loop, plant mass x{args.mass_scale}, "
                  f"effective kp = {np.round(KP * S, 1).tolist()} ---")
            report_tracking(log)
            _plot(t, q, dq, ddq, args.plot)
            return 0

        log = controller.follow(t, q, dq)
        report_tracking(log)
        return 0 if log["complete"] else 2

    except ee_traj.ReferenceError as e:
        print(f"[smooth] {e}")
        return 2
    except KeyboardInterrupt:
        print("\n[smooth] interrupted")
        return 130
    except Exception:
        traceback.print_exc()
        return 1
    finally:
        # Guaranteed: whatever happened above, the arm is ramped down before we exit.
        if controller is not None:
            try:
                controller.safe_return()
            except Exception:
                traceback.print_exc()


if __name__ == "__main__":
    sys.exit(main())
