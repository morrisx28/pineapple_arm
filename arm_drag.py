"""Gravity-compensated hand-guiding and drift measurement.

Drag mode disables position stiffness and supports the arm with calibrated feedforward:

    tau_ff[j] = g_scale[j] * trim[j] * arm_ff.gravity_torque(q)[j]

Untrusted or unobservable fitted factors fall back to the URDF model. Commanded damping
is divided by the measured gain-path scale. The soft fall catch requires persistent
motion in the gravity-losing direction; the independent hard watchdog is always active.
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
import threading
import time

import numpy as np

import arm_ff
import ee_traj as T
import verify_gravity as VG
from arm_tvlqr import check_gains

NUM_MOTORS = 6
DT = 0.005  # 200 Hz, matches pineapple_arm.py
JOINT_NAMES = VG.JOINT_NAMES
JOINT_LOW, JOINT_HIGH = T.JOINT_LOW, T.JOINT_HIGH
# Home rests on two exact limits; an inset would create a large PD command at rest.

# Hold gains are the hardware-proven raw gains; only drag damping is scale-corrected.
KP_HOLD = np.array([20.0, 40.0, 40.0, 20.0, 20.0, 20.0])
KD_HOLD = np.array([0.5, 1.0, 1.0, 0.5, 0.5, 0.5])
# Desired damping before gain-path correction; keep it low for backdrivability.
KD_DRAG = np.array([0.2, 0.2, 0.2, 0.1, 0.1, 0.1])

# Live safety limits (same basis as verify_gravity.py / arm_tvlqr.py).
MOTOR_TAU_LIMIT = arm_ff.TAU_LIMIT                  # [27,27,27,7,7,7]
SAFETY_TAU = 0.90 * MOTOR_TAU_LIMIT
DQ_LIMIT = np.full(NUM_MOTORS, 10.0)                 # rad/s hard abort ceiling

# The soft fall catch restores position hold only after persistent gravity-losing motion.
DQ_CATCH = 3.5          # rad/s sustained -> runaway
CATCH_HOLD_S = 0.20     # how long a trigger must persist before it counts
# Monotonic drift from the last rest pose; deliberately above normal hand motion because
# kinematics cannot distinguish slow intentional motion from sag.
MAX_DRIFT = 1.5
REST_DQ = 0.05          # rad/s below which a joint counts as parked (re-anchors creep)
CREEP_DQ = 0.10         # rad/s: still sliding, for the creep trigger

# Soft joint-limit barrier: an inward spring over the last LIMIT_MARGIN rad of travel,
# so hand-dragging cannot slam a hard stop while kp=0.
LIMIT_MARGIN = 0.15
K_BARRIER = KP_HOLD.copy()                          # Nm/rad -> 6 Nm at full margin on j1
D_BARRIER = KD_HOLD.copy()                          # Nm*s/rad, only when moving outward

# Plausibility bands for the identified factors. Anything outside is a fit artifact,
# not a physical property, so it must not reach the motors.
G_SCALE_BAND = (0.5, 3.0)
KD_DIV_BAND = (0.2, 25.0)
# |centered corr(sag, g)| below this means S is a mean ratio over a nearly constant
# sag, not a verified proportional relation -- see verify_gravity's `lin` column.
LIN_S_MIN = 0.5

HANDSOFF_NM = 0.3       # |tau_est - tau_ff| below this counts as nobody touching it
DRIFT_TAU_S = 0.5       # EWMA time constant for the live drift readout [s]
DRIFT_OK = 0.05         # rad/s: hands-off drift below this = compensation is good
RESID_OK = 0.3          # Nm: residual holding torque below this = good
G_FLOOR_NM = 0.2        # ignore samples whose |tau_ff| is below this in the trim fit
MAX_SAMPLES = 200 * 60 * 10   # 10 min at 200 Hz, then recording stops itself

STATES = ("hold", "engaging", "float", "caught", "homing")
STATE_CODE = {s: i for i, s in enumerate(STATES)}

LOG_SCHEMA_VERSION = 1
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


# Identified-model resolution

def find_latest_log(data_dir=DATA_DIR):
    """Newest real gravity log in ``data/``, or None. Synthetic/derived files out."""
    cands = [p for p in glob.glob(os.path.join(data_dir, "grav_*.npz"))
             if "selftest" not in os.path.basename(p)
             and "scaletest" not in os.path.basename(p)]
    if not cands:
        return None
    return max(cands, key=os.path.getmtime)


def resolve_gravity_scales(npz_path=None, use_ident=True, raw_kd=False):
    """Per-joint gravity scale + kd divisor from an identified gravity log.

    Returns ``dict(g_scale, kd_div, g_why, kd_why, source, rows)``. Fail-safe by
    construction: every path that cannot justify a number leaves it at 1.0, i.e. the
    plain URDF model and the raw commanded gain.
    """
    g_scale = np.ones(NUM_MOTORS)
    kd_div = np.ones(NUM_MOTORS)
    g_why = ["URDF (--no-ident)"] * NUM_MOTORS
    kd_why = ["raw"] * NUM_MOTORS

    if not use_ident:
        return dict(g_scale=g_scale, kd_div=kd_div, g_why=g_why, kd_why=kd_why,
                    source="none (--no-ident): pure URDF", rows=[])

    path = npz_path or find_latest_log()
    if path is None:
        return dict(g_scale=g_scale, kd_div=kd_div,
                    g_why=["URDF (no log found)"] * NUM_MOTORS, kd_why=kd_why,
                    source="none: no data/grav_*.npz found -> pure URDF", rows=[])
    log = VG.load_log(path)
    names = [str(x) for x in np.asarray(log["joint_names"]).ravel()]
    if names != JOINT_NAMES:
        raise ValueError(f"{path}: joint order mismatch {names}")
    if "tau_meas" not in log or len(np.asarray(log["tau_meas"])) == 0:
        raise ValueError(f"{path}: contains no poses")

    fit = VG.compute_joint_scales(log)
    by_j = {r["j"]: r for r in fit["rows"]}
    for j in range(NUM_MOTORS):
        r = by_j.get(j)
        if r is None:
            g_why[j] = f"URDF: not observable (gravity span {fit['gmax'][j]:.2f} Nm)"
            kd_why[j] = "raw: not observable"
            continue
        if not r["good"]:
            g_why[j] = "URDF: LOW CONFIDENCE fit"
            kd_why[j] = "raw: LOW CONFIDENCE fit"
            continue
        # --- torque channel -> gravity scale ---
        if not np.isfinite(r["N_tau"]):
            g_why[j] = "URDF: N_tau not finite"
        elif not (G_SCALE_BAND[0] <= r["N_tau"] <= G_SCALE_BAND[1]):
            g_why[j] = (f"URDF: N_tau={r['N_tau']:.2f} outside "
                        f"[{G_SCALE_BAND[0]}, {G_SCALE_BAND[1]}]")
        else:
            g_scale[j] = float(r["N_tau"])
            g_why[j] = f"fitted N_tau (R2 {r['r2_use']:.3f}, n={r['n_t']})"
        # --- stiffness channel -> kd divisor ---
        if raw_kd:
            kd_why[j] = "raw (--raw-kd)"
        elif not np.isfinite(r["S"]):
            kd_why[j] = "raw: S not finite"
        elif not (np.isfinite(r["lin_s"]) and abs(r["lin_s"]) >= LIN_S_MIN):
            kd_why[j] = f"raw: S unverified (lin={r['lin_s']:.2f}, sag ~constant)"
        elif not (KD_DIV_BAND[0] <= r["S"] <= KD_DIV_BAND[1]):
            kd_why[j] = (f"raw: S={r['S']:.2f} outside "
                         f"[{KD_DIV_BAND[0]}, {KD_DIV_BAND[1]}]")
        else:
            kd_div[j] = float(r["S"])
            kd_why[j] = f"S={r['S']:.2f} (~{r['Nint']}*pi)" if r["Nint"] else f"S={r['S']:.2f}"

    return dict(g_scale=g_scale, kd_div=kd_div, g_why=g_why, kd_why=kd_why,
                source=os.path.basename(path), rows=fit["rows"])


def print_model(res, kd_drag=KD_DRAG):
    """Show exactly what will be commanded, and why, before anything moves."""
    print(f"\n=== resolved gravity model  (source: {res['source']}) ===")
    print(f"  {'joint':18s} {'g_scale':>8s} | {'kd_div':>7s} {'kd_sent':>8s} | why")
    for j, nm in enumerate(JOINT_NAMES):
        kd_sent = kd_drag[j] / res["kd_div"][j]
        print(f"  {nm:18s} {res['g_scale'][j]:8.3f} | {res['kd_div'][j]:7.2f} "
              f"{kd_sent:8.4f} | {res['g_why'][j]}")
        if res["kd_why"][j] not in ("raw",):
            print(f"  {'':18s} {'':8s} | {'':7s} {'':8s} | kd: {res['kd_why'][j]}")
    n_fit = int(np.sum(res["g_scale"] != 1.0))
    print(f"  -> {n_fit}/{NUM_MOTORS} joints use an identified gravity scale; "
          f"the rest fall back to URDF.")


# Controller

def limit_barrier(q, dq, margin=LIMIT_MARGIN, k=K_BARRIER, d=D_BARRIER):
    """Cushion joint stops without applying torque to a stationary legal pose.

    The spring acts only beyond the true limit. Damping ramps in over ``margin`` only
    during outward motion, so home positions lying exactly on a limit remain undisturbed.
    """
    q = np.asarray(q, float); dq = np.asarray(dq, float)
    # Spring from actual penetration, not an inset band.
    tau = k * (np.maximum(0.0, JOINT_LOW - q) - np.maximum(0.0, q - JOINT_HIGH))
    # Damping depth ramps from zero at the band edge to one at the stop.
    depth_lo = np.clip((JOINT_LOW + margin - q) / margin, 0.0, 1.0)
    depth_hi = np.clip((q - JOINT_HIGH + margin) / margin, 0.0, 1.0)
    moving_out_lo = (depth_lo > 0) & (dq < 0)
    moving_out_hi = (depth_hi > 0) & (dq > 0)
    d_eff = d * np.where(moving_out_lo, depth_lo, np.where(moving_out_hi, depth_hi, 0.0))
    return tau - d_eff * dq


class DragController:
    """DDS controller that floats the arm on feedforward gravity so it can be dragged.

    The REPL requests transitions while a daemon runs the 200 Hz loop. Gravity remains
    active in every state, so engaging never removes PD and feedforward simultaneously.
    """

    def __init__(self, g_scale, kd_div, kp_hold=KP_HOLD, kd_hold=KD_HOLD,
                 kd_drag=KD_DRAG, dq_catch=DQ_CATCH, max_drift=MAX_DRIFT,
                 catch_on=True, engage_s=3.0, trip_samples=3, state_timeout=0.1, dt=DT):
        self._init_state(g_scale, kd_div, kp_hold=kp_hold, kd_hold=kd_hold,
                         kd_drag=kd_drag, dq_catch=dq_catch, max_drift=max_drift,
                         catch_on=catch_on, engage_s=engage_s,
                         trip_samples=trip_samples, state_timeout=state_timeout, dt=dt)
        self._init_dds()

    # Keep state construction separate so self-tests do not require DDS.
    def _init_state(self, g_scale, kd_div, kp_hold=KP_HOLD, kd_hold=KD_HOLD,
                    kd_drag=KD_DRAG, dq_catch=DQ_CATCH, max_drift=MAX_DRIFT,
                    catch_on=True, engage_s=3.0, trip_samples=3, state_timeout=0.1,
                    dt=DT):
        self.g_scale = check_gains(g_scale, "g_scale")
        self.kd_div = check_gains(kd_div, "kd_div")
        if np.any(self.kd_div <= 0):
            raise ValueError(f"kd_div must be positive, got {self.kd_div}")
        self.kp_hold = check_gains(kp_hold, "kp_hold")
        self.kd_hold = check_gains(kd_hold, "kd_hold")
        self.kd_drag = check_gains(kd_drag, "kd_drag")
        for label, v in (("dq_catch", dq_catch), ("max_drift", max_drift),
                         ("engage_s", engage_s), ("state_timeout", state_timeout),
                         ("dt", dt)):
            if not np.isfinite(v) or v <= 0:
                raise ValueError(f"{label} must be finite and positive, got {v}")
        self.dq_catch = float(dq_catch)
        self.max_drift = float(max_drift)
        self.catch_on = bool(catch_on)
        self.engage_s = float(engage_s)
        self.trip_samples = max(1, int(trip_samples))
        self.state_timeout = float(state_timeout)
        self.dt = float(dt)
        # A trigger must persist this many ticks. Unlike the watchdog's 3-sample
        # debounce, this is a real time window: a drag is bounded, a fall is not.
        self.catch_samples = max(1, int(round(CATCH_HOLD_S / self.dt)))

        self.trim = np.ones(NUM_MOTORS)
        self.use_friction = False
        self.monitor = False

        self._lock = threading.Lock()          # guards the measured state
        self._cmd_lock = threading.Lock()      # guards trim / requests / recording
        self.qpos = np.zeros(NUM_MOTORS)
        self.qvel = np.zeros(NUM_MOTORS)
        self.qtau = np.zeros(NUM_MOTORS)
        self.t_arrival = 0.0
        self.low_state = None

        self.state = "hold"
        self.kp_blend = 1.0
        self.hold_q = np.zeros(NUM_MOTORS)
        self.home_from = np.zeros(NUM_MOTORS)
        self._request = None
        self._phase = 0                        # tick counter inside engaging/homing
        self._trip = np.zeros(NUM_MOTORS, dtype=int)   # watchdog debounce
        self._catch_trip = np.zeros(NUM_MOTORS, dtype=int)   # runaway / creep
        self._out_trip = np.zeros(NUM_MOTORS, dtype=int)     # past a hard limit
        # Where each joint was last parked while floating. `creep` measures slide from
        # HERE, not from the engage pose, so deliberately repositioning the arm
        # re-anchors instead of leaving the joint permanently over the threshold.
        self.float_q0 = np.zeros(NUM_MOTORS)
        self.drift = np.zeros(NUM_MOTORS)      # EWMA of dq, for the live readout
        self.abort_reason = ""
        self._homed = threading.Event()        # set when a homing ramp completes
        self.is_running = False
        self.thread = None
        self._t0 = 0.0
        self._last_mon = 0.0
        self._rec = None                       # list of sample tuples while recording
        self._rec_name = ""

    def _init_dds(self):
        from unitree_sdk2py.core.channel import ChannelPublisher, ChannelSubscriber
        from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowCmd_
        from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowCmd_, LowState_
        from unitree_sdk2py.utils.crc import CRC

        self.low_cmd = unitree_go_msg_dds__LowCmd_()
        self.crc = CRC()
        self.low_cmd.head[0] = 0xFE
        self.low_cmd.head[1] = 0xEF
        self.low_cmd.level_flag = 0xFF
        self.low_cmd.gpio = 0
        for i in range(NUM_MOTORS):
            self.low_cmd.motor_cmd[i].mode = 0x01
            self.low_cmd.motor_cmd[i].q = 0.0
            self.low_cmd.motor_cmd[i].kp = 0.0
            self.low_cmd.motor_cmd[i].dq = 0.0
            self.low_cmd.motor_cmd[i].kd = 0.0
            self.low_cmd.motor_cmd[i].tau = 0.0
        self.pub = ChannelPublisher("rt/lowcmd", LowCmd_)
        self.pub.Init()
        self.sub = ChannelSubscriber("rt/lowstate", LowState_)
        self.sub.Init(self._on_low_state, 10)

    # State I/O
    def _on_low_state(self, msg):
        q = np.empty(NUM_MOTORS); v = np.empty(NUM_MOTORS); tau = np.empty(NUM_MOTORS)
        for i in range(NUM_MOTORS):
            q[i] = msg.motor_state[i].q
            v[i] = msg.motor_state[i].dq
            tau[i] = msg.motor_state[i].tau_est
        with self._lock:
            self.low_state = msg
            self.t_arrival = time.perf_counter()
            self.qpos[:] = q; self.qvel[:] = v; self.qtau[:] = tau

    def snapshot(self):
        with self._lock:
            return (self.qpos.copy(), self.qvel.copy(), self.qtau.copy(),
                    float(self.t_arrival), self.low_state is not None)

    def has_state(self):
        with self._lock:
            return self.low_state is not None

    def wait_for_state(self, timeout=5.0):
        t0 = time.perf_counter()
        while not self.has_state():
            if time.perf_counter() - t0 > timeout:
                raise TimeoutError("No rt/lowstate received; is the arm up?")
            time.sleep(0.01)

    @staticmethod
    def _prepare_cmd(q_des, kp, kd, tau_ff):
        """Validate and clip one command -> the exact values that reach the motors.

        Shared by the real ``_write`` and the selftest's stub, so the offline tests
        exercise the same guard the hardware path does.
        """
        q_des = np.asarray(q_des, float)
        kp = np.asarray(kp, float); kd = np.asarray(kd, float)
        tau_ff = np.asarray(tau_ff, float)
        # np.clip does NOT remove NaN/Inf, and nothing downstream inspects these.
        for label, a in (("q", q_des), ("kp", kp), ("kd", kd), ("tau", tau_ff)):
            if a.shape != (NUM_MOTORS,) or not np.all(np.isfinite(a)):
                raise ValueError(f"refusing to publish non-finite {label}: {a}")
        if np.any(kp < 0) or np.any(kd < 0):
            raise ValueError(f"refusing to publish negative gains: kp={kp} kd={kd}")
        return (np.clip(q_des, JOINT_LOW, JOINT_HIGH),
                kp, kd, np.clip(tau_ff, -MOTOR_TAU_LIMIT, MOTOR_TAU_LIMIT))

    def _write(self, q_des, kp, kd, tau_ff):
        """Publish one low-cmd. Refuses non-finite values rather than sending them."""
        q_des, kp, kd, tau_ff = self._prepare_cmd(q_des, kp, kd, tau_ff)
        for i in range(NUM_MOTORS):
            self.low_cmd.motor_cmd[i].q = float(q_des[i])
            self.low_cmd.motor_cmd[i].dq = 0.0
            self.low_cmd.motor_cmd[i].kp = float(kp[i])
            self.low_cmd.motor_cmd[i].kd = float(kd[i])
            self.low_cmd.motor_cmd[i].tau = float(tau_ff[i])
        self.low_cmd.crc = self.crc.Crc(self.low_cmd)
        self.pub.Write(self.low_cmd)

    # Safety
    def _violation_str(self, j, qk, dqk, tauk):
        nm = JOINT_NAMES[j]
        if abs(tauk[j]) > SAFETY_TAU[j]:
            return f"{nm} torque {abs(tauk[j]):.1f} > {SAFETY_TAU[j]:.1f} Nm"
        if abs(dqk[j]) > DQ_LIMIT[j]:
            return f"{nm} velocity {abs(dqk[j]):.1f} > {DQ_LIMIT[j]:.1f} rad/s"
        return f"{nm} position {qk[j]:.2f} outside range"

    def _watchdog(self, qk, dqk, tauk, arrival, valid):
        """Debounced hard abort (same structure as verify_gravity's)."""
        if not valid or time.perf_counter() - arrival > self.state_timeout:
            raise RuntimeError(f"stale/missing state (age > {self.state_timeout}s)")
        soft = ((np.abs(tauk) > SAFETY_TAU) | (np.abs(dqk) > DQ_LIMIT)
                | (qk < JOINT_LOW - 0.05) | (qk > JOINT_HIGH + 0.05))
        # Torque is deliberately NOT on the instant-abort path. It cannot exceed the
        # motor limit, so a merely SATURATED joint always reads above any "severe"
        # torque threshold -- which turned one transient spike into a killed session.
        # 3 consecutive samples (15 ms) at the limit still aborts, which is ample
        # protection. Velocity and position keep their instant tier: those are runaways.
        severe = ((np.abs(dqk) > 1.10 * DQ_LIMIT)
                  | (qk < JOINT_LOW - 0.10) | (qk > JOINT_HIGH + 0.10))
        self._trip = np.where(soft, self._trip + 1, 0)
        if np.any(severe):
            j = int(np.flatnonzero(severe)[0])
            raise RuntimeError("severe " + self._violation_str(j, qk, dqk, tauk))
        tripped = np.flatnonzero(self._trip >= self.trip_samples)
        if tripped.size:
            j = int(tripped[0])
            raise RuntimeError(self._violation_str(j, qk, dqk, tauk)
                               + f" ({self.trip_samples} consec)")

    def _catch_check(self, qk, dqk, tau_g):
        """Reason to abandon float and restore kp, or "".

        Speed alone cannot distinguish dragging from falling. Ambiguous triggers require
        persistent gravity-losing motion:

          runaway: fast motion in the gravity direction
          creep: monotonic displacement from the last rest pose
          out: outward motion beyond a hard limit

        ``out`` is unambiguous, so it uses the shorter watchdog debounce without gravity
        gating. Gravity-losing uses ``dq * sign(tau_g) < 0``.
        """
        if not self.catch_on:
            self._catch_trip[:] = 0
            self._out_trip[:] = 0
            return ""
        losing = dqk * np.sign(tau_g) < 0        # moving the way gravity pulls
        moved = qk - self.float_q0
        # Re-anchor at rest or reversal so only monotonic drift accumulates.
        reanchor = (np.abs(dqk) < REST_DQ) | (dqk * np.sign(moved) < 0)
        self.float_q0 = np.where(reanchor, qk, self.float_q0)
        moved = qk - self.float_q0

        runaway = (np.abs(dqk) > self.dq_catch) & losing
        creep = ((np.abs(moved) > self.max_drift) & (np.abs(dqk) > CREEP_DQ)
                 & (moved * np.sign(tau_g) < 0))
        out = ((qk < JOINT_LOW) & (dqk < 0)) | ((qk > JOINT_HIGH) & (dqk > 0))

        self._catch_trip = np.where(runaway | creep, self._catch_trip + 1, 0)
        self._out_trip = np.where(out, self._out_trip + 1, 0)
        tripped = np.flatnonzero((self._catch_trip >= self.catch_samples)
                                 | (self._out_trip >= self.trip_samples))
        if not tripped.size:
            return ""
        j = int(tripped[0])
        held = self.catch_samples * self.dt
        if runaway[j] and self._catch_trip[j] >= self.catch_samples:
            return (f"{JOINT_NAMES[j]} dropping at {dqk[j]:+.2f} rad/s for {held:.2f}s "
                    f"(dq_catch {self.dq_catch}, gravity pulls "
                    f"{'+' if tau_g[j] > 0 else '-'})")
        if creep[j] and self._catch_trip[j] >= self.catch_samples:
            return (f"{JOINT_NAMES[j]} sagging: slid {moved[j]:+.2f} rad from rest "
                    f"(max_drift {self.max_drift}) still at {dqk[j]:+.2f} rad/s")
        return (f"{JOINT_NAMES[j]} past its limit at q={qk[j]:+.3f} rad, "
                f"still moving out at {dqk[j]:+.2f} rad/s")

    # State machine
    def request(self, what):
        """Ask the control thread for a transition ("engage"/"hold"/"home")."""
        if what not in ("engage", "hold", "home"):
            raise ValueError(f"unknown request {what!r}")
        with self._cmd_lock:
            self._request = what

    def reload_model(self, res):
        """Swap in a freshly resolved gravity model. -> (old_g_scale, old_trim).

        Raises RuntimeError when it would invalidate something rather than doing it
        quietly: a session log carries ONE ``g_scale``, so a model change mid-recording
        would leave :func:`session_metrics` describing neither half.

        Floating is handled by holding FIRST (the caller requests it): a g_scale step is
        a torque step, and with kd/S there is almost nothing to absorb it.
        """
        g_scale = check_gains(res["g_scale"], "g_scale")
        kd_div = check_gains(res["kd_div"], "kd_div")
        if np.any(kd_div <= 0):
            raise ValueError(f"kd_div must be positive, got {kd_div}")
        with self._cmd_lock:
            if self._rec is not None:
                raise RuntimeError("a recording is in progress; `stop` it first "
                                   "(one session must describe one model)")
            old_g, old_trim = self.g_scale.copy(), self.trim.copy()
            self.g_scale = g_scale
            self.kd_div = kd_div
            # The new fit already contains whatever a manual trim was compensating for;
            # keeping both would double-correct.
            self.trim = np.ones(NUM_MOTORS)
        return old_g, old_trim

    def set_catch(self, dq_catch=None, max_drift=None, on=None):
        """Retune the fall-catch live, so a catch loop on the robot needs no restart."""
        with self._cmd_lock:
            if dq_catch is not None:
                if not np.isfinite(dq_catch) or dq_catch <= 0:
                    raise ValueError(f"dq_catch must be positive, got {dq_catch}")
                self.dq_catch = float(dq_catch)
            if max_drift is not None:
                if not np.isfinite(max_drift) or max_drift <= 0:
                    raise ValueError(f"max_drift must be positive, got {max_drift}")
                self.max_drift = float(max_drift)
            if on is not None:
                self.catch_on = bool(on)
            self._catch_trip[:] = 0
            self._out_trip[:] = 0

    def _catch(self, reason, qk):
        self.state = "caught"
        self.kp_blend = 1.0
        self.hold_q = qk.copy()
        self._catch_trip[:] = 0
        self._out_trip[:] = 0
        print(f"\n[drag] CATCH: {reason}\n"
              "       -> position hold restored. Adjust the model (`trim` / `model`), "
              "loosen the catch (`catch dq ...`), or `engage` again.")

    def _apply_request(self, qk):
        with self._cmd_lock:
            req, self._request = self._request, None
        if req is None:
            return
        if req == "engage":
            self.state = "engaging"
            self.hold_q = qk.copy()      # the pose the fading PD holds during the ramp
            self.float_q0 = qk.copy()    # creep reference; the catch runs during the ramp
            self._phase = 0
            self._catch_trip[:] = 0
            self._out_trip[:] = 0
            print(f"[drag] engaging: ramping kp -> 0 over {self.engage_s:.1f}s")
        elif req == "hold":
            self.state = "hold"
            self.kp_blend = 1.0
            self.hold_q = qk.copy()
            print("[drag] holding position (kp restored)")
        elif req == "home":
            self.state = "homing"
            self.home_from = qk.copy()
            self.kp_blend = 1.0
            self._phase = 0
            self._homed.clear()
            print("[drag] homing to zero pose")

    def _plan(self, qk, dqk, tau_g, kd_div):
        """-> (q_des, kp, kd) for this tick, advancing the state machine."""
        self._apply_request(qk)
        steps = max(1, int(self.engage_s / self.dt))

        if self.state == "engaging":
            self._phase += 1
            self.kp_blend = max(0.0, 1.0 - self._phase / steps)
            # hold_q stays FIXED at the pose captured on `engage`. It must NOT follow
            # qk: with q_des == q_meas the PD error is zero, so kp does nothing and the
            # ramp gives no fading authority at all -- the arm is already floating from
            # the first tick, which is exactly what the ramp exists to avoid.
            # The catch runs here too, so a bad model degrades to a hold rather than
            # letting the watchdog abort the whole session.
            reason = self._catch_check(qk, dqk, tau_g)
            if reason:
                self._catch(reason, qk)
            elif self.kp_blend <= 0.0:
                self.state = "float"
                self.float_q0 = qk.copy()
                self._catch_trip[:] = 0
                self._out_trip[:] = 0
                print("[drag] FLOAT: kp=0, arm is held by gravity feedforward only. "
                      "Drag it.")
        elif self.state == "float":
            self.kp_blend = 0.0
            self.hold_q = qk.copy()   # so a catch engages in place
            reason = self._catch_check(qk, dqk, tau_g)
            if reason:
                self._catch(reason, qk)
        elif self.state == "homing":
            self._phase += 1
            phase = min(1.0, self._phase / steps)
            self.kp_blend = 1.0
            self.hold_q = self.home_from * (1.0 - phase)
            if phase >= 1.0:
                self.state = "hold"
                self._homed.set()
                print("[drag] home reached; holding zero pose")
        else:  # hold / caught
            self.kp_blend = 1.0

        kp = self.kp_blend * self.kp_hold
        kd = (self.kp_blend * self.kd_hold
              + (1.0 - self.kp_blend) * (self.kd_drag / kd_div))
        return self.hold_q.copy(), kp, kd

    def _tick(self):
        """One control period: read, plan, publish, record. Raises to abort."""
        qk, dqk, tauk, arrival, valid = self.snapshot()
        self._watchdog(qk, dqk, tauk, arrival, valid)

        # The gravity torque is needed BEFORE _plan, because the fall-catch tests the
        # motion against the direction gravity pulls. Take one consistent snapshot of
        # the model under the lock: `model <npz>` can swap g_scale/kd_div from the REPL
        # thread, and a 6-element numpy assignment is not atomic.
        with self._cmd_lock:
            trim = self.trim.copy()
            fric = self.use_friction
            g_scale = self.g_scale.copy()
            kd_div = self.kd_div.copy()
        g_urdf = arm_ff.gravity_torque(qk)
        tau_g = g_scale * trim * g_urdf

        q_des, kp, kd = self._plan(qk, dqk, tau_g, kd_div)
        tau = tau_g.copy()
        if fric:
            tau += arm_ff.friction_torque(dqk)
        tau += limit_barrier(qk, dqk)   # the barrier is a commanded torque too
        lim = arm_ff.FF_CLAMP_FRAC * MOTOR_TAU_LIMIT
        tau = np.clip(tau, -lim, lim)
        self._write(q_des, kp, kd, tau)

        a = self.dt / DRIFT_TAU_S
        self.drift = (1.0 - a) * self.drift + a * dqk
        t = time.perf_counter() - self._t0
        self._record(t, qk, dqk, tauk, tau, g_urdf, tau_g, kp, kd)
        if self.monitor:
            self._readout(t, qk, dqk, tauk, tau)
        return qk, dqk, tauk, tau

    def run(self):
        """200 Hz control loop. On abort: catch, then home, then stop."""
        self._t0 = time.perf_counter()
        while self.is_running:
            step_start = time.perf_counter()
            try:
                self._tick()
            except (RuntimeError, ValueError) as e:
                self.abort_reason = str(e)
                print(f"\n[drag] ABORT: {self.abort_reason}")
                self.is_running = False
                break
            sleep = self.dt - (time.perf_counter() - step_start)
            if sleep > 0:
                time.sleep(sleep)

    def start(self):
        self.is_running = True
        qk = self.snapshot()[0]
        self.hold_q = qk.copy()
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.thread.start()

    def stop(self):
        self.is_running = False
        if self.thread is not None:
            self.thread.join(timeout=2.0)
            self.thread = None

    def go_home(self, timeout=8.0):
        """Request homing and wait for it (best-effort; never raises).

        Waits for the ramp to actually START before waiting for it to finish -- the
        state is already "hold" when the request is made, so checking only for "hold"
        would return immediately without the arm having moved.
        """
        if not self.is_running:
            return
        self._homed.clear()
        self.request("home")
        t0 = time.perf_counter()
        while self.is_running and time.perf_counter() - t0 < timeout:
            if self._homed.wait(0.05):
                return

    def safe_return(self, dt=None):
        """Bring the arm down to zero on a dead loop, watchdog-free (best-effort)."""
        dt = self.dt if dt is None else dt
        try:
            start = self.snapshot()[0]
            steps = max(1, int(3.0 / dt))
            for k in range(steps):
                step_start = time.perf_counter()
                qk, dqk = self.snapshot()[:2]
                phase = (k + 1) / steps
                self._write(start * (1.0 - phase), self.kp_hold, self.kd_hold,
                            self.g_scale * arm_ff.gravity_torque(qk)
                            + limit_barrier(qk, dqk))
                sleep = dt - (time.perf_counter() - step_start)
                if sleep > 0:
                    time.sleep(sleep)
        except Exception as e:  # recovery must never raise
            print(f"[drag] safe_return best-effort error: {e}")

    # Live readout
    def _readout(self, t, qk, dqk, tauk, tau_ff, period=0.5):
        now = time.perf_counter()
        if now - self._last_mon < period:
            return
        self._last_mon = now
        resid = tauk - tau_ff
        j = int(np.argmax(np.abs(self.drift)))
        print(f"\n[drag] {self.state:9s} t={t:6.1f}s  worst drift "
              f"{self.drift[j]:+.3f} rad/s on {JOINT_NAMES[j]}"
              + ("  (recording)" if self._rec is not None else ""))
        print(f"  {'joint':18s} {'q':>7s} {'dq':>7s} {'tau_ff':>7s} "
              f"{'tau_est':>8s} {'resid':>7s} {'drift':>7s}")
        for i, nm in enumerate(JOINT_NAMES):
            print(f"  {nm:18s} {qk[i]:7.3f} {dqk[i]:7.3f} {tau_ff[i]:7.2f} "
                  f"{tauk[i]:8.2f} {resid[i]:7.2f} {self.drift[i]:7.3f}")

    # Recording
    def rec_start(self, name=""):
        with self._cmd_lock:
            self._rec = []
            self._rec_name = name
        print("[drag] recording started")

    def _record(self, t, qk, dqk, tauk, tau_ff, g_urdf, tau_g, kp, kd):
        with self._cmd_lock:
            if self._rec is None:
                return
            if len(self._rec) >= MAX_SAMPLES:
                if len(self._rec) == MAX_SAMPLES:
                    print(f"\n[drag] recording cap ({MAX_SAMPLES} samples) reached; "
                          "stopping capture. Use `stop` to save.")
                    self._rec.append(None)  # sentinel so this prints once
                return
            self._rec.append((t, qk.copy(), dqk.copy(), tauk.copy(), tau_ff.copy(),
                              g_urdf.copy(), tau_g.copy(), kp.copy(), kd.copy(),
                              STATE_CODE[self.state]))

    def rec_stop(self):
        """Finish a capture -> log dict, or None if nothing was recorded."""
        with self._cmd_lock:
            rec, name = self._rec, self._rec_name
            self._rec, self._rec_name = None, ""
        if not rec:
            return None
        rows = [r for r in rec if r is not None]
        if not rows:
            return None
        n = len(rows)
        log = dict(
            t=np.array([r[0] for r in rows]),
            q=np.array([r[1] for r in rows]),
            dq=np.array([r[2] for r in rows]),
            tau_est=np.array([r[3] for r in rows]),
            tau_ff=np.array([r[4] for r in rows]),
            g_urdf=np.array([r[5] for r in rows]),
            tau_g=np.array([r[6] for r in rows]),
            kp_out=np.array([r[7] for r in rows]),
            kd_out=np.array([r[8] for r in rows]),
            state=np.array([r[9] for r in rows]),
            joint_names=np.array(JOINT_NAMES),
            g_scale=self.g_scale.copy(), kd_div=self.kd_div.copy(),
            trim=self.trim.copy(), kp_hold=self.kp_hold.copy(),
            kd_drag=self.kd_drag.copy(),
            states=np.array(STATES), friction=np.array(str(self.use_friction)),
            schema_version=np.array(LOG_SCHEMA_VERSION),
            name=np.array(name),
        )
        print(f"[drag] recording stopped ({n} samples)")
        return log


# Evaluation

def session_metrics(log, handsoff_nm=HANDSOFF_NM, g_floor=G_FLOOR_NM):
    """Return per-joint compensation quality from hands-off floating samples.

    Encoder drift is independent of the torque scale being evaluated. ``drift_g`` projects
    velocity along gravity; a negative value indicates under-compensation.
    """
    tau_est = np.asarray(log["tau_est"], float)
    tau_ff = np.asarray(log["tau_ff"], float)
    dq = np.asarray(log["dq"], float)
    state = np.asarray(log["state"]).ravel().astype(int)
    # Use an untrimmed, session-constant base so the fitted slope remains a valid trim.
    g_scale = (np.asarray(log["g_scale"], float) if "g_scale" in log
               else np.ones(NUM_MOTORS))
    base = np.asarray(log["g_urdf"], float) * g_scale[None, :]

    resid = tau_est - tau_ff
    floating = state == STATE_CODE["float"]
    use = floating[:, None] & (np.abs(resid) < handsoff_nm)

    out = dict(n_float=int(floating.sum()), n_total=len(state),
               n_use=np.zeros(NUM_MOTORS, dtype=int),
               drift=np.full(NUM_MOTORS, np.nan),
               drift_g=np.full(NUM_MOTORS, np.nan),
               resid=np.full(NUM_MOTORS, np.nan),
               effort=np.full(NUM_MOTORS, np.nan),
               sugg_trim=np.full(NUM_MOTORS, np.nan),
               verdict=[""] * NUM_MOTORS)
    for j in range(NUM_MOTORS):
        u = use[:, j]
        out["n_use"][j] = int(u.sum())
        forced = floating & ~u
        if forced.any():
            out["effort"][j] = float(np.abs(resid[forced, j]).max())
        if not u.any():
            out["verdict"][j] = "no hands-off float data"
            continue
        out["drift"][j] = float(np.mean(dq[u, j]))
        out["resid"][j] = float(np.mean(resid[u, j]))
        # Do not infer a compensation verdict where gravity is negligible.
        gj = base[u, j]
        big = np.abs(gj) > g_floor
        if big.sum() >= 2:
            out["drift_g"][j] = float(np.mean(dq[u, j][big] * np.sign(gj[big])))
            k, _ = VG._ls_through_origin(gj[big], tau_est[u, j][big])
            if np.isfinite(k):
                out["sugg_trim"][j] = float(k)
            d = out["drift_g"][j]
            if abs(d) <= DRIFT_OK and abs(out["resid"][j]) <= RESID_OK:
                out["verdict"][j] = "OK"
            elif d < -DRIFT_OK:
                out["verdict"][j] = "UNDER-compensating (raise trim)"
            elif d > DRIFT_OK:
                out["verdict"][j] = "OVER-compensating (lower trim)"
            else:
                out["verdict"][j] = f"drift ok, residual {out['resid'][j]:+.2f} Nm"
        else:
            out["verdict"][j] = "negligible gravity"
    return out


def print_metrics(log, m, path=""):
    print(f"\n=== drag session{': ' + os.path.basename(path) if path else ''} ===")
    print(f"  {m['n_total']} samples, {m['n_float']} in float "
          f"({m['n_float'] * DT:.1f}s of drag time)")
    if not m["n_float"]:
        print("  no float samples -- `engage` before `stop` to evaluate anything.")
        return
    print(f"  {'joint':18s} {'n':>6s} {'drift':>8s} {'drift_g':>8s} {'resid':>7s} "
          f"{'effort':>7s} {'trim*':>6s}  verdict")
    for j, nm in enumerate(JOINT_NAMES):
        print(f"  {nm:18s} {m['n_use'][j]:6d} {m['drift'][j]:8.3f} "
              f"{m['drift_g'][j]:8.3f} {m['resid'][j]:7.2f} {m['effort'][j]:7.2f} "
              f"{m['sugg_trim'][j]:6.2f}  {m['verdict'][j]}")
    print(f"\n  drift   : mean hands-off joint velocity [rad/s]  (|.| < {DRIFT_OK} is good)")
    print("  drift_g : drift projected on the pull direction -- SIGN IS THE VERDICT.")
    print("            Encoder-only, so independent of the torque sensor's scale.")
    print(f"  resid   : mean(tau_est - tau_ff) [Nm]  (|.| < {RESID_OK} is good)")
    print("  effort  : max |tau_est - tau_ff| while being pushed [Nm]")
    print("  trim*   : trim that would make tau_ff match the REPORTED torque. Treat as")
    print("            a hint only -- it trusts the same sensor whose scale is in")
    print("            question. Prefer nudging trim until drift_g crosses zero.")


def plot_session(path, log, m, out=None):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[drag] plot skipped: {e}")
        return None
    t = np.asarray(log["t"], float)
    q = np.asarray(log["q"], float); dq = np.asarray(log["dq"], float)
    tau_est = np.asarray(log["tau_est"], float)
    tau_ff = np.asarray(log["tau_ff"], float)
    state = np.asarray(log["state"]).ravel().astype(int)

    fig, axes = plt.subplots(2, 2, figsize=(15, 9))
    (ax_q, ax_dq), (ax_tau, ax_bar) = axes

    for j, nm in enumerate(JOINT_NAMES):
        ax_q.plot(t, q[:, j], lw=1.0, label=nm)
        ax_dq.plot(t, dq[:, j], lw=1.0, label=nm)
    ax_q.set_ylabel("q [rad]"); ax_q.set_title("measured position")
    ax_dq.set_ylabel("dq [rad/s]"); ax_dq.set_title("measured velocity (drift + drags)")
    ax_dq.axhline(0.0, color="0.6", lw=0.8)
    for ax in (ax_q, ax_dq):
        ax.set_xlabel("t [s]"); ax.legend(fontsize=7, ncol=3)

    # Only the gravity-carrying joints, else six flat lines hide the interesting two.
    show = [j for j in range(NUM_MOTORS)
            if np.abs(tau_ff[:, j]).max() > G_FLOOR_NM] or list(range(NUM_MOTORS))
    for j in show:
        ln, = ax_tau.plot(t, tau_ff[:, j], lw=1.2, label=f"{JOINT_NAMES[j]} ff")
        ax_tau.plot(t, tau_est[:, j], lw=0.9, ls="--", color=ln.get_color(),
                    label=f"{JOINT_NAMES[j]} est")
    ax_tau.set_ylabel("torque [Nm]"); ax_tau.set_xlabel("t [s]")
    ax_tau.set_title("commanded feedforward (solid) vs reported (dashed)")
    ax_tau.legend(fontsize=6, ncol=2)

    # Shade the float windows in the time-series panels: everything else is PD-held.
    fl = state == STATE_CODE["float"]
    edges = np.flatnonzero(np.diff(fl.astype(int)) != 0) + 1
    bounds = np.concatenate([[0], edges, [len(fl)]])
    for a, b in zip(bounds[:-1], bounds[1:]):
        if fl[a]:
            for ax in (ax_q, ax_dq, ax_tau):
                ax.axvspan(t[a], t[b - 1], color="green", alpha=0.08, zorder=0)

    idx = [j for j in range(NUM_MOTORS) if np.isfinite(m["drift_g"][j])]
    if idx:
        xs = np.arange(len(idx)); w = 0.38
        ax_bar.bar(xs - w / 2, [m["drift_g"][j] for j in idx], w,
                   color="C0", label="drift_g [rad/s]")
        ax_bar.bar(xs + w / 2, [m["resid"][j] for j in idx], w,
                   color="C3", label="resid [Nm]")
        ax_bar.axhline(0.0, color="k", lw=1)
        ax_bar.axhspan(-DRIFT_OK, DRIFT_OK, color="green", alpha=0.12)
        ax_bar.set_xticks(xs)
        ax_bar.set_xticklabels([JOINT_NAMES[j] for j in idx], rotation=30,
                               ha="right", fontsize=7)
        for k, j in enumerate(idx):
            if np.isfinite(m["sugg_trim"][j]):
                ax_bar.annotate(f"trim*={m['sugg_trim'][j]:.2f}", (k, 0),
                                textcoords="offset points", xytext=(0, 12),
                                ha="center", fontsize=7)
        ax_bar.legend(fontsize=7)
    else:
        ax_bar.text(0.5, 0.5, "no hands-off float samples", ha="center",
                    va="center", transform=ax_bar.transAxes)
    ax_bar.set_title("hands-off drift (sign = verdict) and residual torque")

    for ax in axes.ravel():
        ax.grid(alpha=0.3)
    fig.suptitle(f"drag session -- {os.path.basename(path)}   "
                 f"(green = float; g_scale={np.round(log['g_scale'], 2)})")
    fig.tight_layout()
    out = out or os.path.splitext(path)[0] + "_drag.png"
    fig.savefig(out, dpi=110); plt.close(fig)
    print(f"[drag] wrote {out}")
    return out


# REPL

HELP = """Commands:
  engage | float          -> ramp kp to 0 and float the arm (then drag it by hand)
  hold                    -> restore the position hold at the current pose
  trim                    -> show the per-joint gravity scales
  trim <joint|idx> <s>    -> set that joint's trim (multiplies its gravity scale)
  trim all <s>            -> set every joint's trim
  model [npz|urdf]        -> reload the identified fit (no arg = newest grav_*.npz,
                             `urdf` = pure-URDF baseline); resets trim
  catch                   -> show the fall-catch thresholds
  catch dq <rad/s>        -> loosen/tighten the runaway speed
  catch drift <rad>       -> loosen/tighten the slow-sag distance
  catch on|off            -> arm / disarm the catch (watchdog still active)
  fric [on|off]           -> friction compensation (helps hand-guiding, may buzz)
  mon [on|off]            -> live per-joint readout at 2 Hz
  rec [name] / stop       -> start / finish a session capture (npz + png + metrics)
  home                    -> ramp back to the zero pose and hold
  help                    -> this list
  exit                    -> home, then shut down"""


def _parse_toggle(tokens, current):
    if len(tokens) > 1:
        return tokens[1].lower() in ("on", "1", "true", "yes")
    return not current  # bare command flips


def _joint_index(tok):
    """Joint name or index -> 0..5, or None."""
    if tok in JOINT_NAMES:
        return JOINT_NAMES.index(tok)
    for j, nm in enumerate(JOINT_NAMES):
        if nm.startswith(tok):
            return j
    try:
        j = int(tok)
    except ValueError:
        return None
    return j if 0 <= j < NUM_MOTORS else None


def _show_trim(ctl):
    print(f"  {'joint':18s} {'g_scale':>8s} {'trim':>7s} {'applied':>8s}")
    with ctl._cmd_lock:
        trim = ctl.trim.copy()
    for j, nm in enumerate(JOINT_NAMES):
        print(f"  {nm:18s} {ctl.g_scale[j]:8.3f} {trim[j]:7.3f} "
              f"{ctl.g_scale[j] * trim[j]:8.3f}")


def _cmd_trim(ctl, tokens):
    if len(tokens) == 1:
        _show_trim(ctl)
        return
    if len(tokens) != 3:
        print("Usage: trim <joint|idx|all> <scale>   (bare `trim` shows the table)")
        return
    try:
        s = float(tokens[2])
    except ValueError:
        print(f"trim scale must be a number, got {tokens[2]!r}")
        return
    if not np.isfinite(s) or not (0.0 <= s <= 3.0):
        print(f"refusing trim {s}: must be finite and within [0, 3]")
        return
    with ctl._cmd_lock:
        if tokens[1] == "all":
            ctl.trim[:] = s
            print(f"[drag] trim set to {s} on all joints")
        else:
            j = _joint_index(tokens[1])
            if j is None:
                print(f"unknown joint {tokens[1]!r}; one of {JOINT_NAMES} or 0..5")
                return
            ctl.trim[j] = s
            print(f"[drag] trim[{JOINT_NAMES[j]}] = {s}  -> applied gravity scale "
                  f"{ctl.g_scale[j] * s:.3f}")


def _cmd_catch(ctl, tokens):
    """`catch` / `catch dq <v>` / `catch drift <v>` / `catch on|off`."""
    if len(tokens) == 1:
        with ctl._cmd_lock:
            on, dq, md = ctl.catch_on, ctl.dq_catch, ctl.max_drift
        print(f"  fall catch: {'ARMED' if on else 'DISARMED'}")
        print(f"    dq    {dq:.2f} rad/s sustained {ctl.catch_samples * ctl.dt:.2f}s"
              "  (runaway: dropping WITH gravity)")
        print(f"    drift {md:.2f} rad from the last rest pose"
              "        (creep: a slow sag)")
        print(f"    hard watchdog stays at {DQ_LIMIT[0]:.0f} rad/s regardless.")
        return
    what = tokens[1].lower()
    if what in ("on", "off"):
        ctl.set_catch(on=(what == "on"))
        print(f"[drag] fall catch {'ARMED' if what == 'on' else 'DISARMED'}"
              + ("" if what == "on" else " -- only the hard watchdog is left"))
        return
    if len(tokens) != 3 or what not in ("dq", "drift"):
        print("Usage: catch | catch dq <rad/s> | catch drift <rad> | catch on|off")
        return
    try:
        v = float(tokens[2])
    except ValueError:
        print(f"catch value must be a number, got {tokens[2]!r}")
        return
    if what == "dq":
        if not np.isfinite(v) or not (0.1 <= v <= DQ_LIMIT[0]):
            print(f"refusing dq_catch {v}: must be within [0.1, {DQ_LIMIT[0]:.0f}] "
                  "(above the watchdog it could never fire first)")
            return
        ctl.set_catch(dq_catch=v)
        print(f"[drag] catch dq = {v} rad/s")
    else:
        span = float(np.max(JOINT_HIGH - JOINT_LOW))
        if not np.isfinite(v) or not (0.05 <= v <= span):
            print(f"refusing max_drift {v}: must be within [0.05, {span:.2f}] rad")
            return
        ctl.set_catch(max_drift=v)
        print(f"[drag] catch drift = {v} rad")


def _cmd_model(ctl, tokens, log_arg=None):
    """`model` / `model <grav_xxx.npz>` / `model urdf` -> re-resolve and swap live."""
    if len(tokens) > 2:
        print("Usage: model [<grav_xxx.npz> | urdf]   (no arg = newest grav_*.npz)")
        return
    arg = tokens[1] if len(tokens) == 2 else None
    if arg is not None and arg.lower() == "urdf":
        use_ident, path = False, None
    else:
        use_ident, path = True, arg or log_arg
        if path is not None and not os.path.isfile(path):
            print(f"[drag] no such file: {path}")
            return
    try:
        res = resolve_gravity_scales(path, use_ident=use_ident)
    except Exception as e:
        print(f"[drag] cannot resolve that model, keeping the current one: {e}")
        return

    old_g = ctl.g_scale.copy()
    # Hold FIRST: a g_scale step is a torque step, and floating there is nothing to
    # absorb it. Wait for the control thread to actually apply it before swapping.
    if ctl.state in ("float", "engaging"):
        print("[drag] floating -- restoring position hold before the swap "
              "(a g_scale change is a torque step).")
        ctl.request("hold")
        t0 = time.perf_counter()
        while ctl.state != "hold" and time.perf_counter() - t0 < 1.0:
            time.sleep(0.02)
        if ctl.state != "hold":
            print("[drag] ABORTED: could not reach hold; model unchanged.")
            return
    try:
        old_g, old_trim = ctl.reload_model(res)
    except (RuntimeError, ValueError) as e:
        print(f"[drag] model unchanged: {e}")
        return

    print(f"[drag] reloaded model from {res['source']}")
    print(f"  {'joint':18s} {'old':>7s} -> {'new':>6s}   why")
    for j, nm in enumerate(JOINT_NAMES):
        mark = " " if abs(old_g[j] - ctl.g_scale[j]) < 5e-4 else "*"
        print(f" {mark}{nm:18s} {old_g[j]:7.3f} -> {ctl.g_scale[j]:6.3f}   "
              f"{res['g_why'][j]}")
    n_fit = int(np.sum(ctl.g_scale != 1.0))
    print(f"  {n_fit}/{NUM_MOTORS} joints identified "
          f"(was {int(np.sum(old_g != 1.0))}); kd_div now "
          f"{np.array2string(np.round(ctl.kd_div, 2))}")
    if np.any(old_trim != 1.0):
        print("  trim reset to 1.0 (was "
              f"{np.array2string(np.round(old_trim, 3))}) -- the new fit already "
              "contains what it was correcting.")
    print("  `engage` when ready.")


def _cmd_stop(ctl, handsoff_nm):
    log = ctl.rec_stop()
    if log is None:
        print("[drag] nothing was recorded (`rec` first)")
        return
    path = os.path.join(DATA_DIR, time.strftime("drag_%Y%m%d_%H%M%S.npz"))
    VG.save_log(path, log)
    print(f"[drag] saved {path}")
    m = session_metrics(log, handsoff_nm=handsoff_nm)
    print_metrics(log, m, path)
    plot_session(path, log, m)


def repl(ctl, handsoff_nm=HANDSOFF_NM, log_arg=None):
    print(HELP)
    while True:
        try:
            line = input("DRAG :").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        tokens = line.split()
        if not tokens:
            continue
        name = tokens[0]
        try:
            if not ctl.is_running and name not in ("exit", "help"):
                print(f"[drag] control loop is down ({ctl.abort_reason or 'stopped'}); "
                      "only `exit` remains.")
                continue
            if name in ("engage", "float"):
                ctl.request("engage")
            elif name == "hold":
                ctl.request("hold")
            elif name == "home":
                ctl.go_home()
            elif name == "trim":
                _cmd_trim(ctl, tokens)
            elif name == "model":
                _cmd_model(ctl, tokens, log_arg)
            elif name == "catch":
                _cmd_catch(ctl, tokens)
            elif name == "fric":
                with ctl._cmd_lock:
                    ctl.use_friction = _parse_toggle(tokens, ctl.use_friction)
                    now = ctl.use_friction
                print(f"[drag] friction compensation: {'on' if now else 'off'}")
            elif name == "mon":
                ctl.monitor = _parse_toggle(tokens, ctl.monitor)
                print(f"[drag] monitor: {'on' if ctl.monitor else 'off'}")
            elif name == "rec":
                ctl.rec_start(tokens[1] if len(tokens) > 1 else "")
            elif name == "stop":
                _cmd_stop(ctl, handsoff_nm)
            elif name == "help":
                print(HELP)
            elif name == "exit":
                break
            else:
                print(f"Unknown command: {name}. `help` for the list.")
        except Exception as e:
            print(f"[drag] command failed: {e}")


# Self-test (no DDS or hardware)

class _FakeDrag(DragController):
    """DragController with the DDS layer stubbed out -- feeds canned state frames to
    the real state machine / watchdog / torque law and captures what was published."""

    def __init__(self, frames, **kw):
        self._init_state(kw.pop("g_scale", np.ones(NUM_MOTORS)),
                         kw.pop("kd_div", np.ones(NUM_MOTORS)), **kw)
        self.frames = frames
        self.i = 0
        self.sent = []
        self.state_timeout = 1e9  # never "stale" in the test
        self._t0 = time.perf_counter()

    def snapshot(self):
        q, dq, tau = self.frames[min(self.i, len(self.frames) - 1)]
        self.i += 1
        return (np.asarray(q, float), np.asarray(dq, float),
                np.asarray(tau, float), time.perf_counter(), True)

    def _write(self, q_des, kp, kd, tau_ff):
        # Same guard and clipping as the hardware path, so `sent` holds exactly what
        # would have reached the motors.
        self.sent.append(tuple(a.copy() for a in
                               self._prepare_cmd(q_des, kp, kd, tau_ff)))


def _check(label, ok):
    print(f"[selftest] {label}: {'ok' if ok else 'FAIL'}")
    return bool(ok)


def _test_resolve():
    """The resolved model must use fitted scales on observable joints and 1.0 elsewhere."""
    path = find_latest_log()
    if path is None:
        print("[selftest] resolve: SKIPPED (no data/grav_*.npz)")
        return True
    res = resolve_gravity_scales(path)
    rows = {r["j"]: r for r in res["rows"]}
    ok = True
    for j in range(NUM_MOTORS):
        r = rows.get(j)
        if r is None or not r["good"]:
            ok = ok and res["g_scale"][j] == 1.0 and res["kd_div"][j] == 1.0
        elif G_SCALE_BAND[0] <= r["N_tau"] <= G_SCALE_BAND[1]:
            ok = ok and abs(res["g_scale"][j] - r["N_tau"]) < 1e-12
    # Unobservable joints must say so, not silently sit at 1.0.
    ok = ok and all(("not observable" in res["g_why"][j]) == (j not in rows)
                    for j in range(NUM_MOTORS))
    # --no-ident is the pure-URDF baseline: every factor exactly 1.
    base = resolve_gravity_scales(path, use_ident=False)
    ok = ok and np.all(base["g_scale"] == 1.0) and np.all(base["kd_div"] == 1.0)
    # --raw-kd must keep the gravity scales but drop every kd correction.
    raw = resolve_gravity_scales(path, raw_kd=True)
    ok = ok and np.all(raw["kd_div"] == 1.0) and np.allclose(raw["g_scale"],
                                                            res["g_scale"])
    return _check(f"resolve model from {os.path.basename(path)}", ok)


def _test_engage_and_catch():
    """kp must reach 0 on engage, and a fall must restore it, naming the joint."""
    q0 = np.array([0.0, 0.8, -0.8, 0.3, 0.0, 0.0])
    still = (q0, np.zeros(NUM_MOTORS), np.zeros(NUM_MOTORS))
    ctl = _FakeDrag([still], engage_s=0.05, dt=DT, trip_samples=3)
    ctl.request("engage")
    for _ in range(int(0.05 / DT) + 5):
        ctl._tick()
    ok = _check("engage ramps kp to 0",
                ctl.state == "float" and ctl.kp_blend == 0.0
                and np.all(ctl.sent[-1][1] == 0.0))

    # kd while floating must be the S-corrected value, not the hold value.
    kd_div = np.array([1.0, 18.81, 12.56, 1.0, 1.0, 1.0])
    ctl2 = _FakeDrag([still], kd_div=kd_div, engage_s=0.05, dt=DT)
    ctl2.request("engage")
    for _ in range(int(0.05 / DT) + 5):
        ctl2._tick()
    ok = _check("float kd = kd_drag / S",
                np.allclose(ctl2.sent[-1][2], KD_DRAG / kd_div)) and ok

    # `hold` must snap kp back at the CURRENT pose (no jump to a stale target).
    ctl5 = _FakeDrag([still], engage_s=0.005, dt=DT)
    ctl5.request("engage")
    for _ in range(5):
        ctl5._tick()
    ctl5.request("hold")
    ctl5._tick()
    ok = _check("hold restores kp in place",
                ctl5.state == "hold" and np.all(ctl5.sent[-1][1] == KP_HOLD)
                and np.allclose(ctl5.sent[-1][0], q0)) and ok
    return ok


# A joint LOSES to gravity when dq * sign(tau_g) < 0, so the losing direction is
# -sign(g). At the pose below, upper_arm's gravity torque is about -7 Nm, hence
# positive dq is the falling direction for it.
_Q_WORK = np.array([0.0, 0.8, -0.8, 0.3, 0.0, 0.0])


def _float_ctl(frames, **kw):
    """A _FakeDrag already in the float state, fed `frames` after engaging."""
    still = (_Q_WORK, np.zeros(NUM_MOTORS), np.zeros(NUM_MOTORS))
    kw.setdefault("engage_s", 0.005)
    kw.setdefault("dt", DT)
    ctl = _FakeDrag([still] * 3 + list(frames), **kw)
    ctl.request("engage")
    for _ in range(3):
        ctl._tick()
    assert ctl.state == "float", ctl.state
    return ctl


def _test_catch_discriminates():
    """The catch must fire on a fall and stay quiet during a drag.

    This is the regression for the bug that made drag mode unusable: CATCH fired on
    every small drag, because the trigger was speed-only at 1.5 rad/s with a 15 ms
    debounce.
    """
    g_up = float(np.sign(arm_ff.gravity_torque(_Q_WORK)[2]))   # -1 at this pose
    fall_dir = -g_up                                            # +1: the losing way
    n = int(CATCH_HOLD_S / DT) + 10

    def dq_on_j2(v):
        d = np.zeros(NUM_MOTORS); d[2] = v
        return (_Q_WORK, d, np.zeros(NUM_MOTORS))

    # 1) Sustained fast motion the way gravity pulls -> CATCH.
    ctl = _float_ctl([dq_on_j2(fall_dir * (DQ_CATCH + 1.0))] * n)
    for _ in range(n):
        ctl._tick()
    ok = _check("catch on a sustained fall", ctl.state == "caught"
                and np.all(ctl.sent[-1][1] == KP_HOLD))

    # 2) Same speed AGAINST gravity (an operator lifting the arm) -> no catch.
    ctl = _float_ctl([dq_on_j2(-fall_dir * (DQ_CATCH + 1.0))] * n)
    for _ in range(n):
        ctl._tick()
    ok = _check("fast drag against gravity does not catch",
                ctl.state == "float") and ok

    # 3) A brisk but BOUNDED drag the falling way, then released -> no catch.
    short = int(0.5 * CATCH_HOLD_S / DT)
    frames = [dq_on_j2(fall_dir * (DQ_CATCH + 1.0))] * short + \
             [dq_on_j2(0.0)] * n
    ctl = _float_ctl(frames)
    for _ in range(short + n):
        ctl._tick()
    ok = _check("bounded drag does not catch", ctl.state == "float") and ok

    # 4) An ordinary slow drag (the speed that used to trip it) -> no catch.
    ctl = _float_ctl([dq_on_j2(fall_dir * 1.5)] * n)
    for _ in range(n):
        ctl._tick()
    ok = _check("1.5 rad/s drag does not catch (the old false trip)",
                ctl.state == "float") and ok

    # 5) A SLOW sag below dq_catch that keeps sliding -> creep catch. Nothing a
    #    speed-only test could ever see. Uses arm_base: it is the only gravity-loaded
    #    joint with more than MAX_DRIFT of travel left in its losing direction, and
    #    sliding past a hard limit would trip the position watchdog instead.
    base_fall = -float(np.sign(arm_ff.gravity_torque(_Q_WORK)[1]))     # +1 here
    slid = _Q_WORK.copy()
    slid[1] += base_fall * (MAX_DRIFT + 0.1)
    assert JOINT_LOW[1] < slid[1] < JOINT_HIGH[1], slid[1]
    sagging = (slid, np.zeros(NUM_MOTORS), np.zeros(NUM_MOTORS))
    d = np.zeros(NUM_MOTORS); d[1] = base_fall * 0.2   # > CREEP_DQ, << dq_catch
    ctl = _float_ctl([(slid, d, np.zeros(NUM_MOTORS))] * n)
    for _ in range(n):
        ctl._tick()
    ok = _check("catch on a slow sag (creep)", ctl.state == "caught") and ok

    # 6) Parking the joint re-anchors, so the same displacement no longer counts.
    ctl = _float_ctl([sagging] * 5 + [(slid, d, np.zeros(NUM_MOTORS))] * n)
    for _ in range(5 + n):
        ctl._tick()
    ok = _check("parking re-anchors the creep reference",
                ctl.state == "float") and ok

    # 6b) Reversing re-anchors too: a back-and-forth drag must not accumulate into a
    #     "sag" it never was. Without this the wiggle of a real hand trips creep.
    fwd = np.zeros(NUM_MOTORS); fwd[1] = base_fall * 0.5
    back = np.zeros(NUM_MOTORS); back[1] = -base_fall * 0.5
    half = int(0.4 / DT)
    frames, q = [], _Q_WORK.copy()
    for _cycle in range(6):
        for dvec in (fwd, back):
            for _ in range(half):
                q = q + dvec * DT
                frames.append((q.copy(), dvec, np.zeros(NUM_MOTORS)))
    ctl = _float_ctl(frames)
    for _ in range(len(frames)):
        ctl._tick()
    ok = _check("reversing re-anchors (back-and-forth drag is not creep)",
                ctl.state == "float") and ok

    # 7) Actually past a hard limit and still going out -> catch, on the SHORT
    #    debounce (no reason to grind for 200 ms).
    past = _Q_WORK.copy(); past[1] = JOINT_LOW[1] - 0.02
    d2 = np.zeros(NUM_MOTORS); d2[1] = -0.2
    ctl = _float_ctl([(past, d2, np.zeros(NUM_MOTORS))] * 10, trip_samples=3)
    for _ in range(6):
        ctl._tick()
    ok = _check("catch when past a hard limit (short debounce)",
                ctl.state == "caught") and ok

    # 8) `catch off` disarms it; the hard watchdog is untouched.
    ctl = _float_ctl([dq_on_j2(fall_dir * (DQ_CATCH + 1.0))] * n, catch_on=False)
    for _ in range(n):
        ctl._tick()
    ok = _check("catch off disarms the catch", ctl.state == "float") and ok
    return ok


def _test_engage_at_home():
    """Engaging at the ZERO pose must be quiet.

    Direct regression for the bug that made drag mode unusable on hardware: the home
    pose is exactly ON arm_base's low limit and upper_arm's high limit, and the barrier
    used to read that as full penetration -> +6/-6 Nm at rest -> a catch within ~5
    ticks, before the arm was touched.
    """
    home = np.zeros(NUM_MOTORS)
    still = (home, np.zeros(NUM_MOTORS), np.zeros(NUM_MOTORS))
    ok = _check("barrier is exactly zero at the home pose",
                np.allclose(limit_barrier(home, np.zeros(NUM_MOTORS)), 0.0))
    ctl = _FakeDrag([still], engage_s=0.05, dt=DT)
    ctl.request("engage")
    for _ in range(200):
        ctl._tick()
    ok = _check("engaging at home does not catch (200 ticks)",
                ctl.state == "float") and ok
    # The published torque must be gravity only -- no barrier contribution.
    ok = _check("home torque is gravity only",
                np.allclose(ctl.sent[-1][3], arm_ff.gravity_torque(home))) and ok
    return ok


def _test_barrier():
    """Zero at rest everywhere legal; spring only past a stop; damping outward only."""
    zero = np.zeros(NUM_MOTORS)
    # At rest the barrier must be silent anywhere inside the range -- INCLUDING
    # exactly on a limit, which the home pose is for two joints.
    ok = np.allclose(limit_barrier(0.5 * (JOINT_LOW + JOINT_HIGH), zero), 0.0)
    ok = ok and np.allclose(limit_barrier(JOINT_LOW, zero), 0.0)
    ok = ok and np.allclose(limit_barrier(JOINT_HIGH, zero), 0.0)
    ok = ok and np.allclose(limit_barrier(JOINT_LOW + 0.5 * LIMIT_MARGIN, zero), 0.0)
    ok = _check("barrier silent at rest inside the range", ok)

    # Past a stop the spring pushes back in, proportional to how far past.
    ok = _check("barrier springs back past a stop",
                np.all(limit_barrier(JOINT_LOW - 0.1, zero) > 0)
                and np.all(limit_barrier(JOINT_HIGH + 0.1, zero) < 0)
                and np.all(limit_barrier(JOINT_LOW - 0.2, zero)
                           > limit_barrier(JOINT_LOW - 0.1, zero))) and ok

    # Inside the band it damps OUTWARD motion only, and not inward motion.
    lo = JOINT_LOW + 0.5 * LIMIT_MARGIN
    out = limit_barrier(lo, np.full(NUM_MOTORS, -1.0))   # moving toward the low stop
    inn = limit_barrier(lo, np.full(NUM_MOTORS, +1.0))   # moving away from it
    ok = _check("barrier damps outward motion only",
                np.all(out > 0) and np.allclose(inn, 0.0)) and ok

    # Damping ramps from 0 at the band edge, so entering the band is not a step.
    edge = JOINT_LOW + LIMIT_MARGIN
    dq_in = np.full(NUM_MOTORS, -1.0)
    ok = _check("barrier damping is continuous at the band edge",
                np.allclose(limit_barrier(edge, dq_in), 0.0)
                and np.all(limit_barrier(edge - 0.01, dq_in) > 0)
                and np.all(limit_barrier(edge - 0.01, dq_in) < 0.2 * D_BARRIER)) and ok
    return ok


def _test_write_guard():
    """Non-finite or negative commands must raise, never reach the motors."""
    q0 = np.zeros(NUM_MOTORS)
    ok = True
    for label, args in (
            ("nan q", (np.full(NUM_MOTORS, np.nan), KP_HOLD, KD_HOLD, q0)),
            ("inf tau", (q0, KP_HOLD, KD_HOLD, np.full(NUM_MOTORS, np.inf))),
            ("short kp", (q0, np.ones(3), KD_HOLD, q0)),
            ("negative kd", (q0, KP_HOLD, -KD_HOLD, q0))):
        try:
            DragController._prepare_cmd(*args)
            ok = False
            print(f"[selftest]   {label} was NOT rejected")
        except ValueError:
            pass
    # A command past the joint stops must be clipped, not published as asked.
    q_out, _, _, tau_out = DragController._prepare_cmd(
        JOINT_HIGH + 1.0, KP_HOLD, KD_HOLD, np.full(NUM_MOTORS, 1e3))
    ok = ok and np.allclose(q_out, JOINT_HIGH)
    ok = ok and np.allclose(tau_out, MOTOR_TAU_LIMIT)
    # Bad gains must be refused at construction too.
    for bad in (dict(g_scale=np.full(NUM_MOTORS, np.nan)),
                dict(kd_div=np.zeros(NUM_MOTORS)),
                dict(kd_div=-np.ones(NUM_MOTORS))):
        try:
            _FakeDrag([(q0, q0, q0)], **bad)
            ok = False
            print(f"[selftest]   construction with {list(bad)[0]} was NOT rejected")
        except ValueError:
            pass
    return _check("_write / construction reject bad values", ok)


def _test_metrics_roundtrip():
    """A synthetic under-compensating session must be diagnosed as UNDER, and the
    suggested trim must recover the factor that was left out."""
    n = 400
    g = np.zeros((n, NUM_MOTORS)); g[:, 1] = 4.0; g[:, 2] = -5.0
    # The arm was run 20% light on j1/j2: applied = g_scale*g, and g_scale < 1, so the
    # trim that fixes it is exactly 1/g_scale.
    g_scale = np.array([1.0, 0.8, 0.8, 1.0, 1.0, 1.0])
    trim_true = 1.0 / g_scale[1]       # 1.25
    tau_ff = g * g_scale[None, :]
    tau_est = g.copy()                 # the joint really needed the full torque
    dq = np.zeros((n, NUM_MOTORS))
    # Losing to gravity: dq points opposite to the pull, i.e. -sign(g).
    dq[:, 1] = -0.20; dq[:, 2] = +0.20
    log = dict(t=np.arange(n) * DT, q=np.zeros((n, NUM_MOTORS)), dq=dq,
               tau_est=tau_est, tau_ff=tau_ff, g_urdf=g, tau_g=tau_ff,
               kp_out=np.zeros((n, NUM_MOTORS)), kd_out=np.zeros((n, NUM_MOTORS)),
               state=np.full(n, STATE_CODE["float"]),
               joint_names=np.array(JOINT_NAMES), g_scale=g_scale,
               kd_div=np.ones(NUM_MOTORS), trim=np.ones(NUM_MOTORS))
    m = session_metrics(log, handsoff_nm=2.0)
    ok = (m["verdict"][1].startswith("UNDER") and m["verdict"][2].startswith("UNDER")
          and abs(m["sugg_trim"][1] - trim_true) < 1e-6
          and abs(m["sugg_trim"][2] - trim_true) < 1e-6
          and m["drift_g"][1] < 0 and m["drift_g"][2] < 0)
    ok = _check("metrics diagnose UNDER + recover trim", ok)

    # The recovered trim must NOT depend on what trim happened to be at stop time:
    # tau_g carries it per sample, the fit regresses on g_scale*g_urdf instead.
    m_alt = session_metrics(dict(log, trim=np.full(NUM_MOTORS, 0.5)), handsoff_nm=2.0)
    ok = _check("suggested trim independent of the stop-time trim",
                abs(m_alt["sugg_trim"][1] - trim_true) < 1e-6) and ok

    # Perfect compensation, no drift -> OK, and joints without gravity stay quiet.
    log_ok = dict(log, tau_ff=g.copy(), tau_g=g.copy(),
                  g_scale=np.ones(NUM_MOTORS), dq=np.zeros((n, NUM_MOTORS)))
    m2 = session_metrics(log_ok, handsoff_nm=2.0)
    ok = _check("metrics call a good session OK",
                m2["verdict"][1] == "OK" and m2["verdict"][2] == "OK"
                and m2["verdict"][0] == "negligible gravity"
                and abs(m2["sugg_trim"][1] - 1.0) < 1e-6) and ok

    # An OVER-compensating session must be called out in the other direction.
    over = dict(log, g_scale=np.array([1.0, 1.2, 1.2, 1.0, 1.0, 1.0]),
                tau_ff=g * 1.2, tau_g=g * 1.2)
    over["dq"] = -dq   # rising against gravity
    m5 = session_metrics(over, handsoff_nm=2.0)
    ok = _check("metrics diagnose OVER",
                m5["verdict"][1].startswith("OVER")
                and abs(m5["sugg_trim"][1] - 1.0 / 1.2) < 1e-6) and ok

    # Held samples must be ignored: the same data marked 'hold' yields no verdict.
    m3 = session_metrics(dict(log, state=np.full(n, STATE_CODE["hold"])),
                         handsoff_nm=2.0)
    ok = _check("metrics ignore non-float samples",
                m3["n_float"] == 0 and m3["n_use"][1] == 0) and ok

    # Log -> save -> load -> metrics -> plot must survive a real round trip.
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "drag_selftest.npz")
        VG.save_log(p, log)
        back = VG.load_log(p)
        m4 = session_metrics(back, handsoff_nm=2.0)
        png = plot_session(p, back, m4)
        ok = _check("save/load/plot round trip",
                    m4["verdict"][1].startswith("UNDER")
                    and (png is None or os.path.isfile(png))) and ok
    return ok


def _test_torque_law():
    """tau_ff must be exactly g_scale*trim*g(q) plus the barrier, and stay clamped."""
    q0 = np.array([0.0, 0.8, -0.8, 0.3, 0.0, 0.0])
    still = (q0, np.zeros(NUM_MOTORS), np.zeros(NUM_MOTORS))
    gs = np.array([1.0, 1.52, 0.98, 0.96, 1.0, 1.0])
    ctl = _FakeDrag([still], g_scale=gs, engage_s=0.005, dt=DT)
    ctl.request("engage")
    for _ in range(5):
        ctl._tick()
    zero = np.zeros(NUM_MOTORS)
    want = gs * arm_ff.gravity_torque(q0) + limit_barrier(q0, zero)
    ok = np.allclose(ctl.sent[-1][3], want)
    with ctl._cmd_lock:
        ctl.trim[1] = 1.3
    ctl._tick()
    want[1] = gs[1] * 1.3 * arm_ff.gravity_torque(q0)[1] + limit_barrier(q0, zero)[1]
    ok = ok and np.allclose(ctl.sent[-1][3], want)
    # The barrier must be scaled too -- it is a commanded torque like any other. Stay
    # inside the watchdog's 0.05 rad position tolerance so only the barrier is exercised.
    past = q0.copy(); past[1] = JOINT_LOW[1] - 0.03
    ctl_b = _FakeDrag([(past, zero, zero)], g_scale=np.ones(NUM_MOTORS),
                      engage_s=0.005, dt=DT)
    ctl_b.request("engage")
    for _ in range(5):
        ctl_b._tick()
    ok = ok and np.allclose(
        ctl_b.sent[-1][3],
        arm_ff.gravity_torque(past) + limit_barrier(past, zero))
    # A wild g_scale must still leave the published torque inside the clamp.
    ctl2 = _FakeDrag([still], g_scale=np.full(NUM_MOTORS, 3.0), engage_s=0.005, dt=DT)
    ctl2.request("engage")
    for _ in range(5):
        ctl2._tick()
    lim = arm_ff.FF_CLAMP_FRAC * MOTOR_TAU_LIMIT
    ok = ok and np.all(np.abs(ctl2.sent[-1][3]) <= lim + 1e-9)
    return _check("torque law = g_scale*trim*g(q) + barrier, clamped", ok)


def _test_watchdog():
    """A runaway aborts instantly; torque and soft breaches need trip_samples."""
    q0 = np.zeros(NUM_MOTORS)
    ctl = _FakeDrag([(q0, q0, q0)], trip_samples=3)
    ok = True
    # A severe VELOCITY is a runaway -> instant.
    sev_dq = q0.copy(); sev_dq[4] = 1.2 * DQ_LIMIT[4]
    try:
        ctl._watchdog(q0, sev_dq, q0, time.perf_counter(), True)
        ok = False
    except RuntimeError as e:
        ok = ok and "severe" in str(e)
    # A SATURATED torque must NOT abort instantly: a saturated joint always reads at
    # the motor limit, so an instant tier there turns one spike into a killed session.
    ctl_t = _FakeDrag([(q0, q0, q0)], trip_samples=3)
    sat = MOTOR_TAU_LIMIT.copy()
    try:
        ctl_t._watchdog(q0, q0, sat, time.perf_counter(), True)
        ctl_t._watchdog(q0, q0, sat, time.perf_counter(), True)
    except RuntimeError:
        ok = False
        print("[selftest]   saturated torque aborted before trip_samples")
    try:  # the third consecutive sample still must abort
        ctl_t._watchdog(q0, q0, sat, time.perf_counter(), True)
        ok = False
        print("[selftest]   saturated torque never aborted")
    except RuntimeError as e:
        ok = ok and "3 consec" in str(e) and "severe" not in str(e)
    soft = q0.copy(); soft[4] = 6.5  # > SAFETY_TAU[4]=6.3
    ctl2 = _FakeDrag([(q0, q0, q0)], trip_samples=3)
    raised = None
    for _ in range(3):
        try:
            ctl2._watchdog(q0, q0, soft, time.perf_counter(), True)
        except RuntimeError as e:
            raised = str(e); break
    ok = ok and raised is not None and "5dof_joint" in raised and "3 consec" in raised
    # Two soft then a clear sample must reset the counter.
    ctl3 = _FakeDrag([(q0, q0, q0)], trip_samples=3)
    ctl3._watchdog(q0, q0, soft, time.perf_counter(), True)
    ctl3._watchdog(q0, q0, soft, time.perf_counter(), True)
    ctl3._watchdog(q0, q0, q0, time.perf_counter(), True)
    try:
        ctl3._watchdog(q0, q0, soft, time.perf_counter(), True)
    except RuntimeError:
        ok = False
    # A stale frame must abort even when every value is in range. _FakeDrag disables
    # the staleness check by default, so re-arm it for this one assertion.
    ctl3.state_timeout = 0.1
    try:
        ctl3._watchdog(q0, q0, q0, time.perf_counter() - 10.0, True)
        ok = False
    except RuntimeError as e:
        ok = ok and "stale" in str(e)
    # ... and a missing state must abort regardless of age.
    try:
        ctl3._watchdog(q0, q0, q0, time.perf_counter(), False)
        ok = False
    except RuntimeError as e:
        ok = ok and "stale" in str(e)
    return _check("watchdog debounce + stale detection", ok)


def _test_homing():
    """`home` must ramp q_des to zero from wherever the arm is, holding kp."""
    q0 = np.array([0.0, 0.8, -0.8, 0.3, 0.0, 0.0])
    still = (q0, np.zeros(NUM_MOTORS), np.zeros(NUM_MOTORS))
    ctl = _FakeDrag([still], engage_s=0.05, dt=DT)
    ctl.request("home")
    steps = int(0.05 / DT)
    for _ in range(steps + 3):
        ctl._tick()
    q_first = ctl.sent[0][0]
    ok = (ctl.state == "hold" and ctl._homed.is_set()
          and np.allclose(ctl.sent[-1][0], np.zeros(NUM_MOTORS))
          and np.all(ctl.sent[-1][1] == KP_HOLD)
          # and it RAMPED: the first command is still near the start pose
          and abs(q_first[1] - q0[1]) < 0.1)
    return _check("home ramps q_des to zero under full kp", ok)


def _test_recording():
    """rec -> ticks -> stop must yield a log whose metrics are computable, and the
    sample cap must stop growth instead of eating memory."""
    q0 = np.array([0.0, 0.8, -0.8, 0.3, 0.0, 0.0])
    still = (q0, np.zeros(NUM_MOTORS), np.zeros(NUM_MOTORS))
    gs = np.array([1.0, 1.52, 0.98, 0.96, 1.0, 1.0])
    ctl = _FakeDrag([still], g_scale=gs, engage_s=0.005, dt=DT)
    ok = _check("stop with nothing recorded returns None", ctl.rec_stop() is None)
    ctl.rec_start("unit")
    ctl.request("engage")
    for _ in range(40):
        ctl._tick()
    log = ctl.rec_stop()
    ok = (log is not None and len(log["t"]) == 40
          and log["q"].shape == (40, NUM_MOTORS)
          and np.all(log["state"][-1] == STATE_CODE["float"])
          and np.allclose(log["g_scale"], gs)
          and str(np.asarray(log["name"]).item()) == "unit") and ok
    ok = _check("rec/stop produces a well-formed log", ok)
    m = session_metrics(log, handsoff_nm=1e6)   # everything counts as hands-off
    ok = _check("metrics run on a recorded session",
                m["n_float"] > 0 and np.isfinite(m["drift"][1])) and ok
    # A second stop must not re-emit the same log.
    ok = _check("stop is idempotent", ctl.rec_stop() is None) and ok
    # The cap must hold, and rec_stop must drop the sentinel row.
    ctl2 = _FakeDrag([still], engage_s=0.005, dt=DT)
    ctl2.rec_start()
    global MAX_SAMPLES
    saved, MAX_SAMPLES = MAX_SAMPLES, 10
    try:
        for _ in range(25):
            ctl2._tick()
        log2 = ctl2.rec_stop()
    finally:
        MAX_SAMPLES = saved
    ok = _check("recording cap stops growth",
                log2 is not None and len(log2["t"]) == 10) and ok
    return ok


def _test_repl_commands():
    """The REPL helpers must accept names/indices/abbreviations and reject nonsense."""
    q0 = np.zeros(NUM_MOTORS)
    ctl = _FakeDrag([(q0, q0, q0)])
    ok = (_joint_index("upper_arm_joint") == 2 and _joint_index("2") == 2
          and _joint_index("fore") == 3 and _joint_index("nope") is None
          and _joint_index("9") is None and _joint_index("-1") is None)
    ok = _check("joint lookup by name / prefix / index", ok)

    _cmd_trim(ctl, ["trim", "arm_base_joint", "1.3"])
    ok = _check("trim by name", ctl.trim[1] == 1.3) and ok
    _cmd_trim(ctl, ["trim", "all", "1.1"])
    ok = _check("trim all", np.all(ctl.trim == 1.1)) and ok
    for bad in (["trim", "arm_base_joint", "abc"],   # not a number
                ["trim", "arm_base_joint", "9"],     # outside [0, 3]
                ["trim", "arm_base_joint", "nan"],   # non-finite
                ["trim", "nosuchjoint", "1.0"],      # unknown joint
                ["trim", "1.0"]):                    # wrong arity
        _cmd_trim(ctl, bad)
    ok = _check("bad trim commands are rejected", np.all(ctl.trim == 1.1)) and ok

    ok = _check("toggle parsing",
                _parse_toggle(["fric"], False) is True
                and _parse_toggle(["fric"], True) is False
                and _parse_toggle(["fric", "on"], False) is True
                and _parse_toggle(["fric", "off"], True) is False) and ok
    ok = _check("bad transition request is rejected",
                _raises(lambda: ctl.request("fly"), ValueError)) and ok
    return ok


def _test_reload_model():
    """`model` must swap the fit live, reset trim, and refuse when it would lie."""
    q0 = np.array([0.0, 0.8, -0.8, 0.3, 0.0, 0.0])
    still = (q0, np.zeros(NUM_MOTORS), np.zeros(NUM_MOTORS))
    gs_a = np.array([1.0, 1.52, 0.98, 0.96, 1.0, 1.0])
    gs_b = np.array([1.0, 1.10, 1.05, 1.00, 1.0, 1.0])
    res_b = dict(g_scale=gs_b, kd_div=np.full(NUM_MOTORS, 2.0),
                 g_why=["fitted"] * NUM_MOTORS, kd_why=["S=2"] * NUM_MOTORS,
                 source="grav_fake_b.npz", rows=[])

    ctl = _FakeDrag([still], g_scale=gs_a, engage_s=0.005, dt=DT)
    with ctl._cmd_lock:
        ctl.trim[1] = 1.3
    old_g, old_trim = ctl.reload_model(res_b)
    ok = _check("reload swaps g_scale/kd_div and resets trim",
                np.allclose(old_g, gs_a) and old_trim[1] == 1.3
                and np.allclose(ctl.g_scale, gs_b)
                and np.allclose(ctl.kd_div, 2.0) and np.all(ctl.trim == 1.0))

    # The new model must actually reach the motors on the next tick.
    ctl._tick()
    ok = _check("reloaded model is what gets published",
                np.allclose(ctl.sent[-1][3],
                            gs_b * arm_ff.gravity_torque(q0)
                            + limit_barrier(q0, np.zeros(NUM_MOTORS)))) and ok

    # Refuse mid-recording: a session log carries ONE g_scale.
    ctl.rec_start()
    ctl._tick()
    ok = _check("reload refused while recording",
                _raises(lambda: ctl.reload_model(res_b), RuntimeError)) and ok
    ctl.rec_stop()

    # Bad models must be rejected before they reach g_scale.
    for bad in (dict(res_b, g_scale=np.full(NUM_MOTORS, np.nan)),
                dict(res_b, kd_div=np.zeros(NUM_MOTORS)),
                dict(res_b, g_scale=-gs_b)):
        ok = _raises(lambda: ctl.reload_model(bad), ValueError) and ok
    ok = _check("reload rejects a non-finite / non-positive model",
                np.allclose(ctl.g_scale, gs_b)) and ok

    # `model urdf` resolves to the pure-URDF baseline.
    base = resolve_gravity_scales(None, use_ident=False)
    ctl.reload_model(base)
    ok = _check("model urdf falls back to all-1.0",
                np.all(ctl.g_scale == 1.0) and np.all(ctl.kd_div == 1.0)) and ok

    # From float, _cmd_model must hold FIRST (a g_scale step is a torque step). The
    # fake has no control thread, so drive the tick that applies the request.
    ctl2 = _FakeDrag([still], g_scale=gs_a, engage_s=0.005, dt=DT)
    ctl2.request("engage")
    for _ in range(3):
        ctl2._tick()
    assert ctl2.state == "float"
    t = threading.Thread(target=_cmd_model, args=(ctl2, ["model", "urdf"]), daemon=True)
    t.start()
    t0 = time.perf_counter()
    while t.is_alive() and time.perf_counter() - t0 < 2.0:
        ctl2._tick()
        time.sleep(0.001)
    t.join(timeout=1.0)
    ok = _check("model from float holds before swapping",
                ctl2.state == "hold" and np.all(ctl2.g_scale == 1.0)) and ok

    # A path that does not exist must leave the model alone.
    ctl3 = _FakeDrag([still], g_scale=gs_a, engage_s=0.005, dt=DT)
    _cmd_model(ctl3, ["model", "/no/such/grav.npz"])
    ok = _check("model with a bad path changes nothing",
                np.allclose(ctl3.g_scale, gs_a)) and ok
    return ok


def _test_catch_command():
    """`catch` must retune live and reject values that could never fire."""
    q0 = np.zeros(NUM_MOTORS)
    ctl = _FakeDrag([(q0, q0, q0)])
    _cmd_catch(ctl, ["catch"])                       # must not raise
    _cmd_catch(ctl, ["catch", "dq", "5.0"])
    ok = _check("catch dq applies", ctl.dq_catch == 5.0)
    _cmd_catch(ctl, ["catch", "drift", "1.2"])
    ok = _check("catch drift applies", ctl.max_drift == 1.2) and ok
    _cmd_catch(ctl, ["catch", "off"])
    ok = _check("catch off disarms", ctl.catch_on is False) and ok
    _cmd_catch(ctl, ["catch", "on"])
    ok = _check("catch on rearms", ctl.catch_on is True) and ok
    for bad in (["catch", "dq", "abc"],                      # not a number
                ["catch", "dq", str(DQ_LIMIT[0] + 1)],       # above the watchdog
                ["catch", "dq", "0.01"],                     # absurdly tight
                ["catch", "drift", "99"],                    # wider than any joint
                ["catch", "nonsense", "1"]):
        _cmd_catch(ctl, bad)
    ok = _check("bad catch commands are rejected",
                ctl.dq_catch == 5.0 and ctl.max_drift == 1.2) and ok
    ok = _check("set_catch rejects non-positive values",
                _raises(lambda: ctl.set_catch(dq_catch=0), ValueError)
                and _raises(lambda: ctl.set_catch(max_drift=-1), ValueError)) and ok
    return ok


def _raises(fn, exc):
    try:
        fn()
    except exc:
        return True
    except Exception:
        return False
    return False


def selftest():
    ok = _test_resolve()
    ok = _test_engage_and_catch() and ok
    ok = _test_engage_at_home() and ok
    ok = _test_catch_discriminates() and ok
    ok = _test_homing() and ok
    ok = _test_barrier() and ok
    ok = _test_write_guard() and ok
    ok = _test_torque_law() and ok
    ok = _test_watchdog() and ok
    ok = _test_recording() and ok
    ok = _test_reload_model() and ok
    ok = _test_catch_command() and ok
    ok = _test_repl_commands() and ok
    ok = _test_metrics_roundtrip() and ok
    print(f"[selftest] {'PASS' if ok else 'FAIL'} (model resolution + engage/home "
          "+ catch discrimination + barrier + command guards + torque law + watchdog "
          "+ recording + model reload + catch tuning + repl + metrics)")
    return 0 if ok else 1


# CLI

def main() -> int:
    ap = argparse.ArgumentParser(
        description="Float the arm on identified gravity compensation so it can be "
                    "hand-dragged, and measure how good that compensation is.")
    ap.add_argument("net", nargs="?", default="lo",
                    help="DDS network interface (e.g. eth0)")
    ap.add_argument("--log", metavar="NPZ", default=None,
                    help="gravity log to take the identified per-joint factors from "
                         "(default: newest data/grav_*.npz)")
    ap.add_argument("--no-ident", action="store_true",
                    help="ignore the identification and use the plain URDF model "
                         "(the A/B baseline)")
    ap.add_argument("--raw-kd", action="store_true",
                    help="do not divide kd by the identified stiffness factor S")
    ap.add_argument("--show-model", action="store_true",
                    help="offline: print the resolved per-joint model and exit")
    ap.add_argument("--selftest", action="store_true",
                    help="offline: state machine / safety / metrics checks (no DDS)")
    ap.add_argument("--engage-s", type=float, default=3.0,
                    help="seconds to ramp kp down to zero on `engage`")
    ap.add_argument("--dq-catch", type=float, default=DQ_CATCH,
                    help=f"sustained joint speed [rad/s] that restores kp (fall catch, "
                         f"default {DQ_CATCH}); also settable live with `catch dq`")
    ap.add_argument("--max-drift", type=float, default=MAX_DRIFT,
                    help=f"slide [rad] from a joint's last rest pose that counts as a "
                         f"slow sag (default {MAX_DRIFT}); live via `catch drift`")
    ap.add_argument("--no-catch", action="store_true",
                    help="disarm the fall catch entirely (the hard watchdog stays). "
                         "Only for a model you already trust.")
    ap.add_argument("--trip-samples", type=int, default=3,
                    help="consecutive violating samples before a watchdog abort")
    ap.add_argument("--handsoff-nm", type=float, default=HANDSOFF_NM,
                    help="|tau_est - tau_ff| below which a sample counts as hands-off")
    ap.add_argument("--kd-drag", type=float, nargs=NUM_MOTORS, default=KD_DRAG.tolist(),
                    metavar=tuple(f"D{i}" for i in range(NUM_MOTORS)),
                    help="damping WANTED while floating (before the S correction)")
    ap.add_argument("--fric", action="store_true",
                    help="start with friction compensation on")
    ap.add_argument("--mon", action="store_true",
                    help="start with the live readout on")
    args = ap.parse_args()

    if args.trip_samples < 1:
        ap.error("--trip-samples must be >= 1")
    if not np.isfinite(args.engage_s) or args.engage_s <= 0:
        ap.error("--engage-s must be positive")
    if not np.isfinite(args.dq_catch) or args.dq_catch <= 0:
        ap.error("--dq-catch must be positive")
    if not np.isfinite(args.max_drift) or args.max_drift <= 0:
        ap.error("--max-drift must be positive")
    if args.handsoff_nm <= 0:
        ap.error("--handsoff-nm must be positive")
    kd_drag = np.asarray(args.kd_drag, float)

    if args.selftest:
        return selftest()

    try:
        res = resolve_gravity_scales(args.log, use_ident=not args.no_ident,
                                     raw_kd=args.raw_kd)
    except Exception as e:
        print(f"[drag] ERROR: cannot resolve the gravity model: {e}")
        return 2
    print_model(res, kd_drag)
    if args.show_model:
        return 0

    print("\nWARNING: the position loop will be switched OFF -- the arm is held by "
          "gravity feedforward alone and WILL fall if the model is wrong. Keep the "
          "workspace clear and a hand near the forearm.")
    input("Press Enter to start drag mode...")

    from unitree_sdk2py.core.channel import ChannelFactoryInitialize
    ChannelFactoryInitialize(1, args.net)

    ctl = None
    try:
        ctl = DragController(res["g_scale"], res["kd_div"], kd_drag=kd_drag,
                             dq_catch=args.dq_catch, max_drift=args.max_drift,
                             catch_on=not args.no_catch, engage_s=args.engage_s,
                             trip_samples=args.trip_samples)
        ctl.use_friction = args.fric
        ctl.monitor = args.mon
        if args.no_catch:
            print("[drag] fall catch DISARMED (--no-catch); only the hard watchdog "
                  f"(|dq| > {DQ_LIMIT[0]:.0f} rad/s, torque, position) is left.")
        ctl.wait_for_state()
        ctl.start()
        print("[drag] holding the current pose with gravity comp. `engage` to float.")
        repl(ctl, handsoff_nm=args.handsoff_nm, log_arg=args.log)
    except KeyboardInterrupt:
        print("\n[drag] interrupted")
    except (TimeoutError, RuntimeError, ValueError) as e:
        print(f"[drag] ERROR: {e}")
        return 2
    finally:
        if ctl is not None:
            if ctl._rec:
                print("[drag] discarding an unsaved recording (use `stop` to keep it)")
            if ctl.is_running:
                ctl.go_home()      # ramp down under the live loop
                ctl.stop()
            elif ctl.has_state():
                ctl.safe_return()  # loop is dead: bring it down here
    return 2 if (ctl is not None and ctl.abort_reason) else 0


if __name__ == "__main__":
    sys.exit(main())
