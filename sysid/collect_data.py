"""Collect a system-identification excitation dataset on the REAL pineapple arm.

Extends pineapple_arm.py's DDS controller: PD position-controls the 6 joints through
a per-joint chirp + slow triangle (for low-speed friction) and logs measured
q/dq/tau plus commands and gains at 200 Hz into an ``.npz`` for ``sysid_fit.py``.
Desired velocity is zero, matching pineapple_arm.py and sysid_fit's PD replay.

Standalone (only unitree_sdk2py + numpy) so it runs in the arm's runtime without
MuJoCo. Keys match ``sysid_common.LOG_KEYS`` plus schema/completeness metadata,
all arrays in motor index order 0..5.

Safety: commands hard-clamped inside each joint's range; smooth ramp to the start
pose; live watchdog aborts on stale state or torque/velocity/range breach; always
ramps back to neutral on finish/Ctrl-C.

Validate offline first (no hardware):
    python collect_data.py --dry-run                 # writes a preview plot
Then, on the robot:
    python collect_data.py [network_interface]       # e.g. eth0 (default: lo)
"""

from __future__ import annotations

import argparse
import sys
import threading
import time

import numpy as np

# Duplicated from sysid_common.py so collection needs no MuJoCo. Motor order 0..5.
NUM_MOTORS = 6
DT = 0.005  # 200 Hz, matches pineapple_arm.py
LOG_SCHEMA_VERSION = 2
TIME_SOURCE = "state_arrival_perf_counter"

JOINT_NAMES = ["arm_joint", "arm_base_joint", "upper_arm_joint",
               "fore_arm_joint", "5dof_joint", "gripper_case_joint"]

# Hard joint limits from pineapple_arm.xml (radians).
JOINT_LOW = np.array([-1.5708, 0.0, -3.1416, -1.5708, -1.5708, -1.5708])
JOINT_HIGH = np.array([1.5708, 3.1416, 0.0, 1.7453, 1.5708, 1.5708])

# Neutral pose to excite around (feasible, mildly extended). Clamped to limits.
CENTER = np.array([0.0, 0.8, -0.8, 0.3, 0.0, 0.0])

# Excitation is sized so realized peaks stay under these: torque saturation
# destroys parameter information (it caused the earlier armature gap). The
# gravity-loaded shoulder/elbow get 18 Nm because upper_arm alone holds ~7 Nm
# against gravity, leaving no room under 10. Distal j3/j4/j5 are DM-4310 (+-7 Nm),
# so their effective cap is ~7 regardless of the number here.
TARGET_TAU = np.array([10.0, 18.0, 18.0, 10.0, 10.0, 10.0])  # peak |joint torque| [Nm]
TARGET_DQ = np.array([20.0, 10.0, 10.0, 20.0, 20.0, 20.0])   # peak |joint velocity| [rad/s]
MOTOR_TAU_LIMIT = np.array([27.0, 27.0, 27.0, 7.0, 7.0, 7.0])
# Just inside ctrlrange, so rail hits are visible before the bridge clamps them.
SAFETY_TAU = np.minimum(TARGET_TAU, 0.98 * MOTOR_TAU_LIMIT)
# Ceiling for the first async/real run: the synchronous sizer otherwise grows
# lightly loaded roll joints to their full range despite DDS-delay resonance.
DESIGN_AMP_MAX = np.array([0.5, 0.4, 0.4, 0.8, 0.6, 0.6])

# Amplitude [rad] and chirp top frequency [Hz], SIZED OFFLINE by
# check_excitation.py. Re-run it whenever the trajectory, gains, CENTER or model
# change, and paste the recommendation here.
# WARNING: that sizer is SYNCHRONOUS and under-predicts the async-DDS simulator
# (~1-step ZOH delay). Its first values drove the roll joints into resonance --
# 5dof hit 51 rad/s and coupled torque spikes into the held joints, violating the
# caps. Roll-joint F1 (j0,j3,j4) is therefore kept LOW. Always re-verify realized
# peaks with `check_excitation.py --analyze <npz>` before raising anything.
AMP = np.array([0.403, 0.220, 0.346, 0.695, 0.600, 0.600])
F1 = np.array([1.8, 1.2, 1.2, 1.6, 1.3, 1.3])

# PD gains used by pineapple_arm.py Controller at the time of collection. They
# are CLI-overridable and always stored in the log.
KP = np.array([20.0, 40.0, 40.0, 20.0, 20.0, 20.0])
KD = np.array([0.5, 1.0, 1.0, 0.5, 0.5, 0.5])

LOG_KEYS = (
    "t", "t_cmd", "q", "dq", "tau", "q_cmd", "dq_cmd", "tau_ff",
    "tick", "kp", "kd",
)
LOG_META_KEYS = ("schema_version", "time_source", "complete", "abort_reason")


def _range_cap(margin: float = 0.1) -> np.ndarray:
    """Largest per-joint amplitude keeping CENTER +- amp within limits (+margin)."""
    hi = JOINT_HIGH - margin - CENTER
    lo = CENTER - (JOINT_LOW + margin)
    return np.maximum(0.0, np.minimum(hi, lo))


def _smooth_envelope(k: int, dt: float, taper_s: float) -> np.ndarray:
    """Raised-cosine endpoint taper with zero position/velocity at boundaries."""
    env = np.ones(k)
    if k < 2 or taper_s <= 0:
        return env
    m = min(k // 2, max(2, int(round(taper_s / dt)) + 1))
    ramp = 0.5 - 0.5 * np.cos(np.linspace(0.0, np.pi, m))
    env[:m] = ramp
    env[-m:] = ramp[::-1]
    return env


def build_excitation(
    dt: float = DT,
    per_joint_s: float = 14.0,
    settle_s: float = 2.0,
    f0: float = 0.1,
    f1=None,
    tri_hz: float = 0.1,
    taper_s: float = 0.5,
    combined_s: float = 0.0,
    amp=None,
    combined_scale: float = 0.4,
) -> tuple[np.ndarray, np.ndarray]:
    """Build the position-command trajectory -> (t, q_cmd (N,6)).

    Settle at CENTER; then per joint a chirp + slow triangle (others held); then an
    optional all-joints multisine. Each moving segment is endpoint-tapered so it
    returns to CENTER with ZERO velocity instead of a one-step torque impulse.
    ``amp``/``f1`` override AMP/F1 (used by check_excitation.py while sizing).
    """
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError(f"dt must be positive, got {dt}")
    # NOTE: every guard below tests finiteness FIRST. A bare comparison lets NaN
    # through (`nan <= 0` is False), and a NaN here silently propagates into the
    # chirp/triangle and out to the motors.
    durations = (per_joint_s, settle_s, combined_s, taper_s)
    if not all(np.isfinite(x) for x in durations):
        raise ValueError(f"durations must be finite, got {durations}")
    if per_joint_s <= 0 or settle_s < 0 or combined_s < 0 or taper_s < 0:
        raise ValueError("durations must be nonnegative and per_joint_s must be positive")
    if not np.isfinite(f0) or f0 < 0:
        raise ValueError(f"f0 must be finite and nonnegative, got {f0}")
    if not np.isfinite(tri_hz) or tri_hz < 0:
        raise ValueError(f"tri_hz must be finite and nonnegative, got {tri_hz}")
    amp = AMP if amp is None else np.asarray(amp, float)
    if amp.shape != (NUM_MOTORS,) or not np.all(np.isfinite(amp)) or np.any(amp < 0):
        raise ValueError(f"amp must be {NUM_MOTORS} finite nonnegative values")
    amp = np.minimum(amp, _range_cap())
    if f1 is None:
        f1 = F1
    f1 = np.full(NUM_MOTORS, f1, float) if np.isscalar(f1) else np.asarray(f1, float)
    if f1.shape != (NUM_MOTORS,) or not np.all(np.isfinite(f1)) or np.any(f1 <= 0):
        raise ValueError(f"f1 must be {NUM_MOTORS} finite positive values")

    def n(sec):
        return max(1, int(round(sec / dt)))

    chunks = [np.tile(CENTER, (n(settle_s), 1))]  # initial settle

    # Per-joint excitation: chirp + slow triangle, others held at CENTER.
    for j in range(NUM_MOTORS):
        k = n(per_joint_s)
        tj = np.arange(k) * dt
        chirp = np.sin(2 * np.pi * (f0 * tj + 0.5 * (f1[j] - f0) / (tj[-1] + 1e-9) * tj**2))
        tri = 2.0 / np.pi * np.arcsin(np.sin(2 * np.pi * tri_hz * tj))  # [-1,1]
        envelope = _smooth_envelope(k, dt, taper_s)
        seg = np.tile(CENTER, (k, 1))
        seg[:, j] = CENTER[j] + amp[j] * envelope * (0.7 * chirp + 0.3 * tri)
        chunks.append(seg)
        chunks.append(np.tile(CENTER, (n(1.0), 1)))  # brief hold between joints

    # Combined multisine on all joints (phase-shifted per joint), reduced
    # amplitude so simultaneous motion still respects the per-joint torque caps.
    if combined_s > 0:
        k = n(combined_s)
        tj = np.arange(k) * dt
        envelope = _smooth_envelope(k, dt, taper_s)
        seg = np.tile(CENTER, (k, 1))
        freqs = [0.2, 0.5, 0.9, 1.5]
        for j in range(NUM_MOTORS):
            acc = np.zeros(k)
            for i, f in enumerate(freqs):
                acc += np.sin(2 * np.pi * f * tj + 0.6 * j + 1.3 * i)
            seg[:, j] = (
                CENTER[j] + envelope * combined_scale * amp[j] * acc / len(freqs)
            )
        chunks.append(seg)

    chunks.append(np.tile(CENTER, (n(settle_s), 1)))  # final settle

    q_cmd = np.vstack(chunks)
    q_cmd = np.clip(q_cmd, JOINT_LOW + 0.05, JOINT_HIGH - 0.05)  # safety clamp
    t = np.arange(len(q_cmd)) * dt
    return t, q_cmd


class ArmCollector:
    def __init__(self, kp=KP, kd=KD):
        from unitree_sdk2py.core.channel import ChannelPublisher, ChannelSubscriber
        from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowCmd_
        from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowCmd_, LowState_
        from unitree_sdk2py.utils.crc import CRC

        self._LowState_ = LowState_
        self.kp = np.asarray(kp, dtype=float)
        self.kd = np.asarray(kd, dtype=float)
        self.low_cmd = unitree_go_msg_dds__LowCmd_()
        self.low_state = None
        self.crc = CRC()
        self._state_lock = threading.Lock()

        self.qpos = np.zeros(NUM_MOTORS, dtype=np.float32)
        self.qvel = np.zeros(NUM_MOTORS, dtype=np.float32)
        self.qtau = np.zeros(NUM_MOTORS, dtype=np.float32)
        self.tick = 0          # controller/sim state timestamp (ms); 0 if source omits it
        self.t_arrival = 0.0   # perf_counter when the latest state MESSAGE arrived

        self._init_low_cmd()
        self.lowcmd_publisher = ChannelPublisher("rt/lowcmd", LowCmd_)
        self.lowcmd_publisher.Init()
        self.lowstate_subscriber = ChannelSubscriber("rt/lowstate", LowState_)
        self.lowstate_subscriber.Init(self._on_low_state, 10)

    def _init_low_cmd(self):
        self.low_cmd.head[0] = 0xFE
        self.low_cmd.head[1] = 0xEF
        self.low_cmd.level_flag = 0xFF
        self.low_cmd.gpio = 0
        for i in range(NUM_MOTORS):
            self.low_cmd.motor_cmd[i].mode = 0x01  # PMSM
            self.low_cmd.motor_cmd[i].q = 0.0
            self.low_cmd.motor_cmd[i].kp = 0.0
            self.low_cmd.motor_cmd[i].dq = 0.0
            self.low_cmd.motor_cmd[i].kd = 0.0
            self.low_cmd.motor_cmd[i].tau = 0.0

    def _on_low_state(self, msg):
        # Build the snapshot first, publish atomically: without the lock the
        # loop can mix joints/timestamps from two DDS callbacks into one row.
        qpos = np.empty(NUM_MOTORS, dtype=np.float32)
        qvel = np.empty(NUM_MOTORS, dtype=np.float32)
        qtau = np.empty(NUM_MOTORS, dtype=np.float32)
        for i in range(NUM_MOTORS):
            qpos[i] = msg.motor_state[i].q
            qvel[i] = msg.motor_state[i].dq
            qtau[i] = msg.motor_state[i].tau_est
        with self._state_lock:
            self.low_state = msg
            self.t_arrival = time.perf_counter()
            self.tick = int(getattr(msg, "tick", 0))
            self.qpos[:] = qpos
            self.qvel[:] = qvel
            self.qtau[:] = qtau

    def snapshot_state(self):
        """Return one internally consistent DDS state snapshot."""
        with self._state_lock:
            return (
                self.qpos.copy(),
                self.qvel.copy(),
                self.qtau.copy(),
                float(self.t_arrival),
                int(self.tick),
                self.low_state is not None,
            )

    def has_state(self) -> bool:
        with self._state_lock:
            return self.low_state is not None

    def wait_for_state(self, timeout=5.0):
        t0 = time.perf_counter()
        while not self.has_state():
            if time.perf_counter() - t0 > timeout:
                raise TimeoutError("No rt/lowstate received; is the arm up?")
            time.sleep(0.01)

    def _write(self, q_des, dq_des=None):
        q_des = np.asarray(q_des, dtype=float)
        dq_des = (
            np.zeros(NUM_MOTORS)
            if dq_des is None
            else np.asarray(dq_des, dtype=float)
        )
        if q_des.shape != (NUM_MOTORS,) or dq_des.shape != (NUM_MOTORS,):
            raise ValueError("q_des/dq_des must each contain six motor commands")
        # Last line of defence: np.clip does NOT remove NaN/Inf, so a non-finite
        # command would be published straight to the motors. Refuse instead.
        if not (np.all(np.isfinite(q_des)) and np.all(np.isfinite(dq_des))):
            raise ValueError(
                f"refusing to publish non-finite command: q={q_des}, dq={dq_des}"
            )
        q_des = np.clip(q_des, JOINT_LOW + 0.05, JOINT_HIGH - 0.05)
        dq_des = np.clip(dq_des, -TARGET_DQ, TARGET_DQ)
        for i in range(NUM_MOTORS):
            self.low_cmd.motor_cmd[i].q = float(q_des[i])
            self.low_cmd.motor_cmd[i].dq = float(dq_des[i])
            self.low_cmd.motor_cmd[i].kp = float(self.kp[i])
            self.low_cmd.motor_cmd[i].kd = float(self.kd[i])
            self.low_cmd.motor_cmd[i].tau = 0.0
        self.low_cmd.crc = self.crc.Crc(self.low_cmd)
        t_send = time.perf_counter()
        self.lowcmd_publisher.Write(self.low_cmd)
        return t_send

    def _limit_violations(self, qk, dqk, tauk, limit_count, trip_samples):
        """Live-limit check -> (reason or "", updated limit_count).

        Trips on a sustained soft breach (``trip_samples`` consecutive) or a single
        severe/out-of-range sample.
        """
        over = (np.abs(tauk) > SAFETY_TAU) | (np.abs(dqk) > TARGET_DQ)
        limit_count = np.where(over, limit_count + 1, 0)
        severe = (np.abs(tauk) > 1.10 * SAFETY_TAU) | (np.abs(dqk) > 1.10 * TARGET_DQ)
        out_of_range = (qk < JOINT_LOW - 0.02) | (qk > JOINT_HIGH + 0.02)
        tripped = severe | out_of_range | (limit_count >= trip_samples)
        if np.any(tripped):
            bad = np.flatnonzero(tripped)
            return ("live safety limit exceeded on "
                    + ", ".join(JOINT_NAMES[j] for j in bad)), limit_count
        return "", limit_count

    def _fresh_start_pose(self, state_timeout=0.1):
        """Measured pose from a FRESH state, else raise: ramping from a stale
        snapshot makes the interpolation start wrong and commands a step jump."""
        qk, _, _, arrival, _, valid = self.snapshot_state()
        age = time.perf_counter() - arrival
        if not valid or not np.isfinite(age) or age > state_timeout:
            raise RuntimeError(f"no fresh state to ramp from (age={age:.3f}s)")
        return qk

    def ramp_to(self, target, duration=3.0, dt=DT, watchdog=True,
                trip_samples=3, state_timeout=0.1):
        """Smoothly interpolate from the current measured pose to ``target``.

        ``watchdog=False`` still requires a FRESH starting state but will not abort
        on limit breaches; it exists solely for ``safe_return``, which must always
        be able to bring the arm down.
        """
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError(f"ramp dt must be finite and positive, got {dt}")
        if not np.isfinite(duration) or duration <= 0:
            raise ValueError(f"ramp duration must be finite and positive, got {duration}")
        start = self._fresh_start_pose(state_timeout)
        target = np.asarray(target, dtype=float)
        if target.shape != (NUM_MOTORS,) or not np.all(np.isfinite(target)):
            raise ValueError(f"ramp target must be {NUM_MOTORS} finite values")
        steps = max(1, int(duration / dt))
        limit_count = np.zeros(NUM_MOTORS, dtype=int)
        for k in range(steps):
            step_start = time.perf_counter()
            if watchdog:
                qk, dqk, tauk, arrival, _, valid = self.snapshot_state()
                age = step_start - arrival
                if not valid or not np.isfinite(age) or age > state_timeout:
                    raise RuntimeError(f"stale/missing state during ramp (age={age:.3f}s)")
                reason, limit_count = self._limit_violations(
                    qk, dqk, tauk, limit_count, trip_samples)
                if reason:
                    raise RuntimeError(reason)
            phase = (k + 1) / steps
            self._write(start * (1 - phase) + target * phase)
            sleep = dt - (time.perf_counter() - step_start)
            if sleep > 0:
                time.sleep(sleep)

    def safe_return(self, dt=DT):
        """Best-effort gentle return to zero after normal exit OR a safety abort.

        Limit violations must NOT block this: a watchdog-gated return would leave
        the arm stuck holding the offending pose. A fresh state is still required
        so the ramp does not start from stale data.
        """
        for target, dur in ((CENTER, 2.0), (np.zeros(NUM_MOTORS), 3.0)):
            try:
                self.ramp_to(target, duration=dur, dt=dt, watchdog=False)
            except (KeyboardInterrupt, RuntimeError, ValueError) as e:
                print(f"[collect] safe_return stopped: {e}")
                return

    def run(
        self,
        t,
        q_cmd,
        dt=DT,
        trip_samples=3,
        state_timeout=0.1,
    ):
        """Execute the command trajectory, logging measured + commanded signals.

        The logged ``t`` is the REAL arrival time of each state MESSAGE
        (perf_counter in the DDS callback), NOT the nominal schedule. This loop is
        not synchronized with the arm/sim's publish thread, so it sometimes reads
        the same state twice or misses one; arrival stamps let ``resample_log``
        detect both. A fabricated uniform ``t`` silently violates the fit's
        one-step-per-sample assumption and BIASES armature/damping.
        """
        # pineapple_arm.py commands zero desired velocity; matching it keeps PD
        # replay faithful and avoids a hidden kd*dq_cmd feedforward.
        dq_cmd = np.zeros_like(q_cmd)
        n = len(t)
        q = np.zeros((n, NUM_MOTORS)); dq = np.zeros((n, NUM_MOTORS))
        tau = np.zeros((n, NUM_MOTORS)); t_meas = np.zeros(n); t_cmd = np.zeros(n)
        tick = np.zeros(n, dtype=np.int64)
        limit_count = np.zeros(NUM_MOTORS, dtype=int)
        abort_reason = ""
        used = 0
        trip_samples = max(1, int(trip_samples))
        for k in range(n):
            step_start = time.perf_counter()
            qk, dqk, tauk, arrival, tickk, valid = self.snapshot_state()
            if not valid or step_start - arrival > state_timeout:
                abort_reason = (
                    f"stale/missing state: age={step_start-arrival:.3f}s "
                    f"> {state_timeout:.3f}s"
                )
                break

            t_meas[k] = arrival
            q[k] = qk; dq[k] = dqk; tau[k] = tauk
            tick[k] = tickk
            t_cmd[k] = step_start
            used = k + 1

            abort_reason, limit_count = self._limit_violations(
                qk, dqk, tauk, limit_count, trip_samples)
            if abort_reason:
                break

            t_cmd[k] = self._write(q_cmd[k], dq_cmd[k])
            sleep = dt - (time.perf_counter() - step_start)
            if sleep > 0:
                time.sleep(sleep)

        t0 = t_meas[0] if used else 0.0
        return dict(
            t=t_meas[:used] - t0,
            t_cmd=t_cmd[:used] - t0,
            q=q[:used], dq=dq[:used], tau=tau[:used],
            q_cmd=q_cmd[:used], dq_cmd=dq_cmd[:used],
            tau_ff=np.zeros_like(q_cmd[:used]), tick=tick[:used],
            kp=np.tile(self.kp, (used, 1)), kd=np.tile(self.kd, (used, 1)),
            schema_version=np.array(LOG_SCHEMA_VERSION),
            time_source=np.array(TIME_SOURCE),
            complete=np.array(not abort_reason),
            abort_reason=np.array(abort_reason),
        )


def save_log(path, log):
    import os
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    np.savez(path, joint_names=np.array(JOINT_NAMES), **log)


def _preview(t, q_cmd, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(11, 5))
    for j in range(NUM_MOTORS):
        ax.plot(t, q_cmd[:, j], label=JOINT_NAMES[j], lw=1.0)
    for lim in (JOINT_LOW, JOINT_HIGH):
        for j in range(NUM_MOTORS):
            ax.axhline(lim[j], color="0.85", lw=0.5)
    ax.set_xlabel("time [s]"); ax.set_ylabel("commanded q [rad]")
    ax.set_title("Excitation position command"); ax.legend(ncol=5, fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("net", nargs="?", default="lo",
                    help="DDS network interface (default: lo, as in pineapple_arm.py)")
    ap.add_argument("--per-joint", type=float, default=14.0,
                    help="seconds of excitation per joint")
    ap.add_argument("--combined", type=float, default=0.0,
                    help="seconds of all-joint multisine (default 0: sequential "
                         "single-joint excitation, which keeps every joint within "
                         "its torque cap and is ideal for these per-joint params)")
    ap.add_argument("--f0", type=float, default=0.1)
    ap.add_argument("--f1", type=float, default=None,
                    help="chirp top freq [Hz]; default uses per-joint F1")
    ap.add_argument("--kp", type=float, nargs=NUM_MOTORS, default=KP.tolist(),
                    metavar=("K0", "K1", "K2", "K3", "K4", "K5"),
                    help="six position gains (stored in the log)")
    ap.add_argument("--kd", type=float, nargs=NUM_MOTORS, default=KD.tolist(),
                    metavar=("D0", "D1", "D2", "D3", "D4", "D5"),
                    help="six velocity gains (stored in the log)")
    ap.add_argument("--trip-samples", type=int, default=3,
                    help="consecutive over-limit states before abort (severe hits abort immediately)")
    ap.add_argument("--state-timeout", type=float, default=0.1,
                    help="abort when the newest DDS state is older than this many seconds")
    ap.add_argument("--out", default=None, help="output .npz path")
    ap.add_argument("--dry-run", action="store_true",
                    help="build + preview the trajectory only; no DDS / hardware")
    args = ap.parse_args()
    # Finiteness FIRST: NaN passes every ordinary comparison (`nan <= 0` is False),
    # so a bare `<`/`>` is not a guard. `--state-timeout nan` would disable the
    # stale-state abort entirely, because `age > nan` is always False.
    for label, value in (("--per-joint", args.per_joint), ("--combined", args.combined),
                         ("--f0", args.f0), ("--f1", args.f1),
                         ("--state-timeout", args.state_timeout)):
        if value is not None and not np.isfinite(value):
            ap.error(f"{label} must be finite, got {value}")
    if args.per_joint <= 0:
        ap.error("--per-joint must be positive")
    if args.combined < 0:
        ap.error("--combined must be nonnegative")
    if args.f0 < 0 or (args.f1 is not None and args.f1 <= 0):
        ap.error("frequencies must be nonnegative and --f1 must be positive")
    if args.trip_samples < 1:
        ap.error("--trip-samples must be >= 1")
    if args.state_timeout <= 0:
        ap.error("--state-timeout must be positive")
    for label, values in (("--kp", args.kp), ("--kd", args.kd)):
        arr = np.asarray(values, dtype=float)
        if not np.all(np.isfinite(arr)) or np.any(arr < 0):
            ap.error(f"{label} values must be finite and nonnegative")

    t, q_cmd = build_excitation(
        per_joint_s=args.per_joint, combined_s=args.combined,
        f0=args.f0, f1=args.f1,
    )
    print(f"[collect] trajectory: {len(t)} steps, {t[-1]:.1f}s; "
          f"per-joint amplitudes [rad] = {np.round(np.minimum(AMP, _range_cap()), 3)}")

    if args.dry_run:
        prev = args.out or "data/excitation_preview.png"
        _preview(t, q_cmd, prev)
        print(f"[collect] DRY RUN -- wrote preview plot to {prev} (no hardware used)")
        return 0

    from unitree_sdk2py.core.channel import ChannelFactoryInitialize
    print("WARNING: ensure the area around the arm is clear before running.")
    input("Press Enter to start data collection...")
    ChannelFactoryInitialize(1, args.net)

    arm = None
    log = None
    interrupted = False
    try:
        arm = ArmCollector(kp=np.asarray(args.kp), kd=np.asarray(args.kd))
        arm.wait_for_state()
        print("[collect] state received; ramping to start pose...")
        arm.ramp_to(CENTER, duration=3.0, trip_samples=args.trip_samples,
                    state_timeout=args.state_timeout)
        print("[collect] running excitation...")
        log = arm.run(
            t, q_cmd,
            trip_samples=args.trip_samples,
            state_timeout=args.state_timeout,
        )
    except KeyboardInterrupt:
        print("\n[collect] interrupted -- returning to neutral.")
        interrupted = True
    finally:
        if arm is not None and arm.has_state():
            arm.safe_return()  # never blocked by limits; always brings the arm down

    if log is not None:
        out = args.out or time.strftime("data/arm_chirp_%Y%m%d_%H%M%S.npz")
        save_log(out, log)
        print(f"[collect] saved {out}  ({len(log['t'])} samples)")
        if len(log["t"]):
            tracking = np.sqrt(np.mean((log["q"] - log["q_cmd"]) ** 2, axis=0))
            print(f"[collect] RMS tracking error per joint [rad]: {np.round(tracking,4)}")
            print(f"[collect] |tau|max per joint [Nm]: {np.round(np.abs(log['tau']).max(0),2)}")
        if not bool(log["complete"]):
            print(f"[collect] ABORTED: {log['abort_reason'].item()}")
            print("[collect] partial log saved for diagnosis; sysid_fit will reject it.")
            return 2
        print("[collect] next: python check_excitation.py --analyze " + out)
        print("[collect] then: python sysid_fit.py --data " + out)
    return 130 if interrupted else 0


if __name__ == "__main__":
    raise SystemExit(main())
