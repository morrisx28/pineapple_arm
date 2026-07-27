"""Verify + calibrate gravity compensation on the REAL pineapple arm.

``pineapple_arm.py`` compensates gravity with ``arm_ff.gravity_torque(q)``, which
pinocchio computes from ``model/robot.urdf``. If the URDF link **masses / CoMs**
are wrong, that feedforward is systematically off (the arm sags or drifts when
back-driven). This tool measures the truth on hardware and calibrates a correction.

Physics: at a STATIC pose (settled, dq~=0, a=0) the measured ``tau_est`` IS the joint
gravity torque, plus Coulomb friction whose sign follows the last direction of motion.
Approaching each pose from BELOW and from ABOVE and averaging cancels that friction,
leaving ground-truth ``g_real(q)`` to compare against ``arm_ff.gravity_torque(q)``.
Half the difference is a bonus per-joint friction estimate.

Gravity is LINEAR in the inertial parameters (only mass ``m`` and first moment ``m*c``
enter when a=0), so regularized least squares on pinocchio's joint-torque regressor
recovers a mass correction.

Two phases (like the sysid tools)
---------------------------------
  collect :  python verify_gravity.py <dds_iface>          # on the robot, comp OFF
             -> writes data/grav_<timestamp>.npz
  analyze :  python verify_gravity.py --analyze <npz>      # offline, needs pinocchio
             -> per-joint error report + plot + calibrated mass/CoM (report only)

  offline: python verify_gravity.py --selftest             # no hardware, no DDS
           python verify_gravity.py --dry-run              # preview the pose sweep

Joint/motor order (i=0..5): arm_joint, arm_base_joint, upper_arm_joint,
fore_arm_joint, 5dof_joint, gripper_case_joint.
"""

from __future__ import annotations

import argparse
import os
import sys
import threading
import time

import numpy as np

# Duplicated from sysid/collect_data.py so on-robot collection needs neither MuJoCo
# nor pinocchio (gravity comp is OFF during collection). Motor order 0..5.
NUM_MOTORS = 6
DT = 0.005  # 200 Hz, matches pineapple_arm.py
LOG_SCHEMA_VERSION = 1
TIME_SOURCE = "state_arrival_perf_counter"

JOINT_NAMES = ["arm_joint", "arm_base_joint", "upper_arm_joint",
               "fore_arm_joint", "5dof_joint", "gripper_case_joint"]

# Hard joint limits from pineapple_arm.xml (radians).
JOINT_LOW = np.array([-1.5708, 0.0, -3.1416, -1.5708, -1.5708, -1.5708])
JOINT_HIGH = np.array([1.5708, 3.1416, 0.0, 1.7453, 1.5708, 1.5708])

# Neutral pose to sweep around (feasible, mildly extended).
CENTER = np.array([0.0, 0.8, -0.8, 0.3, 0.0, 0.0])

# pineapple_arm.py gains. Sag with comp OFF is expected and fine: g_model is
# predicted at the MEASURED q, not the commanded one.
KP = np.array([20.0, 40.0, 40.0, 20.0, 20.0, 20.0])
KD = np.array([0.5, 1.0, 1.0, 0.5, 0.5, 0.5])

# Live safety limits (same basis as sysid/collect_data.py).
MOTOR_TAU_LIMIT = np.array([27.0, 27.0, 27.0, 7.0, 7.0, 7.0])
SAFETY_TAU = 0.90 * MOTOR_TAU_LIMIT
DQ_LIMIT = np.array([30.0, 10.0, 10.0, 30.0, 30.0, 30.0])  # rad/s abort ceiling

# Bodies 0..5 of the pinocchio model are the six arm links; the gravity-relevant
# inertial params are mass + first moment (columns 0..3 of each body's 10-vector).
ARM_LINKS = list(range(NUM_MOTORS))
GRAV_PARAM_COLS = 4  # [m, m*cx, m*cy, m*cz]

LOG_KEYS = (
    "q_des", "q_meas", "dq_stat", "tau_std", "tau_lo", "tau_hi", "tau_meas",
    "friction_est", "static_ok",
)
LOG_META_KEYS = ("joint_names", "kp", "kd", "comp", "schema_version",
                 "time_source", "complete", "abort_reason")


def build_pose_sweep(density: int = 1) -> np.ndarray:
    """Static poses that excite gravity torque, motor order (P, 6).

    Sweeps the gravity-dominant joints (shoulder j1, elbow j2, forearm j3) one at a
    time around CENTER, plus j4 tilts for the distal CoM and a few combined poses.
    j0/j5 barely affect gravity so they stay at CENTER. All poses are clamped inside
    the joint range.
    """
    density = max(1, int(density))

    def lin(lo, hi, base_n):
        return np.linspace(lo, hi, base_n * density)

    poses = []
    # One-joint-at-a-time sweeps (others held at CENTER).
    for v in lin(0.3, 1.5, 5):      # j1 shoulder
        p = CENTER.copy(); p[1] = v; poses.append(p)
    for v in lin(-1.3, -0.3, 5):    # j2 elbow
        p = CENTER.copy(); p[2] = v; poses.append(p)
    for v in lin(-0.3, 1.0, 4):     # j3 forearm
        p = CENTER.copy(); p[3] = v; poses.append(p)
    for v in lin(-0.6, 0.6, 2):     # j4 5dof tilt (distal CoM observability)
        p = CENTER.copy(); p[4] = v; poses.append(p)
    # A few combined poses for cross-coupling observability.
    for combo in ([0.0, 1.2, -0.4, 0.8, 0.4, 0.0],
                  [0.0, 0.4, -1.2, -0.3, -0.4, 0.0],
                  [0.0, 1.0, -1.0, 1.0, 0.0, 0.0]):
        poses.append(np.array(combo))

    P = np.array(poses, dtype=float)
    return np.clip(P, JOINT_LOW + 0.05, JOINT_HIGH - 0.05)


class GravityProbe:
    """DDS controller that PD-holds static poses and logs settled torque.

    Feedforward torque is always ZERO (gravity comp OFF) so the measured ``tau_est``
    at rest equals the true holding torque.
    """

    def __init__(self, kp=KP, kd=KD, trip_samples=3, state_timeout=0.1):
        from unitree_sdk2py.core.channel import ChannelPublisher, ChannelSubscriber
        from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowCmd_
        from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowCmd_, LowState_
        from unitree_sdk2py.utils.crc import CRC

        self.kp = np.asarray(kp, float)
        self.kd = np.asarray(kd, float)
        self.trip_samples = max(1, int(trip_samples))
        self.state_timeout = float(state_timeout)
        self._trip = np.zeros(NUM_MOTORS, dtype=int)  # consecutive soft violations
        self.low_cmd = unitree_go_msg_dds__LowCmd_()
        self.crc = CRC()

        self._lock = threading.Lock()
        self.qpos = np.zeros(NUM_MOTORS, dtype=np.float32)
        self.qvel = np.zeros(NUM_MOTORS, dtype=np.float32)
        self.qtau = np.zeros(NUM_MOTORS, dtype=np.float32)
        self.t_arrival = 0.0
        self.low_state = None

        self._init_low_cmd()
        self.pub = ChannelPublisher("rt/lowcmd", LowCmd_)
        self.pub.Init()
        self.sub = ChannelSubscriber("rt/lowstate", LowState_)
        self.sub.Init(self._on_low_state, 10)

    def _init_low_cmd(self):
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

    def _on_low_state(self, msg):
        qpos = np.empty(NUM_MOTORS, dtype=np.float32)
        qvel = np.empty(NUM_MOTORS, dtype=np.float32)
        qtau = np.empty(NUM_MOTORS, dtype=np.float32)
        for i in range(NUM_MOTORS):
            qpos[i] = msg.motor_state[i].q
            qvel[i] = msg.motor_state[i].dq
            qtau[i] = msg.motor_state[i].tau_est
        with self._lock:
            self.low_state = msg
            self.t_arrival = time.perf_counter()
            self.qpos[:] = qpos
            self.qvel[:] = qvel
            self.qtau[:] = qtau

    def snapshot(self):
        with self._lock:
            return (self.qpos.copy(), self.qvel.copy(), self.qtau.copy(),
                    float(self.t_arrival), self.low_state is not None)

    def has_state(self) -> bool:
        with self._lock:
            return self.low_state is not None

    def wait_for_state(self, timeout=5.0):
        t0 = time.perf_counter()
        while not self.has_state():
            if time.perf_counter() - t0 > timeout:
                raise TimeoutError("No rt/lowstate received; is the arm up?")
            time.sleep(0.01)

    def _write(self, q_des):
        """Send one PD command with zero feedforward (comp OFF)."""
        q_des = np.asarray(q_des, float)
        # np.clip does NOT remove NaN/Inf -- refuse rather than publish it.
        if q_des.shape != (NUM_MOTORS,) or not np.all(np.isfinite(q_des)):
            raise ValueError(f"refusing to publish non-finite command: q={q_des}")
        q_des = np.clip(q_des, JOINT_LOW + 0.05, JOINT_HIGH - 0.05)
        for i in range(NUM_MOTORS):
            self.low_cmd.motor_cmd[i].q = float(q_des[i])
            self.low_cmd.motor_cmd[i].kp = float(self.kp[i])
            self.low_cmd.motor_cmd[i].dq = 0.0
            self.low_cmd.motor_cmd[i].kd = float(self.kd[i])
            self.low_cmd.motor_cmd[i].tau = 0.0  # gravity comp OFF
        self.low_cmd.crc = self.crc.Crc(self.low_cmd)
        self.pub.Write(self.low_cmd)

    def _reset_trips(self):
        self._trip[:] = 0

    def _violation_str(self, j, qk, dqk, tauk):
        name = JOINT_NAMES[j]
        if abs(tauk[j]) > SAFETY_TAU[j]:
            return f"{name} torque {abs(tauk[j]):.1f} > {SAFETY_TAU[j]:.1f} Nm"
        if abs(dqk[j]) > DQ_LIMIT[j]:
            return f"{name} velocity {abs(dqk[j]):.1f} > {DQ_LIMIT[j]:.1f} rad/s"
        return f"{name} position {qk[j]:.2f} outside range"

    def _watchdog(self, qk, dqk, tauk, arrival, valid):
        """Debounced safety check. Raise RuntimeError on a severe single sample, on
        ``trip_samples`` consecutive soft violations of one joint, or on stale state.

        A single transient (e.g. buzz-induced torque spike) no longer aborts the
        whole sweep -- only a sustained or severe breach does.
        """
        if not valid or time.perf_counter() - arrival > self.state_timeout:
            raise RuntimeError(f"stale/missing state (age > {self.state_timeout}s)")
        soft = ((np.abs(tauk) > SAFETY_TAU) | (np.abs(dqk) > DQ_LIMIT)
                | (qk < JOINT_LOW - 0.05) | (qk > JOINT_HIGH + 0.05))
        severe = ((np.abs(tauk) > 1.10 * SAFETY_TAU) | (np.abs(dqk) > 1.10 * DQ_LIMIT)
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

    def ramp_to(self, target, duration=3.0, dt=DT, watchdog=True):
        """Smoothly PD-interpolate from the current measured pose to ``target``.

        ``watchdog=False`` disables the safety check (used by ``safe_return`` so an
        abort can always bring the arm down).
        """
        self._reset_trips()
        start = self.snapshot()[0]
        target = np.asarray(target, float)
        steps = max(1, int(duration / dt))
        for k in range(steps):
            step_start = time.perf_counter()
            qk, dqk, tauk, arrival, valid = self.snapshot()
            if watchdog:
                self._watchdog(qk, dqk, tauk, arrival, valid)
            phase = (k + 1) / steps
            self._write(start * (1 - phase) + target * phase)
            sleep = dt - (time.perf_counter() - step_start)
            if sleep > 0:
                time.sleep(sleep)

    def safe_return(self, dt=DT):
        """Bring the arm down gently to zero, ignoring the watchdog (best-effort).

        Always attempted on exit/abort so the arm never stays stuck holding a pose.
        """
        try:
            self.ramp_to(CENTER, duration=3.0, dt=dt, watchdog=False)
            self.ramp_to(np.zeros(NUM_MOTORS), duration=3.0, dt=dt, watchdog=False)
        except Exception as e:  # never let recovery raise
            print(f"[collect] safe_return best-effort error: {e}")

    def hold_and_measure(self, pose, settle_s=2.5, window_s=0.8, dq_tol=0.1,
                         tau_std_tol=0.5, dt=DT):
        """Hold ``pose``, settle, then average -> (q, dq_stat, tau_std, tau, ok).

        "Static" is judged by torque STEADINESS plus a robust (median) velocity, not
        the single worst sample: small hardware buzz averages out of ``mean(tau)``
        and must not flag an otherwise-good hold.
        """
        self._reset_trips()
        # Settle at the fixed command.
        n_settle = max(1, int(settle_s / dt))
        for _ in range(n_settle):
            step_start = time.perf_counter()
            qk, dqk, tauk, arrival, valid = self.snapshot()
            self._watchdog(qk, dqk, tauk, arrival, valid)
            self._write(pose)
            sleep = dt - (time.perf_counter() - step_start)
            if sleep > 0:
                time.sleep(sleep)
        # Measurement window.
        n_win = max(5, int(window_s / dt))
        qs, dqs, taus = [], [], []
        for _ in range(n_win):
            step_start = time.perf_counter()
            qk, dqk, tauk, arrival, valid = self.snapshot()
            self._watchdog(qk, dqk, tauk, arrival, valid)
            qs.append(qk); dqs.append(dqk); taus.append(tauk)
            self._write(pose)
            sleep = dt - (time.perf_counter() - step_start)
            if sleep > 0:
                time.sleep(sleep)
        qs = np.array(qs); dqs = np.array(dqs); taus = np.array(taus)
        dq_stat = float(np.median(np.max(np.abs(dqs), axis=1)))  # median speed
        tau_std = float(np.max(np.std(taus, axis=0)))            # torque steadiness
        static_ok = dq_stat < dq_tol and tau_std < tau_std_tol
        return qs.mean(0), dq_stat, tau_std, taus.mean(0), static_ok

    def run_sweep(self, poses, approach_delta=0.1, ramp_s=3.0, **hold_kw):
        """Bidirectional static sweep. Returns a log dict (see LOG_KEYS/META)."""
        poses = np.asarray(poses, float)
        P = len(poses)
        out = {k: np.zeros((P, NUM_MOTORS)) for k in
               ("q_des", "q_meas", "tau_lo", "tau_hi", "tau_meas", "friction_est")}
        out["dq_stat"] = np.zeros(P)
        out["tau_std"] = np.zeros(P)
        out["static_ok"] = np.zeros(P, dtype=bool)
        abort_reason = ""
        done = 0
        try:
            for p, pose in enumerate(poses):
                # Approach from BELOW (joints increasing into the pose), then ABOVE.
                below = np.clip(pose - approach_delta, JOINT_LOW + 0.05, JOINT_HIGH - 0.05)
                above = np.clip(pose + approach_delta, JOINT_LOW + 0.05, JOINT_HIGH - 0.05)

                self.ramp_to(below, duration=ramp_s)
                q_lo, dqs_lo, tstd_lo, tau_lo, ok_lo = self.hold_and_measure(pose, **hold_kw)

                self.ramp_to(above, duration=ramp_s)
                q_hi, dqs_hi, tstd_hi, tau_hi, ok_hi = self.hold_and_measure(pose, **hold_kw)

                out["q_des"][p] = pose
                out["q_meas"][p] = 0.5 * (q_lo + q_hi)
                out["tau_lo"][p] = tau_lo
                out["tau_hi"][p] = tau_hi
                out["tau_meas"][p] = 0.5 * (tau_lo + tau_hi)  # friction cancels
                out["friction_est"][p] = 0.5 * (tau_hi - tau_lo)
                out["dq_stat"][p] = max(dqs_lo, dqs_hi)
                out["tau_std"][p] = max(tstd_lo, tstd_hi)
                out["static_ok"][p] = bool(ok_lo and ok_hi)
                done = p + 1
                print(f"[collect] pose {p + 1}/{P}  static={out['static_ok'][p]}  "
                      f"dq={out['dq_stat'][p]:.3f} rad/s  tau_std={out['tau_std'][p]:.2f} Nm  "
                      f"|tau|max={np.abs(out['tau_meas'][p]).max():.2f} Nm")
        except (RuntimeError, TimeoutError) as e:
            abort_reason = str(e)
            print(f"\n[collect] ABORT: {abort_reason}")

        log = {k: out[k][:done] for k in out}
        log.update(
            joint_names=np.array(JOINT_NAMES),
            kp=self.kp.copy(), kd=self.kd.copy(),
            comp=np.array("off"),
            schema_version=np.array(LOG_SCHEMA_VERSION),
            time_source=np.array(TIME_SOURCE),
            complete=np.array(not abort_reason and done == P),
            abort_reason=np.array(abort_reason),
        )
        return log


def save_log(path, log):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    np.savez(path, **log)


def load_log(path):
    with np.load(path, allow_pickle=True) as npz:
        return {k: npz[k] for k in npz.files}


def _gravity_regressor(q_arm):
    """Rows 0..5 of pinocchio's joint-torque regressor at (q, v=0, a=0), so that
    ``Y @ phi = g(q)``. Returns the full (6, 10*nbodies) matrix."""
    import pinocchio
    import arm_ik
    model, data = arm_ik.IK_MODEL, arm_ik.IK_DATA
    q = arm_ik.mj_arm_to_pin(q_arm)
    zero = np.zeros(model.nv)
    Y = pinocchio.computeJointTorqueRegressor(model, data, q, zero, zero)
    return np.asarray(Y[:NUM_MOTORS])


def _nominal_link_params():
    """Per arm link: (column indices into the full regressor, nominal gravity
    params phi0 = [m, m*cx, m*cy, m*cz]). Scaling all four by ``s`` is exactly
    scaling that link's mass by ``s`` with its CoM held fixed."""
    import arm_ik
    model = arm_ik.IK_MODEL
    per_link = []
    for b in ARM_LINKS:
        base = b * 10  # 10 standard inertial params per body
        cols = np.arange(base, base + GRAV_PARAM_COLS)
        phi0 = np.asarray(model.inertias[b + 1].toDynamicParameters()[:GRAV_PARAM_COLS], float)
        per_link.append((cols, phi0))
    return per_link


def _mass_scale_regressor(q_meas):
    """Per-link gravity-contribution regressor ``G`` (6P, 6).

    Column ``k`` is the gravity torque produced by arm link ``k`` at its NOMINAL
    inertial params. Because gravity is exactly linear in mass, scaling arm link
    ``k``'s mass by ``s_k`` (CoM fixed) changes the full gravity by
    ``G @ (s - 1)``; the fixed finger links contribute a constant already inside
    ``arm_ff.gravity_torque``. So we fit ``s`` to the residual
    ``tau_meas - g_model`` (see ``analyze``), which is far better conditioned than
    a free 4-param/link fit and targets the URDF mass magnitudes directly.
    """
    per_link = _nominal_link_params()
    P = len(q_meas)
    G = np.zeros((NUM_MOTORS * P, len(ARM_LINKS)))
    for p in range(P):
        Yp = _gravity_regressor(q_meas[p])  # (6, 80)
        for k, (cols, phi0) in enumerate(per_link):
            G[NUM_MOTORS * p:NUM_MOTORS * (p + 1), k] = Yp[:, cols] @ phi0
    return G


# --- calibration robustness thresholds ------------------------------------- #
OBS_ABS_MIN = 1.0          # min |G column| [Nm] for a link's mass to be observable
SCALE_BAND = (0.8, 1.2)    # plausible per-link mass-scale band
MIN_IMPROVE_NM = 0.05      # min absolute robust RMS drop worth calibrating for


def _detect_outliers(resid, outlier_nm):
    """Flag poses whose residual magnitude is a robust outlier (median + 5*MAD)."""
    rp = np.linalg.norm(resid, axis=1)
    med = float(np.median(rp))
    mad = 1.4826 * float(np.median(np.abs(rp - med)))
    thresh = max(outlier_nm, med + 5.0 * mad)
    return (rp > thresh), rp, thresh


def _fit_scales(G, r, reg):
    """Regularized per-link mass-scale fit -> (s_hat (6,), frozen (6,) bool).

    Links whose gravity contribution is below OBS_ABS_MIN barely move the torque, so
    their mass is unidentifiable: freeze at 1.0 rather than let noise drive them.
    """
    n = G.shape[1]
    col_norm = np.linalg.norm(G, axis=0)
    frozen = col_norm < OBS_ABS_MIN
    free = ~frozen
    s = np.ones(n)
    if np.any(free):
        Gf = G[:, free]
        GtG = Gf.T @ Gf
        lam = reg * (np.trace(GtG) / Gf.shape[1])
        delta = np.linalg.solve(GtG + lam * np.eye(Gf.shape[1]), Gf.T @ r)
        s[free] = 1.0 + delta
    return s, frozen


def _cv_improvement(q_clean, resid_clean, reg, k=4):
    """Mean held-out fractional RMS improvement (k-fold, deterministic folds).

    Guards against overfitting: a fit that only helps its own training poses -- e.g.
    one driven by a lone outlier -- yields ~0 or negative held-out improvement.
    """
    P = len(q_clean)
    if P < 4:
        return 0.0
    k = min(k, P // 2)
    idx = np.arange(P)
    fracs = []
    for f in range(k):
        test = idx[idx % k == f]
        train = idx[idx % k != f]
        if test.size == 0 or train.size < 1:
            continue
        s, _ = _fit_scales(_mass_scale_regressor(q_clean[train]),
                           resid_clean[train].ravel(), reg)
        Gte = _mass_scale_regressor(q_clean[test])
        rte = resid_clean[test].ravel()
        before = float(np.sqrt(np.mean(rte ** 2)))
        after = float(np.sqrt(np.mean((rte - Gte @ (s - 1.0)) ** 2)))
        if before > 1e-9:
            fracs.append(1.0 - after / before)
    return float(np.mean(fracs)) if fracs else 0.0


def _calibrate(q_meas, resid, tol_nm=0.6, reg=1e-2, outlier_nm=0.3,
               min_improve=0.3, force=False):
    """Robust per-link mass-scale calibration with an adequacy/generalization gate.

    Drops outlier poses, fits the clean set (observability-frozen), and decides
    whether a calibration is warranted AT ALL."""
    P = len(q_meas)
    outlier, rp, othresh = _detect_outliers(resid, outlier_nm)
    if (~outlier).sum() < max(GRAV_PARAM_COLS, 4):
        outlier = np.zeros(P, dtype=bool)  # too few clean poses -> keep all
    clean = ~outlier
    q_clean, resid_clean = q_meas[clean], resid[clean]

    per_joint_rms = np.sqrt(np.mean(resid_clean ** 2, 0))
    G = _mass_scale_regressor(q_clean)
    r = resid_clean.ravel()
    s_hat, frozen = _fit_scales(G, r, reg)
    rms_before = float(np.sqrt(np.mean(r ** 2)))
    rms_after = float(np.sqrt(np.mean((r - G @ (s_hat - 1.0)) ** 2)))
    improve_frac = 1.0 - rms_after / max(rms_before, 1e-9)
    improve_nm = rms_before - rms_after
    cv_frac = _cv_improvement(q_clean, resid_clean, reg)
    free = ~frozen
    plausible = (bool(np.all((s_hat[free] >= SCALE_BAND[0])
                             & (s_hat[free] <= SCALE_BAND[1]))) if free.any() else True)
    model_good = bool(np.all(per_joint_rms <= tol_nm))

    reasons = []
    if model_good:
        reasons.append(f"model already within tolerance (max clean RMS {per_joint_rms.max():.3f} Nm)")
    if not (improve_frac > min_improve and improve_nm > MIN_IMPROVE_NM):
        reasons.append(f"improvement not meaningful ({100*improve_frac:.0f}%, {improve_nm:.3f} Nm)")
    if cv_frac <= 0.5 * min_improve:
        reasons.append(f"does not generalize (held-out CV {100*cv_frac:.0f}%)")
    if not plausible:
        reasons.append("implausible mass scales (outside band)")
    emit = force or not reasons

    return dict(outlier=outlier, rp=rp, othresh=othresh, clean=clean, n_clean=int(clean.sum()),
                per_joint_rms=per_joint_rms, s_hat=s_hat, frozen=frozen,
                rms_before=rms_before, rms_after=rms_after, improve_frac=improve_frac,
                improve_nm=improve_nm, cv_frac=cv_frac, plausible=plausible,
                model_good=model_good, reasons=reasons, emit=emit)


def analyze(path, reg=1e-2, tol_nm=0.6, apply=False, force_calib=False,
            outlier_nm=0.3, min_improve=0.3):
    """Verify model vs measured, then calibrate mass scale.

    ``apply=True`` writes the ``model/gravity_calib.json`` overlay that ``arm_ff``
    loads, but ONLY when the fit is valid, the collection complete, and every scale
    in range.
    """
    import arm_ff

    log = load_log(path)
    names = [str(x) for x in np.asarray(log["joint_names"]).ravel()]
    if names != JOINT_NAMES:
        print(f"[analyze] ERROR: joint order mismatch: {names}")
        return 2
    complete_ok = "complete" not in log or bool(np.asarray(log["complete"]).item())
    if not complete_ok:
        reason = str(np.asarray(log.get("abort_reason", "unknown")).item())
        print(f"[analyze] WARNING: collection was incomplete/aborted: {reason}")

    q_meas = np.asarray(log["q_meas"], float)
    tau_meas = np.asarray(log["tau_meas"], float)
    static_ok = np.asarray(log["static_ok"], bool)
    P = len(q_meas)
    if P < GRAV_PARAM_COLS:
        print(f"[analyze] ERROR: need >= {GRAV_PARAM_COLS} poses, got {P}")
        return 2
    if not np.all(static_ok):
        print(f"[analyze] WARNING: {int((~static_ok).sum())}/{P} poses were not "
              "fully static; their torque may include residual motion.")

    # Model prediction at the MEASURED configuration.
    g_model = np.array([arm_ff.gravity_torque(q_meas[p]) for p in range(P)])
    resid = tau_meas - g_model

    # --- global sign / scale sanity BEFORE per-link fitting ------------------
    gm, tm = g_model.ravel(), tau_meas.ravel()
    denom = float(gm @ gm)
    alpha = float(gm @ tm / denom) if denom > 1e-9 else float("nan")
    print("\n=== Gravity-compensation verification ===")
    print(f"poses={P}  global tau_meas ~= alpha * g_model  ->  alpha = {alpha:.3f}")
    if np.isfinite(alpha) and alpha < 0:
        print("  !! alpha < 0: SIGN CONVENTION MISMATCH between tau_est and the model. "
              "Fix the sign before trusting per-link numbers.")
    elif np.isfinite(alpha) and not (0.7 <= alpha <= 1.4):
        print(f"  !! alpha far from 1 ({alpha:.2f}): likely a units/gear-ratio issue, "
              "not a per-link mass error.")

    # --- robust calibration + adequacy/generalization gate -------------------
    cal = _calibrate(q_meas, resid, tol_nm=tol_nm, reg=reg,
                     outlier_nm=outlier_nm, min_improve=min_improve, force=force_calib)
    clean = cal["clean"]

    # Outliers are excluded from the fit AND the verdict: they are almost always
    # measurement artifacts (stiction/contact/cable), not model error.
    if not np.all(clean):
        print("\n[analyze] outlier poses excluded from calibration "
              f"(|resid| > {cal['othresh']:.2f} Nm) -- investigate/re-collect:")
        for p in np.flatnonzero(~clean):
            dom = JOINT_NAMES[int(np.argmax(np.abs(resid[p])))]
            print(f"  pose {p}: |resid|={cal['rp'][p]:.2f} Nm (worst {dom}) "
                  f"q={np.round(q_meas[p], 2)}")

    # --- per-joint error report (clean poses) --------------------------------
    rc = resid[clean]
    mae = np.mean(np.abs(rc), 0); rms = cal["per_joint_rms"]
    bias = np.mean(rc, 0); worst = np.max(np.abs(rc), 0)
    print(f"\nPer-joint gravity error over {cal['n_clean']} clean poses "
          "(measured - model) [Nm]:")
    print(f"  {'joint':18s} {'MAE':>7s} {'RMS':>7s} {'bias':>7s} {'|max|':>7s}  flag")
    for j in range(NUM_MOTORS):
        flag = "OK" if rms[j] <= tol_nm else "OFF!"
        print(f"  {JOINT_NAMES[j]:18s} {mae[j]:7.3f} {rms[j]:7.3f} {bias[j]:7.3f} "
              f"{worst[j]:7.3f}  {flag}")
    fric = np.mean(np.abs(np.asarray(log["friction_est"], float)[clean]), 0)
    print(f"\nEst. Coulomb friction (|tau_hi - tau_lo|/2), mean per joint [Nm]:\n  "
          + np.array2string(np.round(fric, 3)))
    verdict = "PASS" if cal["model_good"] else "NEEDS CALIBRATION"
    print(f"\n[verify] {verdict} (per-joint RMS tolerance {tol_nm} Nm)")

    # --- calibration report -------------------------------------------------
    per_link = _nominal_link_params()
    masses0 = np.array([phi0[0] for _, phi0 in per_link])
    s_hat, frozen = cal["s_hat"], cal["frozen"]
    print("\n=== Per-link mass scale (fit on clean poses) ===")
    calib_rows = []
    for k, b_idx in enumerate(ARM_LINKS):
        m1 = masses0[k] * s_hat[k]
        print(f"  {JOINT_NAMES[b_idx]:18s} scale {s_hat[k]:5.3f}  "
              f"mass {masses0[k]:6.3f} -> {m1:6.3f} kg ({100*(s_hat[k]-1):+5.1f}%)"
              f"{'   (unobservable -> held at 1)' if frozen[k] else ''}")
        calib_rows.append(dict(joint=JOINT_NAMES[b_idx], mass_scale=float(s_hat[k]),
                               mass_nominal=float(masses0[k]), mass_identified=float(m1),
                               weakly_observable=bool(frozen[k])))
    print(f"\nresidual RMS (clean): {cal['rms_before']:.4f} -> {cal['rms_after']:.4f} Nm "
          f"({100*cal['improve_frac']:+.1f}%; held-out CV {100*cal['cv_frac']:+.1f}%)")

    # --- gate verdict + emit decision ---------------------------------------
    if cal["emit"] and not cal["reasons"]:
        print("[calib] EMIT: real, generalizable, plausible model error.")
    elif cal["emit"]:
        print("[calib] FORCED emit despite: " + "; ".join(cal["reasons"]))
    else:
        print("[calib] no calibration written -- " + "; ".join(cal["reasons"]))
        print("        (use --force-calib to override for diagnosis)")

    if cal["emit"]:
        _write_calib_report(path, calib_rows, cal["rms_before"], cal["rms_after"], reg)
    if apply:
        if cal["emit"]:
            applied = np.where(frozen, 1.0, s_hat)
            _apply_overlay(applied, True, complete_ok, path,
                           cal["rms_before"], cal["rms_after"], reg)
        else:
            print("[apply] REFUSED: calibration was withheld (see above); "
                  "use --force-calib to override.")

    _plot(path, tau_meas, g_model)
    return 0 if (cal["model_good"] or cal["emit"]) else 1


CALIB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "model", "gravity_calib.json")
# Must match arm_ff.MASS_SCALE_MIN/MAX (the overlay loader rejects anything wider).
MASS_SCALE_MIN, MASS_SCALE_MAX = 0.5, 1.5


def _apply_overlay(applied, fit_valid, complete_ok, source_npz,
                   rms_before, rms_after, reg):
    """Write model/gravity_calib.json from a validated per-link mass-scale fit."""
    import json
    if not fit_valid:
        print("[apply] REFUSED: fit is invalid; not writing an overlay.")
        return
    if not complete_ok:
        print("[apply] REFUSED: collection was incomplete/aborted; re-collect first.")
        return
    if np.any(applied < MASS_SCALE_MIN) or np.any(applied > MASS_SCALE_MAX):
        bad = ", ".join(f"{JOINT_NAMES[k]}={applied[k]:.2f}"
                        for k in np.flatnonzero((applied < MASS_SCALE_MIN) |
                                                (applied > MASS_SCALE_MAX)))
        print(f"[apply] REFUSED: mass scale outside [{MASS_SCALE_MIN},{MASS_SCALE_MAX}]: "
              f"{bad}. Suspicious data -- re-collect.")
        return
    payload = {
        "joint_names": JOINT_NAMES,
        "mass_scale": [round(float(s), 6) for s in applied],
        "residual_rms_before": round(rms_before, 6),
        "residual_rms_after": round(rms_after, 6),
        "reg": reg,
        "source_npz": os.path.basename(source_npz),
    }
    os.makedirs(os.path.dirname(CALIB_PATH), exist_ok=True)
    with open(CALIB_PATH, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\n[apply] wrote {CALIB_PATH}")
    print("[apply] applied per-link mass scales: "
          + np.array2string(np.round(applied, 3)))
    print("[apply] arm_ff (gravity comp) will use these on its NEXT start. "
          "Revert by deleting the file.")


def _write_calib_report(npz_path, rows, rms_before, rms_after, reg):
    out = os.path.splitext(npz_path)[0] + "_calib.yaml"
    lines = [
        "# Gravity-compensation calibration (report only -- NOT auto-applied).",
        "# Per-link MASS SCALE fit to measured static gravity torque (CoM held at",
        "# nominal). Gravity constrains only base parameters, so weakly_observable",
        "# links are held near 1; trust the residual-RMS drop, not each number.",
        f"# residual RMS: {rms_before:.5f} -> {rms_after:.5f} Nm   (reg={reg})",
        "#",
        "# To apply: multiply each link's <mass> in model/robot.urdf by mass_scale,",
        "# then mirror into the MuJoCo model sysid/results/latest/pineapple_arm.xml.",
        "# Re-run --analyze on a NEW collection to confirm the error dropped.",
        "links:",
    ]
    for r in rows:
        lines += [
            f"  - joint: {r['joint']}",
            f"    mass_scale: {r['mass_scale']:.4f}",
            f"    mass_nominal: {r['mass_nominal']:.6f}",
            f"    mass_identified: {r['mass_identified']:.6f}",
            f"    weakly_observable: {str(r['weakly_observable']).lower()}",
        ]
    with open(out, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[calib] wrote {out}")


def _plot(npz_path, tau_meas, g_model):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[plot] skipped: {e}")
        return
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    for j, ax in enumerate(axes.ravel()):
        gm, tm = g_model[:, j], tau_meas[:, j]
        ax.scatter(gm, tm, s=18, alpha=0.8)
        lo = min(gm.min(), tm.min()); hi = max(gm.max(), tm.max())
        pad = 0.1 * (hi - lo + 1e-6)
        ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], "k--", lw=1, label="y=x")
        ax.set_title(JOINT_NAMES[j]); ax.set_xlabel("model g [Nm]")
        ax.set_ylabel("measured [Nm]"); ax.grid(alpha=0.3); ax.legend(fontsize=7)
    fig.suptitle("Gravity comp: measured static torque vs model prediction")
    fig.tight_layout()
    out = os.path.splitext(npz_path)[0] + "_verify.png"
    fig.savefig(out, dpi=110); plt.close(fig)
    print(f"[plot] wrote {out}")


class _FakeProbe(GravityProbe):
    """GravityProbe with the DDS layer stubbed out -- feeds canned state frames to
    the real hold_and_measure / watchdog logic (no hardware, no unitree_sdk2py)."""

    def __init__(self, frames, trip_samples=3):
        self.frames = frames
        self.i = 0
        self.trip_samples = trip_samples
        self.state_timeout = 1e9  # never "stale" in the test
        self._trip = np.zeros(NUM_MOTORS, dtype=int)
        self.kp = KP.copy()
        self.kd = KD.copy()

    def snapshot(self):
        q, dq, tau = self.frames[min(self.i, len(self.frames) - 1)]
        self.i += 1
        return (np.asarray(q, float), np.asarray(dq, float),
                np.asarray(tau, float), time.perf_counter(), True)

    def _write(self, pose):
        pass


def _collection_selftest() -> bool:
    """Offline checks for static gating + debounced watchdog (no hardware)."""
    q0 = np.zeros(NUM_MOTORS)
    g = np.array([0.0, 3.0, -5.0, -1.0, 0.0, 0.0])  # a steady "gravity" torque
    hold_kw = dict(settle_s=0.0, window_s=0.03, dt=DT, dq_tol=0.1, tau_std_tol=0.5)
    ok = True

    # 1) steady with a single velocity spike -> static (median rejects the spike).
    lo = (q0, np.full(NUM_MOTORS, 0.03), g)
    spike = (q0, np.full(NUM_MOTORS, 0.5), g)
    _, _, _, tau_mean, static = _FakeProbe([lo, lo, lo, lo, lo, spike]).hold_and_measure(q0, **hold_kw)
    ok = ok and static and np.allclose(tau_mean, g, atol=0.2)

    # 2) genuinely moving -> not static.
    mv = (q0, np.full(NUM_MOTORS, 0.3), g)
    _, _, _, _, static = _FakeProbe([mv] * 6).hold_and_measure(q0, **hold_kw)
    ok = ok and not static

    # 3) one severe torque sample -> instant abort.
    sev_tau = g.copy(); sev_tau[4] = 8.0  # > 1.1*SAFETY_TAU[4] (6.93)
    try:
        _FakeProbe([q0])._watchdog(q0, q0, sev_tau, time.perf_counter(), True)
        ok = False  # should have raised
    except RuntimeError as e:
        ok = ok and "severe" in str(e)

    # 4) trip_samples consecutive soft violations -> abort naming the joint+limit.
    fp = _FakeProbe([q0], trip_samples=3)
    soft_tau = g.copy(); soft_tau[4] = 6.5  # > SAFETY_TAU[4]=6.3, < severe
    raised = None
    for k in range(3):
        try:
            fp._watchdog(q0, q0, soft_tau, time.perf_counter(), True)
        except RuntimeError as e:
            raised = str(e); break
    ok = ok and raised is not None and "5dof_joint" in raised and "3 consec" in raised

    # 5) two soft then clear -> counter resets, no abort.
    fp = _FakeProbe([q0], trip_samples=3)
    fp._watchdog(q0, q0, soft_tau, time.perf_counter(), True)  # 1
    fp._watchdog(q0, q0, soft_tau, time.perf_counter(), True)  # 2
    fp._watchdog(q0, q0, g, time.perf_counter(), True)         # clear -> reset
    try:
        fp._watchdog(q0, q0, soft_tau, time.perf_counter(), True)  # 1 again, no abort
        ok = ok and True
    except RuntimeError:
        ok = False

    print(f"[selftest] collection watchdog/static checks: {'PASS' if ok else 'FAIL'}")
    return ok


def _gate_selftest() -> bool:
    """Offline checks for the calibration adequacy/generalization gate (no hardware)."""
    import arm_ff
    poses = build_pose_sweep(density=2)
    P = len(poses)
    g = np.array([arm_ff.gravity_torque(poses[p]) for p in range(P)])
    G = _mass_scale_regressor(poses)
    ok = True

    # Case A: genuine broad, observable, plausible error -> gate EMITS + recovers.
    # tol_nm=0.1 so the (band-capped) 1.2x error counts as out-of-tolerance.
    s_true = np.ones(len(ARM_LINKS)); s_true[2] = 1.2  # upper_arm heavier
    tauA = g + (G @ (s_true - 1.0)).reshape(P, NUM_MOTORS)
    calA = _calibrate(poses, tauA - g, tol_nm=0.1)
    okA = calA["emit"] and not calA["model_good"] and abs(calA["s_hat"][2] - 1.2) < 0.06
    print(f"[selftest] gate A (real error): emit={calA['emit']} "
          f"s2={calA['s_hat'][2]:.3f} cv={100*calA['cv_frac']:.0f}%  {'ok' if okA else 'FAIL'}")

    # Case B: already-good model + one outlier pose -> gate WITHHOLDS + flags it.
    tauB = g.copy(); tauB[5, 2] += 2.5  # artifact at pose 5 on upper_arm
    calB = _calibrate(poses, tauB - g, tol_nm=0.6)
    okB = (not calB["emit"]) and calB["model_good"] and bool(calB["outlier"][5])
    print(f"[selftest] gate B (outlier): emit={calB['emit']} "
          f"outlier@5={bool(calB['outlier'][5])} model_good={calB['model_good']}  "
          f"{'ok' if okB else 'FAIL'}")

    ok = okA and okB
    print(f"[selftest] calibration gate checks: {'PASS' if ok else 'FAIL'}")
    return ok


def selftest(perturb_link=2, mass_scale=1.3, reg=1e-2):
    """No hardware: fabricate a log with link ``perturb_link`` made heavier, then
    check the calibrated model reproduces the true gravity on HELD-OUT poses.

    Held-out reproduction is the honest PASS metric, not exact per-link mass
    recovery: gravity constrains base parameters, so one link's mass is only
    partially separable -- and reproduction is what the feedforward needs.
    """
    import arm_ff

    train = build_pose_sweep(density=2)
    # Held-out check poses (different from training) via a shifted center.
    check = np.clip(build_pose_sweep(density=1) + np.array([0, 0.1, 0.1, 0.1, 0.1, 0]),
                    JOINT_LOW + 0.05, JOINT_HIGH - 0.05)
    s_true = np.ones(len(ARM_LINKS)); s_true[perturb_link] = mass_scale

    # Physically consistent full-model gravity with the perturbed link:
    # g(q; s) = g_model(q) + G_arm(q) @ (s - 1)   (exact, gravity is linear in mass).
    G_train = _mass_scale_regressor(train)
    g_train = np.array([arm_ff.gravity_torque(train[p]) for p in range(len(train))])
    tau = g_train + (G_train @ (s_true - 1.0)).reshape(len(train), NUM_MOTORS)
    log = dict(
        q_des=train, q_meas=train,
        tau_lo=tau, tau_hi=tau, tau_meas=tau,
        friction_est=np.zeros_like(tau),
        dq_max=np.zeros(len(train)), static_ok=np.ones(len(train), dtype=bool),
        joint_names=np.array(JOINT_NAMES), kp=KP, kd=KD, comp=np.array("off"),
        schema_version=np.array(LOG_SCHEMA_VERSION),
        time_source=np.array("synthetic"),
        complete=np.array(True), abort_reason=np.array(""),
    )
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "data", "grav_selftest.npz")
    save_log(path, log)
    print(f"[selftest] synthetic log: link '{JOINT_NAMES[perturb_link]}' mass x{mass_scale}")
    analyze(path, reg=reg)

    # Independent fit (residual r = tau - g_model, toward delta = s - 1).
    r = (tau - g_train).ravel()
    GtG = G_train.T @ G_train
    lam = reg * (np.trace(GtG) / len(ARM_LINKS))
    delta = np.linalg.solve(GtG + lam * np.eye(len(ARM_LINKS)), G_train.T @ r)
    s_hat = 1.0 + delta
    G_check = _mass_scale_regressor(check)
    err = np.abs(G_check @ (s_hat - s_true))
    print(f"\n[selftest] s_hat = {np.round(s_hat, 3)}  (true {np.round(s_true, 3)})")
    print(f"[selftest] held-out gravity reproduction: max err {err.max():.4f} Nm, "
          f"RMS {np.sqrt(np.mean(err**2)):.4f} Nm")
    ok_fit = err.max() < 0.05

    # --- overlay round-trip: write -> arm_ff.load -> apply -> gravity_torque --
    # Exercises the real overlay IO + inertia scaling (via a TEMP file, so the
    # canonical model/gravity_calib.json is never touched by the test).
    import json
    import tempfile
    import pinocchio
    import arm_ik
    scales = np.clip(s_hat, MASS_SCALE_MIN, MASS_SCALE_MAX)
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as tf:
        json.dump({"joint_names": JOINT_NAMES,
                   "mass_scale": [float(x) for x in scales]}, tf)
        tmp = tf.name
    loaded = arm_ff.load_gravity_calibration(tmp)
    os.unlink(tmp)
    gmodel = pinocchio.buildModelFromUrdf(arm_ik.URDF_FILENAME)
    arm_ff._apply_mass_scales(gmodel, loaded)
    gdata = gmodel.createData()
    g_overlay = np.array([
        np.asarray(pinocchio.computeGeneralizedGravity(
            gmodel, gdata, arm_ik.mj_arm_to_pin(check[p]))[:NUM_MOTORS])
        for p in range(len(check))])
    # True full-model gravity with the perturbed link, same linear relation.
    g_check = np.array([arm_ff.gravity_torque(check[p]) for p in range(len(check))])
    true_full = g_check + (G_check @ (s_true - 1.0)).reshape(len(check), NUM_MOTORS)
    rt_err = np.abs(g_overlay - true_full)
    ok_rt = loaded is not None and rt_err.max() < 0.05
    print(f"[selftest] overlay round-trip: max err {rt_err.max():.4f} Nm  "
          f"(reload+apply {'OK' if loaded is not None else 'FAILED'})")

    # Fail-safe: missing / malformed / out-of-range overlays must be ignored.
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as bf:
        bf.write("{not valid json")
        bad_path = bf.name
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as of:
        json.dump({"joint_names": JOINT_NAMES,
                   "mass_scale": [5.0] * NUM_MOTORS}, of)  # outside [0.5,1.5]
        oor_path = of.name
    ok_safe = (arm_ff.load_gravity_calibration("/no/such/file.json") is None
               and arm_ff.load_gravity_calibration(bad_path) is None
               and arm_ff.load_gravity_calibration(oor_path) is None)
    os.unlink(bad_path); os.unlink(oor_path)

    ok_collect = _collection_selftest()
    ok_gate = _gate_selftest()
    ok = ok_fit and ok_rt and ok_safe and ok_collect and ok_gate
    print(f"[selftest] {'PASS' if ok else 'FAIL'} "
          "(fit reproduction + overlay round-trip + fail-safe + collection watchdog + calib gate)")
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser(description="Verify + calibrate arm gravity comp.")
    ap.add_argument("net", nargs="?", default="lo",
                    help="DDS network interface for on-robot collection (e.g. eth0)")
    ap.add_argument("--analyze", metavar="NPZ",
                    help="offline: analyze + calibrate a collected .npz")
    ap.add_argument("--apply", action="store_true",
                    help="with --analyze: write model/gravity_calib.json so arm_ff's "
                         "gravity comp uses the calibrated masses (only from a valid fit)")
    ap.add_argument("--selftest", action="store_true",
                    help="offline: synthetic mass-perturbation recovery test (no DDS)")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the pose sweep and exit (no DDS)")
    ap.add_argument("--density", type=int, default=1, help="poses-per-joint multiplier")
    ap.add_argument("--reg", type=float, default=1e-2,
                    help="Tikhonov weight toward URDF nominal params")
    ap.add_argument("--tol-nm", type=float, default=0.6,
                    help="per-joint RMS error [Nm] to PASS verification")
    ap.add_argument("--outlier-nm", type=float, default=0.3,
                    help="absolute residual floor [Nm] above the median+MAD to flag a pose as outlier")
    ap.add_argument("--min-improve", type=float, default=0.3,
                    help="min fractional robust RMS improvement to emit a calibration")
    ap.add_argument("--force-calib", action="store_true",
                    help="emit a calibration even if the adequacy/generalization gate withholds it")
    # --- collection tuning (on-robot) ---
    ap.add_argument("--dq-tol", type=float, default=0.1,
                    help="median joint speed [rad/s] under which a hold counts as static")
    ap.add_argument("--tau-std-tol", type=float, default=0.5,
                    help="max per-joint torque std [Nm] for a hold to count as static")
    ap.add_argument("--settle", type=float, default=2.5, help="settle time per hold [s]")
    ap.add_argument("--window", type=float, default=0.8, help="averaging window per hold [s]")
    ap.add_argument("--trip-samples", type=int, default=3,
                    help="consecutive soft over-limit samples before abort (severe aborts instantly)")
    ap.add_argument("--approach-delta", type=float, default=0.1,
                    help="rad offset for the below/above bidirectional approach")
    ap.add_argument("--kp", type=float, nargs=NUM_MOTORS, default=KP.tolist(),
                    metavar=tuple(f"K{i}" for i in range(NUM_MOTORS)),
                    help="six position gains for the holds (stored in the log)")
    ap.add_argument("--kd", type=float, nargs=NUM_MOTORS, default=KD.tolist(),
                    metavar=tuple(f"D{i}" for i in range(NUM_MOTORS)),
                    help="six velocity gains; raise to damp comp-OFF buzz")
    ap.add_argument("--out", default=None, help="output .npz path for collection")
    args = ap.parse_args()

    if args.reg < 0:
        ap.error("--reg must be nonnegative")
    if args.trip_samples < 1:
        ap.error("--trip-samples must be >= 1")
    if args.settle < 0 or args.window <= 0:
        ap.error("--settle must be >=0 and --window > 0")

    if args.apply and not args.analyze:
        ap.error("--apply requires --analyze <npz>")

    if args.selftest:
        return selftest(reg=args.reg)
    if args.analyze:
        return analyze(args.analyze, reg=args.reg, tol_nm=args.tol_nm, apply=args.apply,
                       force_calib=args.force_calib, outlier_nm=args.outlier_nm,
                       min_improve=args.min_improve)

    poses = build_pose_sweep(density=args.density)
    if args.dry_run:
        print(f"[dry-run] {len(poses)} poses (motor order j0..j5), rad:")
        for p, pose in enumerate(poses):
            print(f"  {p:2d}: " + np.array2string(np.round(pose, 3)))
        return 0

    # --- on-robot collection -------------------------------------------------
    from unitree_sdk2py.core.channel import ChannelFactoryInitialize
    print("WARNING: keep the workspace around the arm clear. Gravity comp is OFF; "
          "the arm is held by PD only and may sag at each pose.")
    input("Press Enter to start gravity verification...")
    ChannelFactoryInitialize(1, args.net)

    probe = None
    log = None
    try:
        probe = GravityProbe(kp=np.asarray(args.kp), kd=np.asarray(args.kd),
                             trip_samples=args.trip_samples)
        probe.wait_for_state()
        print(f"[collect] state received; sweeping {len(poses)} poses "
              "(bidirectional, comp OFF)...")
        log = probe.run_sweep(
            poses, approach_delta=args.approach_delta,
            settle_s=args.settle, window_s=args.window,
            dq_tol=args.dq_tol, tau_std_tol=args.tau_std_tol,
        )
    except KeyboardInterrupt:
        print("\n[collect] interrupted -- returning to neutral.")
    finally:
        if probe is not None and probe.has_state():
            probe.safe_return()  # watchdog-free; always brings the arm down

    if log is not None and len(log["q_meas"]):
        out = args.out or time.strftime("data/grav_%Y%m%d_%H%M%S.npz")
        save_log(out, log)
        print(f"[collect] saved {out}  ({len(log['q_meas'])} poses)")
        if not bool(log["complete"]):
            print(f"[collect] INCOMPLETE: {str(log['abort_reason'].item())}")
            print("[collect] partial log saved for diagnosis.")
            return 2
        print(f"[collect] next: python verify_gravity.py --analyze {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
