"""Collect and analyze gravity-compensation measurements for the real arm.

Each static pose is approached from both directions so averaging cancels Coulomb
friction. Analysis compares the resulting gravity torque with the Pinocchio model and
fits physically gated inertial or per-joint corrections. Collection always runs with
gravity compensation disabled.
"""

from __future__ import annotations

import argparse
import os
import sys
import threading
import time

import numpy as np

# Keep on-robot collection independent of MuJoCo and Pinocchio.
NUM_MOTORS = 6
DT = 0.005  # 200 Hz, matches pineapple_arm.py
LOG_SCHEMA_VERSION = 1
TIME_SOURCE = "state_arrival_perf_counter"

JOINT_NAMES = ["arm_joint", "arm_base_joint", "upper_arm_joint",
               "fore_arm_joint", "5dof_joint", "gripper_case_joint"]

# Hard joint limits from pineapple_arm.xml (radians).
JOINT_LOW = np.array([-1.5708, 0.0, -3.1416, -1.5708, -1.5708, -1.5708])
JOINT_HIGH = np.array([1.5708, 3.1416, 0.0, 1.7453, 1.5708, 1.5708])

CENTER = np.array([0.0, 0.8, -0.8, 0.3, 0.0, 0.0])

# Predict gravity at measured q, so uncompensated sag does not bias the model input.
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
    # Isolate each joint's gravity contribution.
    for v in lin(0.3, 1.5, 5):      # j1 shoulder
        p = CENTER.copy(); p[1] = v; poses.append(p)
    for v in lin(-1.3, -0.3, 5):    # j2 elbow
        p = CENTER.copy(); p[2] = v; poses.append(p)
    for v in lin(-0.3, 1.0, 4):     # j3 forearm
        p = CENTER.copy(); p[3] = v; poses.append(p)
    for v in lin(-0.6, 0.6, 2):     # j4 5dof tilt (distal CoM observability)
        p = CENTER.copy(); p[4] = v; poses.append(p)
    # Include cross-coupling observability.
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
        n_settle = max(1, int(settle_s / dt))
        for _ in range(n_settle):
            step_start = time.perf_counter()
            qk, dqk, tauk, arrival, valid = self.snapshot()
            self._watchdog(qk, dqk, tauk, arrival, valid)
            self._write(pose)
            sleep = dt - (time.perf_counter() - step_start)
            if sleep > 0:
                time.sleep(sleep)
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


# Calibration robustness thresholds
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

    # Gravity must be evaluated at the measured configuration.
    g_model = np.array([arm_ff.gravity_torque(q_meas[p]) for p in range(P)])
    resid = tau_meas - g_model

    # Check global sign and scale before per-link fitting.
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

    # Robust calibration and generalization gate.
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

    # Per-joint error on clean poses.
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

    # Calibration report.
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

    # Gate verdict and emit decision.
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


GRAV_SIGNAL_MIN = 0.30      # Nm; below this a joint carries too little gravity to judge
RATIO_OK = (0.85, 1.15)     # tau/sag ratio band counted as "matches the model"


def _joint_metrics(log, model_g=None):
    """Per-joint metrics for one log. Needs arm_ff (pinocchio) for the model.

    Returns a dict of (P,6)/(6,) arrays. The KEY pair is ``tau_ratio`` and
    ``sag_ratio``: they are independent channels (torque sensor vs encoder), so
    comparing them separates "the motor lied" from "the motor did not deliver".
    """
    import arm_ff
    q = np.asarray(log["q_meas"], float)
    tau = np.asarray(log["tau_meas"], float)
    qd = np.asarray(log["q_des"], float) if "q_des" in log else None
    kp = np.asarray(log["kp"], float) if "kp" in log else KP
    lo = np.asarray(log["tau_lo"], float) if "tau_lo" in log else None
    hi = np.asarray(log["tau_hi"], float) if "tau_hi" in log else None
    P = len(q)
    g = model_g if model_g is not None else np.array(
        [arm_ff.gravity_torque(q[p]) for p in range(P)])

    m = dict(q=q, qd=qd, tau=tau, g=g, kp=kp, P=P,
             tau_ratio=np.full(NUM_MOTORS, np.nan),
             sag_ratio=np.full(NUM_MOTORS, np.nan),
             kp_factor=np.full(NUM_MOTORS, np.nan),
             corr=np.full(NUM_MOTORS, np.nan),
             rms=np.sqrt(np.mean((tau - g) ** 2, axis=0)),
             gmax=np.abs(g).max(axis=0),
             fric=(np.abs((hi - lo) / 2).mean(axis=0) if lo is not None
                   else np.full(NUM_MOTORS, np.nan)))
    # At static equilibrium with comp OFF: kp*(q_des - q_meas) = g(q_meas),
    # so sag = q_meas - q_des should equal -g/kp using ONLY encoder data.
    m["sag"] = (q - qd) if qd is not None else None
    m["sag_pred"] = -g / kp
    for j in range(NUM_MOTORS):
        gj, tj = g[:, j], tau[:, j]
        if m["gmax"][j] < GRAV_SIGNAL_MIN:
            continue
        big = np.abs(gj) > 0.5 * m["gmax"][j]      # judge where gravity is strong
        m["tau_ratio"][j] = float(np.median(tj[big] / gj[big]))
        if gj.std() > 1e-9 and tj.std() > 1e-9:
            m["corr"][j] = float(np.corrcoef(gj, tj)[0, 1])
        if m["sag"] is not None:
            s, sp = m["sag"][big, j], m["sag_pred"][big, j]
            if np.abs(sp).mean() > 1e-4:
                m["sag_ratio"][j] = float(s.mean() / sp.mean())
            ok = np.abs(m["sag"][:, j]) > 2e-3
            if ok.any():
                m["kp_factor"][j] = float(
                    np.median(np.abs(gj[ok] / m["sag"][ok, j])) / kp[j])
    return m


def _diagnose(tau_ratio, sag_ratio, fric):
    """Turn the two independent ratios into a cause, not just a number.

    torque sensor (tau_ratio) and encoder (sag_ratio) are separate channels:
      both ~1                 -> consistent
      torque low, sag ~1      -> motor delivered, SENSOR under-reports
      torque low, sag low     -> motor did not deliver: something else holds the
                                 load (large friction) or BOTH channels are
                                 mis-scaled (gear/unit config)
    """
    if not np.isfinite(tau_ratio):
        return "no gravity signal"
    tlo, thi = RATIO_OK
    t_ok = tlo <= tau_ratio <= thi
    if not np.isfinite(sag_ratio):
        return "OK" if t_ok else f"torque x{tau_ratio:.2f} (no sag data)"
    s_ok = tlo <= sag_ratio <= thi
    if t_ok and s_ok:
        return "OK"
    if t_ok and not s_ok:
        return f"stiffness off (sag x{sag_ratio:.2f}, torque OK)"
    if s_ok and not t_ok:
        return f"TORQUE SENSOR under-reports (x{tau_ratio:.2f}, sag OK)"
    if np.isfinite(fric) and fric > 0.3 * abs(1 - tau_ratio) * 5:
        return f"held by FRICTION (fric {fric:.2f} Nm)"
    return (f"BOTH channels mis-scaled (torque x{tau_ratio:.2f}, sag x{sag_ratio:.2f})"
            " -> suspect gear/unit config")


FIT_R2_MIN = 0.90          # below this the factor is not trustworthy
FIT_MIN_POSES = 4          # fewer usable poses than this -> low confidence
FIT_REL_FLOOR = 0.15       # keep poses with |g| above this fraction of the joint's peak
FIT_SLOPE_AGREE = 0.10     # through-origin vs offset-corrected slope must agree within this
FIT_MIN_PER_SIDE = 2       # poses needed on EACH sign before an offset is identifiable


def _ls_through_origin(x, y):
    """Least-squares slope k of ``y ~= k*x`` through the origin, plus R^2.

    LS (not a median of y/x ratios) so a residual -- hence a confidence -- comes out
    of the same fit. R^2 is measured about zero, matching the no-intercept model.
    """
    x = np.asarray(x, float).ravel(); y = np.asarray(y, float).ravel()
    den = float(x @ x)
    if den < 1e-12 or len(x) < 2:
        return np.nan, np.nan
    k = float(x @ y / den)
    ss_res = float(np.sum((y - k * x) ** 2))
    ss_tot = float(np.sum(y ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else np.nan
    return k, r2


def _ls_with_intercept(x, y):
    """Least-squares ``y ~= k*x + b``, returning (k, b, R^2 about the mean).

    Only meaningful when x spans BOTH signs: on one-sided data k and b are collinear,
    so the fit happily trades slope for offset (measured: a 0.99 slope collapsing to
    0.69 while R^2 dropped). Callers must gate on identifiability first.
    """
    x = np.asarray(x, float).ravel(); y = np.asarray(y, float).ravel()
    if len(x) < 3:
        return np.nan, np.nan, np.nan
    A = np.column_stack([x, np.ones_like(x)])
    try:
        c, *_ = np.linalg.lstsq(A, y, rcond=None)
    except np.linalg.LinAlgError:
        return np.nan, np.nan, np.nan
    ss_res = float(np.sum((y - A @ c) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else np.nan
    return float(c[0]), float(c[1]), r2


def _identifiable(v):
    """True when v spans both signs with enough leverage to separate slope from offset."""
    v = np.asarray(v, float).ravel()
    return bool(min(int((v > 0).sum()), int((v < 0).sum())) >= FIT_MIN_PER_SIDE)


def compute_joint_scales(log):
    """Per-joint torque and stiffness scale factors from a loaded gravity log.

    Pure: no printing, no plotting, no file IO -- so other tools (``arm_drag.py``)
    can consume the identified numbers instead of scraping :func:`fit_scale`'s table.

    Two INDEPENDENT channels are fitted per joint:
      * torque   : N_tau such that N_tau * tau_meas ~= g_model  (torque sensor)
      * stiffness: S = K/kp where g ~= K * sag                  (encoder only)
    Their ratio ``S / N_tau`` isolates a unit error from a mechanical reduction: a
    pure gear ratio N moves BOTH channels together, so anything left over in the
    ratio (e.g. ~pi) lives in the position/gain path alone.

    Returns a dict with ``rows`` (one entry per OBSERVABLE joint -- joints whose
    gravity span is below GRAV_SIGNAL_MIN are ABSENT, so consumers must treat a
    missing joint as "not identified") plus the shared ``g``/``tau``/``sag``/``gmax``
    arrays the printer and plotter need. ``row["good"]`` is the trust flag.
    """
    import arm_ff

    q = np.asarray(log["q_meas"], float)
    tau = np.asarray(log["tau_meas"], float)
    qd = np.asarray(log["q_des"], float) if "q_des" in log else None
    kp = np.asarray(log["kp"], float) if "kp" in log else KP
    P = len(q)
    g = np.array([arm_ff.gravity_torque(q[p]) for p in range(P)])
    sag = (q - qd) if qd is not None else None
    gmax_all = np.abs(g).max(axis=0)

    rows = []
    for j, nm in enumerate(JOINT_NAMES):
        gmax = float(gmax_all[j])
        if gmax < GRAV_SIGNAL_MIN:
            continue
        # Fit where gravity is strong: near g=0 both ratios are noise-dominated. The cut
        # is relative (leverage) OR'd with an absolute noise floor, so a joint whose
        # torque genuinely sweeps through zero keeps its well-loaded poses on both sides
        # instead of being cut down to the few nearest its peak.
        big = np.abs(g[:, j]) > max(GRAV_SIGNAL_MIN, FIT_REL_FLOOR * gmax)
        n_t = int(big.sum())
        N_tau, r2_t = _ls_through_origin(tau[big, j], g[big, j])
        # A constant offset (sensor bias, model zero error) is identifiable ONLY when the
        # data spans both signs -- and then correcting it is what makes the slope
        # trustworthy. On one-sided data the through-origin slope silently absorbs the
        # offset, which is the genuinely risky case.
        ident = _identifiable(g[big, j])
        k_int, off, r2_int = (_ls_with_intercept(tau[big, j], g[big, j]) if ident
                              else (np.nan, np.nan, np.nan))
        shift = (abs(k_int - N_tau) / abs(k_int)
                 if ident and np.isfinite(k_int) and abs(k_int) > 1e-9 else np.nan)
        if ident and np.isfinite(k_int):
            N_tau = k_int          # offset-corrected slope is the better estimate

        S = r2_s = lin_s = np.nan; n_s = 0
        if sag is not None:
            use = big & (np.abs(sag[:, j]) > 2e-3)
            n_s = int(use.sum())
            if n_s >= 2:
                K, r2_s = _ls_through_origin(sag[use, j], g[use, j])
                # Equilibrium is kp*(q_des-q_meas) = g, i.e. g = -kp_eff*sag, so the
                # fitted slope is NEGATIVE kp_eff. Flip it so S=1 means "as commanded".
                S = -K / kp[j] if kp[j] > 1e-9 else np.nan
                # R^2 through the origin is INFLATED when y has a large mean and small
                # variance -- a nearly constant sag still scores ~0.99. The centered
                # correlation says whether sag genuinely tracks g, or whether S is
                # only an average ratio over a sag that barely moves.
                sj, gj2 = sag[use, j], g[use, j]
                if n_s >= 3 and sj.std() > 1e-12 and gj2.std() > 1e-12:
                    lin_s = float(np.corrcoef(sj, gj2)[0, 1])
                if _identifiable(gj2):
                    Ki, _, _ = _ls_with_intercept(sj, gj2)
                    if np.isfinite(Ki) and kp[j] > 1e-9:
                        S = -Ki / kp[j]
        ratio = S / N_tau if (np.isfinite(S) and np.isfinite(N_tau)
                              and abs(N_tau) > 1e-9) else np.nan
        # Confidence: enough poses, a good fit, and -- when the offset is identifiable --
        # correcting it must barely move the slope. A large move is a real warning.
        r2_use = r2_int if ident and np.isfinite(r2_int) else r2_t
        good = bool(np.isfinite(r2_use) and r2_use >= FIT_R2_MIN and n_t >= FIT_MIN_POSES
                    and (not ident or (np.isfinite(shift) and shift < FIT_SLOPE_AGREE)))
        if not good:
            note = "LOW CONFIDENCE"
        elif ident:
            note = f"ok (offset {off:+.2f}Nm identified)"
        else:
            # Not a defect in the data, but the user must know the slope is unverified.
            note = "ok (one-sided load: offset NOT identifiable)"
        if np.isfinite(lin_s) and abs(lin_s) < 0.5:
            note += "; S=avg ratio only (sag ~constant)"
        s_pi = S / np.pi if np.isfinite(S) else np.nan
        Nint = int(round(s_pi)) if np.isfinite(s_pi) else 0
        rows.append(dict(j=j, name=nm, N_tau=N_tau, r2_t=r2_t, r2_use=r2_use, n_t=n_t,
                         S=S, r2_s=r2_s, lin_s=lin_s, n_s=n_s, ratio=ratio, s_pi=s_pi,
                         Nint=Nint, note=note, good=good, big=big, gmax=gmax, off=off,
                         ident=ident))
    return dict(rows=rows, g=g, tau=tau, sag=sag, gmax=gmax_all, kp=kp, P=P)


def fit_scale(path, out=None):
    """Report the per-joint torque/stiffness scale fit for one log. -> 0 or 2.

    Fitting itself lives in :func:`compute_joint_scales`; this is the CLI view of it
    (table, suggested constants, ``<npz>_scale.png``).
    """
    try:
        import arm_ff  # noqa: F401  -- model needed for g(q)
    except Exception as e:
        print(f"[fit] ERROR: needs pinocchio/arm_ff for the model: {e}")
        return 2
    try:
        log = load_log(path)
    except Exception as e:
        print(f"[fit] ERROR: cannot read {path}: {e}")
        return 2
    if "tau_meas" not in log or len(np.asarray(log["tau_meas"])) == 0:
        print(f"[fit] ERROR: {path} contains no poses")
        return 2

    fit = compute_joint_scales(log)
    rows, g, tau, sag = fit["rows"], fit["g"], fit["tau"], fit["sag"]
    by_j = {r["j"]: r for r in rows}

    print(f"\n=== per-joint scale fit: {os.path.basename(path)} ({fit['P']} poses) ===")
    print(f"  {'joint':18s} {'tau x':>7s} {'R2':>6s} {'n':>3s} {'offset':>7s} | "
          f"{'S(stiff)':>9s} {'lin':>6s} {'n':>3s} | {'S/pi':>6s} confidence")
    for j, nm in enumerate(JOINT_NAMES):
        r = by_j.get(j)
        if r is None:
            print(f"  {nm:18s}   -- gravity span only {fit['gmax'][j]:.2f} Nm, "
                  "not observable --")
            continue
        lin_txt = f"{r['lin_s']:6.2f}" if np.isfinite(r["lin_s"]) else "   n/a"
        off_txt = f"{r['off']:+7.2f}" if np.isfinite(r["off"]) else "    n/a"
        print(f"  {nm:18s} {r['N_tau']:7.2f} {r['r2_use']:6.3f} {r['n_t']:3d} {off_txt}"
              f" | {r['S']:9.2f} {lin_txt} {r['n_s']:3d} | {r['s_pi']:6.2f} {r['note']}")

    print("\n  tau x : STILL-NEEDED torque factor on top of what the interface already")
    print("          applies. 1.0 = torque channel correct.")
    print("  S     : effective stiffness / commanded kp. 1.0 = stiffness correct.")
    print("  lin   : CENTERED corr(sag, g). Near 0 means sag barely varies, so S is a")
    print("          mean ratio rather than a verified proportional relation.")
    print("  S/pi  : with the Q_MAX/pi convention in place (and kp on a fixed 0-500")
    print("          range), S = N*pi, so this column recovers the mechanical gear")
    print("          ratio N. A near-integer here is the tell.")

    # Trust is the fitted confidence flag, NOT a string match: every note carries a
    # parenthetical ("ok (offset ...)"), so `note == "ok"` was never true and this
    # whole block plus the flags below were dead.
    trust = [r for r in rows if r["good"]]
    if trust:
        rr = [(r["name"], r["s_pi"], r["Nint"]) for r in trust
              if np.isfinite(r["s_pi"]) and r["s_pi"] > 0.5]
        if rr:
            print("\n  implied gear ratio N = S/pi on trusted joints:")
            for nm, sp, ni in rr:
                off = abs(sp - ni) / max(ni, 1)
                print(f"    {nm:18s} S/pi = {sp:5.2f}  -> N ~ {ni}"
                      + ("  (within 5%)" if off < 0.05 else
                         f"  ({off*100:.0f}% off an integer -- verify mechanically)"))
    print("\n  --- suggested constants ---")
    for r in rows:
        if not np.isfinite(r["N_tau"]):
            continue
        flag = "" if r["good"] else f"   # {r['note']}"
        tau_ok = abs(r["N_tau"] - 1.0) < 0.1
        stiff_ok = (not np.isfinite(r["S"])) or abs(r["S"] - 1.0) < 0.15
        if tau_ok and stiff_ok:
            print(f"   {r['name']:18s} already correct, no change{flag}")
            continue
        parts = []
        if not tau_ok:
            parts.append(f"multiply reported tau by {r['N_tau']:.2f}")
        if not stiff_ok:
            hint = ""
            if np.isfinite(r["s_pi"]) and abs(r["s_pi"] - r["Nint"]) < 0.15 * max(r["Nint"], 1):
                hint = f" (= {r['Nint']}*pi)"
            parts.append(f"divide commanded kp by {r['S']:.2f}{hint}")
        print(f"   {r['name']:18s} " + "; ".join(parts) + flag)
    if any(not r["good"] for r in rows):
        print("\n  NOTE: a LOW CONFIDENCE joint's factor is not reliable -- its gravity")
        print("  torque changes sign over this sweep. Re-collect with poses that keep")
        print("  that joint loaded in one direction before trusting its number.")

    _scale_plot(path, g, tau, sag, rows, out)
    # Fail closed: a table of NaNs is not a result. Reporting success here would let a
    # log too short to support a 2-point fit look like a clean "no correction needed".
    if not any(np.isfinite(r["N_tau"]) or np.isfinite(r["S"]) for r in rows):
        print("\n[fit] ERROR: no joint produced a usable fit (need >=2 poses with a"
              " gravity signal per joint)")
        return 2
    return 0


def _scale_plot(path, g, tau, sag, rows, out=None):
    """tau-vs-g and sag-vs-g with the fitted lines, so a NON-LINEAR channel
    (saturation, backlash) is visible instead of being averaged into one factor."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[fit] plot skipped: {e}")
        return
    if not rows:
        print("[fit] plot skipped: no observable joints")
        return
    n = len(rows)
    fig, axes = plt.subplots(2, n, figsize=(4.2 * n, 7.5), squeeze=False)
    for c, r in enumerate(rows):
        j, big = r["j"], r["big"]
        ax = axes[0, c]
        ax.scatter(g[:, j], tau[:, j], s=16, alpha=0.5, color="0.6", label="all")
        ax.scatter(g[big, j], tau[big, j], s=20, color="C0", label="fitted")
        if np.isfinite(r["N_tau"]) and abs(r["N_tau"]) > 1e-9:
            lo, hi = float(g[:, j].min()), float(g[:, j].max())
            xs = np.linspace(lo, hi, 10)
            ax.plot(xs, xs / r["N_tau"], "C3-", lw=1.2,
                    label=f"fit 1/{r['N_tau']:.2f}")
        ax.plot([g[:, j].min(), g[:, j].max()], [g[:, j].min(), g[:, j].max()],
                "k--", lw=1, label="y=x")
        ax.set_title(f"{r['name']}\nN_tau={r['N_tau']:.2f} R2={r['r2_t']:.3f}",
                     fontsize=9)
        ax.set_xlabel("model g [Nm]", fontsize=8)
        ax.set_ylabel("reported tau [Nm]", fontsize=8)
        ax.grid(alpha=0.3); ax.legend(fontsize=6)

        ax = axes[1, c]
        if sag is not None:
            ax.scatter(g[:, j], sag[:, j], s=16, alpha=0.5, color="0.6")
            ax.scatter(g[big, j], sag[big, j], s=20, color="C0")
            if np.isfinite(r["S"]):
                lo, hi = float(g[:, j].min()), float(g[:, j].max())
                xs = np.linspace(lo, hi, 10)
                kp_j = KP[j]
                ax.plot(xs, xs / (r["S"] * kp_j), "C3-", lw=1.2,
                        label=f"fit S={r['S']:.2f}")
                ax.plot(xs, xs / kp_j, "k--", lw=1, label="S=1 (ideal)")
                ax.legend(fontsize=6)
            ax.set_ylabel("measured sag [rad]", fontsize=8)
        else:
            ax.text(0.5, 0.5, "no q_des in log", ha="center", va="center",
                    transform=ax.transAxes)
        ax.set_title(f"S={r['S']:.2f}  R2={r['r2_s']:.3f}", fontsize=9)
        ax.set_xlabel("model g [Nm]", fontsize=8); ax.grid(alpha=0.3)
    fig.suptitle(f"per-joint scale fit -- {os.path.basename(path)}")
    fig.tight_layout()
    out = out or os.path.splitext(path)[0] + "_scale.png"
    fig.savefig(out, dpi=110); plt.close(fig)
    print(f"[fit] wrote {out}")


def compare(path_a, path_b, out=None):
    """Compare two gravity logs against the shared model and against each other.

    Built for the sim-vs-real question: run the same pose sweep in MuJoCo and on the
    robot, then see which channels disagree and why. Returns 0, or 2 on a load error.
    """
    try:
        import arm_ff  # noqa: F401  -- model needed for g(q)
    except Exception as e:
        print(f"[cmp] ERROR: needs pinocchio/arm_ff for the model: {e}")
        return 2
    logs, mets, tags = [], [], []
    for tag, p in (("A", path_a), ("B", path_b)):
        try:
            log = load_log(p)
        except Exception as e:
            print(f"[cmp] ERROR: cannot read {p}: {e}")
            return 2
        if "tau_meas" not in log or len(np.asarray(log["tau_meas"])) == 0:
            print(f"[cmp] ERROR: {p} contains no poses")
            return 2
        logs.append(log); mets.append(_joint_metrics(log)); tags.append(tag)
    A, B = mets

    def meta(log, m, tag, path):
        comp = str(np.asarray(log.get("comp", "?")).item()) if "comp" in log else "?"
        done = bool(np.asarray(log["complete"]).item()) if "complete" in log else True
        why = str(np.asarray(log.get("abort_reason", "")).item()) if "abort_reason" in log else ""
        gm, tm = m["g"].ravel(), m["tau"].ravel()
        alpha = float(gm @ tm / (gm @ gm)) if gm @ gm > 1e-9 else np.nan
        print(f"  {tag}: {os.path.basename(path)}")
        print(f"     poses={m['P']}  comp={comp}  complete={done}"
              + (f"  abort={why}" if why else ""))
        print(f"     kp={np.round(m['kp'], 1).tolist()}")
        print(f"     global alpha={alpha:.4f}   resid RMS={np.sqrt(np.mean((m['tau']-m['g'])**2)):.4f} Nm")
        return alpha

    print("\n=== gravity-log comparison ===")
    aa = meta(logs[0], A, "A", path_a)
    ab = meta(logs[1], B, "B", path_b)

    same_sweep = False
    n = min(A["P"], B["P"])
    if A["qd"] is not None and B["qd"] is not None:
        same_sweep = bool(np.allclose(A["qd"][:n], B["qd"][:n], atol=1e-9))
    print(f"\n  identical commanded sweep over the first {n} poses: {same_sweep}")

    hdr = (f"\n  {'joint':18s} | {'A tau/g':>8s} {'A sag':>7s} {'A kpx':>6s} "
           f"| {'B tau/g':>8s} {'B sag':>7s} {'B kpx':>6s} | verdict (B)")
    print(hdr); print("  " + "-" * (len(hdr) - 3))
    flags = []
    for j, nm in enumerate(JOINT_NAMES):
        if A["gmax"][j] < GRAV_SIGNAL_MIN and B["gmax"][j] < GRAV_SIGNAL_MIN:
            continue
        vb = _diagnose(B["tau_ratio"][j], B["sag_ratio"][j], B["fric"][j])
        va = _diagnose(A["tau_ratio"][j], A["sag_ratio"][j], A["fric"][j])
        if vb != "OK":
            flags.append(f"{nm}: {vb}")
        if va != "OK":
            flags.append(f"{nm} (in A): {va}")
        print(f"  {nm:18s} | {A['tau_ratio'][j]:8.2f} {A['sag_ratio'][j]:7.2f} "
              f"{A['kp_factor'][j]:6.1f} | {B['tau_ratio'][j]:8.2f} "
              f"{B['sag_ratio'][j]:7.2f} {B['kp_factor'][j]:6.1f} | {vb}")
    print("  (tau/g = reported torque vs model; sag = measured droop vs -g/kp;")
    print("   kpx = implied stiffness / commanded kp. All three should be ~1.)")

    print(f"\n  {'joint':18s} {'A fric':>8s} {'B fric':>8s}   (|tau_hi-tau_lo|/2, Nm)")
    for j, nm in enumerate(JOINT_NAMES):
        if A["gmax"][j] < GRAV_SIGNAL_MIN and B["gmax"][j] < GRAV_SIGNAL_MIN:
            continue
        print(f"  {nm:18s} {A['fric'][j]:8.3f} {B['fric'][j]:8.3f}")

    for tag, log, m in (("A", logs[0], A), ("B", logs[1], B)):
        ds = np.asarray(log["dq_stat"], float) if "dq_stat" in log else np.array([np.nan])
        ts = np.asarray(log["tau_std"], float) if "tau_std" in log else np.array([np.nan])
        rp = np.linalg.norm(m["tau"] - m["g"], axis=1)
        print(f"\n  {tag} settle/noise: dq_stat med={np.median(ds):.4f} max={ds.max():.4f}"
              f" | tau_std med={np.median(ts):.4f} max={ts.max():.4f}")
        print(f"  {tag} per-pose |resid|: med={np.median(rp):.3f} max={rp.max():.3f} "
              f"Nm (pose {int(rp.argmax())})")

    if same_sweep:
        print(f"\n  matched-pose B/A ratios over {n} poses:")
        print(f"  {'joint':18s} {'tau A':>8s} {'tau B':>8s} {'B/A':>6s} "
              f"{'sag A':>8s} {'sag B':>8s} {'B/A':>6s}")
        for j, nm in enumerate(JOINT_NAMES):
            if A["gmax"][j] < GRAV_SIGNAL_MIN and B["gmax"][j] < GRAV_SIGNAL_MIN:
                continue
            ta, tb = A["tau"][:n, j].mean(), B["tau"][:n, j].mean()
            r = tb / ta if abs(ta) > 1e-6 else np.nan
            sa = sb = rs = np.nan
            if A["sag"] is not None and B["sag"] is not None:
                sa, sb = A["sag"][:n, j].mean(), B["sag"][:n, j].mean()
                rs = sb / sa if abs(sa) > 1e-6 else np.nan
            print(f"  {nm:18s} {ta:8.3f} {tb:8.3f} {r:6.2f} {sa:8.4f} {sb:8.4f} {rs:6.2f}")

    # Can any PHYSICAL (non-negative) mass set explain B? If not, it is not a mass error.
    try:
        from scipy.optimize import nnls
        G = _mass_scale_regressor(B["q"]); bvec = B["tau"].ravel()
        s, _ = nnls(G, bvec)
        r_nom = float(np.sqrt(np.mean((bvec - G @ np.ones(len(ARM_LINKS))) ** 2)))
        r_best = float(np.sqrt(np.mean((bvec - G @ s) ** 2)))
        zeroed = [JOINT_NAMES[k] for k in range(len(s)) if s[k] < 0.05]
        print(f"\n  B mass-error test: nominal RMS {r_nom:.3f} -> best s>=0 RMS {r_best:.3f} Nm")
        print(f"     best non-negative scales: {np.round(s, 3).tolist()}")
        if zeroed:
            print(f"     requires ~zero mass on {zeroed} -> NOT a mass error")
    except Exception as e:
        print(f"\n  (mass-error test skipped: {e})")

    print("\n  --- findings ---")
    if flags:
        for f in dict.fromkeys(flags):
            print(f"   * {f}")
    else:
        print("   * both logs consistent with the model on every observable joint")
    if np.isfinite(aa) and np.isfinite(ab) and abs(aa - ab) > 0.1:
        print(f"   * global alpha differs sharply: A={aa:.3f} vs B={ab:.3f}")

    _compare_plot(path_a, path_b, A, B, n if same_sweep else 0, out)
    return 0


def _compare_plot(path_a, path_b, A, B, n_matched, out=None):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[cmp] plot skipped: {e}")
        return
    fig, axes = plt.subplots(3, 3, figsize=(16, 12))
    for j in range(NUM_MOTORS):
        ax = axes[j // 3, j % 3]
        for m, tag, c in ((A, "A", "C0"), (B, "B", "C3")):
            ax.scatter(m["g"][:, j], m["tau"][:, j], s=16, alpha=0.75, color=c, label=tag)
        allv = np.concatenate([A["g"][:, j], A["tau"][:, j],
                               B["g"][:, j], B["tau"][:, j]])
        lo, hi = float(allv.min()), float(allv.max())
        pad = 0.1 * (hi - lo + 1e-6)
        ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], "k--", lw=1, label="y=x")
        ax.set_title(JOINT_NAMES[j], fontsize=9)
        ax.set_xlabel("model g [Nm]", fontsize=8)
        ax.set_ylabel("measured [Nm]", fontsize=8)
        ax.grid(alpha=0.3)
        if j == 0:
            ax.legend(fontsize=7)

    idx = [j for j in range(NUM_MOTORS)
           if max(A["gmax"][j], B["gmax"][j]) >= GRAV_SIGNAL_MIN]
    xs = np.arange(len(idx)); w = 0.38
    for ax, key, ttl in ((axes[2, 0], "tau_ratio", "reported torque / model"),
                         (axes[2, 1], "sag_ratio", "measured sag / model sag")):
        ax.bar(xs - w / 2, [A[key][j] for j in idx], w, label="A", color="C0")
        ax.bar(xs + w / 2, [B[key][j] for j in idx], w, label="B", color="C3")
        ax.axhline(1.0, color="k", ls="--", lw=1)
        ax.axhspan(RATIO_OK[0], RATIO_OK[1], color="green", alpha=0.10)
        ax.set_xticks(xs)
        ax.set_xticklabels([JOINT_NAMES[j] for j in idx], rotation=30,
                           ha="right", fontsize=7)
        ax.set_title(ttl + "  (1.0 = agrees)", fontsize=9)
        ax.grid(alpha=0.3); ax.legend(fontsize=7)

    ax = axes[2, 2]
    rpa = np.linalg.norm(A["tau"] - A["g"], axis=1)
    rpb = np.linalg.norm(B["tau"] - B["g"], axis=1)
    ax.plot(rpa, "o-", ms=3, color="C0", label="A")
    ax.plot(rpb, "s-", ms=3, color="C3", label="B")
    ax.set_title("per-pose |measured - model| [Nm]", fontsize=9)
    ax.set_xlabel("pose index", fontsize=8); ax.grid(alpha=0.3); ax.legend(fontsize=7)

    ttl = (f"A = {os.path.basename(path_a)}   vs   B = {os.path.basename(path_b)}"
           + (f"   ({n_matched} matched poses)" if n_matched else "   (sweeps differ)"))
    fig.suptitle(ttl)
    fig.tight_layout()
    out = out or (os.path.splitext(path_a)[0] + "_vs_"
                  + os.path.basename(os.path.splitext(path_b)[0]) + "_compare.png")
    fig.savefig(out, dpi=110); plt.close(fig)
    print(f"[cmp] wrote {out}")


def plot_raw(path, out=None, dq_tol=0.1, tau_std_tol=0.5):
    """Plot ONLY what the log contains, indexed by pose -> <npz>_raw.png. Returns 0/2.

    The ``--analyze`` figure is a DERIVED view (measured vs model). This one makes no
    model prediction at all, so it works with no pinocchio and on a broken/aborted log
    -- which is exactly when you need to see the data rather than a comparison.

    Note the .npz holds per-pose AGGREGATES: ``hold_and_measure`` averages its sample
    window and keeps only the mean/spread, so there is no time axis to plot.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[raw] ERROR: matplotlib unavailable: {e}")
        return 2
    try:
        log = load_log(path)
    except Exception as e:
        print(f"[raw] ERROR: cannot read {path}: {e}")
        return 2

    def arr(key, shape2=True):
        if key not in log:
            return None
        a = np.asarray(log[key], float)
        return a if (a.ndim == 2 or not shape2) else None

    q_des, q_meas = arr("q_des"), arr("q_meas")
    tau_meas, tau_lo, tau_hi = arr("tau_meas"), arr("tau_lo"), arr("tau_hi")
    fric = arr("friction_est")
    P = 0 if tau_meas is None else len(tau_meas)
    if P == 0:
        print(f"[raw] ERROR: {path} contains no poses")
        return 2

    names = [str(x) for x in np.asarray(log["joint_names"]).ravel()] \
        if "joint_names" in log else list(JOINT_NAMES)
    if len(names) != NUM_MOTORS:
        print(f"[raw] WARNING: joint_names has {len(names)} entries; using defaults")
        names = list(JOINT_NAMES)

    dq_stat = np.asarray(log["dq_stat"], float).ravel() if "dq_stat" in log else None
    tau_std = np.asarray(log["tau_std"], float).ravel() if "tau_std" in log else None
    static = (np.asarray(log["static_ok"], bool).ravel() if "static_ok" in log
              else np.ones(P, bool))
    complete = bool(np.asarray(log["complete"]).item()) if "complete" in log else True
    reason = str(np.asarray(log.get("abort_reason", "")).item()) if "abort_reason" in log else ""
    comp = str(np.asarray(log.get("comp", "?")).item()) if "comp" in log else "?"

    x = np.arange(P)
    fig, axes = plt.subplots(2, 2, figsize=(15, 9))
    (ax_tau, ax_sag), (ax_fric, ax_set) = axes

    # Measured torque, with the lo/hi (bidirectional approach) bracket per point.
    for j in range(min(NUM_MOTORS, tau_meas.shape[1])):
        ln, = ax_tau.plot(x, tau_meas[:, j], "o-", ms=3, lw=1.0, label=names[j])
        if tau_lo is not None and tau_hi is not None:
            ax_tau.vlines(x, np.minimum(tau_lo[:, j], tau_hi[:, j]),
                          np.maximum(tau_lo[:, j], tau_hi[:, j]),
                          color=ln.get_color(), alpha=0.35, lw=2)
    ax_tau.set_ylabel("measured torque [Nm]")
    ax_tau.set_title("tau_meas per pose (bar = tau_lo..tau_hi spread)")
    ax_tau.legend(fontsize=7, ncol=3)

    # PD sag: with gravity comp OFF this droop is why g_model is evaluated at q_meas.
    if q_des is not None and q_meas is not None:
        for j in range(min(NUM_MOTORS, q_meas.shape[1])):
            ax_sag.plot(x, q_meas[:, j] - q_des[:, j], "o-", ms=3, lw=1.0, label=names[j])
        ax_sag.axhline(0.0, color="0.6", lw=0.8)
        ax_sag.legend(fontsize=7, ncol=3)
    else:
        ax_sag.text(0.5, 0.5, "q_des / q_meas not in log", ha="center", va="center",
                    transform=ax_sag.transAxes)
    ax_sag.set_ylabel("q_meas - q_des [rad]")
    ax_sag.set_title("PD sag (comp OFF)")

    if fric is not None:
        for j in range(min(NUM_MOTORS, fric.shape[1])):
            ax_fric.plot(x, fric[:, j], "o-", ms=3, lw=1.0, label=names[j])
        ax_fric.legend(fontsize=7, ncol=3)
    else:
        ax_fric.text(0.5, 0.5, "friction_est not in log", ha="center", va="center",
                     transform=ax_fric.transAxes)
    ax_fric.set_ylabel("friction est [Nm]")
    ax_fric.set_title("(tau_hi - tau_lo)/2")
    ax_fric.set_xlabel("pose index")

    # Settle quality against the thresholds that set static_ok. Log scale when the
    # data sits far below its threshold -- otherwise the threshold line stretches the
    # axis and squashes the actual values into an unreadable line at the bottom.
    def _settle(ax, vals, tol, color, marker, label, unit):
        ax.plot(x, vals, marker + "-", ms=3, lw=1.0, color=color, label=label)
        ax.axhline(tol, color=color, ls="--", lw=0.9, label=f"{label}_tol={tol:g}")
        ax.set_ylabel(f"{label} [{unit}]", color=color)
        finite = vals[np.isfinite(vals)]
        # Decide on the MEDIAN, not the max: one spike must not force a linear axis
        # that squashes the other N-1 poses into a flat line.
        if (finite.size and finite.min() > 0 and tol > 0
                and tol / max(np.median(finite), 1e-12) > 5):
            ax.set_yscale("log")

    if dq_stat is not None:
        _settle(ax_set, dq_stat, dq_tol, "C0", "o", "dq_stat", "rad/s")
        ax_set.legend(fontsize=7, loc="upper left")
    ax_set.set_xlabel("pose index")
    if tau_std is not None:
        ax2 = ax_set.twinx()
        _settle(ax2, tau_std, tau_std_tol, "C3", "s", "tau_std", "Nm")
        ax2.legend(fontsize=7, loc="upper right")
    ax_set.set_title("settle quality (log scale when far below tolerance)")

    # Shade non-static poses in EVERY panel so one bad hold is obvious everywhere.
    bad = np.flatnonzero(~static)
    for ax in axes.ravel():
        for p in bad:
            ax.axvspan(p - 0.4, p + 0.4, color="red", alpha=0.12, zorder=0)
        ax.grid(alpha=0.3)

    title = f"RAW logged data -- {os.path.basename(path)}   ({P} poses, comp={comp}"
    title += f", {len(bad)} non-static)" if len(bad) else ")"
    if not complete:
        title += f"\nINCOMPLETE: {reason}"
    fig.suptitle(title)
    fig.tight_layout()
    out = out or os.path.splitext(path)[0] + "_raw.png"
    fig.savefig(out, dpi=110); plt.close(fig)

    print(f"\n=== raw log: {path} ===")
    print(f"  poses={P}  comp={comp}  complete={complete}"
          + (f"  abort={reason}" if reason else ""))
    if len(bad):
        print(f"  non-static poses: {bad.tolist()}")
    print(f"  {'joint':18s} {'tau min':>9s} {'tau mean':>9s} {'tau max':>9s} "
          f"{'|sag|max':>9s} {'fric mean':>9s}")
    for j in range(min(NUM_MOTORS, tau_meas.shape[1])):
        sag = (np.abs(q_meas[:, j] - q_des[:, j]).max()
               if (q_des is not None and q_meas is not None) else np.nan)
        fm = np.abs(fric[:, j]).mean() if fric is not None else np.nan
        print(f"  {names[j]:18s} {tau_meas[:, j].min():9.3f} "
              f"{tau_meas[:, j].mean():9.3f} {tau_meas[:, j].max():9.3f} "
              f"{sag:9.3f} {fm:9.3f}")
    print(f"[raw] wrote {out}")
    return 0


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

    # A median-based static check rejects an isolated velocity spike.
    lo = (q0, np.full(NUM_MOTORS, 0.03), g)
    spike = (q0, np.full(NUM_MOTORS, 0.5), g)
    _, _, _, tau_mean, static = _FakeProbe([lo, lo, lo, lo, lo, spike]).hold_and_measure(q0, **hold_kw)
    ok = ok and static and np.allclose(tau_mean, g, atol=0.2)

    # Sustained motion is not static.
    mv = (q0, np.full(NUM_MOTORS, 0.3), g)
    _, _, _, _, static = _FakeProbe([mv] * 6).hold_and_measure(q0, **hold_kw)
    ok = ok and not static

    # A severe torque sample aborts immediately.
    sev_tau = g.copy(); sev_tau[4] = 8.0  # > 1.1*SAFETY_TAU[4] (6.93)
    try:
        _FakeProbe([q0])._watchdog(q0, q0, sev_tau, time.perf_counter(), True)
        ok = False  # should have raised
    except RuntimeError as e:
        ok = ok and "severe" in str(e)

    # Consecutive soft violations identify the joint and limit.
    fp = _FakeProbe([q0], trip_samples=3)
    soft_tau = g.copy(); soft_tau[4] = 6.5  # > SAFETY_TAU[4]=6.3, < severe
    raised = None
    for k in range(3):
        try:
            fp._watchdog(q0, q0, soft_tau, time.perf_counter(), True)
        except RuntimeError as e:
            raised = str(e); break
    ok = ok and raised is not None and "5dof_joint" in raised and "3 consec" in raised

    # Clearing a soft violation resets its debounce counter.
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

    # Broad observable error should pass the calibration gate.
    # tol_nm=0.1 so the (band-capped) 1.2x error counts as out-of-tolerance.
    s_true = np.ones(len(ARM_LINKS)); s_true[2] = 1.2  # upper_arm heavier
    tauA = g + (G @ (s_true - 1.0)).reshape(P, NUM_MOTORS)
    calA = _calibrate(poses, tauA - g, tol_nm=0.1)
    okA = calA["emit"] and not calA["model_good"] and abs(calA["s_hat"][2] - 1.2) < 0.06
    print(f"[selftest] gate A (real error): emit={calA['emit']} "
          f"s2={calA['s_hat'][2]:.3f} cv={100*calA['cv_frac']:.0f}%  {'ok' if okA else 'FAIL'}")

    # One outlier in an otherwise accurate model should be rejected.
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
    # Shifted poses keep validation independent from training.
    check = np.clip(build_pose_sweep(density=1) + np.array([0, 0.1, 0.1, 0.1, 0.1, 0]),
                    JOINT_LOW + 0.05, JOINT_HIGH - 0.05)
    s_true = np.ones(len(ARM_LINKS)); s_true[perturb_link] = mass_scale

    # Gravity is linear in link mass: g(q; s) = g_model(q) + G_arm(q) @ (s - 1).
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

    # Independently fit delta = s - 1 from the gravity residual.
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

    # Exercise overlay write, load, apply, and gravity evaluation.
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

    # Raw-plot smoke test: renders, and rejects a log with no poses.
    raw_png = os.path.splitext(path)[0] + "_raw.png"
    ok_raw = (plot_raw(path) == 0 and os.path.isfile(raw_png)
              and plot_raw(os.path.join(os.path.dirname(path), "nonexistent.npz")) == 2)
    print(f"[selftest] raw-plot checks: {'PASS' if ok_raw else 'FAIL'}")

    # Scale-estimator round trip: scale a synthetic log's torque by a known factor
    # and confirm fit_scale recovers exactly that factor (and 1.0 when unscaled).
    ok_scale = True
    try:
        import io, contextlib, re as _re
        base = load_log(path)

        def fitted_factor(scale):
            """upper_arm tau-factor reported for a log whose torque is /scale."""
            l2 = dict(base)
            for k in ("tau_meas", "tau_lo", "tau_hi"):
                l2[k] = np.asarray(base[k], float) / scale
            p2 = os.path.join(os.path.dirname(path), f"grav_scaletest_{scale:g}.npz")
            save_log(p2, l2)
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                fit_scale(p2)
            for ext in (".npz", "_scale.png"):
                f = os.path.splitext(p2)[0] + ext
                if os.path.exists(f):
                    os.unlink(f)
            m = _re.search(r"upper_arm_joint\s+([-\d.]+)", buf.getvalue())
            return float(m.group(1)) if m else float("nan")

        # Test the RATIO, not the absolute factor: this synthetic log carries a 1.3x
        # mass perturbation on upper_arm, so its torque is legitimately ~6% off the
        # nominal model and the absolute factor is not 1.0. The ratio is what the
        # estimator must get exactly right.
        b = fitted_factor(1.0)
        for want in (4.0, 6.0):
            got = fitted_factor(want)
            ratio = got / b if abs(b) > 1e-9 else float("nan")
            hit = abs(ratio - want) < 0.02 * want
            ok_scale = ok_scale and hit
            print(f"[selftest] scale round-trip x{want:g}: ratio {ratio:.3f} "
                  f"{'ok' if hit else 'FAIL'}")
        # A log with no poses must be rejected, not crash.
        ok_scale = ok_scale and fit_scale(os.path.join(
            os.path.dirname(path), "nonexistent.npz")) == 2
    except Exception as e:
        print(f"[selftest] scale round-trip FAILED: {e}")
        ok_scale = False
    print(f"[selftest] scale-estimator checks: {'PASS' if ok_scale else 'FAIL'}")

    ok = (ok_fit and ok_rt and ok_safe and ok_collect and ok_gate and ok_raw
          and ok_scale)
    print(f"[selftest] {'PASS' if ok else 'FAIL'} "
          "(fit reproduction + overlay round-trip + fail-safe + collection watchdog "
          "+ calib gate + raw plot + scale estimator)")
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
    ap.add_argument("--plot-raw", metavar="NPZ",
                    help="offline: plot ONLY the logged per-pose data (torque, PD sag, "
                         "friction, settle quality) -> <npz>_raw.png. No model, no IK, "
                         "no pinocchio; works on an aborted/incomplete log.")
    ap.add_argument("--fit-scale", metavar="NPZ",
                    help="offline: fit the per-joint torque factor (N_tau) and stiffness "
                         "factor (S=K/kp) from a log, with R2/confidence, and print the "
                         "constants to apply -> <npz>_scale.png")
    ap.add_argument("--compare", nargs=2, metavar=("A_NPZ", "B_NPZ"),
                    help="offline: compare two logs (e.g. sim vs real) against the "
                         "shared model and each other. Separates 'torque sensor lies' "
                         "from 'motor did not deliver' by checking the encoder sag "
                         "independently -> <A>_vs_<B>_compare.png")
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
    # On-robot collection tuning
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
    modes = [m for m, on in (("--analyze", bool(args.analyze)),
                             ("--plot-raw", bool(args.plot_raw)),
                             ("--compare", bool(args.compare)),
                             ("--fit-scale", bool(args.fit_scale)),
                             ("--selftest", args.selftest)) if on]
    if len(modes) > 1:
        ap.error(f"{', '.join(modes)} are separate modes; pick one")
    if args.fit_scale:
        return fit_scale(args.fit_scale)
    if args.compare:
        return compare(args.compare[0], args.compare[1])
    if args.plot_raw:
        return plot_raw(args.plot_raw, dq_tol=args.dq_tol,
                        tau_std_tol=args.tau_std_tol)

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

    # On-robot collection
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
