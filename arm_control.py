"""Shared dynamics, gain validation, reporting, and DDS transport helpers.

The experimental MPC controller uses these helpers; smooth PD motion lives in
``arm_smooth_move.py``. This module contains no trajectory feedback controller.
"""

from __future__ import annotations

import threading
import time

import numpy as np
import pinocchio

import arm_ff
import arm_ik
import ee_traj as T

NUM_MOTORS = T.NUM_ARM_DOF
NX, NU = 2 * NUM_MOTORS, NUM_MOTORS
DT = 0.005
KP = np.array([20.0, 40.0, 40.0, 20.0, 20.0, 20.0])
KD = np.array([0.5, 1.0, 1.0, 0.5, 0.5, 0.5])
MOTOR_TAU_LIMIT = arm_ff.TAU_LIMIT
SAFETY_TAU = 0.90 * MOTOR_TAU_LIMIT
DQ_LIMIT = np.full(NUM_MOTORS, 6.0)
TRACK_ABORT_RAD = 0.35
Q_POS, Q_VEL, R_TAU = 100.0, 1.0, 0.05
W_EE, W_ROT = 5.0e3, 5.0e2



def linearize(model, data, q, dq, tau, dt):
    """Discrete-time linearization about (q, dq, tau).

    ``A_c = [[0, I], [da/dq, da/dv]]``, ``B_c = [[0], [M^-1]]`` -- all three blocks
    come from ``computeABADerivatives``. Euler is accurate enough at dt=5 ms.
    """
    pinocchio.computeABADerivatives(model, data,
                                    np.asarray(q, float),
                                    np.asarray(dq, float),
                                    np.asarray(tau, float))
    A = np.eye(NX)
    A[:NUM_MOTORS, NUM_MOTORS:] = dt * np.eye(NUM_MOTORS)
    A[NUM_MOTORS:, :NUM_MOTORS] = dt * data.ddq_dq
    A[NUM_MOTORS:, NUM_MOTORS:] += dt * data.ddq_dv
    B = np.zeros((NX, NU))
    B[NUM_MOTORS:, :] = dt * data.Minv
    return A, B


def _check_weights(q_pos, q_vel, w_ee, w_rot, eps, r_tau=None):
    """Reject non-finite or indefinite state and input cost weights."""
    for label, v in (("q_pos", q_pos), ("q_vel", q_vel),
                     ("w_ee", w_ee), ("w_rot", w_rot), ("eps", eps)):
        if not np.isfinite(v) or v < 0:
            raise ValueError(f"{label} must be finite and nonnegative, got {v}")
    if eps <= 0:
        raise ValueError(f"eps must be positive (keeps Q PD), got {eps}")
    if r_tau is not None and (not np.isfinite(r_tau) or r_tau <= 0):
        raise ValueError(f"r_tau must be finite and positive, got {r_tau}")


def state_cost(model, data, q, q_pos=Q_POS, q_vel=Q_VEL, task_space=False,
               w_ee=W_EE, w_rot=W_ROT, eps=1.0):
    """State weighting matrix Q (12x12) at configuration ``q``.

    ``task_space=True`` makes the position block ``J^T W J + eps*I`` from the FULL
    6x6 Jacobian with ``W = diag([w_ee]*3 + [w_rot]*3)``, so the cost penalizes EE
    POSE error. Using only the 3x6 translational Jacobian (as this did originally)
    left rotation penalized by ``eps`` alone: at the home pose a pure-EE-rotation
    direction cost 1.0 vs 1006 for a translating one -- ~1000x under-weighted even
    with an explicit --rpy. ``eps*I`` keeps Q PD and regularizes near singularities.
    """
    _check_weights(q_pos, q_vel, w_ee, w_rot, eps)
    Q = np.zeros((NX, NX))
    if task_space:
        joint_id = min(arm_ik.JOINT_ID, model.njoints - 1)
        _, Jf = T._frame_jacobian(model, data, np.asarray(q, float), joint_id)
        W = np.diag(np.concatenate([np.full(3, float(w_ee)), np.full(3, float(w_rot))]))
        Q[:NUM_MOTORS, :NUM_MOTORS] = Jf.T @ W @ Jf + eps * np.eye(NUM_MOTORS)
    else:
        Q[:NUM_MOTORS, :NUM_MOTORS] = q_pos * np.eye(NUM_MOTORS)
    Q[NUM_MOTORS:, NUM_MOTORS:] = q_vel * np.eye(NUM_MOTORS)
    return Q


def check_gains(gains, label):
    """Validated (6,) gain array, or raise.

    Gains go straight into ``motor_cmd.kp/.kd``; ``np.clip`` does not remove NaN and
    nothing downstream inspects them, so an unvalidated gain reaches the motors --
    verified: ``--kp nan ...`` previously published ``kp=[nan, nan, nan]``.
    """
    arr = np.asarray(gains, dtype=float)
    if arr.shape != (NUM_MOTORS,):
        raise ValueError(f"{label} must have {NUM_MOTORS} values, got {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{label} must be finite, got {arr}")
    if np.any(arr < 0):
        raise ValueError(f"{label} must be nonnegative (negative feedback gain is "
                         f"destabilizing), got {arr}")
    return arr


def k_pd_matrix(kp=KP, kd=KD):
    """The hardware PD expressed as a state-feedback gain (6x12)."""
    K = np.zeros((NU, NX))
    K[:, :NUM_MOTORS] = np.diag(kp)
    K[:, NUM_MOTORS:] = np.diag(kd)
    return K


def perturbed_model(scale=1.2, links=(2, 3)):
    """Design-model copy with link masses scaled: a deliberate plant mismatch."""
    model = T.build_arm_model()
    for b in links:
        if b + 1 < model.njoints:
            I = model.inertias[b + 1]
            model.inertias[b + 1] = pinocchio.Inertia(
                I.mass * scale, I.lever, I.inertia * scale)
    return model


class DDSController:
    """DDS state snapshots, guarded publishing, ramps, and return-to-home."""

    def __init__(self, kp=KP, kd=KD, trip_samples=3, state_timeout=0.1):
        from unitree_sdk2py.core.channel import ChannelPublisher, ChannelSubscriber
        from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowCmd_
        from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowCmd_, LowState_
        from unitree_sdk2py.utils.crc import CRC

        # Validate BEFORE the publisher exists. Negative kd is positive velocity
        # feedback, i.e. actively destabilizing.
        self.kp = check_gains(kp, "kp")
        self.kd = check_gains(kd, "kd")
        self.trip_samples = max(1, int(trip_samples))
        if not np.isfinite(state_timeout) or state_timeout <= 0:
            raise ValueError(f"state_timeout must be finite and positive, got {state_timeout}")
        self.state_timeout = float(state_timeout)
        self._trip = np.zeros(NUM_MOTORS, dtype=int)

        self.low_cmd = unitree_go_msg_dds__LowCmd_()
        self.crc = CRC()
        self._lock = threading.Lock()
        self.qpos = np.zeros(NUM_MOTORS)
        self.qvel = np.zeros(NUM_MOTORS)
        self.qtau = np.zeros(NUM_MOTORS)
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

    def _write(self, q_des, dq_des=None, tau_ff=None):
        q_des = np.asarray(q_des, float)
        dq_des = np.zeros(NUM_MOTORS) if dq_des is None else np.asarray(dq_des, float)
        tau_ff = np.zeros(NUM_MOTORS) if tau_ff is None else np.asarray(tau_ff, float)
        # np.clip does not remove NaN -- refuse rather than publish. Gains are
        # re-checked here because this is the line that publishes them.
        for label, a in (("q", q_des), ("dq", dq_des), ("tau", tau_ff),
                         ("kp", self.kp), ("kd", self.kd)):
            if a.shape != (NUM_MOTORS,) or not np.all(np.isfinite(a)):
                raise ValueError(f"refusing to publish non-finite {label}: {a}")
        q_des = np.clip(q_des, T.JOINT_LOW + 0.05, T.JOINT_HIGH - 0.05)
        tau_ff = np.clip(tau_ff, -MOTOR_TAU_LIMIT, MOTOR_TAU_LIMIT)
        for i in range(NUM_MOTORS):
            self.low_cmd.motor_cmd[i].q = float(q_des[i])
            self.low_cmd.motor_cmd[i].dq = float(dq_des[i])
            self.low_cmd.motor_cmd[i].kp = float(self.kp[i])
            self.low_cmd.motor_cmd[i].kd = float(self.kd[i])
            self.low_cmd.motor_cmd[i].tau = float(tau_ff[i])
        self.low_cmd.crc = self.crc.Crc(self.low_cmd)
        self.pub.Write(self.low_cmd)

    def _watchdog(self, qk, dqk, tauk, arrival, valid):
        if not valid or time.perf_counter() - arrival > self.state_timeout:
            raise RuntimeError(f"stale/missing state (age > {self.state_timeout}s)")
        soft = ((np.abs(tauk) > SAFETY_TAU) | (np.abs(dqk) > DQ_LIMIT)
                | (qk < T.JOINT_LOW - 0.05) | (qk > T.JOINT_HIGH + 0.05))
        severe = ((np.abs(tauk) > 1.10 * SAFETY_TAU) | (np.abs(dqk) > 1.10 * DQ_LIMIT)
                  | (qk < T.JOINT_LOW - 0.10) | (qk > T.JOINT_HIGH + 0.10))
        self._trip = np.where(soft, self._trip + 1, 0)
        if np.any(severe):
            j = int(np.flatnonzero(severe)[0])
            raise RuntimeError(f"severe safety limit on {arm_ik.IK_MODEL.names[j+1]}")
        bad = np.flatnonzero(self._trip >= self.trip_samples)
        if bad.size:
            j = int(bad[0])
            raise RuntimeError(f"safety limit on {arm_ik.IK_MODEL.names[j+1]} "
                               f"({self.trip_samples} consec)")

    def _fresh_start_pose(self):
        qk, _, _, arrival, valid = self.snapshot()
        age = time.perf_counter() - arrival
        if not valid or not np.isfinite(age) or age > self.state_timeout:
            raise RuntimeError(f"no fresh state to ramp from (age={age:.3f}s)")
        return qk

    def ramp_to(self, target, duration=3.0, dt=DT, watchdog=True):
        """Gentle PD ramp with gravity feedforward (no trajectory tracking yet)."""
        if not np.isfinite(dt) or dt <= 0 or not np.isfinite(duration) or duration <= 0:
            raise ValueError("ramp dt/duration must be finite and positive")
        self._trip[:] = 0
        start = self._fresh_start_pose()
        target = np.asarray(target, float)
        steps = max(1, int(duration / dt))
        for k in range(steps):
            t0 = time.perf_counter()
            qk, dqk, tauk, arrival, valid = self.snapshot()
            if watchdog:
                self._watchdog(qk, dqk, tauk, arrival, valid)
            phase = (k + 1) / steps
            q_des = start * (1 - phase) + target * phase
            # Only gravity here: there is no reference acceleration during a ramp.
            self._write(q_des, tau_ff=arm_ff.gravity_torque(qk))
            sleep = dt - (time.perf_counter() - t0)
            if sleep > 0:
                time.sleep(sleep)

    def safe_return(self, dt=DT):
        """Always bring the arm down; limit breaches must not block this."""
        try:
            self.ramp_to(np.zeros(NUM_MOTORS), duration=3.0, dt=dt, watchdog=False)
        except (KeyboardInterrupt, RuntimeError, ValueError) as e:
            print(f"[control] safe_return stopped: {e}")


def _exit_code(log, interrupted=False):
    """Process exit code for a tracking run.

    ``complete`` is checked INDEPENDENTLY of the sample count: an abort on the first
    sample leaves the log empty, and gating on ``len(log["q"])`` made an immediate
    hardware safety abort exit 0. No log at all (Ctrl-C, or a pre-track ramp failure)
    is likewise not a success.
    """
    if log is None:
        return 130 if interrupted else 2
    return 0 if bool(log["complete"]) else 2


def report_tracking(q, q_ref, model, p_ee=None, label="tracking"):
    """Per-joint and EE pose tracking error -> (joint_rms, ee_pos_rms, ee_rot_rms).

    Reports BOTH position and orientation: the task-space cost weights both, so a
    position-only report cannot show whether --w-rot is doing anything.
    """
    n = min(len(q), len(q_ref))
    err = q[:n] - q_ref[:n]
    jrms = np.sqrt(np.mean(err ** 2, axis=0))
    p_act = T.ee_positions_of(q[:n], model)
    ref = T.ee_positions_of(q_ref[:n], model) if p_ee is None else np.asarray(p_ee)[:n]
    ee = np.linalg.norm(p_act - ref, axis=1)
    rot = T.orientation_error(q[:n], q_ref[:n], model)
    print(f"\n=== {label} ===")
    print(f"  joint RMS [rad]: {np.round(jrms, 4)}")
    print(f"  EE position: mean {ee.mean()*1000:6.2f} mm | "
          f"max {ee.max()*1000:6.2f} mm | final {ee[-1]*1000:6.2f} mm")
    print(f"  EE orientation: mean {np.degrees(rot.mean()):6.2f} deg | "
          f"max {np.degrees(rot.max()):6.2f} deg | final {np.degrees(rot[-1]):6.2f} deg")
    return (jrms, float(np.sqrt(np.mean(ee ** 2))),
            float(np.sqrt(np.mean(rot ** 2))))


def _plot(plan, sim=None, out="ee_tracking.png"):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[plot] skipped: {e}")
        return
    t, model = plan["t"], plan["model"]
    fig, axes = plt.subplots(3, 1, figsize=(10, 10))
    p_ref = plan["p_ee"]
    axes[0].plot(p_ref[:, 0], p_ref[:, 2], "k--", label="EE reference")
    if sim is not None:
        p_act = T.ee_positions_of(sim["q"], model)
        axes[0].plot(p_act[:, 0], p_act[:, 2], "C0", label="EE achieved")
    axes[0].set_xlabel("x [m]"); axes[0].set_ylabel("z [m]")
    axes[0].set_title("EE path (x-z)"); axes[0].axis("equal")
    axes[0].grid(alpha=0.3); axes[0].legend(fontsize=8)

    for j in range(NUM_MOTORS):
        axes[1].plot(t, plan["q_ref"][:, j], lw=1.0,
                     label=arm_ik.IK_MODEL.names[j + 1])
    axes[1].set_ylabel("q_ref [rad]"); axes[1].grid(alpha=0.3)
    axes[1].legend(fontsize=7, ncol=3)

    for j in range(NUM_MOTORS):
        axes[2].plot(t, plan["tau_ref"][:, j], lw=1.0)
    axes[2].set_ylabel("tau_ref [Nm]"); axes[2].set_xlabel("time [s]")
    axes[2].grid(alpha=0.3)
    fig.suptitle("EE trajectory reference" + ("" if sim is None else " + simulation"))
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    plt.close(fig)
    print(f"[plot] wrote {out}")
