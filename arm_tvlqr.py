"""Track a moving end-effector trajectory with time-varying LQR.

The reference, inverse-dynamics torque, and Riccati gains are computed offline. The
200 Hz loop combines the model feedback with the low-level motor PD using the idealized
command law

    tau_applied = tau_sent + kp*(q_des - q) + kd*(dq_des - dq)

    tau_sent = tau_ref - (K_lqr - K_pd) @ x_err     =>   tau_applied = tau_ref - K_lqr @ x_err

This file does not model the repository's measured command-torque and gain-path scaling,
so that cancellation is exact only under the idealized law. ``tau_ref`` already includes
gravity. Torque limits are enforced by clamping rather than as hard constraints.
"""

from __future__ import annotations

import argparse
import sys
import threading
import time

import numpy as np
import pinocchio

import arm_ff
import arm_ik
import ee_traj as T

NUM_MOTORS = T.NUM_ARM_DOF          # 6
NX, NU = 2 * NUM_MOTORS, NUM_MOTORS  # state [q; dq], input tau
DT = 0.005                           # 200 Hz, matches pineapple_arm.py

# Nominal motor PD used by the idealized cancellation law.
KP = np.array([20.0, 40.0, 40.0, 20.0, 20.0, 20.0])
KD = np.array([0.5, 1.0, 1.0, 0.5, 0.5, 0.5])

MOTOR_TAU_LIMIT = arm_ff.TAU_LIMIT                 # [27,27,27,7,7,7]
SAFETY_TAU = 0.90 * MOTOR_TAU_LIMIT
DQ_LIMIT = np.full(NUM_MOTORS, 6.0)                # rad/s abort ceiling
TRACK_ABORT_RAD = 0.35                             # per-joint tracking error abort

# Default weights: joint position, joint velocity, and torque.
Q_POS, Q_VEL, R_TAU = 100.0, 1.0, 0.05
W_EE = 5.0e3        # task-space POSITION weight [1/m^2]
# Task-space orientation weight [1/rad^2], chosen to improve rotation without the
# high gains and noise sensitivity of larger values.
W_ROT = 5.0e2


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
    """Reject weights that make the LQR problem ill-posed.

    A negative state weight makes Q indefinite and silently destroys the Riccati
    stability/optimality guarantees -- observed: --q-vel -1 raised |K|max from 117
    to ~32000 while still reporting "feasible". R must be PD so (R+B'PB) inverts.
    """
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


def tvlqr_gains(model, q_ref, dq_ref, tau_ref, dt, q_pos=Q_POS, q_vel=Q_VEL,
                r_tau=R_TAU, task_space=False, w_ee=W_EE, w_rot=W_ROT,
                qf_scale=10.0):
    """Backward Riccati recursion along the reference -> K (N,6,12).

    ``P_N = qf_scale * Q_N`` (heavier terminal cost drives the arm onto the end of
    the trajectory), then ``K_k = (R+B'PB)^-1 B'PA``, ``P = Q_k + A'P(A - B K_k)``.
    """
    _check_weights(q_pos, q_vel, w_ee, w_rot, 1.0, r_tau)
    n = len(q_ref)
    data = model.createData()
    R = r_tau * np.eye(NU)

    # Precompute per-step linearization + cost (the Riccati pass then runs backward).
    As, Bs, Qs = [], [], []
    for k in range(n):
        A, B = linearize(model, data, q_ref[k], dq_ref[k], tau_ref[k], dt)
        As.append(A); Bs.append(B)
        Qs.append(state_cost(model, data, q_ref[k], q_pos, q_vel, task_space,
                             w_ee, w_rot))

    if max(np.abs(Q).max() for Q in Qs) <= 0.0:
        print("[tvlqr] WARNING: all state weights are zero -> K = 0 (no feedback)")
    K = np.zeros((n, NU, NX))
    P = qf_scale * Qs[-1]
    for k in range(n - 1, -1, -1):
        A, B, Q = As[k], Bs[k], Qs[k]
        S = R + B.T @ P @ B
        Kk = np.linalg.solve(S, B.T @ P @ A)
        P = Q + A.T @ P @ (A - B @ Kk)
        P = 0.5 * (P + P.T)          # keep it symmetric against drift
        K[k] = Kk
    return K


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


def tau_to_send(tau_ref_k, K_k, x_err, kp=KP, kd=KD, clamp=True):
    """Feedforward torque for motor_cmd.tau. Subtracting K_pd avoids double-counting
    the hardware PD; see the module docstring."""
    tau = tau_ref_k - (K_k - k_pd_matrix(kp, kd)) @ x_err
    if not np.all(np.isfinite(tau)):
        raise ValueError(f"non-finite control torque: {tau}")
    return np.clip(tau, -MOTOR_TAU_LIMIT, MOTOR_TAU_LIMIT) if clamp else tau


def simulate(plan, plant_model, dt, kp=KP, kd=KD, q0=None, dq0=None,
             mode="lqr"):
    """Closed-loop rollout against ``plant_model`` (may differ from the design model).

    Reproduces the hardware law exactly
    (``tau_applied = clamp(tau_sent) + kp*(q_ref-q) + kd*(dq_ref-dq)``) then
    integrates with ``aba`` + semi-implicit Euler. -> (q, dq, tau_applied) each (N,6).

    ``mode``: "pd" (nothing sent), "ff" (``tau_ref`` only -- the HONEST baseline for
    judging TVLQR, since it isolates feedback from feedforward benefit, which
    otherwise dominates), or "lqr" (full :func:`tau_to_send`).
    """
    q_ref, dq_ref, tau_ref, K = plan["q_ref"], plan["dq_ref"], plan["tau_ref"], plan["K"]
    n = len(q_ref)
    data = plant_model.createData()
    q = np.zeros((n, NUM_MOTORS)); dq = np.zeros((n, NUM_MOTORS))
    tau_ap = np.zeros((n, NUM_MOTORS))
    qk = q_ref[0].copy() if q0 is None else np.asarray(q0, float).copy()
    vk = dq_ref[0].copy() if dq0 is None else np.asarray(dq0, float).copy()

    for k in range(n):
        q[k] = qk; dq[k] = vk
        x_err = np.concatenate([qk - q_ref[k], vk - dq_ref[k]])
        if mode == "lqr":
            sent = tau_to_send(tau_ref[k], K[k], x_err, kp, kd)
        elif mode == "ff":
            sent = np.clip(tau_ref[k], -MOTOR_TAU_LIMIT, MOTOR_TAU_LIMIT)
        else:
            sent = np.zeros(NUM_MOTORS)
        applied = sent + kp * (q_ref[k] - qk) + kd * (dq_ref[k] - vk)
        applied = np.clip(applied, -MOTOR_TAU_LIMIT, MOTOR_TAU_LIMIT)
        tau_ap[k] = applied
        acc = pinocchio.aba(plant_model, data, qk, vk, applied)
        vk = vk + dt * acc                 # semi-implicit Euler
        qk = qk + dt * vk
    return q, dq, tau_ap


def perturbed_model(scale=1.2, links=(2, 3)):
    """Design-model copy with link masses scaled: a deliberate plant mismatch."""
    model = T.build_arm_model()
    for b in links:
        if b + 1 < model.njoints:
            I = model.inertias[b + 1]
            model.inertias[b + 1] = pinocchio.Inertia(
                I.mass * scale, I.lever, I.inertia * scale)
    return model


def build_plan(args, seed=None, verbose=True):
    """EE path -> validated joint reference -> TVLQR gains. Raises on infeasibility."""
    model = T.build_arm_model()
    kw = {}
    if args.shape in ("hold", "line"):
        kw["p0"] = args.p0
    if args.shape == "line":
        kw["p1"] = args.p1
    if args.shape == "circle":
        kw.update(center=args.center, radius=args.radius, axis=args.axis,
                  turns=args.turns)
    if args.shape == "waypoints":
        pts = np.asarray(args.points, float).reshape(-1, 3)
        kw["points"] = pts

    t, p_ee, v_ee = T.build_ee_path(args.shape, args.dt, args.duration, **kw)
    q_ref, dq_ref, ddq_ref, tau_ref = T.make_joint_reference(
        t, p_ee, v_ee, rpy=args.rpy, model=model, seed=seed)

    issues = T.validate_reference(q_ref, dq_ref, tau_ref, p_ee=p_ee, model=model)
    if issues:
        raise T.ReferenceError("infeasible reference:\n  - " + "\n  - ".join(issues))

    K = tvlqr_gains(model, q_ref, dq_ref, tau_ref, args.dt,
                    q_pos=args.q_pos, q_vel=args.q_vel, r_tau=args.r_tau,
                    task_space=args.task_space, w_ee=args.w_ee,
                    w_rot=getattr(args, "w_rot", W_ROT))
    if verbose:
        print(f"[tvlqr] {args.shape}: N={len(t)} dt={args.dt}s dur={args.duration}s  "
              f"cost={'task-space' if args.task_space else 'joint-space'}")
        print(f"[tvlqr] |dq_ref|max={np.abs(dq_ref).max():.2f} rad/s  "
              f"|tau_ref|max={np.abs(tau_ref).max():.2f} Nm  "
              f"|K|max={np.abs(K).max():.1f}")
    return dict(t=t, p_ee=p_ee, v_ee=v_ee, q_ref=q_ref, dq_ref=dq_ref,
                ddq_ref=ddq_ref, tau_ref=tau_ref, K=K, model=model)


class TVLQRController:
    """DDS controller tracking a precomputed plan with TVLQR + hardware PD.

    Same thread-safe snapshot, debounced watchdog and guaranteed safe return as
    ``verify_gravity.py`` and ``sysid/collect_data.py``.
    """

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
            print(f"[tvlqr] safe_return stopped: {e}")

    def track(self, plan, dt=DT, track_abort=TRACK_ABORT_RAD):
        """Execute the plan. Returns a log dict of measured vs reference."""
        q_ref, dq_ref, tau_ref, K = (plan["q_ref"], plan["dq_ref"],
                                     plan["tau_ref"], plan["K"])
        n = len(q_ref)
        self._trip[:] = 0
        q = np.zeros((n, NUM_MOTORS)); dq = np.zeros((n, NUM_MOTORS))
        tau_m = np.zeros((n, NUM_MOTORS)); tau_c = np.zeros((n, NUM_MOTORS))
        abort_reason = ""
        used = 0
        t_start = time.perf_counter()
        try:
            for k in range(n):
                t0 = time.perf_counter()
                qk, dqk, tauk, arrival, valid = self.snapshot()
                self._watchdog(qk, dqk, tauk, arrival, valid)

                x_err = np.concatenate([qk - q_ref[k], dqk - dq_ref[k]])
                if np.max(np.abs(x_err[:NUM_MOTORS])) > track_abort:
                    j = int(np.argmax(np.abs(x_err[:NUM_MOTORS])))
                    raise RuntimeError(
                        f"tracking error {x_err[j]:+.3f} rad on "
                        f"{arm_ik.IK_MODEL.names[j+1]} exceeds {track_abort} rad")

                sent = tau_to_send(tau_ref[k], K[k], x_err, self.kp, self.kd)
                self._write(q_ref[k], dq_ref[k], sent)

                q[k] = qk; dq[k] = dqk; tau_m[k] = tauk; tau_c[k] = sent
                used = k + 1
                # Pace against the trajectory clock, not the loop, so tracking
                # does not drift in time.
                sleep = (k + 1) * dt - (time.perf_counter() - t_start)
                if sleep > 0:
                    time.sleep(sleep)
                elif k % 200 == 0:
                    print(f"[tvlqr] WARNING: loop overran by {-sleep*1000:.1f} ms")
        except (RuntimeError, TimeoutError, ValueError) as e:
            abort_reason = str(e)
            print(f"\n[tvlqr] ABORT: {abort_reason}")

        return dict(q=q[:used], dq=dq[:used], tau_meas=tau_m[:used],
                    tau_cmd=tau_c[:used], q_ref=q_ref[:used], dq_ref=dq_ref[:used],
                    complete=(not abort_reason and used == n),
                    abort_reason=abort_reason)


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


def _plot(plan, sim=None, out="ee_tvlqr.png"):
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


def main() -> int:
    ap = argparse.ArgumentParser(
        description="EE trajectory tracking with TVLQR.",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("net", nargs="?", default="lo",
                    help="DDS interface for on-robot tracking (e.g. eth0)")
    ap.add_argument("--dry-run", action="store_true",
                    help="plan + validate + plot only; no DDS, no hardware")
    ap.add_argument("--simulate", action="store_true",
                    help="offline closed-loop test against a perturbed plant")
    # trajectory
    ap.add_argument("--shape", default="line",
                    choices=["hold", "line", "circle", "waypoints"])
    # z targets are in the shared URDF/MJCF base frame. They read 0.43/0.53/0.45 while
    # model/robot.urdf mounted the arm 72.735 mm lower than the MJCF; +0.072735 keeps the
    # same physical motion now that the two frames agree.
    ap.add_argument("--p0", type=float, nargs=3, default=[0.205, 0.0, 0.502735])
    ap.add_argument("--p1", type=float, nargs=3, default=[0.205, 0.0, 0.602735])
    ap.add_argument("--center", type=float, nargs=3, default=[0.205, 0.0, 0.522735])
    ap.add_argument("--radius", type=float, default=0.05)
    ap.add_argument("--axis", type=float, nargs=3, default=[0.0, 1.0, 0.0])
    ap.add_argument("--turns", type=float, default=1.0)
    ap.add_argument("--points", type=float, nargs="+", default=None,
                    help="waypoints as a flat x y z x y z ... list")
    ap.add_argument("--rpy", type=float, nargs=3, default=None,
                    help="fixed EE orientation to hold, as roll pitch yaw [rad]. "
                         "The IK ALWAYS solves a full 6-DOF pose; omitting this means "
                         "identity orientation, not position-only.")
    ap.add_argument("--duration", type=float, default=4.0)
    ap.add_argument("--dt", type=float, default=DT)
    # cost
    # Task-space weighting is the DEFAULT: this tool tracks an EE pose, and
    # J^T W J measurably beats joint-space weighting on EE error (see --simulate).
    ap.add_argument("--joint-space", dest="task_space", action="store_false",
                    help="weight joint error instead of EE error (default: EE)")
    ap.set_defaults(task_space=True)
    ap.add_argument("--w-ee", type=float, default=W_EE,
                    help="task-space POSITION weight [1/m^2]")
    ap.add_argument("--w-rot", type=float, default=W_ROT,
                    help="task-space ORIENTATION weight [1/rad^2]; 0 = position-only cost")
    ap.add_argument("--q-pos", type=float, default=Q_POS)
    ap.add_argument("--q-vel", type=float, default=Q_VEL)
    ap.add_argument("--r-tau", type=float, default=R_TAU)
    # runtime
    ap.add_argument("--kp", type=float, nargs=NUM_MOTORS, default=KP.tolist())
    ap.add_argument("--kd", type=float, nargs=NUM_MOTORS, default=KD.tolist())
    ap.add_argument("--track-abort", type=float, default=TRACK_ABORT_RAD)
    ap.add_argument("--mass-perturb", type=float, default=1.2,
                    help="--simulate: plant link-mass scale (model mismatch)")
    ap.add_argument("--no-plots", action="store_true")
    args = ap.parse_args()

    # Validate EVERY numeric arg before any DDS init. NaN passes every ordinary
    # comparison (`nan <= 0` is False), so finiteness is checked first: missing
    # entries previously let `--kp nan` reach the motors and `--track-abort nan`
    # silently disable the abort check.
    scalars = (("--duration", args.duration), ("--dt", args.dt),
               ("--radius", args.radius), ("--turns", args.turns),
               ("--w-ee", args.w_ee), ("--w-rot", args.w_rot),
               ("--q-pos", args.q_pos), ("--q-vel", args.q_vel),
               ("--r-tau", args.r_tau), ("--track-abort", args.track_abort),
               ("--mass-perturb", args.mass_perturb))
    vectors = (("--p0", args.p0), ("--p1", args.p1), ("--center", args.center),
               ("--axis", args.axis), ("--kp", args.kp), ("--kd", args.kd),
               ("--rpy", args.rpy), ("--points", args.points))
    for label, v in scalars:
        if not np.isfinite(v):
            ap.error(f"{label} must be finite")
    for label, v in vectors:
        if v is not None and not np.all(np.isfinite(np.asarray(v, dtype=float))):
            ap.error(f"{label} must be finite")

    if args.r_tau <= 0:
        ap.error("--r-tau must be positive (R must be invertible)")
    # Negative state weights make Q indefinite and void the Riccati guarantees
    # (measured: --q-vel -1 raised |K|max from 117 to ~32000).
    for label, v in (("--q-pos", args.q_pos), ("--q-vel", args.q_vel),
                     ("--w-ee", args.w_ee), ("--w-rot", args.w_rot)):
        if v < 0:
            ap.error(f"{label} must be nonnegative (a negative weight makes Q indefinite)")
    for label, v in (("--kp", args.kp), ("--kd", args.kd)):
        if np.any(np.asarray(v, dtype=float) < 0):
            ap.error(f"{label} must be nonnegative (negative feedback gain is destabilizing)")
    if args.track_abort <= 0:
        ap.error("--track-abort must be positive (a nonpositive/NaN threshold never trips)")
    if args.mass_perturb <= 0:
        ap.error("--mass-perturb must be positive")
    if args.shape == "waypoints" and (not args.points or len(args.points) % 3):
        ap.error("--points needs a multiple of 3 values (x y z per waypoint)")

    # Shared plan for all modes.
    try:
        plan = build_plan(args)
    except (T.ReferenceError, ValueError) as e:
        print(f"[tvlqr] ERROR: {e}")
        return 2

    if args.dry_run:
        print("[tvlqr] reference is feasible (IK, joint limits, dq/tau caps all OK)")
        if not args.no_plots:
            _plot(plan)
        return 0

    if args.simulate:
        plant = perturbed_model(args.mass_perturb)
        # Start off-reference so the feedback has something to correct.
        q0 = plan["q_ref"][0] + 0.05
        kp, kd = np.asarray(args.kp), np.asarray(args.kd)
        runs = {m: simulate(plan, plant, args.dt, kp, kd, q0=q0, mode=m)
                for m in ("pd", "ff", "lqr")}
        print(f"\n[sim] plant link mass x{args.mass_perturb}, "
              "start offset 0.05 rad/joint")
        labels = {"pd": "PD only (no feedforward)",
                  "ff": "PD + feedforward (honest baseline)",
                  "lqr": "TVLQR + feedforward"}
        ee, rot = {}, {}
        for m in ("pd", "ff", "lqr"):
            _, ee[m], rot[m] = report_tracking(runs[m][0], plan["q_ref"],
                                               plan["model"], plan["p_ee"], labels[m])
        # Judge TVLQR against PD+feedforward, not against bare PD: the feedforward
        # alone removes most of the gravity error, so comparing to "pd" would
        # overstate what the optimal FEEDBACK contributes.
        better = ee["lqr"] < ee["ff"]
        print(f"\n[sim] EE pos RMS  pd {ee['pd']*1000:7.2f} mm | ff {ee['ff']*1000:7.2f} mm "
              f"| tvlqr {ee['lqr']*1000:7.2f} mm")
        print(f"[sim] EE rot RMS  pd {np.degrees(rot['pd']):7.2f} deg | "
              f"ff {np.degrees(rot['ff']):7.2f} deg | tvlqr {np.degrees(rot['lqr']):7.2f} deg")
        print(f"[sim] TVLQR vs PD+feedforward: position "
              f"{'IMPROVED' if better else 'NO IMPROVEMENT'} "
              f"({ee['ff']/max(ee['lqr'],1e-9):.2f}x), orientation "
              f"{rot['ff']/max(rot['lqr'],1e-9):.2f}x")
        if not args.no_plots:
            _plot(plan, sim=dict(q=runs["lqr"][0]))
        return 0 if better else 1

    # On-robot tracking
    from unitree_sdk2py.core.channel import ChannelFactoryInitialize
    print("WARNING: clear the workspace. The arm will move along the EE trajectory.")
    input("Press Enter to start TVLQR tracking...")
    ChannelFactoryInitialize(1, args.net)

    ctrl = None
    log = None
    interrupted = False
    try:
        ctrl = TVLQRController(kp=np.asarray(args.kp), kd=np.asarray(args.kd))
        ctrl.wait_for_state()
        print("[tvlqr] state received; ramping to trajectory start...")
        ctrl.ramp_to(plan["q_ref"][0], duration=3.0, dt=args.dt)
        print(f"[tvlqr] tracking {args.shape} for {args.duration}s...")
        log = ctrl.track(plan, dt=args.dt, track_abort=args.track_abort)
    except KeyboardInterrupt:
        print("\n[tvlqr] interrupted.")
        interrupted = True
    finally:
        if ctrl is not None and ctrl.has_state():
            ctrl.safe_return(dt=args.dt)

    if log is not None:
        if len(log["q"]):
            report_tracking(log["q"], log["q_ref"], plan["model"],
                            plan["p_ee"], "on-robot tracking")
        else:
            print("[tvlqr] no samples logged (aborted on the first control step)")
        if not bool(log["complete"]):
            print(f"[tvlqr] INCOMPLETE: {log['abort_reason']}")
    elif not interrupted:
        print("[tvlqr] tracking never started (see the error above)")
    return _exit_code(log, interrupted)


if __name__ == "__main__":
    sys.exit(main())
