"""Recover injected dynamics parameters in a hardware-free sim-to-sim test.

PD-driven chirps keep range-limited joints moving without saturating at their stops, then
the normal fitting pipeline starts from nominal parameters and checks recovery.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

import absl.logging
import mujoco
from mujoco import sysid

absl.logging.set_verbosity(absl.logging.ERROR)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sysid_common as C


# Injected parameters in motor order, including wrist-like values for j5.
TRUE = {
    "armature":     np.array([0.10, 0.030, 0.025, 0.008, 0.006, 0.006]),
    "frictionloss": np.array([0.20, 0.50, 0.45, 0.20, 0.15, 0.15]),
    "damping":      np.array([0.10, 0.40, 0.35, 0.10, 0.08, 0.08]),
}


def apply_true(spec: mujoco.MjSpec) -> None:
    for attr, vals in TRUE.items():
        for j, jname in enumerate(C.JOINTS):
            C.set_joint_dyn(spec, jname, attr, vals[j])


def chirp_reference(t, centers, amp=0.25, f0=0.1, f1=1.5):
    """Per-joint frequency-swept position reference around ``centers``."""
    T = t[-1]
    phase = 2 * np.pi * (f0 * t + 0.5 * (f1 - f0) / T * t**2)
    ref = np.zeros((len(t), C.NUM_MOTORS))
    for j in range(C.NUM_MOTORS):
        ref[:, j] = centers[j] + amp * np.sin(phase + 0.7 * j)
    return ref


def generate_pd(true_spec, t, q_cmd, kp, kd, seed=0, noise=0.0):
    """PD-control the true (torque) model to track ``q_cmd``.

    Returns aligned arrays of length ``len(t)`` (toolbox convention: ``tau[k]``
    is applied at ``t[k]`` and produces the state at ``t[k+1]``; the last torque
    is unused by the fitter):
      q, dq   : measured joint position / velocity
      tau     : applied, actuator-clamped joint torque
      q0, dq0 : initial state
    """
    model = true_spec.compile()
    data = mujoco.MjData(model)
    data.qpos[:] = q_cmd[0]
    mujoco.mj_forward(model, data)

    n = len(t)
    q = np.zeros((n, C.NUM_MOTORS))
    dq = np.zeros((n, C.NUM_MOTORS))
    tau = np.zeros((n, C.NUM_MOTORS))
    lo = model.actuator_ctrlrange[:, 0]
    hi = model.actuator_ctrlrange[:, 1]
    for k in range(n):
        q[k] = data.qpos
        dq[k] = data.qvel
        tau_cmd = kp * (q_cmd[k] - data.qpos) - kd * data.qvel
        tau[k] = np.clip(tau_cmd, lo, hi)  # PD, dq_des = 0
        data.ctrl[:] = tau[k]
        mujoco.mj_step(model, data)

    if noise > 0:
        rng = np.random.default_rng(seed)
        q = q + rng.normal(0, noise, q.shape)
        dq = dq + rng.normal(0, 10 * noise, dq.shape)
    return q, dq, tau


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--drive", choices=["torque", "pd"], default="torque")
    ap.add_argument("--optimizer", default="scipy_parallel_fd",
                    choices=["scipy", "scipy_parallel_fd", "mujoco"])
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--dt", type=float, default=C.DT_DEFAULT)
    ap.add_argument("--noise", type=float, default=0.0,
                    help="std of Gaussian position noise added to measured q [rad]")
    ap.add_argument("--max-iters", type=int, default=100)
    ap.add_argument("--seg", type=int, default=100,
                    help="multiple-shooting segment length in steps")
    ap.add_argument("--tol", type=float, default=0.10,
                    help="max allowed median relative error to PASS")
    ap.add_argument("--max-tol", type=float, default=0.25,
                    help="max allowed worst-parameter relative error to PASS")
    args = ap.parse_args()

    t = np.arange(args.steps) * args.dt
    kp, kd = C.DEFAULT_KP, C.DEFAULT_KD

    # Centre on each joint's range midpoint so the chirp stays within limits.
    m0 = C.fresh_spec(drive="torque", dt=args.dt).compile()
    centers = np.array([m0.jnt_range[i].mean() for i in range(m0.njnt)])
    q_cmd = chirp_reference(t, centers)

    true_spec = C.fresh_spec(drive="torque", dt=args.dt)  # gravity ON (realistic)
    apply_true(true_spec)
    q_meas, dq_meas, tau_meas = generate_pd(
        true_spec, t, q_cmd, kp, kd, seed=2, noise=args.noise
    )
    print(f"[selftest] drive={args.drive} steps={args.steps} dt={args.dt} "
          f"noise={args.noise}")
    print(f"[selftest] q range [{q_meas.min():.2f}, {q_meas.max():.2f}] rad, "
          f"|dq|max {np.abs(dq_meas).max():.2f} rad/s, "
          f"|tau|max {np.abs(tau_meas).max():.2f} Nm")

    ctrl = tau_meas if args.drive == "torque" else q_cmd
    fit_spec = C.fresh_spec(drive=args.drive, dt=args.dt, kp=kp, kd=kd)
    ms = C.make_model_sequences(fit_spec, t, ctrl, q_meas, dq_meas, seg_steps=args.seg)
    params = C.build_param_dict()  # nominal init = XML placeholders
    init_vec = params.as_vector().copy()

    residual_fn = sysid.build_residual_fn(models_sequences=[ms])
    opt_params, _ = sysid.optimize(
        params, residual_fn, optimizer=args.optimizer,
        verbose=False, max_iters=args.max_iters, x_scale="jac",
    )

    # build_param_dict order is [armature x6, frictionloss x6, damping x6].
    true_vec = np.concatenate([TRUE[a] for a in C.ALL_ATTRS])
    est_vec = opt_params.as_vector()
    print(opt_params.compare_parameters(init_vec, est_vec, measured_params=true_vec))

    rel = np.abs(est_vec - true_vec) / np.maximum(np.abs(true_vec), 1e-6)
    med, mx = float(np.median(rel)), float(np.max(rel))
    print(f"[selftest] relative error: median={med:.3%}  max={mx:.3%}")
    ok = med < args.tol and mx < args.max_tol
    print(f"[selftest] {'PASS' if ok else 'FAIL'} "
          f"(median={med:.3%}, max={mx:.3%}; limits "
          f"{args.tol:.1%}/{args.max_tol:.1%})")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
