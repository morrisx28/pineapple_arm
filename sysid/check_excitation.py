"""Size and verify system-identification excitation without hardware.

The rollout mirrors the bridge's torque-clamped PD law:

    ctrl = kp*(q_cmd - q) - kd*dq,  then clamp to the actuator ctrlrange

It sizes amplitudes below torque and velocity caps, reports saturation, and checks
observability by recovering injected parameters. Avoiding saturation is essential because
armature information appears through acceleration.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

import absl.logging
import mujoco
from mujoco import sysid

absl.logging.set_verbosity(absl.logging.ERROR)  # silence allow_missing_sensors info

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sysid_common as C
import collect_data as cd
from sim2sim_selftest import TRUE  # large distinct known params (default GT)

# Small ground truth near the real regime makes marginal armature observability visible.
SMALL_GT = {
    "armature":     np.array([0.004, 0.006, 0.005, 0.007, 0.004, 0.005]),
    "frictionloss": np.array([0.10, 0.13, 0.11, 0.09, 0.08, 0.09]),
    "damping":      np.array([0.02, 0.03, 0.025, 0.015, 0.02, 0.02]),
}

# Saturating baseline retained for ``--compare-old``.
OLD_AMP = np.array([0.6, 0.6, 0.6, 0.5, 0.7, 0.7])
OLD_F1 = 2.0
OLD_COMBINED = 0.6


def simulate_clamped_pd(model, q_cmd, kp, kd):
    """Bridge-accurate rollout: clamped-PD torque -> mj_step.
    -> q, dq, tau_applied (post-clamp, == tau_est), tau_cmd (pre-clamp)."""
    dt = model.opt.timestep
    lo = model.actuator_ctrlrange[:, 0].copy()
    hi = model.actuator_ctrlrange[:, 1].copy()
    data = mujoco.MjData(model)
    data.qpos[:] = q_cmd[0]
    mujoco.mj_forward(model, data)
    n = len(q_cmd)
    q = np.zeros((n, C.NUM_MOTORS)); dq = np.zeros((n, C.NUM_MOTORS))
    tau_ap = np.zeros((n, C.NUM_MOTORS)); tau_cmd = np.zeros((n, C.NUM_MOTORS))
    for k in range(n):
        q[k] = data.qpos; dq[k] = data.qvel
        u = kp * (q_cmd[k] - data.qpos) - kd * data.qvel
        tau_cmd[k] = u
        uc = np.clip(u, lo, hi)
        tau_ap[k] = uc
        data.ctrl[:] = uc
        mujoco.mj_step(model, data)
    return q, dq, tau_ap, tau_cmd


def _single_joint_cmd(j, a, f1j, dt, dur=10.0, f0=0.1, tri_hz=0.1):
    """One joint's chirp+triangle around CENTER, others held (matches the
    per-joint phase of build_excitation)."""
    k = max(2, int(dur / dt))
    tj = np.arange(k) * dt
    chirp = np.sin(2 * np.pi * (f0 * tj + 0.5 * (f1j - f0) / (tj[-1] + 1e-9) * tj**2))
    tri = 2.0 / np.pi * np.arcsin(np.sin(2 * np.pi * tri_hz * tj))
    envelope = cd._smooth_envelope(k, dt, taper_s=0.5)
    qc = np.tile(cd.CENTER, (k, 1))
    qc[:, j] = cd.CENTER[j] + a * envelope * (0.7 * chirp + 0.3 * tri)
    return np.clip(qc, cd.JOINT_LOW + 0.05, cd.JOINT_HIGH - 0.05)


def size_amplitudes(model, kp, kd, f1, dt, per_joint_s=14.0,
                    margin_tau=0.8, margin_dq=0.85, iters=30):
    """Size per-joint amplitudes so the full trajectory's realized peak |tau|/|dq|
    sit near margin*caps in the SYNCHRONOUS model.

    Simulates the WHOLE excitation and attributes each over-cap peak to the joint
    actually MOVING, so the coupling culprit is scaled, not just the victim (a held
    joint spikes when a neighbour sweeps it through gravity).

    WARNING: margins are <1 because the async-DDS simulator runs hotter than this
    synchronous check. Always re-verify realized peaks with --analyze.
    """
    cap = np.minimum(cd._range_cap(), cd.DESIGN_AMP_MAX)
    hi_ctrl = np.abs(model.actuator_ctrlrange[:, 1])
    # Never above the design cap, and safely below the motor's real ctrlrange --
    # else the joint saturates, as j3/j4 did (+-7 Nm motors under a 10 Nm cap).
    tau_lim = np.minimum(margin_tau * np.asarray(cd.TARGET_TAU, float), 0.9 * hi_ctrl)
    dq_lim = margin_dq * np.asarray(cd.TARGET_DQ, float)
    amp = np.minimum(cap, np.full(C.NUM_MOTORS, 0.3))
    peak_tau = np.zeros(C.NUM_MOTORS)
    peak_dq = np.zeros(C.NUM_MOTORS)
    for _ in range(iters):
        _, qc = cd.build_excitation(dt=dt, per_joint_s=per_joint_s, amp=amp,
                                    f1=f1, combined_s=0.0)
        _, dq, tau_ap, _ = simulate_clamped_pd(model, qc, kp, kd)
        peak_tau = np.abs(tau_ap).max(0)
        peak_dq = np.abs(dq).max(0)
        disp = np.abs(qc - cd.CENTER)
        moving = disp.max(1) > 1e-3                              # exclude hold/settle
        active = np.where(moving, np.argmax(disp, axis=1), -1)  # joint moving each step
        ratio = np.minimum(tau_lim / np.maximum(peak_tau, 1e-6),
                           dq_lim / np.maximum(peak_dq, 1e-6))     # victim scaling
        for j in range(C.NUM_MOTORS):
            if peak_tau[j] <= tau_lim[j]:
                continue
            # shrink EVERY joint that is moving while joint j is over its limit
            # (coupling spikes on a gravity-loaded joint come from all of them,
            # not just the single worst instant -> avoids whack-a-mole).
            r_j = tau_lim[j] / peak_tau[j]
            for c in np.unique(active[np.abs(tau_ap[:, j]) > tau_lim[j]]):
                if c >= 0:
                    ratio[int(c)] = min(ratio[int(c)], r_j)
        prev_amp = amp
        amp = np.minimum(amp * np.clip(ratio ** 0.6, 0.5, 1.25), cap)  # damped
        # Stop only once the amplitudes have CONVERGED. The old code broke as soon
        # as every peak was under its limit, so the "gentle growth toward the cap"
        # never actually happened and the excitation stayed weaker than necessary.
        if np.all(peak_tau <= tau_lim + 1e-3) and np.all(peak_dq <= dq_lim + 1e-2) \
                and np.allclose(amp, prev_amp, rtol=1e-3, atol=1e-4):
            break
    # Ensure reported peaks correspond to the returned amplitude even if the
    # iteration budget was exhausted immediately after an update.
    _, qc = cd.build_excitation(
        dt=dt, per_joint_s=per_joint_s, amp=amp, f1=f1, combined_s=0.0
    )
    _, dq, tau_ap, _ = simulate_clamped_pd(model, qc, kp, kd)
    peak_tau = np.abs(tau_ap).max(0)
    peak_dq = np.abs(dq).max(0)
    rows = [(C.JOINTS[j], amp[j], peak_tau[j], peak_dq[j],
             cd.TARGET_TAU[j], cd.TARGET_DQ[j], 0.0,
             "tau" if peak_tau[j] >= tau_lim[j] - 1e-2 else "dq/interior")
            for j in range(C.NUM_MOTORS)]
    return amp, rows


def report_peaks(model, t, q_cmd, kp, kd, label):
    q, dq, tau_ap, tau_cmd = simulate_clamped_pd(model, q_cmd, kp, kd)
    dt = model.opt.timestep
    ddq = np.gradient(dq, dt, axis=0)
    print(f"\n=== realized peaks over full excitation [{label}] "
          f"({len(t)} steps, {t[-1]:.0f}s) ===")
    print(f"{'joint':16s} {'|tau|max':>8s} {'cap':>5s} {'ctrlrng':>7s} {'sat%':>6s} "
          f"{'|dq|max':>8s} {'cap':>5s} {'ddqRMS':>7s}")
    ok = True
    for j in range(C.NUM_MOTORS):
        ptau = np.abs(tau_ap[:, j]).max()
        crng = abs(model.actuator_ctrlrange[j, 1])
        sat = 100.0 * np.mean(np.abs(tau_cmd[:, j]) > crng - 1e-9)
        pdq = np.abs(dq[:, j]).max()
        flag = "" if (ptau <= cd.TARGET_TAU[j] + 1e-2 and pdq <= cd.TARGET_DQ[j] + 1e-1 and sat == 0) else "  <-- over"
        ok = ok and flag == ""
        print(f"{C.JOINTS[j]:16s} {ptau:8.2f} {cd.TARGET_TAU[j]:5.0f} {crng:7.0f} "
              f"{sat:6.1f} {pdq:8.2f} {cd.TARGET_DQ[j]:5.0f} {np.sqrt(np.mean(ddq[:,j]**2)):7.1f}{flag}")
    print(f"  -> {'PASS: within caps, no saturation' if ok else 'OVER caps/saturating'}")
    return ok


def observability(amp, f1, combined_scale, kp, kd, dt, label, gt=TRUE,
                  per_joint_s=8.0, combined_s=8.0, max_relerr=0.25):
    """Inject known params, regenerate, refit, report recovery + 95% CI.

    True only if the optimizer converged AND armature recovery is within
    ``max_relerr`` (median) / ``2*max_relerr`` (worst), so a useless excitation
    cannot silently report success."""
    t, q_cmd = cd.build_excitation(dt=dt, per_joint_s=per_joint_s, combined_s=combined_s,
                                   amp=amp, f1=f1, combined_scale=combined_scale)
    # ground truth (distinct, known)
    gspec = C.fresh_spec(drive="torque", dt=dt)
    for a in C.ALL_ATTRS:
        for j, jn in enumerate(C.JOINTS):
            C.set_joint_dyn(gspec, jn, a, gt[a][j])
    gmodel = gspec.compile()
    q, dq, tau, _ = simulate_clamped_pd(gmodel, q_cmd, kp, kd)

    fspec = C.fresh_spec(drive="torque", dt=dt)
    ms = C.make_model_sequences(fspec, t, tau, q, dq, seg_steps=100)
    params = C.build_param_dict()
    rfn = sysid.build_residual_fn(models_sequences=[ms])
    opt, res = sysid.optimize(params, rfn, optimizer="scipy_parallel_fd",
                              verbose=False, max_iters=40, x_scale="jac")
    names = opt.get_non_frozen_parameter_names()
    est = opt.as_vector()
    try:
        rstar, _, _ = rfn(res.x, opt, return_pred_all=True)
        _, ci = sysid.calculate_intervals(rstar, res.jac)
    except Exception:
        ci = np.full(len(names), np.nan)
    gtn = {n: gt[n.split("/")[1]][C.JOINTS.index(n.split("/")[0])] for n in names}
    print(f"\n=== observability [{label}] : recover distinct GT, +-95% CI ===")
    print(f"{'param':30s} {'GT':>9s} {'ident':>9s} {'relerr':>7s} {'95%CI':>9s}")
    arm_err = []
    for n, v, c in zip(names, est, ci):
        g = gtn[n]
        rel = abs(v - g) / max(abs(g), 1e-9)
        if n.endswith("/armature"):
            arm_err.append(rel)
        print(f"{n:30s} {g:9.4g} {v:9.4g} {rel*100:6.1f}% {c:9.2g}")
    med, mx = float(np.median(arm_err)), float(np.max(arm_err))
    ok = bool(getattr(res, "success", False)) and med <= max_relerr and mx <= 2.0 * max_relerr
    print(f"  -> armature median rel err: {med*100:.1f}%  max: {mx*100:.1f}%  "
          f"(limits {max_relerr*100:.0f}%/{2*max_relerr*100:.0f}%)")
    if not bool(getattr(res, "success", False)):
        print("  -> FAIL: optimizer did not converge")
    print(f"  -> {'PASS: excitation identifies armature' if ok else 'FAIL: poor recovery'}")
    return ok


def analyze_data(path, dt=None):
    """Report REALIZED per-joint peaks/saturation/tracking from a COLLECTED .npz.

    The faithful check -- the offline sizer under-predicts the async sim. No fit.
    """
    log = C.load_log(path)
    try:
        C.validate_log(log, drive="pd")
    except (TypeError, ValueError) as e:
        print(f"[analyze] ERROR: invalid log: {e}")
        return 2
    t = np.asarray(log["t"], float)
    q, dq, tau = (np.asarray(log[k], float) for k in ("q", "dq", "tau"))
    q_cmd = np.asarray(log["q_cmd"], float)
    dt = dt or float(np.median(np.diff(t)))
    model = C.fresh_spec(drive="torque", dt=dt).compile()
    crng = np.abs(model.actuator_ctrlrange[:, 1])
    dt_std = float(np.std(np.diff(t)))
    print(f"[analyze] {path}")
    print(f"[analyze] N={len(t)} dur={t[-1]-t[0]:.1f}s dt={dt:.4f}s "
          f"dt_std={dt_std:.1e} ({'uniform/synthetic grid' if dt_std < 1e-9 else 'real/jittery'})")
    print(f"{'joint':16s} {'|tau|':>6s} {'cap':>4s} {'ctrl':>5s} {'sat%':>5s} "
          f"{'|dq|':>6s} {'cap':>4s} {'trackRMS':>8s} {'motionSD':>8s}")
    ok = True
    for j in range(C.NUM_MOTORS):
        ptau = np.abs(tau[:, j]).max()
        sat = 100.0 * np.mean(np.abs(tau[:, j]) >= 0.99 * crng[j])
        pdq = np.abs(dq[:, j]).max()
        # True RMS about zero: std would subtract a constant tracking offset,
        # which is itself a tracking failure.
        trk = float(np.sqrt(np.mean((q[:, j] - q_cmd[:, j]) ** 2)))
        mot = float(np.std(q[:, j]))
        over_tau = ptau > cd.TARGET_TAU[j] + 0.1
        over_dq = pdq > cd.TARGET_DQ[j] + 0.5
        bad_track = trk > 0.5 * max(mot, 1e-6)
        flag = ""
        if over_tau or sat > 0: flag += " TAU!"
        if over_dq: flag += " DQ!"
        if bad_track: flag += " TRACK!"
        ok = ok and not flag
        print(f"{C.JOINTS[j]:16s} {ptau:6.2f} {cd.TARGET_TAU[j]:4.0f} {crng[j]:5.0f} "
              f"{sat:5.2f} {pdq:6.2f} {cd.TARGET_DQ[j]:4.0f} {trk:8.4f} {mot:8.4f}{flag}")
    print(f"  -> {'PASS: within caps and tracking well' if ok else 'ISSUES flagged above'}")
    print("  (TAU!=over torque cap/saturating, DQ!=over velocity cap, "
          "TRACK!=tracking error > half the motion). Next: sysid_fit.py --data " + path)
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--compare-old", action="store_true",
                    help="also evaluate the previous (saturating) excitation")
    ap.add_argument("--dt", type=float, default=C.DT_DEFAULT)
    ap.add_argument("--max-relerr", type=float, default=0.25,
                    help="max median armature relative error for the observability "
                         "check to PASS (worst-case limit is 2x this)")
    ap.add_argument("--no-observability", action="store_true",
                    help="skip the (slower) synthetic recovery fit")
    ap.add_argument("--small-gt", action="store_true",
                    help="use a small GT near the arm's real regime (armature ~0.004) "
                         "for the observability test -- exposes the excitation effect")
    ap.add_argument("--analyze", metavar="NPZ", default=None,
                    help="analyze a COLLECTED .npz: realized peaks/saturation/tracking "
                         "vs caps (the faithful check; no sizing/fit)")
    args = ap.parse_args()

    if args.analyze:
        return analyze_data(args.analyze, dt=args.dt if args.dt != C.DT_DEFAULT else None)

    gt = SMALL_GT if args.small_gt else TRUE

    kp, kd = C.DEFAULT_KP, C.DEFAULT_KD
    nominal = C.fresh_spec(drive="torque", dt=args.dt).compile()  # GT-defaults model

    # Size amplitudes to the caps.
    print("Sizing per-joint amplitudes to TARGET_TAU/TARGET_DQ (bridge-clamped PD)...")
    amp_new, rows = size_amplitudes(nominal, kp, kd, cd.F1, args.dt)
    print(f"\n{'joint':16s} {'amp[rad]':>8s} {'|tau|':>6s} {'|dq|':>6s} "
          f"{'tauCap':>6s} {'dqCap':>6s} {'hold':>5s} {'binds':>10s}")
    for jn, a, pt, pd, tc, dc, pt0, bind in rows:
        print(f"{jn:16s} {a:8.3f} {pt:6.2f} {pd:6.2f} {tc:6.0f} {dc:6.0f} {pt0:5.2f} {bind:>10s}")
    print("\nRecommended constants for collect_data.py:")
    print(f"  AMP = np.array([{', '.join(f'{a:.3f}' for a in amp_new)}])")
    print(f"  F1  = np.array([{', '.join(f'{f:.2f}' for f in cd.F1)}])")

    # Verify the full sequential excitation.
    t, q_new = cd.build_excitation(dt=args.dt, per_joint_s=14.0, amp=amp_new,
                                   f1=cd.F1, combined_s=0.0)
    # Fail-closed so CI cannot treat printed failures as success.
    peaks_ok = report_peaks(nominal, t, q_new, kp, kd, "NEW (sized)")

    if args.compare_old:
        t_o, q_old = cd.build_excitation(dt=args.dt, per_joint_s=12.0, amp=OLD_AMP,
                                         f1=OLD_F1, combined_s=15.0,
                                         combined_scale=OLD_COMBINED)
        report_peaks(nominal, t_o, q_old, kp, kd, "OLD (previous)")  # reference only

    # Verify armature observability.
    obs_ok = True
    if not args.no_observability:
        obs_ok = observability(amp_new, cd.F1, 0.0, kp, kd, args.dt, "NEW (sized)",
                               gt=gt, per_joint_s=5.0, combined_s=0.0,
                               max_relerr=args.max_relerr)
        if args.compare_old:
            observability(OLD_AMP, OLD_F1, OLD_COMBINED, kp, kd, args.dt,
                          "OLD (previous)", gt=gt, per_joint_s=5.0, combined_s=6.0,
                          max_relerr=args.max_relerr)  # reference only

    if peaks_ok and obs_ok:
        return 0
    print("\n[check] FAILED: "
          + ", ".join(m for m, k in (("excitation exceeds caps", not peaks_ok),
                                     ("armature not identifiable", not obs_ok)) if k))
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
