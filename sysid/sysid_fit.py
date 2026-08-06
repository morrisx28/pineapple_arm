"""Fit per-joint armature, dry friction, and damping from a collector log.

The MuJoCo sysid fit uses box-constrained nonlinear least squares with threaded rollout
and multiple shooting.

Drive modes:
  * ``torque`` (default): replay measured joint torque. Most sensitive to the
    parameters; validated in ``sim2sim_selftest.py``.
  * ``pd``: replay the position command. Robust to a biased torque sensor, but the
    closed loop MASKS the parameters -- cross-check only.

An identified XML is emitted only when validation passes; diagnostic overrides remain
non-successful and visibly marked.
"""

from __future__ import annotations

import argparse
import os
import pathlib
import pickle
import sys

import numpy as np

import absl.logging
from mujoco import sysid

absl.logging.set_verbosity(absl.logging.ERROR)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sysid_common as C
import collect_data as D


def _infer_dt(t: np.ndarray) -> float:
    d = np.diff(np.asarray(t, dtype=float).ravel())
    if d.size == 0 or not np.all(np.isfinite(d)) or np.median(d) <= 0:
        raise ValueError("cannot infer a positive timestep from t")
    return float(np.median(d))


def _seg_steps(value: str) -> int:
    seg = int(value)
    if seg < 2:
        raise argparse.ArgumentTypeError("segment length must be >= 2")
    return seg


def _timing_artifacts(log: dict) -> tuple[int, np.ndarray, bool]:
    q = np.asarray(log["q"], float)
    t = np.asarray(log["t"], float).ravel()
    keep = C.dedup_mask(
        q, t, dq=log.get("dq"), tau=log.get("tau"), tick=log.get("tick")
    )
    dropped = int(len(t) - np.count_nonzero(keep))
    d = np.diff(t)
    jitter = d.size > 0 and np.median(d) > 0 and (
        np.std(d) / np.median(d) > 0.1
        or d.min() <= 0.0
        or d.max() > 1.5 * np.median(d)
    )
    return dropped, d, bool(jitter)


def _excitation_issues(log: dict, drive: str = "torque") -> list[str]:
    """Reasons a raw collector log is unsafe/uninformative to fit.

    Tracking error is split into DYNAMIC (does the joint follow the command's
    shape?) and constant BIAS (steady-state PD sag). Dynamic always matters. A bias
    corrupts ``--drive pd`` (the model must reproduce the offset) but is harmless
    for ``--drive torque``, which never uses ``q_cmd`` -- so it fails only for PD.
    """
    q = np.asarray(log["q"], float)
    dq = np.asarray(log["dq"], float)
    tau = np.asarray(log["tau"], float)
    issues: list[str] = []
    peak_tau = np.max(np.abs(tau), axis=0)
    peak_dq = np.max(np.abs(dq), axis=0)
    saturated = np.any(np.abs(tau) >= 0.99 * D.MOTOR_TAU_LIMIT, axis=0)
    for j, name in enumerate(C.JOINTS):
        reasons = []
        if peak_tau[j] > D.TARGET_TAU[j] + 0.1:
            reasons.append(
                f"|tau|max={peak_tau[j]:.2f}>{D.TARGET_TAU[j]:.2f} Nm"
            )
        if saturated[j]:
            reasons.append(f"torque reached the {D.MOTOR_TAU_LIMIT[j]:.0f} Nm rail")
        if peak_dq[j] > D.TARGET_DQ[j] + 0.5:
            reasons.append(f"|dq|max={peak_dq[j]:.2f}>{D.TARGET_DQ[j]:.2f} rad/s")
        if "q_cmd" in log:
            q_cmd = np.asarray(log["q_cmd"], float)
            err = q[:, j] - q_cmd[:, j]
            rms = float(np.sqrt(np.mean(err ** 2)))   # true RMS: includes the bias
            bias = float(np.mean(err))
            dynamic = float(np.std(err))              # bias-free tracking
            motion = float(np.std(q[:, j]))
            limit = 0.5 * max(motion, 1e-6)
            if dynamic > limit:
                reasons.append(
                    f"dynamic tracking={dynamic:.3g} > 0.5*motion={limit:.3g} "
                    f"(RMS={rms:.3g}, bias={bias:+.3g})"
                )
            elif abs(bias) > limit:
                msg = (f"constant tracking bias={bias:+.3g} > 0.5*motion={limit:.3g} "
                       f"(RMS={rms:.3g})")
                if drive == "pd":
                    reasons.append(msg + " -- PD replay must reproduce this offset")
                else:
                    print(f"[fit] NOTE: {name}: {msg}; harmless for torque replay "
                          "(q_cmd is not used), but it is a real steady-state sag.")
        if reasons:
            issues.append(f"{name}: " + ", ".join(reasons))
    return issues


def _bound_hits(params: sysid.ParameterDict, fraction: float = 1e-4) -> list[str]:
    hits = []
    for name in params.get_non_frozen_parameter_names():
        param = params[name]
        value = float(np.asarray(param.value).ravel()[0])
        lo = float(np.asarray(param.min_value).ravel()[0])
        hi = float(np.asarray(param.max_value).ravel()[0])
        tol = fraction * max(hi - lo, 1.0)
        if value <= lo + tol or value >= hi - tol:
            hits.append(name)
    return hits


def _confidence_issues(
    params: sysid.ParameterDict,
    names: list[str],
    estimates: np.ndarray,
    intervals: np.ndarray,
) -> list[str]:
    """Parameters whose linearized 95% interval is missing or crosses a bound."""
    if len(intervals) != len(names):
        return ["confidence interval count does not match fitted parameters"]
    issues = []
    for name, value, interval in zip(names, estimates, intervals):
        param = params[name]
        lo = float(np.asarray(param.min_value).ravel()[0])
        hi = float(np.asarray(param.max_value).ravel()[0])
        if not np.isfinite(interval):
            issues.append(name + " (CI unavailable)")
        elif value - interval <= lo or value + interval >= hi:
            issues.append(name + " (CI reaches a bound)")
    return issues


def _reject(out, reason):
    """Fail an input check: clear any stale XML in a REUSED --out dir and leave an
    INVALID_FIT.txt, so old artifacts cannot look like this run's."""
    print(f"[fit] ERROR: {reason}")
    try:
        folder = pathlib.Path(out)
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "pineapple_arm_identified.xml").unlink(missing_ok=True)
        (folder / "INVALID_FIT.txt").write_text(
            "INVALID SYSTEM IDENTIFICATION FIT\n\n- " + reason
            + "\n\nInput was rejected before fitting; no XML was emitted.\n"
        )
    except Exception as e:
        print(f"[fit] (could not clear stale artifacts in {out}: {e})")
    return 2


def _atomic_write_bytes(path, data: bytes):
    """Write via a temp file + os.replace so readers never see a partial file."""
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _stamp_invalid_xml(path, reasons):
    """Prepend a loud warning comment inside an override-written model XML."""
    try:
        text = pathlib.Path(path).read_text()
        banner = ("<!-- WARNING: INVALID FIT. Written by sysid_fit.py "
                  "--allow-invalid-result for DIAGNOSIS ONLY -- DO NOT DEPLOY.\n"
                  + "\n".join("     - " + r for r in reasons) + " -->\n")
        # Keep any XML declaration first so the file stays well-formed.
        if text.startswith("<?xml"):
            head, _, rest = text.partition("\n")
            text = head + "\n" + banner + rest
        else:
            text = banner + text
        pathlib.Path(path).write_text(text)
    except Exception as e:
        print(f"[fit] (could not stamp invalid XML: {e})")


def _segment_rollout(spec, t, ctrl, q, dq, seg):
    """Predicted q from multiple-shooting segments (re-anchored to the measured
    state every ``seg`` steps -- what the fit actually optimizes)."""
    import mujoco
    import mujoco.rollout
    model = spec.compile()
    data = mujoco.MjData(model)
    qp = np.full_like(q, np.nan)
    start = 0
    while start + 2 <= len(t):
        end = min(start + seg, len(t))
        x0 = sysid.create_initial_state(model, q[start], dq[start])
        st, _ = mujoco.rollout.rollout(model, data, x0[None, :], ctrl[start:end][None, :-1, :])
        qp[start + 1:end] = st[0][:, 1:1 + model.nq]
        qp[start] = q[start]
        start = end
    return qp


def _fit_quality(spec, t, ctrl, q, dq, seg):
    """Per-joint position RMSE [rad], segment-wise AND full-trajectory.

    The segment rollout re-anchors every ``seg`` steps so it CANNOT reveal long-term
    drift; the full open-loop rollout can. A fit can converge "successfully" onto a
    bad model, so ``main`` gates on both.
    """
    seg_rmse = full_rmse = np.full(C.NUM_MOTORS, np.nan)
    try:
        qp = _segment_rollout(spec, t, ctrl, q, dq, seg)
        m = np.isfinite(qp).all(axis=1)
        if m.any():
            seg_rmse = np.sqrt(np.mean((qp[m] - q[m]) ** 2, axis=0))
    except Exception as e:
        print(f"[fit] (segment rollout unavailable: {e})")
    try:
        qf = _rollout_full(spec, t, ctrl, q, dq)
        n = min(len(qf), len(q))
        good = np.isfinite(qf[:n]).all(axis=1)
        if good.any():
            full_rmse = np.sqrt(np.mean((qf[:n][good] - q[:n][good]) ** 2, axis=0))
    except Exception as e:
        print(f"[fit] (full-trajectory rollout unavailable: {e})")
    return seg_rmse, full_rmse


def _save_results(
    out: str,
    model_sequences: sysid.ModelSequences,
    initial_params: sysid.ParameterDict,
    opt_params: sysid.ParameterDict,
    opt_result,
    covariance: np.ndarray,
    intervals: np.ndarray,
    interval_kind: str = "95% CI",
) -> None:
    """Persist results. ``interval_kind`` records whether the intervals are
    measurement-only CIs or prior-shrunk regularized ones (``--reg > 0``), so a
    later reader of confidence.pkl cannot misinterpret them."""
    folder = pathlib.Path(out)
    folder.mkdir(parents=True, exist_ok=True)
    initial_params.save_to_disk(folder / "params_x_0.yaml")
    opt_params.save_to_disk(folder / "params_x_hat.yaml")
    # Atomic: a crash mid-write must not leave a truncated pickle.
    _atomic_write_bytes(folder / "results.pkl",
                        pickle.dumps(opt_result, protocol=pickle.HIGHEST_PROTOCOL))
    _atomic_write_bytes(folder / "confidence.pkl",
                        pickle.dumps({"cov": covariance, "intervals": intervals,
                                      "kind": interval_kind},
                                     protocol=pickle.HIGHEST_PROTOCOL))
    model_sequences.spec.to_file((folder / f"{model_sequences.name}.xml").as_posix())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="path to logged .npz from collect_data.py")
    ap.add_argument("--drive", choices=["torque", "pd"], default="torque")
    ap.add_argument("--attrs", nargs="+", default=list(C.ALL_ATTRS),
                    choices=list(C.ALL_ATTRS),
                    help="which per-joint parameters to identify")
    ap.add_argument("--optimizer", default="scipy_parallel_fd",
                    choices=["scipy", "scipy_parallel_fd", "mujoco"])
    ap.add_argument("--seg", type=_seg_steps, default=100,
                    help="multiple-shooting segment length in steps")
    ap.add_argument("--dt", type=float, default=None,
                    help="rollout timestep; default = median dt of the data")
    ap.add_argument("--resample", choices=["auto", "on", "off"], default="auto",
                    help="resample async/jittered DDS data onto a uniform dt grid "
                         "(using the log's REAL timestamps) before fitting. auto: "
                         "when duplicate full states or non-uniform timing are detected. "
                         "Needs real timestamps in the log (collect_data.py records them).")
    ap.add_argument("--allow-legacy-time", action="store_true",
                    help="override rejection of duplicate/jittered logs without proven real "
                         "timestamps (diagnostic use only; estimates may be biased)")
    ap.add_argument("--allow-unsafe-data", action="store_true",
                    help="fit despite torque/velocity/saturation/tracking failures "
                         "(diagnostic use only)")
    ap.add_argument("--trim", type=float, default=0.0,
                    help="drop the first N seconds (initial settling)")
    ap.add_argument("--max-iters", type=int, default=200)
    ap.add_argument("--freeze", nargs="*", default=[], choices=list(C.ALL_ATTRS),
                    help="hold these attrs FIXED at nominal (don't optimize) -- for "
                         "parameters the data can't observe, e.g. --freeze armature")
    ap.add_argument("--reg", type=float, default=0.0,
                    help="Tikhonov regularization weight pulling params toward their "
                         "nominal (relative units). >0 stops unobservable params from "
                         "drifting 'confidently wrong'; try 0.01-0.1")
    ap.add_argument("--out", default="results/latest", help="output directory")
    ap.add_argument("--max-rmse", type=float, default=0.05,
                    help="max per-joint position RMSE [rad] of the identified model's "
                         "open-loop rollout for the fit to count as VALID. Guards "
                         "against an optimizer that converges onto a poor model. The "
                         "full-trajectory limit is 4x this (drift accumulates).")
    ap.add_argument("--allow-invalid-result", action="store_true",
                    help="write the drop-in XML even if optimization fails, a parameter "
                         "hits a bound, or the rollout error exceeds --max-rmse "
                         "(diagnostic use only; keeps INVALID_FIT.txt and exits nonzero)")
    ap.add_argument("--no-plots", action="store_true")
    args = ap.parse_args()

    # Load and prepare arrays in motor order.
    if args.dt is not None and args.dt <= 0:
        ap.error("--dt must be positive")
    if args.trim < 0:
        ap.error("--trim must be nonnegative")
    if args.reg < 0:
        ap.error("--reg must be nonnegative")
    if args.max_iters < 1:
        ap.error("--max-iters must be >= 1")
    if not np.isfinite(args.max_rmse) or args.max_rmse <= 0:
        ap.error("--max-rmse must be finite and positive")

    log = C.load_log(args.data)
    try:
        C.validate_log(log, drive=args.drive)
    except (TypeError, ValueError) as e:
        return _reject(args.out, f"invalid log: {e}")

    diagnostic_input_reasons: list[str] = []
    quality_issues = _excitation_issues(log, drive=args.drive)
    if quality_issues:
        print("[fit] excitation-quality failures:")
        for issue in quality_issues:
            print("  - " + issue)
        if not args.allow_unsafe_data:
            return _reject(args.out, "excitation-quality failures: "
                           + "; ".join(quality_issues)
                           + " (re-collect, or --allow-unsafe-data for diagnosis)")
        print("[fit] WARNING: continuing because --allow-unsafe-data was supplied.")
        diagnostic_input_reasons.append("unsafe excitation-data override was used")

    # DDS collection is asynchronous: duplicate frames (~12% on the sim) and
    # jittered timing, while the fixed-timestep rollout assumes one dt step per
    # sample. Resample first. `auto` is a no-op on already-clean data.
    grid_dt = args.dt or C.DT_DEFAULT
    _dropped, _d, _dt_jit = _timing_artifacts(log)
    # ANY duplicate/non-monotonic frame counts: a >1% floor silently left
    # tick-detected duplicates in the fitted data.
    _needs_resample = _dropped > 0 or _dt_jit
    _do_resample = args.resample == "on" or (
        args.resample == "auto" and _needs_resample
    )
    _missing_real_time = _needs_resample and not C.has_real_timestamps(log)
    if _missing_real_time:
        if not args.allow_legacy_time:
            return _reject(args.out,
                "duplicate/jittered samples need resampling, but this log does not "
                "contain proven real state-arrival timestamps. Legacy synthetic time "
                "grids cannot be repaired; re-collect with the current collector.")
        diagnostic_input_reasons.append("legacy/unproven timestamp override was used")
    if args.resample == "off" and _needs_resample:
        print(
            "[fit] WARNING: fitting duplicate/non-uniform data with --resample off "
            "can bias armature and damping."
        )
        diagnostic_input_reasons.append(
            "duplicate/non-uniform data was fitted with resampling disabled"
        )
    if _do_resample:
        _n0 = len(log["t"])
        log = C.resample_log(log, dt=grid_dt)
        print(f"[fit] cleaned async data: {_n0} -> {len(log['t'])} samples @ uniform "
              f"dt={grid_dt}s ({_dropped} duplicate/non-monotonic frames dropped, "
              f"raw dt med={np.median(_d):.5f}s std={np.std(_d):.5f}s)")

    t = np.asarray(log["t"], dtype=float)
    if args.trim > 0:
        keep = t >= (t[0] + args.trim)
        log = {k: (v[keep] if getattr(v, "ndim", 0) >= 1 and len(v) == len(t) else v)
               for k, v in log.items()}
        t = t[keep]
    if len(t) < 2:
        return _reject(args.out, "fewer than 2 samples remain after trimming")
    t = t - t[0]
    dt = args.dt or _infer_dt(t)
    q, dq = np.asarray(log["q"], float), np.asarray(log["dq"], float)

    if args.drive == "torque":
        ctrl = np.asarray(log["tau"], float)
        kp = kd = None
    else:
        ctrl = np.asarray(log["q_cmd"], float)
        # PD replay uses ONE constant gain set and zero desired velocity: reject
        # anything the fitted model cannot represent rather than silently using
        # row 0 and ignoring the rest.
        for key in ("kp", "kd"):
            if key not in log:
                continue
            arr = np.asarray(log[key], float)
            if arr.ndim == 2 and not np.allclose(arr, arr[0]):
                return _reject(args.out, f"{key} varies over time; PD replay fits a "
                               "single constant gain set. Re-collect with fixed gains.")
        for key in ("dq_cmd", "tau_ff"):
            if key in log and np.any(np.asarray(log[key], float) != 0.0):
                print(f"[fit] WARNING: logged {key} is non-zero but the PD-replay model "
                      f"ignores it; the fit will not reproduce that {key} contribution.")
        kp = np.asarray(log["kp"][0], float) if "kp" in log else C.DEFAULT_KP
        kd = np.asarray(log["kd"][0], float) if "kd" in log else C.DEFAULT_KD

    print(f"[fit] data={args.data}  N={len(t)}  dt={dt:.4f}s  drive={args.drive}  "
          f"attrs={args.attrs}  seg={args.seg}  freeze={args.freeze}  reg={args.reg}")
    print(f"[fit] q range [{q.min():.2f}, {q.max():.2f}] rad, "
          f"|dq|max {np.abs(dq).max():.2f} rad/s, |tau|max {np.abs(ctrl).max():.2f}"
          f"{' Nm' if args.drive=='torque' else ' rad(cmd)'}")

    # Build model and parameters.
    fit_spec = C.fresh_spec(drive=args.drive, dt=dt, kp=kp, kd=kd)
    ms = C.make_model_sequences(fit_spec, t, ctrl, q, dq, seg_steps=args.seg)
    params = C.build_param_dict(attrs=args.attrs, freeze_attrs=args.freeze)
    if params.as_vector().size == 0:
        return _reject(args.out, "all selected parameters are frozen")
    init_params = params.copy()

    # Optimize.
    base_residual_fn = sysid.build_residual_fn(models_sequences=[ms])
    if args.reg > 0:
        # Tikhonov rows pull unconstrained directions toward nominal. Computed
        # from the decision vector `x` so the shape matches the toolbox's batched
        # finite-difference Jacobian: x is (n,) normally, (n, n_fd) during FD.
        # Confidence must see these rows too, or residual and Jacobian dimensions
        # become statistically inconsistent.
        x_nom = init_params.as_vector()
        scale = np.maximum(np.abs(x_nom), 1e-6)

        def residual_fn(x, p, **kw):
            res, preds, recs = base_residual_fn(x, p, **kw)
            xa = np.asarray(x, float)
            if xa.ndim == 1:
                reg = np.sqrt(args.reg) * (xa - x_nom) / scale
            else:
                reg = np.sqrt(args.reg) * (xa - x_nom[:, None]) / scale[:, None]
            res = list(res) + [reg]
            return res, preds, recs
    else:
        residual_fn = base_residual_fn

    opt_params, opt_result = sysid.optimize(
        params, residual_fn,
        optimizer=args.optimizer,
        verbose=False,
        max_iters=args.max_iters,
        x_scale="jac",
    )

    # Report.
    print("\n" + opt_params.compare_parameters(
        init_params.as_vector(), opt_params.as_vector()))

    names = opt_params.get_non_frozen_parameter_names()
    est = opt_params.as_vector()
    bound_hits = _bound_hits(opt_params)
    try:
        res_star, _, _ = residual_fn(opt_result.x, opt_params, return_pred_all=True)
        covariance, intervals = sysid.calculate_intervals(res_star, opt_result.jac)
    except Exception as e:  # confidence intervals are best-effort
        print(f"[fit] (confidence intervals unavailable: {e})")
        covariance = np.full((len(names), len(names)), np.nan)
        intervals = np.full(len(names), np.nan)
    confidence_issues = _confidence_issues(opt_params, names, est, intervals)
    # With --reg these are prior-shrunk (MAP-style) intervals, NOT measurement-only
    # 95% CIs. Label them honestly rather than overstating confidence.
    ci_label = ("regularized ~95% interval (prior-shrunk, reg=%g)" % args.reg
                if args.reg > 0 else "95% CI")
    print(f"\nIdentified parameters (value +- {ci_label}):")
    for name, v, ci in zip(names, est, intervals):
        ci_s = f"+-{ci:.3g}" if np.isfinite(ci) else "(CI n/a)"
        flags = []
        if name in bound_hits:
            flags.append("near bound")
        if any(item.startswith(name + " ") for item in confidence_issues):
            flags.append("CI invalid")
        flag = "  <-- " + ", ".join(flags) if flags else ""
        print(f"  {name:30s} {v:11.5g} {ci_s}{flag}")

    # An optimizer can report success while sitting on a poor model, so gate the
    # deployment XML on the actual rollout error too.
    ident_spec = C.fresh_spec(drive=args.drive, dt=dt, kp=kp, kd=kd)
    sysid.apply_param_modifiers_spec(opt_params, ident_spec)
    seg_rmse, full_rmse = _fit_quality(ident_spec, t, ctrl, q, dq, args.seg)
    full_tol = 4.0 * args.max_rmse  # open-loop drift accumulates over the trajectory
    print(f"\nFit quality -- per-joint position RMSE [rad] "
          f"(limits: seg {args.max_rmse:.3g}, full {full_tol:.3g}):")
    print(f"  {'joint':18s} {'segment':>10s} {'full-traj':>10s}")
    for j, jn in enumerate(C.JOINTS):
        print(f"  {jn:18s} {seg_rmse[j]:10.4g} {full_rmse[j]:10.4g}")
    quality_failures = []
    if not np.all(np.isfinite(seg_rmse)) or np.any(seg_rmse > args.max_rmse):
        quality_failures.append(
            f"segment rollout RMSE {np.nanmax(seg_rmse):.4g} > {args.max_rmse:.4g} rad")
    if not np.all(np.isfinite(full_rmse)) or np.any(full_rmse > full_tol):
        quality_failures.append(
            f"full-trajectory rollout RMSE {np.nanmax(full_rmse):.4g} > {full_tol:.4g} rad")
    for msg in quality_failures:
        print(f"  <-- FAIL: {msg}")

    # Save results and the validated model.
    _save_results(
        args.out, ms, init_params, opt_params, opt_result, covariance, intervals,
        interval_kind=ci_label,
    )
    ident_xml = os.path.join(args.out, "pineapple_arm_identified.xml")
    fit_valid = (
        bool(getattr(opt_result, "success", False))
        and np.all(np.isfinite(est))
        and not bound_hits
        and not confidence_issues
        and not diagnostic_input_reasons
        and not quality_failures
    )
    marker = pathlib.Path(args.out) / "INVALID_FIT.txt"
    reasons = []
    if not bool(getattr(opt_result, "success", False)):
        reasons.append(f"optimizer failed: {getattr(opt_result, 'message', 'unknown')}")
    if not np.all(np.isfinite(est)):
        reasons.append("one or more estimates are NaN/Inf")
    if bound_hits:
        reasons.append("parameters at/near bounds: " + ", ".join(bound_hits))
    if confidence_issues:
        reasons.append(
            "unreliable confidence intervals: " + ", ".join(confidence_issues)
        )
    reasons.extend(quality_failures)
    reasons.extend(diagnostic_input_reasons)

    if fit_valid:
        C.save_identified_xml(opt_params, ident_xml)
        marker.unlink(missing_ok=True)
    elif args.allow_invalid_result:
        # Diagnostic override: emit the XML but KEEP the marker, stamp the file,
        # and still exit nonzero, so nothing can mistake it for a valid fit.
        C.save_identified_xml(opt_params, ident_xml)
        _stamp_invalid_xml(ident_xml, reasons)
        marker.write_text(
            "INVALID SYSTEM IDENTIFICATION FIT (XML written by --allow-invalid-result)\n\n"
            + "\n".join("- " + reason for reason in reasons)
            + "\n\nDO NOT DEPLOY pineapple_arm_identified.xml: it was emitted by an\n"
              "explicit diagnostic override, not by passing validation.\n"
        )
    else:
        pathlib.Path(ident_xml).unlink(missing_ok=True)
        marker.write_text(
            "INVALID SYSTEM IDENTIFICATION FIT\n\n"
            + "\n".join("- " + reason for reason in reasons)
            + "\n\nNo pineapple_arm_identified.xml was emitted. Re-collect/re-fit, or "
              "use --allow-invalid-result only for diagnosis.\n"
        )
    print(f"\n[fit] results saved to {args.out}/")
    if fit_valid:
        print(f"[fit] drop-in identified model: {ident_xml}")
    elif args.allow_invalid_result:
        print(f"[fit] WARNING: wrote INVALID XML by explicit override: {ident_xml}")
        print(f"[fit] {marker} kept and the XML stamped; exit code is nonzero. DO NOT DEPLOY.")
    else:
        print(f"[fit] INVALID result; no drop-in XML written. See {marker}")

    if not args.no_plots:
        try:
            _plots(args.out, t, dt, q, dq, ctrl, fit_spec, opt_params, init_params,
                   args.drive, names, est, intervals, args.seg, kp, kd)
            print(f"[fit] plots: {args.out}/fit_overview.png, {args.out}/fit_params.png")
        except Exception as e:
            print(f"[fit] (plotting skipped: {e})")
    # Nonzero on override too: a diagnostic XML must never look like a passing run.
    return 0 if fit_valid else 2


def _rollout_full(spec, t, ctrl, q, dq):
    """Open-loop rollout of ``spec`` over the whole trajectory (for plotting)."""
    import mujoco
    import mujoco.rollout
    model = spec.compile()
    x0 = sysid.create_initial_state(model, q[0], dq[0])
    data = mujoco.MjData(model)
    state, _ = mujoco.rollout.rollout(model, data, x0[None, :], ctrl[None, :-1, :])
    nq = model.nq
    qp = np.vstack([q[0], state[0][:, 1:1 + nq]])
    return qp


def _plots(out, t, dt, q, dq, ctrl, fit_spec, opt_params, init_params,
           drive, names, est, intervals, seg, kp, kd):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Segment rollouts show fit quality; the full rollout exposes accumulated drift.
    ident_spec = C.fresh_spec(drive=drive, dt=dt,
                              kp=C.DEFAULT_KP if kp is None else kp,
                              kd=C.DEFAULT_KD if kd is None else kd)
    sysid.apply_param_modifiers_spec(opt_params, ident_spec)
    qp = _segment_rollout(ident_spec, t, ctrl, q, dq, seg)
    try:
        qf = _rollout_full(ident_spec, t, ctrl, q, dq)[:len(t)]
    except Exception as e:
        print(f"[fit] (full-trajectory plot unavailable: {e})")
        qf = np.full_like(q, np.nan)

    fig, axes = plt.subplots(C.NUM_MOTORS, 1, figsize=(10, 12), sharex=True)
    for j in range(C.NUM_MOTORS):
        axes[j].plot(t, q[:, j], "k", lw=1.0, label="measured")
        axes[j].plot(t, qp[:, j], "C1--", lw=1.0, label="identified (per-seg)")
        axes[j].plot(t[:len(qf)], qf[:, j], "C2:", lw=1.0,
                     label="identified (full open-loop)")
        axes[j].set_ylabel(C.JOINTS[j] + "\nq [rad]")
        if j == 0:
            axes[j].legend(loc="upper right", fontsize=8)
    axes[-1].set_xlabel("time [s]")
    fig.suptitle("Measured vs. identified-model joint position")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fit_overview.png"), dpi=110)
    plt.close(fig)

    # Identified parameters grouped by attribute, with confidence intervals.
    attrs = sorted({n.split("/")[1] for n in names})
    fig, axes = plt.subplots(1, len(attrs), figsize=(4 * len(attrs), 4))
    if len(attrs) == 1:
        axes = [axes]
    init_vec = init_params.as_vector()
    for ax, attr in zip(axes, attrs):
        idx = [i for i, n in enumerate(names) if n.endswith("/" + attr)]
        xs = np.arange(len(idx))
        ax.bar(xs - 0.2, init_vec[idx], width=0.4, label="initial", color="0.7")
        ci = np.where(np.isfinite(intervals[idx]), intervals[idx], 0.0)
        ax.bar(xs + 0.2, est[idx], width=0.4, yerr=ci, capsize=3,
               label="identified", color="C0")
        ax.set_xticks(xs)
        ax.set_xticklabels([C.JOINTS[i] for i in range(len(idx))],
                           rotation=45, ha="right", fontsize=8)
        ax.set_title(attr)
        ax.legend(fontsize=8)
    fig.suptitle("Identified joint parameters")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fit_params.png"), dpi=110)
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
