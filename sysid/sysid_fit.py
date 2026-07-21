"""Identify per-joint armature / frictionloss / damping for the pineapple arm.

Given a trajectory logged from the real arm by ``collect_data.py`` (measured
q / dq / tau plus the position command), this fits the MuJoCo model's per-joint
dynamics parameters with DeepMind's ``mujoco.sysid`` toolbox: box-constrained
nonlinear least squares over a threaded ``mujoco.rollout``, using multiple
shooting so the open-loop rollout stays close to the data.

Drive modes (how the measured motion is reproduced in simulation):
  * ``torque`` (default, recommended): replay the measured joint torque through
    the torque motors and match the resulting joint motion. Most sensitive to
    the parameters; validated in ``sim2sim_selftest.py``.
  * ``pd``: replay the position command through PD position actuators. Robust to
    a biased torque sensor, but the closed loop masks the parameters (low
    sensitivity) -- use only as a cross-check.

Outputs (under ``--out``): a console table + confidence intervals, the toolbox
result files (params_x_0.yaml, params_x_hat.yaml, results.pkl, confidence.pkl),
an identified full model ``pineapple_arm_identified.xml``, and diagnostic plots.

Example:
    python sysid_fit.py --data data/arm_chirp_XXXX.npz --out results/run1
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

import absl.logging
from mujoco import sysid

# Silence the per-segment "sensor data dimension ... does not match" info that
# allow_missing_sensors=True emits on every model compile.
absl.logging.set_verbosity(absl.logging.ERROR)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sysid_common as C


def _infer_dt(t: np.ndarray) -> float:
    return float(np.median(np.diff(t)))


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
    ap.add_argument("--seg", type=int, default=100,
                    help="multiple-shooting segment length in steps")
    ap.add_argument("--dt", type=float, default=None,
                    help="rollout timestep; default = median dt of the data")
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
    ap.add_argument("--no-plots", action="store_true")
    args = ap.parse_args()

    # --- load + prepare data (all arrays in JOINTS / motor order) ------------
    log = C.load_log(args.data)
    t = np.asarray(log["t"], dtype=float)
    if args.trim > 0:
        keep = t >= (t[0] + args.trim)
        log = {k: (v[keep] if getattr(v, "ndim", 0) >= 1 and len(v) == len(t) else v)
               for k, v in log.items()}
        t = t[keep]
    t = t - t[0]
    dt = args.dt or _infer_dt(t)
    q, dq = np.asarray(log["q"], float), np.asarray(log["dq"], float)

    if args.drive == "torque":
        ctrl = np.asarray(log["tau"], float)
        kp = kd = None
    else:
        ctrl = np.asarray(log["q_cmd"], float)
        kp = np.asarray(log["kp"][0], float) if "kp" in log else C.DEFAULT_KP
        kd = np.asarray(log["kd"][0], float) if "kd" in log else C.DEFAULT_KD

    print(f"[fit] data={args.data}  N={len(t)}  dt={dt:.4f}s  drive={args.drive}  "
          f"attrs={args.attrs}  seg={args.seg}  freeze={args.freeze}  reg={args.reg}")
    print(f"[fit] q range [{q.min():.2f}, {q.max():.2f}] rad, "
          f"|dq|max {np.abs(dq).max():.2f} rad/s, |tau|max {np.abs(ctrl).max():.2f}"
          f"{' Nm' if args.drive=='torque' else ' rad(cmd)'}")

    # --- build model + parameters -------------------------------------------
    fit_spec = C.fresh_spec(drive=args.drive, dt=dt, kp=kp, kd=kd)
    ms = C.make_model_sequences(fit_spec, t, ctrl, q, dq, seg_steps=args.seg)
    params = C.build_param_dict(attrs=args.attrs, freeze_attrs=args.freeze)
    init_params = params.copy()

    # --- optimize ------------------------------------------------------------
    base_residual_fn = sysid.build_residual_fn(models_sequences=[ms])
    if args.reg > 0:
        # Tikhonov: append sqrt(reg)*(x - x_nominal)/|x_nominal| residual rows so
        # the optimizer prefers staying near nominal on directions the data does
        # not constrain. Computed from the decision vector `x` (not params) so it
        # matches the toolbox's batched shape during the finite-difference
        # Jacobian: x is (n,) on a normal eval and (n, n_fd) during FD. Skip on
        # the CI pass (return_pred_all) so confidence uses data residuals only.
        x_nom = init_params.as_vector()
        scale = np.maximum(np.abs(x_nom), 1e-6)

        def residual_fn(x, p, **kw):
            res, preds, recs = base_residual_fn(x, p, **kw)
            if not kw.get("return_pred_all", False):
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

    # --- report --------------------------------------------------------------
    print("\n" + opt_params.compare_parameters(
        init_params.as_vector(), opt_params.as_vector()))

    names = opt_params.get_non_frozen_parameter_names()
    est = opt_params.as_vector()
    try:
        res_star, _, _ = residual_fn(opt_result.x, opt_params, return_pred_all=True)
        _, intervals = sysid.calculate_intervals(res_star, opt_result.jac)
    except Exception as e:  # confidence intervals are best-effort
        print(f"[fit] (confidence intervals unavailable: {e})")
        intervals = np.full(len(names), np.nan)
    print("\nIdentified parameters (value +- 95% CI):")
    for name, v, ci in zip(names, est, intervals):
        ci_s = f"+-{ci:.3g}" if np.isfinite(ci) else "(CI n/a)"
        flag = "  <-- near bound" if (v <= params[name.split('[')[0]].min_value[0] + 1e-9
                                      or v >= params[name.split('[')[0]].max_value[0] - 1e-9) else ""
        print(f"  {name:30s} {v:11.5g} {ci_s}{flag}")

    # --- save results + identified model ------------------------------------
    sysid.save_results(args.out, [ms], init_params, opt_params, opt_result, residual_fn)
    ident_xml = os.path.join(args.out, "pineapple_arm_identified.xml")
    C.save_identified_xml(opt_params, ident_xml)
    print(f"\n[fit] results saved to {args.out}/")
    print(f"[fit] drop-in identified model: {ident_xml}")

    if not args.no_plots:
        try:
            _plots(args.out, t, dt, q, dq, ctrl, fit_spec, opt_params, init_params,
                   args.drive, names, est, intervals)
            print(f"[fit] plots: {args.out}/fit_overview.png, {args.out}/fit_params.png")
        except Exception as e:
            print(f"[fit] (plotting skipped: {e})")
    return 0


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
           drive, names, est, intervals):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # 1) data + fit-quality overview: measured vs identified one-shot rollout
    #    (segment-wise, matching the multiple-shooting the fit actually used).
    seg = 100
    ident_spec = C.fresh_spec(drive=drive, dt=dt,
                              kp=C.DEFAULT_KP, kd=C.DEFAULT_KD)
    sysid.apply_param_modifiers_spec(opt_params, ident_spec)
    qp = np.full_like(q, np.nan)
    start = 0
    import mujoco, mujoco.rollout
    model = ident_spec.compile()
    data = mujoco.MjData(model)
    while start + 2 <= len(t):
        end = min(start + seg, len(t))
        x0 = sysid.create_initial_state(model, q[start], dq[start])
        st, _ = mujoco.rollout.rollout(model, data, x0[None, :], ctrl[start:end][None, :-1, :])
        qp[start + 1:end] = st[0][:, 1:1 + model.nq]
        qp[start] = q[start]
        start = end

    fig, axes = plt.subplots(C.NUM_MOTORS, 1, figsize=(10, 12), sharex=True)
    for j in range(C.NUM_MOTORS):
        axes[j].plot(t, q[:, j], "k", lw=1.0, label="measured")
        axes[j].plot(t, qp[:, j], "C1--", lw=1.0, label="identified (per-seg)")
        axes[j].set_ylabel(C.JOINTS[j] + "\nq [rad]")
        if j == 0:
            axes[j].legend(loc="upper right", fontsize=8)
    axes[-1].set_xlabel("time [s]")
    fig.suptitle("Measured vs. identified-model joint position")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fit_overview.png"), dpi=110)
    plt.close(fig)

    # 2) identified parameters grouped by attribute, with CI error bars
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
