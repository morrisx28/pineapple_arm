"""Generate and validate jerk-limited joint-space trajectories.

    quintic_point_to_point   two endpoints, closed-form duration
    quintic_spline_via       C^2 through intermediate configurations
    linear_blend             pineapple_arm.py's profile, for A/B comparison only
    validate                 fail-closed feasibility gate (limits, caps, RNEA torque)

The module is intentionally independent of controller tuning; it only shares model and
limit definitions with ``ee_traj``.
"""

from __future__ import annotations

import numpy as np
import pinocchio

import arm_ff
import arm_ik
import ee_traj

NUM_ARM_DOF = ee_traj.NUM_ARM_DOF  # 6

JOINT_LOW = ee_traj.JOINT_LOW
JOINT_HIGH = ee_traj.JOINT_HIGH

# Kinematic caps
# DQ_MAX matches ee_traj.DQ_REF_MAX so a trajectory accepted here is also accepted by the
# rest of the pipeline. DDQ_MAX and JERK_MAX have NO measured basis -- they are
# conservative knobs, and callers are expected to print them (see arm_smooth_move.py's
# report) rather than let them pass silently. The real feasibility gate is the RNEA torque
# check in validate().
#
# Do NOT source these from the URDF: it declares velocity="0" effort="0" on the first five
# joints, so pinocchio reports velocityLimit = [0, 0, 0, 0, 0, 30, ...].
DQ_MAX = np.full(NUM_ARM_DOF, 2.0)      # rad/s
DDQ_MAX = np.full(NUM_ARM_DOF, 4.0)     # rad/s^2
JERK_MAX = np.full(NUM_ARM_DOF, 20.0)   # rad/s^3

T_MIN = 0.5   # s: never command a move shorter than this, however small the span

# Tolerance when comparing an ACHIEVED peak against a cap. The peaks are measured on the
# generated samples, not evaluated analytically, so a profile sitting exactly on a cap reads
# a hair over it. Shared by validate() and quintic_from_state()'s duration search so the
# gate that accepts a trajectory and the search that produces one cannot disagree.
CAP_SLACK = 0.01   # 1%

# Floating-point tolerance, not a safety margin. It permits an endpoint exactly on a joint
# limit while remaining far below encoder resolution and the watchdog penetration limit.
LIMIT_EPS = 1e-9   # rad

# How far outside a nominal joint limit a START configuration may sit before validate()
# warns. The measured resting pose routinely reads a fraction of a mrad outside (arm_base's
# low bound is exactly 0.0 and the arm rests there), which must stay silent; a miscalibrated
# zero must not.
START_OUTSIDE_WARN = 0.05   # rad

# Peak derivatives of the quintic scaling s(tau) = 10 tau^3 - 15 tau^4 + 6 tau^5, tau=t/T.
# These are exact, not fitted:
#   max|ds/dt|   at tau=1/2            -> 15/8      / T
#   max|dds/dt^2| at tau=(3-sqrt3)/6   -> 10/sqrt3  / T^2
#   max|d3s/dt^3| at tau=0 and tau=1   -> 60        / T^3
# They are what makes the duration solve closed-form instead of an optimization.
PEAK_DS = 15.0 / 8.0                  # 1.875
PEAK_DDS = 10.0 / np.sqrt(3.0)        # 5.7735026918962575
PEAK_D3S = 60.0

# Use the hardware command path's limit table; the older EE reference table differs.
TAU_LIMIT = arm_ff.TAU_LIMIT
TAU_FRAC = 0.7   # leave headroom for the PD feedback term on top of the reference


def _as_caps(dq_max, ddq_max, jerk_max):
    """Broadcast the three caps to (6,) arrays, rejecting non-positive values."""
    out = []
    for name, val, default in (("dq_max", dq_max, DQ_MAX),
                               ("ddq_max", ddq_max, DDQ_MAX),
                               ("jerk_max", jerk_max, JERK_MAX)):
        arr = np.broadcast_to(np.asarray(default if val is None else val, float),
                             (NUM_ARM_DOF,)).astype(float)
        if not np.all(np.isfinite(arr)) or np.any(arr <= 0):
            raise ValueError(f"{name} must be finite and positive, got {arr}")
        out.append(arr)
    return tuple(out)


def min_duration(span, dq_max=None, ddq_max=None, jerk_max=None, t_min=T_MIN):
    """Shortest quintic duration [s] whose vel/accel/jerk peaks all fit the caps.

    ``span`` is the per-joint displacement (6,). Closed form: the quintic's peaks scale as
    T^-1, T^-2, T^-3 exactly, so each cap gives a lower bound on T and the answer is their
    maximum over joints. No optimizer, no iteration.
    """
    span = np.abs(np.asarray(span, float))
    if span.shape != (NUM_ARM_DOF,):
        raise ValueError(f"span must have shape ({NUM_ARM_DOF},), got {span.shape}")
    if not np.all(np.isfinite(span)):
        raise ValueError(f"span contains NaN/Inf: {span}")
    dq_max, ddq_max, jerk_max = _as_caps(dq_max, ddq_max, jerk_max)

    return float(max(
        t_min,
        np.max(PEAK_DS * span / dq_max),
        np.max(np.sqrt(PEAK_DDS * span / ddq_max)),
        np.max(np.cbrt(PEAK_D3S * span / jerk_max)),
    ))


def _time_grid(duration, dt):
    """Sample grid covering ``duration``, and the grid's own end time.

    Rounds the sample count UP so the returned duration is >= the requested one: the caps
    are satisfied for any T at least as long, so lengthening is always safe while
    shortening would silently violate them.
    """
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError(f"dt must be finite and positive, got {dt}")
    n = max(2, int(np.ceil(duration / dt - 1e-9)) + 1)
    return np.arange(n) * dt, float((n - 1) * dt)


def quintic_point_to_point(q0, q1, dt, duration=None, dq_max=None, ddq_max=None,
                           jerk_max=None, t_min=T_MIN):
    """Straight joint-space move q0 -> q1. -> (t, q, dq, ddq), each (N,6) after t.

    Zero velocity AND zero acceleration at both ends, so neither the entry nor the exit
    puts a step into the torque. ``duration=None`` picks the shortest duration meeting the
    caps; an explicit ``duration`` is used as given and is NOT checked against the caps
    (``validate`` is what rejects an over-fast move).
    """
    q0 = np.asarray(q0, float)[:NUM_ARM_DOF]
    q1 = np.asarray(q1, float)[:NUM_ARM_DOF]
    if q0.shape != (NUM_ARM_DOF,) or q1.shape != (NUM_ARM_DOF,):
        raise ValueError(f"q0/q1 must have {NUM_ARM_DOF} entries, "
                         f"got {q0.shape} and {q1.shape}")
    if not (np.all(np.isfinite(q0)) and np.all(np.isfinite(q1))):
        raise ValueError(f"q0/q1 contain NaN/Inf: {q0}, {q1}")

    d = q1 - q0
    T = min_duration(d, dq_max, ddq_max, jerk_max, t_min) if duration is None \
        else float(duration)
    if not np.isfinite(T) or T <= 0:
        raise ValueError(f"duration must be finite and positive, got {T}")

    t, T = _time_grid(T, dt)
    s, ds, dds = ee_traj.quintic_scaling(t, T)
    return t, q0 + s[:, None] * d, ds[:, None] * d, dds[:, None] * d


# Moving-start quintic
# A moving start needs per-joint coefficients; the shared rest-to-rest scale has ds(0)=0.
T_GROWTH = 1.25        # duration search step; peaks are monotone in T so this converges
T_SEARCH_MAX_ITER = 40  # geometric growth then bisection; far above any real need
T_BISECT_ITER = 20      # ~1e-6 relative resolution on T


def _quintic_state_coeffs(q0, dq0, ddq0, q1, T):
    """Return per-joint coefficients for ``(q0,dq0,ddq0) -> (q1,0,0)`` over ``T``.

    In tau = t/T the six boundary conditions q(0)=q0, q'(0)=dq0, q''(0)=ddq0, q(1)=q1,
    q'(1)=0, q''(1)=0 give a closed form -- no matrix solve:

        a0 = q0,  a1 = dq0*T,  a2 = ddq0*T^2/2,  h = q1 - (a0+a1+a2)
        a3 =  10h + 4a1 +  7a2
        a4 = -15h - 7a1 - 12a2
        a5 =   6h + 3a1 +  5a2

    With zero initial derivatives this reduces to the standard ``10,-15,6`` scaling.
    """
    a0 = q0
    a1 = dq0 * T
    a2 = ddq0 * T * T / 2.0
    h = q1 - (a0 + a1 + a2)
    return np.stack([a0, a1, a2,
                     10.0 * h + 4.0 * a1 + 7.0 * a2,
                     -15.0 * h - 7.0 * a1 - 12.0 * a2,
                     6.0 * h + 3.0 * a1 + 5.0 * a2], axis=0)


def _sample_quintic_state(coef, t, T):
    """Evaluate the (6,6) coefficient set on time grid ``t``. -> (q, dq, ddq) each (N,6)."""
    tau = (t / T)[:, None]
    powers = np.stack([tau ** i for i in range(6)], axis=0)          # (6,N,1)
    q = np.einsum("ij,ink->nj", coef, powers)
    dq = np.einsum("ij,ink->nj", coef[1:] * np.arange(1, 6)[:, None],
                   powers[:5]) / T
    ddq = np.einsum("ij,ink->nj",
                    coef[2:] * (np.arange(2, 6) * np.arange(1, 5))[:, None],
                    powers[:4]) / (T * T)
    return q, dq, ddq


def quintic_from_state(q0, dq0, ddq0, q1, dt, duration=None, dq_max=None, ddq_max=None,
                       jerk_max=None, t_min=T_MIN):
    """Return a jerk-limited quintic from an arbitrary state to rest at ``q1``.

    Returns ``(t,q,dq,ddq)``. Without an explicit duration, geometric growth and bisection
    find the shortest cap-compliant time because derivative peaks decrease with duration.
    ``|dq0| <= dq_max`` is required: no duration can reduce the initial sample's speed.
    Zero initial derivatives delegate to the faster rest-to-rest path.
    """
    q0, dq0, ddq0, q1 = (np.asarray(v, float)[:NUM_ARM_DOF] for v in (q0, dq0, ddq0, q1))
    for name, v in (("q0", q0), ("dq0", dq0), ("ddq0", ddq0), ("q1", q1)):
        if v.shape != (NUM_ARM_DOF,):
            raise ValueError(f"{name} must have {NUM_ARM_DOF} entries, got {v.shape}")
        if not np.all(np.isfinite(v)):
            raise ValueError(f"{name} contains NaN/Inf: {v}")
    caps = _as_caps(dq_max, ddq_max, jerk_max)
    dq_cap, ddq_cap, jerk_cap = caps

    if duration is not None:
        T = float(duration)
        if not np.isfinite(T) or T <= 0:
            raise ValueError(f"duration must be finite and positive, got {T}")
        t, T = _time_grid(T, dt)
        return (t,) + _sample_quintic_state(
            _quintic_state_coeffs(q0, dq0, ddq0, q1, T), t, T)

    # Unsatisfiable-by-construction check, before any searching.
    over = np.abs(dq0) > dq_cap * (1.0 + CAP_SLACK)
    if np.any(over):
        j = int(np.argmax(np.abs(dq0) - dq_cap))
        raise ValueError(
            f"start velocity |dq0[{j}]| = {abs(dq0[j]):.3f} rad/s already exceeds "
            f"dq_max[{j}] = {dq_cap[j]:.3f}; no duration can satisfy the velocity cap. "
            f"Seed from a validated trajectory's reference state, not a measured velocity.")

    def fits(T):
        t, T = _time_grid(T, dt)
        _, dq, ddq = _sample_quintic_state(
            _quintic_state_coeffs(q0, dq0, ddq0, q1, T), t, T)
        got = peaks(dq, ddq, dt)
        return not (np.any(got["dq"] > dq_cap * (1.0 + CAP_SLACK))
                    or np.any(got["ddq"] > ddq_cap * (1.0 + CAP_SLACK))
                    or np.any(got["jerk"] > jerk_cap * (1.0 + CAP_SLACK)))

    # Seed: the rest-to-rest bound on the span, plus the time to bleed off dq0 at ddq_max.
    lo = max(float(t_min),
             min_duration(q1 - q0, *caps, t_min=t_min),
             float(np.max(np.abs(dq0) / ddq_cap)))
    hi = lo
    for _ in range(T_SEARCH_MAX_ITER):
        if fits(hi):
            break
        lo, hi = hi, hi * T_GROWTH
    else:
        raise ValueError(
            f"no duration within {T_SEARCH_MAX_ITER} growth steps satisfies the caps for "
            f"dq0={np.round(dq0, 3)} ddq0={np.round(ddq0, 3)} span={np.round(q1 - q0, 3)}")

    if hi > lo:                       # bisect so the move is not left over-long
        for _ in range(T_BISECT_ITER):
            mid = 0.5 * (lo + hi)
            if fits(mid):
                hi = mid
            else:
                lo = mid

    t, T = _time_grid(hi, dt)
    return (t,) + _sample_quintic_state(
        _quintic_state_coeffs(q0, dq0, ddq0, q1, T), t, T)


def quintic_spline_via(q_vias, dt, duration=None, dq_max=None, ddq_max=None,
                       jerk_max=None, t_min=T_MIN, samples=4001):
    """Return a C2 spline through ``q_vias`` with rest at both endpoints.

    Interior vias do not stop. Duration is derived from densely sampled normalized
    derivative peaks and their exact inverse-power scaling with time.
    """
    from scipy.interpolate import make_interp_spline

    Q = np.atleast_2d(np.asarray(q_vias, float))
    if Q.ndim != 2 or Q.shape[1] != NUM_ARM_DOF:
        raise ValueError(f"q_vias must be (M, {NUM_ARM_DOF}), got {Q.shape}")
    if not np.all(np.isfinite(Q)):
        raise ValueError("q_vias contains NaN/Inf")
    if len(Q) < 2:
        raise ValueError(f"q_vias needs at least 2 configurations, got {len(Q)}")
    if len(Q) == 2:
        return quintic_point_to_point(Q[0], Q[1], dt, duration,
                                      dq_max, ddq_max, jerk_max, t_min)

    # Knots at cumulative joint-space chord length: a long leg gets proportionally more of
    # the parameter range, so the spline does not race through it.
    seg = np.linalg.norm(np.diff(Q, axis=0), axis=1)
    total = float(seg.sum())
    if total < 1e-9:
        return quintic_point_to_point(Q[0], Q[-1], dt, duration,
                                      dq_max, ddq_max, jerk_max, t_min)
    # Coincident consecutive vias would give a repeated knot; nudge them apart.
    seg = np.maximum(seg, 1e-6 * total)
    u_knots = np.concatenate([[0.0], np.cumsum(seg)]) / seg.sum()

    zero = np.zeros(NUM_ARM_DOF)
    bc = ([(1, zero), (2, zero)], [(1, zero), (2, zero)])
    spl = make_interp_spline(u_knots, Q, k=5, bc_type=bc)
    d1, d2, d3 = spl.derivative(1), spl.derivative(2), spl.derivative(3)

    if duration is None:
        u = np.linspace(0.0, 1.0, int(samples))
        p1 = np.max(np.abs(d1(u)), axis=0)
        p2 = np.max(np.abs(d2(u)), axis=0)
        p3 = np.max(np.abs(d3(u)), axis=0)
        dq_c, ddq_c, jerk_c = _as_caps(dq_max, ddq_max, jerk_max)
        T = float(max(
            t_min,
            np.max(p1 / dq_c),
            np.max(np.sqrt(p2 / ddq_c)),
            np.max(np.cbrt(p3 / jerk_c)),
        ))
    else:
        T = float(duration)
        if not np.isfinite(T) or T <= 0:
            raise ValueError(f"duration must be finite and positive, got {T}")

    t, T = _time_grid(T, dt)
    u = np.clip(t / T, 0.0, 1.0)
    return t, spl(u), d1(u) / T, d2(u) / T**2


def linear_blend(q0, q1, dt, duration=3.0):
    """``pineapple_arm.py``'s profile: linear in time. -> (t, q, dq, ddq).

    Here ONLY so ``arm_smooth_move.py --compare-linear`` can plot what it replaces.

    ``dq`` is written analytically as the RECTANGLE it is -- zero at t=0 where the arm is
    still at rest, ``dq/T`` for the whole interior, zero again at t=T -- rather than
    differentiated from ``q``. Differentiating would hide the entire point: the gradient of
    a linear ramp is constant even at the edges, so the two velocity steps that make this
    profile jerk would come back as exactly zero acceleration.
    """
    q0 = np.asarray(q0, float)[:NUM_ARM_DOF]
    q1 = np.asarray(q1, float)[:NUM_ARM_DOF]
    t, T = _time_grid(float(duration), dt)
    phase = np.clip(t / T, 0.0, 1.0)
    q = q0 + phase[:, None] * (q1 - q0)

    dq = np.tile((q1 - q0) / T, (len(t), 1))
    dq[0] = 0.0     # at rest when the command lands
    dq[-1] = 0.0    # at rest again once it ends
    ddq = np.gradient(dq, dt, axis=0, edge_order=1)
    return t, q, dq, ddq


def peaks(dq, ddq, dt):
    """Achieved per-joint peaks -> dict of (6,) arrays for ``dq``, ``ddq``, ``jerk``.

    Measured on the generated samples, so this reports what will actually be published
    rather than what the closed form predicted. ``jerk`` is differentiated from ``ddq``.
    """
    dq = np.asarray(dq, float)
    ddq = np.asarray(ddq, float)
    jerk = np.gradient(ddq, dt, axis=0, edge_order=1) if len(ddq) > 1 \
        else np.zeros_like(ddq)
    return {
        "dq": np.max(np.abs(dq), axis=0),
        "ddq": np.max(np.abs(ddq), axis=0),
        "jerk": np.max(np.abs(jerk), axis=0),
    }


def rnea_torque(q, dq, ddq, model=None):
    """Inverse-dynamics joint torque along a trajectory. (N,6), gravity INCLUDED."""
    model = ee_traj.build_arm_model() if model is None else model
    data = model.createData()
    q = np.asarray(q, float)
    dq = np.asarray(dq, float)
    ddq = np.asarray(ddq, float)
    tau = np.zeros_like(q)
    for k in range(len(q)):
        tau[k] = pinocchio.rnea(model, data, q[k], dq[k], ddq[k])
    return tau


def start_outside_limits(q_start):
    """How far past a nominal limit ``q_start`` sits, per joint (>=0). (6,)"""
    q_start = np.asarray(q_start, float)[:NUM_ARM_DOF]
    return np.maximum.reduce([JOINT_LOW - q_start, q_start - JOINT_HIGH,
                              np.zeros(NUM_ARM_DOF)])


def validate(t, q, dq, ddq, model=None, dq_max=None, ddq_max=None, jerk_max=None,
             tau_limit=None, tau_frac=TAU_FRAC, warn=True) -> list[str]:
    """Return fail-closed finiteness, limit, cap, and RNEA-torque issues.

    The limit envelope includes an initially out-of-range measured pose so the arm can move
    back toward legality, but rejects any greater penetration. Large initial offsets warn
    because they indicate calibration error rather than encoder noise.
    """
    issues: list[str] = []
    for label, arr in (("q", q), ("dq", dq), ("ddq", ddq)):
        if not np.all(np.isfinite(arr)):
            issues.append(f"{label} contains NaN/Inf")
    if issues:
        return issues

    q = np.asarray(q, float)
    dt = float(np.median(np.diff(t))) if len(t) > 1 else 1.0
    dq_c, ddq_c, jerk_c = _as_caps(dq_max, ddq_max, jerk_max)
    got = peaks(dq, ddq, dt)
    lim = np.broadcast_to(np.asarray(TAU_LIMIT if tau_limit is None else tau_limit,
                                     float), (NUM_ARM_DOF,))

    outside = start_outside_limits(q[0])
    lo = np.minimum(JOINT_LOW, q[0]) - LIMIT_EPS
    hi = np.maximum(JOINT_HIGH, q[0]) + LIMIT_EPS
    if warn:
        for j in (int(i) for i in np.flatnonzero(outside > START_OUTSIDE_WARN)):
            # int(): pinocchio's names container rejects numpy integer indices.
            print(f"[traj] WARNING: {arm_ik.IK_MODEL.names[j + 1]} starts "
                  f"{outside[j]:.4f} rad outside its limit "
                  f"[{JOINT_LOW[j]:+.3f}, {JOINT_HIGH[j]:+.3f}] -- the limit check is "
                  "widened to allow it, but this is large enough to suggest a calibration "
                  "problem, not encoder offset")

    tau = rnea_torque(q, dq, ddq, model)
    for j in range(NUM_ARM_DOF):
        name = arm_ik.IK_MODEL.names[j + 1]
        col = q[:, j]
        if np.any(col < lo[j]) or np.any(col > hi[j]):
            worst = col[np.argmax(np.abs(col - np.clip(col, lo[j], hi[j])))]
            issues.append(f"{name}: q reaches {worst:+.4f} rad, outside "
                          f"[{lo[j]:+.4f}, {hi[j]:+.4f}]"
                          + (f" (envelope widened from {JOINT_LOW[j]:+.3f}/"
                             f"{JOINT_HIGH[j]:+.3f} by the start pose)"
                             if outside[j] > 0 else ""))
        for key, cap, unit in (("dq", dq_c, "rad/s"), ("ddq", ddq_c, "rad/s^2"),
                               ("jerk", jerk_c, "rad/s^3")):
            if got[key][j] > cap[j] * (1.0 + CAP_SLACK):
                issues.append(f"{name}: |{key}|max={got[key][j]:.3f} > "
                              f"{cap[j]:.3f} {unit} (slow the move down)")
        pk = float(np.max(np.abs(tau[:, j])))
        cap_tau = tau_frac * lim[j]
        if pk > cap_tau:
            issues.append(f"{name}: |tau|max={pk:.2f} > {cap_tau:.2f} Nm "
                          f"({tau_frac:.0%} of the {lim[j]:.0f} Nm motor limit; "
                          "no headroom left for feedback)")
    return issues


def clip_to_limits(q):
    """Clip a commanded configuration to the EXACT joint limits, no inset margin.

    An inset margin would command +margin at t=0 from the resting pose, since that pose is
    itself a limit for two joints -- a step injected by the safety code. README quirk 2.
    """
    return np.clip(np.asarray(q, float), JOINT_LOW, JOINT_HIGH)


def format_report(t, q, dq, ddq, dq_max=None, ddq_max=None, jerk_max=None,
                  tau_limit=None, model=None) -> str:
    """Per-joint peaks vs caps and torque vs motor limit, as a printable table.

    Printed by every mode so the caps -- two of which have no measured basis -- are visible
    rather than implied.
    """
    dt = float(np.median(np.diff(t))) if len(t) > 1 else 1.0
    dq_c, ddq_c, jerk_c = _as_caps(dq_max, ddq_max, jerk_max)
    got = peaks(dq, ddq, dt)
    tau = rnea_torque(q, dq, ddq, model)
    lim = np.broadcast_to(np.asarray(TAU_LIMIT if tau_limit is None else tau_limit,
                                     float), (NUM_ARM_DOF,))

    lines = [f"duration {t[-1]:.3f} s over {len(t)} samples at dt={dt*1e3:.1f} ms",
             f"{'joint':<20}{'|dq|':>16}{'|ddq|':>18}{'|jerk|':>18}{'|tau|':>16}"]
    for j in range(NUM_ARM_DOF):
        name = arm_ik.IK_MODEL.names[j + 1]
        pk = float(np.max(np.abs(tau[:, j])))
        lines.append(
            f"{name:<20}"
            f"{got['dq'][j]:7.3f}/{dq_c[j]:<8.2f}"
            f"{got['ddq'][j]:8.3f}/{ddq_c[j]:<9.2f}"
            f"{got['jerk'][j]:8.2f}/{jerk_c[j]:<9.2f}"
            f"{pk:7.2f}/{TAU_FRAC * lim[j]:<8.2f}"
        )
    return "\n".join(lines)
