"""Task B: orbit-determination impact assessment for stellar aberration.

Runs identical position-only BLS-LM orbit determination on the same SPICE-fixture
arc with the stellar aberration model disabled vs enabled, and quantifies the
impact. Reality is taken to include stellar aberration (the physically correct,
SPICE-faithful SSB frame); the question is whether *omitting* the correction from
the measurement model measurably degrades the estimate.

Two frames are assessed:
  * spice_ssb  -- SPICE-faithful (SSB observer velocity, ~11 arcsec for Earth-Moon)
  * local_mci  -- cheaper Moon-relative approximation (~0.8 arcsec)

For each: measurement-level systematic bias (truth vs model-off), the post-fit OD
solution (state error, residual RMS, iterations, cost, conditioning), and the
formal covariance. Regenerates ``docs/stellar_aberration_od_impact.md``.

Run from the project root (SPICE kernels required)::

    python python_port/examples/od_impact_stellar_aberration.py
"""

from __future__ import annotations

import dataclasses
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import spiceypy as spice  # noqa: E402

from lunar_od import (  # noqa: E402
    MoonCenteredEphemeris,
    compute_position_residuals,
    estimate_position_bls_lm,
    generate_position_measurements,
    load_spice_kernels,
    range_rate_stations,
)
from lunar_od.geometry import wrap_to_pi  # noqa: E402

PYTHON_PORT = Path(__file__).resolve().parents[1]
FIXTURE = PYTHON_PORT / "fixtures" / "spice_snapshots.json"
OUT = PYTHON_PORT / "docs" / "stellar_aberration_od_impact.md"
ARCSEC = np.degrees(1.0) * 3600.0


def _load():
    fx = json.loads(FIXTURE.read_text(encoding="utf-8"))
    truth, meas, C = fx["truth_propagation"], fx["position_measurements"], fx["constants"]
    eph = MoonCenteredEphemeris(
        t_ephem_s=truth["t_ephem_s"], earth_pos_m=truth["earth_pos_grid_m"],
        sun_pos_m=truth["sun_pos_grid_m"], earth_vel_mps=truth["earth_vel_grid_mps"])
    by = {s.name: s for s in range_rate_stations()}
    stations = [by[n] for n in meas["station_names"]]
    return {
        "eph": eph, "stations": stations,
        "state": np.asarray(truth["state_history_mci_m_mps"], float),
        "tp": np.asarray(meas["t_pass_s"], float),
        "vis": meas["vis_mask_raw"], "et": fx["et"],
        "mu_moon": C["mu_moon_km3_s2"] * 1e9,
        "mu_earth": C["mu_earth_km3_s2"] * 1e9,
        "mu_sun": C["mu_sun_km3_s2"] * 1e9,
    }


def _gen(d, model):
    load_spice_kernels()
    try:
        _, pg, clean = generate_position_measurements(
            d["tp"], d["state"], d["stations"], d["vis"], d["eph"].earth_position,
            d["eph"].earth_velocity, d["et"], noise=False, apply_light_time=True,
            apply_stellar_aberration=True, stellar_aberration_model=model)
    finally:
        spice.kclear()
    return pg, clean


def _channel_residuals(state, clean, pg):
    """Per-channel residual array (n,3): range[m], az[rad], el[rad]."""
    _, h = compute_position_residuals(state, clean, pg)
    diff = clean[:, 1:4] - h
    diff[:, 1] = wrap_to_pi(diff[:, 1])
    diff[:, 2] = wrap_to_pi(diff[:, 2])
    return diff


def _rms(x):
    return float(np.sqrt(np.mean(np.square(x))))


def _bls(d, clean, pg, x0):
    x, stop, stats = estimate_position_bls_lm(
        d["tp"], clean, x0, pg, d["mu_moon"], d["mu_earth"], d["mu_sun"],
        d["eph"].earth_position, d["eph"].sun_position,
        max_iter=60, rtol=1e-11, atol=1e-12, return_posterior=True)
    return x, stop, stats


def assess(d, model):
    state = d["state"]
    x_true0 = state[0].copy()
    pg_on, clean = _gen(d, model)                  # truth measurements (with +S)
    pg_off = dataclasses.replace(pg_on, apply_stellar_aberration=False)

    # measurement-level systematic bias: truth vs model-off prediction at truth
    bias = _channel_residuals(state, clean, pg_off)     # (n,3)
    sig_range = np.array([d["stations"][int(clean[i, 4]) - 1].sigma_range_m
                          for i in range(clean.shape[0])])
    sig_ang = np.array([d["stations"][int(clean[i, 4]) - 1].sigma_angle_rad
                        for i in range(clean.shape[0])])
    ang_bias = np.sqrt(bias[:, 1] ** 2 + bias[:, 2] ** 2)   # combined az/el [rad]
    ratio_ang = ang_bias / sig_ang

    # OD off vs on (start from truth to isolate the measurement-model effect)
    x_on, stop_on, st_on = _bls(d, clean, pg_on, x_true0)
    x_off, stop_off, st_off = _bls(d, clean, pg_off, x_true0)

    def errs(x):
        return float(np.linalg.norm(x[:3] - x_true0[:3])), float(np.linalg.norm(x[3:6] - x_true0[3:6]))

    pe_on, ve_on = errs(x_on)
    pe_off, ve_off = errs(x_off)

    # post-fit residual RMS per channel (recompute at the estimated initial state,
    # propagated) -- use the estimator's own consistency: rebuild from x_est by a
    # cheap re-evaluation through compute_position_residuals on the *_on geometry.
    # (range residual is dominated by light-time geometry; az/el by aberration.)
    post_on = _post_fit_rms(d, clean, pg_on, x_on)
    post_off = _post_fit_rms(d, clean, pg_off, x_off)

    # formal 1-sigma position from the ON posterior covariance
    cov_on = getattr(st_on, "posterior_covariance", None)
    formal_pos_sigma = (float(np.sqrt(np.trace(np.asarray(cov_on)[:3, :3])))
                        if cov_on is not None else float("nan"))

    return {
        "model": model,
        "ang_bias_med_as": float(np.median(ang_bias) * ARCSEC),
        "ang_bias_max_as": float(np.max(ang_bias) * ARCSEC),
        "range_bias_max_m": float(np.max(np.abs(bias[:, 0]))),
        "ratio_ang_med": float(np.median(ratio_ang)),
        "ratio_ang_max": float(np.max(ratio_ang)),
        "pe_on": pe_on, "ve_on": ve_on, "pe_off": pe_off, "ve_off": ve_off,
        "iters_on": st_on.iterations, "iters_off": st_off.iterations,
        "cost_on": float(st_on.final_cost), "cost_off": float(st_off.final_cost),
        "cond_on": float(st_on.condition_number), "cond_off": float(st_off.condition_number),
        "stop_on": stop_on, "stop_off": stop_off,
        "post_on": post_on, "post_off": post_off,
        "formal_pos_sigma": formal_pos_sigma,
    }


def _post_fit_rms(d, clean, pg, x_est):
    """Propagate x_est over the arc and report per-channel residual RMS."""
    from lunar_od.dynamics import propagate_augmented_state

    x_aug0 = np.concatenate([x_est, np.eye(6).reshape(-1, order="F")])
    hist = propagate_augmented_state(
        d["tp"], x_aug0, d["mu_moon"], d["mu_earth"], d["mu_sun"],
        d["eph"].earth_position, d["eph"].sun_position, rtol=1e-11, atol=1e-12)
    diff = _channel_residuals(hist[:, :6], clean, pg)
    return (_rms(diff[:, 0]), _rms(diff[:, 1]) * ARCSEC, _rms(diff[:, 2]) * ARCSEC)


def write_report(results):
    L = []
    L.append("# Stellar Aberration — Orbit Determination Impact\n")
    L.append("_Task B deliverable — generated by "
             "`examples/od_impact_stellar_aberration.py`._\n")
    L.append("## Setup\n")
    L.append("Identical position-only (range/az/el) BLS-LM orbit determination on "
             "the same SPICE-fixture arc (single pass, 12 observations, two "
             "stations: Goldstone DSN σ_angle = 3.6″, ITU Ayazağa σ_angle = 18″). "
             "Truth measurements **include** stellar aberration (reality); the two "
             "runs differ only in whether the measurement **model** applies it. "
             "Both runs are started from the truth initial state, so any residual "
             "or state error is attributable solely to the measurement model. Same "
             "force model, ephemerides, and estimator settings throughout.\n")

    L.append("## 1. Measurement-level systematic bias (model OFF vs aberrated truth)\n")
    L.append("| frame | az/el bias (median / max) [arcsec] | range bias [m] | "
             "bias/σ_angle (median / max) |")
    L.append("|---|---:|---:|---:|")
    for r in results:
        L.append(f"| `{r['model']}` | {r['ang_bias_med_as']:.2f} / {r['ang_bias_max_as']:.2f} "
                 f"| {r['range_bias_max_m']:.2e} | {r['ratio_ang_med']:.2f} / {r['ratio_ang_max']:.2f} |")
    L.append("\nRange is unaffected (stellar aberration is a pure rotation). The "
             "az/el bias is the omitted aberration; for `spice_ssb` it exceeds the "
             "DSN angular noise floor, for `local_mci` it sits below it.\n")

    L.append("## 2. Orbit-determination outcome (off vs on)\n")
    L.append("| frame | run | pos err [m] | vel err [m/s] | post-fit RMS "
             "(range[m] / az[″] / el[″]) | iters | final cost | cond |")
    L.append("|---|---|---:|---:|---|---:|---:|---:|")
    for r in results:
        L.append(f"| `{r['model']}` | ON (modelled) | {r['pe_on']:.3f} | {r['ve_on']:.4f} "
                 f"| {r['post_on'][0]:.2e} / {r['post_on'][1]:.3f} / {r['post_on'][2]:.3f} "
                 f"| {r['iters_on']} | {r['cost_on']:.2e} | {r['cond_on']:.2e} |")
        L.append(f"| `{r['model']}` | OFF (omitted) | {r['pe_off']:.1f} | {r['ve_off']:.4f} "
                 f"| {r['post_off'][0]:.2e} / {r['post_off'][1]:.3f} / {r['post_off'][2]:.3f} "
                 f"| {r['iters_off']} | {r['cost_off']:.2e} | {r['cond_off']:.2e} |")
    L.append("")
    for r in results:
        L.append(f"- `{r['model']}`: formal 1σ position (ON posterior) = "
                 f"{r['formal_pos_sigma']:.2f} m; omitting the correction moves the "
                 f"estimate by {r['pe_off']:.0f} m / {r['ve_off']:.2f} m/s.")
    L.append("")

    L.append("## 3. Conclusion\n")
    ssb = next((r for r in results if r["model"] == "spice_ssb"), results[0])
    loc = next((r for r in results if r["model"] == "local_mci"), None)
    L.append("**Case A — measurable OD impact (for the physically-correct SSB "
             "frame).**\n")
    ssb_sigma_ratio = ssb["pe_off"] / ssb["formal_pos_sigma"] if ssb["formal_pos_sigma"] else float("nan")
    L.append(f"With the SPICE-faithful `spice_ssb` model, omitting stellar "
             f"aberration biases the angular measurements by ~{ssb['ang_bias_max_as']:.0f}″ "
             f"(≈{ssb['ratio_ang_max']:.1f}× the DSN angular noise) and shifts the "
             f"recovered state by **{ssb['pe_off']:.0f} m / {ssb['ve_off']:.2f} m/s** — "
             f"above the {ssb['formal_pos_sigma']:.0f} m formal 1σ of this weakly-observed "
             f"single arc (≈{ssb_sigma_ratio:.1f}σ). Modelling it recovers the truth to "
             f"{ssb['pe_on']:.3f} m. The effect is real because the ~11″ "
             "Earth-barycentric aberration, acting over the ~384 000 km Earth-Moon "
             "lever arm, maps to a kilometre-scale transverse position error that the "
             "range channel cannot absorb.\n")
    if loc is not None:
        L.append(f"With the cheaper `local_mci` model the bias is only "
                 f"~{loc['ang_bias_max_as']:.1f}″ (≈{loc['ratio_ang_max']:.1f}× the DSN "
                 f"noise) and the state impact is {loc['pe_off']:.0f} m — about an order "
                 "of magnitude smaller, since `local_mci` omits the dominant barycentric "
                 "term (confirming the Task A cross-validation: the two frames differ by "
                 "the Moon's barycentric aberration). For this arc that offset is below "
                 "the formal 1σ, i.e. **Case B** (negligible) — but a systematic bias is "
                 "coherent across all measurements and does not average down like random "
                 "noise, so it can still matter in a better-observed, multi-arc scenario.\n")
    L.append("Range observables are unaffected in every case (pure rotation), so "
             "the impact is confined to the angular channels and the orbit "
             "geometry they constrain.\n")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"wrote {OUT}")


def main() -> int:
    d = _load()
    results = [assess(d, "spice_ssb"), assess(d, "local_mci")]
    write_report(results)
    for r in results:
        print(f"{r['model']:>10}: OFF pos_err={r['pe_off']:.1f} m vel_err={r['ve_off']:.3f} "
              f"| ON pos_err={r['pe_on']:.4f} m | ang_bias_max={r['ang_bias_max_as']:.2f}\" "
              f"| bias/sig max={r['ratio_ang_max']:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
