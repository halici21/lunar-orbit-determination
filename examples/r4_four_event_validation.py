"""R4 (CD-4) four-event counted-Doppler validation campaign.

R4 productionises the already accepted four-event model-F capability for
explicit nonzero constant transponder delay.  This is a capability expansion,
not a claim that the preceding R2 campaign established an operational need for
nonzero delay.

Two campaigns, both written to CSV so every headline recomputes from raw rows:

* **migration (R4-P13)** -- accepted R3 exact zero-delay single-bounce versus R4
  four-event under the frozen *hypothetical* delay sweep, over the frozen R2
  count intervals and cadences, normalised by the station range-rate sigma.
* **estimator qualification (R4-P21)** -- BLS-LM and SRIF driven by R4
  generation and R4 evaluation.  Self-consistency only; no absolute physical
  validation is claimed, and the transponder delay is never a solve-for state.

The delay values come from repository authority
(``examples/r2_measurement_fidelity_validation.py`` ``FROZEN_TRANSPONDER_DELAYS_S``)
and are HYPOTHETICAL: no mission hardware delay is asserted anywhere.

Usage::

    python examples/r4_four_event_validation.py --output-dir <dir>
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import numpy as np

from lunar_od.config import Station
from lunar_od.dynamics import propagate_augmented_state
from lunar_od.diagnostics import analyze_convergence
from lunar_od.estimators import estimate_range_rate_bls_lm, estimate_range_rate_srif
from lunar_od.measurements import compute_range_rate_residuals, generate_range_rate_measurements
from lunar_od.radiometrics import RangeRatePhysicsConfig

# Frozen hypothetical sweep, recovered from repository authority.
FROZEN_TRANSPONDER_DELAYS_S = (0.0, 1e-6, 1e-5, 1e-4, 1e-3)
FROZEN_COUNT_INTERVALS_S = (1.0, 10.0, 30.0, 60.0, 100.0)
FROZEN_CADENCES_S = (0.1, 1.0, 3.0, 10.0, 30.0, 60.0, 120.0)

# Frozen materiality classification (frame-error budget + R2 decision contract).
NEGLIGIBLE_SIGMA = 0.1
BLOCKER_SIGMA = 0.5

MU_MOON = 4.9028e12


def _spice_epoch() -> float:
    """Reuse the accepted R2 campaign kernel loader and its covered epoch."""
    from examples import r2_measurement_fidelity_validation as campaign

    return campaign._spice_epoch_and_loader()


ET0 = None  # resolved in main(); generation needs a kernel-covered epoch.


def _station(name: str, lat_deg: float, lon_deg: float) -> Station:
    return Station(
        name=name,
        lat_deg=lat_deg,
        lon_deg=lon_deg,
        alt_m=0.0,
        color_rgb=(0.0, 0.0, 0.0),
        sigma_range_m=1.0,
        sigma_angle_rad=1e-5,
        sigma_range_rate_mps=1e-3,
    )


STATIONS = (
    _station("Synthetic A", 35.0, 20.0),
    _station("Synthetic B", -30.0, 140.0),
)


def _earth_pos(t):
    return np.tile(np.array([384400e3, 0.0, 0.0]), (np.size(np.asarray(t)), 1))


def _earth_vel(t):
    return np.tile(np.array([0.0, 1000.0, 0.0]), (np.size(np.asarray(t)), 1))


def _sun_pos(t):
    return np.tile(np.array([149.6e9, 0.0, 0.0]), (np.size(np.asarray(t)), 1))


# FA-03B: pre-roll ownership sits at the fixture level, never in the guard. The
# four-event chain reaches back one round-trip light time plus half a count
# interval before each receive tag, so the propagated history is extended and
# the outer epochs are simply not tagged as observations.
PRE_ROLL_S = 200.0


def _truth_arc(cadence_s: float, span_s: float = 900.0):
    radius = 1.938e6
    x0 = np.array([radius, 30e3, -20e3, -15.0, math.sqrt(MU_MOON / radius), 4.0])
    t_pass_s = np.arange(-PRE_ROLL_S, span_s + PRE_ROLL_S + 1e-9, cadence_s)
    x_aug0 = np.concatenate([x0, np.eye(6).reshape(-1, order="F")])
    x_aug = propagate_augmented_state(
        t_pass_s, x_aug0, MU_MOON, 0.0, 0.0, _earth_pos, _sun_pos, rtol=1e-12, atol=1e-13
    )
    return t_pass_s, x_aug[:, :6], x_aug


# The frozen matrix specifies interval/cadence COMBINATIONS, not arc length. A
# 0.1 s cadence over the full arc would tag ~26k observations per cell and make
# the campaign intractable without adding any coverage, so each cell is capped:
# the tagged epochs are subsampled uniformly inside the supported window. This
# is a fixture-size choice and changes no threshold or classification rule.
MAX_TAGS_PER_CELL = 24


def _visibility(t_pass_s):
    """Tag observations only where the full event chain has history support."""
    mask = (t_pass_s >= -PRE_ROLL_S + 150.0) & (t_pass_s <= t_pass_s[-1] - 150.0)
    supported = np.flatnonzero(mask)
    if supported.size > MAX_TAGS_PER_CELL:
        keep = supported[
            np.linspace(0, supported.size - 1, MAX_TAGS_PER_CELL).astype(int)
        ]
        mask = np.zeros_like(mask)
        mask[keep] = True
    return np.tile(mask.reshape(-1, 1), (1, len(STATIONS)))


def _physics(count_interval_s: float, delay_s: float, four_event: bool):
    kwargs = {}
    if four_event:
        kwargs["counted_doppler_model"] = "four_event_delay"
    return RangeRatePhysicsConfig(
        mode="two_way_counted_doppler",
        count_interval_s=count_interval_s,
        transponder_delay_s=delay_s if four_event else 0.0,
        **kwargs,
    )


def run_migration_campaign(output_dir: Path) -> dict:
    """R4-P13: classify the R3 -> R4 observable shift against the frozen budget."""
    rows = []
    worst = {"sigma": 0.0, "delay_s": None, "count_interval_s": None, "cadence_s": None}
    for cadence_s in FROZEN_CADENCES_S:
        t_pass_s, x_truth, _ = _truth_arc(cadence_s)
        visibility = _visibility(t_pass_s)
        for count_interval_s in FROZEN_COUNT_INTERVALS_S:
            if count_interval_s <= max(FROZEN_TRANSPONDER_DELAYS_S):
                continue
            baseline_obs, baseline_geo = generate_range_rate_measurements(
                t_pass_s, x_truth, STATIONS, visibility, _earth_pos, _earth_vel, ET0,
                noise=False, arc_id=1,
                range_rate_physics=_physics(count_interval_s, 0.0, four_event=False),
            )
            sigma = float(STATIONS[0].sigma_range_rate_mps)
            for delay_s in FROZEN_TRANSPONDER_DELAYS_S:
                geo = generate_range_rate_measurements(
                    t_pass_s, x_truth, STATIONS, visibility, _earth_pos, _earth_vel, ET0,
                    noise=False, arc_id=1,
                    range_rate_physics=_physics(count_interval_s, delay_s, four_event=True),
                )[0]
                shift = geo[:, 2] - baseline_obs[:, 2]
                max_shift = float(np.max(np.abs(shift)))
                over_sigma = max_shift / sigma
                if over_sigma > NEGLIGIBLE_SIGMA:
                    label = "blocker" if over_sigma > BLOCKER_SIGMA else "material"
                else:
                    label = "negligible"
                rows.append(
                    [cadence_s, count_interval_s, delay_s, len(shift), repr(max_shift),
                     repr(over_sigma), repr(sigma), label, "hypothetical_capability_sweep"]
                )
                if over_sigma > worst["sigma"]:
                    worst = {"sigma": over_sigma, "delay_s": delay_s,
                             "count_interval_s": count_interval_s, "cadence_s": cadence_s}
    _write(output_dir / "migration_results.csv",
           ["cadence_s", "count_interval_s", "transponder_delay_s", "n_observations",
            "max_abs_observable_shift_mps", "observable_shift_over_sigma",
            "sigma_range_rate_mps", "classification", "delay_provenance"], rows)

    # First-order scaling check: the shift must be linear in the delay.
    scale_rows = []
    t_pass_s, x_truth, _ = _truth_arc(10.0)
    visibility = _visibility(t_pass_s)
    baseline_obs = generate_range_rate_measurements(
        t_pass_s, x_truth, STATIONS, visibility, _earth_pos, _earth_vel, ET0,
        noise=False, arc_id=1, range_rate_physics=_physics(60.0, 0.0, four_event=False),
    )[0]
    previous = None
    orders = []
    for delay_s in (1e-3, 5e-4, 2.5e-4, 1.25e-4, 6.25e-5):
        obs = generate_range_rate_measurements(
            t_pass_s, x_truth, STATIONS, visibility, _earth_pos, _earth_vel, ET0,
            noise=False, arc_id=1, range_rate_physics=_physics(60.0, delay_s, True),
        )[0]
        shift = float(np.max(np.abs(obs[:, 2] - baseline_obs[:, 2])))
        order = ""
        if previous is not None and shift > 0.0:
            order = math.log(previous[1] / shift) / math.log(previous[0] / delay_s)
            orders.append(order)
        scale_rows.append([delay_s, repr(shift), repr(shift / delay_s), repr(order)])
        previous = (delay_s, shift)
    _write(output_dir / "migration_delay_scaling.csv",
           ["transponder_delay_s", "max_abs_shift_mps", "shift_over_delay",
            "measured_scaling_order"], scale_rows)
    worst["scaling_order_min"] = min(orders) if orders else float("nan")
    worst["scaling_order_max"] = max(orders) if orders else float("nan")
    worst["rows"] = len(rows)
    return worst


def run_estimator_qualification(output_dir: Path) -> list:
    """R4-P21: BLS-LM and SRIF self-consistency under the R4 model."""
    rows = []
    t_pass_s, x_truth, _ = _truth_arc(30.0)
    visibility = _visibility(t_pass_s)
    for delay_s in (0.0, 1e-4, 1e-3):
        physics = _physics(60.0, delay_s, four_event=True)
        obs_data, pass_geo = generate_range_rate_measurements(
            t_pass_s, x_truth, STATIONS, visibility, _earth_pos, _earth_vel, ET0,
            noise=False, arc_id=1, range_rate_physics=physics,
        )
        perturbation = np.array([500.0, -400.0, 300.0, 0.4, -0.3, 0.2])
        x_nominal0 = x_truth[0] + perturbation
        error_before = float(np.linalg.norm(perturbation[:3]))
        for name, estimator in (("bls_lm", estimate_range_rate_bls_lm),
                                ("srif", estimate_range_rate_srif)):
            x_hat, stop_reason, stats = estimator(
                t_pass_s, obs_data, x_nominal0, pass_geo, MU_MOON, 0.0, 0.0,
                _earth_pos, _sun_pos,
            )
            x_hat = np.asarray(x_hat, dtype=float)
            error_after = float(np.linalg.norm(x_hat[:3] - x_truth[0][:3]))
            diagnostics = analyze_convergence(
                stop_reason,
                stats=stats,
                rank=stats.rank,
                expected_rank=6,
                condition_number=stats.condition_number,
                final_cost=stats.final_cost,
            )
            # Residuals must come from the PROPAGATED estimated trajectory, not
            # from a constant state tiled across the arc.
            x_est_aug = propagate_augmented_state(
                t_pass_s,
                np.concatenate([x_hat, np.eye(6).reshape(-1, order="F")]),
                MU_MOON, 0.0, 0.0, _earth_pos, _sun_pos, rtol=1e-12, atol=1e-13,
            )
            residuals = compute_range_rate_residuals(
                x_est_aug[:, :6], obs_data, pass_geo
            )[0]
            residual_rms = float(
                np.sqrt(np.mean(np.square(np.asarray(residuals).reshape(-1))))
            )
            rows.append([
                name, delay_s, bool(np.all(np.isfinite(x_hat))), stop_reason,
                diagnostics.category, diagnostics.converged,
                diagnostics.finite_final_cost, int(stats.iterations),
                repr(float(stats.final_cost)), repr(float(stats.condition_number)),
                int(stats.rank), repr(error_before), repr(error_after),
                repr(error_after / error_before if error_before else float("nan")),
                repr(residual_rms), "self_consistency_only", False,
            ])
    _write(output_dir / "estimator_qualification.csv",
           ["estimator", "transponder_delay_s", "finite_result", "stop_reason",
            "convergence_category", "converged", "finite_final_cost", "iterations",
            "final_cost", "condition_number", "rank", "position_error_before_m",
            "position_error_after_m", "error_ratio", "residual_rms_mps",
            "interpretation", "delay_is_solve_for"], rows)
    return rows


def _write(path: Path, header, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)
    print(f"  wrote {path.name}: {len(rows)} rows")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    global ET0
    ET0 = _spice_epoch()
    print(f"SPICE epoch {ET0!r} (kernel-covered, from the accepted R2 loader)")
    print("R4 migration campaign (hypothetical capability sweep)...")
    worst = run_migration_campaign(args.output_dir)
    print(f"  rows={worst['rows']}")
    print(f"  maximum observable shift = {worst['sigma']:.6f} sigma "
          f"at delay {worst['delay_s']!r} s, Tc {worst['count_interval_s']!r} s, "
          f"cadence {worst['cadence_s']!r} s")
    print(f"  measured delay scaling order in "
          f"[{worst['scaling_order_min']:.4f}, {worst['scaling_order_max']:.4f}]")
    print("R4 estimator qualification (self-consistency only)...")
    rows = run_estimator_qualification(args.output_dir)
    for row in rows:
        print(f"  {row[0]:7s} delay={row[1]:g} stop={row[3]} category={row[4]} "
              f"converged={row[5]} iters={row[7]} error {float(row[11]):.2f} -> "
              f"{float(row[12]):.5f} m  rms={float(row[14]):.3e} m/s")


if __name__ == "__main__":
    main()
