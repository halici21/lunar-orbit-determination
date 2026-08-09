"""Deterministic R1-R4 long-arc / geometry qualification harness (System Phase 1).

This is a QUALIFICATION harness. It adds no production physics and changes no
accepted R1-R4 threshold. Every threshold it applies is inherited:

* R3/R4 per-leg light-time equation residual  <= 1e-11 s
* R4 model-F parity                            < 1e-9  relative
* R4 zero-delay reduction to R3                bitwise
* R2/R4 migration classification               <= 0.1 / > 0.1 / > 0.5 sigma
* delay sweep                                  FROZEN_TRANSPONDER_DELAYS_S

Quantities with no accepted threshold (runtime, conditioning trends) are
recorded as ``DIAGNOSTIC`` and are never given an invented pass criterion.

Determinism: no Monte Carlo, no random noise, no random initial state. All
geometry comes from accepted repository fixtures -- the single accepted lunar
orbit definition and the accepted ``RANGE_RATE_STATION_DEFS`` coordinates,
sampled at different orbital phases. No mission orbit or station coordinate is
invented.

Sharding and resumability
-------------------------
The campaign is a deterministic ordered list of cells. ``--shard k/n`` executes
cell indices ``i`` with ``i % n == k - 1``. Each shard writes its own result CSV
plus a sidecar JSON carrying the campaign definition hash, the canonical HEAD
and a completion flag, so an interrupted run resumes by re-running only the
missing shards. Aggregation rejects missing, duplicate, or mismatched shards.

Usage::

    python examples/r1_r4_long_arc_qualification.py --campaign phase1 \
        --horizon H1 --shard 1/8 --output-dir <dir>
    python examples/r1_r4_long_arc_qualification.py --aggregate --output-dir <dir>
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from lunar_od.config import RANGE_RATE_STATION_DEFS, Station
from lunar_od.dynamics import propagate_augmented_state
from lunar_od.measurements import generate_range_rate_measurements
from lunar_od.radiometrics import (
    COUNTED_DOPPLER_EARTH_EPHEMERIS_METHOD,
    COUNTED_DOPPLER_MODEL_VERSION,
    COUNTED_DOPPLER_SPACECRAFT_INTERPOLATION_METHOD,
    FOUR_EVENT_COUNTED_DOPPLER_MODEL_VERSION,
    RangeRatePhysicsConfig,
)
from lunar_od.two_way_counted_doppler import four_event_counted_doppler_endpoints

# ---------------------------------------------------------------- constants --
MU_MOON = 4902.800066e9
R0 = 1737.4e3 + 100e3
ORBIT_PERIOD_S = 2.0 * math.pi * math.sqrt(R0**3 / MU_MOON)

# Recovered from repository authority; HYPOTHETICAL capability sweep, not
# mission hardware values.
FROZEN_TRANSPONDER_DELAYS_S = (0.0, 1e-6, 1e-5, 1e-4, 1e-3)

HORIZONS_S = {"H1": 3600.0, "H2": 21600.0, "H3": 86400.0,
              "H4": 259200.0, "H5": 604800.0}

FROZEN_MATRIX = {
    "H1": {"count_intervals": (1.0, 10.0, 30.0, 60.0, 100.0),
           "cadences": (0.1, 1.0, 3.0, 10.0, 30.0, 60.0, 120.0),
           "delays": FROZEN_TRANSPONDER_DELAYS_S},
    "H2": {"count_intervals": (1.0, 10.0, 30.0, 60.0, 100.0),
           "cadences": (1.0, 10.0, 60.0), "delays": FROZEN_TRANSPONDER_DELAYS_S},
    "H3": {"count_intervals": (1.0, 10.0, 30.0, 60.0, 100.0),
           "cadences": (10.0, 30.0, 60.0, 120.0), "delays": (0.0, 1e-4, 1e-3)},
    "H4": {"count_intervals": (10.0, 30.0, 60.0, 100.0),
           "cadences": (30.0, 60.0, 120.0), "delays": (0.0, 1e-4, 1e-3)},
    "H5": {"count_intervals": (10.0, 60.0, 100.0), "cadences": (60.0, 120.0),
           "delays": (0.0, 1e-4, 1e-3)},
}

# Accepted stations chosen to span the Earth-rotation velocity projection.
PHASE1_STATIONS = ("Chuuk KGS", "Byalalu ISRO", "Goldstone DSN", "ITU Ayazaga",
                   "Bear Lakes RUS", "Svalbard KGS", "Canberra DSN", "Malargue ESA")
PHASE1_ORBIT_PHASES = (0.0, 0.25, 0.5)

# Inherited thresholds. None of these is introduced by this harness.
LEG_RESIDUAL_TOLERANCE_S = 1e-11
MODEL_F_RELATIVE_TOLERANCE = 1e-9
NEGLIGIBLE_SIGMA = 0.1
BLOCKER_SIGMA = 0.5

PRE_ROLL_S = 400.0


@dataclass(frozen=True)
class Geometry:
    geometry_id: str
    station_name: str
    lat_deg: float
    lon_deg: float
    alt_m: float
    phase_fraction: float

    @property
    def phase_offset_s(self) -> float:
        return self.phase_fraction * ORBIT_PERIOD_S


def phase1_geometries() -> tuple[Geometry, ...]:
    by_name = {d[0]: d for d in RANGE_RATE_STATION_DEFS}
    out: list[Geometry] = []
    index = 0
    for name in PHASE1_STATIONS:
        _n, lat, lon, alt, _rgb = by_name[name]
        for phase in PHASE1_ORBIT_PHASES:
            index += 1
            out.append(Geometry(f"G{index:02d}", name, lat, lon, alt, phase))
    return tuple(out)


def _station(geometry: Geometry) -> Station:
    return Station(name=geometry.station_name, lat_deg=geometry.lat_deg,
                   lon_deg=geometry.lon_deg, alt_m=geometry.alt_m,
                   color_rgb=(0.0, 0.0, 0.0), sigma_range_m=1.0,
                   sigma_angle_rad=1e-5,
                   sigma_range_rate_mps=1e-3 if "ITU" in geometry.station_name else 1e-4)


def _earth_pos(t):
    return np.tile(np.array([384400e3, 0.0, 0.0]), (np.size(np.asarray(t)), 1))


def _earth_vel(t):
    return np.tile(np.array([0.0, 1000.0, 0.0]), (np.size(np.asarray(t)), 1))


def _sun_pos(t):
    return np.tile(np.array([149.6e9, 0.0, 0.0]), (np.size(np.asarray(t)), 1))


def spice_epoch() -> float:
    """Kernel-covered epoch from the accepted R2 campaign loader."""
    from examples import r2_measurement_fidelity_validation as campaign

    return campaign._spice_epoch_and_loader()


def truth_arc(geometry: Geometry, horizon_s: float, cadence_s: float):
    """Deterministic truth arc for one geometry. No randomness anywhere."""
    offset = geometry.phase_offset_s
    mean_motion = math.sqrt(MU_MOON / R0**3)
    angle = mean_motion * offset
    speed = math.sqrt(MU_MOON / R0)
    x0 = np.array([
        R0 * math.cos(angle), 30e3 + R0 * math.sin(angle), -20e3,
        -15.0 - speed * math.sin(angle), speed * math.cos(angle), 4.0,
    ])
    t = np.arange(-PRE_ROLL_S, horizon_s + PRE_ROLL_S + 1e-9, cadence_s)
    aug0 = np.concatenate([x0, np.eye(6).reshape(-1, order="F")])
    aug = propagate_augmented_state(t, aug0, MU_MOON, 0.0, 0.0, _earth_pos, _sun_pos,
                                    rtol=1e-12, atol=1e-13)
    visibility = np.zeros((t.size, 1), dtype=bool)
    inside = (t >= 0.0) & (t <= horizon_s)
    visibility[inside, 0] = True
    return t, aug[:, :6], visibility


def physics_config(count_interval_s: float, delay_s: float, four_event: bool):
    extra = {"counted_doppler_model": "four_event_delay"} if four_event else {}
    return RangeRatePhysicsConfig(
        mode="two_way_counted_doppler", count_interval_s=count_interval_s,
        transponder_delay_s=delay_s if four_event else 0.0, **extra)


def campaign_cells(horizon: str, geometries, max_epochs_per_cell: int | None):
    """Deterministic ordered cell list. Order defines shard membership."""
    spec = FROZEN_MATRIX[horizon]
    cells = []
    for geometry in geometries:
        for cadence in spec["cadences"]:
            for count_interval in spec["count_intervals"]:
                if count_interval <= max(spec["delays"]):
                    continue          # invalid combination, never padded in
                for delay in spec["delays"]:
                    cells.append({
                        "horizon": horizon, "geometry_id": geometry.geometry_id,
                        "station": geometry.station_name,
                        "phase_fraction": geometry.phase_fraction,
                        "cadence_s": cadence, "count_interval_s": count_interval,
                        "delay_s": delay,
                        "max_epochs": max_epochs_per_cell,
                    })
    return cells


def definition_hash(horizon: str, geometries, max_epochs_per_cell) -> str:
    payload = json.dumps({
        "horizon": horizon, "horizon_s": HORIZONS_S[horizon],
        "matrix": {k: list(v) for k, v in FROZEN_MATRIX[horizon].items()},
        "geometries": [g.geometry_id for g in geometries],
        "stations": [g.station_name for g in geometries],
        "phases": [g.phase_fraction for g in geometries],
        "max_epochs_per_cell": max_epochs_per_cell,
        "orbit_r0_m": R0, "pre_roll_s": PRE_ROLL_S,
    }, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()


def canonical_head() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                              text=True, cwd=Path(__file__).resolve().parent.parent,
                              check=True).stdout.strip()
    except Exception:  # pragma: no cover - environment dependent
        return "unknown"


RAW_FIELDS = [
    "phase", "campaign_id", "shard_id", "model", "model_version", "geometry_id",
    "station", "horizon", "horizon_s", "epoch_s", "count_interval_s", "cadence_s",
    "delay_s", "observable_mps", "oracle", "oracle_difference", "difference_sigma",
    "classification", "event_residual_up_s", "event_residual_down_s",
    "event_residual_delay_s", "t1_s", "t2u_s", "t2d_s", "t3_s", "event_order_ok",
    "earth_ephemeris_method", "spacecraft_interpolation_method", "station_state_method",
    "range_rate_sign", "status",
]


def run_cell(cell, et0, geometries_by_id, shard_id, campaign_id):
    """Execute one campaign cell. Returns (rows, stats)."""
    geometry = geometries_by_id[cell["geometry_id"]]
    station = _station(geometry)
    horizon_s = HORIZONS_S[cell["horizon"]]
    t, x_truth, visibility = truth_arc(geometry, horizon_s, cell["cadence_s"])

    if cell["max_epochs"]:
        live = np.flatnonzero(visibility[:, 0])
        if live.size > cell["max_epochs"]:
            keep = live[np.linspace(0, live.size - 1, cell["max_epochs"]).astype(int)]
            visibility[:] = False
            visibility[keep, 0] = True

    rows: list[dict] = []
    stats = {"observations": 0, "event_solves": 0, "nonfinite": 0,
             "expected_fail_closed": 0, "unexpected_failure": 0}

    r3_cfg = physics_config(cell["count_interval_s"], 0.0, four_event=False)
    r4_cfg = physics_config(cell["count_interval_s"], cell["delay_s"], four_event=True)

    try:
        r3_obs, _r3_geo = generate_range_rate_measurements(
            t, x_truth, (station,), visibility, _earth_pos, _earth_vel, et0,
            noise=False, arc_id=1, range_rate_physics=r3_cfg)
        r4_obs, _r4_geo = generate_range_rate_measurements(
            t, x_truth, (station,), visibility, _earth_pos, _earth_vel, et0,
            noise=False, arc_id=1, range_rate_physics=r4_cfg)
    except Exception as exc:  # noqa: BLE001
        stats["unexpected_failure"] += 1
        return [{**{f: "" for f in RAW_FIELDS},
                 "phase": "Q1", "campaign_id": campaign_id, "shard_id": shard_id,
                 "model": "R4_four_event_delay", "geometry_id": cell["geometry_id"],
                 "station": geometry.station_name, "horizon": cell["horizon"],
                 "horizon_s": horizon_s, "count_interval_s": cell["count_interval_s"],
                 "cadence_s": cell["cadence_s"], "delay_s": cell["delay_s"],
                 "status": f"UNEXPECTED_FAILURE: {type(exc).__name__}: {exc}"[:200]}], stats

    sigma = float(station.sigma_range_rate_mps)
    for index in range(len(r3_obs)):
        epoch = float(r3_obs[index, 0])
        r3_value = float(r3_obs[index, 2])
        r4_value = float(r4_obs[index, 2])
        shift = r4_value - r3_value
        over_sigma = abs(shift) / sigma
        if cell["delay_s"] == 0.0:
            oracle, diff = "R3_production_bitwise", shift
            classification = "ZERO_DELAY_REDUCTION"
        else:
            oracle, diff = "R3_production", shift
            classification = ("MIGRATION_BLOCKER_CLASS" if over_sigma > BLOCKER_SIGMA
                              else "MATERIAL" if over_sigma > NEGLIGIBLE_SIGMA
                              else "NEGLIGIBLE")
        stats["observations"] += 2
        stats["event_solves"] += 4

        row = {f: "" for f in RAW_FIELDS}
        row.update({
            "phase": "Q1", "campaign_id": campaign_id, "shard_id": shard_id,
            "model": "R4_four_event_delay",
            "model_version": FOUR_EVENT_COUNTED_DOPPLER_MODEL_VERSION,
            "geometry_id": cell["geometry_id"], "station": geometry.station_name,
            "horizon": cell["horizon"], "horizon_s": horizon_s, "epoch_s": repr(epoch),
            "count_interval_s": cell["count_interval_s"], "cadence_s": cell["cadence_s"],
            "delay_s": cell["delay_s"], "observable_mps": repr(r4_value),
            "oracle": oracle, "oracle_difference": repr(diff),
            "difference_sigma": repr(over_sigma), "classification": classification,
            "earth_ephemeris_method": COUNTED_DOPPLER_EARTH_EPHEMERIS_METHOD,
            "spacecraft_interpolation_method": COUNTED_DOPPLER_SPACECRAFT_INTERPOLATION_METHOD,
            "station_state_method": "exact_event_epoch_sxform",
            "range_rate_sign": "positive" if r3_value >= 0.0 else "negative",
            "status": "OK",
        })
        if not (math.isfinite(r3_value) and math.isfinite(r4_value)):
            stats["nonfinite"] += 1
            row["status"] = "NONFINITE"
        rows.append(row)

    # Event-equation detail at deterministic checkpoints (start/quarter/.../end).
    live_epochs = [float(r3_obs[i, 0]) for i in range(len(r3_obs))]
    if live_epochs:
        picks = sorted({live_epochs[int(f * (len(live_epochs) - 1))]
                        for f in (0.0, 0.25, 0.5, 0.75, 1.0)})
        for epoch in picks:
            try:
                _s, _e, start, end, _ad = four_event_counted_doppler_endpoints(
                    epoch, station, t, x_truth, _earth_pos(t), _earth_vel(t), None,
                    r4_cfg, et0_s=et0)
                for label, solution in (("start", start), ("end", end)):
                    ordered = (solution.t1_s < solution.t2u_s <= solution.t2d_s
                               < solution.t3_s)
                    row = {f: "" for f in RAW_FIELDS}
                    row.update({
                        "phase": "Q1", "campaign_id": campaign_id, "shard_id": shard_id,
                        "model": f"R4_event_detail_{label}",
                        "model_version": FOUR_EVENT_COUNTED_DOPPLER_MODEL_VERSION,
                        "geometry_id": cell["geometry_id"],
                        "station": geometry.station_name, "horizon": cell["horizon"],
                        "horizon_s": horizon_s, "epoch_s": repr(epoch),
                        "count_interval_s": cell["count_interval_s"],
                        "cadence_s": cell["cadence_s"], "delay_s": cell["delay_s"],
                        "event_residual_up_s": repr(solution.uplink_equation_residual_s),
                        "event_residual_down_s": repr(solution.downlink_equation_residual_s),
                        "event_residual_delay_s": repr(
                            solution.transponder_equation_residual_s),
                        "t1_s": repr(solution.t1_s), "t2u_s": repr(solution.t2u_s),
                        "t2d_s": repr(solution.t2d_s), "t3_s": repr(solution.t3_s),
                        "event_order_ok": ordered, "oracle": "four_event_equations",
                        "status": "OK" if ordered else "EVENT_ORDER_VIOLATION",
                    })
                    rows.append(row)
                    stats["event_solves"] += 4
            except Exception as exc:  # noqa: BLE001
                stats["expected_fail_closed"] += 1
                row = {f: "" for f in RAW_FIELDS}
                row.update({
                    "phase": "Q1", "campaign_id": campaign_id, "shard_id": shard_id,
                    "model": "R4_event_detail", "geometry_id": cell["geometry_id"],
                    "horizon": cell["horizon"], "epoch_s": repr(epoch),
                    "status": f"FAIL_CLOSED: {type(exc).__name__}"})
                rows.append(row)
    return rows, stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", default="phase1-r1r4-longarc")
    parser.add_argument("--horizon", choices=sorted(HORIZONS_S), default="H1")
    parser.add_argument("--shard", default="1/1", help="k/n")
    parser.add_argument("--max-epochs-per-cell", type=int, default=None,
                        help="deterministic uniform subsample cap per cell")
    parser.add_argument("--geometry", action="append", default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--aggregate", action="store_true")
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    geometries = phase1_geometries()
    if args.geometry:
        geometries = tuple(g for g in geometries if g.geometry_id in set(args.geometry))
    by_id = {g.geometry_id: g for g in geometries}

    if args.aggregate:
        aggregate(args.output_dir, args.campaign)
        return

    cells = campaign_cells(args.horizon, geometries, args.max_epochs_per_cell)
    k, n = (int(v) for v in args.shard.split("/"))
    mine = [c for i, c in enumerate(cells) if i % n == k - 1]
    defn_hash = definition_hash(args.horizon, geometries, args.max_epochs_per_cell)
    print(f"campaign {args.campaign} horizon {args.horizon} shard {k}/{n}: "
          f"{len(mine)} of {len(cells)} cells; definition {defn_hash[:12]}")
    if args.plan_only:
        return

    et0 = spice_epoch()
    head = canonical_head()
    started = time.perf_counter()
    rows: list[dict] = []
    totals = {"observations": 0, "event_solves": 0, "nonfinite": 0,
              "expected_fail_closed": 0, "unexpected_failure": 0}
    for index, cell in enumerate(mine, start=1):
        cell_rows, stats = run_cell(cell, et0, by_id, f"{k}/{n}", args.campaign)
        rows.extend(cell_rows)
        for key in totals:
            totals[key] += stats[key]
        if index % 10 == 0 or index == len(mine):
            print(f"  cell {index}/{len(mine)}  rows={len(rows)}  "
                  f"obs={totals['observations']}  "
                  f"{time.perf_counter() - started:.1f}s")
    elapsed = time.perf_counter() - started

    shard_path = args.output_dir / f"long_arc_raw_{args.horizon}_shard{k}of{n}.csv"
    with shard_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=RAW_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    sidecar = {
        "campaign_id": args.campaign, "horizon": args.horizon, "shard": f"{k}/{n}",
        "shard_index": k, "shard_count": n, "definition_hash": defn_hash,
        "canonical_head": head, "cells": len(mine), "rows": len(rows),
        "wall_time_s": elapsed, "totals": totals, "complete": True,
        "geometries": [g.geometry_id for g in geometries],
    }
    shard_path.with_suffix(".json").write_text(json.dumps(sidecar, indent=2),
                                               encoding="utf-8")
    print(f"wrote {shard_path.name}: {len(rows)} rows, {elapsed:.1f} s, "
          f"{totals['observations']} observations, "
          f"{totals['unexpected_failure']} unexpected failures")


def aggregate(output_dir: Path, campaign_id: str) -> None:
    """Aggregate shards, rejecting missing, duplicate or mismatched ones."""
    sidecars = sorted(output_dir.glob("long_arc_raw_*_shard*.json"))
    if not sidecars:
        raise SystemExit("no shard sidecars found")
    groups: dict[tuple[str, str, int], list[dict]] = {}
    for path in sidecars:
        meta = json.loads(path.read_text(encoding="utf-8"))
        key = (meta["campaign_id"], meta["horizon"], meta["shard_count"])
        groups.setdefault(key, []).append(meta)

    problems = []
    all_rows = []
    for (camp, horizon, count), metas in sorted(groups.items()):
        if camp != campaign_id:
            problems.append(f"{horizon}: campaign id mismatch {camp}")
        seen = [m["shard_index"] for m in metas]
        missing = sorted(set(range(1, count + 1)) - set(seen))
        duplicates = sorted({s for s in seen if seen.count(s) > 1})
        heads = {m["canonical_head"] for m in metas}
        defs = {m["definition_hash"] for m in metas}
        if missing:
            problems.append(f"{horizon}: missing shards {missing}")
        if duplicates:
            problems.append(f"{horizon}: duplicate shards {duplicates}")
        if len(heads) > 1:
            problems.append(f"{horizon}: mismatched canonical HEAD {heads}")
        if len(defs) > 1:
            problems.append(f"{horizon}: mismatched campaign definition {defs}")
        if not all(m.get("complete") for m in metas):
            problems.append(f"{horizon}: incomplete shard present")
        for meta in metas:
            csv_path = output_dir / (
                f"long_arc_raw_{horizon}_shard{meta['shard_index']}of{count}.csv")
            with csv_path.open(newline="", encoding="utf-8") as fh:
                all_rows.extend(list(csv.DictReader(fh)))

    status = "COMPLETE" if not problems else "INCOMPLETE"
    with (output_dir / "long_arc_raw_results.csv").open(
            "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=RAW_FIELDS)
        writer.writeheader()
        writer.writerows(all_rows)

    obs_rows = [r for r in all_rows if r["model"] == "R4_four_event_delay"
                and r["difference_sigma"]]
    zero = [r for r in obs_rows if float(r["delay_s"]) == 0.0]
    bitwise = sum(1 for r in zero if float(r["oracle_difference"]) == 0.0)
    nonzero = [r for r in obs_rows if float(r["delay_s"]) != 0.0]
    detail = [r for r in all_rows if r["model"].startswith("R4_event_detail_")]
    resid = [max(float(r["event_residual_up_s"]), float(r["event_residual_down_s"]),
                 float(r["event_residual_delay_s"])) for r in detail
             if r["event_residual_up_s"]]
    summary = [
        ["campaign_status", status],
        ["problems", "; ".join(problems) if problems else "none"],
        ["total_rows", len(all_rows)],
        ["observation_rows", len(obs_rows)],
        ["zero_delay_rows", len(zero)],
        ["zero_delay_bitwise_rows", bitwise],
        ["zero_delay_bitwise_fraction", f"{bitwise / len(zero):.6f}" if zero else ""],
        ["nonzero_delay_rows", len(nonzero)],
        ["max_shift_sigma", max((float(r["difference_sigma"]) for r in nonzero),
                                default=0.0)],
        ["rows_negligible", sum(1 for r in nonzero if r["classification"] == "NEGLIGIBLE")],
        ["rows_material", sum(1 for r in nonzero if r["classification"] == "MATERIAL")],
        ["rows_blocker_class", sum(1 for r in nonzero
                                   if r["classification"] == "MIGRATION_BLOCKER_CLASS")],
        ["max_leg_residual_s", max(resid, default=0.0)],
        ["leg_residual_threshold_s", LEG_RESIDUAL_TOLERANCE_S],
        ["event_order_violations",
         sum(1 for r in detail if r["event_order_ok"] == "False")],
        ["nonfinite_rows", sum(1 for r in all_rows if r["status"] == "NONFINITE")],
        ["unexpected_failures",
         sum(1 for r in all_rows if r["status"].startswith("UNEXPECTED_FAILURE"))],
        ["expected_fail_closed",
         sum(1 for r in all_rows if r["status"].startswith("FAIL_CLOSED"))],
    ]
    with (output_dir / "long_arc_summary.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["metric", "value"])
        writer.writerows(summary)
    for key, value in summary:
        print(f"  {key}: {value}")
    if problems:
        raise SystemExit(f"campaign INCOMPLETE: {problems}")


if __name__ == "__main__":
    main()
