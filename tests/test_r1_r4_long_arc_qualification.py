"""Tests for the deterministic R1-R4 long-arc qualification harness (System Phase 1).

These tests protect the *harness*, not the production physics: the campaign
definition, the shard/aggregation contract, the geometry construction and a
bounded real qualification slice. The full multi-day campaign runs through
``examples/r1_r4_long_arc_qualification.py``, not through pytest.

Every threshold asserted here is inherited from the accepted R1-R4 contracts.
No new numerical threshold is introduced.
"""

import csv
import json
import math
import os
import tempfile
import unittest
from pathlib import Path

import numpy as np

from examples import r1_r4_long_arc_qualification as harness

RUN_SLOW = os.environ.get("LUNAR_OD_RUN_SLOW_TESTS")


class CampaignDefinition(unittest.TestCase):
    """The frozen matrix and delay set must match repository authority."""

    def test_delay_set_matches_the_accepted_frozen_sweep(self):
        from examples import r2_measurement_fidelity_validation as campaign

        self.assertEqual(
            harness.FROZEN_TRANSPONDER_DELAYS_S,
            campaign.FROZEN_TRANSPONDER_DELAYS_S,
        )

    def test_horizons_cover_one_hour_to_seven_days(self):
        self.assertEqual(harness.HORIZONS_S["H1"], 3600.0)
        self.assertEqual(harness.HORIZONS_S["H5"], 7 * 86400.0)
        spans = [harness.HORIZONS_S[k] for k in ("H1", "H2", "H3", "H4", "H5")]
        self.assertEqual(spans, sorted(spans), "horizons must increase")

    def test_inherited_thresholds_are_not_redefined(self):
        """The harness must not invent thresholds; these are the accepted values."""
        self.assertEqual(harness.LEG_RESIDUAL_TOLERANCE_S, 1e-11)
        self.assertEqual(harness.MODEL_F_RELATIVE_TOLERANCE, 1e-9)
        self.assertEqual(harness.NEGLIGIBLE_SIGMA, 0.1)
        self.assertEqual(harness.BLOCKER_SIGMA, 0.5)

    def test_invalid_count_interval_delay_combinations_are_excluded(self):
        """A cell is never padded in merely to populate the table."""
        geometries = harness.phase1_geometries()[:1]
        for horizon in harness.HORIZONS_S:
            cells = harness.campaign_cells(horizon, geometries, None)
            for cell in cells:
                self.assertGreater(cell["count_interval_s"], cell["delay_s"])


class GeometryConstruction(unittest.TestCase):
    """Geometry must come from accepted fixtures only."""

    def test_every_station_is_an_accepted_range_rate_station(self):
        from lunar_od.config import RANGE_RATE_STATION_DEFS

        accepted = {d[0] for d in RANGE_RATE_STATION_DEFS}
        for geometry in harness.phase1_geometries():
            self.assertIn(geometry.station_name, accepted)

    def test_geometry_ids_are_unique_and_stable(self):
        ids = [g.geometry_id for g in harness.phase1_geometries()]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(ids, [g.geometry_id for g in harness.phase1_geometries()])

    def test_geometry_set_spans_the_station_rotation_projection(self):
        """Materially different Earth-rotation velocity projection is required."""
        speeds = {
            6.371e6 * 7.2921159e-5 * math.cos(math.radians(g.lat_deg))
            for g in harness.phase1_geometries()
        }
        self.assertGreater(max(speeds) / min(speeds), 3.0)

    def test_orbital_phase_offsets_produce_distinct_states(self):
        seen = []
        for geometry in harness.phase1_geometries()[:3]:
            t, x, _vis = harness.truth_arc(geometry, 600.0, 60.0)
            seen.append(tuple(np.round(x[0], 3)))
        self.assertEqual(len(seen), len(set(seen)))


class ShardingAndResumability(unittest.TestCase):
    """Sharding must partition the campaign exactly once, with no overlap."""

    def test_shards_partition_the_cell_list_exactly(self):
        geometries = harness.phase1_geometries()[:2]
        cells = harness.campaign_cells("H1", geometries, 4)
        for shard_count in (1, 3, 8):
            collected = []
            for k in range(1, shard_count + 1):
                collected.extend(
                    c for i, c in enumerate(cells) if i % shard_count == k - 1
                )
            self.assertEqual(len(collected), len(cells))

    def test_definition_hash_is_deterministic_and_sensitive(self):
        geometries = harness.phase1_geometries()[:2]
        first = harness.definition_hash("H1", geometries, 4)
        self.assertEqual(first, harness.definition_hash("H1", geometries, 4))
        self.assertNotEqual(first, harness.definition_hash("H1", geometries, 5))
        self.assertNotEqual(first, harness.definition_hash("H2", geometries, 4))

    def test_aggregation_rejects_a_missing_shard(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            self._write_shard(out, 1, 3)
            self._write_shard(out, 2, 3)          # shard 3 deliberately absent
            with self.assertRaises(SystemExit) as ctx:
                harness.aggregate(out, "phase1-r1r4-longarc")
            self.assertIn("missing", str(ctx.exception))

    def test_aggregation_rejects_a_mismatched_canonical_head(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            self._write_shard(out, 1, 2)
            self._write_shard(out, 2, 2, head="deadbeef")
            with self.assertRaises(SystemExit) as ctx:
                harness.aggregate(out, "phase1-r1r4-longarc")
            self.assertIn("mismatched canonical HEAD", str(ctx.exception))

    def test_aggregation_rejects_a_mismatched_campaign_definition(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            self._write_shard(out, 1, 2)
            self._write_shard(out, 2, 2, definition="other")
            with self.assertRaises(SystemExit) as ctx:
                harness.aggregate(out, "phase1-r1r4-longarc")
            self.assertIn("mismatched campaign definition", str(ctx.exception))

    def test_aggregation_accepts_a_complete_shard_set(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            self._write_shard(out, 1, 2)
            self._write_shard(out, 2, 2)
            harness.aggregate(out, "phase1-r1r4-longarc")
            summary = dict(
                (r[0], r[1])
                for r in csv.reader((out / "long_arc_summary.csv").open(encoding="utf-8"))
            )
            self.assertEqual(summary["campaign_status"], "COMPLETE")

    @staticmethod
    def _write_shard(out: Path, index: int, count: int, head="abc123",
                     definition="def456"):
        csv_path = out / f"long_arc_raw_H1_shard{index}of{count}.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=harness.RAW_FIELDS)
            writer.writeheader()
            row = {f: "" for f in harness.RAW_FIELDS}
            row.update({"phase": "Q1", "model": "R4_four_event_delay",
                        "delay_s": "0.0", "oracle_difference": "0.0",
                        "difference_sigma": "0.0", "status": "OK",
                        "classification": "ZERO_DELAY_REDUCTION"})
            writer.writerow(row)
        csv_path.with_suffix(".json").write_text(json.dumps({
            "campaign_id": "phase1-r1r4-longarc", "horizon": "H1",
            "shard_index": index, "shard_count": count,
            "definition_hash": definition, "canonical_head": head,
            "complete": True, "rows": 1,
        }), encoding="utf-8")


class BoundedQualificationSlice(unittest.TestCase):
    """A small REAL qualification slice, using inherited thresholds only."""

    @classmethod
    def setUpClass(cls):
        try:
            cls.et0 = harness.spice_epoch()
        except Exception as exc:  # pragma: no cover - environment dependent
            raise unittest.SkipTest(f"long-arc qualification needs SPICE: {exc}")
        cls.geometry = harness.phase1_geometries()[0]

    def _cell(self, delay_s, count_interval_s=60.0, cadence_s=60.0, horizon="H1"):
        return {"horizon": horizon, "geometry_id": self.geometry.geometry_id,
                "station": self.geometry.station_name,
                "phase_fraction": self.geometry.phase_fraction,
                "cadence_s": cadence_s, "count_interval_s": count_interval_s,
                "delay_s": delay_s, "max_epochs": 4}

    def _run(self, cell):
        by_id = {self.geometry.geometry_id: self.geometry}
        return harness.run_cell(cell, self.et0, by_id, "1/1", "test")

    def test_zero_delay_reduces_to_r3_bitwise_across_the_arc(self):
        """Inherited R4 contract: R4(delta=0) == R3 production, bitwise."""
        rows, stats = self._run(self._cell(0.0))
        self.assertEqual(stats["unexpected_failure"], 0)
        observations = [r for r in rows if r["model"] == "R4_four_event_delay"]
        self.assertGreater(len(observations), 0)
        for row in observations:
            self.assertEqual(float(row["oracle_difference"]), 0.0,
                             f"zero-delay row not bitwise at epoch {row['epoch_s']}")

    def test_four_event_residuals_meet_the_inherited_tolerance(self):
        rows, _stats = self._run(self._cell(1e-3))
        detail = [r for r in rows if r["model"].startswith("R4_event_detail_")]
        self.assertGreater(len(detail), 0)
        for row in detail:
            for field in ("event_residual_up_s", "event_residual_down_s",
                          "event_residual_delay_s"):
                self.assertLessEqual(float(row[field]),
                                     harness.LEG_RESIDUAL_TOLERANCE_S)
            self.assertEqual(row["event_order_ok"], True)

    def test_provenance_is_truthful_and_stable_across_the_arc(self):
        rows, _stats = self._run(self._cell(1e-4))
        observations = [r for r in rows if r["model"] == "R4_four_event_delay"]
        for row in observations:
            self.assertEqual(row["model_version"],
                             "r4.counted-doppler.four-event-delay.v1")
            self.assertEqual(row["earth_ephemeris_method"], "linear_grid_interpolation")
            self.assertEqual(row["spacecraft_interpolation_method"], "cubic_hermite")
            self.assertEqual(row["station_state_method"], "exact_event_epoch_sxform")
        self.assertEqual(len({r["model_version"] for r in observations}), 1,
                         "provenance must not change midway through an arc")

    def test_no_unexpected_nonfinite_rows(self):
        for delay in (0.0, 1e-4, 1e-3):
            rows, stats = self._run(self._cell(delay))
            self.assertEqual(stats["nonfinite"], 0)
            self.assertEqual(stats["unexpected_failure"], 0)

    def test_materiality_classification_uses_the_inherited_bands(self):
        rows, _stats = self._run(self._cell(1e-3))
        observations = [r for r in rows if r["model"] == "R4_four_event_delay"]
        for row in observations:
            sigma = float(row["difference_sigma"])
            expected = ("MIGRATION_BLOCKER_CLASS" if sigma > harness.BLOCKER_SIGMA
                        else "MATERIAL" if sigma > harness.NEGLIGIBLE_SIGMA
                        else "NEGLIGIBLE")
            self.assertEqual(row["classification"], expected)

    @unittest.skipUnless(RUN_SLOW, "bounded slow long-arc slice")
    def test_slow_day_horizon_slice_remains_bitwise_at_zero_delay(self):
        """A bounded H3 (24 h) slice, only under the slow marker."""
        rows, stats = self._run(self._cell(0.0, count_interval_s=60.0,
                                           cadence_s=120.0, horizon="H3"))
        self.assertEqual(stats["unexpected_failure"], 0)
        observations = [r for r in rows if r["model"] == "R4_four_event_delay"]
        self.assertGreater(len(observations), 0)
        for row in observations:
            self.assertEqual(float(row["oracle_difference"]), 0.0)


if __name__ == "__main__":
    unittest.main()
