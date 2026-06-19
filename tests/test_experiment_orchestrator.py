import json
import tempfile
import unittest
from pathlib import Path

from examples.run_all_experiments import (
    EXPERIMENTS,
    run_experiments,
    select_experiments,
    write_run_summaries,
)


class ExperimentOrchestratorTests(unittest.TestCase):
    def test_quick_manifest_selects_fast_non_spice_reports(self):
        selected = select_experiments("quick")

        self.assertEqual([item.experiment_id for item in selected], ["synthetic_hot_start", "visibility_fixture"])
        self.assertTrue(all(item.quick for item in selected))
        self.assertTrue(all(not item.requires_spice for item in selected))
        self.assertTrue(all(not item.long_running for item in selected))

    def test_full_manifest_includes_all_experiments(self):
        selected = select_experiments("full")

        self.assertEqual(len(selected), len(EXPERIMENTS))
        self.assertTrue(any(item.requires_spice for item in selected))
        self.assertTrue(any(item.long_running for item in selected))

    def test_only_and_skip_filter_by_id_or_script_stem(self):
        selected = select_experiments(
            "full",
            only=("synthetic_hot_start_report", "formal_handoff"),
            skip=("formal_handoff",),
        )

        self.assertEqual([item.experiment_id for item in selected], ["synthetic_hot_start"])

    def test_dry_run_writes_csv_and_json_summaries(self):
        selected = select_experiments("quick")
        with tempfile.TemporaryDirectory() as tmpdir:
            runs = run_experiments(
                selected,
                mode="quick",
                dry_run=True,
                stop_on_fail=False,
                timeout_s=None,
                python_executable="python",
                project_root=Path.cwd(),
            )
            csv_path, json_path = write_run_summaries(runs, tmpdir)

            self.assertEqual([run.status for run in runs], ["planned", "planned"])
            self.assertTrue(csv_path.is_file())
            self.assertTrue(json_path.is_file())
            csv_text = csv_path.read_text(encoding="utf-8")
            payload = json.loads(json_path.read_text(encoding="utf-8"))

        self.assertIn("experiment_id,script,mode,status", csv_text)
        self.assertEqual(payload[0]["experiment_id"], "synthetic_hot_start")
        self.assertEqual(payload[0]["status"], "planned")


if __name__ == "__main__":
    unittest.main()
