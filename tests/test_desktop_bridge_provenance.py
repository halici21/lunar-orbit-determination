"""Approved desktop bridge provenance and rejection of later edits."""
from pathlib import Path
import hashlib
import subprocess

from examples import r2_measurement_fidelity_validation as campaign


ROOT = Path(__file__).resolve().parents[1]
BRIDGE = "examples/run_scenario_config.py"
APPROVED_COMMIT = "a9f9e56c2556b98a68493e3425ae42b836f34f8c"
NUMERICAL_CORE_COMMIT = "71e088fabaf793935bcf8a346d91aca8337215f0"


def _filtered_sha256(commit):
    blob = subprocess.check_output(["git", "cat-file", "--filters", commit + ":" + BRIDGE], cwd=ROOT)
    return hashlib.sha256(blob).hexdigest()


def test_refreeze_keeps_the_historical_record_and_protects_approved_bytes():
    assert campaign.R2_HISTORICAL_FILTERED_CHECKOUT_SHA256[BRIDGE] == (
        "f6cf74b69fa15a15148cb6ca59db56980b9afa3340b59fcaf4de8a2a2a533bac"
    )
    assert BRIDGE in campaign.R2_CURRENT_TREE_PROTECTED_PATHS
    assert campaign.DESKTOP_WORKSPACE_REFROZEN_CURRENT_TREE_SHA256[BRIDGE] == _filtered_sha256(APPROVED_COMMIT)
    # The numerical-core correction is the later additive identity of the bridge.
    assert campaign.NUMERICAL_CORE_REFROZEN_CURRENT_TREE_SHA256[BRIDGE] == _filtered_sha256(NUMERICAL_CORE_COMMIT)
    # F1 WP1.9 (UKF initial covariance) is the current one.
    assert campaign.expected_current_tree_sha256(BRIDGE) == campaign.F1_WP1_9_REFROZEN_CURRENT_TREE_SHA256[BRIDGE]
    assert campaign.expected_current_tree_sha256(BRIDGE) == hashlib.sha256((ROOT / BRIDGE).read_bytes()).hexdigest()


def test_a_later_bridge_edit_still_fails_the_provenance_gate(tmp_path):
    protected = set(campaign.R2_CURRENT_TREE_PROTECTED_PATHS)
    protected.update(campaign.CLOSURE_REFROZEN_CURRENT_TREE_SHA256)
    for name in protected:
        target = tmp_path / campaign.current_source_path(name)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((ROOT / campaign.current_source_path(name)).read_bytes())
    assert all(row["pass"] for row in campaign.build_r2_current_tree_protection_rows(tmp_path))
    path = tmp_path / BRIDGE
    path.write_bytes(path.read_bytes() + b"\n# unauthorized later edit\n")
    failed = [row["item"] for row in campaign.build_r2_current_tree_protection_rows(tmp_path)
              if not row["pass"]]
    assert failed == [BRIDGE]
