"""WP0.4: LRO hold-out days are sealed and every tracking reader enforces it."""
from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "studies" / "phase17" / "scripts"
sys.path.insert(0, str(SCRIPTS))

import lro_data_partition as P  # noqa: E402


def test_partition_days_are_disjoint_and_holdout_is_sealed():
    part = P.load_partition()
    dev, hold, excl = (set(part[k]["days"]) for k in ("development", "holdout", "excluded"))
    assert not (dev & hold) and not (dev & excl) and not (hold & excl)
    assert part["holdout"]["status"] == "sealed"
    assert part["holdout"]["unseal"]["required_value"] is None


@pytest.mark.parametrize("day", ["2009264", "2009265", "2009266", "2009267", "2009268"])
def test_holdout_and_excluded_days_are_refused(day, monkeypatch):
    monkeypatch.setenv("LUNAR_OD_LRO_HOLDOUT_UNSEAL", "anything")
    with pytest.raises(P.HoldoutDataError):
        P.require_readable_day("trk/lro_no_01/" + day)


def test_development_days_are_readable():
    for day in P.load_partition()["development"]["days"]:
        P.require_readable_day("trk/lro_no_01/" + day)


def test_unseal_needs_the_predeclared_value(monkeypatch):
    part = P.load_partition()
    part["holdout"]["unseal"]["required_value"] = "studies/phase17/configs/wp5_4_predeclaration.json"
    with pytest.raises(P.HoldoutDataError):
        P.require_readable_day("2009264", part)
    monkeypatch.setenv("LUNAR_OD_LRO_HOLDOUT_UNSEAL", "studies/phase17/configs/wp5_4_predeclaration.json")
    P.require_readable_day("2009264", part)


@pytest.mark.parametrize("module,reader", [("phase_f1_lro_residuals", "read_day"),
                                           ("phase_f1c_lro_doppler", "read_doppler")])
def test_tracking_readers_refuse_holdout_days(module, reader, tmp_path):
    mod = pytest.importorskip(module)
    (tmp_path / "trk" / "lro_no_01" / "2009265").mkdir(parents=True)
    with pytest.raises(P.HoldoutDataError):
        getattr(mod, reader)(tmp_path, "trk/lro_no_01/2009265")


def test_no_study_script_or_config_names_a_holdout_day():
    holdout = P.load_partition()["holdout"]["days"]
    pattern = re.compile("|".join(rf"{d[:4]}[_/ -]?{d[4:]}" for d in holdout))
    offenders = []
    for path in list(SCRIPTS.glob("*.py")) + list((ROOT / "studies" / "phase17" / "configs").glob("*.json")):
        if path.name in {"lro_data_partition.py", "lro_data_partition.json"}:
            continue
        if pattern.search(path.read_text(encoding="utf-8", errors="replace")):
            offenders.append(path.name)
    assert offenders == []
