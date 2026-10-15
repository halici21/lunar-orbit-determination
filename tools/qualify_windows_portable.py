"""Run inside a relocated portable bundle with external Python/PYTHONPATH removed."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main(bundle):
    bundle = bundle.resolve()
    interpreter = bundle / "runtime/python.exe"
    if Path(sys.executable).resolve() != interpreter:
        raise RuntimeError("Qualification must use the bundled interpreter")
    app_root = bundle / "app"
    record = json.loads((bundle / "BUNDLE_MANIFEST.json").read_text(encoding="utf-8"))
    print(f"Checking {len(record['files_sha256'])} bundled file identities...", flush=True)
    def verify_files():
        for name, expected in record["files_sha256"].items():
            if hashlib.sha256((bundle / name).read_bytes()).hexdigest() != expected:
                raise ValueError(f"Bundle content changed: {name}")
    verify_files()
    if record["source"]["working_tree_dirty"]:
        raise ValueError("Dirty source bundle cannot pass release qualification")
    print("Bundle identities verified; importing scientific and Qt packages...", flush=True)
    os.environ.pop("PYTHONPATH", None)
    os.environ.pop("PYTHONHOME", None)
    os.environ["PATH"] = os.environ.get("SystemRoot", "C:/Windows") + "/System32"
    os.environ["QT_QPA_PLATFORM"] = "offscreen"
    os.environ["QT_QPA_FONTDIR"] = os.environ.get("SystemRoot", "C:/Windows") + "/Fonts"
    os.environ.pop("PYTHONDONTWRITEBYTECODE", None)
    sys.dont_write_bytecode = False
    for key in ("LUNAR_OD_KERNEL_DIR", "LUNAR_OD_CAMPAIGN_ROOT", "LUNAR_OD_GRAVITY_ROOT", "LUNAR_OD_PROJECT_ROOT"):
        folder = bundle / "qualification" / key
        folder.mkdir(parents=True, exist_ok=True)
        os.environ[key] = str(folder)
    import numpy
    import scipy
    import numba
    import spiceypy
    import matplotlib
    import pandas
    import h5py
    import pyqtgraph
    import lunar_od
    if not Path(lunar_od.__file__).resolve().is_relative_to(app_root):
        raise ValueError("Scientific library escaped the portable bundle")
    print("Imports passed; checking QProcess and desktop workflow...", flush=True)
    from PyQt5.QtCore import QSettings
    from PyQt5.QtWidgets import QApplication
    settings_dir = bundle / "qualification/settings"
    settings_dir.mkdir(parents=True, exist_ok=True)
    QSettings.setDefaultFormat(QSettings.IniFormat)
    QSettings.setPath(QSettings.IniFormat, QSettings.UserScope, str(settings_dir))
    QSettings.setPath(QSettings.NativeFormat, QSettings.UserScope, str(settings_dir))
    sys.path.insert(0, str(app_root / "desktop_app"))
    from workers.process_worker import ProcessWorker
    qt = QApplication.instance() or QApplication([])

    def wait(predicate, timeout=60):
        deadline = time.monotonic() + timeout
        while not predicate() and time.monotonic() < deadline:
            qt.processEvents()
            time.sleep(.005)
        qt.processEvents()
        if not predicate():
            raise TimeoutError("Packaged Qt workflow timed out")

    worker = ProcessWorker()
    codes, lines = [], []
    worker.finished.connect(codes.append)
    worker.output_line.connect(lines.append)
    worker.error_line.connect(lines.append)
    worker.run(str(interpreter), ["-E", "-c", "import lunar_od,sys,pandas; assert not sys.dont_write_bytecode; print(sys.executable); print(lunar_od.__file__)"])
    wait(lambda: bool(codes))
    if codes[-1] != 0 or not any(str(bundle).lower() in line.lower() for line in lines):
        raise RuntimeError(f"Bundled QProcess imports failed: {lines}")
    codes.clear()
    worker.run(str(interpreter), ["-c", "import time; time.sleep(30)"])
    wait(lambda: worker.is_running)
    worker.cancel()
    wait(lambda: bool(codes))
    if worker.is_running or codes[-1] == 0:
        raise RuntimeError("QProcess cancellation did not stop the child")
    import app as desktop
    isolated = QSettings(str(settings_dir / "window.ini"), QSettings.IniFormat)
    desktop.QSettings = lambda *args, **kwargs: isolated
    window = desktop.MainWindow()
    window.show()
    window.start_tutorial()
    monitor = window._pages["run_monitor"]
    wait(lambda: monitor._result_path is not None and not monitor._worker.is_running)
    result_path = monitor._result_path
    from services.tracking_results import load_tracking_results
    result = load_tracking_results(result_path)
    if len(result.updates) != 124 or result.updates[-1]["position_error_m"] >= 1000:
        raise RuntimeError("Bundled scientific tutorial did not recover the declared case")
    window.close()
    qt.processEvents()
    if window.environment.busy or monitor._worker.is_running:
        raise RuntimeError("Desktop shutdown left a child running")
    # Normal imports may populate caches. Distributed source/binaries must still match.
    generated_caches = list((bundle / "runtime/Lib/site-packages").rglob("*.pyc"))
    if not generated_caches:
        raise RuntimeError("Qualification did not exercise ordinary bytecode cache creation")
    verify_files()
    report = {"schema": "lunar-od-portable-qualification-v1", "passed": True,
        "source_commit": record["source"]["git_commit"], "python": sys.version,
        "interpreter": str(interpreter), "library": str(lunar_od.__file__),
        "verified_files": len(record["files_sha256"]), "external_python_path": os.environ["PATH"],
        "checks": ["bundle hashes", "scientific and Qt imports", "QProcess bundled child",
                   "QProcess cancellation", "desktop launch", "124-update scientific tutorial", "desktop shutdown",
                   "immutable file hashes after normal cache-writing imports and application use"],
        "cache_policy": record["cache_policy"],
        "generated_cache_files_after_use": len(generated_caches),
        "tutorial_final_position_error_m": result.updates[-1]["position_error_m"],
        "kernel_setup": "empty explicit directory; tutorial requires no kernels"}
    path = bundle / "qualification/verification.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    main(parser.parse_args().bundle)
