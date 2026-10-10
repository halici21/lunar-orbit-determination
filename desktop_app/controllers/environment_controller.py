"""Own the asynchronous environment report and the desktop kernel preference."""
from __future__ import annotations
from datetime import datetime
import json
import os
from pathlib import Path
import sys

from PyQt5.QtCore import QObject, QTimer, pyqtSignal

from services.environment_readiness import KERNEL_KEY, apply_kernel_preference
from services.project_paths import DESKTOP_APP_DIR, PYTHON_PORT
from workers.process_worker import ProcessWorker


class EnvironmentController(QObject):
    changed = pyqtSignal()

    def __init__(self, settings, parent=None):
        super().__init__(parent)
        self.settings = settings
        self._inherited_kernel_dir = os.environ.get("LUNAR_OD_KERNEL_DIR")
        self.selected_path = apply_kernel_preference(settings, self._inherited_kernel_dir)
        self.report = None
        self._closed = False
        self.error = ""
        self._pending_report = None
        self._callback = None
        self._stderr = []
        self.worker = ProcessWorker(self)
        self.worker.output_line.connect(self._on_line)
        self.worker.error_line.connect(self._stderr.append)
        self.worker.finished.connect(self._on_finished)
        self._deadline = QTimer(self)
        self._deadline.setSingleShot(True)
        self._deadline.timeout.connect(self._timeout)

    @property
    def busy(self):
        return self.worker.is_running

    def select_directory(self, path):
        if self.busy or self._closed:
            return False
        selected = str(Path(path).expanduser().resolve()) if path else ""
        self.settings.setValue(KERNEL_KEY, selected)
        self.settings.sync()
        self.selected_path = apply_kernel_preference(self.settings, self._inherited_kernel_dir)
        self.report = None
        self.refresh()
        return True

    def refresh(self, *, model=None, callback=None):
        if self.busy or self._closed:
            return False
        self.report = None
        self.error = ""
        self._pending_report = None
        self._callback = callback
        self._stderr = []
        args = ["-u", str(DESKTOP_APP_DIR / "environment_probe.py")]
        if self.selected_path:
            args += ["--kernel-dir", self.selected_path]
        if model is not None:
            args += ["--duration-h", str(model.duration_days * 24), "--state-step-s", str(model.state_step_s),
                     "--ephemeris-step-s", str(model.ephemeris_step_s), "--output-dir=" + model.output_dir]
        self.worker.run(sys.executable, args, working_dir=str(PYTHON_PORT),
                        env={"PYTHONIOENCODING": "utf-8", "PYTHONUNBUFFERED": "1"})
        self._deadline.start(60000)
        self.changed.emit()
        return True

    def _on_line(self, line):
        if line.startswith("LUNAR_OD_ENVIRONMENT "):
            try:
                report = json.loads(line.removeprefix("LUNAR_OD_ENVIRONMENT "))
                if (not isinstance(report, dict) or report.get("schema") != "lunar-od-environment-v1"
                        or type(report.get("ready")) is not bool
                        or not isinstance(report.get("checks"), list) or not isinstance(report.get("kernels"), list)
                        or not isinstance(report.get("checked_at"), str)):
                    raise ValueError("Ortam raporu sözleşmesi geçersiz.")
                datetime.fromisoformat(report["checked_at"])
                rows = report["checks"] + report["kernels"]
                if any(not isinstance(row, dict) or not isinstance(row.get("name"), str)
                       or not isinstance(row.get("detail"), str)
                       or row.get("status") not in {"ok", "warning", "error", "pending"} for row in rows):
                    raise ValueError("Ortam raporu satırları geçersiz.")
                spice_ok = any(row["name"] == "SPICE yükleme / örnek epoklar" and row["status"] == "ok"
                               for row in report["checks"])
                if report["ready"] and (not spice_ok or any(row["status"] == "error" for row in rows)):
                    raise ValueError("Ortam raporunda hazır durumu ve kontrol sonuçları uyuşmuyor.")
                self._pending_report = report
            except (ValueError, TypeError) as exc:
                self.error = str(exc)

    def _on_finished(self, code):
        self._deadline.stop()
        if code == 0 and self._pending_report is not None and not self.error:
            self.report = self._pending_report
        else:
            self.error = self.error or "\n".join(self._stderr[-8:]) or "Ortam kontrolü tamamlanamadı."
        callback, self._callback = self._callback, None
        self.changed.emit()
        if callback:
            callback(self.report is not None and self.report["ready"])

    def _timeout(self):
        self.error = "Ortam kontrolü 60 saniyede tamamlanamadı. Yeniden deneyin."
        self.worker.cancel()

    def shutdown(self):
        self._closed = True
        self._callback = None
        self._deadline.stop()
        if self.busy:
            self.worker.cancel()
            self.worker._proc.waitForFinished(2000)
