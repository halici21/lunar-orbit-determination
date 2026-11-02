"""Run page controller — monitors the job started from Senaryo (or the tutorial).

The page launches nothing on its own: a job arrives through ``preset_script``
with its persisted job file, and the page shows its status, log and result.
"""
from __future__ import annotations
import sys
import json
from pathlib import Path

from PyQt5.QtWidgets import QWidget
from PyQt5.QtCore import Qt
from PyQt5 import uic

from services.project_paths import DESKTOP_APP_DIR, PYTHON_PORT
from workers.process_worker import ProcessWorker
from widgets.log_console import LogConsole
from styles.theme import C
from styles.artifact_workspace import ARTIFACT_STYLE

UI_PATH = DESKTOP_APP_DIR / "ui" / "pages" / "run_monitor_page.ui"


class RunMonitorController(QWidget):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        uic.loadUi(str(UI_PATH), self)
        self._worker = ProcessWorker(self)
        self._last_exit_code: int | None = None
        self._command: tuple[Path, list[str]] | None = None
        self._job_context = None
        self._result_path = None
        self._pending_result = None
        self._cancelled = False
        self._setup()
        self.setObjectName("artifactWorkspace")
        self.setAttribute(Qt.WA_StyledBackground, True)
        self.setStyleSheet(ARTIFACT_STYLE)
        self._set_status("Hazır", C.TEXT_MUTED)

    def _setup(self) -> None:
        self._log = LogConsole(self)
        self._log.setPlaceholderText("Henüz çalışan bir koşu yok. Senaryo sayfasından başlatın.")
        placeholder = self.consolePlaceholder
        parent_layout = placeholder.parent().layout()
        parent_layout.insertWidget(parent_layout.indexOf(placeholder), self._log, stretch=1)
        placeholder.hide()
        self.commandPreviewLabel.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.commandPreviewLabel.setProperty("role", "caption")
        self.elapsedLabel.setProperty("role", "caption")

        self.cancelBtn.clicked.connect(self._cancel)
        self.clearLogBtn.clicked.connect(self._log.clear_log)
        self.openOutputBtn.clicked.connect(self._open_output)
        self.openConsoleBtn.setProperty("primary", True)
        self.openConsoleBtn.clicked.connect(self._open_console)

        self._worker.output_line.connect(self._log.append_stdout)
        self._worker.output_line.connect(self._capture_result)
        self._worker.error_line.connect(self._log.append_stderr)
        self._worker.started.connect(self._on_started)
        self._worker.finished.connect(self._on_finished)
        self._worker.elapsed_tick.connect(self._on_tick)

    def on_shown(self) -> None:
        pass  # nothing to refresh on show

    # ------------------------------------------------------------------
    def _update_command_preview(self) -> None:
        if self._command is None:
            self.commandPreviewLabel.clear()
            return
        script, extra_args = self._command
        quoted_args = " ".join(_quote_arg(arg) for arg in extra_args)
        self.commandPreviewLabel.setText(f"python {script.name}" + (f" {quoted_args}" if quoted_args else ""))
        self.commandPreviewLabel.setToolTip(f"python {script} {quoted_args}".strip())

    def _start(self) -> None:
        if self._worker.is_running or self._command is None:
            return
        self._log.clear_log()
        self._result_path = None
        self._pending_result = None
        self._cancelled = False
        self.openConsoleBtn.setEnabled(False)
        script, extra_args = self._command
        args = [str(script), *extra_args]
        self._log.append_info(f"Başlatılıyor: {sys.executable} {' '.join(args)}")
        self._worker.run(sys.executable, ["-u", *args], working_dir=str(PYTHON_PORT if self._job_context else PYTHON_PORT.parent),
                         env={"PYTHONIOENCODING": "utf-8", "PYTHONUNBUFFERED": "1"})

    def _cancel(self) -> None:
        if not self._worker.is_running:
            return
        self._cancelled = True
        self._log.append_warning("Koşu iptal ediliyor; kısmi dosyalar tamamlanmış sonuç olarak açılmayacak.")
        self._worker.cancel()

    def _on_started(self) -> None:
        self.cancelBtn.setEnabled(True)
        self.openOutputBtn.setEnabled(False)
        self.progressBar.setVisible(True)
        self._set_status("Çalışıyor…", C.YELLOW)

    def _on_finished(self, exit_code: int) -> None:
        self._last_exit_code = exit_code
        self.cancelBtn.setEnabled(False)
        self.openOutputBtn.setEnabled(True)
        self.progressBar.setVisible(False)
        if self._cancelled:
            self._set_status("İptal edildi — kısmi dosyalar korunuyor", C.YELLOW)
        elif exit_code == 0:
            self._log.append_success("İşlem başarıyla bitti (çıkış kodu 0)")
            self._set_status("Tamamlandı · çözüm durumu Sonuçlar sayfasında", C.GREEN)
            if self._pending_result is not None:
                self._result_path = self._pending_result
                self.openConsoleBtn.setEnabled(True)
                if self.autoConsoleCheck.isChecked():
                    self._open_console()
            elif self._job_context:
                self._set_status("İşlem bitti, tamamlanmış sonuç paketi alınamadı", C.YELLOW)
        else:
            self._log.append_stderr(f"İşlem çıkış kodu {exit_code} ile bitti")
            self._set_status(f"Hata (çıkış kodu {exit_code})", C.RED)

    def _on_tick(self, elapsed: float) -> None:
        mins, secs = divmod(int(elapsed), 60)
        self.elapsedLabel.setText(f"Geçen süre: {mins:02d}:{secs:02d}")

    def _set_status(self, text: str, color: str) -> None:
        self.statusLabel.setText(text)
        self.statusLabel.setStyleSheet(f"QLabel {{ color: {color}; }}")
        self.statusDot.setStyleSheet(f"QLabel {{ color: {color}; font-size: 16px; }}")

    def _open_output(self) -> None:
        try:
            import subprocess as sp
            from services.project_paths import RESULTS_DIR
            folder = self._job_context["folder"] if self._job_context else RESULTS_DIR
            sp.Popen(["explorer", str(folder)])
        except Exception:
            pass

    # ------------------------------------------------------------------
    def preset_script(
        self,
        script_path: str,
        extra_args: list[str] | None = None,
        auto_start: bool = False,
        job_context: dict | None = None,
    ) -> None:
        """Load the job to monitor: its script, arguments and run identity."""
        if self._worker.is_running:
            self._log.append_warning("Çalışan koşu tamamlanmadan yeni koşu yüklenemez.")
            return
        self._job_context = job_context
        self._result_path = None
        self._pending_result = None
        self.openConsoleBtn.setEnabled(False)
        target = Path(script_path)
        if not target.is_absolute():
            target = PYTHON_PORT / target
        self._command = (target, list(extra_args or []))
        if job_context:
            self.jobLabel.setText(f"Koşu {str(job_context.get('run_id', ''))[:12]} · {job_context.get('folder', '')}")
        else:
            self.jobLabel.setText(f"Betik · {target.name}")
        self._update_command_preview()
        if auto_start:
            self._start()

    def _capture_result(self, line):
        if not self._job_context or not line.startswith("LUNAR_OD_RESULT "):
            return
        try:
            info = json.loads(line.removeprefix("LUNAR_OD_RESULT "))
            if not isinstance(info, dict):
                raise ValueError("Sonuç bildirimi nesne olmalı.")
            path = Path(info["result_path"]).resolve()
            folder = Path(self._job_context["folder"]).resolve()
            completion = json.loads((folder / "completed.json").read_text(encoding="utf-8"))
            if (info.get("schema") != "lunar-od-completed-job-v1" or info.get("run_id") != self._job_context["run_id"]
                    or path.parent != folder or not path.name.endswith(("_arc_solutions.csv", "_ukf_updates.csv"))
                    or not path.is_file() or completion != info):
                raise ValueError("Koşu kimliği veya tamamlanma kaydı uyuşmuyor.")
            self._pending_result = path
        except (ValueError, TypeError, KeyError, OSError) as exc:
            self._log.append_warning(f"Sonuç bildirimi geçersiz: {exc}")

    def _open_console(self):
        if self._result_path is not None and hasattr(self.window(), "open_in_result_console"):
            self.window().open_in_result_console(self._result_path)

    def shutdown(self):
        if self._worker.is_running:
            self._cancelled = True
            self._worker.cancel()
            self._worker._proc.waitForFinished(2000)


def _quote_arg(arg: str) -> str:
    text = str(arg)
    if not text or any(ch.isspace() for ch in text):
        return f'"{text}"'
    return text
