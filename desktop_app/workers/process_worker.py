"""QProcess-based worker for launching Python scripts non-blocking."""
from __future__ import annotations
import sys
import time
from pathlib import Path

from PyQt5.QtCore import QObject, QProcess, QTimer, pyqtSignal


class ProcessWorker(QObject):
    """Launch a subprocess, stream stdout/stderr line-by-line, report elapsed time.

    Usage::

        worker = ProcessWorker()
        worker.output_line.connect(console.append_stdout)
        worker.error_line.connect(console.append_stderr)
        worker.finished.connect(on_done)
        worker.run(sys.executable, ["path/to/script.py"], working_dir="...")
    """

    output_line = pyqtSignal(str)   # a line from stdout
    error_line = pyqtSignal(str)    # a line from stderr
    started = pyqtSignal()          # process started
    finished = pyqtSignal(int)      # exit code
    elapsed_tick = pyqtSignal(float)  # seconds since start (every 500 ms)

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._proc = QProcess(self)
        self._proc.readyReadStandardOutput.connect(self._read_stdout)
        self._proc.readyReadStandardError.connect(self._read_stderr)
        self._proc.started.connect(self._on_started)
        self._proc.finished.connect(self._on_finished)

        self._timer = QTimer(self)
        self._timer.setInterval(500)
        self._timer.timeout.connect(self._on_tick)
        self._start_time: float = 0.0
        self._stdout_buf: str = ""
        self._stderr_buf: str = ""

    # ------------------------------------------------------------------
    def run(
        self,
        program: str,
        args: list[str],
        working_dir: str | None = None,
        env: dict[str, str] | None = None,
    ) -> None:
        """Start the subprocess. No-op if already running."""
        if self._proc.state() != QProcess.NotRunning:
            return
        if working_dir:
            self._proc.setWorkingDirectory(working_dir)
        if env is not None:
            from PyQt5.QtCore import QProcessEnvironment
            qenv = QProcessEnvironment.systemEnvironment()
            for k, v in env.items():
                qenv.insert(k, v)
            self._proc.setProcessEnvironment(qenv)
        self._proc.start(program, args)

    def run_script(self, script_path: str | Path, working_dir: str | None = None) -> None:
        """Convenience: run a Python script with the current interpreter."""
        self.run(sys.executable, [str(script_path)], working_dir=working_dir)

    def cancel(self) -> None:
        """Kill the running process."""
        if self._proc.state() == QProcess.Running:
            self._proc.kill()

    @property
    def is_running(self) -> bool:
        return self._proc.state() == QProcess.Running

    # ------------------------------------------------------------------
    def _on_started(self) -> None:
        self._start_time = time.monotonic()
        self._stdout_buf = ""
        self._stderr_buf = ""
        self._timer.start()
        self.started.emit()

    def _on_finished(self, exit_code: int, _status) -> None:
        self._timer.stop()
        # Flush any remaining buffered output
        self._flush(self._stdout_buf, stdout=True)
        self._stdout_buf = ""
        self._flush(self._stderr_buf, stdout=False)
        self._stderr_buf = ""
        self.finished.emit(exit_code)

    def _on_tick(self) -> None:
        self.elapsed_tick.emit(time.monotonic() - self._start_time)

    def _read_stdout(self) -> None:
        raw = bytes(self._proc.readAllStandardOutput()).decode("utf-8", errors="replace")
        self._stdout_buf += raw
        self._stdout_buf = self._emit_lines(self._stdout_buf, stdout=True)

    def _read_stderr(self) -> None:
        raw = bytes(self._proc.readAllStandardError()).decode("utf-8", errors="replace")
        self._stderr_buf += raw
        self._stderr_buf = self._emit_lines(self._stderr_buf, stdout=False)

    def _emit_lines(self, buf: str, stdout: bool) -> str:
        lines = buf.split("\n")
        for line in lines[:-1]:
            line = line.rstrip("\r")
            if line:
                (self.output_line if stdout else self.error_line).emit(line)
        return lines[-1]  # remainder (incomplete line)

    def _flush(self, buf: str, stdout: bool) -> None:
        line = buf.strip()
        if line:
            (self.output_line if stdout else self.error_line).emit(line)
