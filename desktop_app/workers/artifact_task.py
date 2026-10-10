"""Cooperative background artifact tasks with stale-response protection."""
from PyQt5.QtCore import QThread, pyqtSignal
from services.tracking_results import LoadCancelled


class ArtifactTask(QThread):
    ready = pyqtSignal(int, object)
    failed = pyqtSignal(int, str)

    def __init__(self, generation, function, args, parent):
        super().__init__(parent)
        self.generation, self.function, self.args = generation, function, args

    def run(self):
        try:
            result = self.function(*self.args, cancel=self.isInterruptionRequested)
            if not self.isInterruptionRequested():
                self.ready.emit(self.generation, result)
        except LoadCancelled:
            pass
        except Exception as exc:
            self.failed.emit(self.generation, str(exc))


class ArtifactJobs:
    """Controller mixin; source operations never execute on the UI thread."""
    def init_jobs(self):
        self._jobs, self._generation, self._closing = set(), 0, False

    def start_job(self, function, args):
        self.cancel_jobs()
        worker = ArtifactTask(self._generation, function, args, self)
        self._jobs.add(worker)
        worker.ready.connect(self._job_ready)
        worker.failed.connect(self._job_failed)
        worker.finished.connect(lambda w=worker: self._retire_job(w))
        worker.start()

    def cancel_jobs(self):
        self._generation += 1
        for worker in self._jobs:
            worker.requestInterruption()

    def _retire_job(self, worker):
        self._jobs.discard(worker)
        worker.deleteLater()

    def shutdown(self):
        self._closing = True
        self.cancel_jobs()
        for worker in tuple(self._jobs):
            worker.wait(3000)

    def closeEvent(self, event):
        self.shutdown()
        super().closeEvent(event)
