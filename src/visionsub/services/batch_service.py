from typing import List

from PyQt6.QtCore import QObject, QThreadPool, pyqtSignal

from visionsub.batch_processing.batch_processor import BatchJob
from visionsub.models.config import OcrConfig


class BatchServiceSignals(QObject):
    """Defines signals for the overall batch processing service."""
    batch_started = pyqtSignal()
    batch_finished = pyqtSignal()
    file_started = pyqtSignal(str) # file_path
    file_finished = pyqtSignal(str) # file_path
    file_error = pyqtSignal(str, str) # file_path, error_message
    total_progress = pyqtSignal(int) # Overall percentage

class BatchService(QObject):
    """
    A service to manage the batch processing of multiple video files.
    """
    def __init__(self):
        super().__init__()
        self.thread_pool = QThreadPool()
        self.thread_pool.setMaxThreadCount(2) # Limit concurrent jobs
        self.signals = BatchServiceSignals()

        self.video_queue: List[str] = []
        self.config: OcrConfig | None = None
        self.completed_files = 0
        self.total_files = 0

    def start(self, video_queue: List[str], config: OcrConfig):
        """Starts the batch processing workflow."""
        if not video_queue:
            return

        self.video_queue = video_queue
        self.config = config
        self.completed_files = 0
        self.total_files = len(video_queue)

        self.signals.batch_started.emit()
        self.signals.total_progress.emit(0)

        self._start_next_job()

    def _start_next_job(self):
        """Pops the next video from the queue and starts a job for it."""
        if not self.video_queue:
            # No more files, batch is done
            self.signals.total_progress.emit(100)
            self.signals.batch_finished.emit()
            return

        file_path = self.video_queue.pop(0)
        self.signals.file_started.emit(file_path)

        job = BatchJob(file_path, self.config)
        job.signals.finished.connect(self._on_job_finished)
        job.signals.error.connect(self._on_job_error)

        self.thread_pool.start(job)

    def _on_job_finished(self, file_path: str):
        """Handles a single job's completion."""
        self.completed_files += 1
        progress = int((self.completed_files / self.total_files) * 100)
        self.signals.total_progress.emit(progress)
        self.signals.file_finished.emit(file_path)

        # Start the next job in the queue
        self._start_next_job()

    def _on_job_error(self, file_path: str, error_message: str):
        """Handles a single job's failure."""
        self.completed_files += 1
        progress = int((self.completed_files / self.total_files) * 100)
        self.signals.total_progress.emit(progress)
        self.signals.file_error.emit(file_path, error_message)

        # Continue with the next job
        self._start_next_job()
