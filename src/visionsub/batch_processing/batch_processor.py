import os
import traceback
from typing import List

from PyQt6.QtCore import QObject, QRunnable, pyqtSignal, pyqtSlot

from visionsub.core.ocr import get_ocr_engine
from visionsub.core.video import VideoReader
from visionsub.models.config import OcrConfig

# We need a subtitle processor, let's create a simple one for now.
# This highlights a need to refactor subtitle processing out of the old core file.
from visionsub.models.subtitle import SubtitleItem


class SubtitleProcessor:
    def process(self, raw_results: List[tuple[float, str]]) -> List[SubtitleItem]:
        # This is a placeholder. A real implementation would merge, clean, and format.
        items = []
        for i, (timestamp, text) in enumerate(raw_results):
            start_h, rem = divmod(timestamp, 3600)
            start_m, start_s = divmod(rem, 60)
            start_ms = (start_s - int(start_s)) * 1000

            end_timestamp = timestamp + 2.0 # Default 2s duration
            end_h, rem = divmod(end_timestamp, 3600)
            end_m, end_s = divmod(rem, 60)
            end_ms = (end_s - int(end_s)) * 1000

            items.append(SubtitleItem(
                index=i + 1,
                start_time=f"{int(start_h):02}:{int(start_m):02}:{int(start_s):02},{int(start_ms):03}",
                end_time=f"{int(end_h):02}:{int(end_m):02}:{int(end_s):02},{int(end_ms):03}",
                content=text
            ))
        return items

    def generate_srt_content(self, subtitles: List[SubtitleItem]) -> str:
        return "".join(item.to_srt_format() + "\n" for item in subtitles)

class BatchJobSignals(QObject):
    """Signals for a single batch processing job."""
    finished = pyqtSignal(str) # file_path
    error = pyqtSignal(str, str) # file_path, error_message
    progress = pyqtSignal(str, int) # file_path, percentage

class BatchJob(QRunnable):
    """A QRunnable that processes a single video file."""
    def __init__(self, file_path: str, config: OcrConfig):
        super().__init__()
        self.file_path = file_path
        self.config = config
        self.signals = BatchJobSignals()

    @pyqtSlot()
    def run(self):
        try:
            ocr_engine = get_ocr_engine(self.config.engine, lang=self.config.language)
            subtitle_processor = SubtitleProcessor()
            raw_results = []

            with VideoReader(self.file_path) as reader:
                total_frames = reader.metadata.frame_count
                frame_interval = int(reader.metadata.fps) # Process 1 frame per second

                for i in range(0, total_frames, frame_interval):
                    reader.cap.set(1, i)
                    success, frame = reader.cap.read()
                    if not success:
                        continue

                    # Extract ROI and run OCR
                    x, y, w, h = self.config.roi_rect
                    if w > 0 and h > 0:
                        roi_frame = frame[y:y+h, x:x+w]
                        text = ocr_engine.recognize(roi_frame)
                        if text:
                            timestamp = i / reader.metadata.fps
                            raw_results.append((timestamp, text))

                    progress_percent = int((i / total_frames) * 100)
                    self.signals.progress.emit(self.file_path, progress_percent)

            # Process and save subtitles
            processed_subs = subtitle_processor.process(raw_results)
            srt_content = subtitle_processor.generate_srt_content(processed_subs)

            output_path = f"{os.path.splitext(self.file_path)[0]}.srt"
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(srt_content)

            self.signals.progress.emit(self.file_path, 100)
            self.signals.finished.emit(self.file_path)

        except Exception:
            error_message = traceback.format_exc()
            self.signals.error.emit(self.file_path, error_message)
