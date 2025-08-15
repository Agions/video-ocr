from pathlib import Path
from typing import Generator, Tuple, Union

import cv2
import numpy as np

from visionsub.models.video import VideoMetadata


class VideoReader:
    """
    A class to handle video file reading and frame extraction using OpenCV.
    It provides metadata and an efficient way to iterate through frames.
    """

    def __init__(self, file_path: Union[str, Path]):
        """
        Initializes the VideoReader.

        Args:
            file_path: The absolute path to the video file.

        Raises:
            FileNotFoundError: If the video file does not exist.
            IOError: If the video file cannot be opened.
        """
        self.video_path = Path(file_path)
        if not self.video_path.exists():
            raise FileNotFoundError(f"Video file not found at: {self.video_path}")

        self.cap = cv2.VideoCapture(str(self.video_path))
        if not self.cap.isOpened():
            raise IOError(f"Could not open video file: {self.video_path}")

        self._metadata = self._load_metadata()

    def _load_metadata(self) -> VideoMetadata:
        """Loads metadata from the video file."""
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps if fps > 0 else 0

        return VideoMetadata(
            file_path=str(self.video_path),
            duration=duration,
            fps=fps,
            width=width,
            height=height,
            frame_count=frame_count,
        )

    @property
    def metadata(self) -> VideoMetadata:
        """Returns the cached video metadata."""
        return self._metadata

    def iter_frames(self, interval_seconds: float = 1.0) -> Generator[Tuple[float, np.ndarray], None, None]:
        """
        A generator that yields frames from the video at a specified interval.

        Args:
            interval_seconds: The time interval in seconds between captured frames.

        Yields:
            A tuple containing the timestamp (in seconds) and the frame (as a NumPy array).
        """
        if self.metadata.fps == 0:
            return

        frame_interval = max(1, int(self.metadata.fps * interval_seconds))
        frame_idx = 0
        while self.cap.isOpened():
            # Set the next frame position
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)

            success, frame = self.cap.read()
            if not success:
                break

            timestamp = frame_idx / self.metadata.fps
            yield timestamp, frame

            frame_idx += frame_interval
            if frame_idx >= self.metadata.frame_count:
                break

    @staticmethod
    def extract_subtitle_region(frame: np.ndarray, region: Tuple[float, float, float, float]) -> np.ndarray:
        """
        Extracts a specific region of interest (ROI) from a frame.

        Args:
            frame: The input frame.
            region: A tuple (x_start_ratio, y_start_ratio, width_ratio, height_ratio)
                    defining the subtitle area (e.g., (0.0, 0.8, 1.0, 0.2)).

        Returns:
            The cropped region of the frame.
        """
        height, width, _ = frame.shape
        x_start = int(width * region[0])
        y_start = int(height * region[1])
        roi_width = int(width * region[2])
        roi_height = int(height * region[3])

        return frame[y_start : y_start + roi_height, x_start : x_start + roi_width]


    def release(self) -> None:
        """Releases the video capture object."""
        if self.cap.isOpened():
            self.cap.release()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()
