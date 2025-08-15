"""
Scene change detection for optimized video processing
"""
import logging
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class SceneChangeDetector:
    """
    Detects scene changes in video frames to optimize OCR processing
    """

    def __init__(self, threshold: float = 0.3):
        """
        Initialize scene change detector

        Args:
            threshold: Scene change detection threshold (0.0 to 1.0)
                      Lower values detect more scene changes
        """
        self.threshold = threshold
        self.last_frame: Optional[np.ndarray] = None
        self.last_histogram: Optional[np.ndarray] = None
        self.frame_count = 0
        self.scene_count = 0

    def is_scene_changed(self, frame: np.ndarray) -> bool:
        """
        Detect if current frame represents a scene change

        Args:
            frame: Current video frame as numpy array

        Returns:
            True if scene has changed, False otherwise
        """
        self.frame_count += 1

        # First frame is always considered a scene change
        if self.last_frame is None:
            self.last_frame = frame.copy()
            self.last_histogram = self._calculate_histogram(frame)
            self.scene_count += 1
            return True

        # Calculate histogram difference
        current_hist = self._calculate_histogram(frame)
        difference = self._calculate_histogram_difference(
            self.last_histogram, current_hist
        )

        # Apply adaptive threshold based on frame content
        adaptive_threshold = self._calculate_adaptive_threshold(frame)

        # Update last frame if significant change
        if difference > adaptive_threshold:
            self.last_frame = frame.copy()
            self.last_histogram = current_hist
            self.scene_count += 1
            logger.debug(f"Scene change detected at frame {self.frame_count} "
                        f"(difference: {difference:.3f}, threshold: {adaptive_threshold:.3f})")
            return True

        return False

    def _calculate_histogram(self, frame: np.ndarray) -> np.ndarray:
        """
        Calculate color histogram for frame comparison

        Args:
            frame: Input frame as numpy array

        Returns:
            Normalized histogram as numpy array
        """
        # Convert to HSV for better color representation
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Calculate 2D histogram for Hue and Saturation
        hist = cv2.calcHist(
            [hsv_frame],
            [0, 1],  # Hue and Saturation channels
            None,
            [30, 32],  # Bins for H and S
            [0, 180, 0, 256]  # Ranges for H and S
        )

        # Normalize histogram
        return cv2.normalize(hist, hist).flatten()

    def _calculate_histogram_difference(
        self,
        hist1: np.ndarray,
        hist2: np.ndarray
    ) -> float:
        """
        Calculate histogram difference using correlation method

        Args:
            hist1: First histogram
            hist2: Second histogram

        Returns:
            Difference score (0.0 to 1.0, where 1.0 is identical)
        """
        correlation = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
        return 1.0 - correlation  # Convert to difference metric

    def _calculate_adaptive_threshold(self, frame: np.ndarray) -> float:
        """
        Calculate adaptive threshold based on frame characteristics

        Args:
            frame: Current frame

        Returns:
            Adaptive threshold value
        """
        # Calculate frame brightness
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        brightness = np.mean(gray_frame) / 255.0

        # Calculate frame contrast
        contrast = np.std(gray_frame) / 255.0

        # Adjust threshold based on frame characteristics
        # Dark frames or low contrast frames need lower thresholds
        brightness_factor = 1.0 - (brightness * 0.3)
        contrast_factor = 1.0 - (contrast * 0.2)

        adaptive_threshold = self.threshold * brightness_factor * contrast_factor

        # Ensure threshold stays within reasonable bounds
        return max(0.1, min(0.8, adaptive_threshold))

    def reset(self):
        """Reset detector state"""
        self.last_frame = None
        self.last_histogram = None
        self.frame_count = 0
        self.scene_count = 0

    def get_stats(self) -> dict:
        """Get detector statistics"""
        return {
            'frames_processed': self.frame_count,
            'scenes_detected': self.scene_count,
            'scene_ratio': self.scene_count / max(1, self.frame_count),
            'current_threshold': self.threshold
        }
