"""
Real-time preview and editing capabilities for VisionSub
"""
import asyncio
import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
from PyQt6.QtCore import QObject, QRect, QTimer, pyqtSignal

from ..features.advanced_ocr import AdvancedOCRProcessor, EnhancedOCRResult

logger = logging.getLogger(__name__)


class PreviewMode(Enum):
    """Preview modes"""
    DISABLED = "disabled"
    LIVE_OCR = "live_ocr"
    EDITING = "editing"
    REVIEW = "review"


@dataclass
class PreviewConfig:
    """Preview configuration"""
    enabled: bool = True
    mode: PreviewMode = PreviewMode.LIVE_OCR
    auto_refresh: bool = True
    refresh_interval: float = 1.0  # seconds
    show_confidence: bool = True
    show_bounding_boxes: bool = True
    highlight_low_confidence: bool = True
    confidence_threshold: float = 0.7


class OverlayManager:
    """Manages overlay rendering on video frames"""

    def __init__(self):
        self.overlay_config = {
            'text_color': (255, 255, 0),  # Yellow
            'box_color': (0, 255, 0),     # Green
            'low_confidence_color': (255, 0, 0),  # Red
            'font_scale': 0.7,
            'thickness': 2,
            'alpha': 0.8
        }

    def create_overlay_frame(
        self,
        frame: np.ndarray,
        ocr_result: EnhancedOCRResult,
        config: PreviewConfig
    ) -> np.ndarray:
        """
        Create overlay frame with OCR results

        Args:
            frame: Original video frame
            ocr_result: OCR result to overlay
            config: Preview configuration

        Returns:
            Frame with overlay
        """
        overlay = frame.copy()

        if not ocr_result.text and not config.show_bounding_boxes:
            return overlay

        # Draw bounding boxes if available
        if config.show_bounding_boxes and ocr_result.bounding_boxes:
            for i, box in enumerate(ocr_result.bounding_boxes):
                x, y, w, h = box

                # Choose color based on confidence
                if config.highlight_low_confidence and ocr_result.confidence < config.confidence_threshold:
                    color = self.overlay_config['low_confidence_color']
                else:
                    color = self.overlay_config['box_color']

                # Draw rectangle
                cv2.rectangle(overlay, (x, y), (x + w, y + h), color, self.overlay_config['thickness'])

        # Draw text overlay
        if ocr_result.text:
            self._draw_text_overlay(overlay, ocr_result, config)

        # Draw confidence indicator
        if config.show_confidence:
            self._draw_confidence_indicator(overlay, ocr_result)

        return overlay

    def _draw_text_overlay(
        self,
        frame: np.ndarray,
        ocr_result: EnhancedOCRResult,
        config: PreviewConfig
    ):
        """Draw text overlay on frame"""
        if not ocr_result.text:
            return

        # Choose text color
        if config.highlight_low_confidence and ocr_result.confidence < config.confidence_threshold:
            color = self.overlay_config['low_confidence_color']
        else:
            color = self.overlay_config['text_color']

        # Calculate text position (bottom-left corner)
        height, width = frame.shape[:2]
        text = ocr_result.enhanced_text or ocr_result.text

        # Split text into lines if too long
        max_width = width - 20
        lines = self._wrap_text(text, max_width)

        # Draw text background
        line_height = int(30 * self.overlay_config['font_scale'])
        total_height = len(lines) * line_height + 20
        bg_y = height - total_height - 10

        # Semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, bg_y), (width - 10, height - 10), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

        # Draw text lines
        for i, line in enumerate(lines):
            y_pos = bg_y + 20 + i * line_height
            cv2.putText(
                frame,
                line,
                (20, y_pos),
                cv2.FONT_HERSHEY_SIMPLEX,
                self.overlay_config['font_scale'],
                color,
                self.overlay_config['thickness']
            )

    def _draw_confidence_indicator(
        self,
        frame: np.ndarray,
        ocr_result: EnhancedOCRResult
    ):
        """Draw confidence indicator on frame"""
        if ocr_result.confidence <= 0:
            return

        # Position (top-right corner)
        height, width = frame.shape[:2]
        x_pos = width - 150
        y_pos = 30

        # Draw confidence bar background
        bar_width = 100
        bar_height = 10
        cv2.rectangle(frame, (x_pos, y_pos), (x_pos + bar_width, y_pos + bar_height), (255, 255, 255), 1)

        # Draw confidence level
        fill_width = int(bar_width * ocr_result.confidence)
        if ocr_result.confidence >= 0.7:
            color = (0, 255, 0)  # Green
        elif ocr_result.confidence >= 0.5:
            color = (255, 255, 0)  # Yellow
        else:
            color = (255, 0, 0)  # Red

        cv2.rectangle(frame, (x_pos, y_pos), (x_pos + fill_width, y_pos + bar_height), color, -1)

        # Draw confidence text
        confidence_text = f"Conf: {ocr_result.confidence:.2f}"
        cv2.putText(
            frame,
            confidence_text,
            (x_pos, y_pos - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1
        )

    def _wrap_text(self, text: str, max_width: int) -> List[str]:
        """Wrap text to fit within maximum width"""
        words = text.split()
        lines = []
        current_line = []

        for word in words:
            test_line = ' '.join(current_line + [word])
            # Approximate text width (this is a rough estimate)
            text_width = len(test_line) * 10  # Rough estimate

            if text_width <= max_width:
                current_line.append(word)
            else:
                if current_line:
                    lines.append(' '.join(current_line))
                current_line = [word]

        if current_line:
            lines.append(' '.join(current_line))

        return lines


class RealTimePreview(QObject):
    """Real-time preview system for OCR results"""

    # Signals
    preview_updated = pyqtSignal(np.ndarray)  # Updated preview frame
    ocr_result_updated = pyqtSignal(EnhancedOCRResult)  # Updated OCR result
    preview_error = pyqtSignal(str)  # Preview error message
    mode_changed = pyqtSignal(PreviewMode)  # Preview mode changed

    def __init__(
        self,
        ocr_processor: AdvancedOCRProcessor,
        config: PreviewConfig = None
    ):
        super().__init__()
        self.ocr_processor = ocr_processor
        self.config = config or PreviewConfig()
        self.overlay_manager = OverlayManager()

        # State
        self.current_frame: Optional[np.ndarray] = None
        self.current_result: Optional[EnhancedOCRResult] = None
        self.roi_rect: Optional[QRect] = None
        self.is_enabled = False
        self.preview_mode = PreviewMode.DISABLED

        # Timer for auto-refresh
        self.refresh_timer = QTimer()
        self.refresh_timer.timeout.connect(self._refresh_preview)

        # Processing queue
        self.processing_queue = asyncio.Queue()
        self.is_processing = False

    def enable_preview(self, mode: PreviewMode = PreviewMode.LIVE_OCR):
        """Enable real-time preview"""
        self.is_enabled = True
        self.preview_mode = mode
        self.config.mode = mode

        if self.config.auto_refresh:
            self.refresh_timer.start(int(self.config.refresh_interval * 1000))

        self.mode_changed.emit(mode)
        logger.info(f"Real-time preview enabled in {mode.value} mode")

    def disable_preview(self):
        """Disable real-time preview"""
        self.is_enabled = False
        self.preview_mode = PreviewMode.DISABLED
        self.refresh_timer.stop()
        self.mode_changed.emit(PreviewMode.DISABLED)
        logger.info("Real-time preview disabled")

    def update_frame(self, frame: np.ndarray, roi_rect: Optional[QRect] = None):
        """Update current frame for preview"""
        self.current_frame = frame
        self.roi_rect = roi_rect

        if self.is_enabled and self.config.auto_refresh:
            self._refresh_preview()

    async def _refresh_preview(self):
        """Refresh preview with current frame"""
        if not self.is_enabled or self.current_frame is None:
            return

        try:
            # Extract ROI if specified
            if self.roi_rect and self.roi_rect.width() > 0 and self.roi_rect.height() > 0:
                x, y, w, h = self.roi_rect.x(), self.roi_rect.y(), self.roi_rect.width(), self.roi_rect.height()
                roi_frame = self.current_frame[y:y+h, x:x+w]
            else:
                roi_frame = self.current_frame

            # Process OCR
            result = await self.ocr_processor.process_with_enhancement(
                roi_frame,
                0.0,  # timestamp
                0     # frame_number
            )

            self.current_result = result

            # Create overlay
            overlay_frame = self.overlay_manager.create_overlay_frame(
                self.current_frame,
                result,
                self.config
            )

            # Emit signals
            self.preview_updated.emit(overlay_frame)
            self.ocr_result_updated.emit(result)

        except Exception as e:
            error_msg = f"Preview refresh failed: {e}"
            logger.error(error_msg)
            self.preview_error.emit(error_msg)

    def update_config(self, config: PreviewConfig):
        """Update preview configuration"""
        self.config = config

        # Update timer if auto-refresh setting changed
        if self.config.auto_refresh:
            self.refresh_timer.setInterval(int(self.config.refresh_interval * 1000))
            if not self.refresh_timer.isActive() and self.is_enabled:
                self.refresh_timer.start()
        else:
            self.refresh_timer.stop()

    def get_current_result(self) -> Optional[EnhancedOCRResult]:
        """Get current OCR result"""
        return self.current_result

    def is_preview_enabled(self) -> bool:
        """Check if preview is enabled"""
        return self.is_enabled

    def get_current_mode(self) -> PreviewMode:
        """Get current preview mode"""
        return self.preview_mode


class SubtitleEditor:
    """Subtitle editing functionality"""

    def __init__(self):
        self.edit_history = []
        self.current_index = -1
        self.max_history = 50

    def edit_subtitle_text(
        self,
        original_text: str,
        new_text: str,
        subtitle_index: int
    ) -> Dict[str, Any]:
        """
        Edit subtitle text with history tracking

        Args:
            original_text: Original subtitle text
            new_text: New subtitle text
            subtitle_index: Index of subtitle being edited

        Returns:
            Edit result dictionary
        """
        edit_result = {
            'index': subtitle_index,
            'original_text': original_text,
            'new_text': new_text,
            'timestamp': asyncio.get_event_loop().time(),
            'edit_type': 'text_edit'
        }

        self._add_to_history(edit_result)
        return edit_result

    def edit_subtitle_timing(
        self,
        original_start: str,
        original_end: str,
        new_start: str,
        new_end: str,
        subtitle_index: int
    ) -> Dict[str, Any]:
        """
        Edit subtitle timing with history tracking

        Args:
            original_start: Original start time
            original_end: Original end time
            new_start: New start time
            new_end: New end time
            subtitle_index: Index of subtitle being edited

        Returns:
            Edit result dictionary
        """
        edit_result = {
            'index': subtitle_index,
            'original_start': original_start,
            'original_end': original_end,
            'new_start': new_start,
            'new_end': new_end,
            'timestamp': asyncio.get_event_loop().time(),
            'edit_type': 'timing_edit'
        }

        self._add_to_history(edit_result)
        return edit_result

    def _add_to_history(self, edit_result: Dict[str, Any]):
        """Add edit to history"""
        # Remove any edits after current index (for redo functionality)
        self.edit_history = self.edit_history[:self.current_index + 1]

        # Add new edit
        self.edit_history.append(edit_result)
        self.current_index += 1

        # Limit history size
        if len(self.edit_history) > self.max_history:
            self.edit_history.pop(0)
            self.current_index -= 1

    def undo(self) -> Optional[Dict[str, Any]]:
        """Undo last edit"""
        if self.current_index >= 0:
            edit = self.edit_history[self.current_index]
            self.current_index -= 1
            return edit
        return None

    def redo(self) -> Optional[Dict[str, Any]]:
        """Redo next edit"""
        if self.current_index < len(self.edit_history) - 1:
            self.current_index += 1
            return self.edit_history[self.current_index]
        return None

    def can_undo(self) -> bool:
        """Check if undo is available"""
        return self.current_index >= 0

    def can_redo(self) -> bool:
        """Check if redo is available"""
        return self.current_index < len(self.edit_history) - 1

    def clear_history(self):
        """Clear edit history"""
        self.edit_history.clear()
        self.current_index = -1

    def get_edit_history(self) -> List[Dict[str, Any]]:
        """Get complete edit history"""
        return self.edit_history.copy()
