"""
Core module for VisionSub processing engine
"""

from .engine import ProcessingEngine
from .errors import (
    ConfigurationError,
    ErrorHandler,
    FileIOError,
    OCRError,
    VideoProcessingError,
    VisionSubError,
    error_handler,
)
from .frame_cache import FrameCache
from .scene_detection import SceneChangeDetector
from .video_processor import UnifiedVideoProcessor

__all__ = [
    "ProcessingEngine",
    "UnifiedVideoProcessor",
    "SceneChangeDetector",
    "FrameCache",
    "VisionSubError",
    "VideoProcessingError",
    "OCRError",
    "FileIOError",
    "ConfigurationError",
    "ErrorHandler",
    "error_handler"
]
