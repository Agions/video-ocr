"""
VisionSub - Professional Video OCR Subtitle Extraction Tool

A modern, feature-rich video OCR application with advanced text extraction,
real-time preview, and multiple export formats.
"""

__version__ = "1.0.0"
__author__ = "Gemini AI"
__description__ = "Professional Video OCR Subtitle Extraction Tool"

# Core imports
from .core.engine import ProcessingEngine
from .models.config import OcrConfig, ProcessingConfig
from .models.subtitle import SubtitleItem
from .models.video import VideoMetadata

# Export main classes
__all__ = [
    "ProcessingEngine",
    "ProcessingConfig",
    "OcrConfig",
    "SubtitleItem",
    "VideoMetadata",
    "__version__",
    "__author__",
    "__description"
]
