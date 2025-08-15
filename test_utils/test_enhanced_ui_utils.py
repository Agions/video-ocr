"""
Comprehensive Test Configuration and Utilities for VisionSub UI Testing

This module provides centralized test configuration, fixtures, and utilities
for comprehensive UI testing across all enhanced components.
"""

import pytest
import asyncio
import numpy as np
import cv2
import tempfile
import json
import time
import gc
import psutil
import threading
import logging
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
from contextlib import contextmanager
import sys
import os

# PyQt6 imports
from PyQt6.QtCore import (
    Qt, QTimer, QEvent, QRect, QSize, QPoint, 
    pyqtSignal, QThread, QPropertyAnimation, QEasingCurve
)
from PyQt6.QtGui import (
    QImage, QPixmap, QFont, QColor, QPalette, QPainter,
    QBrush, QPen, QTextCharFormat, QTextCursor, QKeySequence,
    QMouseEvent, QKeyEvent, QWheelEvent, QDragEnterEvent,
    QDropEvent, QCloseEvent
)
from PyQt6.QtWidgets import (
    QApplication, QWidget, QMainWindow, QDialog, QPushButton,
    QLabel, QLineEdit, QTextEdit, QComboBox, QSpinBox, QDoubleSpinBox,
    QCheckBox, QSlider, QProgressBar, QTabWidget, QTableWidget,
    QTableWidgetItem, QHeaderView, QSplitter, QScrollArea,
    QGroupBox, QVBoxLayout, QHBoxLayout, QFormLayout, QMenuBar,
    QMenu, QAction, QToolBar, QStatusBar, QMessageBox,
    QFileDialog, QSystemTrayIcon, QStyle, QStyleOption
)
from PyQt6.QtTest import QTest

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# VisionSub imports
from visionsub.ui.enhanced_main_window import EnhancedMainWindow
from visionsub.ui.enhanced_video_player import EnhancedVideoPlayer
from visionsub.ui.enhanced_ocr_preview import EnhancedOCRPreview
from visionsub.ui.enhanced_subtitle_editor import EnhancedSubtitleEditor
from visionsub.ui.enhanced_settings_dialog import EnhancedSettingsDialog
from visionsub.ui.enhanced_main import EnhancedApplication
from visionsub.ui.theme_system import ThemeManager, get_theme_manager, ThemeColors
from visionsub.models.config import AppConfig, OcrConfig, ProcessingConfig, UIConfig, SecurityConfig
from visionsub.view_models.main_view_model import MainViewModel

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Test Configuration
@dataclass
class TestConfig:
    """Test configuration class"""
    # Performance thresholds
    max_widget_creation_time: float = 2.0
    max_theme_switching_time: float = 2.0
    max_large_data_loading_time: float = 3.0
    max_memory_increase: int = 100 * 1024 * 1024  # 100MB
    max_responsiveness_time: float = 1.0
    
    # Test data sizes
    large_result_count: int = 1000
    concurrent_thread_count: int = 10
    
    # File settings
    max_file_size_mb: int = 100
    allowed_video_formats: List[str] = field(default_factory=lambda: ["mp4", "avi", "mkv", "mov", "wmv", "flv"])
    
    # Security settings
    max_input_length: int = 1000
    rate_limit_requests: int = 60
    rate_limit_window: int = 60
    
    # UI settings
    default_window_size: Tuple[int, int] = (1200, 800)
    default_timeout: float = 5.0
    
    # Test timeouts
    short_timeout: float = 0.1
    medium_timeout: float = 1.0
    long_timeout: float = 5.0


# Test Categories
class TestCategory(Enum):
    """Test categories for organization"""
    UNIT = "unit"
    INTEGRATION = "integration"
    SECURITY = "security"
    PERFORMANCE = "performance"
    ACCESSIBILITY = "accessibility"
    USER_EXPERIENCE = "ux"
    REGRESSION = "regression"
    SMOKE = "smoke"
    E2E = "e2e"


# Test Data Classes
@dataclass
class VideoTestData:
    """Test video data"""
    path: str
    width: int
    height: int
    duration: float
    frame_count: int
    fps: float


@dataclass
class ImageTestData:
    """Test image data"""
    array: np.ndarray
    width: int
    height: int
    channels: int
    text_content: str


@dataclass
class OCRTestData:
    """Test OCR data"""
    results: List[Dict[str, Any]]
    total_count: int
    avg_confidence: float
    languages: List[str]


@dataclass
class SubtitleTestData:
    """Test subtitle data"""
    entries: List[Dict[str, Any]]
    total_duration: float
    language: str
    format_type: str


@dataclass
class TestData:
    """Complete test data package"""
    video: VideoTestData
    image: ImageTestData
    ocr: OCRTestData
    subtitle: SubtitleTestData
    config: AppConfig
    temp_files: List[str] = field(default_factory=list)


# Performance Metrics
@dataclass
class PerformanceMetrics:
    """Performance metrics collection"""
    execution_time: float = 0.0
    memory_usage: int = 0
    cpu_usage: float = 0.0
    peak_memory: int = 0
    operation_count: int = 0
    
    def add_measurement(self, execution_time: float, memory_usage: int = 0):
        """Add performance measurement"""
        self.execution_time += execution_time
        self.memory_usage = max(self.memory_usage, memory_usage)
        self.peak_memory = max(self.peak_memory, memory_usage)
        self.operation_count += 1
    
    def get_average_time(self) -> float:
        """Get average execution time"""
        return self.execution_time / self.operation_count if self.operation_count > 0 else 0.0


# Test Utilities
class TestUtils:
    """Test utility functions"""
    
    @staticmethod
    def create_test_video(duration: float = 1.0, fps: float = 30.0, 
                        width: int = 640, height: int = 480) -> str:
        """Create test video file"""
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as f:
            video_path = f.name
            
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(video_path, fourcc, fps, (width, height))
        
        frame_count = int(duration * fps)
        for i in range(frame_count):
            frame = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
            # Add frame number and timestamp
            cv2.putText(frame, f"Frame {i}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, f"Time: {i/fps:.2f}s", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            out.write(frame)
        
        out.release()
        return video_path
    
    @staticmethod
    def create_test_image(width: int = 640, height: int = 480, 
                         text: str = "Test OCR Text") -> np.ndarray:
        """Create test image with text"""
        image = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
        cv2.putText(image, text, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        return image
    
    @staticmethod
    def create_test_ocr_results(count: int = 10) -> List[Dict[str, Any]]:
        """Create test OCR results"""
        results = []
        for i in range(count):
            result = {
                "text": f"Test text {i}",
                "confidence": 0.5 + (i % 50) * 0.01,
                "language": "en" if i % 2 == 0 else "zh",
                "position": {
                    "x": (i * 50) % 600,
                    "y": (i * 30) % 400,
                    "width": 100 + (i % 50),
                    "height": 30 + (i % 20)
                },
                "timestamp": i * 0.5
            }
            results.append(result)
        return results
    
    @staticmethod
    def create_test_subtitles(count: int = 10, duration: float = 5.0) -> List[Dict[str, Any]]:
        """Create test subtitles"""
        subtitles = []
        for i in range(count):
            start_time = i * duration
            end_time = (i + 1) * duration
            subtitle = {
                "start": int(start_time * 1000),  # Convert to milliseconds
                "end": int(end_time * 1000),
                "text": f"Subtitle {i + 1}",
                "index": i + 1
            }
            subtitles.append(subtitle)
        return subtitles
    
    @staticmethod
    def create_test_config() -> AppConfig:
        """Create test configuration"""
        return AppConfig(
            processing=ProcessingConfig(
                ocr_config=OcrConfig(
                    engine="PaddleOCR",
                    language="en",
                    threshold=128,
                    confidence_threshold=0.8,
                    enable_preprocessing=True,
                    enable_postprocessing=True
                ),
                scene_threshold=0.3,
                cache_size=100,
                max_concurrent_jobs=4,
                frame_interval=1.0
            ),
            ui=UIConfig(
                theme="dark",
                language="en",
                window_size=[1200, 800],
                font_size=12,
                enable_animations=True,
                show_performance_metrics=True
            ),
            security=SecurityConfig(
                enable_input_validation=True,
                max_file_size_mb=100,
                allowed_video_formats=["mp4", "avi", "mkv"],
                enable_rate_limiting=True,
                rate_limit_requests_per_minute=60
            )
        )
    
    @staticmethod
    def measure_performance(func: Callable, *args, **kwargs) -> Tuple[Any, PerformanceMetrics]:
        """Measure performance of a function"""
        metrics = PerformanceMetrics()
        
        # Measure memory before
        process = psutil.Process()
        memory_before = process.memory_info().rss
        
        # Measure CPU before
        cpu_before = process.cpu_percent()
        
        # Execute function and measure time
        start_time = time.time()
        result = func(*args, **kwargs)
        execution_time = time.time() - start_time
        
        # Measure memory after
        memory_after = process.memory_info().rss
        
        # Measure CPU after
        cpu_after = process.cpu_percent()
        
        # Update metrics
        metrics.add_measurement(execution_time, memory_after - memory_before)
        metrics.cpu_usage = max(cpu_before, cpu_after)
        
        return result, metrics
    
    @staticmethod
    def wait_for_condition(condition_func: Callable, timeout: float = 5.0, 
                          interval: float = 0.1) -> bool:
        """Wait for a condition to become true"""
        start_time = time.time()
        while time.time() - start_time < timeout:
            if condition_func():
                return True
            time.sleep(interval)
        return False
    
    @staticmethod
    def simulate_user_interaction(widget: QWidget, actions: List[Tuple[str, Any]]):
        """Simulate user interaction with widget"""
        for action_type, action_data in actions:
            if action_type == "click":
                if isinstance(action_data, QPoint):
                    QTest.mouseClick(widget, Qt.MouseButton.LeftButton, pos=action_data)
                else:
                    QTest.mouseClick(action_data, Qt.MouseButton.LeftButton)
            elif action_type == "key_click":
                QTest.keyClick(widget, action_data)
            elif action_type == "key_clicks":
                QTest.keyClicks(widget, action_data)
            elif action_type == "drag":
                start_pos, end_pos = action_data
                QTest.mousePress(widget, Qt.MouseButton.LeftButton, pos=start_pos)
                QTest.mouseMove(widget, pos=end_pos)
                QTest.mouseRelease(widget, Qt.MouseButton.LeftButton, pos=end_pos)
    
    @staticmethod
    def get_widget_screenshot(widget: QWidget) -> QImage:
        """Get screenshot of widget"""
        pixmap = widget.grab()
        return pixmap.toImage()
    
    @staticmethod
    def compare_images(image1: QImage, image2: QImage, tolerance: float = 0.01) -> bool:
        """Compare two images with tolerance"""
        if image1.size() != image2.size():
            return False
        
        # Convert to numpy arrays for comparison
        ptr1 = image1.constBits()
        ptr2 = image2.constBits()
        
        if ptr1 is None or ptr2 is None:
            return False
        
        # Simple pixel comparison
        width = image1.width()
        height = image1.height()
        
        different_pixels = 0
        total_pixels = width * height
        
        for y in range(height):
            for x in range(width):
                pixel1 = image1.pixelColor(x, y)
                pixel2 = image2.pixelColor(x, y)
                
                if pixel1 != pixel2:
                    different_pixels += 1
        
        difference_ratio = different_pixels / total_pixels
        return difference_ratio <= tolerance


# Test Context Manager
@contextmanager
def temporary_widget(widget_class, *args, **kwargs):
    """Context manager for temporary widget creation and cleanup"""
    widget = widget_class(*args, **kwargs)
    try:
        yield widget
    finally:
        widget.close()
        widget.deleteLater()


@contextmanager
def measure_time(description: str = ""):
    """Context manager for measuring execution time"""
    start_time = time.time()
    try:
        yield
    finally:
        end_time = time.time()
        execution_time = end_time - start_time
        if description:
            logger.info(f"{description}: {execution_time:.3f}s")
        else:
            logger.info(f"Execution time: {execution_time:.3f}s")


@contextmanager
def measure_memory():
    """Context manager for measuring memory usage"""
    process = psutil.Process()
    memory_before = process.memory_info().rss
    try:
        yield
    finally:
        memory_after = process.memory_info().rss
        memory_increase = memory_after - memory_before
        logger.info(f"Memory increase: {memory_increase / 1024 / 1024:.2f}MB")


# Test Data Factory
class TestDataFactory:
    """Factory for creating test data"""
    
    @staticmethod
    def create_comprehensive_test_data() -> TestData:
        """Create comprehensive test data"""
        # Create test video
        video_path = TestUtils.create_test_video(duration=2.0, fps=30.0)
        video_data = VideoTestData(
            path=video_path,
            width=640,
            height=480,
            duration=2.0,
            frame_count=60,
            fps=30.0
        )
        
        # Create test image
        image_array = TestUtils.create_test_image(text="Comprehensive Test")
        image_data = ImageTestData(
            array=image_array,
            width=640,
            height=480,
            channels=3,
            text_content="Comprehensive Test"
        )
        
        # Create test OCR results
        ocr_results = TestUtils.create_test_ocr_results(count=20)
        ocr_data = OCRTestData(
            results=ocr_results,
            total_count=len(ocr_results),
            avg_confidence=sum(r['confidence'] for r in ocr_results) / len(ocr_results),
            languages=list(set(r['language'] for r in ocr_results))
        )
        
        # Create test subtitles
        subtitle_entries = TestUtils.create_test_subtitles(count=15, duration=3.0)
        subtitle_data = SubtitleTestData(
            entries=subtitle_entries,
            total_duration=45.0,
            language="en",
            format_type="srt"
        )
        
        # Create test config
        config = TestUtils.create_test_config()
        
        return TestData(
            video=video_data,
            image=image_data,
            ocr=ocr_data,
            subtitle=subtitle_data,
            config=config,
            temp_files=[video_path]
        )
    
    @staticmethod
    def cleanup_test_data(test_data: TestData):
        """Clean up test data"""
        for temp_file in test_data.temp_files:
            try:
                Path(temp_file).unlink(missing_ok=True)
            except Exception as e:
                logger.warning(f"Failed to cleanup temp file {temp_file}: {e}")


# Custom Test Fixtures
@pytest.fixture(scope="session")
def test_config():
    """Test configuration fixture"""
    return TestConfig()


@pytest.fixture(scope="session")
def test_utils():
    """Test utilities fixture"""
    return TestUtils


@pytest.fixture(scope="session")
def test_data_factory():
    """Test data factory fixture"""
    return TestDataFactory


@pytest.fixture(scope="function")
def comprehensive_test_data(test_data_factory):
    """Comprehensive test data fixture"""
    data = test_data_factory.create_comprehensive_test_data()
    yield data
    test_data_factory.cleanup_test_data(data)


@pytest.fixture(scope="session")
def app():
    """QApplication fixture"""
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    yield app
    app.quit()


@pytest.fixture(scope="function")
def view_model():
    """View model fixture"""
    return MainViewModel()


@pytest.fixture(scope="function")
def config():
    """Configuration fixture"""
    return TestUtils.create_test_config()


@pytest.fixture(scope="function")
def performance_metrics():
    """Performance metrics fixture"""
    return PerformanceMetrics()


# Custom Test Markers
def pytest_configure(config):
    """Configure custom test markers"""
    config.addinivalue_line(
        "markers", "ui: marks tests as UI tests"
    )
    config.addinivalue_line(
        "markers", "security: marks tests as security tests"
    )
    config.addinivalue_line(
        "markers", "performance: marks tests as performance tests"
    )
    config.addinivalue_line(
        "markers", "accessibility: marks tests as accessibility tests"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "slow: marks tests as slow tests"
    )


# Custom Test Hooks
def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers"""
    for item in items:
        # Add markers based on test file names
        if "security" in item.fspath.basename:
            item.add_marker(pytest.mark.security)
        elif "performance" in item.fspath.basename:
            item.add_marker(pytest.mark.performance)
        elif "accessibility" in item.fspath.basename:
            item.add_marker(pytest.mark.accessibility)
        elif "integration" in item.fspath.basename:
            item.add_marker(pytest.mark.integration)
        elif "ui" in item.fspath.basename:
            item.add_marker(pytest.mark.ui)
        
        # Add slow marker for performance tests
        if "performance" in item.fspath.basename:
            item.add_marker(pytest.mark.slow)


# Custom Test Report
@pytest.hookimpl(tryfirst=True, hookwrapper=True)
def pytest_runtest_makereport(item, call):
    """Custom test report hook"""
    outcome = yield
    report = outcome.get_result()
    
    if report.when == "call":
        # Add performance metrics to report
        if hasattr(item, "performance_metrics"):
            report.performance_metrics = item.performance_metrics
        
        # Add screenshots to report for UI tests
        if "ui" in item.keywords and hasattr(item, "screenshots"):
            report.screenshots = item.screenshots


# Test Helpers
class TestHelpers:
    """Helper functions for tests"""
    
    @staticmethod
    def wait_for_signal(signal, timeout: float = 5.0) -> bool:
        """Wait for a signal to be emitted"""
        result = {"received": False}
        
        def on_signal_received(*args, **kwargs):
            result["received"] = True
        
        signal.connect(on_signal_received)
        
        start_time = time.time()
        while time.time() - start_time < timeout:
            if result["received"]:
                signal.disconnect(on_signal_received)
                return True
            QTest.qWait(10)
        
        signal.disconnect(on_signal_received)
        return False
    
    @staticmethod
    def get_widget_by_name(widget: QWidget, name: str) -> Optional[QWidget]:
        """Get widget by object name"""
        if widget.objectName() == name:
            return widget
        
        for child in widget.findChildren(QWidget):
            found = TestHelpers.get_widget_by_name(child, name)
            if found:
                return found
        
        return None
    
    @staticmethod
    def get_widget_by_type(widget: QWidget, widget_type: type) -> List[QWidget]:
        """Get all widgets of specific type"""
        return widget.findChildren(widget_type)
    
    @staticmethod
    def simulate_file_drop(widget: QWidget, file_path: str):
        """Simulate file drop on widget"""
        from PyQt6.QtCore import QMimeData, QUrl
        
        mime_data = QMimeData()
        mime_data.setUrls([QUrl.fromLocalFile(file_path)])
        
        drop_event = QDropEvent(
            QPoint(100, 100),
            Qt.DropAction.CopyAction,
            mime_data,
            Qt.MouseButton.NoButton,
            Qt.KeyboardModifier.NoModifier
        )
        
        widget.dropEvent(drop_event)
    
    @staticmethod
    def verify_widget_properties(widget: QWidget, expected_properties: Dict[str, Any]):
        """Verify widget properties"""
        for property_name, expected_value in expected_properties.items():
            actual_value = getattr(widget, property_name, None)
            assert actual_value == expected_value, f"Property {property_name}: expected {expected_value}, got {actual_value}"
    
    @staticmethod
    def verify_signal_connections(widget: QWidget, expected_signals: List[str]):
        """Verify signal connections"""
        for signal_name in expected_signals:
            assert hasattr(widget, signal_name), f"Signal {signal_name} not found"
            signal = getattr(widget, signal_name)
            assert isinstance(signal, pyqtSignal), f"{signal_name} is not a signal"


# Performance Test Decorator
def performance_test(max_time: float = None, max_memory: int = None):
    """Decorator for performance tests"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            # Measure performance
            result, metrics = TestUtils.measure_performance(func, *args, **kwargs)
            
            # Store metrics on test item
            if hasattr(args[0], "performance_metrics"):
                args[0].performance_metrics = metrics
            
            # Assert performance requirements
            if max_time is not None:
                assert metrics.execution_time <= max_time, f"Execution time {metrics.execution_time:.3f}s exceeds maximum {max_time:.3f}s"
            
            if max_memory is not None:
                assert metrics.memory_usage <= max_memory, f"Memory usage {metrics.memory_usage} bytes exceeds maximum {max_memory} bytes"
            
            return result
        
        return wrapper
    return decorator


# Security Test Decorator
def security_test():
    """Decorator for security tests"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            # Store original function for security testing
            original_func = func
            
            def secure_func(*args, **kwargs):
                # Add security checks
                result = original_func(*args, **kwargs)
                
                # Verify security properties
                # This can be customized based on specific security requirements
                
                return result
            
            return secure_func(*args, **kwargs)
        
        return wrapper
    return decorator


# Accessibility Test Decorator
def accessibility_test():
    """Decorator for accessibility tests"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            # Add accessibility checks
            result = func(*args, **kwargs)
            
            # Verify accessibility properties
            # This can be customized based on specific accessibility requirements
            
            return result
        
        return wrapper
    return decorator


# Export utilities
__all__ = [
    "TestConfig",
    "TestCategory",
    "TestData",
    "VideoTestData",
    "ImageTestData",
    "OCRTestData",
    "SubtitleTestData",
    "PerformanceMetrics",
    "TestUtils",
    "TestDataFactory",
    "TestHelpers",
    "temporary_widget",
    "measure_time",
    "measure_memory",
    "performance_test",
    "security_test",
    "accessibility_test",
]