"""
Test configuration for VisionSub
"""
import pytest
import tempfile
import os
import sys
import numpy as np
import cv2
from pathlib import Path
from unittest.mock import Mock, AsyncMock
from visionsub.models.config import AppConfig, ProcessingConfig, OcrConfig


@pytest.fixture
def temp_config_file():
    """Create a temporary config file"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        config_data = {
            "processing": {
                "ocr_config": {
                    "engine": "PaddleOCR",
                    "language": "中文",
                    "confidence_threshold": 0.8
                },
                "scene_threshold": 0.3,
                "cache_size": 100
            },
            "ui": {
                "theme": "dark",
                "language": "zh_CN"
            }
        }
        import json
        json.dump(config_data, f, indent=2)
        f.flush()
        yield f.name
    
    Path(f.name).unlink()


@pytest.fixture
def sample_ocr_config():
    """Sample OCR configuration for testing"""
    return OcrConfig(
        engine="PaddleOCR",
        language="中文",
        confidence_threshold=0.8,
        roi_rect=(0, 900, 1920, 180),
        denoise=True,
        enhance_contrast=True,
        threshold=180,
        sharpen=True,
        auto_detect_language=True
    )


@pytest.fixture
def sample_processing_config(sample_ocr_config):
    """Sample processing configuration for testing"""
    return ProcessingConfig(
        ocr_config=sample_ocr_config,
        scene_threshold=0.3,
        cache_size=100,
        frame_interval=1.0,
        enable_scene_detection=True,
        enable_parallel_processing=True,
        memory_limit_mb=1024,
        output_formats=["srt", "vtt"],
        output_directory="./output",
        create_subdirectories=True
    )


@pytest.fixture
def sample_app_config(sample_processing_config):
    """Sample app configuration for testing"""
    return AppConfig(
        processing=sample_processing_config,
        ui={
            "theme": "dark",
            "language": "zh_CN",
            "window_size": (1200, 800),
            "font_size": 10,
            "enable_animations": True,
            "show_performance_metrics": True
        },
        logging={
            "level": "INFO",
            "log_file": None,
            "max_file_size_mb": 10,
            "backup_count": 5,
            "enable_structured_logging": True
        },
        security={
            "enable_input_validation": True,
            "max_file_size_mb": 500,
            "allowed_video_formats": ["mp4", "avi", "mkv", "mov", "wmv", "flv", "webm"],
            "enable_rate_limiting": True,
            "rate_limit_requests_per_minute": 60
        }
    )


# Additional test fixtures and configuration
def pytest_configure(config):
    """Configure pytest with custom settings"""
    # Set test environment variables
    os.environ.setdefault('VISIONSUB_ENVIRONMENT', 'test')
    os.environ.setdefault('VISIONSUB_LOG_LEVEL', 'DEBUG')
    os.environ.setdefault('VISIONSUB_TEST_MODE', 'true')
    
    # Register custom markers
    config.addinivalue_line(
        "markers", "unit: Unit tests for individual components"
    )
    config.addinivalue_line(
        "markers", "integration: Integration tests for component interactions"
    )
    config.addinivalue_line(
        "markers", "performance: Performance tests for benchmarking"
    )
    config.addinivalue_line(
        "markers", "slow: Tests that take a long time to run"
    )
    config.addinivalue_line(
        "markers", "gpu: Tests that require GPU acceleration"
    )
    config.addinivalue_line(
        "markers", "gui: Tests that require GUI environment"
    )
    config.addinivalue_line(
        "markers", "network: Tests that require network access"
    )
    config.addinivalue_line(
        "markers", "database: Tests that require database access"
    )
    config.addinivalue_line(
        "markers", "security: Security-related tests"
    )
    config.addinivalue_line(
        "markers", "api: API-related tests"
    )
    config.addinivalue_line(
        "markers", "ocr: OCR-related tests"
    )
    config.addinivalue_line(
        "markers", "video: Video processing tests"
    )
    config.addinivalue_line(
        "markers", "memory: Memory management tests"
    )
    config.addinivalue_line(
        "markers", "logging: Logging system tests"
    )
    config.addinivalue_line(
        "markers", "health: Health check tests"
    )
    config.addinivalue_line(
        "markers", "config: Configuration tests"
    )
    config.addinivalue_line(
        "markers", "async: Asynchronous operation tests"
    )
    config.addinivalue_line(
        "markers", "cache: Cache-related tests"
    )
    config.addinivalue_line(
        "markers", "ui: User interface tests"
    )
    config.addinivalue_line(
        "markers", "web: Web interface tests"
    )
    config.addinivalue_line(
        "markers", "batch: Batch processing tests"
    )
    config.addinivalue_line(
        "markers", "realtime: Real-time processing tests"
    )
    config.addinivalue_line(
        "markers", "regression: Regression tests"
    )
    config.addinivalue_line(
        "markers", "smoke: Smoke tests for basic functionality"
    )
    config.addinivalue_line(
        "markers", "e2e: End-to-end tests"
    )


@pytest.fixture(scope="session")
def test_config():
    """Test configuration fixture"""
    return {
        "environment": "test",
        "log_level": "DEBUG",
        "test_mode": True,
        "timeout": 30,
        "temp_dir": "/tmp/visionsub_test"
    }


@pytest.fixture(scope="session")
def test_data_dir():
    """Test data directory fixture"""
    test_dir = Path(__file__).parent / "test_data"
    test_dir.mkdir(exist_ok=True)
    return test_dir


@pytest.fixture(scope="session")
def temp_dir():
    """Temporary directory fixture"""
    import tempfile
    temp_path = Path(tempfile.mkdtemp(prefix="visionsub_test_"))
    yield temp_path
    # Cleanup
    import shutil
    shutil.rmtree(temp_path, ignore_errors=True)


@pytest.fixture
def sample_image():
    """Sample image fixture for testing"""
    return np.zeros((100, 100, 3), dtype=np.uint8)


@pytest.fixture
def sample_video_file(temp_dir):
    """Sample video file fixture for testing"""
    video_path = temp_dir / "test_video.mp4"
    
    # Create a simple test video
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    out = cv2.VideoWriter(str(video_path), fourcc, 30.0, (640, 480))
    
    for i in range(30):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(frame, f"Test Frame {i}", (50, 240), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        out.write(frame)
    
    out.release()
    return video_path


@pytest.fixture
def mock_ocr_engine():
    """Mock OCR engine fixture"""
    mock_engine = Mock()
    mock_engine.process_image = AsyncMock(return_value={
        "text": "Mock OCR Result",
        "confidence": 0.95,
        "boxes": [[0, 0, 100, 100]]
    })
    mock_engine.process_batch = AsyncMock(return_value=[
        {"text": f"Mock Result {i}", "confidence": 0.9, "boxes": [[0, 0, 100, 100]]}
        for i in range(5)
    ])
    
    return mock_engine


@pytest.fixture
def mock_video_processor():
    """Mock video processor fixture"""
    mock_processor = Mock()
    mock_processor.get_video_info.return_value = {
        "fps": 30.0,
        "frame_count": 300,
        "duration": 10.0,
        "width": 1920,
        "height": 1080
    }
    mock_processor.extract_frames.return_value = [
        np.zeros((480, 640, 3), dtype=np.uint8) for _ in range(10)
    ]
    
    return mock_processor


@pytest.fixture
def event_loop():
    """Event loop fixture for async tests"""
    import asyncio
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


# Performance test helpers
class PerformanceTracker:
    """Helper class for tracking performance metrics"""
    
    def __init__(self):
        self.metrics = {}
    
    def start_timing(self, name):
        """Start timing a operation"""
        import time
        self.metrics[name] = {"start_time": time.time()}
    
    def end_timing(self, name):
        """End timing a operation"""
        import time
        if name in self.metrics:
            self.metrics[name]["end_time"] = time.time()
            self.metrics[name]["duration"] = (
                self.metrics[name]["end_time"] - self.metrics[name]["start_time"]
            )
    
    def get_metric(self, name):
        """Get a specific metric"""
        return self.metrics.get(name, {})
    
    def get_all_metrics(self):
        """Get all metrics"""
        return self.metrics


@pytest.fixture
def performance_tracker():
    """Performance tracker fixture"""
    return PerformanceTracker()


# Test data generators
def generate_test_images(count=10, size=(100, 100)):
    """Generate test images"""
    return [np.random.randint(0, 255, (*size, 3), dtype=np.uint8) for _ in range(count)]


def generate_test_video_frames(count=30, size=(640, 480)):
    """Generate test video frames"""
    return [np.random.randint(0, 255, (*size, 3), dtype=np.uint8) for _ in range(count)]


def generate_test_config():
    """Generate test configuration"""
    return {
        "app": {
            "name": "VisionSub Test",
            "version": "1.0.0",
            "environment": "test"
        },
        "ocr": {
            "default_engine": "mock",
            "preprocessing": {"enabled": False}
        },
        "logging": {
            "level": "DEBUG",
            "format": "text"
        }
    }


# Custom assertions
def assert_image_equal(img1, img2, tolerance=1e-6):
    """Assert that two images are equal within tolerance"""
    assert np.allclose(img1, img2, rtol=tolerance, atol=tolerance)


def assert_performance_metric(metric, expected_min, expected_max=None):
    """Assert that a performance metric is within expected range"""
    if expected_max is not None:
        assert expected_min <= metric <= expected_max
    else:
        assert metric >= expected_min


def assert_memory_usage_within_limit(usage_mb, limit_mb):
    """Assert that memory usage is within limit"""
    assert usage_mb <= limit_mb, f"Memory usage {usage_mb}MB exceeds limit {limit_mb}MB"


def assert_processing_time_within_limit(time_s, limit_s):
    """Assert that processing time is within limit"""
    assert time_s <= limit_s, f"Processing time {time_s}s exceeds limit {limit_s}s"


# Skip markers for different environments
def pytest_collection_modifyitems(config, items):
    """Modify test collection to add skip markers"""
    skip_gui = pytest.mark.skip(reason="GUI not available in test environment")
    skip_gpu = pytest.mark.skip(reason="GPU not available")
    skip_network = pytest.mark.skip(reason="Network access not available")
    skip_database = pytest.mark.skip(reason="Database not available")
    
    for item in items:
        # Skip GUI tests in non-GUI environment
        if "gui" in item.keywords and not os.environ.get('DISPLAY'):
            item.add_marker(skip_gui)
        
        # Skip GPU tests if no GPU available
        if "gpu" in item.keywords:
            try:
                import torch
                if not torch.cuda.is_available():
                    item.add_marker(skip_gpu)
            except ImportError:
                item.add_marker(skip_gpu)
        
        # Skip network tests if network not available
        if "network" in item.keywords:
            try:
                import socket
                socket.create_connection(("8.8.8.8", 53), timeout=1)
            except:
                item.add_marker(skip_network)
        
        # Skip database tests if database not available
        if "database" in item.keywords:
            try:
                import sqlite3
                sqlite3.connect(":memory:")
            except:
                item.add_marker(skip_database)