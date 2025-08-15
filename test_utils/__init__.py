"""
Test Utilities for VisionSub Application

This module provides utility functions and fixtures for testing:
- Test data generation
- Mock objects and factories
- Test environment setup
- Performance monitoring
- Test helpers and utilities
"""

import pytest
import tempfile
import json
import numpy as np
import cv2
from pathlib import Path
from unittest.mock import Mock, AsyncMock
from PIL import Image
import time
import psutil
import threading
from factory import Factory, Faker, LazyAttribute
from faker import Faker as RealFaker
from PyQt6.QtCore import QObject, pyqtSignal

# Add src to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from visionsub.models.config import AppConfig, ProcessingConfig, OcrConfig
from visionsub.video_utils import VideoFrameExtractor
from visionsub.models.ocr import OCRResult
from visionsub.models.subtitle import SubtitleItem


class TestDataFactory:
    """Factory for generating test data"""
    
    def __init__(self):
        self.faker = RealFaker()
    
    def create_sample_image(self, width=640, height=480, text="Test Text"):
        """Create a sample image with text"""
        image = np.full((height, width, 3), 255, dtype=np.uint8)
        cv2.putText(image, text, (50, height//2), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        return image
    
    def create_sample_video(self, duration_seconds=5, fps=30, width=640, height=480, filename=None):
        """Create a sample video file"""
        if filename is None:
            with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as f:
                filename = f.name
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(filename, fourcc, fps, (width, height))
        
        total_frames = duration_seconds * fps
        for i in range(total_frames):
            frame = np.full((height, width, 3), [i % 256, 100, 200], dtype=np.uint8)
            cv2.putText(frame, f"Frame {i}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
            cv2.putText(frame, f"Time: {i/fps:.2f}s", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
            out.write(frame)
        
        out.release()
        return filename
    
    def create_ocr_result(self, text="Sample Text", confidence=0.9, language="en"):
        """Create a sample OCR result"""
        return OCRResult(
            text=text,
            confidence=confidence,
            boxes=[[50, 80, 200, 120]],
            language=language
        )
    
    def create_subtitle_entry(self, text="Sample Subtitle", start=0, end=2000, index=1):
        """Create a sample subtitle entry"""
        return SubtitleItem(
            index=index,
            start_time=f"00:00:{start//1000:02d},{start%1000:03d}",
            end_time=f"00:00:{end//1000:02d},{end%1000:03d}",
            content=text
        )
    
    def create_config(self, engine="PaddleOCR", language="中文", confidence_threshold=0.8):
        """Create a sample configuration"""
        return AppConfig(
            processing=ProcessingConfig(
                ocr_config=OcrConfig(
                    engine=engine,
                    language=language,
                    confidence_threshold=confidence_threshold
                ),
                scene_threshold=0.3,
                cache_size=100
            ),
            ui={
                "theme": "dark",
                "language": "zh_CN"
            }
        )


class MockOCRProcessor:
    """Mock OCR processor for testing"""
    
    def __init__(self, config=None):
        self.config = config or OcrConfig(engine="MockOCR", language="en", confidence_threshold=0.8)
        self.processed_images = []
    
    def process_image(self, image):
        """Mock image processing"""
        self.processed_images.append(image)
        return [self.create_mock_result()]
    
    def process_batch(self, images):
        """Mock batch processing"""
        self.processed_images.extend(images)
        return [[self.create_mock_result()] for _ in images]
    
    def create_mock_result(self):
        """Create mock OCR result"""
        return OCRResult(
            text="Mock OCR Result",
            confidence=0.95,
            boxes=[[0, 0, 100, 30]],
            language="en"
        )


class MockVideoProcessor:
    """Mock video processor for testing"""
    
    def __init__(self, config=None):
        self.config = config
        self.processed_videos = []
    
    def get_video_info(self, video_path):
        """Mock video info extraction"""
        self.processed_videos.append(video_path)
        return {
            'fps': 30.0,
            'frame_count': 150,
            'duration': 5.0,
            'width': 640,
            'height': 480
        }
    
    def extract_frame(self, video_path, timestamp):
        """Mock frame extraction"""
        self.processed_videos.append(video_path)
        return np.full((480, 640, 3), 255, dtype=np.uint8)
    
    def extract_frames_batch(self, video_path, timestamps):
        """Mock batch frame extraction"""
        self.processed_videos.append(video_path)
        return [np.full((480, 640, 3), 255, dtype=np.uint8) for _ in timestamps]


class PerformanceMonitor:
    """Performance monitoring utility"""
    
    def __init__(self):
        self.process = psutil.Process()
        self.measurements = []
    
    def start_measurement(self, name):
        """Start a performance measurement"""
        return PerformanceMeasurement(name, self)
    
    def get_cpu_usage(self):
        """Get current CPU usage"""
        return self.process.cpu_percent()
    
    def get_memory_usage(self):
        """Get current memory usage"""
        memory_info = self.process.memory_info()
        return {
            'rss': memory_info.rss,
            'vms': memory_info.vms,
            'percent': self.process.memory_percent()
        }
    
    def get_system_info(self):
        """Get system information"""
        return {
            'cpu_count': psutil.cpu_count(),
            'memory_total': psutil.virtual_memory().total,
            'disk_usage': psutil.disk_usage('/').total
        }


class PerformanceMeasurement:
    """Context manager for performance measurement"""
    
    def __init__(self, name, monitor):
        self.name = name
        self.monitor = monitor
        self.start_time = None
        self.start_memory = None
        self.start_cpu = None
        self.result = None
    
    def __enter__(self):
        self.start_time = time.time()
        self.start_memory = self.monitor.get_memory_usage()
        self.start_cpu = self.monitor.get_cpu_usage()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        end_time = time.time()
        end_memory = self.monitor.get_memory_usage()
        end_cpu = self.monitor.get_cpu_usage()
        
        self.result = {
            'name': self.name,
            'duration': end_time - self.start_time,
            'memory_increase': end_memory['rss'] - self.start_memory['rss'],
            'cpu_usage': (self.start_cpu + end_cpu) / 2,
            'timestamp': time.time()
        }
        
        self.monitor.measurements.append(self.result)


class TestEnvironment:
    """Test environment setup and management"""
    
    def __init__(self):
        self.temp_files = []
        self.temp_dirs = []
    
    def create_temp_file(self, suffix='', content=None):
        """Create a temporary file"""
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
            if content:
                f.write(content)
            temp_file = f.name
        self.temp_files.append(temp_file)
        return temp_file
    
    def create_temp_dir(self):
        """Create a temporary directory"""
        temp_dir = tempfile.mkdtemp()
        self.temp_dirs.append(temp_dir)
        return temp_dir
    
    def cleanup(self):
        """Clean up temporary files and directories"""
        for file_path in self.temp_files:
            try:
                Path(file_path).unlink()
            except FileNotFoundError:
                pass
        
        for dir_path in self.temp_dirs:
            try:
                import shutil
                shutil.rmtree(dir_path)
            except FileNotFoundError:
                pass
        
        self.temp_files.clear()
        self.temp_dirs.clear()


class TestHelpers:
    """Helper functions for testing"""
    
    @staticmethod
    def wait_for_signal(signal, timeout=5000):
        """Wait for a signal to be emitted"""
        result = {'called': False, 'args': None, 'kwargs': None}
        
        def slot(*args, **kwargs):
            result['called'] = True
            result['args'] = args
            result['kwargs'] = kwargs
        
        signal.connect(slot)
        
        # Wait for signal (simplified - in real implementation would use QTest.qWait)
        start_time = time.time()
        while not result['called'] and (time.time() - start_time) * 1000 < timeout:
            time.sleep(0.01)
        
        signal.disconnect(slot)
        return result
    
    @staticmethod
    def compare_images(image1, image2, threshold=0.99):
        """Compare two images for similarity"""
        if image1.shape != image2.shape:
            return False
        
        # Convert to grayscale if needed
        if len(image1.shape) == 3:
            gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
        else:
            gray1 = image1
            gray2 = image2
        
        # Calculate similarity
        correlation = cv2.matchTemplate(gray1, gray2, cv2.TM_CCOEFF_NORMED)
        max_corr = np.max(correlation)
        
        return max_corr >= threshold
    
    @staticmethod
    def create_test_subtitles(count=5):
        """Create test subtitles"""
        subtitles = []
        for i in range(count):
            subtitle = SubtitleEntry(
                start=i * 2000,
                end=(i + 1) * 2000,
                text=f"Test Subtitle {i + 1}",
                index=i + 1
            )
            subtitles.append(subtitle)
        return subtitles
    
    @staticmethod
    def create_test_video_with_subtitles(filename=None, subtitle_count=5):
        """Create a test video with embedded subtitles"""
        if filename is None:
            with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as f:
                filename = f.name
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(filename, fourcc, 30.0, (640, 480))
        
        subtitle_texts = [f"Subtitle {i + 1}" for i in range(subtitle_count)]
        
        for i, text in enumerate(subtitle_texts):
            for frame in range(60):  # 2 seconds per subtitle
                frame_img = np.full((480, 640, 3), [50, 50, 80], dtype=np.uint8)
                cv2.putText(frame_img, text, (50, 400), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv2.putText(frame_img, f"Frame {i*60 + frame}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)
                out.write(frame_img)
        
        out.release()
        return filename


# Pytest fixtures
@pytest.fixture
def test_env():
    """Test environment fixture"""
    env = TestEnvironment()
    yield env
    env.cleanup()


@pytest.fixture
def data_factory():
    """Test data factory fixture"""
    return TestDataFactory()


@pytest.fixture
def performance_monitor():
    """Performance monitor fixture"""
    return PerformanceMonitor()


@pytest.fixture
def mock_ocr_processor():
    """Mock OCR processor fixture"""
    return MockOCRProcessor()


@pytest.fixture
def mock_video_processor():
    """Mock video processor fixture"""
    return MockVideoProcessor()


@pytest.fixture
def sample_config():
    """Sample configuration fixture"""
    return AppConfig(
        processing=ProcessingConfig(
            ocr_config=OcrConfig(
                engine="PaddleOCR",
                language="中文",
                confidence_threshold=0.8
            ),
            scene_threshold=0.3,
            cache_size=100
        ),
        ui={
            "theme": "dark",
            "language": "zh_CN"
        }
    )


@pytest.fixture
def sample_image():
    """Sample image fixture"""
    image = np.full((480, 640, 3), 255, dtype=np.uint8)
    cv2.putText(image, "Test Image", (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    return image


@pytest.fixture
def sample_video(test_env):
    """Sample video fixture"""
    return test_env.create_temp_file(suffix='.mp4')


@pytest.fixture
def sample_ocr_result():
    """Sample OCR result fixture"""
    return OCRResult(
        text="Sample OCR Result",
        confidence=0.9,
        bbox=(50, 80, 200, 120),
        language="en"
    )


@pytest.fixture
def sample_subtitle():
    """Sample subtitle fixture"""
    return SubtitleEntry(
        start=0,
        end=2000,
        text="Sample Subtitle",
        index=1
    )


@pytest.fixture
def sample_video_with_subtitles(test_env):
    """Sample video with subtitles fixture"""
    return TestHelpers.create_test_video_with_subtitles()


@pytest.fixture
def sample_subtitles():
    """Sample subtitles fixture"""
    return TestHelpers.create_test_subtitles()


# Factory classes for pytest-factoryboy
class OCRResultFactory(Factory):
    """Factory for OCR results"""
    class Meta:
        model = OCRResult
    
    text = Faker('sentence', nb_words=3)
    confidence = Faker('pyfloat', min_value=0.5, max_value=1.0)
    bbox = LazyAttribute(lambda: [50, 80, 200, 120])
    language = Faker('random_element', elements=['en', 'zh', 'ja', 'ko'])


class SubtitleEntryFactory(Factory):
    """Factory for subtitle entries"""
    class Meta:
        model = SubtitleEntry
    
    start = Faker('pyint', min_value=0, max_value=10000)
    end = LazyAttribute(lambda obj: obj.start + 2000)
    text = Faker('sentence', nb_words=5)
    index = Faker('pyint', min_value=1, max_value=100)


class AppConfigFactory(Factory):
    """Factory for application configuration"""
    class Meta:
        model = AppConfig
    
    processing = LazyAttribute(lambda: {
        'ocr_config': {
            'engine': Faker('random_element', elements=['PaddleOCR', 'TesseractOCR', 'EasyOCR']),
            'language': Faker('random_element', elements=['en', 'zh', 'ja', 'ko']),
            'confidence_threshold': Faker('pyfloat', min_value=0.5, max_value=1.0)
        },
        'scene_threshold': Faker('pyfloat', min_value=0.1, max_value=0.5),
        'cache_size': Faker('pyint', min_value=50, max_value=500)
    })
    
    ui = LazyAttribute(lambda: {
        'theme': Faker('random_element', elements=['dark', 'light']),
        'language': Faker('random_element', elements=['en_US', 'zh_CN', 'ja_JP', 'ko_KR'])
    })


# Custom pytest markers
def pytest_configure(config):
    """Configure custom pytest markers"""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "performance: marks tests as performance tests"
    )
    config.addinivalue_line(
        "markers", "gui: marks tests as GUI tests"
    )
    config.addinivalue_line(
        "markers", "security: marks tests as security tests"
    )
    config.addinivalue_line(
        "markers", "e2e: marks tests as end-to-end tests"
    )
    config.addinivalue_line(
        "markers", "load: marks tests as load tests"
    )
    config.addinivalue_line(
        "markers", "accessibility: marks tests as accessibility tests"
    )


# Test helpers
def create_test_suite():
    """Create a comprehensive test suite"""
    return {
        'unit_tests': [
            'test_enhanced_ui.py',
            'test_enhanced_backend.py',
            'test_security.py'
        ],
        'integration_tests': [
            'test_integration.py'
        ],
        'performance_tests': [
            'test_performance.py'
        ],
        'e2e_tests': [
            'test_e2e.py'
        ]
    }


def run_test_with_coverage(test_file, coverage_threshold=80):
    """Run a test file with coverage reporting"""
    import subprocess
    import sys
    
    cmd = [
        sys.executable, '-m', 'pytest',
        test_file,
        '--cov=src/visionsub',
        '--cov-report=term-missing',
        '--cov-report=html:htmlcov',
        '--cov-fail-under=' + str(coverage_threshold),
        '-v'
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result


def generate_test_report():
    """Generate a comprehensive test report"""
    report = {
        'timestamp': time.time(),
        'test_suites': create_test_suite(),
        'system_info': {
            'platform': sys.platform,
            'python_version': sys.version,
            'cpu_count': psutil.cpu_count(),
            'memory_total': psutil.virtual_memory().total
        }
    }
    
    return report


if __name__ == "__main__":
    # Example usage
    factory = TestDataFactory()
    image = factory.create_sample_image(text="Hello World")
    video = factory.create_sample_video(duration_seconds=3)
    
    print(f"Created test image with shape: {image.shape}")
    print(f"Created test video at: {video}")
    
    # Cleanup
    Path(video).unlink()