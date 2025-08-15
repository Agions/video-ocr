"""
Test Configuration and Setup

This module provides configuration for the testing framework:
- Test environment setup
- Test discovery and configuration
- Test fixtures and markers
- Test runner configuration
"""

import pytest
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def pytest_configure(config):
    """Configure pytest with custom settings"""
    # Add custom markers
    config.addinivalue_line("markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')")
    config.addinivalue_line("markers", "integration: marks tests as integration tests")
    config.addinivalue_line("markers", "unit: marks tests as unit tests")
    config.addinivalue_line("markers", "performance: marks tests as performance tests")
    config.addinivalue_line("markers", "gui: marks tests as GUI tests")
    config.addinivalue_line("markers", "security: marks tests as security tests")
    config.addinivalue_line("markers", "e2e: marks tests as end-to-end tests")
    config.addinivalue_line("markers", "load: marks tests as load tests")
    config.addinivalue_line("markers", "accessibility: marks tests as accessibility tests")
    config.addinivalue_line("markers", "regression: marks tests as regression tests")
    config.addinivalue_line("markers", "smoke: marks tests as smoke tests")


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers"""
    for item in items:
        # Add slow marker to performance tests
        if "performance" in item.nodeid:
            item.add_marker(pytest.mark.slow)
        
        # Add gui marker to UI tests
        if "gui" in item.nodeid or "ui" in item.nodeid:
            item.add_marker(pytest.mark.gui)
        
        # Add integration marker to integration tests
        if "integration" in item.nodeid:
            item.add_marker(pytest.mark.integration)
        
        # Add e2e marker to end-to-end tests
        if "e2e" in item.nodeid:
            item.add_marker(pytest.mark.e2e)
        
        # Add security marker to security tests
        if "security" in item.nodeid:
            item.add_marker(pytest.mark.security)


@pytest.fixture(scope="session")
def test_environment():
    """Setup test environment for the session"""
    # Create test directories
    test_dirs = [
        "test_output",
        "test_output/screenshots",
        "test_output/reports",
        "test_output/logs",
        "test_data",
        "test_data/videos",
        "test_data/images",
        "test_data/configs"
    ]
    
    for dir_path in test_dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    yield
    
    # Cleanup (optional)
    # for dir_path in test_dirs:
    #     import shutil
    #     shutil.rmtree(dir_path, ignore_errors=True)


@pytest.fixture(scope="session")
def app_config():
    """Application configuration for testing"""
    return {
        "test_mode": True,
        "log_level": "DEBUG",
        "temp_dir": "test_output/temp",
        "data_dir": "test_data",
        "screenshot_dir": "test_output/screenshots",
        "report_dir": "test_output/reports",
        "log_dir": "test_output/logs"
    }


@pytest.fixture
def temp_dir(test_environment):
    """Temporary directory for individual tests"""
    import tempfile
    import shutil
    
    temp_path = tempfile.mkdtemp(prefix="test_")
    yield temp_path
    
    # Cleanup
    shutil.rmtree(temp_path, ignore_errors=True)


@pytest.fixture
def temp_file(temp_dir):
    """Temporary file for individual tests"""
    import tempfile
    
    with tempfile.NamedTemporaryFile(dir=temp_dir, delete=False) as f:
        yield f.name
    
    # Cleanup is handled by temp_dir fixture


# Configure pytest options
pytest_plugins = [
    "pytest_qt",
    "pytest_mock",
    "pytest_asyncio",
    "pytest_benchmark",
    "pytest_html",
    "pytest_cov"
]

# Test configuration
TEST_CONFIG = {
    "min_python_version": (3, 9),
    "coverage_threshold": 80,
    "performance_threshold": {
        "video_loading": 2.0,
        "ocr_processing": 3.0,
        "subtitle_export": 1.0
    },
    "memory_threshold": 500 * 1024 * 1024,  # 500MB
    "timeout": 30,  # 30 seconds default timeout
    "retry_count": 3,
    "parallel_workers": 4
}


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    """Add custom summary information"""
    terminalreporter.section("VisionSub Test Summary")
    
    # Add test statistics
    total = terminalreporter._numcollected
    passed = len(terminalreporter.stats.get('passed', []))
    failed = len(terminalreporter.stats.get('failed', []))
    skipped = len(terminalreporter.stats.get('skipped', []))
    
    terminalreporter.write_line(f"Total tests: {total}")
    terminalreporter.write_line(f"Passed: {passed}")
    terminalreporter.write_line(f"Failed: {failed}")
    terminalreporter.write_line(f"Skipped: {skipped}")
    
    if failed > 0:
        terminalreporter.write_line(f"Failure rate: {failed/total*100:.1f}%")
    
    # Add performance summary if available
    if 'benchmark' in terminalreporter.stats:
        terminalreporter.section("Performance Summary")
        benchmarks = terminalreporter.stats['benchmark']
        for benchmark in benchmarks:
            if hasattr(benchmark, 'stats'):
                stats = benchmark.stats
                terminalreporter.write_line(f"{benchmark.name}: {stats.mean:.3f}s ± {stats.stddev:.3f}s")


# Custom assertion helpers
def assert_image_similarity(image1, image2, threshold=0.99):
    """Assert that two images are similar"""
    import cv2
    import numpy as np
    
    if image1.shape != image2.shape:
        pytest.fail("Images have different shapes")
    
    # Convert to grayscale
    if len(image1.shape) == 3:
        gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    else:
        gray1 = image1
        gray2 = image2
    
    # Calculate similarity
    correlation = cv2.matchTemplate(gray1, gray2, cv2.TM_CCOEFF_NORMED)
    max_corr = np.max(correlation)
    
    if max_corr < threshold:
        pytest.fail(f"Images are not similar enough: {max_corr} < {threshold}")


def assert_performance_within_threshold(duration, threshold, operation_name):
    """Assert that performance is within acceptable threshold"""
    if duration > threshold:
        pytest.fail(f"{operation_name} too slow: {duration:.3f}s > {threshold:.3f}s")


def assert_memory_usage_within_limit(memory_usage, limit, operation_name):
    """Assert that memory usage is within acceptable limit"""
    if memory_usage > limit:
        pytest.fail(f"{operation_name} uses too much memory: {memory_usage} > {limit}")


# Test data generation utilities
class TestDataGenerator:
    """Generate test data for various scenarios"""
    
    @staticmethod
    def generate_video_frames(count=10, width=640, height=480):
        """Generate video frames for testing"""
        frames = []
        for i in range(count):
            frame = np.full((height, width, 3), [i % 256, 100, 200], dtype=np.uint8)
            cv2.putText(frame, f"Frame {i}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
            frames.append(frame)
        return frames
    
    @staticmethod
    def generate_ocr_results(count=5):
        """Generate OCR results for testing"""
        results = []
        for i in range(count):
            result = {
                "text": f"OCR Result {i + 1}",
                "confidence": 0.8 + (i * 0.05),
                "bbox": [50 + i*10, 80 + i*5, 200 + i*10, 120 + i*5],
                "language": "en"
            }
            results.append(result)
        return results
    
    @staticmethod
    def generate_subtitle_entries(count=5):
        """Generate subtitle entries for testing"""
        entries = []
        for i in range(count):
            entry = {
                "start": i * 2000,
                "end": (i + 1) * 2000,
                "text": f"Subtitle {i + 1}",
                "index": i + 1
            }
            entries.append(entry)
        return entries


# Error handling utilities
class TestErrorHandler:
    """Handle test errors and provide helpful messages"""
    
    @staticmethod
    def handle_qt_error(error):
        """Handle Qt-related errors"""
        return f"Qt Error: {str(error)}"
    
    @staticmethod
    def handle_opencv_error(error):
        """Handle OpenCV-related errors"""
        return f"OpenCV Error: {str(error)}"
    
    @staticmethod
    def handle_file_error(error):
        """Handle file-related errors"""
        return f"File Error: {str(error)}"
    
    @staticmethod
    def handle_config_error(error):
        """Handle configuration-related errors"""
        return f"Configuration Error: {str(error)}"


# Test runner utilities
class TestRunner:
    """Utilities for running tests"""
    
    @staticmethod
    def run_category_tests(category, markers=None):
        """Run tests in a specific category"""
        import subprocess
        
        cmd = [sys.executable, "-m", "pytest"]
        
        if markers:
            cmd.extend(["-m", " or ".join(markers)])
        
        if category == "unit":
            cmd.extend(["test_enhanced_ui.py", "test_enhanced_backend.py"])
        elif category == "integration":
            cmd.append("test_integration.py")
        elif category == "performance":
            cmd.append("test_performance.py")
        elif category == "security":
            cmd.append("test_security.py")
        elif category == "e2e":
            cmd.append("test_e2e.py")
        
        cmd.extend(["-v", "--tb=short"])
        
        return subprocess.run(cmd, capture_output=True, text=True)
    
    @staticmethod
    def run_all_tests():
        """Run all tests"""
        return TestRunner.run_category_tests("all", ["not slow"])
    
    @staticmethod
    def run_ci_tests():
        """Run tests suitable for CI environment"""
        return TestRunner.run_category_tests("ci", ["unit", "integration", "not gui", "not slow"])


# Export commonly used fixtures and utilities
__all__ = [
    "test_environment",
    "app_config",
    "temp_dir",
    "temp_file",
    "assert_image_similarity",
    "assert_performance_within_threshold",
    "assert_memory_usage_within_limit",
    "TestDataGenerator",
    "TestErrorHandler",
    "TestRunner",
    "TEST_CONFIG"
]