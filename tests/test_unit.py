"""
Unit tests for VisionSub core components
"""

import pytest
import numpy as np
import cv2
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import tempfile
import os

from visionsub.core.video_processor import VideoProcessor
from visionsub.core.error_handler import ErrorHandler, VisionSubError
from visionsub.core.security_manager import SecurityManager
from visionsub.core.memory_manager import MemoryManager
from visionsub.core.logging_system import StructuredLogger


class TestVideoProcessorUnit:
    """Unit tests for VideoProcessor"""
    
    @pytest.fixture
    def processor(self):
        """Create video processor instance"""
        return VideoProcessor()
    
    def test_processor_initialization(self, processor):
        """Test processor initialization"""
        assert processor is not None
        assert hasattr(processor, 'config')
        assert hasattr(processor, 'logger')
    
    def test_get_video_info_nonexistent_file(self, processor):
        """Test getting info from non-existent file"""
        with pytest.raises(FileNotFoundError):
            processor.get_video_info("/nonexistent/file.mp4")
    
    def test_get_video_info_invalid_format(self, processor):
        """Test getting info from invalid video format"""
        # Create a text file instead of video
        with tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as f:
            f.write(b"Not a video file")
            temp_path = f.name
        
        try:
            with pytest.raises(Exception):
                processor.get_video_info(temp_path)
        finally:
            os.unlink(temp_path)
    
    @patch('cv2.VideoCapture')
    def test_get_video_info_success(self, mock_videocapture, processor):
        """Test successful video info extraction"""
        # Mock video capture
        mock_cap = Mock()
        mock_cap.get.side_effect = lambda prop: {
            cv2.CAP_PROP_FPS: 30.0,
            cv2.CAP_PROP_FRAME_COUNT: 300,
            cv2.CAP_PROP_FRAME_WIDTH: 1920,
            cv2.CAP_PROP_FRAME_HEIGHT: 1080
        }.get(prop, 0)
        mock_cap.isOpened.return_value = True
        mock_videocapture.return_value = mock_cap
        
        # Create temporary video file
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as f:
            temp_path = f.name
        
        try:
            info = processor.get_video_info(temp_path)
            
            assert info['fps'] == 30.0
            assert info['frame_count'] == 300
            assert info['width'] == 1920
            assert info['height'] == 1080
            assert info['duration'] == 10.0  # 300 frames / 30 fps
        finally:
            os.unlink(temp_path)
    
    def test_extract_frames_invalid_interval(self, processor):
        """Test frame extraction with invalid interval"""
        with pytest.raises(ValueError):
            processor.extract_frames("dummy.mp4", interval=0)
        
        with pytest.raises(ValueError):
            processor.extract_frames("dummy.mp4", interval=-1)
    
    @patch('cv2.VideoCapture')
    def test_extract_frames_success(self, mock_videocapture, processor):
        """Test successful frame extraction"""
        # Mock video capture with test frames
        mock_cap = Mock()
        mock_cap.get.side_effect = lambda prop: {
            cv2.CAP_PROP_FPS: 30.0,
            cv2.CAP_PROP_FRAME_COUNT: 90,
            cv2.CAP_PROP_FRAME_WIDTH: 640,
            cv2.CAP_PROP_FRAME_HEIGHT: 480
        }.get(prop, 0)
        
        # Create test frames
        test_frames = [
            np.zeros((480, 640, 3), dtype=np.uint8) + i * 10
            for i in range(90)
        ]
        
        mock_cap.isOpened.return_value = True
        mock_cap.read.side_effect = [(True, frame) for frame in test_frames] + [(False, None)]
        mock_videocapture.return_value = mock_cap
        
        # Create temporary video file
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as f:
            temp_path = f.name
        
        try:
            frames = processor.extract_frames(temp_path, interval=1.0)
            
            # Should extract 30 frames (90 frames at 30 fps with 1.0s interval)
            assert len(frames) == 30
            
            # Check frame properties
            for frame in frames:
                assert isinstance(frame, np.ndarray)
                assert frame.shape == (480, 640, 3)
        finally:
            os.unlink(temp_path)


class TestErrorHandlerUnit:
    """Unit tests for ErrorHandler"""
    
    @pytest.fixture
    def handler(self):
        """Create error handler instance"""
        return ErrorHandler()
    
    def test_handler_initialization(self, handler):
        """Test error handler initialization"""
        assert handler is not None
        assert hasattr(handler, 'error_log')
        assert hasattr(handler, 'recovery_strategies')
    
    def test_categorize_error_value_error(self, handler):
        """Test ValueError categorization"""
        error = ValueError("Invalid input value")
        category = handler.categorize_error(error)
        assert category == 'validation'
    
    def test_categorize_error_file_not_found(self, handler):
        """Test FileNotFoundError categorization"""
        error = FileNotFoundError("File not found")
        category = handler.categorize_error(error)
        assert category == 'file_system'
    
    def test_categorize_error_permission_error(self, handler):
        """Test PermissionError categorization"""
        error = PermissionError("Permission denied")
        category = handler.categorize_error(error)
        assert category == 'security'
    
    def test_categorize_error_runtime_error(self, handler):
        """Test RuntimeError categorization"""
        error = RuntimeError("Runtime error occurred")
        category = handler.categorize_error(error)
        assert category == 'runtime'
    
    def test_categorize_error_unknown(self, handler):
        """Test unknown error categorization"""
        error = Exception("Unknown error")
        category = handler.categorize_error(error)
        assert category == 'unknown'
    
    def test_get_recovery_strategy_file_not_found(self, handler):
        """Test recovery strategy for file not found"""
        strategy = handler.get_recovery_strategy(FileNotFoundError("test.txt"))
        assert strategy is not None
        assert strategy['action'] == 'retry_with_alternative'
        assert 'description' in strategy
    
    def test_get_recovery_strategy_permission_error(self, handler):
        """Test recovery strategy for permission error"""
        strategy = handler.get_recovery_strategy(PermissionError("Access denied"))
        assert strategy is not None
        assert strategy['action'] == 'request_permissions'
        assert 'description' in strategy
    
    def test_handle_error_with_exception(self, handler):
        """Test handling an exception"""
        error = ValueError("Test error message")
        result = handler.handle_error(error)
        
        assert result['success'] is False
        assert 'error_id' in result
        assert result['message'] == "Test error message"
        assert result['category'] == 'validation'
        assert result['severity'] == 'error'
    
    def test_handle_error_with_context(self, handler):
        """Test handling error with context"""
        error = ValueError("Test error")
        context = {"file": "test.py", "line": 42, "function": "test_function"}
        
        result = handler.handle_error(error, context=context)
        
        assert result['context'] == context
        assert result['file'] == "test.py"
        assert result['line'] == 42
        assert result['function'] == "test_function"
    
    def test_error_logging(self, handler):
        """Test error logging"""
        error = ValueError("Test error for logging")
        
        # Handle error (should log it)
        result = handler.handle_error(error)
        
        # Check if error was logged
        assert len(handler.error_log) > 0
        logged_error = handler.error_log[-1]
        assert logged_error['error_id'] == result['error_id']
        assert logged_error['message'] == "Test error for logging"


class TestSecurityManagerUnit:
    """Unit tests for SecurityManager"""
    
    @pytest.fixture
    def manager(self):
        """Create security manager instance"""
        return SecurityManager()
    
    def test_manager_initialization(self, manager):
        """Test security manager initialization"""
        assert manager is not None
        assert hasattr(manager, 'validation_rules')
        assert hasattr(manager, 'permission_matrix')
    
    def test_validate_string_valid(self, manager):
        """Test valid string validation"""
        result = manager.validate_string("test_user", max_length=20, 
                                        allowed_chars="abcdefghijklmnopqrstuvwxyz_")
        assert result['valid'] is True
        assert result['sanitized'] == "test_user"
    
    def test_validate_string_too_long(self, manager):
        """Test string validation with too long string"""
        result = manager.validate_string("a" * 50, max_length=20)
        assert result['valid'] is False
        assert 'too_long' in result['errors']
    
    def test_validate_string_invalid_chars(self, manager):
        """Test string validation with invalid characters"""
        result = manager.validate_string("test@user", allowed_chars="abcdefghijklmnopqrstuvwxyz_")
        assert result['valid'] is False
        assert 'invalid_chars' in result['errors']
    
    def test_validate_string_xss_attempt(self, manager):
        """Test string validation with XSS attempt"""
        xss_string = "<script>alert('xss')</script>"
        result = manager.validate_string(xss_string)
        assert result['valid'] is False
        assert 'xss_detected' in result['errors']
    
    def test_validate_string_sql_injection(self, manager):
        """Test string validation with SQL injection attempt"""
        sql_string = "'; DROP TABLE users; --"
        result = manager.validate_string(sql_string)
        assert result['valid'] is False
        assert 'sql_injection' in result['errors']
    
    def test_validate_file_path_valid(self, manager):
        """Test valid file path validation"""
        result = manager.validate_file_path("/tmp/test.txt")
        assert result['valid'] is True
        assert result['sanitized'] == "/tmp/test.txt"
    
    def test_validate_file_path_traversal(self, manager):
        """Test file path validation with path traversal"""
        result = manager.validate_file_path("/tmp/../../../etc/passwd")
        assert result['valid'] is False
        assert 'path_traversal' in result['errors']
    
    def test_validate_file_path_windows_traversal(self, manager):
        """Test file path validation with Windows path traversal"""
        result = manager.validate_file_path("C:\\tmp\\..\\..\\windows\\system32")
        assert result['valid'] is False
        assert 'path_traversal' in result['errors']
    
    def test_validate_file_path_null_bytes(self, manager):
        """Test file path validation with null bytes"""
        result = manager.validate_file_path("/tmp/test\0.txt")
        assert result['valid'] is False
        assert 'null_bytes' in result['errors']
    
    def test_check_permission_admin(self, manager):
        """Test permission checking for admin"""
        user = {"role": "admin", "permissions": ["read", "write", "execute"]}
        
        assert manager.check_permission(user, "read") is True
        assert manager.check_permission(user, "write") is True
        assert manager.check_permission(user, "execute") is True
        assert manager.check_permission(user, "admin") is True
    
    def test_check_permission_user(self, manager):
        """Test permission checking for regular user"""
        user = {"role": "user", "permissions": ["read"]}
        
        assert manager.check_permission(user, "read") is True
        assert manager.check_permission(user, "write") is False
        assert manager.check_permission(user, "execute") is False
        assert manager.check_permission(user, "admin") is False
    
    def test_check_permission_guest(self, manager):
        """Test permission checking for guest"""
        user = {"role": "guest", "permissions": []}
        
        assert manager.check_permission(user, "read") is False
        assert manager.check_permission(user, "write") is False
        assert manager.check_permission(user, "execute") is False
    
    def test_check_permission_no_role(self, manager):
        """Test permission checking for user without role"""
        user = {}
        
        assert manager.check_permission(user, "read") is False
        assert manager.check_permission(user, "write") is False
        assert manager.check_permission(user, "execute") is False
    
    def test_sanitize_input_html(self, manager):
        """Test input sanitization for HTML"""
        html_input = "<script>alert('xss')</script><p>Safe content</p>"
        sanitized = manager.sanitize_input(html_input)
        
        assert "<script>" not in sanitized
        assert "Safe content" in sanitized
    
    def test_sanitize_input_sql(self, manager):
        """Test input sanitization for SQL"""
        sql_input = "SELECT * FROM users WHERE name = 'admin' OR '1'='1'"
        sanitized = manager.sanitize_input(sql_input)
        
        # SQL keywords should be escaped or removed
        assert "SELECT" not in sanitized or "SELECT" in sanitized and sanitized.count("'") < sql_input.count("'")
    
    def test_generate_api_key(self, manager):
        """Test API key generation"""
        api_key = manager.generate_api_key()
        
        assert len(api_key) == 32  # Default length
        assert api_key.isalnum()
        assert api_key.islower()


class TestMemoryManagerUnit:
    """Unit tests for MemoryManager"""
    
    @pytest.fixture
    def manager(self):
        """Create memory manager instance"""
        return MemoryManager(max_size="10MB")
    
    def test_manager_initialization(self, manager):
        """Test memory manager initialization"""
        assert manager is not None
        assert hasattr(manager, 'cache')
        assert hasattr(manager, 'max_size')
        assert manager.max_size == 10 * 1024 * 1024  # 10MB
    
    def test_cache_set_and_get(self, manager):
        """Test cache set and get operations"""
        test_data = np.zeros((100, 100, 3), dtype=np.uint8)
        key = "test_key"
        
        # Set data
        manager.cache.set(key, test_data)
        
        # Get data
        retrieved_data = manager.cache.get(key)
        
        assert retrieved_data is not None
        assert np.array_equal(retrieved_data, test_data)
    
    def test_cache_get_nonexistent(self, manager):
        """Test getting non-existent cache key"""
        result = manager.cache.get("nonexistent_key")
        assert result is None
    
    def test_cache_delete(self, manager):
        """Test cache delete operation"""
        test_data = np.zeros((100, 100, 3), dtype=np.uint8)
        key = "test_key"
        
        # Set data
        manager.cache.set(key, test_data)
        
        # Verify data exists
        assert manager.cache.get(key) is not None
        
        # Delete data
        manager.cache.delete(key)
        
        # Verify data is deleted
        assert manager.cache.get(key) is None
    
    def test_cache_clear(self, manager):
        """Test cache clear operation"""
        # Add multiple items
        for i in range(5):
            data = np.zeros((50, 50, 3), dtype=np.uint8) + i
            manager.cache.set(f"key_{i}", data)
        
        # Clear cache
        manager.cache.clear()
        
        # Verify all items are deleted
        for i in range(5):
            assert manager.cache.get(f"key_{i}") is None
    
    def test_cache_size_limit(self, manager):
        """Test cache size limit enforcement"""
        # Try to add data larger than max size
        large_data = np.zeros((2000, 2000, 3), dtype=np.uint8)  # ~12MB
        
        # Should not cache data that exceeds size limit
        manager.cache.set("large_data", large_data)
        
        # Data should not be cached
        assert manager.cache.get("large_data") is None
    
    def test_memory_usage_calculation(self, manager):
        """Test memory usage calculation"""
        # Add some data to cache
        for i in range(3):
            data = np.zeros((100, 100, 3), dtype=np.uint8)
            manager.cache.set(f"data_{i}", data)
        
        usage = manager.get_memory_usage()
        
        assert 'total' in usage
        assert 'used' in usage
        assert 'available' in usage
        assert 'percentage' in usage
        assert usage['total'] == manager.max_size
        assert usage['used'] > 0
        assert usage['percentage'] > 0
        assert usage['percentage'] <= 100


class TestStructuredLoggerUnit:
    """Unit tests for StructuredLogger"""
    
    @pytest.fixture
    def logger(self):
        """Create logger instance"""
        return StructuredLogger("test_logger")
    
    def test_logger_initialization(self, logger):
        """Test logger initialization"""
        assert logger is not None
        assert logger.name == "test_logger"
        assert hasattr(logger, 'logger')
    
    def test_info_logging(self, logger, caplog):
        """Test info logging"""
        logger.info("Test info message")
        
        assert len(caplog.records) == 1
        record = caplog.records[0]
        assert record.message == "Test info message"
        assert record.levelname == "INFO"
    
    def test_warning_logging(self, logger, caplog):
        """Test warning logging"""
        logger.warning("Test warning message")
        
        assert len(caplog.records) == 1
        record = caplog.records[0]
        assert record.message == "Test warning message"
        assert record.levelname == "WARNING"
    
    def test_error_logging(self, logger, caplog):
        """Test error logging"""
        logger.error("Test error message")
        
        assert len(caplog.records) == 1
        record = caplog.records[0]
        assert record.message == "Test error message"
        assert record.levelname == "ERROR"
    
    def test_debug_logging(self, logger, caplog):
        """Test debug logging"""
        logger.debug("Test debug message")
        
        assert len(caplog.records) == 1
        record = caplog.records[0]
        assert record.message == "Test debug message"
        assert record.levelname == "DEBUG"
    
    def test_structured_data_logging(self, logger, caplog):
        """Test logging with structured data"""
        extra_data = {"user_id": 123, "action": "login", "ip": "192.168.1.1"}
        logger.info("User action", extra=extra_data)
        
        assert len(caplog.records) == 1
        record = caplog.records[0]
        assert record.message == "User action"
        assert hasattr(record, 'user_id')
        assert record.user_id == 123
        assert hasattr(record, 'action')
        assert record.action == "login"
        assert hasattr(record, 'ip')
        assert record.ip == "192.168.1.1"
    
    def test_exception_logging(self, logger, caplog):
        """Test exception logging"""
        try:
            raise ValueError("Test exception")
        except Exception as e:
            logger.exception("Exception occurred", exc_info=e)
        
        assert len(caplog.records) == 1
        record = caplog.records[0]
        assert record.message == "Exception occurred"
        assert record.levelname == "ERROR"
        assert hasattr(record, 'exc_info')
    
    def test_performance_timer(self, logger, caplog):
        """Test performance timer"""
        import time
        
        with logger.performance_timer("test_operation"):
            time.sleep(0.01)
        
        # Check if performance log was created
        performance_logs = [r for r in caplog.records if hasattr(r, 'operation')]
        assert len(performance_logs) > 0
        
        perf_log = performance_logs[0]
        assert perf_log.operation == "test_operation"
        assert hasattr(perf_log, 'duration')
        assert perf_log.duration > 0
    
    def test_log_level_filtering(self, logger, caplog):
        """Test log level filtering"""
        # Set logger to WARNING level
        logger.logger.setLevel("WARNING")
        
        logger.debug("Debug message")  # Should not appear
        logger.info("Info message")    # Should not appear
        logger.warning("Warning message")  # Should appear
        logger.error("Error message")    # Should appear
        
        # Only WARNING and ERROR should appear
        assert len(caplog.records) == 2
        levels = [record.levelname for record in caplog.records]
        assert "WARNING" in levels
        assert "ERROR" in levels
        assert "DEBUG" not in levels
        assert "INFO" not in levels


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])