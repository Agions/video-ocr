"""
Error handling system for VisionSub
"""
import logging
import time
import traceback
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


class ErrorSeverity(Enum):
    """Error severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ErrorCategory(Enum):
    """Error categories"""
    VIDEO_PROCESSING = "video_processing"
    OCR_PROCESSING = "ocr_processing"
    FILE_IO = "file_io"
    CONFIGURATION = "configuration"
    USER_INTERFACE = "user_interface"
    SYSTEM = "system"
    SECURITY = "security"


@dataclass
class ErrorContext:
    """Context information for error handling"""
    operation: str
    file_path: Optional[str] = None
    frame_number: Optional[int] = None
    timestamp: Optional[float] = None
    additional_info: Optional[Dict[str, Any]] = None


@dataclass
class ErrorInfo:
    """Detailed error information"""
    error_type: str
    message: str
    severity: ErrorSeverity
    category: ErrorCategory
    context: ErrorContext
    traceback: Optional[str] = None
    timestamp: float = 0.0
    user_message: Optional[str] = None
    suggestions: Optional[List[str]] = None

    def __post_init__(self):
        if self.timestamp == 0.0:
            self.timestamp = time.time()

        # Generate user-friendly message if not provided
        if self.user_message is None:
            self.user_message = self._generate_user_message()

        # Generate suggestions if not provided
        if self.suggestions is None:
            self.suggestions = self._generate_suggestions()

    def _generate_user_message(self) -> str:
        """Generate user-friendly error message"""
        base_messages = {
            ErrorCategory.VIDEO_PROCESSING: "视频处理出错",
            ErrorCategory.OCR_PROCESSING: "文字识别出错",
            ErrorCategory.FILE_IO: "文件操作出错",
            ErrorCategory.CONFIGURATION: "配置错误",
            ErrorCategory.USER_INTERFACE: "界面错误",
            ErrorCategory.SYSTEM: "系统错误"
        }

        base = base_messages.get(self.category, "发生错误")
        return f"{base}：{self.message}"

    def _generate_suggestions(self) -> List[str]:
        """Generate troubleshooting suggestions"""
        suggestions = {
            ErrorCategory.VIDEO_PROCESSING: [
                "检查视频文件是否完整",
                "尝试使用其他视频文件",
                "确保视频格式受支持"
            ],
            ErrorCategory.OCR_PROCESSING: [
                "检查字幕区域是否正确设置",
                "尝试调整OCR参数",
                "确保视频质量足够清晰"
            ],
            ErrorCategory.FILE_IO: [
                "检查文件路径是否正确",
                "确保有足够的磁盘空间",
                "检查文件权限"
            ],
            ErrorCategory.CONFIGURATION: [
                "检查配置文件格式",
                "恢复默认设置",
                "重新启动应用程序"
            ],
            ErrorCategory.USER_INTERFACE: [
                "重新启动应用程序",
                "检查系统资源使用情况",
                "更新显卡驱动程序"
            ],
            ErrorCategory.SYSTEM: [
                "重新启动应用程序",
                "检查系统日志",
                "联系技术支持"
            ]
        }

        return suggestions.get(self.category, ["请重试操作"])


class VisionSubError(Exception):
    """Base exception class for VisionSub"""

    def __init__(
        self,
        message: str,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        category: ErrorCategory = ErrorCategory.SYSTEM,
        context: Optional[ErrorContext] = None
    ):
        super().__init__(message)
        self.message = message
        self.severity = severity
        self.category = category
        self.context = context or ErrorContext("unknown")

    def to_error_info(self) -> ErrorInfo:
        """Convert to ErrorInfo object"""
        return ErrorInfo(
            error_type=self.__class__.__name__,
            message=self.message,
            severity=self.severity,
            category=self.category,
            context=self.context,
            traceback=traceback.format_exc()
        )


class VideoProcessingError(VisionSubError):
    """Video processing related errors"""
    def __init__(self, message: str, context: Optional[ErrorContext] = None):
        super().__init__(
            message,
            severity=ErrorSeverity.HIGH,
            category=ErrorCategory.VIDEO_PROCESSING,
            context=context
        )


class OCRError(VisionSubError):
    """OCR processing related errors"""
    def __init__(self, message: str, context: Optional[ErrorContext] = None):
        super().__init__(
            message,
            severity=ErrorSeverity.MEDIUM,
            category=ErrorCategory.OCR_PROCESSING,
            context=context
        )


class FileIOError(VisionSubError):
    """File I/O related errors"""
    def __init__(self, message: str, context: Optional[ErrorContext] = None):
        super().__init__(
            message,
            severity=ErrorSeverity.HIGH,
            category=ErrorCategory.FILE_IO,
            context=context
        )


class ConfigurationError(VisionSubError):
    """Configuration related errors"""
    def __init__(self, message: str, context: Optional[ErrorContext] = None):
        super().__init__(
            message,
            severity=ErrorSeverity.MEDIUM,
            category=ErrorCategory.CONFIGURATION,
            context=context
        )


class SecurityError(VisionSubError):
    """Security related errors"""
    def __init__(self, message: str, context: Optional[ErrorContext] = None):
        super().__init__(
            message,
            severity=ErrorSeverity.HIGH,
            category=ErrorCategory.SECURITY,
            context=context
        )


class ErrorHandler:
    """Centralized error handling system"""

    def __init__(self):
        self.error_callbacks: Dict[ErrorCategory, List[Callable]] = {}
        self.error_history: List[ErrorInfo] = []
        self.max_history_size = 100

    def register_callback(
        self,
        category: ErrorCategory,
        callback: Callable[[ErrorInfo], None]
    ):
        """Register error callback for specific category"""
        if category not in self.error_callbacks:
            self.error_callbacks[category] = []
        self.error_callbacks[category].append(callback)

    def handle_error(
        self,
        error: Exception,
        context: Optional[ErrorContext] = None
    ) -> ErrorInfo:
        """Handle error with appropriate strategies"""
        # Convert to ErrorInfo
        if isinstance(error, VisionSubError):
            error_info = error.to_error_info()
        else:
            # Generic exception
            error_info = ErrorInfo(
                error_type=error.__class__.__name__,
                message=str(error),
                severity=ErrorSeverity.HIGH,
                category=ErrorCategory.SYSTEM,
                context=context or ErrorContext("unknown"),
                traceback=traceback.format_exc()
            )

        # Update context if provided
        if context:
            error_info.context = context

        # Log error
        self._log_error(error_info)

        # Add to history
        self._add_to_history(error_info)

        # Trigger callbacks
        self._trigger_callbacks(error_info)

        # Attempt recovery if possible
        self._attempt_recovery(error_info)

        return error_info

    def _log_error(self, error_info: ErrorInfo):
        """Log error with appropriate level"""
        log_message = f"{error_info.category.value}: {error_info.message}"

        if error_info.severity == ErrorSeverity.CRITICAL:
            logger.critical(log_message, extra={'error_info': error_info})
        elif error_info.severity == ErrorSeverity.HIGH:
            logger.error(log_message, extra={'error_info': error_info})
        elif error_info.severity == ErrorSeverity.MEDIUM:
            logger.warning(log_message, extra={'error_info': error_info})
        else:
            logger.info(log_message, extra={'error_info': error_info})

    def _add_to_history(self, error_info: ErrorInfo):
        """Add error to history"""
        self.error_history.append(error_info)

        # Maintain history size
        if len(self.error_history) > self.max_history_size:
            self.error_history.pop(0)

    def _trigger_callbacks(self, error_info: ErrorInfo):
        """Trigger registered callbacks"""
        callbacks = self.error_callbacks.get(error_info.category, [])

        for callback in callbacks:
            try:
                callback(error_info)
            except Exception as e:
                logger.error(f"Error callback failed: {e}")

    def _attempt_recovery(self, error_info: ErrorInfo):
        """Attempt to recover from error"""
        recovery_strategies = {
            ErrorCategory.VIDEO_PROCESSING: self._recover_video_error,
            ErrorCategory.OCR_PROCESSING: self._recover_ocr_error,
            ErrorCategory.FILE_IO: self._recover_file_error,
            ErrorCategory.CONFIGURATION: self._recover_config_error,
        }

        strategy = recovery_strategies.get(error_info.category)
        if strategy:
            try:
                success = strategy(error_info)
                if success:
                    logger.info(f"Successfully recovered from {error_info.category.value} error")
            except Exception as e:
                logger.error(f"Recovery attempt failed: {e}")

    def _recover_video_error(self, error_info: ErrorInfo) -> bool:
        """Attempt to recover from video processing error"""
        # Implementation would depend on specific error
        # For now, just log the attempt
        logger.info("Attempting video error recovery...")
        return False

    def _recover_ocr_error(self, error_info: ErrorInfo) -> bool:
        """Attempt to recover from OCR error"""
        # Try fallback OCR engine
        logger.info("Attempting OCR error recovery...")
        return False

    def _recover_file_error(self, error_info: ErrorInfo) -> bool:
        """Attempt to recover from file I/O error"""
        # Check file permissions, try alternative paths
        logger.info("Attempting file I/O error recovery...")
        return False

    def _recover_config_error(self, error_info: ErrorInfo) -> bool:
        """Attempt to recover from configuration error"""
        # Reset to default configuration
        logger.info("Attempting configuration error recovery...")
        return False

    def get_error_stats(self) -> Dict[str, Any]:
        """Get error statistics"""
        if not self.error_history:
            return {'total_errors': 0}

        # Count errors by category
        category_counts = {}
        severity_counts = {}

        for error in self.error_history:
            category_counts[error.category.value] = category_counts.get(error.category.value, 0) + 1
            severity_counts[error.severity.value] = severity_counts.get(error.severity.value, 0) + 1

        return {
            'total_errors': len(self.error_history),
            'by_category': category_counts,
            'by_severity': severity_counts,
            'recent_errors': len([e for e in self.error_history if time.time() - e.timestamp < 3600])
        }

    def clear_history(self):
        """Clear error history"""
        self.error_history.clear()


# Global error handler instance
error_handler = ErrorHandler()
