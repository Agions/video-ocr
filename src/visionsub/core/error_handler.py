"""
Enhanced error handling with detailed error information and recovery strategies
"""
import asyncio
import logging
import traceback
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Callable, Union
from pathlib import Path

from ..models.config import AppConfig, SecurityConfig


class ErrorSeverity(Enum):
    """Error severity levels"""
    LOW = "low"          # Non-critical, can continue
    MEDIUM = "medium"    # May affect functionality but can recover
    HIGH = "high"        # Critical, requires immediate attention
    CRITICAL = "critical"  # System cannot continue


class ErrorCategory(Enum):
    """Error categories"""
    FILE_IO = "file_io"
    NETWORK = "network"
    PROCESSING = "processing"
    CONFIGURATION = "configuration"
    MEMORY = "memory"
    PERMISSION = "permission"
    VALIDATION = "validation"
    EXTERNAL_SERVICE = "external_service"
    UNKNOWN = "unknown"


class RecoveryAction(Enum):
    """Recovery actions"""
    RETRY = "retry"
    FALLBACK = "fallback"
    SKIP = "skip"
    ROLLBACK = "rollback"
    RESTART = "restart"
    NOTIFY = "notify"
    TERMINATE = "terminate"


@dataclass
class ErrorContext:
    """Context information for errors"""
    component: str
    operation: str
    input_data: Optional[Dict[str, Any]] = None
    environment: Optional[Dict[str, Any]] = None
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    trace_id: Optional[str] = None


@dataclass
class ErrorDetails:
    """Detailed error information"""
    error_id: str
    error_type: str
    error_message: str
    severity: ErrorSeverity
    category: ErrorCategory
    context: ErrorContext
    timestamp: datetime
    stack_trace: Optional[str] = None
    recovery_suggested: Optional[RecoveryAction] = None
    recovery_details: Optional[str] = None
    additional_info: Dict[str, Any] = field(default_factory=dict)


class VisionSubError(Exception):
    """Base exception class for VisionSub with enhanced error information"""
    
    def __init__(
        self,
        message: str,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        category: ErrorCategory = ErrorCategory.UNKNOWN,
        context: Optional[ErrorContext] = None,
        recovery_action: Optional[RecoveryAction] = None,
        recovery_details: Optional[str] = None,
        original_error: Optional[Exception] = None
    ):
        super().__init__(message)
        self.message = message
        self.severity = severity
        self.category = category
        self.context = context or ErrorContext("unknown", "unknown")
        self.recovery_action = recovery_action
        self.recovery_details = recovery_details
        self.original_error = original_error
        self.timestamp = datetime.now()
        self.error_id = f"vs_{int(self.timestamp.timestamp() * 1000)}_{id(self)}"
        
        # Capture stack trace
        if original_error:
            self.stack_trace = traceback.format_exception(
                type(original_error), 
                original_error, 
                original_error.__traceback__
            )
        else:
            self.stack_trace = traceback.format_stack()


class ErrorHandler:
    """Centralized error handling with recovery strategies"""
    
    def __init__(self, config: AppConfig):
        self.config = config
        self.error_history: List[ErrorDetails] = []
        self.error_callbacks: Dict[ErrorCategory, List[Callable]] = {}
        self.recovery_strategies: Dict[ErrorCategory, Callable] = {}
        
        # Initialize recovery strategies
        self._init_recovery_strategies()
        
        # Error statistics
        self.error_stats = {
            'total_errors': 0,
            'errors_by_category': {},
            'errors_by_severity': {},
            'recovery_success_rate': 0.0,
            'recent_errors': []
        }
        
        logger = logging.getLogger(__name__)
        self.logger = logger

    def _init_recovery_strategies(self):
        """Initialize recovery strategies for different error categories"""
        self.recovery_strategies = {
            ErrorCategory.FILE_IO: self._recover_file_io,
            ErrorCategory.NETWORK: self._recover_network,
            ErrorCategory.PROCESSING: self._recover_processing,
            ErrorCategory.CONFIGURATION: self._recover_configuration,
            ErrorCategory.MEMORY: self._recover_memory,
            ErrorCategory.PERMISSION: self._recover_permission,
            ErrorCategory.VALIDATION: self._recover_validation,
            ErrorCategory.EXTERNAL_SERVICE: self._recover_external_service,
        }

    def register_error_callback(
        self, 
        category: ErrorCategory, 
        callback: Callable[[ErrorDetails], None]
    ):
        """Register callback for specific error category"""
        if category not in self.error_callbacks:
            self.error_callbacks[category] = []
        self.error_callbacks[category].append(callback)

    async def handle_error(
        self, 
        error: Exception, 
        context: ErrorContext,
        auto_recover: bool = True
    ) -> ErrorDetails:
        """
        Handle error with automatic recovery and notification
        
        Args:
            error: The exception that occurred
            context: Error context information
            auto_recover: Whether to attempt automatic recovery
            
        Returns:
            ErrorDetails: Detailed error information
        """
        # Convert to VisionSubError if needed
        if not isinstance(error, VisionSubError):
            vision_sub_error = self._convert_to_vision_sub_error(error, context)
        else:
            vision_sub_error = error
            
        # Create error details
        error_details = ErrorDetails(
            error_id=vision_sub_error.error_id,
            error_type=type(error).__name__,
            error_message=str(error),
            severity=vision_sub_error.severity,
            category=vision_sub_error.category,
            context=vision_sub_error.context,
            timestamp=vision_sub_error.timestamp,
            stack_trace=''.join(vision_sub_error.stack_trace) if vision_sub_error.stack_trace else None,
            recovery_suggested=vision_sub_error.recovery_action,
            recovery_details=vision_sub_error.recovery_details
        )
        
        # Log error
        self._log_error(error_details)
        
        # Update statistics
        self._update_error_stats(error_details)
        
        # Store in history
        self.error_history.append(error_details)
        
        # Keep only recent errors in memory
        if len(self.error_history) > 1000:
            self.error_history = self.error_history[-1000:]
        
        # Notify callbacks
        await self._notify_callbacks(error_details)
        
        # Attempt recovery if enabled
        if auto_recover and vision_sub_error.recovery_action:
            recovery_result = await self._attempt_recovery(error_details)
            error_details.additional_info['recovery_attempted'] = True
            error_details.additional_info['recovery_success'] = recovery_result
        
        return error_details

    def _convert_to_vision_sub_error(self, error: Exception, context: ErrorContext) -> VisionSubError:
        """Convert generic exception to VisionSubError with appropriate categorization"""
        error_type = type(error).__name__
        error_message = str(error)
        
        # Categorize error based on type and message
        if isinstance(error, (FileNotFoundError, PermissionError, IOError)):
            category = ErrorCategory.FILE_IO
            severity = ErrorSeverity.HIGH
            recovery = RecoveryAction.FALLBACK
        elif isinstance(error, (ConnectionError, TimeoutError)):
            category = ErrorCategory.NETWORK
            severity = ErrorSeverity.MEDIUM
            recovery = RecoveryAction.RETRY
        elif isinstance(error, (MemoryError, OverflowError)):
            category = ErrorCategory.MEMORY
            severity = ErrorSeverity.CRITICAL
            recovery = RecoveryAction.RESTART
        elif isinstance(error, (ValueError, TypeError, AttributeError)):
            category = ErrorCategory.VALIDATION
            severity = ErrorSeverity.MEDIUM
            recovery = RecoveryAction.SKIP
        elif "config" in error_message.lower():
            category = ErrorCategory.CONFIGURATION
            severity = ErrorSeverity.HIGH
            recovery = RecoveryAction.ROLLBACK
        else:
            category = ErrorCategory.UNKNOWN
            severity = ErrorSeverity.MEDIUM
            recovery = RecoveryAction.NOTIFY
        
        return VisionSubError(
            message=error_message,
            severity=severity,
            category=category,
            context=context,
            recovery_action=recovery,
            original_error=error
        )

    def _log_error(self, error_details: ErrorDetails):
        """Log error with appropriate level"""
        log_message = f"[{error_details.error_id}] {error_details.context.component}.{error_details.context.operation}: {error_details.error_message}"
        
        if error_details.severity == ErrorSeverity.CRITICAL:
            self.logger.critical(log_message, extra={'error_details': error_details})
        elif error_details.severity == ErrorSeverity.HIGH:
            self.logger.error(log_message, extra={'error_details': error_details})
        elif error_details.severity == ErrorSeverity.MEDIUM:
            self.logger.warning(log_message, extra={'error_details': error_details})
        else:
            self.logger.info(log_message, extra={'error_details': error_details})

    def _update_error_stats(self, error_details: ErrorDetails):
        """Update error statistics"""
        self.error_stats['total_errors'] += 1
        
        # Update by category
        category = error_details.category.value
        self.error_stats['errors_by_category'][category] = self.error_stats['errors_by_category'].get(category, 0) + 1
        
        # Update by severity
        severity = error_details.severity.value
        self.error_stats['errors_by_severity'][severity] = self.error_stats['errors_by_severity'].get(severity, 0) + 1
        
        # Update recent errors
        self.error_stats['recent_errors'].append(error_details)
        if len(self.error_stats['recent_errors']) > 100:
            self.error_stats['recent_errors'] = self.error_stats['recent_errors'][-100:]
        
        # Calculate recovery success rate
        successful_recoveries = sum(
            1 for error in self.error_history 
            if error.additional_info.get('recovery_success', False)
        )
        total_recoveries = sum(
            1 for error in self.error_history 
            if error.additional_info.get('recovery_attempted', False)
        )
        
        if total_recoveries > 0:
            self.error_stats['recovery_success_rate'] = successful_recoveries / total_recoveries

    async def _notify_callbacks(self, error_details: ErrorDetails):
        """Notify registered callbacks for error category"""
        callbacks = self.error_callbacks.get(error_details.category, [])
        
        for callback in callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(error_details)
                else:
                    callback(error_details)
            except Exception as e:
                self.logger.error(f"Error callback failed: {e}")

    async def _attempt_recovery(self, error_details: ErrorDetails) -> bool:
        """Attempt recovery based on error category"""
        recovery_strategy = self.recovery_strategies.get(error_details.category)
        
        if recovery_strategy:
            try:
                return await recovery_strategy(error_details)
            except Exception as e:
                self.logger.error(f"Recovery strategy failed: {e}")
                return False
        
        return False

    # Recovery strategies
    async def _recover_file_io(self, error_details: ErrorDetails) -> bool:
        """Recover from file I/O errors"""
        context = error_details.context
        
        # Check if file exists
        if 'file_path' in (context.input_data or {}):
            file_path = Path(context.input_data['file_path'])
            if not file_path.exists():
                # Try to find similar file or create default
                self.logger.warning(f"File not found: {file_path}")
                return False
        
        # Check permissions
        if 'file_path' in (context.input_data or {}):
            file_path = Path(context.input_data['file_path'])
            if file_path.exists() and not os.access(file_path, os.R_OK):
                self.logger.warning(f"Permission denied: {file_path}")
                return False
        
        return False

    async def _recover_network(self, error_details: ErrorDetails) -> bool:
        """Recover from network errors"""
        # Retry with exponential backoff would be implemented here
        self.logger.info("Attempting network recovery...")
        # For now, just log and return False
        return False

    async def _recover_processing(self, error_details: ErrorDetails) -> bool:
        """Recover from processing errors"""
        context = error_details.context
        
        # If OCR processing failed, try with different settings
        if context.operation == 'ocr_processing':
            self.logger.info("Attempting OCR processing recovery...")
            # Could try different OCR engine or settings
            return False
        
        return False

    async def _recover_configuration(self, error_details: ErrorDetails) -> bool:
        """Recover from configuration errors"""
        self.logger.info("Attempting configuration recovery...")
        # Could reset to default configuration
        return False

    async def _recover_memory(self, error_details: ErrorDetails) -> bool:
        """Recover from memory errors"""
        self.logger.warning("Memory error detected, attempting cleanup...")
        # Clear caches, release memory
        import gc
        gc.collect()
        return True

    async def _recover_permission(self, error_details: ErrorDetails) -> bool:
        """Recover from permission errors"""
        self.logger.warning("Permission error detected")
        # Could request elevated permissions or use alternative paths
        return False

    async def _recover_validation(self, error_details: ErrorDetails) -> bool:
        """Recover from validation errors"""
        self.logger.info("Validation error, attempting to use defaults...")
        # Use default values or skip validation
        return True

    async def _recover_external_service(self, error_details: ErrorDetails) -> bool:
        """Recover from external service errors"""
        self.logger.info("External service error, attempting fallback...")
        # Use fallback service or cached data
        return False

    def get_error_history(
        self, 
        category: Optional[ErrorCategory] = None,
        severity: Optional[ErrorSeverity] = None,
        limit: int = 100
    ) -> List[ErrorDetails]:
        """Get filtered error history"""
        filtered_errors = self.error_history
        
        if category:
            filtered_errors = [e for e in filtered_errors if e.category == category]
        
        if severity:
            filtered_errors = [e for e in filtered_errors if e.severity == severity]
        
        return filtered_errors[-limit:]

    def get_error_stats(self) -> Dict[str, Any]:
        """Get error statistics"""
        return self.error_stats.copy()

    def clear_error_history(self):
        """Clear error history (for privacy or maintenance)"""
        self.error_history.clear()
        self.error_stats['recent_errors'].clear()
        self.logger.info("Error history cleared")

    def create_error_report(self) -> Dict[str, Any]:
        """Create comprehensive error report"""
        return {
            'generated_at': datetime.now().isoformat(),
            'total_errors': self.error_stats['total_errors'],
            'errors_by_category': self.error_stats['errors_by_category'],
            'errors_by_severity': self.error_stats['errors_by_severity'],
            'recovery_success_rate': self.error_stats['recovery_success_rate'],
            'recent_errors': [
                {
                    'error_id': error.error_id,
                    'timestamp': error.timestamp.isoformat(),
                    'component': error.context.component,
                    'operation': error.context.operation,
                    'message': error.error_message,
                    'severity': error.severity.value,
                    'category': error.category.value,
                    'recovery_attempted': error.additional_info.get('recovery_attempted', False),
                    'recovery_success': error.additional_info.get('recovery_success', False)
                }
                for error in self.error_stats['recent_errors']
            ]
        }


# Decorator for automatic error handling
def handle_errors(
    component: str,
    operation: str,
    severity: ErrorSeverity = ErrorSeverity.MEDIUM,
    auto_recover: bool = True
):
    """Decorator for automatic error handling"""
    def decorator(func):
        async def async_wrapper(*args, **kwargs):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                context = ErrorContext(component, operation, {
                    'args': str(args)[:200],  # Limit length
                    'kwargs': str(kwargs)[:200]
                })
                
                # Get error handler from app or create default
                error_handler = kwargs.get('error_handler')
                if error_handler:
                    await error_handler.handle_error(e, context, auto_recover)
                else:
                    logging.error(f"Unhandled error in {component}.{operation}: {e}")
                    raise
                
                # Re-raise if critical
                if severity in [ErrorSeverity.CRITICAL, ErrorSeverity.HIGH]:
                    raise
        
        def sync_wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                context = ErrorContext(component, operation, {
                    'args': str(args)[:200],
                    'kwargs': str(kwargs)[:200]
                })
                
                error_handler = kwargs.get('error_handler')
                if error_handler:
                    # Run async error handler in event loop
                    try:
                        loop = asyncio.get_event_loop()
                        loop.run_until_complete(
                            error_handler.handle_error(e, context, auto_recover)
                        )
                    except RuntimeError:
                        # No event loop, log error
                        logging.error(f"Unhandled error in {component}.{operation}: {e}")
                else:
                    logging.error(f"Unhandled error in {component}.{operation}: {e}")
                    raise
                
                if severity in [ErrorSeverity.CRITICAL, ErrorSeverity.HIGH]:
                    raise
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator