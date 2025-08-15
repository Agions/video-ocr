"""
Security module for VisionSub 2.0 - Input validation, sanitization, and access control
"""
import os
import re
import hashlib
import mimetypes
from pathlib import Path
from typing import Optional, List, Dict, Any, Union
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class SecurityLevel(Enum):
    """Security levels for different operations"""
    LOW = "low"      # Basic validation
    MEDIUM = "medium"  # Enhanced validation
    HIGH = "high"     # Strict validation
    CRITICAL = "critical"  # Maximum security


@dataclass
class SecurityPolicy:
    """Security policy configuration"""
    max_file_size: int = 1024 * 1024 * 1024  # 1GB default
    allowed_video_formats: List[str] = None
    allowed_image_formats: List[str] = None
    max_filename_length: int = 255
    sanitize_paths: bool = True
    validate_mime_types: bool = True
    security_level: SecurityLevel = SecurityLevel.MEDIUM

    def __post_init__(self):
        if self.allowed_video_formats is None:
            self.allowed_video_formats = [
                '.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm', '.m4v'
            ]
        if self.allowed_image_formats is None:
            self.allowed_image_formats = [
                '.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif', '.webp'
            ]


class SecurityValidator:
    """Security validator for file operations and user inputs"""

    def __init__(self, policy: SecurityPolicy = None):
        self.policy = policy or SecurityPolicy()
        self._blocked_patterns = [
            r'\.\.',  # Directory traversal
            r'~$',   # Backup files
            r'\.bak$',  # Backup files
            r'\.tmp$',  # Temporary files
            r'\.exe$',  # Executables
            r'\.scr$',  # Screensavers
            r'\.bat$',  # Batch files
            r'\.cmd$',  # Command files
            r'\.pif$',  # Program information files
        ]

    def validate_file_path(self, file_path: Union[str, Path]) -> bool:
        """
        Validate file path for security issues

        Args:
            file_path: Path to validate

        Returns:
            bool: True if path is valid and safe
        """
        try:
            path = Path(file_path)

            # Check for empty path
            if not path.name:
                logger.warning("Empty file path")
                return False

            # Check filename length
            if len(path.name) > self.policy.max_filename_length:
                logger.warning(f"Filename too long: {path.name}")
                return False

            # Check for blocked patterns
            filename_lower = path.name.lower()
            for pattern in self._blocked_patterns:
                if re.search(pattern, filename_lower):
                    logger.warning(f"Blocked pattern in filename: {path.name}")
                    return False

            # Check path traversal attempts
            if self.policy.sanitize_paths:
                try:
                    # Resolve to absolute path and check if it's safe
                    resolved_path = path.resolve()
                    if '..' in str(resolved_path):
                        logger.warning(f"Path traversal attempt: {file_path}")
                        return False
                except (OSError, ValueError):
                    logger.warning(f"Invalid path: {file_path}")
                    return False

            return True

        except Exception as e:
            logger.error(f"Error validating file path {file_path}: {e}")
            return False

    def validate_video_file(self, file_path: Union[str, Path]) -> bool:
        """
        Validate video file for security and format support

        Args:
            file_path: Path to video file

        Returns:
            bool: True if file is valid and safe
        """
        # First validate basic path security
        if not self.validate_file_path(file_path):
            return False

        try:
            path = Path(file_path)

            # Check if file exists
            if not path.exists():
                logger.warning(f"File does not exist: {file_path}")
                return False

            # Check if it's actually a file
            if not path.is_file():
                logger.warning(f"Path is not a file: {file_path}")
                return False

            # Check file size
            file_size = path.stat().st_size
            if file_size > self.policy.max_file_size:
                logger.warning(f"File too large: {file_size} bytes")
                return False

            # Check file extension
            ext = path.suffix.lower()
            if ext not in self.policy.allowed_video_formats:
                logger.warning(f"Unsupported video format: {ext}")
                return False

            # Validate MIME type if enabled
            if self.policy.validate_mime_types:
                mime_type, _ = mimetypes.guess_type(str(path))
                if not mime_type or not mime_type.startswith('video/'):
                    logger.warning(f"Invalid MIME type: {mime_type}")
                    return False

            return True

        except Exception as e:
            logger.error(f"Error validating video file {file_path}: {e}")
            return False

    def validate_image_file(self, file_path: Union[str, Path]) -> bool:
        """
        Validate image file for security and format support

        Args:
            file_path: Path to image file

        Returns:
            bool: True if file is valid and safe
        """
        # First validate basic path security
        if not self.validate_file_path(file_path):
            return False

        try:
            path = Path(file_path)

            # Check if file exists
            if not path.exists():
                logger.warning(f"File does not exist: {file_path}")
                return False

            # Check if it's actually a file
            if not path.is_file():
                logger.warning(f"Path is not a file: {file_path}")
                return False

            # Check file size
            file_size = path.stat().st_size
            if file_size > self.policy.max_file_size:
                logger.warning(f"File too large: {file_size} bytes")
                return False

            # Check file extension
            ext = path.suffix.lower()
            if ext not in self.policy.allowed_image_formats:
                logger.warning(f"Unsupported image format: {ext}")
                return False

            # Validate MIME type if enabled
            if self.policy.validate_mime_types:
                mime_type, _ = mimetypes.guess_type(str(path))
                if not mime_type or not mime_type.startswith('image/'):
                    logger.warning(f"Invalid MIME type: {mime_type}")
                    return False

            return True

        except Exception as e:
            logger.error(f"Error validating image file {file_path}: {e}")
            return False

    def sanitize_input(self, input_string: str, max_length: int = 1000) -> str:
        """
        Sanitize user input strings

        Args:
            input_string: Input string to sanitize
            max_length: Maximum allowed length

        Returns:
            str: Sanitized string
        """
        if not isinstance(input_string, str):
            return ""

        # Truncate to max length
        sanitized = input_string[:max_length]

        # Remove potentially dangerous characters
        sanitized = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', '', sanitized)

        # Normalize whitespace
        sanitized = re.sub(r'\s+', ' ', sanitized).strip()

        return sanitized

    def validate_config_value(self, key: str, value: Any) -> bool:
        """
        Validate configuration values for security

        Args:
            key: Configuration key
            value: Configuration value

        Returns:
            bool: True if value is valid
        """
        try:
            # Validate numeric ranges
            if key.endswith(('_threshold', '_interval', '_size', '_timeout')):
                if isinstance(value, (int, float)):
                    if value < 0:
                        logger.warning(f"Negative value for {key}: {value}")
                        return False
                    if key.endswith('_threshold') and not (0 <= value <= 1):
                        logger.warning(f"Threshold out of range for {key}: {value}")
                        return False
                else:
                    logger.warning(f"Invalid numeric type for {key}: {type(value)}")
                    return False

            # Validate string values
            elif isinstance(value, str):
                if len(value) > 1000:  # Reasonable limit for config strings
                    logger.warning(f"Config value too long for {key}")
                    return False

                # Check for potentially dangerous patterns
                dangerous_patterns = [
                    r'<script', r'javascript:', r'vbscript:', r'onload=', r'onerror=',
                    r'<iframe', r'<object', r'<embed', r'file://', r'ftp://'
                ]
                for pattern in dangerous_patterns:
                    if re.search(pattern, value.lower()):
                        logger.warning(f"Dangerous pattern in config {key}: {value}")
                        return False

            return True

        except Exception as e:
            logger.error(f"Error validating config {key}: {e}")
            return False

    def generate_file_hash(self, file_path: Union[str, Path]) -> Optional[str]:
        """
        Generate SHA-256 hash of a file for integrity verification

        Args:
            file_path: Path to file

        Returns:
            Optional[str]: SHA-256 hash or None if error
        """
        try:
            if not self.validate_file_path(file_path):
                return None

            path = Path(file_path)
            if not path.exists():
                return None

            sha256_hash = hashlib.sha256()
            with open(path, 'rb') as f:
                # Read file in chunks to handle large files
                for chunk in iter(lambda: f.read(4096), b""):
                    sha256_hash.update(chunk)

            return sha256_hash.hexdigest()

        except Exception as e:
            logger.error(f"Error generating file hash for {file_path}: {e}")
            return None

    def check_directory_permissions(self, directory: Union[str, Path]) -> bool:
        """
        Check if directory has appropriate permissions

        Args:
            directory: Directory path to check

        Returns:
            bool: True if permissions are appropriate
        """
        try:
            path = Path(directory)

            # Check if directory exists
            if not path.exists():
                logger.warning(f"Directory does not exist: {directory}")
                return False

            # Check if it's a directory
            if not path.is_dir():
                logger.warning(f"Path is not a directory: {directory}")
                return False

            # Check read/write permissions
            if not os.access(path, os.R_OK | os.W_OK):
                logger.warning(f"Insufficient permissions for directory: {directory}")
                return False

            return True

        except Exception as e:
            logger.error(f"Error checking directory permissions for {directory}: {e}")
            return False

    def get_security_info(self) -> Dict[str, Any]:
        """
        Get current security configuration and status

        Returns:
            Dict[str, Any]: Security information
        """
        return {
            'security_level': self.policy.security_level.value,
            'max_file_size': self.policy.max_file_size,
            'allowed_video_formats': self.policy.allowed_video_formats,
            'allowed_image_formats': self.policy.allowed_image_formats,
            'max_filename_length': self.policy.max_filename_length,
            'sanitize_paths': self.policy.sanitize_paths,
            'validate_mime_types': self.policy.validate_mime_types,
            'blocked_patterns_count': len(self._blocked_patterns)
        }


# Global security validator instance
_security_validator: Optional[SecurityValidator] = None


def get_security_validator(policy: SecurityPolicy = None) -> SecurityValidator:
    """
    Get or create the global security validator

    Args:
        policy: Security policy to use

    Returns:
        SecurityValidator: Global security validator instance
    """
    global _security_validator
    if _security_validator is None:
        _security_validator = SecurityValidator(policy)
    return _security_validator


def validate_file_operation(file_path: Union[str, Path], operation_type: str = 'read') -> bool:
    """
    Convenience function for file operation validation

    Args:
        file_path: Path to validate
        operation_type: Type of operation ('read', 'write', 'video', 'image')

    Returns:
        bool: True if operation is allowed
    """
    validator = get_security_validator()

    if operation_type == 'video':
        return validator.validate_video_file(file_path)
    elif operation_type == 'image':
        return validator.validate_image_file(file_path)
    else:
        return validator.validate_file_path(file_path)


def sanitize_user_input(input_string: str, max_length: int = 1000) -> str:
    """
    Convenience function for input sanitization

    Args:
        input_string: Input to sanitize
        max_length: Maximum allowed length

    Returns:
        str: Sanitized input
    """
    validator = get_security_validator()
    return validator.sanitize_input(input_string, max_length)