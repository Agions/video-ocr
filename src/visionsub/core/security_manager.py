"""
Input validation and permission management system
"""
import hashlib
import logging
import os
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union, Callable
from uuid import uuid4

from ..core.errors import VisionSubError, ErrorCategory, ErrorSeverity, RecoveryAction
from ..models.config import SecurityConfig


class Permission(Enum):
    """Permission types"""
    READ = "read"
    WRITE = "write"
    EXECUTE = "execute"
    DELETE = "delete"
    ADMIN = "admin"


class ResourceType(Enum):
    """Resource types"""
    FILE = "file"
    DIRECTORY = "directory"
    CONFIGURATION = "configuration"
    OCR_ENGINE = "ocr_engine"
    VIDEO_PROCESSOR = "video_processor"
    SYSTEM = "system"


class ValidationRule(Enum):
    """Validation rule types"""
    REQUIRED = "required"
    TYPE = "type"
    RANGE = "range"
    PATTERN = "pattern"
    CUSTOM = "custom"
    FILE_EXISTS = "file_exists"
    DIRECTORY_EXISTS = "directory_exists"
    FILE_READABLE = "file_readable"
    FILE_WRITABLE = "file_writable"


@dataclass
class ValidationResult:
    """Result of validation operation"""
    is_valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    sanitized_data: Optional[Dict[str, Any]] = None


@dataclass
class User:
    """User information"""
    user_id: str
    username: str
    email: Optional[str] = None
    permissions: Set[Permission] = field(default_factory=set)
    role: str = "user"
    is_active: bool = True
    created_at: datetime = field(default_factory=datetime.now)
    last_login: Optional[datetime] = None


@dataclass
class Resource:
    """Resource information"""
    resource_id: str
    resource_type: ResourceType
    path: str
    owner_id: str
    permissions: Dict[Permission, Set[str]] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Session:
    """User session information"""
    session_id: str
    user_id: str
    created_at: datetime
    expires_at: datetime
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    is_active: bool = True


class InputValidator:
    """Input validation system with comprehensive rules"""
    
    def __init__(self):
        self.validation_rules: Dict[str, List[Dict]] = {}
        self.custom_validators: Dict[str, Callable] = {}
        self.logger = logging.getLogger(__name__)

    def add_validation_rule(self, field: str, rule: Dict):
        """Add validation rule for a field"""
        if field not in self.validation_rules:
            self.validation_rules[field] = []
        self.validation_rules[field].append(rule)

    def add_custom_validator(self, name: str, validator: Callable):
        """Add custom validation function"""
        self.custom_validators[name] = validator

    def validate(self, data: Dict[str, Any], schema: Dict[str, List[Dict]]) -> ValidationResult:
        """
        Validate input data against schema
        
        Args:
            data: Input data to validate
            schema: Validation schema
            
        Returns:
            ValidationResult: Validation result
        """
        result = ValidationResult(is_valid=True, sanitized_data=data.copy())
        
        for field, rules in schema.items():
            field_value = data.get(field)
            
            for rule in rules:
                rule_type = rule.get('type')
                
                try:
                    if rule_type == ValidationRule.REQUIRED.value:
                        self._validate_required(field, field_value, result)
                    elif rule_type == ValidationRule.TYPE.value:
                        self._validate_type(field, field_value, rule['expected_type'], result)
                    elif rule_type == ValidationRule.RANGE.value:
                        self._validate_range(field, field_value, rule['min'], rule['max'], result)
                    elif rule_type == ValidationRule.PATTERN.value:
                        self._validate_pattern(field, field_value, rule['pattern'], result)
                    elif rule_type == ValidationRule.CUSTOM.value:
                        self._validate_custom(field, field_value, rule['validator'], result)
                    elif rule_type == ValidationRule.FILE_EXISTS.value:
                        self._validate_file_exists(field, field_value, result)
                    elif rule_type == ValidationRule.DIRECTORY_EXISTS.value:
                        self._validate_directory_exists(field, field_value, result)
                    elif rule_type == ValidationRule.FILE_READABLE.value:
                        self._validate_file_readable(field, field_value, result)
                    elif rule_type == ValidationRule.FILE_WRITABLE.value:
                        self._validate_file_writable(field, field_value, result)
                        
                except Exception as e:
                    result.errors.append(f"Validation error for field '{field}': {str(e)}")
                    result.is_valid = False
        
        result.is_valid = len(result.errors) == 0
        return result

    def _validate_required(self, field: str, value: Any, result: ValidationResult):
        """Validate required field"""
        if value is None or value == '':
            result.errors.append(f"Field '{field}' is required")

    def _validate_type(self, field: str, value: Any, expected_type: str, result: ValidationResult):
        """Validate field type"""
        if value is None:
            return
            
        type_map = {
            'str': str,
            'int': int,
            'float': float,
            'bool': bool,
            'list': list,
            'dict': dict,
            'path': (str, Path)
        }
        
        expected_python_type = type_map.get(expected_type)
        if expected_python_type and not isinstance(value, expected_python_type):
            result.errors.append(f"Field '{field}' must be of type {expected_type}")

    def _validate_range(self, field: str, value: Any, min_val: Any, max_val: Any, result: ValidationResult):
        """Validate numeric range"""
        if value is None:
            return
            
        if isinstance(value, (int, float)):
            if min_val is not None and value < min_val:
                result.errors.append(f"Field '{field}' must be >= {min_val}")
            if max_val is not None and value > max_val:
                result.errors.append(f"Field '{field}' must be <= {max_val}")

    def _validate_pattern(self, field: str, value: Any, pattern: str, result: ValidationResult):
        """Validate string pattern"""
        if value is None:
            return
            
        if isinstance(value, str):
            if not re.match(pattern, value):
                result.errors.append(f"Field '{field}' does not match required pattern")

    def _validate_custom(self, field: str, value: Any, validator_name: str, result: ValidationResult):
        """Validate using custom validator"""
        if validator_name not in self.custom_validators:
            result.errors.append(f"Custom validator '{validator_name}' not found")
            return
            
        try:
            validator_result = self.custom_validators[validator_name](value)
            if not validator_result:
                result.errors.append(f"Field '{field}' failed custom validation '{validator_name}'")
        except Exception as e:
            result.errors.append(f"Custom validator '{validator_name}' failed: {str(e)}")

    def _validate_file_exists(self, field: str, value: Any, result: ValidationResult):
        """Validate file exists"""
        if value is None:
            return
            
        if isinstance(value, (str, Path)):
            file_path = Path(value)
            if not file_path.exists():
                result.errors.append(f"File '{value}' does not exist")
            elif not file_path.is_file():
                result.errors.append(f"Path '{value}' is not a file")

    def _validate_directory_exists(self, field: str, value: Any, result: ValidationResult):
        """Validate directory exists"""
        if value is None:
            return
            
        if isinstance(value, (str, Path)):
            dir_path = Path(value)
            if not dir_path.exists():
                result.errors.append(f"Directory '{value}' does not exist")
            elif not dir_path.is_dir():
                result.errors.append(f"Path '{value}' is not a directory")

    def _validate_file_readable(self, field: str, value: Any, result: ValidationResult):
        """Validate file is readable"""
        if value is None:
            return
            
        if isinstance(value, (str, Path)):
            file_path = Path(value)
            if not file_path.exists():
                result.errors.append(f"File '{value}' does not exist")
            elif not os.access(file_path, os.R_OK):
                result.errors.append(f"File '{value}' is not readable")

    def _validate_file_writable(self, field: str, value: Any, result: ValidationResult):
        """Validate file is writable"""
        if value is None:
            return
            
        if isinstance(value, (str, Path)):
            file_path = Path(value)
            if file_path.exists() and not os.access(file_path, os.W_OK):
                result.errors.append(f"File '{value}' is not writable")
            elif not file_path.parent.exists():
                result.errors.append(f"Parent directory of '{value}' does not exist")

    def sanitize_input(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Sanitize input data to prevent injection attacks"""
        sanitized = {}
        
        for key, value in data.items():
            if isinstance(value, str):
                # Remove potentially dangerous characters
                sanitized[key] = self._sanitize_string(value)
            elif isinstance(value, dict):
                sanitized[key] = self.sanitize_input(value)
            elif isinstance(value, list):
                sanitized[key] = [self.sanitize_input(item) if isinstance(item, dict) else item for item in value]
            else:
                sanitized[key] = value
                
        return sanitized

    def _sanitize_string(self, value: str) -> str:
        """Sanitize string input"""
        # Remove potentially dangerous characters
        dangerous_chars = ['<', '>', '&', '"', "'", '`', '\\']
        for char in dangerous_chars:
            value = value.replace(char, '')
        
        # Limit length
        if len(value) > 1000:
            value = value[:1000]
            
        return value.strip()


class PermissionManager:
    """Permission management system"""
    
    def __init__(self, config: SecurityConfig):
        self.config = config
        self.users: Dict[str, User] = {}
        self.resources: Dict[str, Resource] = {}
        self.sessions: Dict[str, Session] = {}
        self.role_permissions: Dict[str, Set[Permission]] = {
            'admin': {Permission.ADMIN, Permission.READ, Permission.WRITE, Permission.EXECUTE, Permission.DELETE},
            'user': {Permission.READ, Permission.WRITE, Permission.EXECUTE},
            'viewer': {Permission.READ},
            'guest': set()
        }
        self.logger = logging.getLogger(__name__)

    def create_user(self, username: str, email: Optional[str] = None, role: str = "user") -> User:
        """Create a new user"""
        user_id = str(uuid4())
        permissions = self.role_permissions.get(role, set())
        
        user = User(
            user_id=user_id,
            username=username,
            email=email,
            permissions=permissions.copy(),
            role=role
        )
        
        self.users[user_id] = user
        self.logger.info(f"Created user: {username} ({user_id})")
        return user

    def create_resource(
        self, 
        resource_type: ResourceType, 
        path: str, 
        owner_id: str,
        permissions: Optional[Dict[Permission, Set[str]]] = None
    ) -> Resource:
        """Create a new resource"""
        resource_id = str(uuid4())
        
        resource = Resource(
            resource_id=resource_id,
            resource_type=resource_type,
            path=path,
            owner_id=owner_id,
            permissions=permissions or {}
        )
        
        self.resources[resource_id] = resource
        self.logger.info(f"Created resource: {resource_type.value} at {path}")
        return resource

    def create_session(self, user_id: str, ip_address: Optional[str] = None, user_agent: Optional[str] = None) -> Session:
        """Create a new user session"""
        session_id = self._generate_session_id()
        expires_at = datetime.now() + timedelta(hours=self.config.session_timeout_hours)
        
        session = Session(
            session_id=session_id,
            user_id=user_id,
            created_at=datetime.now(),
            expires_at=expires_at,
            ip_address=ip_address,
            user_agent=user_agent
        )
        
        self.sessions[session_id] = session
        
        # Update user's last login
        if user_id in self.users:
            self.users[user_id].last_login = datetime.now()
        
        self.logger.info(f"Created session for user {user_id}")
        return session

    def check_permission(self, user_id: str, resource_id: str, permission: Permission) -> bool:
        """Check if user has permission for resource"""
        # Check if user exists
        if user_id not in self.users:
            return False
        
        user = self.users[user_id]
        
        # Admin has all permissions
        if Permission.ADMIN in user.permissions:
            return True
        
        # Check if resource exists
        if resource_id not in self.resources:
            return False
        
        resource = self.resources[resource_id]
        
        # Owner has all permissions for their resources
        if resource.owner_id == user_id:
            return True
        
        # Check specific permissions
        if permission in resource.permissions:
            return user_id in resource.permissions[permission]
        
        # Check user's global permissions
        return permission in user.permissions

    def grant_permission(self, user_id: str, resource_id: str, permission: Permission):
        """Grant permission to user for resource"""
        if user_id not in self.users or resource_id not in self.resources:
            raise VisionSubError("User or resource not found", ErrorCategory.VALIDATION)
        
        resource = self.resources[resource_id]
        if permission not in resource.permissions:
            resource.permissions[permission] = set()
        
        resource.permissions[permission].add(user_id)
        self.logger.info(f"Granted {permission.value} permission to user {user_id} for resource {resource_id}")

    def revoke_permission(self, user_id: str, resource_id: str, permission: Permission):
        """Revoke permission from user for resource"""
        if resource_id not in self.resources:
            return
        
        resource = self.resources[resource_id]
        if permission in resource.permissions and user_id in resource.permissions[permission]:
            resource.permissions[permission].remove(user_id)
            self.logger.info(f"Revoked {permission.value} permission from user {user_id} for resource {resource_id}")

    def validate_session(self, session_id: str) -> Optional[User]:
        """Validate session and return user"""
        if session_id not in self.sessions:
            return None
        
        session = self.sessions[session_id]
        
        # Check if session is expired
        if datetime.now() > session.expires_at:
            del self.sessions[session_id]
            return None
        
        # Check if session is active
        if not session.is_active:
            return None
        
        # Check if user exists and is active
        user = self.users.get(session.user_id)
        if not user or not user.is_active:
            return None
        
        return user

    def revoke_session(self, session_id: str):
        """Revoke a session"""
        if session_id in self.sessions:
            self.sessions[session_id].is_active = False
            del self.sessions[session_id]
            self.logger.info(f"Revoked session {session_id}")

    def cleanup_expired_sessions(self):
        """Clean up expired sessions"""
        current_time = datetime.now()
        expired_sessions = [
            session_id for session_id, session in self.sessions.items()
            if current_time > session.expires_at
        ]
        
        for session_id in expired_sessions:
            del self.sessions[session_id]
        
        if expired_sessions:
            self.logger.info(f"Cleaned up {len(expired_sessions)} expired sessions")

    def get_user_permissions(self, user_id: str) -> Set[Permission]:
        """Get all permissions for a user"""
        if user_id not in self.users:
            return set()
        
        return self.users[user_id].permissions.copy()

    def add_role_permission(self, role: str, permission: Permission):
        """Add permission to a role"""
        if role not in self.role_permissions:
            self.role_permissions[role] = set()
        
        self.role_permissions[role].add(permission)
        
        # Update existing users with this role
        for user in self.users.values():
            if user.role == role:
                user.permissions.add(permission)

    def _generate_session_id(self) -> str:
        """Generate secure session ID"""
        return hashlib.sha256(os.urandom(32)).hexdigest()


class SecurityManager:
    """Main security management system"""
    
    def __init__(self, config: SecurityConfig):
        self.config = config
        self.validator = InputValidator()
        self.permission_manager = PermissionManager(config)
        self.logger = logging.getLogger(__name__)
        
        # Initialize default validation schemas
        self._init_validation_schemas()

    def _init_validation_schemas(self):
        """Initialize default validation schemas"""
        # Video processing schema
        self.validator.add_validation_rule('video_path', {
            'type': ValidationRule.FILE_EXISTS.value,
            'required': True
        })
        
        # OCR configuration schema
        self.validator.add_validation_rule('ocr_engine', {
            'type': ValidationRule.TYPE.value,
            'expected_type': 'str',
            'required': True
        })
        
        # File path schema
        self.validator.add_validation_rule('output_path', {
            'type': ValidationRule.DIRECTORY_EXISTS.value,
            'required': True
        })

    def validate_input(self, data: Dict[str, Any], schema_name: str) -> ValidationResult:
        """Validate input data against named schema"""
        # Get schema from predefined schemas or use custom validation
        if schema_name == 'video_processing':
            schema = {
                'video_path': [{'type': ValidationRule.FILE_EXISTS.value, 'required': True}],
                'output_dir': [{'type': ValidationRule.DIRECTORY_EXISTS.value, 'required': True}],
                'frame_interval': [{'type': ValidationRule.TYPE.value, 'expected_type': 'float', 'min': 0.1, 'max': 10.0}]
            }
        elif schema_name == 'ocr_config':
            schema = {
                'engine': [{'type': ValidationRule.TYPE.value, 'expected_type': 'str', 'required': True}],
                'language': [{'type': ValidationRule.TYPE.value, 'expected_type': 'str'}],
                'confidence_threshold': [{'type': ValidationRule.TYPE.value, 'expected_type': 'float', 'min': 0.0, 'max': 1.0}]
            }
        else:
            # Default schema
            schema = {}
        
        return self.validator.validate(data, schema)

    def check_access(self, session_id: str, resource_id: str, permission: Permission) -> bool:
        """Check if session has access to resource"""
        user = self.permission_manager.validate_session(session_id)
        if not user:
            return False
        
        return self.permission_manager.check_permission(user.user_id, resource_id, permission)

    def create_default_admin_user(self) -> User:
        """Create default admin user"""
        return self.permission_manager.create_user(
            username="admin",
            email="admin@visionsub.local",
            role="admin"
        )

    def create_default_resources(self):
        """Create default system resources"""
        # Create system resource
        self.permission_manager.create_resource(
            ResourceType.SYSTEM,
            "/system",
            "admin"  # Default admin user ID
        )
        
        # Create OCR engine resource
        self.permission_manager.create_resource(
            ResourceType.OCR_ENGINE,
            "/ocr",
            "admin"
        )
        
        # Create video processor resource
        self.permission_manager.create_resource(
            ResourceType.VIDEO_PROCESSOR,
            "/video",
            "admin"
        )

    def get_security_status(self) -> Dict[str, Any]:
        """Get security status information"""
        return {
            'total_users': len(self.permission_manager.users),
            'active_sessions': len(self.permission_manager.sessions),
            'total_resources': len(self.permission_manager.resources),
            'security_enabled': self.config.enable_security,
            'session_timeout_hours': self.config.session_timeout_hours,
            'max_login_attempts': self.config.max_login_attempts
        }


# Decorator for permission checking
def require_permission(permission: Permission, resource_type: ResourceType):
    """Decorator to check permissions before executing function"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            # Get security manager from kwargs or create default
            security_manager = kwargs.get('security_manager')
            if not security_manager:
                raise VisionSubError("Security manager not available", ErrorCategory.SECURITY)
            
            # Get session ID from kwargs
            session_id = kwargs.get('session_id')
            if not session_id:
                raise VisionSubError("Session ID required", ErrorCategory.VALIDATION)
            
            # Get resource ID from kwargs or determine from context
            resource_id = kwargs.get('resource_id')
            if not resource_id:
                # Try to find resource based on context
                if resource_type == ResourceType.FILE and 'file_path' in kwargs:
                    resource_id = kwargs['file_path']
                else:
                    raise VisionSubError("Resource ID required", ErrorCategory.VALIDATION)
            
            # Check permission
            if not security_manager.check_access(session_id, resource_id, permission):
                raise VisionSubError(
                    f"Permission denied: {permission.value} required for {resource_type.value}",
                    ErrorCategory.PERMISSION,
                    RecoveryAction.TERMINATE
                )
            
            return func(*args, **kwargs)
        
        return wrapper
    return decorator


# Decorator for input validation
def validate_input(schema_name: str):
    """Decorator to validate input before executing function"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            # Get security manager from kwargs or create default
            security_manager = kwargs.get('security_manager')
            if not security_manager:
                raise VisionSubError("Security manager not available", ErrorCategory.SECURITY)
            
            # Extract input data from kwargs
            input_data = {k: v for k, v in kwargs.items() if k not in ['security_manager', 'session_id']}
            
            # Validate input
            result = security_manager.validate_input(input_data, schema_name)
            if not result.is_valid:
                raise VisionSubError(
                    f"Input validation failed: {'; '.join(result.errors)}",
                    ErrorCategory.VALIDATION,
                    RecoveryAction.SKIP
                )
            
            # Use sanitized data
            if result.sanitized_data:
                kwargs.update(result.sanitized_data)
            
            return func(*args, **kwargs)
        
        return wrapper
    return decorator