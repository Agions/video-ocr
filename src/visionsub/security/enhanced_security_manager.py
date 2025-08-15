"""
Enhanced Security Components for VisionSub
"""
import asyncio
import hashlib
import logging
import os
import secrets
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from typing import Callable

import aiofiles
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from pydantic import BaseModel, Field, validator

from ..core.errors import SecurityError, ConfigurationError
from ..utils.audit_logger import AuditLogger
from ..utils.rate_limiter import RateLimiter

logger = logging.getLogger(__name__)


class SecurityEventType(Enum):
    """Security event types"""
    AUTHENTICATION = "authentication"
    AUTHORIZATION = "authorization"
    FILE_ACCESS = "file_access"
    CONFIG_CHANGE = "config_change"
    DATA_ACCESS = "data_access"
    SYSTEM_OPERATION = "system_operation"
    SECURITY_VIOLATION = "security_violation"


class SecurityLevel(Enum):
    """Security levels"""
    MINIMAL = 0
    BASIC = 1
    STANDARD = 2
    ENHANCED = 3
    PARANOID = 4


@dataclass
class SecurityEvent:
    """Security event record"""
    event_type: SecurityEventType
    timestamp: float
    user_id: Optional[str] = None
    resource: Optional[str] = None
    action: Optional[str] = None
    result: str = "success"  # success, failure, blocked
    details: Dict[str, Any] = field(default_factory=dict)
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    session_id: Optional[str] = None


@dataclass
class SecurityPolicy:
    """Comprehensive security policy"""
    level: SecurityLevel = SecurityLevel.ENHANCED
    enable_encryption: bool = True
    enable_audit_logging: bool = True
    enable_rate_limiting: bool = True
    enable_input_validation: bool = True
    enable_file_validation: bool = True
    enable_session_management: bool = True
    max_login_attempts: int = 5
    session_timeout: int = 3600  # 1 hour
    password_min_length: int = 12
    password_require_special: bool = True
    password_require_numbers: bool = True
    password_require_uppercase: bool = True
    allowed_file_extensions: Set[str] = field(default_factory=lambda: {
        '.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm', '.m4v',
        '.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif', '.webp'
    })
    max_file_size: int = 1024 * 1024 * 1024  # 1GB
    blocked_ip_addresses: Set[str] = field(default_factory=set)
    rate_limit_requests: int = 100
    rate_limit_window: int = 60  # seconds
    encryption_key_rotation_days: int = 30


class SecurityContext:
    """Security context for operations"""
    
    def __init__(self, user_id: Optional[str] = None, session_id: Optional[str] = None):
        self.user_id = user_id
        self.session_id = session_id
        self.ip_address = None
        self.user_agent = None
        self.permissions: Set[str] = set()
        self.roles: Set[str] = set()
        self.created_at = time.time()
        self.last_activity = time.time()

    def update_activity(self):
        """Update last activity timestamp"""
        self.last_activity = time.time()

    def has_permission(self, permission: str) -> bool:
        """Check if context has specific permission"""
        return permission in self.permissions

    def has_role(self, role: str) -> bool:
        """Check if context has specific role"""
        return role in self.roles


class EnhancedSecurityManager:
    """Enhanced security manager with comprehensive protection"""

    def __init__(self, policy: Optional[SecurityPolicy] = None):
        self.policy = policy or SecurityPolicy()
        self._initialize_components()
        
        # Security context management
        self._active_contexts: Dict[str, SecurityContext] = {}
        self._context_lock = asyncio.Lock()
        
        # Security state
        self._is_initialized = False
        self._encryption_key: Optional[bytes] = None
        self._key_rotation_time = 0

    def _initialize_components(self):
        """Initialize security components"""
        try:
            # Initialize audit logger
            if self.policy.enable_audit_logging:
                self.audit_logger = AuditLogger()
            else:
                self.audit_logger = None
            
            # Initialize rate limiter
            if self.policy.enable_rate_limiting:
                self.rate_limiter = RateLimiter(
                    max_requests=self.policy.rate_limit_requests,
                    window_seconds=self.policy.rate_limit_window
                )
            else:
                self.rate_limiter = None
            
            # Initialize encryption
            if self.policy.enable_encryption:
                self._initialize_encryption()
            
            # Initialize session management
            if self.policy.enable_session_management:
                self._initialize_session_management()
            
            logger.info("Enhanced security components initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize security components: {e}")
            raise ConfigurationError(f"Security initialization failed: {e}")

    def _initialize_encryption(self):
        """Initialize encryption components"""
        try:
            # Generate or load encryption key
            self._encryption_key = self._get_or_create_encryption_key()
            self._key_rotation_time = time.time()
            
            # Initialize Fernet cipher
            self.cipher = Fernet(self._encryption_key)
            
            logger.info("Encryption initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize encryption: {e}")
            raise ConfigurationError(f"Encryption initialization failed: {e}")

    def _get_or_create_encryption_key(self) -> bytes:
        """Get or create encryption key"""
        try:
            key_file = Path.home() / ".visionsub" / "security" / "encryption.key"
            key_file.parent.mkdir(parents=True, exist_ok=True)
            
            if key_file.exists():
                # Load existing key
                with open(key_file, 'rb') as f:
                    key = f.read()
                logger.info("Loaded existing encryption key")
            else:
                # Generate new key
                key = Fernet.generate_key()
                with open(key_file, 'wb') as f:
                    f.write(key)
                os.chmod(key_file, 0o600)
                logger.info("Generated new encryption key")
            
            return key
            
        except Exception as e:
            logger.error(f"Failed to get/create encryption key: {e}")
            raise ConfigurationError(f"Key management failed: {e}")

    def _initialize_session_management(self):
        """Initialize session management"""
        try:
            self._session_store: Dict[str, Dict[str, Any]] = {}
            self._session_cleanup_task = asyncio.create_task(
                self._cleanup_expired_sessions()
            )
            
            logger.info("Session management initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize session management: {e}")
            raise ConfigurationError(f"Session management initialization failed: {e}")

    async def initialize(self) -> bool:
        """Initialize security manager"""
        try:
            # Validate configuration
            if not self._validate_policy():
                return False
            
            # Check encryption key rotation
            if self.policy.enable_encryption:
                await self._check_key_rotation()
            
            # Perform security health check
            if not await self._perform_security_health_check():
                return False
            
            self._is_initialized = True
            logger.info("Enhanced security manager initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize security manager: {e}")
            return False

    def _validate_policy(self) -> bool:
        """Validate security policy"""
        try:
            # Validate numeric ranges
            if self.policy.max_login_attempts <= 0:
                logger.error("Invalid max login attempts")
                return False
            
            if self.policy.session_timeout <= 0:
                logger.error("Invalid session timeout")
                return False
            
            if self.policy.password_min_length < 8:
                logger.error("Password too short")
                return False
            
            # Validate file extensions
            if not self.policy.allowed_file_extensions:
                logger.error("No allowed file extensions")
                return False
            
            logger.debug("Security policy validation passed")
            return True
            
        except Exception as e:
            logger.error(f"Policy validation failed: {e}")
            return False

    async def _check_key_rotation(self):
        """Check and perform encryption key rotation"""
        try:
            current_time = time.time()
            rotation_interval = self.policy.encryption_key_rotation_days * 24 * 3600
            
            if current_time - self._key_rotation_time > rotation_interval:
                await self._rotate_encryption_key()
                self._key_rotation_time = current_time
                
        except Exception as e:
            logger.error(f"Key rotation check failed: {e}")

    async def _rotate_encryption_key(self):
        """Rotate encryption key"""
        try:
            logger.info("Starting encryption key rotation")
            
            # Generate new key
            new_key = Fernet.generate_key()
            
            # Re-encrypt sensitive data with new key
            # (Implementation depends on what data needs to be re-encrypted)
            
            # Save new key
            key_file = Path.home() / ".visionsub" / "security" / "encryption.key"
            with open(key_file, 'wb') as f:
                f.write(new_key)
            os.chmod(key_file, 0o600)
            
            # Update cipher
            self._encryption_key = new_key
            self.cipher = Fernet(new_key)
            
            logger.info("Encryption key rotation completed")
            
        except Exception as e:
            logger.error(f"Key rotation failed: {e}")
            raise SecurityError(f"Key rotation failed: {e}")

    async def _perform_security_health_check(self) -> bool:
        """Perform security health check"""
        try:
            # Check key file permissions
            key_file = Path.home() / ".visionsub" / "security" / "encryption.key"
            if key_file.exists():
                stat = key_file.stat()
                if stat.st_mode & 0o077:  # Check if others have permissions
                    logger.error("Encryption key file has insecure permissions")
                    return False
            
            # Check audit log accessibility
            if self.audit_logger:
                if not self.audit_logger.is_accessible():
                    logger.error("Audit log not accessible")
                    return False
            
            logger.debug("Security health check passed")
            return True
            
        except Exception as e:
            logger.error(f"Security health check failed: {e}")
            return False

    async def create_security_context(
        self, 
        user_id: str, 
        permissions: Set[str], 
        roles: Set[str]
    ) -> SecurityContext:
        """Create security context"""
        try:
            context = SecurityContext(user_id=user_id)
            context.permissions = permissions
            context.roles = roles
            
            # Generate session ID
            session_id = secrets.token_urlsafe(32)
            context.session_id = session_id
            
            # Store context
            async with self._context_lock:
                self._active_contexts[session_id] = context
            
            # Log security event
            await self._log_security_event(
                SecurityEventType.AUTHENTICATION,
                user_id=user_id,
                action="create_context",
                result="success",
                details={"session_id": session_id}
            )
            
            logger.info(f"Created security context for user: {user_id}")
            return context
            
        except Exception as e:
            logger.error(f"Failed to create security context: {e}")
            raise SecurityError(f"Context creation failed: {e}")

    async def get_security_context(self, session_id: str) -> Optional[SecurityContext]:
        """Get security context by session ID"""
        try:
            async with self._context_lock:
                context = self._active_contexts.get(session_id)
                if context:
                    context.update_activity()
                return context
                
        except Exception as e:
            logger.error(f"Failed to get security context: {e}")
            return None

    async def validate_permission(
        self, 
        session_id: str, 
        permission: str,
        resource: Optional[str] = None
    ) -> bool:
        """Validate permission for session"""
        try:
            context = await self.get_security_context(session_id)
            if not context:
                return False
            
            # Check rate limiting
            if self.rate_limiter:
                client_id = context.ip_address or session_id
                if not await self.rate_limiter.is_allowed(client_id):
                    await self._log_security_event(
                        SecurityEventType.SECURITY_VIOLATION,
                        user_id=context.user_id,
                        action="rate_limit_exceeded",
                        result="blocked",
                        details={"permission": permission, "resource": resource}
                    )
                    return False
            
            # Check permission
            has_permission = context.has_permission(permission)
            
            # Log authorization event
            await self._log_security_event(
                SecurityEventType.AUTHORIZATION,
                user_id=context.user_id,
                resource=resource,
                action=f"check_permission:{permission}",
                result="granted" if has_permission else "denied"
            )
            
            return has_permission
            
        except Exception as e:
            logger.error(f"Permission validation failed: {e}")
            return False

    async def encrypt_data(self, data: Union[str, bytes]) -> str:
        """Encrypt data"""
        try:
            if not self.policy.enable_encryption:
                if isinstance(data, str):
                    return data
                return data.decode('utf-8')
            
            if isinstance(data, str):
                data = data.encode('utf-8')
            
            encrypted_data = self.cipher.encrypt(data)
            return encrypted_data.decode('utf-8')
            
        except Exception as e:
            logger.error(f"Data encryption failed: {e}")
            raise SecurityError(f"Encryption failed: {e}")

    async def decrypt_data(self, encrypted_data: str) -> bytes:
        """Decrypt data"""
        try:
            if not self.policy.enable_encryption:
                return encrypted_data.encode('utf-8')
            
            encrypted_bytes = encrypted_data.encode('utf-8')
            decrypted_data = self.cipher.decrypt(encrypted_bytes)
            return decrypted_data
            
        except Exception as e:
            logger.error(f"Data decryption failed: {e}")
            raise SecurityError(f"Decryption failed: {e}")

    async def validate_file_operation(
        self, 
        file_path: Union[str, Path], 
        operation: str = "read",
        context: Optional[SecurityContext] = None
    ) -> bool:
        """Validate file operation"""
        try:
            path = Path(file_path)
            
            # Check file extension
            if path.suffix.lower() not in self.policy.allowed_file_extensions:
                logger.warning(f"Blocked file extension: {path.suffix}")
                return False
            
            # Check file size
            if path.exists() and path.stat().st_size > self.policy.max_file_size:
                logger.warning(f"File too large: {path.stat().st_size} bytes")
                return False
            
            # Check path traversal
            if ".." in str(path) or str(path).startswith("/"):
                logger.warning(f"Potential path traversal: {path}")
                return False
            
            # Log file access
            if context:
                await self._log_security_event(
                    SecurityEventType.FILE_ACCESS,
                    user_id=context.user_id,
                    resource=str(path),
                    action=operation,
                    result="success"
                )
            
            return True
            
        except Exception as e:
            logger.error(f"File operation validation failed: {e}")
            return False

    async def _log_security_event(self, event_type: SecurityEventType, **kwargs):
        """Log security event"""
        try:
            if not self.audit_logger:
                return
            
            event = SecurityEvent(
                event_type=event_type,
                timestamp=time.time(),
                **kwargs
            )
            
            await self.audit_logger.log_event(event)
            
        except Exception as e:
            logger.error(f"Failed to log security event: {e}")

    async def _cleanup_expired_sessions(self):
        """Cleanup expired sessions"""
        try:
            while True:
                await asyncio.sleep(300)  # Check every 5 minutes
                
                current_time = time.time()
                expired_sessions = []
                
                async with self._context_lock:
                    for session_id, context in self._active_contexts.items():
                        if current_time - context.last_activity > self.policy.session_timeout:
                            expired_sessions.append(session_id)
                    
                    # Remove expired sessions
                    for session_id in expired_sessions:
                        del self._active_contexts[session_id]
                        logger.info(f"Expired session: {session_id}")
                
        except asyncio.CancelledError:
            logger.info("Session cleanup task cancelled")
        except Exception as e:
            logger.error(f"Session cleanup failed: {e}")

    async def get_security_status(self) -> Dict[str, Any]:
        """Get security status information"""
        try:
            return {
                "initialized": self._is_initialized,
                "policy_level": self.policy.level.value,
                "encryption_enabled": self.policy.enable_encryption,
                "audit_logging_enabled": self.policy.enable_audit_logging,
                "rate_limiting_enabled": self.policy.enable_rate_limiting,
                "active_sessions": len(self._active_contexts),
                "key_rotation_days": self.policy.encryption_key_rotation_days,
                "days_until_rotation": int(
                    (self.policy.encryption_key_rotation_days * 24 * 3600 - 
                     (time.time() - self._key_rotation_time)) / (24 * 3600)
                )
            }
            
        except Exception as e:
            logger.error(f"Failed to get security status: {e}")
            return {"error": str(e)}

    async def revoke_session(self, session_id: str):
        """Revoke a session"""
        try:
            async with self._context_lock:
                if session_id in self._active_contexts:
                    context = self._active_contexts[session_id]
                    del self._active_contexts[session_id]
                    
                    await self._log_security_event(
                        SecurityEventType.SYSTEM_OPERATION,
                        user_id=context.user_id,
                        action="revoke_session",
                        result="success",
                        details={"session_id": session_id}
                    )
                    
                    logger.info(f"Revoked session: {session_id}")
                    
        except Exception as e:
            logger.error(f"Failed to revoke session: {e}")

    async def cleanup(self):
        """Cleanup security manager"""
        try:
            # Cancel session cleanup task
            if hasattr(self, '_session_cleanup_task'):
                self._session_cleanup_task.cancel()
            
            # Clear active sessions
            async with self._context_lock:
                self._active_contexts.clear()
            
            # Cleanup audit logger
            if self.audit_logger:
                await self.audit_logger.cleanup()
            
            logger.info("Enhanced security manager cleaned up")
            
        except Exception as e:
            logger.error(f"Security manager cleanup failed: {e}")

    @asynccontextmanager
    async def security_context(
        self, 
        user_id: str, 
        permissions: Set[str], 
        roles: Set[str]
    ):
        """Context manager for security operations"""
        context = None
        try:
            context = await self.create_security_context(user_id, permissions, roles)
            yield context
        finally:
            if context:
                await self.revoke_session(context.session_id)


class SecureDataStore:
    """Secure data storage with encryption"""

    def __init__(self, storage_dir: Path, security_manager: EnhancedSecurityManager):
        self.storage_dir = storage_dir
        self.security_manager = security_manager
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        os.chmod(self.storage_dir, 0o700)

    async def store_data(self, key: str, data: Any) -> bool:
        """Store data securely"""
        try:
            import json
            
            # Serialize data
            if isinstance(data, (dict, list)):
                data_str = json.dumps(data, ensure_ascii=False)
            else:
                data_str = str(data)
            
            # Encrypt data
            encrypted_data = await self.security_manager.encrypt_data(data_str)
            
            # Store to file
            file_path = self.storage_dir / f"{key}.enc"
            async with aiofiles.open(file_path, 'w', encoding='utf-8') as f:
                await f.write(encrypted_data)
            
            os.chmod(file_path, 0o600)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to store data: {e}")
            return False

    async def retrieve_data(self, key: str) -> Optional[Any]:
        """Retrieve data securely"""
        try:
            file_path = self.storage_dir / f"{key}.enc"
            if not file_path.exists():
                return None
            
            # Read encrypted data
            async with aiofiles.open(file_path, 'r', encoding='utf-8') as f:
                encrypted_data = await f.read()
            
            # Decrypt data
            decrypted_data = await self.security_manager.decrypt_data(encrypted_data)
            
            # Parse as JSON if possible
            try:
                import json
                return json.loads(decrypted_data.decode('utf-8'))
            except json.JSONDecodeError:
                return decrypted_data.decode('utf-8')
                
        except Exception as e:
            logger.error(f"Failed to retrieve data: {e}")
            return None

    async def delete_data(self, key: str) -> bool:
        """Delete data securely"""
        try:
            file_path = self.storage_dir / f"{key}.enc"
            if file_path.exists():
                file_path.unlink()
                return True
            return False
            
        except Exception as e:
            logger.error(f"Failed to delete data: {e}")
            return False


# Global security manager instance
_security_manager: Optional[EnhancedSecurityManager] = None


async def get_security_manager() -> EnhancedSecurityManager:
    """Get or create global security manager"""
    global _security_manager
    if _security_manager is None:
        _security_manager = EnhancedSecurityManager()
        await _security_manager.initialize()
    return _security_manager


async def create_secure_context(
    user_id: str, 
    permissions: Set[str], 
    roles: Set[str]
) -> SecurityContext:
    """Create secure context using global security manager"""
    security_manager = await get_security_manager()
    return await security_manager.create_security_context(user_id, permissions, roles)