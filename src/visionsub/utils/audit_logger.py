"""
Audit Logging for Security Events
"""
import asyncio
import json
import logging
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from typing import AsyncIterator

from ..security.enhanced_security_manager import SecurityEvent

logger = logging.getLogger(__name__)


@dataclass
class AuditLogEntry:
    """Audit log entry"""
    timestamp: float
    event_type: str
    user_id: Optional[str]
    resource: Optional[str]
    action: Optional[str]
    result: str
    details: Dict[str, Any]
    ip_address: Optional[str]
    user_agent: Optional[str]
    session_id: Optional[str]
    log_level: str = "INFO"


class AuditLogger:
    """Enhanced audit logger for security events"""

    def __init__(
        self,
        log_file_path: Optional[Path] = None,
        max_log_size_mb: int = 100,
        max_log_files: int = 10,
        enable_console_logging: bool = True
    ):
        self.log_file_path = log_file_path or Path.home() / ".visionsub" / "logs" / "audit.log"
        self.max_log_size_bytes = max_log_size_mb * 1024 * 1024
        self.max_log_files = max_log_files
        self.enable_console_logging = enable_console_logging
        
        # Ensure log directory exists
        self.log_file_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Async logging queue
        self._log_queue: asyncio.Queue[AuditLogEntry] = asyncio.Queue()
        self._logging_task: Optional[asyncio.Task] = None
        self._is_logging = False
        
        # Statistics
        self._total_events = 0
        self._events_by_type: Dict[str, int] = {}
        self._events_by_result: Dict[str, int] = {}

    async def start(self):
        """Start audit logging"""
        if self._is_logging:
            return
        
        self._is_logging = True
        self._logging_task = asyncio.create_task(self._logging_loop())
        logger.info("Audit logging started")

    async def stop(self):
        """Stop audit logging"""
        if not self._is_logging:
            return
        
        self._is_logging = False
        if self._logging_task:
            self._logging_task.cancel()
            try:
                await self._logging_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Audit logging stopped")

    async def log_event(self, event: SecurityEvent):
        """Log a security event"""
        try:
            # Convert to audit log entry
            log_entry = AuditLogEntry(
                timestamp=event.timestamp,
                event_type=event.event_type.value,
                user_id=event.user_id,
                resource=event.resource,
                action=event.action,
                result=event.result,
                details=event.details,
                ip_address=event.ip_address,
                user_agent=event.user_agent,
                session_id=event.session_id,
                log_level=self._determine_log_level(event)
            )
            
            # Queue for async logging
            await self._log_queue.put(log_entry)
            
            # Update statistics
            self._update_statistics(log_entry)
            
        except Exception as e:
            logger.error(f"Failed to log security event: {e}")

    def _determine_log_level(self, event: SecurityEvent) -> str:
        """Determine log level based on event"""
        if event.result == "blocked" or event.event_type == SecurityEventType.SECURITY_VIOLATION:
            return "WARNING"
        elif event.result == "failure":
            return "ERROR"
        else:
            return "INFO"

    def _update_statistics(self, entry: AuditLogEntry):
        """Update logging statistics"""
        self._total_events += 1
        self._events_by_type[entry.event_type] = self._events_by_type.get(entry.event_type, 0) + 1
        self._events_by_result[entry.result] = self._events_by_result.get(entry.result, 0) + 1

    async def _logging_loop(self):
        """Async logging loop"""
        while self._is_logging:
            try:
                # Get log entry with timeout
                log_entry = await asyncio.wait_for(self._log_queue.get(), timeout=1.0)
                
                # Log to file
                await self._log_to_file(log_entry)
                
                # Log to console if enabled
                if self.enable_console_logging:
                    self._log_to_console(log_entry)
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Audit logging error: {e}")

    async def _log_to_file(self, entry: AuditLogEntry):
        """Log entry to file"""
        try:
            # Check log file size and rotate if needed
            await self._rotate_log_if_needed()
            
            # Write log entry
            log_line = json.dumps(asdict(entry), ensure_ascii=False) + "\n"
            
            async with aiofiles.open(self.log_file_path, 'a', encoding='utf-8') as f:
                await f.write(log_line)
                
        except Exception as e:
            logger.error(f"Failed to write to audit log: {e}")

    def _log_to_console(self, entry: AuditLogEntry):
        """Log entry to console"""
        try:
            log_message = (
                f"[AUDIT] {entry.timestamp:.3f} "
                f"{entry.event_type} "
                f"user={entry.user_id or 'anonymous'} "
                f"action={entry.action or 'unknown'} "
                f"result={entry.result}"
            )
            
            if entry.log_level == "ERROR":
                logger.error(log_message)
            elif entry.log_level == "WARNING":
                logger.warning(log_message)
            else:
                logger.info(log_message)
                
        except Exception as e:
            logger.error(f"Failed to log to console: {e}")

    async def _rotate_log_if_needed(self):
        """Rotate log file if it's too large"""
        try:
            if self.log_file_path.exists():
                file_size = self.log_file_path.stat().st_size
                
                if file_size > self.max_log_size_bytes:
                    # Rotate log files
                    await self._rotate_log_files()
                    
        except Exception as e:
            logger.error(f"Log rotation failed: {e}")

    async def _rotate_log_files(self):
        """Rotate log files"""
        try:
            # Move existing log files
            for i in range(self.max_log_files - 1, 0, -1):
                old_file = self.log_file_path.with_suffix(f".log.{i}")
                new_file = self.log_file_path.with_suffix(f".log.{i + 1}")
                
                if old_file.exists():
                    if new_file.exists():
                        new_file.unlink()
                    old_file.rename(new_file)
            
            # Move current log file
            if self.log_file_path.exists():
                backup_file = self.log_file_path.with_suffix(".log.1")
                if backup_file.exists():
                    backup_file.unlink()
                self.log_file_path.rename(backup_file)
                
        except Exception as e:
            logger.error(f"Log file rotation failed: {e}")

    async def get_audit_logs(
        self,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
        event_type: Optional[str] = None,
        user_id: Optional[str] = None,
        limit: int = 100
    ) -> List[AuditLogEntry]:
        """Get audit logs with filtering"""
        try:
            logs = []
            
            # Read from current log file
            if self.log_file_path.exists():
                async with aiofiles.open(self.log_file_path, 'r', encoding='utf-8') as f:
                    async for line in f:
                        try:
                            entry_data = json.loads(line.strip())
                            entry = AuditLogEntry(**entry_data)
                            
                            # Apply filters
                            if start_time and entry.timestamp < start_time:
                                continue
                            if end_time and entry.timestamp > end_time:
                                continue
                            if event_type and entry.event_type != event_type:
                                continue
                            if user_id and entry.user_id != user_id:
                                continue
                            
                            logs.append(entry)
                            
                            if len(logs) >= limit:
                                break
                                
                        except (json.JSONDecodeError, TypeError) as e:
                            logger.warning(f"Failed to parse audit log entry: {e}")
                            continue
            
            # Sort by timestamp (newest first)
            logs.sort(key=lambda x: x.timestamp, reverse=True)
            
            return logs[:limit]
            
        except Exception as e:
            logger.error(f"Failed to get audit logs: {e}")
            return []

    async def get_statistics(self) -> Dict[str, Any]:
        """Get audit logging statistics"""
        try:
            return {
                "total_events": self._total_events,
                "events_by_type": self._events_by_type.copy(),
                "events_by_result": self._events_by_result.copy(),
                "log_file_path": str(self.log_file_path),
                "log_file_exists": self.log_file_path.exists(),
                "max_log_size_mb": self.max_log_size_bytes / (1024 * 1024),
                "is_logging": self._is_logging
            }
            
        except Exception as e:
            logger.error(f"Failed to get audit statistics: {e}")
            return {"error": str(e)}

    async def search_logs(
        self,
        query: str,
        search_fields: List[str] = None,
        limit: int = 100
    ) -> List[AuditLogEntry]:
        """Search audit logs"""
        try:
            if search_fields is None:
                search_fields = ["event_type", "action", "details", "user_id"]
            
            logs = []
            
            # Read from current log file
            if self.log_file_path.exists():
                async with aiofiles.open(self.log_file_path, 'r', encoding='utf-8') as f:
                    async for line in f:
                        try:
                            entry_data = json.loads(line.strip())
                            entry = AuditLogEntry(**entry_data)
                            
                            # Search in specified fields
                            for field in search_fields:
                                field_value = getattr(entry, field, None)
                                if field_value and query.lower() in str(field_value).lower():
                                    logs.append(entry)
                                    break
                            
                            if len(logs) >= limit:
                                break
                                
                        except (json.JSONDecodeError, TypeError) as e:
                            continue
            
            return logs
            
        except Exception as e:
            logger.error(f"Failed to search audit logs: {e}")
            return []

    def is_accessible(self) -> bool:
        """Check if audit log is accessible"""
        try:
            return self.log_file_path.parent.exists() and \
                   os.access(self.log_file_path.parent, os.W_OK)
        except Exception:
            return False

    async def cleanup(self):
        """Cleanup audit logger"""
        await self.stop()
        
        # Clear statistics
        self._total_events = 0
        self._events_by_type.clear()
        self._events_by_result.clear()


# Import required modules
try:
    import aiofiles
except ImportError:
    aiofiles = None

try:
    import os
except ImportError:
    os = None

# Import enums
try:
    from ..security.enhanced_security_manager import SecurityEventType
except ImportError:
    SecurityEventType = None