"""
Configuration management with hot reload support
"""
import asyncio
import json
import yaml
import logging
from typing import Dict, Any, Optional, Callable, List
from pathlib import Path
from dataclasses import dataclass, field, asdict
from pydantic import BaseModel, Field
import asyncio
import watchdog.observers
import watchdog.events
from datetime import datetime
import os
from enum import Enum


class ConfigSource(Enum):
    """Configuration source types"""
    FILE = "file"
    ENVIRONMENT = "environment"
    DATABASE = "database"
    REMOTE = "remote"


@dataclass
class ConfigChangeEvent:
    """Configuration change event"""
    config_path: str
    old_value: Any
    new_value: Any
    source: ConfigSource
    timestamp: datetime = field(default_factory=datetime.utcnow)


class HotReloadConfig(BaseModel):
    """Hot reload configuration"""
    enabled: bool = True
    watch_files: bool = True
    watch_environment: bool = True
    debounce_interval: float = 1.0  # seconds
    max_file_size: int = 10 * 1024 * 1024  # 10MB
    allowed_extensions: List[str] = Field(default_factory=lambda: [".yaml", ".yml", ".json", ".toml"])
    ignored_paths: List[str] = Field(default_factory=lambda: ["__pycache__", ".git", "node_modules"])


class DatabaseConfig(BaseModel):
    """Database configuration"""
    host: str = "localhost"
    port: int = 5432
    database: str = "visionsub"
    username: str = "visionsub"
    password: str = ""
    pool_size: int = 10
    max_overflow: int = 20
    pool_timeout: int = 30
    pool_recycle: int = 3600


class RedisConfig(BaseModel):
    """Redis configuration"""
    host: str = "localhost"
    port: int = 6379
    db: int = 0
    password: Optional[str] = None
    max_connections: int = 20
    socket_timeout: float = 5.0
    connection_pool_timeout: float = 10.0


class OCRServiceConfig(BaseModel):
    """OCR service configuration"""
    host: str = "localhost"
    port: int = 8081
    max_workers: int = 4
    queue_size: int = 1000
    timeout: float = 30.0
    retry_attempts: int = 3
    cache_ttl: int = 3600


class UIServiceConfig(BaseModel):
    """UI service configuration"""
    host: str = "localhost"
    port: int = 8080
    static_files_path: str = "./static"
    template_path: str = "./templates"
    session_timeout: int = 3600
    max_upload_size: int = 50 * 1024 * 1024  # 50MB


class GatewayConfig(BaseModel):
    """Gateway configuration"""
    host: str = "localhost"
    port: int = 8000
    cors_origins: List[str] = Field(default_factory=lambda: ["*"])
    rate_limit_requests: int = 100
    rate_limit_window: int = 60
    jwt_secret: str = "your-secret-key-change-in-production"
    jwt_expiration: int = 3600


class MonitoringConfig(BaseModel):
    """Monitoring configuration"""
    enabled: bool = True
    metrics_port: int = 9090
    health_check_interval: int = 30
    log_level: str = "INFO"
    tracing_enabled: bool = False
    profiling_enabled: bool = False


class VisionSubConfig(BaseModel):
    """Main VisionSub configuration"""
    environment: str = "development"
    debug: bool = False
    log_level: str = "INFO"
    hot_reload: HotReloadConfig = Field(default_factory=HotReloadConfig)
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    redis: RedisConfig = Field(default_factory=RedisConfig)
    ocr_service: OCRServiceConfig = Field(default_factory=OCRServiceConfig)
    ui_service: UIServiceConfig = Field(default_factory=UIServiceConfig)
    gateway: GatewayConfig = Field(default_factory=GatewayConfig)
    monitoring: MonitoringConfig = Field(default_factory=MonitoringConfig)
    
    class Config:
        extra = "allow"


class ConfigWatcher(watchdog.events.FileSystemEventHandler):
    """File system watcher for configuration changes"""
    
    def __init__(self, config_manager: 'ConfigManager'):
        self.config_manager = config_manager
        self.observer = watchdog.observers.Observer()
        self.debounce_timers = {}
    
    def start(self, paths: List[str]):
        """Start watching configuration files"""
        for path in paths:
            if os.path.exists(path):
                self.observer.schedule(self, path, recursive=True)
        
        self.observer.start()
        logging.info(f"Started watching configuration files: {paths}")
    
    def stop(self):
        """Stop watching configuration files"""
        self.observer.stop()
        self.observer.join()
        logging.info("Stopped watching configuration files")
    
    def on_modified(self, event):
        """Handle file modification events"""
        if event.is_directory:
            return
        
        file_path = Path(event.src_path)
        
        # Check if file should be watched
        if not self._should_watch_file(file_path):
            return
        
        # Debounce file changes
        self._debounce_file_change(file_path)
    
    def _should_watch_file(self, file_path: Path) -> bool:
        """Check if file should be watched for changes"""
        config = self.config_manager.config.hot_reload
        
        # Check file extension
        if file_path.suffix not in config.allowed_extensions:
            return False
        
        # Check file size
        try:
            if file_path.stat().st_size > config.max_file_size:
                return False
        except OSError:
            return False
        
        # Check ignored paths
        for ignored_path in config.ignored_paths:
            if ignored_path in str(file_path):
                return False
        
        return True
    
    def _debounce_file_change(self, file_path: Path):
        """Debounce file changes to avoid multiple reloads"""
        file_key = str(file_path)
        
        # Cancel existing timer
        if file_key in self.debounce_timers:
            self.debounce_timers[file_key].cancel()
        
        # Create new timer
        delay = self.config_manager.config.hot_reload.debounce_interval
        timer = asyncio.get_event_loop().call_later(
            delay,
            lambda: asyncio.create_task(self._handle_file_change(file_path))
        )
        self.debounce_timers[file_key] = timer
    
    async def _handle_file_change(self, file_path: Path):
        """Handle debounced file change"""
        try:
            # Clean up timer
            file_key = str(file_path)
            if file_key in self.debounce_timers:
                del self.debounce_timers[file_key]
            
            # Reload configuration
            await self.config_manager.reload_config_file(file_path)
            
        except Exception as e:
            logging.error(f"Error handling file change {file_path}: {e}")


class ConfigManager:
    """Configuration manager with hot reload support"""
    
    def __init__(self, config_path: Optional[Path] = None):
        self.config_path = config_path or Path("config/visionsub.yaml")
        self.config: VisionSubConfig = VisionSubConfig()
        self.config_watchers: Dict[str, Callable] = {}
        self.file_watcher: Optional[ConfigWatcher] = None
        self.environment_watcher_task: Optional[asyncio.Task] = None
        self.last_file_mtime: Dict[str, float] = {}
        self.last_environment_vars: Dict[str, str] = {}
        
        # Load initial configuration
        self.load_configuration()
    
    def load_configuration(self):
        """Load configuration from file and environment"""
        # Load from file
        if self.config_path.exists():
            self.load_from_file(self.config_path)
        
        # Load from environment variables
        self.load_from_environment()
        
        # Validate configuration
        self.validate_configuration()
    
    def load_from_file(self, file_path: Path):
        """Load configuration from file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                if file_path.suffix.lower() in ['.yaml', '.yml']:
                    data = yaml.safe_load(f)
                elif file_path.suffix.lower() == '.json':
                    data = json.load(f)
                elif file_path.suffix.lower() == '.toml':
                    # Simple TOML parsing (you might want to use a proper TOML library)
                    import tomllib
                    data = tomllib.load(f)
                else:
                    logging.warning(f"Unsupported config file format: {file_path.suffix}")
                    return
            
            # Update configuration
            self.config = VisionSubConfig(**data)
            
            # Store file modification time
            try:
                self.last_file_mtime[str(file_path)] = file_path.stat().st_mtime
            except OSError:
                pass
            
            logging.info(f"Configuration loaded from {file_path}")
            
        except Exception as e:
            logging.error(f"Error loading configuration from {file_path}: {e}")
    
    def load_from_environment(self):
        """Load configuration from environment variables"""
        env_mappings = {
            'VISIONSUB_ENVIRONMENT': ('environment', str),
            'VISIONSUB_DEBUG': ('debug', bool),
            'VISIONSUB_LOG_LEVEL': ('log_level', str),
            'VISIONSUB_DB_HOST': ('database', 'host', str),
            'VISIONSUB_DB_PORT': ('database', 'port', int),
            'VISIONSUB_DB_NAME': ('database', 'database', str),
            'VISIONSUB_DB_USER': ('database', 'username', str),
            'VISIONSUB_DB_PASSWORD': ('database', 'password', str),
            'VISIONSUB_REDIS_HOST': ('redis', 'host', str),
            'VISIONSUB_REDIS_PORT': ('redis', 'port', int),
            'VISIONSUB_REDIS_DB': ('redis', 'db', int),
            'VISIONSUB_REDIS_PASSWORD': ('redis', 'password', str),
            'VISIONSUB_OCR_HOST': ('ocr_service', 'host', str),
            'VISIONSUB_OCR_PORT': ('ocr_service', 'port', int),
            'VISIONSUB_UI_HOST': ('ui_service', 'host', str),
            'VISIONSUB_UI_PORT': ('ui_service', 'port', str),
            'VISIONSUB_GATEWAY_HOST': ('gateway', 'host', str),
            'VISIONSUB_GATEWAY_PORT': ('gateway', 'port', int),
            'VISIONSUB_GATEWAY_JWT_SECRET': ('gateway', 'jwt_secret', str),
            'VISIONSUB_MONITORING_ENABLED': ('monitoring', 'enabled', bool),
            'VISIONSUB_HOT_RELOAD_ENABLED': ('hot_reload', 'enabled', bool),
        }
        
        changed_vars = {}
        
        for env_var, config_path in env_mappings.items():
            env_value = os.environ.get(env_var)
            if env_value is not None:
                # Convert environment variable to appropriate type
                if len(config_path) == 2:
                    section_name, field_name = config_path
                    section = getattr(self.config, section_name)
                    field_type = section.__annotations__.get(field_name, str)
                else:
                    section_name, field_name, field_type = config_path
                    section = getattr(self.config, section_name)
                
                # Convert value
                if field_type == bool:
                    converted_value = env_value.lower() in ['true', '1', 'yes', 'on']
                elif field_type == int:
                    converted_value = int(env_value)
                elif field_type == float:
                    converted_value = float(env_value)
                else:
                    converted_value = env_value
                
                # Check if value changed
                old_value = getattr(section, field_name)
                if old_value != converted_value:
                    changed_vars[env_var] = (old_value, converted_value)
                    setattr(section, field_name, converted_value)
        
        # Store current environment variables
        current_env = {k: v for k, v in os.environ.items() if k.startswith('VISIONSUB_')}
        self.last_environment_vars = current_env
        
        if changed_vars:
            logging.info(f"Configuration updated from environment: {list(changed_vars.keys())}")
            
            # Notify watchers
            for env_var, (old_value, new_value) in changed_vars.items():
                self._notify_watchers(
                    config_path=f"environment:{env_var}",
                    old_value=old_value,
                    new_value=new_value,
                    source=ConfigSource.ENVIRONMENT
                )
    
    def validate_configuration(self):
        """Validate configuration"""
        try:
            # Pydantic validation
            self.config = VisionSubConfig(**self.config.dict())
            
            # Custom validation
            if self.config.gateway.rate_limit_requests <= 0:
                raise ValueError("Rate limit requests must be positive")
            
            if self.config.gateway.rate_limit_window <= 0:
                raise ValueError("Rate limit window must be positive")
            
            if not (1 <= self.config.ocr_service.max_workers <= 32):
                raise ValueError("OCR max workers must be between 1 and 32")
            
            logging.info("Configuration validation passed")
            
        except Exception as e:
            logging.error(f"Configuration validation failed: {e}")
            raise
    
    def add_config_watcher(self, config_path: str, callback: Callable[[ConfigChangeEvent], None]):
        """Add configuration change watcher"""
        if config_path not in self.config_watchers:
            self.config_watchers[config_path] = []
        self.config_watchers[config_path].append(callback)
    
    def remove_config_watcher(self, config_path: str, callback: Callable):
        """Remove configuration change watcher"""
        if config_path in self.config_watchers:
            self.config_watchers[config_path].remove(callback)
            if not self.config_watchers[config_path]:
                del self.config_watchers[config_path]
    
    def _notify_watchers(self, config_path: str, old_value: Any, new_value: Any, source: ConfigSource):
        """Notify configuration watchers of changes"""
        event = ConfigChangeEvent(
            config_path=config_path,
            old_value=old_value,
            new_value=new_value,
            source=source
        )
        
        # Notify watchers for specific config path
        if config_path in self.config_watchers:
            for callback in self.config_watchers[config_path]:
                try:
                    callback(event)
                except Exception as e:
                    logging.error(f"Error in config watcher callback: {e}")
        
        # Notify global watchers
        if "*" in self.config_watchers:
            for callback in self.config_watchers["*"]:
                try:
                    callback(event)
                except Exception as e:
                    logging.error(f"Error in global config watcher callback: {e}")
    
    async def start_hot_reload(self):
        """Start hot reload monitoring"""
        if not self.config.hot_reload.enabled:
            logging.info("Hot reload is disabled")
            return
        
        # Start file watching
        if self.config.hot_reload.watch_files:
            self.file_watcher = ConfigWatcher(self)
            watch_paths = [str(self.config_path)]
            
            # Add config directory if it exists
            config_dir = self.config_path.parent
            if config_dir.exists():
                watch_paths.append(str(config_dir))
            
            self.file_watcher.start(watch_paths)
        
        # Start environment variable watching
        if self.config.hot_reload.watch_environment:
            self.environment_watcher_task = asyncio.create_task(
                self._watch_environment_variables()
            )
        
        logging.info("Hot reload monitoring started")
    
    async def stop_hot_reload(self):
        """Stop hot reload monitoring"""
        if self.file_watcher:
            self.file_watcher.stop()
            self.file_watcher = None
        
        if self.environment_watcher_task:
            self.environment_watcher_task.cancel()
            try:
                await self.environment_watcher_task
            except asyncio.CancelledError:
                pass
            self.environment_watcher_task = None
        
        logging.info("Hot reload monitoring stopped")
    
    async def _watch_environment_variables(self):
        """Watch for environment variable changes"""
        while True:
            try:
                # Get current environment variables
                current_env = {k: v for k, v in os.environ.items() if k.startswith('VISIONSUB_')}
                
                # Check for changes
                if current_env != self.last_environment_vars:
                    # Reload from environment
                    self.load_from_environment()
                
                self.last_environment_vars = current_env
                
                # Wait before next check
                await asyncio.sleep(5.0)  # Check every 5 seconds
                
            except Exception as e:
                logging.error(f"Error watching environment variables: {e}")
                await asyncio.sleep(10.0)  # Wait longer on error
    
    async def reload_config_file(self, file_path: Path):
        """Reload configuration file"""
        try:
            # Check if file actually changed
            try:
                current_mtime = file_path.stat().st_mtime
                last_mtime = self.last_file_mtime.get(str(file_path), 0)
                
                if current_mtime <= last_mtime:
                    return  # File didn't change
            except OSError:
                return  # File not accessible
            
            # Store old configuration for comparison
            old_config = self.config.dict()
            
            # Reload configuration
            self.load_from_file(file_path)
            
            # Find changed values
            new_config = self.config.dict()
            changed_keys = self._find_changed_keys(old_config, new_config)
            
            # Notify watchers of changes
            for key in changed_keys:
                old_value = self._get_nested_value(old_config, key)
                new_value = self._get_nested_value(new_config, key)
                
                self._notify_watchers(
                    config_path=key,
                    old_value=old_value,
                    new_value=new_value,
                    source=ConfigSource.FILE
                )
            
            logging.info(f"Configuration reloaded from {file_path}")
            
        except Exception as e:
            logging.error(f"Error reloading configuration file {file_path}: {e}")
    
    def _find_changed_keys(self, old_dict: Dict[str, Any], new_dict: Dict[str, Any], 
                          prefix: str = "") -> List[str]:
        """Find changed keys between two dictionaries"""
        changed_keys = []
        
        for key in set(old_dict.keys()) | set(new_dict.keys()):
            full_key = f"{prefix}.{key}" if prefix else key
            
            if key not in old_dict:
                changed_keys.append(full_key)
            elif key not in new_dict:
                changed_keys.append(full_key)
            else:
                old_value = old_dict[key]
                new_value = new_dict[key]
                
                if isinstance(old_value, dict) and isinstance(new_value, dict):
                    changed_keys.extend(self._find_changed_keys(old_value, new_value, full_key))
                elif old_value != new_value:
                    changed_keys.append(full_key)
        
        return changed_keys
    
    def _get_nested_value(self, data: Dict[str, Any], key_path: str) -> Any:
        """Get nested value from dictionary using dot notation"""
        keys = key_path.split('.')
        value = data
        
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return None
        
        return value
    
    def get_config(self) -> VisionSubConfig:
        """Get current configuration"""
        return self.config
    
    def get_service_config(self, service_name: str) -> Dict[str, Any]:
        """Get configuration for a specific service"""
        if service_name == "ocr":
            return self.config.ocr_service.dict()
        elif service_name == "ui":
            return self.config.ui_service.dict()
        elif service_name == "gateway":
            return self.config.gateway.dict()
        else:
            raise ValueError(f"Unknown service: {service_name}")
    
    def update_config(self, updates: Dict[str, Any]):
        """Update configuration with new values"""
        old_config = self.config.dict()
        
        # Update configuration
        for key, value in updates.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
        
        # Validate new configuration
        self.validate_configuration()
        
        # Find changed values
        new_config = self.config.dict()
        changed_keys = self._find_changed_keys(old_config, new_config)
        
        # Notify watchers
        for key in changed_keys:
            old_value = self._get_nested_value(old_config, key)
            new_value = self._get_nested_value(new_config, key)
            
            self._notify_watchers(
                config_path=key,
                old_value=old_value,
                new_value=new_value,
                source=ConfigSource.REMOTE
            )
    
    def save_config(self, file_path: Optional[Path] = None):
        """Save current configuration to file"""
        if file_path is None:
            file_path = self.config_path
        
        try:
            # Create directory if it doesn't exist
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save configuration
            with open(file_path, 'w', encoding='utf-8') as f:
                if file_path.suffix.lower() in ['.yaml', '.yml']:
                    yaml.dump(self.config.dict(), f, default_flow_style=False, indent=2)
                elif file_path.suffix.lower() == '.json':
                    json.dump(self.config.dict(), f, indent=2)
                else:
                    raise ValueError(f"Unsupported file format: {file_path.suffix}")
            
            logging.info(f"Configuration saved to {file_path}")
            
        except Exception as e:
            logging.error(f"Error saving configuration to {file_path}: {e}")
            raise


# Global configuration manager instance
config_manager = ConfigManager()


async def get_config() -> VisionSubConfig:
    """Get global configuration instance"""
    return config_manager.get_config()


async def get_service_config(service_name: str) -> Dict[str, Any]:
    """Get service configuration"""
    return config_manager.get_service_config(service_name)


async def start_config_hot_reload():
    """Start configuration hot reload"""
    await config_manager.start_hot_reload()


async def stop_config_hot_reload():
    """Stop configuration hot reload"""
    await config_manager.stop_hot_reload()