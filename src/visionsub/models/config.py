from typing import List, Literal, Optional, Tuple

from pydantic import BaseModel, Field


class OcrConfig(BaseModel):
    """
    A Pydantic model to hold all user-configurable OCR parameters.
    This ensures that all configuration is type-safe and centrally managed.
    """

    # Literal type enforces that the value must be one of the given strings.
    engine: Literal["PaddleOCR", "Tesseract"] = Field(
        default="PaddleOCR",
        description="The OCR engine to use for text recognition."
    )

    language: str = Field(
        default="中文",  # 默认使用中文
        description="OCR引擎的语言显示名称 (如: '中文', '英文', '韩文')."
    )

    threshold: int = Field(
        default=180,
        ge=0,
        le=255,
        description="The binarization threshold (0-255). Text pixels lighter than this will be converted to white."
    )

    # The region of interest (ROI) rectangle: [x, y, width, height]
    # Using Tuple[int, int, int, int] for a fixed-size tuple.
    roi_rect: Tuple[int, int, int, int] = Field(
        default=(0, 0, 0, 0),
        description="The subtitle region selected by the user on the video frame."
    )

    # Advanced OCR settings
    confidence_threshold: float = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
        description="Minimum confidence threshold for OCR results."
    )

    enable_preprocessing: bool = Field(
        default=True,
        description="Enable image preprocessing (denoising, contrast enhancement, etc.)."
    )

    enable_postprocessing: bool = Field(
        default=True,
        description="Enable text postprocessing (error correction, punctuation fixing)."
    )

    custom_params: Optional[dict] = Field(
        default=None,
        description="Custom parameters for OCR engine."
    )

    # Enhanced preprocessing options
    denoise: bool = Field(
        default=True,
        description="Enable image denoising."
    )

    enhance_contrast: bool = Field(
        default=True,
        description="Enable contrast enhancement."
    )

    sharpen: bool = Field(
        default=True,
        description="Enable image sharpening."
    )

    auto_detect_language: bool = Field(
        default=True,
        description="Automatically detect text language."
    )

    def get_paddle_lang_code(self) -> str:
        """Get PaddleOCR language code from display language"""
        lang_mapping = {
            "中文": "ch",
            "英文": "en", 
            "韩文": "ko",
            "日文": "ja",
            "法文": "fr",
            "德文": "de",
            "西班牙文": "es",
            "俄文": "ru",
            "阿拉伯文": "ar",
            "印地文": "hi"
        }
        return lang_mapping.get(self.language, "ch")  # Default to Chinese


class ProcessingConfig(BaseModel):
    """
    Main processing configuration for VisionSub.
    Controls video processing, OCR, and export settings.
    """

    ocr_config: OcrConfig = Field(
        default_factory=OcrConfig,
        description="OCR engine configuration."
    )

    scene_threshold: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Threshold for scene change detection (0.0-1.0)."
    )

    cache_size: int = Field(
        default=100,
        ge=1,
        le=10000,
        description="Maximum number of frames to cache for OCR processing."
    )

    max_concurrent_jobs: int = Field(
        default=4,
        ge=1,
        le=16,
        description="Maximum number of concurrent processing jobs."
    )

    enable_batch_processing: bool = Field(
        default=True,
        description="Enable batch processing capabilities."
    )

    output_formats: List[str] = Field(
        default=["srt", "vtt"],
        description="Default output formats for subtitle export."
    )

    # Video processing settings
    frame_interval: float = Field(
        default=1.0,
        ge=0.1,
        le=10.0,
        description="Interval between frames to process (in seconds)."
    )

    enable_scene_detection: bool = Field(
        default=True,
        description="Enable intelligent scene change detection."
    )

    # Performance settings
    enable_parallel_processing: bool = Field(
        default=True,
        description="Enable parallel processing of frames."
    )

    memory_limit_mb: int = Field(
        default=1024,
        ge=256,
        le=8192,
        description="Memory usage limit in megabytes."
    )

    # Output settings
    output_directory: str = Field(
        default="./output",
        description="Default output directory for processed files."
    )

    create_subdirectories: bool = Field(
        default=True,
        description="Create subdirectories for each processed video."
    )

    # Advanced settings
    debug_mode: bool = Field(
        default=False,
        description="Enable debug mode with additional logging."
    )

    enable_performance_monitoring: bool = Field(
        default=True,
        description="Enable performance monitoring and metrics collection."
    )


class UIConfig(BaseModel):
    """
    User interface configuration settings.
    """

    theme: str = Field(
        default="dark",
        description="UI theme (dark, light, system)."
    )

    language: str = Field(
        default="zh_CN",
        description="Interface language."
    )

    window_size: Tuple[int, int] = Field(
        default=(1200, 800),
        description="Default window size."
    )

    font_size: int = Field(
        default=10,
        ge=8,
        le=24,
        description="Base font size for UI elements."
    )

    enable_animations: bool = Field(
        default=True,
        description="Enable UI animations and transitions."
    )

    show_performance_metrics: bool = Field(
        default=True,
        description="Show performance metrics in the UI."
    )


class LoggingConfig(BaseModel):
    """
    Logging configuration settings.
    """

    level: str = Field(
        default="INFO",
        description="Logging level (DEBUG, INFO, WARNING, ERROR)."
    )

    log_file: Optional[str] = Field(
        default=None,
        description="Path to log file. If None, logs to console only."
    )

    max_file_size_mb: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Maximum log file size in megabytes."
    )

    backup_count: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Number of backup log files to keep."
    )

    enable_structured_logging: bool = Field(
        default=True,
        description="Enable structured JSON logging."
    )


class SecurityConfig(BaseModel):
    """
    Security configuration settings.
    """

    enable_input_validation: bool = Field(
        default=True,
        description="Enable input validation and sanitization."
    )

    max_file_size_mb: int = Field(
        default=500,
        ge=10,
        le=2048,
        description="Maximum file size for upload in megabytes."
    )

    allowed_video_formats: List[str] = Field(
        default=["mp4", "avi", "mkv", "mov", "wmv", "flv", "webm"],
        description="Allowed video file formats."
    )

    enable_rate_limiting: bool = Field(
        default=True,
        description="Enable rate limiting for API endpoints."
    )

    rate_limit_requests_per_minute: int = Field(
        default=60,
        ge=1,
        le=1000,
        description="Rate limit: requests per minute."
    )


class AppConfig(BaseModel):
    """
    Complete application configuration.
    """

    processing: ProcessingConfig = Field(
        default_factory=ProcessingConfig,
        description="Processing configuration."
    )

    ui: UIConfig = Field(
        default_factory=UIConfig,
        description="UI configuration."
    )

    logging: LoggingConfig = Field(
        default_factory=LoggingConfig,
        description="Logging configuration."
    )

    security: SecurityConfig = Field(
        default_factory=SecurityConfig,
        description="Security configuration."
    )

    # Application metadata
    app_name: str = Field(
        default="VisionSub",
        description="Application name."
    )

    version: str = Field(
        default="2.0.0",
        description="Application version."
    )

    # Configuration file settings
    config_file_path: Optional[str] = Field(
        default=None,
        description="Path to configuration file."
    )

    @classmethod
    def load_from_file(cls, config_path: str) -> 'AppConfig':
        """
        Load configuration from a JSON or YAML file.

        Args:
            config_path: Path to the configuration file.

        Returns:
            AppConfig instance.
        """
        import json
        from pathlib import Path

        import yaml

        path = Path(config_path)

        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        with open(path, 'r', encoding='utf-8') as f:
            if path.suffix.lower() in ['.yaml', '.yml']:
                data = yaml.safe_load(f)
            else:
                data = json.load(f)

        return cls(**data)

    def save_to_file(self, config_path: str):
        """
        Save configuration to a JSON or YAML file.

        Args:
            config_path: Path to save the configuration file.
        """
        import json
        from pathlib import Path

        import yaml

        path = Path(config_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = self.model_dump()

        with open(path, 'w', encoding='utf-8') as f:
            if path.suffix.lower() in ['.yaml', '.yml']:
                yaml.safe_dump(data, f, default_flow_style=False, allow_unicode=True)
            else:
                json.dump(data, f, indent=2, ensure_ascii=False)

    def get_ocr_config(self) -> OcrConfig:
        """Get OCR configuration."""
        return self.processing.ocr_config

    def update_ocr_config(self, **kwargs):
        """Update OCR configuration."""
        ocr_data = self.processing.ocr_config.model_dump()
        ocr_data.update(kwargs)
        self.processing.ocr_config = OcrConfig(**ocr_data)

    def validate_config(self) -> List[str]:
        """
        Validate configuration and return list of validation errors.

        Returns:
            List of validation error messages.
        """
        errors = []

        # Validate processing config
        if self.processing.cache_size < 1:
            errors.append("Cache size must be at least 1")

        if self.processing.max_concurrent_jobs < 1:
            errors.append("Max concurrent jobs must be at least 1")

        if self.processing.memory_limit_mb < 256:
            errors.append("Memory limit must be at least 256MB")

        # Validate security config
        if self.security.max_file_size_mb < 10:
            errors.append("Max file size must be at least 10MB")

        if len(self.security.allowed_video_formats) == 0:
            errors.append("At least one video format must be allowed")

        return errors
