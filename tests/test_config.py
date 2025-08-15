"""
Tests for configuration models
"""
import pytest
from pathlib import Path
from visionsub.models.config import OcrConfig, ProcessingConfig, AppConfig


class TestOcrConfig:
    """Test OCR configuration"""
    
    def test_default_ocr_config(self):
        """Test default OCR configuration"""
        config = OcrConfig()
        assert config.engine == "PaddleOCR"
        assert config.language == "中文"
        assert config.confidence_threshold == 0.8
        assert config.roi_rect == (0, 0, 0, 0)
        assert config.denoise is True
        assert config.enhance_contrast is True
        assert config.threshold == 180
        assert config.sharpen is True
        assert config.auto_detect_language is True
    
    def test_custom_ocr_config(self):
        """Test custom OCR configuration"""
        config = OcrConfig(
            engine="Tesseract",
            language="英文",
            confidence_threshold=0.9,
            roi_rect=(100, 200, 300, 400),
            denoise=False,
            enhance_contrast=False,
            threshold=150,
            sharpen=False,
            auto_detect_language=False
        )
        assert config.engine == "Tesseract"
        assert config.language == "英文"
        assert config.confidence_threshold == 0.9
        assert config.roi_rect == (100, 200, 300, 400)
        assert config.denoise is False
        assert config.enhance_contrast is False
        assert config.threshold == 150
        assert config.sharpen is False
        assert config.auto_detect_language is False
    
    def test_paddle_lang_code_mapping(self):
        """Test PaddleOCR language code mapping"""
        config = OcrConfig(language="中文")
        assert config.get_paddle_lang_code() == "ch"
        
        config = OcrConfig(language="英文")
        assert config.get_paddle_lang_code() == "en"
        
        config = OcrConfig(language="韩文")
        assert config.get_paddle_lang_code() == "ko"
        
        # Test default for unknown language
        config = OcrConfig(language="未知语言")
        assert config.get_paddle_lang_code() == "ch"
    
    def test_ocr_config_validation(self):
        """Test OCR configuration validation"""
        # Valid configuration
        config = OcrConfig(
            confidence_threshold=0.5,
            threshold=128
        )
        assert config.confidence_threshold == 0.5
        assert config.threshold == 128
        
        # Test boundary values
        config = OcrConfig(confidence_threshold=0.0)
        assert config.confidence_threshold == 0.0
        
        config = OcrConfig(confidence_threshold=1.0)
        assert config.confidence_threshold == 1.0
        
        config = OcrConfig(threshold=0)
        assert config.threshold == 0
        
        config = OcrConfig(threshold=255)
        assert config.threshold == 255


class TestProcessingConfig:
    """Test processing configuration"""
    
    def test_default_processing_config(self):
        """Test default processing configuration"""
        config = ProcessingConfig()
        assert config.scene_threshold == 0.3
        assert config.cache_size == 100
        assert config.frame_interval == 1.0
        assert config.enable_scene_detection is True
        assert config.enable_parallel_processing is True
        assert config.memory_limit_mb == 1024
        assert config.output_formats == ["srt", "vtt"]
        assert config.output_directory == "./output"
        assert config.create_subdirectories is True
    
    def test_custom_processing_config(self):
        """Test custom processing configuration"""
        ocr_config = OcrConfig()
        config = ProcessingConfig(
            ocr_config=ocr_config,
            scene_threshold=0.5,
            cache_size=200,
            frame_interval=0.5,
            enable_scene_detection=False,
            enable_parallel_processing=False,
            memory_limit_mb=2048,
            output_formats=["srt", "vtt", "json"],
            output_directory="./custom_output",
            create_subdirectories=False
        )
        assert config.scene_threshold == 0.5
        assert config.cache_size == 200
        assert config.frame_interval == 0.5
        assert config.enable_scene_detection is False
        assert config.enable_parallel_processing is False
        assert config.memory_limit_mb == 2048
        assert config.output_formats == ["srt", "vtt", "json"]
        assert config.output_directory == "./custom_output"
        assert config.create_subdirectories is False


class TestAppConfig:
    """Test application configuration"""
    
    def test_default_app_config(self):
        """Test default application configuration"""
        config = AppConfig()
        assert config.app_name == "VisionSub"
        assert config.version == "2.0.0"
        assert config.processing.scene_threshold == 0.3
        assert config.ui.theme == "dark"
        assert config.ui.language == "zh_CN"
        assert config.logging.level == "INFO"
        assert config.security.enable_input_validation is True
    
    def test_app_config_validation(self):
        """Test application configuration validation"""
        config = AppConfig()
        
        # Test valid configuration
        errors = config.validate_config()
        assert len(errors) == 0
        
        # Test invalid configuration
        config.processing.cache_size = 0
        config.processing.max_concurrent_jobs = 0
        config.processing.memory_limit_mb = 100
        config.security.max_file_size_mb = 1
        config.security.allowed_video_formats = []
        
        errors = config.validate_config()
        assert len(errors) > 0
        assert any("Cache size must be at least 1" in error for error in errors)
        assert any("Max concurrent jobs must be at least 1" in error for error in errors)
        assert any("Memory limit must be at least 256MB" in error for error in errors)
        assert any("Max file size must be at least 10MB" in error for error in errors)
        assert any("At least one video format must be allowed" in error for error in errors)
    
    def test_config_file_operations(self, sample_app_config, temp_config_file):
        """Test configuration file save and load operations"""
        # Save configuration
        sample_app_config.save_to_file(temp_config_file)
        assert Path(temp_config_file).exists()
        
        # Load configuration
        loaded_config = AppConfig.load_from_file(temp_config_file)
        assert loaded_config.app_name == sample_app_config.app_name
        assert loaded_config.version == sample_app_config.version
        assert loaded_config.processing.scene_threshold == sample_app_config.processing.scene_threshold
        
        # Test YAML format
        yaml_file = temp_config_file.replace('.json', '.yaml')
        sample_app_config.save_to_file(yaml_file)
        assert Path(yaml_file).exists()
        
        loaded_yaml_config = AppConfig.load_from_file(yaml_file)
        assert loaded_yaml_config.app_name == sample_app_config.app_name
        
        # Cleanup
        Path(yaml_file).unlink(missing_ok=True)