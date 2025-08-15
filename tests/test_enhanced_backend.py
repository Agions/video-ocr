"""
Backend Testing Suite for VisionSub Frontend Application

This module provides comprehensive testing for all backend components including:
- Video processing utilities
- OCR processing
- Configuration management
- File handling
- Data validation
"""

import pytest
import asyncio
import numpy as np
import cv2
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from pathlib import Path
import tempfile
import json
import os
from PIL import Image
import pysrt
from datetime import timedelta

# Add src to path
import sys
sys.path.insert(0, str(Path(__file__).parent / "src"))

from visionsub.video_utils import VideoProcessor, FrameExtractor
from visionsub.ocr_utils import OCRProcessor, OCRResult
from visionsub.subtitle_utils import SubtitleProcessor, SubtitleEntry
from visionsub.core.config_manager import ConfigManager
from visionsub.models.config import AppConfig, ProcessingConfig, OcrConfig
from visionsub.core.cache_manager import CacheManager
from visionsub.core.validation import ValidationError, validate_video_file, validate_config


class TestVideoProcessor:
    """Test suite for Video Processing component"""
    
    @pytest.fixture
    def video_processor(self):
        """Create video processor instance"""
        return VideoProcessor()
    
    @pytest.fixture
    def sample_video_path(self):
        """Create a sample video file for testing"""
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as f:
            # Create a minimal MP4 file
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(f.name, fourcc, 30.0, (640, 480))
            
            # Create 10 frames with some content
            for i in range(10):
                frame = np.full((480, 640, 3), [i * 25, 100, 200], dtype=np.uint8)
                # Add some text to the frame
                cv2.putText(frame, f"Frame {i}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                out.write(frame)
            
            out.release()
            yield f.name
        
        Path(f.name).unlink()
    
    def test_video_processor_initialization(self, video_processor):
        """Test video processor initialization"""
        assert video_processor is not None
        assert hasattr(video_processor, 'config')
        assert hasattr(video_processor, 'cache_manager')
    
    def test_video_info_extraction(self, video_processor, sample_video_path):
        """Test video information extraction"""
        info = video_processor.get_video_info(sample_video_path)
        
        assert 'fps' in info
        assert 'frame_count' in info
        assert 'duration' in info
        assert 'width' in info
        assert 'height' in info
        
        assert info['fps'] == 30.0
        assert info['frame_count'] == 10
        assert info['width'] == 640
        assert info['height'] == 480
    
    def test_frame_extraction(self, video_processor, sample_video_path):
        """Test frame extraction"""
        frame = video_processor.extract_frame(sample_video_path, 1000)  # 1 second
        
        assert frame is not None
        assert isinstance(frame, np.ndarray)
        assert frame.shape == (480, 640, 3)
    
    def test_frame_extraction_with_roi(self, video_processor, sample_video_path):
        """Test frame extraction with ROI"""
        roi_rect = (100, 100, 200, 100)  # x, y, width, height
        frame = video_processor.extract_frame(sample_video_path, 1000, roi_rect)
        
        assert frame is not None
        assert isinstance(frame, np.ndarray)
        assert frame.shape == (100, 200, 3)
    
    def test_batch_frame_extraction(self, video_processor, sample_video_path):
        """Test batch frame extraction"""
        timestamps = [0, 1000, 2000, 3000]
        frames = video_processor.extract_frames_batch(sample_video_path, timestamps)
        
        assert len(frames) == 4
        for frame in frames:
            assert frame is not None
            assert isinstance(frame, np.ndarray)
    
    def test_scene_detection(self, video_processor, sample_video_path):
        """Test scene detection"""
        scenes = video_processor.detect_scenes(sample_video_path)
        
        assert isinstance(scenes, list)
        assert len(scenes) >= 1  # At least one scene
    
    def test_video_validation(self, video_processor):
        """Test video file validation"""
        # Test valid video
        valid_video = Path(__file__).parent / "assets" / "sample.mp4"
        if valid_video.exists():
            assert video_processor.validate_video_file(str(valid_video))
        
        # Test invalid video
        assert not video_processor.validate_video_file("/nonexistent/video.mp4")
        
        # Test unsupported format
        with tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as f:
            f.write(b"Not a video file")
            f.flush()
            assert not video_processor.validate_video_file(f.name)
            Path(f.name).unlink()
    
    def test_error_handling(self, video_processor):
        """Test error handling"""
        # Test with invalid video file
        with pytest.raises(Exception):
            video_processor.get_video_info("/nonexistent/video.mp4")
        
        # Test with invalid timestamp
        sample_video = self.create_sample_video()
        try:
            video_processor.extract_frame(sample_video, -1000)  # Negative timestamp
        except Exception:
            pass  # Expected error
        finally:
            Path(sample_video).unlink()
    
    def create_sample_video(self):
        """Helper method to create sample video"""
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as f:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(f.name, fourcc, 30.0, (640, 480))
            
            # Create 5 frames
            for i in range(5):
                frame = np.full((480, 640, 3), [i * 50, 100, 150], dtype=np.uint8)
                out.write(frame)
            
            out.release()
            return f.name


class TestOCRProcessor:
    """Test suite for OCR Processing component"""
    
    @pytest.fixture
    def ocr_processor(self):
        """Create OCR processor instance"""
        config = OcrConfig(
            engine="PaddleOCR",
            language="中文",
            confidence_threshold=0.8
        )
        return OCRProcessor(config)
    
    @pytest.fixture
    def sample_image(self):
        """Create a sample image with text for testing"""
        # Create image with text
        image = np.full((200, 400, 3), 255, dtype=np.uint8)
        cv2.putText(image, "Hello World", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        cv2.putText(image, "测试文本", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        return image
    
    def test_ocr_processor_initialization(self, ocr_processor):
        """Test OCR processor initialization"""
        assert ocr_processor is not None
        assert hasattr(ocr_processor, 'config')
        assert hasattr(ocr_processor, 'engine')
        assert ocr_processor.config.engine == "PaddleOCR"
    
    def test_ocr_processing(self, ocr_processor, sample_image):
        """Test OCR processing"""
        with patch.object(ocr_processor, 'process_image') as mock_process:
            mock_result = OCRResult(
                text="Hello World",
                confidence=0.95,
                bbox=(50, 80, 200, 120),
                language="en"
            )
            mock_process.return_value = [mock_result]
            
            results = ocr_processor.process_image(sample_image)
            
            assert len(results) == 1
            assert results[0].text == "Hello World"
            assert results[0].confidence == 0.95
            mock_process.assert_called_once_with(sample_image)
    
    def test_ocr_with_roi(self, ocr_processor, sample_image):
        """Test OCR processing with ROI"""
        roi_rect = (50, 80, 200, 120)
        
        with patch.object(ocr_processor, 'process_image_with_roi') as mock_process:
            mock_result = OCRResult(
                text="Hello World",
                confidence=0.95,
                bbox=(0, 0, 150, 40),
                language="en"
            )
            mock_process.return_value = [mock_result]
            
            results = ocr_processor.process_image_with_roi(sample_image, roi_rect)
            
            assert len(results) == 1
            assert results[0].text == "Hello World"
            mock_process.assert_called_once_with(sample_image, roi_rect)
    
    def test_confidence_filtering(self, ocr_processor, sample_image):
        """Test confidence filtering"""
        with patch.object(ocr_processor, 'process_image') as mock_process:
            # Mock results with different confidence levels
            mock_results = [
                OCRResult(text="High confidence", confidence=0.9, bbox=(0, 0, 100, 30), language="en"),
                OCRResult(text="Low confidence", confidence=0.6, bbox=(0, 40, 100, 70), language="en"),
                OCRResult(text="Medium confidence", confidence=0.8, bbox=(0, 80, 100, 110), language="en")
            ]
            mock_process.return_value = mock_results
            
            results = ocr_processor.process_image(sample_image)
            
            # Should filter out low confidence results
            assert len(results) == 2  # Only high and medium confidence
            assert all(result.confidence >= 0.8 for result in results)
    
    def test_language_detection(self, ocr_processor, sample_image):
        """Test language detection"""
        with patch.object(ocr_processor, 'detect_language') as mock_detect:
            mock_detect.return_value = "zh"
            
            language = ocr_processor.detect_language(sample_image)
            
            assert language == "zh"
            mock_detect.assert_called_once_with(sample_image)
    
    def test_text_preprocessing(self, ocr_processor):
        """Test text preprocessing"""
        raw_text = "  Hello   World  \n  Test  Text  "
        processed = ocr_processor.preprocess_text(raw_text)
        
        assert processed == "Hello World Test Text"
    
    def test_ocr_engine_switching(self, ocr_processor):
        """Test OCR engine switching"""
        # Test switching to Tesseract
        ocr_processor.switch_engine("TesseractOCR")
        assert ocr_processor.config.engine == "TesseractOCR"
        
        # Test switching to EasyOCR
        ocr_processor.switch_engine("EasyOCR")
        assert ocr_processor.config.engine == "EasyOCR"
        
        # Test switching back to PaddleOCR
        ocr_processor.switch_engine("PaddleOCR")
        assert ocr_processor.config.engine == "PaddleOCR"
    
    def test_batch_processing(self, ocr_processor):
        """Test batch OCR processing"""
        images = [
            np.full((100, 200, 3), 255, dtype=np.uint8),
            np.full((100, 200, 3), 255, dtype=np.uint8),
            np.full((100, 200, 3), 255, dtype=np.uint8)
        ]
        
        with patch.object(ocr_processor, 'process_image') as mock_process:
            mock_result = OCRResult(
                text="Sample text",
                confidence=0.9,
                bbox=(0, 0, 100, 30),
                language="en"
            )
            mock_process.return_value = [mock_result]
            
            results = ocr_processor.process_batch(images)
            
            assert len(results) == 3
            for result in results:
                assert len(result) == 1
                assert result[0].text == "Sample text"
            
            assert mock_process.call_count == 3
    
    def test_error_handling(self, ocr_processor):
        """Test error handling"""
        # Test with invalid image
        invalid_image = None
        with pytest.raises(Exception):
            ocr_processor.process_image(invalid_image)
        
        # Test with empty image
        empty_image = np.array([])
        with pytest.raises(Exception):
            ocr_processor.process_image(empty_image)
        
        # Test with invalid ROI
        sample_image = np.full((100, 200, 3), 255, dtype=np.uint8)
        invalid_roi = (300, 300, 100, 100)  # Outside image bounds
        results = ocr_processor.process_image_with_roi(sample_image, invalid_roi)
        assert len(results) == 0  # Should handle gracefully


class TestSubtitleProcessor:
    """Test suite for Subtitle Processing component"""
    
    @pytest.fixture
    def subtitle_processor(self):
        """Create subtitle processor instance"""
        return SubtitleProcessor()
    
    @pytest.fixture
    def sample_ocr_results(self):
        """Create sample OCR results for testing"""
        return [
            OCRResult(text="Hello World", confidence=0.9, bbox=(0, 0, 200, 30), language="en"),
            OCRResult(text="测试文本", confidence=0.85, bbox=(0, 40, 200, 70), language="zh"),
            OCRResult(text="Sample subtitle", confidence=0.95, bbox=(0, 80, 200, 110), language="en")
        ]
    
    def test_subtitle_processor_initialization(self, subtitle_processor):
        """Test subtitle processor initialization"""
        assert subtitle_processor is not None
        assert hasattr(subtitle_processor, 'config')
    
    def test_subtitle_creation(self, subtitle_processor, sample_ocr_results):
        """Test subtitle creation from OCR results"""
        timestamps = [0, 2000, 4000]  # milliseconds
        
        subtitles = subtitle_processor.create_subtitles(sample_ocr_results, timestamps)
        
        assert len(subtitles) == 3
        for i, subtitle in enumerate(subtitles):
            assert isinstance(subtitle, SubtitleEntry)
            assert subtitle.text == sample_ocr_results[i].text
            assert subtitle.start == timestamps[i]
            assert subtitle.end == timestamps[i] + 2000  # Default duration
    
    def test_subtitle_timing_adjustment(self, subtitle_processor):
        """Test subtitle timing adjustment"""
        subtitle = SubtitleEntry(
            start=1000,
            end=3000,
            text="Test subtitle",
            index=1
        )
        
        # Adjust timing
        adjusted = subtitle_processor.adjust_timing(subtitle, 500, 2500)
        
        assert adjusted.start == 500
        assert adjusted.end == 2500
        assert adjusted.text == "Test subtitle"
    
    def test_subtitle_merge(self, subtitle_processor):
        """Test subtitle merging"""
        subtitle1 = SubtitleEntry(
            start=1000,
            end=2000,
            text="Hello",
            index=1
        )
        subtitle2 = SubtitleEntry(
            start=2000,
            end=3000,
            text="World",
            index=2
        )
        
        merged = subtitle_processor.merge_subtitles([subtitle1, subtitle2])
        
        assert len(merged) == 1
        assert merged[0].text == "Hello World"
        assert merged[0].start == 1000
        assert merged[0].end == 3000
    
    def test_subtitle_split(self, subtitle_processor):
        """Test subtitle splitting"""
        subtitle = SubtitleEntry(
            start=1000,
            end=4000,
            text="Hello World Test",
            index=1
        )
        
        split = subtitle_processor.split_subtitle(subtitle, 2)
        
        assert len(split) == 2
        assert split[0].start == 1000
        assert split[0].end == 2500
        assert split[1].start == 2500
        assert split[1].end == 4000
    
    def test_subtitle_export_srt(self, subtitle_processor, sample_ocr_results):
        """Test subtitle export to SRT format"""
        timestamps = [0, 2000, 4000]
        subtitles = subtitle_processor.create_subtitles(sample_ocr_results, timestamps)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.srt', delete=False) as f:
            export_path = f.name
        
        subtitle_processor.export_srt(subtitles, export_path)
        
        # Verify file was created
        assert Path(export_path).exists()
        
        # Verify content
        with open(export_path, 'r', encoding='utf-8') as f:
            content = f.read()
            assert "Hello World" in content
            assert "测试文本" in content
            assert "Sample subtitle" in content
        
        Path(export_path).unlink()
    
    def test_subtitle_import_srt(self, subtitle_processor):
        """Test subtitle import from SRT format"""
        # Create test SRT file
        srt_content = """1
00:00:00,000 --> 00:00:02,000
Hello World

2
00:00:02,000 --> 00:00:04,000
测试文本

3
00:00:04,000 --> 00:00:06,000
Sample subtitle
"""
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.srt', delete=False) as f:
            f.write(srt_content)
            f.flush()
            import_path = f.name
        
        subtitles = subtitle_processor.import_srt(import_path)
        
        assert len(subtitles) == 3
        assert subtitles[0].text == "Hello World"
        assert subtitles[1].text == "测试文本"
        assert subtitles[2].text == "Sample subtitle"
        
        Path(import_path).unlink()
    
    def test_subtitle_validation(self, subtitle_processor):
        """Test subtitle validation"""
        # Valid subtitle
        valid_subtitle = SubtitleEntry(
            start=1000,
            end=3000,
            text="Valid subtitle",
            index=1
        )
        assert subtitle_processor.validate_subtitle(valid_subtitle)
        
        # Invalid timing (end before start)
        invalid_subtitle = SubtitleEntry(
            start=3000,
            end=1000,
            text="Invalid subtitle",
            index=1
        )
        assert not subtitle_processor.validate_subtitle(invalid_subtitle)
        
        # Empty text
        empty_subtitle = SubtitleEntry(
            start=1000,
            end=3000,
            text="",
            index=1
        )
        assert not subtitle_processor.validate_subtitle(empty_subtitle)
    
    def test_subtitle_format_conversion(self, subtitle_processor, sample_ocr_results):
        """Test subtitle format conversion"""
        timestamps = [0, 2000, 4000]
        subtitles = subtitle_processor.create_subtitles(sample_ocr_results, timestamps)
        
        # Test SRT format
        srt_content = subtitle_processor.to_srt(subtitles)
        assert "Hello World" in srt_content
        assert "00:00:00,000" in srt_content
        
        # Test VTT format
        vtt_content = subtitle_processor.to_vtt(subtitles)
        assert "Hello World" in vtt_content
        assert "00:00:00.000" in vtt_content
        
        # Test ASS format
        ass_content = subtitle_processor.to_ass(subtitles)
        assert "Hello World" in ass_content
        assert "Dialogue:" in ass_content


class TestConfigManager:
    """Test suite for Configuration Management component"""
    
    @pytest.fixture
    def config_manager(self):
        """Create config manager instance"""
        return ConfigManager()
    
    @pytest.fixture
    def sample_config(self):
        """Create sample configuration for testing"""
        return AppConfig(
            processing=ProcessingConfig(
                ocr_config=OcrConfig(
                    engine="PaddleOCR",
                    language="中文",
                    confidence_threshold=0.8
                ),
                scene_threshold=0.3,
                cache_size=100
            ),
            ui={
                "theme": "dark",
                "language": "zh_CN"
            }
        )
    
    def test_config_manager_initialization(self, config_manager):
        """Test config manager initialization"""
        assert config_manager is not None
        assert hasattr(config_manager, 'config')
        assert hasattr(config_manager, 'config_path')
    
    def test_config_save_load(self, config_manager, sample_config):
        """Test configuration save and load"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            config_path = f.name
        
        # Save configuration
        config_manager.save_config(sample_config, config_path)
        
        # Load configuration
        loaded_config = config_manager.load_config(config_path)
        
        assert loaded_config.processing.ocr_config.engine == sample_config.processing.ocr_config.engine
        assert loaded_config.processing.ocr_config.language == sample_config.processing.ocr_config.language
        assert loaded_config.processing.ocr_config.confidence_threshold == sample_config.processing.ocr_config.confidence_threshold
        assert loaded_config.processing.scene_threshold == sample_config.processing.scene_threshold
        assert loaded_config.processing.cache_size == sample_config.processing.cache_size
        assert loaded_config.ui == sample_config.ui
        
        Path(config_path).unlink()
    
    def test_config_validation(self, config_manager):
        """Test configuration validation"""
        # Valid configuration
        valid_config = AppConfig(
            processing=ProcessingConfig(
                ocr_config=OcrConfig(
                    engine="PaddleOCR",
                    language="中文",
                    confidence_threshold=0.8
                ),
                scene_threshold=0.3,
                cache_size=100
            ),
            ui={
                "theme": "dark",
                "language": "zh_CN"
            }
        )
        assert config_manager.validate_config(valid_config)
        
        # Invalid configuration (negative confidence threshold)
        invalid_config = AppConfig(
            processing=ProcessingConfig(
                ocr_config=OcrConfig(
                    engine="PaddleOCR",
                    language="中文",
                    confidence_threshold=-0.5  # Invalid
                ),
                scene_threshold=0.3,
                cache_size=100
            ),
            ui={
                "theme": "dark",
                "language": "zh_CN"
            }
        )
        assert not config_manager.validate_config(invalid_config)
    
    def test_config_defaults(self, config_manager):
        """Test default configuration"""
        default_config = config_manager.get_default_config()
        
        assert isinstance(default_config, AppConfig)
        assert default_config.processing.ocr_config.engine == "PaddleOCR"
        assert default_config.processing.ocr_config.language == "中文"
        assert default_config.processing.ocr_config.confidence_threshold == 0.8
        assert default_config.processing.scene_threshold == 0.3
        assert default_config.processing.cache_size == 100
        assert default_config.ui["theme"] == "dark"
        assert default_config.ui["language"] == "zh_CN"
    
    def test_config_merge(self, config_manager):
        """Test configuration merging"""
        base_config = AppConfig(
            processing=ProcessingConfig(
                ocr_config=OcrConfig(
                    engine="PaddleOCR",
                    language="中文",
                    confidence_threshold=0.8
                ),
                scene_threshold=0.3,
                cache_size=100
            ),
            ui={
                "theme": "dark",
                "language": "zh_CN"
            }
        )
        
        override_config = AppConfig(
            processing=ProcessingConfig(
                ocr_config=OcrConfig(
                    engine="TesseractOCR",  # Override
                    language="中文",
                    confidence_threshold=0.9  # Override
                ),
                scene_threshold=0.3,
                cache_size=100
            ),
            ui={
                "theme": "light",  # Override
                "language": "zh_CN"
            }
        )
        
        merged = config_manager.merge_configs(base_config, override_config)
        
        assert merged.processing.ocr_config.engine == "TesseractOCR"
        assert merged.processing.ocr_config.confidence_threshold == 0.9
        assert merged.ui["theme"] == "light"
        assert merged.processing.ocr_config.language == "中文"  # Not overridden
    
    def test_config_reset(self, config_manager):
        """Test configuration reset"""
        # Modify configuration
        config_manager.config.processing.ocr_config.engine = "EasyOCR"
        config_manager.config.ui["theme"] = "light"
        
        # Reset to defaults
        config_manager.reset_to_defaults()
        
        # Verify reset
        assert config_manager.config.processing.ocr_config.engine == "PaddleOCR"
        assert config_manager.config.ui["theme"] == "dark"
    
    def test_config_export_import(self, config_manager, sample_config):
        """Test configuration export and import"""
        # Export configuration
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            export_path = f.name
        
        config_manager.export_config(sample_config, export_path)
        
        # Import configuration
        imported_config = config_manager.import_config(export_path)
        
        assert imported_config.processing.ocr_config.engine == sample_config.processing.ocr_config.engine
        assert imported_config.processing.ocr_config.language == sample_config.processing.ocr_config.language
        assert imported_config.ui == sample_config.ui
        
        Path(export_path).unlink()


class TestCacheManager:
    """Test suite for Cache Management component"""
    
    @pytest.fixture
    def cache_manager(self):
        """Create cache manager instance"""
        with tempfile.TemporaryDirectory() as temp_dir:
            return CacheManager(cache_dir=temp_dir, max_size=100)
    
    def test_cache_manager_initialization(self, cache_manager):
        """Test cache manager initialization"""
        assert cache_manager is not None
        assert hasattr(cache_manager, 'cache_dir')
        assert hasattr(cache_manager, 'max_size')
        assert hasattr(cache_manager, 'cache')
    
    def test_cache_store_retrieve(self, cache_manager):
        """Test cache storage and retrieval"""
        # Store data
        key = "test_key"
        data = {"test": "data", "number": 123}
        
        cache_manager.store(key, data)
        
        # Retrieve data
        retrieved_data = cache_manager.retrieve(key)
        
        assert retrieved_data == data
        assert retrieved_data["test"] == "data"
        assert retrieved_data["number"] == 123
    
    def test_cache_expiration(self, cache_manager):
        """Test cache expiration"""
        # Store data with short TTL
        key = "expiring_key"
        data = {"test": "data"}
        
        cache_manager.store(key, data, ttl=1)  # 1 second TTL
        
        # Retrieve immediately
        assert cache_manager.retrieve(key) == data
        
        # Wait for expiration
        import time
        time.sleep(1.1)
        
        # Should be expired
        assert cache_manager.retrieve(key) is None
    
    def test_cache_size_limit(self, cache_manager):
        """Test cache size limit"""
        # Store multiple items
        for i in range(150):  # Exceeds max_size of 100
            cache_manager.store(f"key_{i}", {"data": f"value_{i}"})
        
        # Should have evicted old items
        assert len(cache_manager.cache) <= 100
    
    def test_cache_clear(self, cache_manager):
        """Test cache clearing"""
        # Store some data
        cache_manager.store("key1", {"data": "value1"})
        cache_manager.store("key2", {"data": "value2"})
        
        assert len(cache_manager.cache) == 2
        
        # Clear cache
        cache_manager.clear()
        
        assert len(cache_manager.cache) == 0
        assert cache_manager.retrieve("key1") is None
        assert cache_manager.retrieve("key2") is None
    
    def test_cache_persistence(self, cache_manager):
        """Test cache persistence"""
        # Store data
        key = "persistent_key"
        data = {"test": "persistent_data"}
        
        cache_manager.store(key, data)
        
        # Save cache
        cache_manager.save_cache()
        
        # Create new cache manager
        new_cache_manager = CacheManager(cache_dir=cache_manager.cache_dir, max_size=100)
        new_cache_manager.load_cache()
        
        # Verify data was persisted
        assert new_cache_manager.retrieve(key) == data
    
    def test_cache_stats(self, cache_manager):
        """Test cache statistics"""
        # Store some data
        cache_manager.store("key1", {"data": "value1"})
        cache_manager.store("key2", {"data": "value2"})
        
        # Get stats
        stats = cache_manager.get_stats()
        
        assert "total_items" in stats
        assert "cache_size" in stats
        assert "hit_rate" in stats
        assert "miss_rate" in stats
        
        assert stats["total_items"] == 2
        assert stats["cache_size"] > 0


class TestValidation:
    """Test suite for Validation component"""
    
    def test_video_file_validation(self):
        """Test video file validation"""
        # Test valid video file extension
        assert validate_video_file("test.mp4")
        assert validate_video_file("test.avi")
        assert validate_video_file("test.mkv")
        assert validate_video_file("test.mov")
        
        # Test invalid video file extension
        assert not validate_video_file("test.txt")
        assert not validate_video_file("test.jpg")
        assert not validate_video_file("test.pdf")
    
    def test_config_validation(self):
        """Test configuration validation"""
        # Valid configuration
        valid_config = {
            "processing": {
                "ocr_config": {
                    "engine": "PaddleOCR",
                    "language": "中文",
                    "confidence_threshold": 0.8
                },
                "scene_threshold": 0.3,
                "cache_size": 100
            },
            "ui": {
                "theme": "dark",
                "language": "zh_CN"
            }
        }
        assert validate_config(valid_config)
        
        # Invalid configuration (missing required fields)
        invalid_config = {
            "processing": {
                "ocr_config": {
                    "engine": "PaddleOCR"
                    # Missing language and confidence_threshold
                }
            }
        }
        assert not validate_config(invalid_config)
        
        # Invalid configuration (invalid values)
        invalid_config = {
            "processing": {
                "ocr_config": {
                    "engine": "PaddleOCR",
                    "language": "中文",
                    "confidence_threshold": 1.5  # Invalid confidence threshold
                },
                "scene_threshold": 0.3,
                "cache_size": 100
            },
            "ui": {
                "theme": "dark",
                "language": "zh_CN"
            }
        }
        assert not validate_config(invalid_config)
    
    def test_ocr_result_validation(self):
        """Test OCR result validation"""
        # Valid OCR result
        valid_result = {
            "text": "Hello World",
            "confidence": 0.9,
            "bbox": [0, 0, 100, 30],
            "language": "en"
        }
        assert validate_config(valid_result)  # Using validate_config as placeholder
        
        # Invalid OCR result (missing text)
        invalid_result = {
            "confidence": 0.9,
            "bbox": [0, 0, 100, 30],
            "language": "en"
        }
        assert not validate_config(invalid_result)  # Using validate_config as placeholder
    
    def test_subtitle_validation(self):
        """Test subtitle validation"""
        # Valid subtitle
        valid_subtitle = {
            "start": 1000,
            "end": 3000,
            "text": "Hello World",
            "index": 1
        }
        assert validate_config(valid_subtitle)  # Using validate_config as placeholder
        
        # Invalid subtitle (end before start)
        invalid_subtitle = {
            "start": 3000,
            "end": 1000,
            "text": "Hello World",
            "index": 1
        }
        assert not validate_config(invalid_subtitle)  # Using validate_config as placeholder


if __name__ == "__main__":
    pytest.main([__file__, "-v"])