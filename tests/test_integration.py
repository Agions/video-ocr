"""
Integration Testing Suite for VisionSub Application

This module provides comprehensive integration testing including:
- Component interaction testing
- UI-backend integration testing
- Data flow testing
- Event handling testing
- End-to-end workflow testing
"""

import pytest
import asyncio
import numpy as np
import cv2
from unittest.mock import Mock, patch, AsyncMock
from pathlib import Path
import tempfile
import json
import time
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QObject
from PyQt6.QtWidgets import QApplication, QPushButton, QLabel, QLineEdit
from PyQt6.QtTest import QTest

# Add src to path
import sys
sys.path.insert(0, str(Path(__file__).parent / "src"))

from visionsub.ui.enhanced_main_window import EnhancedMainWindow
from visionsub.ui.enhanced_video_player import EnhancedVideoPlayer
from visionsub.ui.enhanced_ocr_preview import EnhancedOCRPreview
from visionsub.ui.enhanced_subtitle_editor import EnhancedSubtitleEditor
from visionsub.video_utils import VideoProcessor
from visionsub.ocr_utils import OCRProcessor, OCRResult
from visionsub.subtitle_utils import SubtitleProcessor, SubtitleEntry
from visionsub.core.config_manager import ConfigManager
from visionsub.models.config import AppConfig, ProcessingConfig, OcrConfig


class TestUIComponentIntegration:
    """Test suite for UI component integration"""
    
    @pytest.fixture
    def main_window(self, qtbot):
        """Create main window instance"""
        window = EnhancedMainWindow()
        qtbot.addWidget(window)
        return window
    
    def test_main_window_component_initialization(self, main_window):
        """Test main window component initialization"""
        assert main_window.video_player is not None
        assert main_window.ocr_preview is not None
        assert main_window.subtitle_editor is not None
        assert main_window.settings_dialog is not None
    
    def test_video_player_ocr_integration(self, main_window, qtbot):
        """Test video player and OCR preview integration"""
        # Create sample video
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as f:
            video_path = f.name
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(video_path, fourcc, 30.0, (640, 480))
        
        # Create frames with text
        for i in range(10):
            frame = np.full((480, 640, 3), 255, dtype=np.uint8)
            cv2.putText(frame, f"Test Frame {i}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
            out.write(frame)
        
        out.release()
        
        # Test video loading
        with patch('PyQt6.QtWidgets.QFileDialog.getOpenFileName') as mock_dialog:
            mock_dialog.return_value = (video_path, "Video Files (*.mp4)")
            
            main_window.load_video()
            assert main_window.video_player.current_video == video_path
        
        # Test frame extraction to OCR
        frame = main_window.video_player.extract_frame(1000)
        assert frame is not None
        
        # Test OCR preview integration
        main_window.ocr_preview.load_image(frame)
        assert main_window.ocr_preview.current_image is not None
        
        Path(video_path).unlink()
    
    def test_ocr_subtitle_integration(self, main_window, qtbot):
        """Test OCR and subtitle editor integration"""
        # Create sample image with text
        image = np.full((200, 400, 3), 255, dtype=np.uint8)
        cv2.putText(image, "Hello World", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        
        # Load image to OCR preview
        main_window.ocr_preview.load_image(image)
        
        # Mock OCR processing
        with patch.object(main_window.ocr_preview, 'process_ocr') as mock_ocr:
            mock_result = OCRResult(
                text="Hello World",
                confidence=0.95,
                bbox=(50, 80, 200, 120),
                language="en"
            )
            main_window.ocr_preview.ocr_results = [mock_result]
            
            # Test transfer to subtitle editor
            main_window.transfer_to_subtitle_editor()
            
            # Verify subtitle was created
            assert len(main_window.subtitle_editor.subtitles) > 0
            assert main_window.subtitle_editor.subtitles[0]["text"] == "Hello World"
    
    def test_settings_integration(self, main_window, qtbot):
        """Test settings dialog integration"""
        # Open settings dialog
        main_window.open_settings()
        
        # Verify settings dialog is shown
        assert main_window.settings_dialog.isVisible()
        
        # Test setting changes
        engine_combo = main_window.settings_dialog.findChild(QComboBox, "engine_combo")
        engine_combo.setCurrentText("TesseractOCR")
        
        # Apply settings
        apply_button = main_window.settings_dialog.findChild(QPushButton, "apply_button")
        QTest.mouseClick(apply_button, Qt.MouseButton.LeftButton)
        
        # Verify settings were applied
        assert main_window.config.processing.ocr_config.engine == "TesseractOCR"
    
    def test_signal_slot_connections(self, main_window):
        """Test signal-slot connections between components"""
        # Test video player signals
        assert hasattr(main_window.video_player, 'frame_extracted')
        assert hasattr(main_window.video_player, 'video_loaded')
        
        # Test OCR preview signals
        assert hasattr(main_window.ocr_preview, 'ocr_completed')
        assert hasattr(main_window.ocr_preview, 'image_loaded')
        
        # Test subtitle editor signals
        assert hasattr(main_window.subtitle_editor, 'subtitle_changed')
        assert hasattr(main_window.subtitle_editor, 'subtitle_exported')
        
        # Test main window signals
        assert hasattr(main_window, 'settings_changed')
        assert hasattr(main_window, 'workflow_completed')
    
    def test_state_synchronization(self, main_window, qtbot):
        """Test state synchronization between components"""
        # Modify video player state
        main_window.video_player.current_video = "/test/video.mp4"
        
        # Verify state is synchronized
        assert main_window.current_video == "/test/video.mp4"
        
        # Modify OCR preview state
        main_window.ocr_preview.current_image = np.zeros((100, 100, 3), dtype=np.uint8)
        
        # Verify state is synchronized
        assert main_window.current_image is not None
        
        # Modify subtitle editor state
        main_window.subtitle_editor.subtitles = [{"text": "Test", "start": 0, "end": 1000}]
        
        # Verify state is synchronized
        assert len(main_window.subtitles) > 0


class TestBackendIntegration:
    """Test suite for backend component integration"""
    
    @pytest.fixture
    def video_processor(self):
        """Create video processor instance"""
        config = ProcessingConfig(
            ocr_config=OcrConfig(
                engine="PaddleOCR",
                language="中文",
                confidence_threshold=0.8
            ),
            scene_threshold=0.3,
            cache_size=100
        )
        return VideoProcessor(config)
    
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
    def subtitle_processor(self):
        """Create subtitle processor instance"""
        return SubtitleProcessor()
    
    def test_video_ocr_integration(self, video_processor, ocr_processor):
        """Test video processing and OCR integration"""
        # Create sample video
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as f:
            video_path = f.name
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(video_path, fourcc, 30.0, (640, 480))
        
        # Create frames with text
        for i in range(5):
            frame = np.full((480, 640, 3), 255, dtype=np.uint8)
            cv2.putText(frame, f"Test Text {i}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
            out.write(frame)
        
        out.release()
        
        # Process video
        info = video_processor.get_video_info(video_path)
        assert info is not None
        
        # Extract frames
        frames = video_processor.extract_frames_batch(video_path, [0, 1000, 2000])
        assert len(frames) == 3
        
        # Process OCR
        with patch.object(ocr_processor, 'process_batch') as mock_ocr:
            mock_results = [
                [OCRResult(text="Test Text 0", confidence=0.9, bbox=(0, 0, 200, 30), language="en")],
                [OCRResult(text="Test Text 1", confidence=0.9, bbox=(0, 0, 200, 30), language="en")],
                [OCRResult(text="Test Text 2", confidence=0.9, bbox=(0, 0, 200, 30), language="en")]
            ]
            mock_ocr.return_value = mock_results
            
            ocr_results = ocr_processor.process_batch(frames)
            assert len(ocr_results) == 3
        
        Path(video_path).unlink()
    
    def test_ocr_subtitle_integration(self, ocr_processor, subtitle_processor):
        """Test OCR and subtitle processing integration"""
        # Create sample OCR results
        ocr_results = [
            OCRResult(text="Hello World", confidence=0.9, bbox=(0, 0, 200, 30), language="en"),
            OCRResult(text="测试文本", confidence=0.85, bbox=(0, 40, 200, 70), language="zh"),
            OCRResult(text="Sample Text", confidence=0.95, bbox=(0, 80, 200, 110), language="en")
        ]
        
        # Create subtitles from OCR results
        timestamps = [0, 2000, 4000]
        subtitles = subtitle_processor.create_subtitles(ocr_results, timestamps)
        
        assert len(subtitles) == 3
        assert subtitles[0].text == "Hello World"
        assert subtitles[1].text == "测试文本"
        assert subtitles[2].text == "Sample Text"
        
        # Test subtitle export
        with tempfile.NamedTemporaryFile(mode='w', suffix='.srt', delete=False) as f:
            export_path = f.name
        
        subtitle_processor.export_srt(subtitles, export_path)
        assert Path(export_path).exists()
        
        # Test subtitle import
        imported_subtitles = subtitle_processor.import_srt(export_path)
        assert len(imported_subtitles) == 3
        assert imported_subtitles[0].text == "Hello World"
        
        Path(export_path).unlink()
    
    def test_config_integration(self):
        """Test configuration management integration"""
        # Create sample configuration
        config = AppConfig(
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
        
        # Test config manager
        config_manager = ConfigManager()
        
        # Save configuration
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            config_path = f.name
        
        config_manager.save_config(config, config_path)
        
        # Load configuration
        loaded_config = config_manager.load_config(config_path)
        
        # Verify configuration
        assert loaded_config.processing.ocr_config.engine == "PaddleOCR"
        assert loaded_config.processing.ocr_config.language == "中文"
        assert loaded_config.processing.ocr_config.confidence_threshold == 0.8
        assert loaded_config.processing.scene_threshold == 0.3
        assert loaded_config.processing.cache_size == 100
        assert loaded_config.ui["theme"] == "dark"
        assert loaded_config.ui["language"] == "zh_CN"
        
        Path(config_path).unlink()
    
    def test_cache_integration(self):
        """Test cache management integration"""
        with tempfile.TemporaryDirectory() as temp_dir:
            cache_manager = CacheManager(cache_dir=temp_dir, max_size=100)
            
            # Test cache storage
            cache_manager.store("test_key", {"test": "data"})
            
            # Test cache retrieval
            data = cache_manager.retrieve("test_key")
            assert data == {"test": "data"}
            
            # Test cache persistence
            cache_manager.save_cache()
            
            # Create new cache manager
            new_cache_manager = CacheManager(cache_dir=temp_dir, max_size=100)
            new_cache_manager.load_cache()
            
            # Verify data was persisted
            data = new_cache_manager.retrieve("test_key")
            assert data == {"test": "data"}


class TestEventHandlingIntegration:
    """Test suite for event handling integration"""
    
    @pytest.fixture
    def main_window(self, qtbot):
        """Create main window instance"""
        window = EnhancedMainWindow()
        qtbot.addWidget(window)
        return window
    
    def test_video_loaded_event(self, main_window, qtbot):
        """Test video loaded event handling"""
        # Create sample video
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as f:
            video_path = f.name
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(video_path, fourcc, 30.0, (640, 480))
        
        for i in range(5):
            frame = np.full((480, 640, 3), 255, dtype=np.uint8)
            out.write(frame)
        
        out.release()
        
        # Test video loaded event
        with patch('PyQt6.QtWidgets.QFileDialog.getOpenFileName') as mock_dialog:
            mock_dialog.return_value = (video_path, "Video Files (*.mp4)")
            
            main_window.load_video()
            
            # Verify event was handled
            assert main_window.current_video == video_path
            assert main_window.video_player.current_video == video_path
            assert main_window.statusBar().currentMessage() != ""
        
        Path(video_path).unlink()
    
    def test_ocr_completed_event(self, main_window, qtbot):
        """Test OCR completed event handling"""
        # Create sample image
        image = np.full((200, 400, 3), 255, dtype=np.uint8)
        cv2.putText(image, "Test Text", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        
        # Load image to OCR preview
        main_window.ocr_preview.load_image(image)
        
        # Mock OCR completion
        mock_result = OCRResult(
            text="Test Text",
            confidence=0.9,
            bbox=(50, 80, 200, 120),
            language="en"
        )
        main_window.ocr_preview.ocr_results = [mock_result]
        
        # Trigger OCR completed event
        main_window.ocr_preview.ocr_completed.emit([mock_result])
        
        # Verify event was handled
        assert len(main_window.ocr_results) > 0
        assert main_window.statusBar().currentMessage() != ""
    
    def test_subtitle_changed_event(self, main_window, qtbot):
        """Test subtitle changed event handling"""
        # Create sample subtitle
        subtitle = {
            "start": 0,
            "end": 2000,
            "text": "Test Subtitle",
            "index": 1
        }
        
        # Load subtitle to editor
        main_window.subtitle_editor.subtitles = [subtitle]
        
        # Trigger subtitle changed event
        main_window.subtitle_editor.subtitle_changed.emit([subtitle])
        
        # Verify event was handled
        assert len(main_window.subtitles) > 0
        assert main_window.subtitles[0]["text"] == "Test Subtitle"
    
    def test_settings_changed_event(self, main_window, qtbot):
        """Test settings changed event handling"""
        # Open settings dialog
        main_window.open_settings()
        
        # Change settings
        engine_combo = main_window.settings_dialog.findChild(QComboBox, "engine_combo")
        engine_combo.setCurrentText("EasyOCR")
        
        # Apply settings
        apply_button = main_window.settings_dialog.findChild(QPushButton, "apply_button")
        QTest.mouseClick(apply_button, Qt.MouseButton.LeftButton)
        
        # Verify settings changed event was handled
        assert main_window.config.processing.ocr_config.engine == "EasyOCR"
        assert main_window.statusBar().currentMessage() != ""


class TestWorkflowIntegration:
    """Test suite for complete workflow integration"""
    
    @pytest.fixture
    def main_window(self, qtbot):
        """Create main window instance"""
        window = EnhancedMainWindow()
        qtbot.addWidget(window)
        return window
    
    def test_complete_workflow(self, main_window, qtbot):
        """Test complete workflow from video to subtitles"""
        # Create sample video
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as f:
            video_path = f.name
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(video_path, fourcc, 30.0, (640, 480))
        
        # Create frames with text
        for i in range(5):
            frame = np.full((480, 640, 3), 255, dtype=np.uint8)
            cv2.putText(frame, f"Subtitle {i}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
            out.write(frame)
        
        out.release()
        
        # Step 1: Load video
        with patch('PyQt6.QtWidgets.QFileDialog.getOpenFileName') as mock_dialog:
            mock_dialog.return_value = (video_path, "Video Files (*.mp4)")
            
            main_window.load_video()
            assert main_window.current_video == video_path
        
        # Step 2: Extract frame
        frame = main_window.video_player.extract_frame(1000)
        assert frame is not None
        
        # Step 3: Process OCR
        main_window.ocr_preview.load_image(frame)
        
        with patch.object(main_window.ocr_preview, 'process_ocr') as mock_ocr:
            mock_result = OCRResult(
                text="Subtitle 1",
                confidence=0.9,
                bbox=(50, 80, 200, 120),
                language="en"
            )
            main_window.ocr_preview.ocr_results = [mock_result]
            
            main_window.ocr_preview.process_ocr()
            assert len(main_window.ocr_preview.ocr_results) > 0
        
        # Step 4: Create subtitles
        main_window.transfer_to_subtitle_editor()
        assert len(main_window.subtitle_editor.subtitles) > 0
        
        # Step 5: Export subtitles
        with tempfile.NamedTemporaryFile(mode='w', suffix='.srt', delete=False) as f:
            export_path = f.name
        
        main_window.subtitle_editor.export_srt(export_path)
        assert Path(export_path).exists()
        
        # Step 6: Verify exported subtitles
        with open(export_path, 'r', encoding='utf-8') as f:
            content = f.read()
            assert "Subtitle 1" in content
        
        Path(video_path).unlink()
        Path(export_path).unlink()
    
    def test_error_handling_workflow(self, main_window, qtbot):
        """Test error handling in workflow"""
        # Test with invalid video file
        invalid_video = "/nonexistent/video.mp4"
        
        with patch('PyQt6.QtWidgets.QFileDialog.getOpenFileName') as mock_dialog:
            mock_dialog.return_value = (invalid_video, "Video Files (*.mp4)")
            
            with patch('PyQt6.QtWidgets.QMessageBox.critical') as mock_message:
                main_window.load_video()
                mock_message.assert_called_once()
        
        # Test with invalid image
        invalid_image = None
        
        main_window.ocr_preview.load_image(invalid_image)
        
        # Should handle gracefully
        assert main_window.ocr_preview.current_image is None
        
        # Test with empty OCR results
        main_window.ocr_preview.ocr_results = []
        main_window.transfer_to_subtitle_editor()
        
        # Should handle gracefully
        assert len(main_window.subtitle_editor.subtitles) == 0
    
    def test_state_persistence_workflow(self, main_window, qtbot):
        """Test state persistence in workflow"""
        # Create sample video
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as f:
            video_path = f.name
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(video_path, fourcc, 30.0, (640, 480))
        
        for i in range(3):
            frame = np.full((480, 640, 3), 255, dtype=np.uint8)
            out.write(frame)
        
        out.release()
        
        # Load video
        with patch('PyQt6.QtWidgets.QFileDialog.getOpenFileName') as mock_dialog:
            mock_dialog.return_value = (video_path, "Video Files (*.mp4)")
            
            main_window.load_video()
        
        # Create some state
        main_window.current_project = {
            "video_path": video_path,
            "subtitles": [{"text": "Test", "start": 0, "end": 1000}],
            "settings": {"theme": "dark"}
        }
        
        # Save state
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            state_path = f.name
        
        main_window.save_state(state_path)
        
        # Create new window and load state
        new_window = EnhancedMainWindow()
        new_window.load_state(state_path)
        
        # Verify state was loaded
        assert new_window.current_project["video_path"] == video_path
        assert len(new_window.current_project["subtitles"]) > 0
        
        Path(video_path).unlink()
        Path(state_path).unlink()


class TestAsyncIntegration:
    """Test suite for async integration testing"""
    
    @pytest.mark.asyncio
    async def test_async_video_processing(self):
        """Test async video processing"""
        # Create sample video
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as f:
            video_path = f.name
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(video_path, fourcc, 30.0, (640, 480))
        
        for i in range(10):
            frame = np.full((480, 640, 3), 255, dtype=np.uint8)
            out.write(frame)
        
        out.release()
        
        # Test async processing
        config = ProcessingConfig(
            ocr_config=OcrConfig(
                engine="PaddleOCR",
                language="中文",
                confidence_threshold=0.8
            ),
            scene_threshold=0.3,
            cache_size=100
        )
        
        processor = VideoProcessor(config)
        
        # Get video info
        info = await asyncio.to_thread(processor.get_video_info, video_path)
        assert info is not None
        
        # Extract frames async
        frames = await asyncio.to_thread(
            processor.extract_frames_batch, 
            video_path, 
            [0, 1000, 2000]
        )
        assert len(frames) == 3
        
        Path(video_path).unlink()
    
    @pytest.mark.asyncio
    async def test_async_ocr_processing(self):
        """Test async OCR processing"""
        # Create sample images
        images = []
        for i in range(5):
            image = np.full((200, 400, 3), 255, dtype=np.uint8)
            cv2.putText(image, f"Test {i}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
            images.append(image)
        
        # Test async OCR processing
        config = OcrConfig(
            engine="PaddleOCR",
            language="中文",
            confidence_threshold=0.8
        )
        
        processor = OCRProcessor(config)
        
        # Process images async
        with patch.object(processor, 'process_batch') as mock_ocr:
            mock_results = [
                [OCRResult(text=f"Test {i}", confidence=0.9, bbox=(0, 0, 200, 30), language="en")]
                for i in range(5)
            ]
            mock_ocr.return_value = mock_results
            
            results = await asyncio.to_thread(processor.process_batch, images)
            assert len(results) == 5
    
    @pytest.mark.asyncio
    async def test_concurrent_processing(self):
        """Test concurrent processing"""
        # Create test data
        videos = []
        for i in range(3):
            with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as f:
                video_path = f.name
                
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(video_path, fourcc, 30.0, (640, 480))
            
            for j in range(5):
                frame = np.full((480, 640, 3), 255, dtype=np.uint8)
                out.write(frame)
            
            out.release()
            videos.append(video_path)
        
        # Process videos concurrently
        async def process_video(video_path):
            config = ProcessingConfig(
                ocr_config=OcrConfig(
                    engine="PaddleOCR",
                    language="中文",
                    confidence_threshold=0.8
                ),
                scene_threshold=0.3,
                cache_size=100
            )
            
            processor = VideoProcessor(config)
            return await asyncio.to_thread(processor.get_video_info, video_path)
        
        tasks = [process_video(video_path) for video_path in videos]
        results = await asyncio.gather(*tasks)
        
        assert len(results) == 3
        for result in results:
            assert result is not None
        
        # Cleanup
        for video_path in videos:
            Path(video_path).unlink()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])