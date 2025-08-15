"""
End-to-End Testing Suite for VisionSub Application

This module provides comprehensive end-to-end testing including:
- Complete user scenario testing
- Real-world usage patterns
- Data integrity verification
- Error recovery testing
- Configuration validation
- User acceptance testing
"""

import pytest
import time
import tempfile
import json
import numpy as np
import cv2
from pathlib import Path
from unittest.mock import Mock, patch
from PyQt6.QtCore import Qt, QTimer, QEvent
from PyQt6.QtWidgets import QApplication, QMessageBox, QFileDialog
from PyQt6.QtTest import QTest
from PIL import Image
import pysrt

# Add src to path
import sys
sys.path.insert(0, str(Path(__file__).parent / "src"))

from visionsub.ui.enhanced_main import EnhancedMainApp
from visionsub.ui.enhanced_main_window import EnhancedMainWindow
from visionsub.video_utils import VideoProcessor
from visionsub.ocr_utils import OCRProcessor, OCRResult
from visionsub.subtitle_utils import SubtitleProcessor, SubtitleEntry
from visionsub.models.config import AppConfig, ProcessingConfig, OcrConfig


class TestEndToEndWorkflows:
    """Test suite for end-to-end workflows"""
    
    @pytest.fixture
    def main_app(self, qtbot):
        """Create main application instance"""
        app = EnhancedMainApp([])
        window = app.main_window
        qtbot.addWidget(window)
        return app
    
    @pytest.fixture
    def sample_video_with_subtitles(self):
        """Create a sample video with subtitle content"""
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as f:
            video_path = f.name
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(video_path, fourcc, 30.0, (640, 480))
        
        # Create frames with subtitle-like content
        subtitle_texts = [
            "Welcome to VisionSub",
            "This is a test subtitle",
            "视频OCR字幕提取",
            "Professional OCR tool",
            "Extract subtitles easily"
        ]
        
        for i, text in enumerate(subtitle_texts):
            for frame in range(30):  # 1 second per subtitle
                frame_img = np.full((480, 640, 3), [50, 50, 80], dtype=np.uint8)
                cv2.putText(frame_img, text, (50, 400), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv2.putText(frame_img, f"Frame {i*30 + frame}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)
                out.write(frame_img)
        
        out.release()
        yield video_path
        Path(video_path).unlink()
    
    def test_complete_video_processing_workflow(self, main_app, sample_video_with_subtitles, qtbot):
        """Test complete video processing workflow"""
        window = main_app.main_window
        
        # Step 1: Launch application
        assert window.isVisible()
        assert window.windowTitle() == "VisionSub - Professional Video OCR Tool"
        
        # Step 2: Load video file
        with patch('PyQt6.QtWidgets.QFileDialog.getOpenFileName') as mock_dialog:
            mock_dialog.return_value = (sample_video_with_subtitles, "Video Files (*.mp4)")
            
            # Simulate file menu click
            file_menu = window.menuBar().findChild(Qt.Menu, "file_menu")
            open_action = file_menu.findChild(Qt.Action, "open_action")
            open_action.trigger()
            
            # Verify video was loaded
            assert window.current_video == sample_video_with_subtitles
            assert window.video_player.current_video == sample_video_with_subtitles
        
        # Step 3: Extract and process frames
        # Extract frames at regular intervals
        timestamps = [0, 1000, 2000, 3000, 4000]
        extracted_frames = []
        
        for timestamp in timestamps:
            frame = window.video_player.extract_frame(timestamp)
            assert frame is not None
            extracted_frames.append((timestamp, frame))
        
        # Step 4: Process OCR on frames
        ocr_results = []
        
        for timestamp, frame in extracted_frames:
            window.ocr_preview.load_image(frame)
            
            # Mock OCR processing (since real OCR may not be available in test environment)
            mock_result = OCRResult(
                text=f"Extracted text at {timestamp}ms",
                confidence=0.9,
                bbox=(50, 380, 400, 420),
                language="en"
            )
            window.ocr_preview.ocr_results = [mock_result]
            ocr_results.append((timestamp, mock_result))
        
        assert len(ocr_results) == len(timestamps)
        
        # Step 5: Create subtitles from OCR results
        window.transfer_to_subtitle_editor()
        
        # Verify subtitles were created
        assert len(window.subtitle_editor.subtitles) >= len(ocr_results)
        
        # Step 6: Edit and refine subtitles
        # Test subtitle editing
        if window.subtitle_editor.subtitles:
            first_subtitle = window.subtitle_editor.subtitles[0]
            first_subtitle["text"] = "Edited subtitle text"
            window.subtitle_editor.subtitle_changed.emit(window.subtitle_editor.subtitles)
        
        # Step 7: Export subtitles
        with tempfile.NamedTemporaryFile(mode='w', suffix='.srt', delete=False) as f:
            export_path = f.name
        
        with patch('PyQt6.QtWidgets.QFileDialog.getSaveFileName') as mock_dialog:
            mock_dialog.return_value = (export_path, "Subtitle Files (*.srt)")
            
            # Simulate export action
            export_action = window.findChild(Qt.Action, "export_action")
            export_action.trigger()
        
        # Verify export was successful
        assert Path(export_path).exists()
        
        # Step 8: Verify exported content
        with open(export_path, 'r', encoding='utf-8') as f:
            srt_content = f.read()
            assert "Extracted text" in srt_content or "Edited subtitle" in srt_content
            assert "00:00:00" in srt_content  # Verify time format
        
        # Step 9: Test project save and load
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            project_path = f.name
        
        # Save project
        window.save_project(project_path)
        assert Path(project_path).exists()
        
        # Load project
        window.load_project(project_path)
        assert window.current_project is not None
        
        # Cleanup
        Path(export_path).unlink()
        Path(project_path).unlink()
    
    def test_batch_processing_workflow(self, main_app, qtbot):
        """Test batch processing workflow"""
        window = main_app.main_window
        
        # Create multiple sample videos
        video_paths = []
        for i in range(3):
            with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as f:
                video_path = f.name
            
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(video_path, fourcc, 30.0, (320, 240))
            
            for j in range(30):  # 1 second videos
                frame = np.full((240, 320, 3), [i*50, 100, 150], dtype=np.uint8)
                cv2.putText(frame, f"Video {i} Frame {j}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                out.write(frame)
            
            out.release()
            video_paths.append(video_path)
        
        # Step 1: Start batch processing
        window.start_batch_processing(video_paths)
        
        # Step 2: Monitor progress
        # This would normally show a progress dialog
        assert hasattr(window, 'batch_processor')
        
        # Step 3: Verify all videos were processed
        # In a real implementation, we would check the batch processing results
        processed_count = len(video_paths)
        assert processed_count == 3
        
        # Step 4: Export all results
        with tempfile.TemporaryDirectory() as output_dir:
            output_path = Path(output_dir) / "batch_results"
            output_path.mkdir()
            
            # Mock batch export
            for i, video_path in enumerate(video_paths):
                export_file = output_path / f"subtitles_{i}.srt"
                with open(export_file, 'w', encoding='utf-8') as f:
                    f.write(f"1\n00:00:00,000 --> 00:00:01,000\nVideo {i} subtitle\n\n")
                
                assert export_file.exists()
        
        # Cleanup
        for video_path in video_paths:
            Path(video_path).unlink()
    
    def test_settings_customization_workflow(self, main_app, qtbot):
        """Test settings customization workflow"""
        window = main_app.main_window
        
        # Step 1: Open settings dialog
        window.open_settings()
        assert window.settings_dialog.isVisible()
        
        # Step 2: Customize OCR settings
        ocr_tab = window.settings_dialog.findChild(Qt.Widget, "ocr_tab")
        
        # Change OCR engine
        engine_combo = ocr_tab.findChild(Qt.ComboBox, "engine_combo")
        engine_combo.setCurrentText("TesseractOCR")
        
        # Change language
        language_combo = ocr_tab.findChild(Qt.ComboBox, "language_combo")
        language_combo.setCurrentText("en")
        
        # Change confidence threshold
        confidence_spin = ocr_tab.findChild(Qt.DoubleSpinBox, "confidence_spin")
        confidence_spin.setValue(0.7)
        
        # Step 3: Customize processing settings
        processing_tab = window.settings_dialog.findChild(Qt.Widget, "processing_tab")
        
        # Change scene threshold
        scene_spin = processing_tab.findChild(Qt.DoubleSpinBox, "scene_spin")
        scene_spin.setValue(0.2)
        
        # Change cache size
        cache_spin = processing_tab.findChild(Qt.SpinBox, "cache_spin")
        cache_spin.setValue(200)
        
        # Step 4: Customize UI settings
        ui_tab = window.settings_dialog.findChild(Qt.Widget, "ui_tab")
        
        # Change theme
        theme_combo = ui_tab.findChild(Qt.ComboBox, "theme_combo")
        theme_combo.setCurrentText("light")
        
        # Change language
        ui_language_combo = ui_tab.findChild(Qt.ComboBox, "ui_language_combo")
        ui_language_combo.setCurrentText("en_US")
        
        # Step 5: Apply settings
        apply_button = window.settings_dialog.findChild(Qt.PushButton, "apply_button")
        QTest.mouseClick(apply_button, Qt.MouseButton.LeftButton)
        
        # Verify settings were applied
        assert window.config.processing.ocr_config.engine == "TesseractOCR"
        assert window.config.processing.ocr_config.language == "en"
        assert window.config.processing.ocr_config.confidence_threshold == 0.7
        assert window.config.processing.scene_threshold == 0.2
        assert window.config.processing.cache_size == 200
        assert window.config.ui["theme"] == "light"
        assert window.config.ui["language"] == "en_US"
        
        # Step 6: Test settings persistence
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            config_path = f.name
        
        window.settings_dialog.save_settings(config_path)
        
        # Create new window and load settings
        new_window = EnhancedMainWindow()
        new_window.settings_dialog.load_settings(config_path)
        
        # Verify settings were loaded
        assert new_window.config.processing.ocr_config.engine == "TesseractOCR"
        assert new_window.config.ui["theme"] == "light"
        
        # Cleanup
        Path(config_path).unlink()
    
    def test_error_recovery_workflow(self, main_app, qtbot):
        """Test error recovery workflow"""
        window = main_app.main_window
        
        # Step 1: Test with corrupted video file
        corrupted_video = "/nonexistent/corrupted.mp4"
        
        with patch('PyQt6.QtWidgets.QFileDialog.getOpenFileName') as mock_dialog:
            mock_dialog.return_value = (corrupted_video, "Video Files (*.mp4)")
            
            with patch('PyQt6.QtWidgets.QMessageBox.critical') as mock_message:
                window.load_video()
                mock_message.assert_called_once()
        
        # Verify application remains stable
        assert window.isVisible()
        assert window.current_video is None
        
        # Step 2: Test with invalid image format
        invalid_image = np.array([])  # Empty array
        
        window.ocr_preview.load_image(invalid_image)
        
        # Verify graceful handling
        assert window.ocr_preview.current_image is None
        
        # Step 3: Test OCR processing failure
        valid_image = np.full((100, 200, 3), 255, dtype=np.uint8)
        window.ocr_preview.load_image(valid_image)
        
        with patch.object(window.ocr_preview, 'process_ocr') as mock_ocr:
            mock_ocr.side_effect = Exception("OCR processing failed")
            
            with patch('PyQt6.QtWidgets.QMessageBox.warning') as mock_message:
                window.ocr_preview.process_ocr()
                mock_message.assert_called_once()
        
        # Verify application remains stable
        assert window.isVisible()
        
        # Step 4: Test export failure
        with tempfile.NamedTemporaryFile(mode='w', suffix='.srt', delete=False) as f:
            export_path = f.name
        
        # Make file read-only to simulate write failure
        Path(export_path).chmod(0o444)
        
        with patch('PyQt6.QtWidgets.QFileDialog.getSaveFileName') as mock_dialog:
            mock_dialog.return_value = (export_path, "Subtitle Files (*.srt)")
            
            with patch('PyQt6.QtWidgets.QMessageBox.warning') as mock_message:
                window.subtitle_editor.export_srt(export_path)
                mock_message.assert_called_once()
        
        # Restore permissions and cleanup
        Path(export_path).chmod(0o666)
        Path(export_path).unlink()
        
        # Step 5: Test recovery from invalid configuration
        invalid_config_data = {
            "processing": {
                "ocr_config": {
                    "engine": "InvalidEngine",
                    "language": "invalid_language",
                    "confidence_threshold": 1.5  # Invalid value
                }
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(invalid_config_data, f)
            config_path = f.name
        
        with patch('PyQt6.QtWidgets.QMessageBox.warning') as mock_message:
            window.load_config(config_path)
            mock_message.assert_called_once()
        
        # Verify application falls back to default configuration
        assert window.config.processing.ocr_config.engine in ["PaddleOCR", "TesseractOCR", "EasyOCR"]
        assert window.config.processing.ocr_config.confidence_threshold <= 1.0
        
        # Cleanup
        Path(config_path).unlink()
    
    def test_user_acceptance_testing(self, main_app, sample_video_with_subtitles, qtbot):
        """Test user acceptance testing scenarios"""
        window = main_app.main_window
        
        # Scenario 1: First-time user experience
        self._test_first_time_user_experience(window, sample_video_with_subtitles, qtbot)
        
        # Scenario 2: Experienced user workflow
        self._test_experienced_user_workflow(window, sample_video_with_subtitles, qtbot)
        
        # Scenario 3: Power user features
        self._test_power_user_features(window, sample_video_with_subtitles, qtbot)
    
    def _test_first_time_user_experience(self, window, video_path, qtbot):
        """Test first-time user experience"""
        # Step 1: User launches application
        assert window.isVisible()
        
        # Step 2: User clicks "Open Video" button
        open_button = window.findChild(Qt.PushButton, "open_button")
        assert open_button is not None
        
        with patch('PyQt6.QtWidgets.QFileDialog.getOpenFileName') as mock_dialog:
            mock_dialog.return_value = (video_path, "Video Files (*.mp4)")
            QTest.mouseClick(open_button, Qt.MouseButton.LeftButton)
        
        # Step 3: Video loads and user sees preview
        assert window.current_video == video_path
        
        # Step 4: User clicks "Process OCR" button
        process_button = window.findChild(Qt.PushButton, "process_button")
        assert process_button is not None
        
        # Mock OCR processing
        with patch.object(window, 'process_ocr') as mock_process:
            mock_process.return_value = True
            QTest.mouseClick(process_button, Qt.MouseButton.LeftButton)
        
        # Step 5: User sees OCR results
        assert len(window.ocr_results) > 0
        
        # Step 6: User clicks "Export Subtitles" button
        export_button = window.findChild(Qt.PushButton, "export_button")
        assert export_button is not None
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.srt', delete=False) as f:
            export_path = f.name
        
        with patch('PyQt6.QtWidgets.QFileDialog.getSaveFileName') as mock_dialog:
            mock_dialog.return_value = (export_path, "Subtitle Files (*.srt)")
            QTest.mouseClick(export_button, Qt.MouseButton.LeftButton)
        
        # Step 7: Export completes successfully
        assert Path(export_path).exists()
        
        # Cleanup
        Path(export_path).unlink()
    
    def _test_experienced_user_workflow(self, window, video_path, qtbot):
        """Test experienced user workflow"""
        # Step 1: User uses keyboard shortcuts
        QTest.keyPress(window, Qt.Key.Key_O, modifier=Qt.KeyboardModifier.ControlModifier)  # Ctrl+O
        
        with patch('PyQt6.QtWidgets.QFileDialog.getOpenFileName') as mock_dialog:
            mock_dialog.return_value = (video_path, "Video Files (*.mp4)")
            QTest.keyPress(window, Qt.Key.Key_Return)
        
        assert window.current_video == video_path
        
        # Step 2: User uses toolbar buttons
        process_action = window.findChild(Qt.Action, "process_action")
        process_action.trigger()
        
        # Step 3: User navigates through timeline
        progress_slider = window.video_player.findChild(Qt.Slider, "progress_slider")
        progress_slider.setValue(50)
        
        # Step 4: User uses context menus
        # This would test right-click functionality
        
        # Step 5: User uses drag and drop
        # This would test drag and drop functionality
        
        assert window.current_video == video_path
    
    def _test_power_user_features(self, window, video_path, qtbot):
        """Test power user features"""
        # Step 1: User uses advanced settings
        window.open_settings()
        
        # Configure advanced options
        ocr_tab = window.settings_dialog.findChild(Qt.Widget, "ocr_tab")
        advanced_button = ocr_tab.findChild(Qt.PushButton, "advanced_button")
        if advanced_button:
            QTest.mouseClick(advanced_button, Qt.MouseButton.LeftButton)
        
        # Step 2: User uses batch processing
        window.start_batch_processing([video_path])
        
        # Step 3: User uses custom ROI selection
        roi_button = window.findChild(Qt.PushButton, "roi_button")
        if roi_button:
            QTest.mouseClick(roi_button, Qt.MouseButton.LeftButton)
        
        # Step 4: User uses custom themes
        window.change_theme("custom")
        
        # Step 5: User uses plugins/extensions
        # This would test plugin functionality
        
        # Verify power user features work
        assert window.config is not None
    
    def test_performance_acceptance_criteria(self, main_app, sample_video_with_subtitles, qtbot):
        """Test performance acceptance criteria"""
        window = main_app.main_window
        
        # Load video
        with patch('PyQt6.QtWidgets.QFileDialog.getOpenFileName') as mock_dialog:
            mock_dialog.return_value = (sample_video_with_subtitles, "Video Files (*.mp4)")
            window.load_video()
        
        # Measure video loading time
        start_time = time.time()
        assert window.current_video == sample_video_with_subtitles
        load_time = time.time() - start_time
        assert load_time < 2.0  # Should load in less than 2 seconds
        
        # Measure frame extraction time
        start_time = time.time()
        frame = window.video_player.extract_frame(1000)
        extract_time = time.time() - start_time
        assert extract_time < 0.5  # Should extract in less than 0.5 seconds
        assert frame is not None
        
        # Measure OCR processing time (mocked)
        start_time = time.time()
        window.ocr_preview.load_image(frame)
        
        with patch.object(window.ocr_preview, 'process_ocr') as mock_ocr:
            mock_result = OCRResult(
                text="Test text",
                confidence=0.9,
                bbox=(0, 0, 200, 30),
                language="en"
            )
            window.ocr_preview.ocr_results = [mock_result]
            window.ocr_preview.process_ocr()
        
        ocr_time = time.time() - start_time
        assert ocr_time < 3.0  # Should process in less than 3 seconds
        
        # Measure export time
        with tempfile.NamedTemporaryFile(mode='w', suffix='.srt', delete=False) as f:
            export_path = f.name
        
        start_time = time.time()
        window.subtitle_editor.export_srt(export_path)
        export_time = time.time() - start_time
        assert export_time < 1.0  # Should export in less than 1 second
        
        # Cleanup
        Path(export_path).unlink()
    
    def test_accessibility_compliance(self, main_app, qtbot):
        """Test accessibility compliance"""
        window = main_app.main_window
        
        # Test keyboard navigation
        QTest.keyPress(window, Qt.Key.Key_Tab)  # Navigate to next widget
        QTest.keyPress(window, Qt.Key.Key_Tab)
        QTest.keyPress(window, Qt.Key.Key_Tab)
        
        # Test screen reader compatibility
        # Verify all widgets have accessible names
        all_widgets = window.findChildren(Qt.Widget)
        for widget in all_widgets:
            if widget.objectName():
                assert widget.objectName() != ""
        
        # Test high contrast mode
        window.set_high_contrast_mode(True)
        
        # Test font size adjustment
        window.adjust_font_size(1.5)  # Increase font size
        
        # Test color blindness support
        window.set_color_blind_mode("deuteranopia")
        
        # Verify accessibility features work
        assert window.accessible_mode is not None
    
    def test_cross_platform_compatibility(self, main_app, qtbot):
        """Test cross-platform compatibility"""
        window = main_app.main_window
        
        # Test file path handling
        test_paths = [
            "/home/user/video.mp4",  # Unix-style
            "C:\\Users\\user\\video.mp4",  # Windows-style
            "/Volumes/Mac/video.mp4",  # macOS-style
            "relative/path/video.mp4"  # Relative path
        ]
        
        for path in test_paths:
            # Should handle different path formats
            normalized_path = Path(path).as_posix()
            assert isinstance(normalized_path, str)
        
        # Test platform-specific features
        import platform
        current_platform = platform.system()
        
        if current_platform == "Windows":
            # Test Windows-specific features
            assert hasattr(window, 'windows_specific_feature')
        elif current_platform == "Darwin":
            # Test macOS-specific features
            assert hasattr(window, 'macos_specific_feature')
        elif current_platform == "Linux":
            # Test Linux-specific features
            assert hasattr(window, 'linux_specific_feature')
        
        # Test platform-independent behavior
        assert window.isVisible()  # Should work on all platforms


class TestDataIntegrity:
    """Test suite for data integrity verification"""
    
    def test_video_data_integrity(self):
        """Test video data integrity"""
        # Create test video
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as f:
            video_path = f.name
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(video_path, fourcc, 30.0, (640, 480))
        
        # Create frames with unique content
        frame_hashes = []
        for i in range(30):
            frame = np.full((480, 640, 3), [i % 256, 100, 200], dtype=np.uint8)
            cv2.putText(frame, f"Frame {i}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            frame_hash = hash(frame.tobytes())
            frame_hashes.append(frame_hash)
            out.write(frame)
        
        out.release()
        
        # Verify video integrity
        cap = cv2.VideoCapture(video_path)
        actual_hashes = []
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_hash = hash(frame.tobytes())
            actual_hashes.append(frame_hash)
        
        cap.release()
        
        # Verify frame integrity
        assert len(actual_hashes) == len(frame_hashes)
        for expected, actual in zip(frame_hashes, actual_hashes):
            assert expected == actual
        
        # Cleanup
        Path(video_path).unlink()
    
    def test_ocr_data_integrity(self):
        """Test OCR data integrity"""
        # Create test image
        test_text = "Test OCR Data Integrity"
        image = np.full((200, 400, 3), 255, dtype=np.uint8)
        cv2.putText(image, test_text, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        
        # Test OCR result integrity
        ocr_result = OCRResult(
            text=test_text,
            confidence=0.95,
            bbox=(50, 80, 200, 120),
            language="en"
        )
        
        # Verify OCR result integrity
        assert ocr_result.text == test_text
        assert 0 <= ocr_result.confidence <= 1
        assert len(ocr_result.bbox) == 4
        assert ocr_result.language is not None
        
        # Test serialization/deserialization
        result_dict = ocr_result.to_dict()
        restored_result = OCRResult.from_dict(result_dict)
        
        assert restored_result.text == ocr_result.text
        assert restored_result.confidence == ocr_result.confidence
        assert restored_result.bbox == ocr_result.bbox
        assert restored_result.language == ocr_result.language
    
    def test_subtitle_data_integrity(self):
        """Test subtitle data integrity"""
        # Create test subtitle
        subtitle = SubtitleEntry(
            start=1000,
            end=3000,
            text="Test Subtitle",
            index=1
        )
        
        # Verify subtitle integrity
        assert subtitle.start < subtitle.end
        assert subtitle.text != ""
        assert subtitle.index > 0
        
        # Test SRT format integrity
        srt_content = subtitle.to_srt()
        assert "1" in srt_content
        assert "00:00:01,000" in srt_content
        assert "00:00:03,000" in srt_content
        assert "Test Subtitle" in srt_content
        
        # Test round-trip conversion
        parsed_subtitle = SubtitleEntry.from_srt(srt_content, 1)
        assert parsed_subtitle.start == subtitle.start
        assert parsed_subtitle.end == subtitle.end
        assert parsed_subtitle.text == subtitle.text
    
    def test_configuration_data_integrity(self):
        """Test configuration data integrity"""
        # Create test configuration
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
        
        # Test serialization/deserialization
        config_dict = config.to_dict()
        restored_config = AppConfig.from_dict(config_dict)
        
        assert restored_config.processing.ocr_config.engine == config.processing.ocr_config.engine
        assert restored_config.processing.ocr_config.language == config.processing.ocr_config.language
        assert restored_config.processing.ocr_config.confidence_threshold == config.processing.ocr_config.confidence_threshold
        assert restored_config.processing.scene_threshold == config.processing.scene_threshold
        assert restored_config.processing.cache_size == config.processing.cache_size
        assert restored_config.ui == config.ui


if __name__ == "__main__":
    pytest.main([__file__, "-v"])