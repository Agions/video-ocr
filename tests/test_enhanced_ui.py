"""
Enhanced UI Testing Suite for VisionSub

This module provides comprehensive testing for all UI components including:
- Enhanced video player
- OCR preview and editing
- Theme system
- Advanced settings
- ROI selection
- Subtitle editing
"""

import pytest
import asyncio
import numpy as np
import cv2
from unittest.mock import Mock, patch, AsyncMock
from pathlib import Path
import tempfile
import json
from PyQt6.QtCore import Qt, QTimer, QEvent, QRect
from PyQt6.QtGui import QImage, QPixmap, QFont, QColor
from PyQt6.QtWidgets import QApplication, QPushButton, QLabel, QLineEdit, QComboBox
from PyQt6.QtTest import QTest

# Add src to path
import sys
sys.path.insert(0, str(Path(__file__).parent / "src"))

from visionsub.ui.enhanced_video_player import EnhancedVideoPlayer
from visionsub.ui.enhanced_ocr_preview import EnhancedOCRPreview
from visionsub.ui.theme_system import ThemeSystem
from visionsub.ui.enhanced_settings_dialog import EnhancedSettingsDialog
from visionsub.ui.roi_selection import ROISelectionDialog
from visionsub.ui.enhanced_subtitle_editor import EnhancedSubtitleEditor
from visionsub.ui.enhanced_main_window import EnhancedMainWindow
from visionsub.models.config import AppConfig, ProcessingConfig, OcrConfig


class TestEnhancedVideoPlayer:
    """Test suite for Enhanced Video Player component"""
    
    @pytest.fixture
    def video_player(self, qtbot):
        """Create video player instance"""
        player = EnhancedVideoPlayer()
        qtbot.addWidget(player)
        return player
    
    @pytest.fixture
    def sample_video_path(self):
        """Create a sample video file for testing"""
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as f:
            # Create a minimal MP4 file
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(f.name, fourcc, 30.0, (640, 480))
            
            # Create 10 frames
            for _ in range(10):
                frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
                out.write(frame)
            
            out.release()
            yield f.name
        
        Path(f.name).unlink()
    
    def test_video_player_initialization(self, video_player):
        """Test video player initialization"""
        assert video_player is not None
        assert hasattr(video_player, 'media_player')
        assert hasattr(video_player, 'video_widget')
        assert hasattr(video_player, 'controls')
    
    def test_load_video_file(self, video_player, sample_video_path, qtbot):
        """Test loading video file"""
        with patch('visionsub.ui.enhanced_video_player.QFileDialog.getOpenFileName') as mock_dialog:
            mock_dialog.return_value = (sample_video_path, "Video Files (*.mp4 *.avi)")
            
            # Simulate file loading
            video_player.load_video()
            
            # Verify video was loaded
            assert video_player.current_video == sample_video_path
    
    def test_play_pause_controls(self, video_player, sample_video_path, qtbot):
        """Test play/pause functionality"""
        video_player.load_video(sample_video_path)
        
        # Test play button
        play_button = video_player.findChild(QPushButton, "play_button")
        QTest.mouseClick(play_button, Qt.MouseButton.LeftButton)
        
        # Test pause button
        QTest.mouseClick(play_button, Qt.MouseButton.LeftButton)
        
        # Verify state changes
        assert play_button.text() in ["Play", "Pause"]
    
    def test_video_controls(self, video_player, sample_video_path, qtbot):
        """Test video control functionality"""
        video_player.load_video(sample_video_path)
        
        # Test volume control
        volume_slider = video_player.findChild(QSlider, "volume_slider")
        volume_slider.setValue(50)
        assert volume_slider.value() == 50
        
        # Test progress slider
        progress_slider = video_player.findChild(QSlider, "progress_slider")
        progress_slider.setValue(25)
        assert progress_slider.value() == 25
    
    def test_frame_extraction(self, video_player, sample_video_path, qtbot):
        """Test frame extraction functionality"""
        video_player.load_video(sample_video_path)
        
        # Extract frame at specific time
        frame = video_player.extract_frame(1000)  # 1 second
        assert frame is not None
        assert isinstance(frame, np.ndarray)
        assert len(frame.shape) == 3
    
    def test_keyboard_shortcuts(self, video_player, sample_video_path, qtbot):
        """Test keyboard shortcuts"""
        video_player.load_video(sample_video_path)
        
        # Test space bar for play/pause
        QTest.keyPress(video_player, Qt.Key.Key_Space)
        
        # Test arrow keys for seeking
        QTest.keyPress(video_player, Qt.Key.Key_Right)
        QTest.keyPress(video_player, Qt.Key.Key_Left)
        
        # Test volume control
        QTest.keyPress(video_player, Qt.Key.Key_Up)
        QTest.keyPress(video_player, Qt.Key.Key_Down)
    
    def test_error_handling(self, video_player, qtbot):
        """Test error handling for invalid video files"""
        invalid_path = "/nonexistent/video.mp4"
        
        with patch('PyQt6.QtWidgets.QMessageBox.critical') as mock_message:
            video_player.load_video(invalid_path)
            mock_message.assert_called_once()
    
    def test_fullscreen_mode(self, video_player, sample_video_path, qtbot):
        """Test fullscreen mode"""
        video_player.load_video(sample_video_path)
        
        # Test entering fullscreen
        video_player.toggle_fullscreen()
        assert video_player.isFullScreen()
        
        # Test exiting fullscreen
        video_player.toggle_fullscreen()
        assert not video_player.isFullScreen()


class TestEnhancedOCRPreview:
    """Test suite for Enhanced OCR Preview component"""
    
    @pytest.fixture
    def ocr_preview(self, qtbot):
        """Create OCR preview instance"""
        preview = EnhancedOCRPreview()
        qtbot.addWidget(preview)
        return preview
    
    @pytest.fixture
    def sample_image(self):
        """Create a sample image for testing"""
        image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        return image
    
    def test_ocr_preview_initialization(self, ocr_preview):
        """Test OCR preview initialization"""
        assert ocr_preview is not None
        assert hasattr(ocr_preview, 'image_label')
        assert hasattr(ocr_preview, 'text_editor')
        assert hasattr(ocr_preview, 'controls')
    
    def test_load_image(self, ocr_preview, sample_image, qtbot):
        """Test loading image"""
        ocr_preview.load_image(sample_image)
        
        # Verify image was loaded
        assert ocr_preview.current_image is not None
        assert np.array_equal(ocr_preview.current_image, sample_image)
    
    def test_ocr_processing(self, ocr_preview, sample_image, qtbot):
        """Test OCR processing"""
        with patch('visionsub.ui.enhanced_ocr_preview.perform_ocr') as mock_ocr:
            mock_ocr.return_value = ["Sample text 1", "Sample text 2"]
            
            ocr_preview.load_image(sample_image)
            ocr_preview.process_ocr()
            
            # Verify OCR was called
            mock_ocr.assert_called_once()
            assert len(ocr_preview.ocr_results) == 2
    
    def test_text_editing(self, ocr_preview, sample_image, qtbot):
        """Test text editing functionality"""
        ocr_preview.load_image(sample_image)
        
        # Simulate text editing
        text_editor = ocr_preview.findChild(QTextEdit, "text_editor")
        QTest.keyClicks(text_editor, "Test text")
        
        assert text_editor.toPlainText() == "Test text"
    
    def test_zoom_controls(self, ocr_preview, sample_image, qtbot):
        """Test zoom controls"""
        ocr_preview.load_image(sample_image)
        
        # Test zoom in
        zoom_in_button = ocr_preview.findChild(QPushButton, "zoom_in_button")
        QTest.mouseClick(zoom_in_button, Qt.MouseButton.LeftButton)
        
        # Test zoom out
        zoom_out_button = ocr_preview.findChild(QPushButton, "zoom_out_button")
        QTest.mouseClick(zoom_out_button, Qt.MouseButton.LeftButton)
        
        # Test reset zoom
        reset_button = ocr_preview.findChild(QPushButton, "reset_zoom_button")
        QTest.mouseClick(reset_button, Qt.MouseButton.LeftButton)
    
    def test_save_results(self, ocr_preview, sample_image, qtbot):
        """Test saving OCR results"""
        ocr_preview.load_image(sample_image)
        ocr_preview.ocr_results = ["Test result"]
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            save_path = f.name
        
        with patch('visionsub.ui.enhanced_ocr_preview.QFileDialog.getSaveFileName') as mock_dialog:
            mock_dialog.return_value = (save_path, "Text Files (*.txt)")
            
            ocr_preview.save_results()
            
            # Verify file was saved
            assert Path(save_path).exists()
            with open(save_path, 'r') as f:
                content = f.read()
                assert "Test result" in content
        
        Path(save_path).unlink()
    
    def test_export_formats(self, ocr_preview, sample_image, qtbot):
        """Test export to different formats"""
        ocr_preview.load_image(sample_image)
        ocr_preview.ocr_results = ["Test result"]
        
        # Test SRT export
        with tempfile.NamedTemporaryFile(mode='w', suffix='.srt', delete=False) as f:
            srt_path = f.name
        
        ocr_preview.export_srt(srt_path)
        assert Path(srt_path).exists()
        
        # Test JSON export
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json_path = f.name
        
        ocr_preview.export_json(json_path)
        assert Path(json_path).exists()
        
        # Cleanup
        Path(srt_path).unlink()
        Path(json_path).unlink()


class TestThemeSystem:
    """Test suite for Theme System component"""
    
    @pytest.fixture
    def theme_system(self):
        """Create theme system instance"""
        return ThemeSystem()
    
    def test_theme_system_initialization(self, theme_system):
        """Test theme system initialization"""
        assert theme_system is not None
        assert hasattr(theme_system, 'current_theme')
        assert hasattr(theme_system, 'themes')
    
    def test_available_themes(self, theme_system):
        """Test available themes"""
        themes = theme_system.get_available_themes()
        assert isinstance(themes, list)
        assert len(themes) > 0
        assert "dark" in themes
        assert "light" in themes
    
    def test_theme_switching(self, theme_system):
        """Test theme switching"""
        original_theme = theme_system.current_theme
        
        # Switch to dark theme
        theme_system.apply_theme("dark")
        assert theme_system.current_theme == "dark"
        
        # Switch to light theme
        theme_system.apply_theme("light")
        assert theme_system.current_theme == "light"
        
        # Restore original theme
        theme_system.apply_theme(original_theme)
    
    def test_custom_theme_creation(self, theme_system):
        """Test custom theme creation"""
        custom_theme = {
            "name": "custom",
            "colors": {
                "background": "#1a1a1a",
                "foreground": "#ffffff",
                "accent": "#00ff00"
            },
            "fonts": {
                "main": "Arial",
                "monospace": "Courier New"
            }
        }
        
        theme_system.add_custom_theme(custom_theme)
        assert "custom" in theme_system.get_available_themes()
        
        # Apply custom theme
        theme_system.apply_theme("custom")
        assert theme_system.current_theme == "custom"
    
    def test_theme_persistence(self, theme_system):
        """Test theme persistence"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            config_path = f.name
        
        # Save theme configuration
        theme_system.save_config(config_path)
        
        # Create new theme system and load configuration
        new_theme_system = ThemeSystem()
        new_theme_system.load_config(config_path)
        
        assert new_theme_system.current_theme == theme_system.current_theme
        
        Path(config_path).unlink()
    
    def test_theme_application_to_widget(self, theme_system, qtbot):
        """Test theme application to widgets"""
        from PyQt6.QtWidgets import QWidget
        
        widget = QWidget()
        qtbot.addWidget(widget)
        
        # Apply theme to widget
        theme_system.apply_theme_to_widget(widget, "dark")
        
        # Verify theme was applied
        assert widget.styleSheet() != ""


class TestEnhancedSettingsDialog:
    """Test suite for Enhanced Settings Dialog component"""
    
    @pytest.fixture
    def settings_dialog(self, qtbot):
        """Create settings dialog instance"""
        dialog = EnhancedSettingsDialog()
        qtbot.addWidget(dialog)
        return dialog
    
    def test_settings_dialog_initialization(self, settings_dialog):
        """Test settings dialog initialization"""
        assert settings_dialog is not None
        assert hasattr(settings_dialog, 'tabs')
        assert hasattr(settings_dialog, 'button_box')
    
    def test_ocr_settings_tab(self, settings_dialog, qtbot):
        """Test OCR settings tab"""
        # Find OCR settings tab
        ocr_tab = settings_dialog.findChild(QWidget, "ocr_tab")
        assert ocr_tab is not None
        
        # Test engine selection
        engine_combo = ocr_tab.findChild(QComboBox, "engine_combo")
        assert engine_combo is not None
        
        # Test language selection
        language_combo = ocr_tab.findChild(QComboBox, "language_combo")
        assert language_combo is not None
        
        # Test confidence threshold
        confidence_spin = ocr_tab.findChild(QDoubleSpinBox, "confidence_spin")
        assert confidence_spin is not None
        
        # Test setting values
        engine_combo.setCurrentText("PaddleOCR")
        language_combo.setCurrentText("中文")
        confidence_spin.setValue(0.8)
        
        assert engine_combo.currentText() == "PaddleOCR"
        assert language_combo.currentText() == "中文"
        assert confidence_spin.value() == 0.8
    
    def test_processing_settings_tab(self, settings_dialog, qtbot):
        """Test processing settings tab"""
        # Find processing settings tab
        processing_tab = settings_dialog.findChild(QWidget, "processing_tab")
        assert processing_tab is not None
        
        # Test scene threshold
        scene_spin = processing_tab.findChild(QDoubleSpinBox, "scene_spin")
        assert scene_spin is not None
        
        # Test cache size
        cache_spin = processing_tab.findChild(QSpinBox, "cache_spin")
        assert cache_spin is not None
        
        # Test setting values
        scene_spin.setValue(0.3)
        cache_spin.setValue(100)
        
        assert scene_spin.value() == 0.3
        assert cache_spin.value() == 100
    
    def test_ui_settings_tab(self, settings_dialog, qtbot):
        """Test UI settings tab"""
        # Find UI settings tab
        ui_tab = settings_dialog.findChild(QWidget, "ui_tab")
        assert ui_tab is not None
        
        # Test theme selection
        theme_combo = ui_tab.findChild(QComboBox, "theme_combo")
        assert theme_combo is not None
        
        # Test language selection
        ui_language_combo = ui_tab.findChild(QComboBox, "ui_language_combo")
        assert ui_language_combo is not None
        
        # Test setting values
        theme_combo.setCurrentText("dark")
        ui_language_combo.setCurrentText("zh_CN")
        
        assert theme_combo.currentText() == "dark"
        assert ui_language_combo.currentText() == "zh_CN"
    
    def test_settings_validation(self, settings_dialog, qtbot):
        """Test settings validation"""
        # Test invalid confidence threshold
        confidence_spin = settings_dialog.findChild(QDoubleSpinBox, "confidence_spin")
        confidence_spin.setValue(1.5)  # Invalid value
        
        # Should clamp to valid range
        assert confidence_spin.value() <= 1.0
        
        # Test invalid cache size
        cache_spin = settings_dialog.findChild(QSpinBox, "cache_spin")
        cache_spin.setValue(-10)  # Invalid value
        
        # Should clamp to valid range
        assert cache_spin.value() >= 0
    
    def test_settings_save_load(self, settings_dialog, qtbot):
        """Test settings save and load"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            config_path = f.name
        
        # Modify settings
        engine_combo = settings_dialog.findChild(QComboBox, "engine_combo")
        engine_combo.setCurrentText("TesseractOCR")
        
        # Save settings
        settings_dialog.save_settings(config_path)
        
        # Create new dialog and load settings
        new_dialog = EnhancedSettingsDialog()
        new_dialog.load_settings(config_path)
        
        # Verify settings were loaded
        new_engine_combo = new_dialog.findChild(QComboBox, "engine_combo")
        assert new_engine_combo.currentText() == "TesseractOCR"
        
        Path(config_path).unlink()
    
    def test_settings_reset(self, settings_dialog, qtbot):
        """Test settings reset"""
        # Modify settings
        engine_combo = settings_dialog.findChild(QComboBox, "engine_combo")
        engine_combo.setCurrentText("EasyOCR")
        
        # Reset settings
        reset_button = settings_dialog.findChild(QPushButton, "reset_button")
        QTest.mouseClick(reset_button, Qt.MouseButton.LeftButton)
        
        # Verify settings were reset
        assert engine_combo.currentText() == "PaddleOCR"  # Default value


class TestROISelectionDialog:
    """Test suite for ROI Selection Dialog component"""
    
    @pytest.fixture
    def roi_dialog(self, qtbot):
        """Create ROI selection dialog instance"""
        dialog = ROISelectionDialog()
        qtbot.addWidget(dialog)
        return dialog
    
    @pytest.fixture
    def sample_image(self):
        """Create a sample image for testing"""
        image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        return image
    
    def test_roi_dialog_initialization(self, roi_dialog):
        """Test ROI dialog initialization"""
        assert roi_dialog is not None
        assert hasattr(roi_dialog, 'image_label')
        assert hasattr(roi_dialog, 'controls')
    
    def test_load_image_for_roi(self, roi_dialog, sample_image, qtbot):
        """Test loading image for ROI selection"""
        roi_dialog.load_image(sample_image)
        
        # Verify image was loaded
        assert roi_dialog.current_image is not None
        assert np.array_equal(roi_dialog.current_image, sample_image)
    
    def test_roi_selection(self, roi_dialog, sample_image, qtbot):
        """Test ROI selection"""
        roi_dialog.load_image(sample_image)
        
        # Simulate mouse drag for ROI selection
        image_label = roi_dialog.findChild(QLabel, "image_label")
        
        # Start drag
        QTest.mousePress(image_label, Qt.MouseButton.LeftButton, pos=(100, 100))
        
        # Drag to create ROI
        QTest.mouseMove(image_label, pos=(200, 200))
        
        # End drag
        QTest.mouseRelease(image_label, Qt.MouseButton.LeftButton, pos=(200, 200))
        
        # Verify ROI was created
        assert roi_dialog.current_roi is not None
        assert isinstance(roi_dialog.current_roi, QRect)
    
    def test_roi_preset_sizes(self, roi_dialog, sample_image, qtbot):
        """Test ROI preset sizes"""
        roi_dialog.load_image(sample_image)
        
        # Test preset buttons
        preset_buttons = roi_dialog.findChildren(QPushButton)
        for button in preset_buttons:
            if "preset" in button.objectName().lower():
                QTest.mouseClick(button, Qt.MouseButton.LeftButton)
                assert roi_dialog.current_roi is not None
    
    def test_roi_adjustment(self, roi_dialog, sample_image, qtbot):
        """Test ROI adjustment"""
        roi_dialog.load_image(sample_image)
        
        # Create initial ROI
        roi_dialog.set_roi(QRect(100, 100, 200, 200))
        
        # Test adjustment controls
        x_spin = roi_dialog.findChild(QSpinBox, "x_spin")
        y_spin = roi_dialog.findChild(QSpinBox, "y_spin")
        width_spin = roi_dialog.findChild(QSpinBox, "width_spin")
        height_spin = roi_dialog.findChild(QSpinBox, "height_spin")
        
        # Adjust values
        x_spin.setValue(150)
        y_spin.setValue(150)
        width_spin.setValue(250)
        height_spin.setValue(250)
        
        # Verify ROI was updated
        assert roi_dialog.current_roi.x() == 150
        assert roi_dialog.current_roi.y() == 150
        assert roi_dialog.current_roi.width() == 250
        assert roi_dialog.current_roi.height() == 250
    
    def test_roi_validation(self, roi_dialog, sample_image, qtbot):
        """Test ROI validation"""
        roi_dialog.load_image(sample_image)
        
        # Test invalid ROI (outside image bounds)
        invalid_roi = QRect(1000, 1000, 100, 100)
        roi_dialog.set_roi(invalid_roi)
        
        # Should be adjusted to fit within image bounds
        assert roi_dialog.current_roi.x() < 640
        assert roi_dialog.current_roi.y() < 480
    
    def test_roi_export_import(self, roi_dialog, sample_image, qtbot):
        """Test ROI export and import"""
        roi_dialog.load_image(sample_image)
        
        # Create ROI
        roi_dialog.set_roi(QRect(100, 100, 200, 200))
        
        # Export ROI
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            roi_path = f.name
        
        roi_dialog.export_roi(roi_path)
        
        # Import ROI
        new_dialog = ROISelectionDialog()
        new_dialog.load_image(sample_image)
        new_dialog.import_roi(roi_path)
        
        # Verify ROI was imported
        assert new_dialog.current_roi == roi_dialog.current_roi
        
        Path(roi_path).unlink()


class TestEnhancedSubtitleEditor:
    """Test suite for Enhanced Subtitle Editor component"""
    
    @pytest.fixture
    def subtitle_editor(self, qtbot):
        """Create subtitle editor instance"""
        editor = EnhancedSubtitleEditor()
        qtbot.addWidget(editor)
        return editor
    
    def test_subtitle_editor_initialization(self, subtitle_editor):
        """Test subtitle editor initialization"""
        assert subtitle_editor is not None
        assert hasattr(subtitle_editor, 'subtitle_table')
        assert hasattr(subtitle_editor, 'text_editor')
        assert hasattr(subtitle_editor, 'controls')
    
    def test_load_subtitles(self, subtitle_editor, qtbot):
        """Test loading subtitles"""
        sample_subtitles = [
            {"start": 0, "end": 5000, "text": "Hello world"},
            {"start": 5000, "end": 10000, "text": "This is a test"}
        ]
        
        subtitle_editor.load_subtitles(sample_subtitles)
        
        # Verify subtitles were loaded
        assert len(subtitle_editor.subtitles) == 2
        assert subtitle_editor.subtitles[0]["text"] == "Hello world"
    
    def test_subtitle_editing(self, subtitle_editor, qtbot):
        """Test subtitle editing"""
        sample_subtitles = [
            {"start": 0, "end": 5000, "text": "Original text"}
        ]
        
        subtitle_editor.load_subtitles(sample_subtitles)
        
        # Edit subtitle text
        text_editor = subtitle_editor.findChild(QTextEdit, "text_editor")
        text_editor.clear()
        QTest.keyClicks(text_editor, "Edited text")
        
        # Save changes
        save_button = subtitle_editor.findChild(QPushButton, "save_button")
        QTest.mouseClick(save_button, Qt.MouseButton.LeftButton)
        
        # Verify changes were saved
        assert subtitle_editor.subtitles[0]["text"] == "Edited text"
    
    def test_subtitle_timing_adjustment(self, subtitle_editor, qtbot):
        """Test subtitle timing adjustment"""
        sample_subtitles = [
            {"start": 0, "end": 5000, "text": "Test subtitle"}
        ]
        
        subtitle_editor.load_subtitles(sample_subtitles)
        
        # Adjust timing
        start_spin = subtitle_editor.findChild(QSpinBox, "start_spin")
        end_spin = subtitle_editor.findChild(QSpinBox, "end_spin")
        
        start_spin.setValue(1000)
        end_spin.setValue(6000)
        
        # Verify timing was updated
        assert subtitle_editor.subtitles[0]["start"] == 1000
        assert subtitle_editor.subtitles[0]["end"] == 6000
    
    def test_subtitle_add_delete(self, subtitle_editor, qtbot):
        """Test adding and deleting subtitles"""
        # Add new subtitle
        add_button = subtitle_editor.findChild(QPushButton, "add_button")
        QTest.mouseClick(add_button, Qt.MouseButton.LeftButton)
        
        # Verify subtitle was added
        assert len(subtitle_editor.subtitles) == 1
        
        # Delete subtitle
        delete_button = subtitle_editor.findChild(QPushButton, "delete_button")
        QTest.mouseClick(delete_button, Qt.MouseButton.LeftButton)
        
        # Verify subtitle was deleted
        assert len(subtitle_editor.subtitles) == 0
    
    def test_subtitle_search(self, subtitle_editor, qtbot):
        """Test subtitle search functionality"""
        sample_subtitles = [
            {"start": 0, "end": 5000, "text": "Hello world"},
            {"start": 5000, "end": 10000, "text": "This is a test"},
            {"start": 10000, "end": 15000, "text": "Another subtitle"}
        ]
        
        subtitle_editor.load_subtitles(sample_subtitles)
        
        # Search for text
        search_edit = subtitle_editor.findChild(QLineEdit, "search_edit")
        QTest.keyClicks(search_edit, "test")
        
        search_button = subtitle_editor.findChild(QPushButton, "search_button")
        QTest.mouseClick(search_button, Qt.MouseButton.LeftButton)
        
        # Verify search results
        assert len(subtitle_editor.search_results) == 1
        assert subtitle_editor.search_results[0]["text"] == "This is a test"
    
    def test_subtitle_export_formats(self, subtitle_editor, qtbot):
        """Test subtitle export to different formats"""
        sample_subtitles = [
            {"start": 0, "end": 5000, "text": "Test subtitle"}
        ]
        
        subtitle_editor.load_subtitles(sample_subtitles)
        
        # Test SRT export
        with tempfile.NamedTemporaryFile(mode='w', suffix='.srt', delete=False) as f:
            srt_path = f.name
        
        subtitle_editor.export_srt(srt_path)
        assert Path(srt_path).exists()
        
        # Test VTT export
        with tempfile.NamedTemporaryFile(mode='w', suffix='.vtt', delete=False) as f:
            vtt_path = f.name
        
        subtitle_editor.export_vtt(vtt_path)
        assert Path(vtt_path).exists()
        
        # Test ASS export
        with tempfile.NamedTemporaryFile(mode='w', suffix='.ass', delete=False) as f:
            ass_path = f.name
        
        subtitle_editor.export_ass(ass_path)
        assert Path(ass_path).exists()
        
        # Cleanup
        Path(srt_path).unlink()
        Path(vtt_path).unlink()
        Path(ass_path).unlink()


class TestEnhancedMainWindow:
    """Test suite for Enhanced Main Window component"""
    
    @pytest.fixture
    def main_window(self, qtbot):
        """Create main window instance"""
        window = EnhancedMainWindow()
        qtbot.addWidget(window)
        return window
    
    def test_main_window_initialization(self, main_window):
        """Test main window initialization"""
        assert main_window is not None
        assert hasattr(main_window, 'video_player')
        assert hasattr(main_window, 'ocr_preview')
        assert hasattr(main_window, 'subtitle_editor')
        assert hasattr(main_window, 'menu_bar')
    
    def test_menu_actions(self, main_window, qtbot):
        """Test menu actions"""
        # Test File menu
        file_menu = main_window.findChild(QMenu, "file_menu")
        assert file_menu is not None
        
        # Test Open action
        open_action = file_menu.findChild(QAction, "open_action")
        assert open_action is not None
        
        # Test Save action
        save_action = file_menu.findChild(QAction, "save_action")
        assert save_action is not None
        
        # Test Exit action
        exit_action = file_menu.findChild(QAction, "exit_action")
        assert exit_action is not None
    
    def test_toolbar_actions(self, main_window, qtbot):
        """Test toolbar actions"""
        # Test Open action
        open_action = main_window.findChild(QAction, "open_action")
        assert open_action is not None
        
        # Test Process action
        process_action = main_window.findChild(QAction, "process_action")
        assert process_action is not None
        
        # Test Settings action
        settings_action = main_window.findChild(QAction, "settings_action")
        assert settings_action is not None
    
    def test_workflow_integration(self, main_window, qtbot):
        """Test workflow integration between components"""
        # Test that components are properly connected
        assert main_window.video_player is not None
        assert main_window.ocr_preview is not None
        assert main_window.subtitle_editor is not None
        
        # Test signal-slot connections
        assert main_window.video_player.signals.frame_extracted is not None
        assert main_window.ocr_preview.signals.ocr_completed is not None
        assert main_window.subtitle_editor.signals.subtitle_changed is not None
    
    def test_state_persistence(self, main_window, qtbot):
        """Test state persistence"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            state_path = f.name
        
        # Modify state
        main_window.resize(800, 600)
        
        # Save state
        main_window.save_state(state_path)
        
        # Create new window and load state
        new_window = EnhancedMainWindow()
        new_window.load_state(state_path)
        
        # Verify state was loaded
        assert new_window.size() == main_window.size()
        
        Path(state_path).unlink()
    
    def test_error_handling(self, main_window, qtbot):
        """Test error handling"""
        # Test error dialog display
        with patch('PyQt6.QtWidgets.QMessageBox.critical') as mock_message:
            main_window.show_error("Test error", "Test error message")
            mock_message.assert_called_once_with(
                main_window, "Test error", "Test error message"
            )
    
    def test_recent_files(self, main_window, qtbot):
        """Test recent files functionality"""
        # Add recent file
        test_file = "/path/to/test.mp4"
        main_window.add_recent_file(test_file)
        
        # Verify file was added
        assert test_file in main_window.recent_files
        
        # Test recent files menu
        recent_menu = main_window.findChild(QMenu, "recent_menu")
        assert recent_menu is not None
        
        # Verify menu item was created
        recent_actions = recent_menu.findChildren(QAction)
        assert len(recent_actions) > 0
        assert recent_actions[0].text() == test_file


# Integration tests
class TestUIIntegration:
    """Integration tests for UI components"""
    
    @pytest.fixture
    def integrated_app(self, qtbot):
        """Create integrated application"""
        from visionsub.ui.enhanced_main import EnhancedMainApp
        
        app = EnhancedMainApp([])
        window = app.main_window
        qtbot.addWidget(window)
        return app
    
    def test_complete_workflow(self, integrated_app, qtbot):
        """Test complete workflow from video to subtitles"""
        window = integrated_app.main_window
        
        # This would be a comprehensive test of the entire workflow
        # In practice, this would involve:
        # 1. Loading a video
        # 2. Processing OCR
        # 3. Editing subtitles
        # 4. Exporting results
        
        # For now, just verify components are properly initialized
        assert window.video_player is not None
        assert window.ocr_preview is not None
        assert window.subtitle_editor is not None
    
    def test_signal_propagation(self, integrated_app, qtbot):
        """Test signal propagation between components"""
        window = integrated_app.main_window
        
        # Test that signals are properly connected
        assert window.video_player.signals.frame_extracted is not None
        assert window.ocr_preview.signals.ocr_completed is not None
        assert window.subtitle_editor.signals.subtitle_changed is not None
    
    def test_resource_management(self, integrated_app, qtbot):
        """Test resource management"""
        window = integrated_app.main_window
        
        # Test that resources are properly managed
        assert hasattr(window, 'cleanup')
        
        # Simulate cleanup
        window.cleanup()
        
        # Verify resources were cleaned up
        # (This would need more specific checks based on actual implementation)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])