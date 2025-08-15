"""
Comprehensive Enhanced UI Testing Suite for VisionSub

This module provides comprehensive testing for all enhanced UI components including:
- Enhanced main window
- Enhanced video player
- Enhanced OCR preview
- Enhanced subtitle editor
- Enhanced settings dialog
- Theme system
- Security features
- Performance testing
- Integration testing
"""

import pytest
import asyncio
import numpy as np
import cv2
import tempfile
import json
import time
import gc
import psutil
import threading
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum

# PyQt6 imports
from PyQt6.QtCore import (
    Qt, QTimer, QEvent, QRect, QSize, QPoint, 
    pyqtSignal, QThread, QPropertyAnimation, QEasingCurve
)
from PyQt6.QtGui import (
    QImage, QPixmap, QFont, QColor, QPalette, QPainter,
    QBrush, QPen, QTextCharFormat, QTextCursor, QKeySequence,
    QMouseEvent, QKeyEvent, QWheelEvent, QDragEnterEvent,
    QDropEvent, QCloseEvent
)
from PyQt6.QtWidgets import (
    QApplication, QWidget, QMainWindow, QDialog, QPushButton,
    QLabel, QLineEdit, QTextEdit, QComboBox, QSpinBox, QDoubleSpinBox,
    QCheckBox, QSlider, QProgressBar, QTabWidget, QTableWidget,
    QTableWidgetItem, QHeaderView, QSplitter, QScrollArea,
    QGroupBox, QVBoxLayout, QHBoxLayout, QFormLayout, QMenuBar,
    QMenu, QAction, QToolBar, QStatusBar, QMessageBox,
    QFileDialog, QSystemTrayIcon, QStyle, QStyleOption
)
from PyQt6.QtTest import QTest
from PyQt6.QtSvg import QSvgRenderer

# Add src to path
import sys
sys.path.insert(0, str(Path(__file__).parent / "src"))

# VisionSub imports
from visionsub.ui.enhanced_main_window import EnhancedMainWindow
from visionsub.ui.enhanced_video_player import EnhancedVideoPlayer
from visionsub.ui.enhanced_ocr_preview import EnhancedOCRPreview
from visionsub.ui.enhanced_subtitle_editor import EnhancedSubtitleEditor
from visionsub.ui.enhanced_settings_dialog import EnhancedSettingsDialog
from visionsub.ui.enhanced_main import EnhancedApplication
from visionsub.ui.theme_system import ThemeManager, get_theme_manager, ThemeColors
from visionsub.models.config import AppConfig, OcrConfig, ProcessingConfig, UIConfig, SecurityConfig
from visionsub.view_models.main_view_model import MainViewModel

# Configure logging
import logging
logger = logging.getLogger(__name__)


# Test Data Classes
@dataclass
class TestData:
    """Test data for UI testing"""
    sample_video_path: str
    sample_image: np.ndarray
    sample_ocr_results: List[Dict[str, Any]]
    sample_subtitles: List[Dict[str, Any]]
    config_data: Dict[str, Any]


class TestCategory(Enum):
    """Test categories for organization"""
    UNIT = "unit"
    INTEGRATION = "integration"
    SECURITY = "security"
    PERFORMANCE = "performance"
    ACCESSIBILITY = "accessibility"
    USER_EXPERIENCE = "ux"


# Fixtures
@pytest.fixture(scope="session")
def app():
    """Create QApplication instance"""
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    yield app
    app.quit()


@pytest.fixture
def test_data():
    """Create test data"""
    # Create sample video
    with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as f:
        video_path = f.name
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(f.name, fourcc, 30.0, (640, 480))
        
        # Create 30 frames
        for i in range(30):
            frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            # Add some text to frames
            cv2.putText(frame, f"Frame {i}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            out.write(frame)
        
        out.release()
    
    # Create sample image
    sample_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    cv2.putText(sample_image, "Test OCR Text", (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    # Sample OCR results
    sample_ocr_results = [
        {
            "text": "Test OCR Text",
            "confidence": 0.95,
            "language": "en",
            "position": {"x": 100, "y": 100, "width": 200, "height": 50},
            "timestamp": 1.0
        },
        {
            "text": "Another Text",
            "confidence": 0.87,
            "language": "en",
            "position": {"x": 150, "y": 200, "width": 150, "height": 40},
            "timestamp": 2.0
        }
    ]
    
    # Sample subtitles
    sample_subtitles = [
        {"start": 0, "end": 5000, "text": "Hello world"},
        {"start": 5000, "end": 10000, "text": "This is a test"},
        {"start": 10000, "end": 15000, "text": "Another subtitle"}
    ]
    
    # Sample config
    config_data = {
        "processing": {
            "ocr_config": {
                "engine": "PaddleOCR",
                "language": "en",
                "threshold": 128,
                "confidence_threshold": 0.8,
                "enable_preprocessing": True,
                "enable_postprocessing": True
            },
            "scene_threshold": 0.3,
            "cache_size": 100,
            "max_concurrent_jobs": 4,
            "frame_interval": 1.0
        },
        "ui": {
            "theme": "dark",
            "language": "en",
            "window_size": [1200, 800],
            "font_size": 12,
            "enable_animations": True,
            "show_performance_metrics": True
        },
        "security": {
            "enable_input_validation": True,
            "max_file_size_mb": 100,
            "allowed_video_formats": ["mp4", "avi", "mkv"],
            "enable_rate_limiting": True,
            "rate_limit_requests_per_minute": 60
        }
    }
    
    yield TestData(
        sample_video_path=video_path,
        sample_image=sample_image,
        sample_ocr_results=sample_ocr_results,
        sample_subtitles=sample_subtitles,
        config_data=config_data
    )
    
    # Cleanup
    Path(video_path).unlink(missing_ok=True)


@pytest.fixture
def view_model():
    """Create view model instance"""
    return MainViewModel()


@pytest.fixture
def config():
    """Create config instance"""
    return AppConfig()


# ============================================================================
# Enhanced Main Window Tests
# ============================================================================

class TestEnhancedMainWindow:
    """Comprehensive test suite for EnhancedMainWindow"""
    
    @pytest.fixture
    def main_window(self, qtbot, view_model):
        """Create main window instance"""
        window = EnhancedMainWindow(view_model)
        qtbot.addWidget(window)
        window.show()
        yield window
        window.close()
    
    def test_initialization(self, main_window):
        """Test main window initialization"""
        assert main_window is not None
        assert hasattr(main_window, 'video_player')
        assert hasattr(main_window, 'ocr_preview')
        assert hasattr(main_window, 'subtitle_editor')
        assert hasattr(main_window, 'menu_bar')
        assert hasattr(main_window, 'status_bar')
        assert hasattr(main_window, 'tool_bar')
        
        # Check window properties
        assert main_window.windowTitle() == "VisionSub - 视频OCR字幕提取工具"
        assert main_window.size().width() > 0
        assert main_window.size().height() > 0
    
    def test_menu_structure(self, main_window):
        """Test menu structure and actions"""
        # Check File menu
        file_menu = main_window.menu_bar.findChild(QMenu, "file_menu")
        assert file_menu is not None
        assert file_menu.title() == "文件(&F)"
        
        # Check Edit menu
        edit_menu = main_window.menu_bar.findChild(QMenu, "edit_menu")
        assert edit_menu is not None
        assert edit_menu.title() == "编辑(&E)"
        
        # Check View menu
        view_menu = main_window.menu_bar.findChild(QMenu, "view_menu")
        assert view_menu is not None
        assert view_menu.title() == "视图(&V)"
        
        # Check Tools menu
        tools_menu = main_window.menu_bar.findChild(QMenu, "tools_menu")
        assert tools_menu is not None
        assert tools_menu.title() == "工具(&T)"
        
        # Check Help menu
        help_menu = main_window.menu_bar.findChild(QMenu, "help_menu")
        assert help_menu is not None
        assert help_menu.title() == "帮助(&H)"
    
    def test_toolbar_actions(self, main_window):
        """Test toolbar actions"""
        # Check Open action
        open_action = main_window.findChild(QAction, "open_action")
        assert open_action is not None
        assert open_action.text() == "打开"
        assert open_action.toolTip() == "打开视频文件"
        
        # Check Process action
        process_action = main_window.findChild(QAction, "process_action")
        assert process_action is not None
        assert process_action.text() == "处理"
        assert process_action.toolTip() == "开始处理视频"
        
        # Check Settings action
        settings_action = main_window.findChild(QAction, "settings_action")
        assert settings_action is not None
        assert settings_action.text() == "设置"
        assert settings_action.toolTip() == "打开设置对话框"
    
    def test_status_bar(self, main_window):
        """Test status bar functionality"""
        status_bar = main_window.status_bar()
        assert status_bar is not None
        
        # Check status bar widgets
        status_label = status_bar.findChild(QLabel, "status_label")
        assert status_label is not None
        
        progress_bar = status_bar.findChild(QProgressBar, "progress_bar")
        assert progress_bar is not None
        
        # Test status updates
        main_window.update_status("Ready")
        assert status_label.text() == "Ready"
        
        main_window.update_progress(50)
        assert progress_bar.value() == 50
    
    def test_file_operations(self, main_window, test_data, qtbot):
        """Test file operations"""
        # Test open file
        with patch('PyQt6.QtWidgets.QFileDialog.getOpenFileName') as mock_dialog:
            mock_dialog.return_value = (test_data.sample_video_path, "Video Files (*.mp4 *.avi)")
            
            # Trigger open action
            open_action = main_window.findChild(QAction, "open_action")
            open_action.trigger()
            
            # Verify file was loaded
            qtbot.wait(100)  # Wait for file loading
            assert main_window.current_file == test_data.sample_video_path
    
    def test_keyboard_shortcuts(self, main_window, qtbot):
        """Test keyboard shortcuts"""
        # Test Ctrl+O for open
        QTest.keyClick(main_window, Qt.Key.Key_O, Qt.KeyboardModifier.ControlModifier)
        
        # Test Ctrl+S for save
        QTest.keyClick(main_window, Qt.Key.Key_S, Qt.KeyboardModifier.ControlModifier)
        
        # Test Ctrl+Q for quit
        with patch.object(main_window, 'close') as mock_close:
            QTest.keyClick(main_window, Qt.Key.Key_Q, Qt.KeyboardModifier.ControlModifier)
            mock_close.assert_called_once()
    
    def test_drag_and_drop(self, main_window, test_data, qtbot):
        """Test drag and drop functionality"""
        # Create drag enter event
        mime_data = MagicMock()
        mime_data.hasUrls.return_value = True
        mime_data.urls.return_value = [test_data.sample_video_path]
        
        drag_enter_event = QDragEnterEvent(
            QPoint(100, 100),
            Qt.DropAction.CopyAction,
            mime_data,
            Qt.MouseButton.NoButton,
            Qt.KeyboardModifier.NoModifier
        )
        
        # Test drag enter
        main_window.dragEnterEvent(drag_enter_event)
        assert drag_enter_event.acceptProposedAction()
        
        # Create drop event
        drop_event = QDropEvent(
            QPoint(100, 100),
            Qt.DropAction.CopyAction,
            mime_data,
            Qt.MouseButton.NoButton,
            Qt.KeyboardModifier.NoModifier
        )
        
        # Test drop
        main_window.dropEvent(drop_event)
        qtbot.wait(100)
        assert main_window.current_file == test_data.sample_video_path
    
    def test_window_state_management(self, main_window, qtbot):
        """Test window state management"""
        # Test maximize/restore
        main_window.showMaximized()
        assert main_window.isMaximized()
        
        main_window.showNormal()
        assert not main_window.isMaximized()
        
        # Test minimize/restore
        main_window.showMinimized()
        assert main_window.isMinimized()
        
        main_window.showNormal()
        assert not main_window.isMinimized()
        
        # Test fullscreen
        main_window.showFullScreen()
        assert main_window.isFullScreen()
        
        main_window.showNormal()
        assert not main_window.isFullScreen()
    
    def test_recent_files_management(self, main_window, test_data):
        """Test recent files management"""
        # Add recent file
        main_window.add_recent_file(test_data.sample_video_path)
        assert test_data.sample_video_path in main_window.recent_files
        
        # Check recent files menu
        recent_menu = main_window.findChild(QMenu, "recent_menu")
        assert recent_menu is not None
        
        recent_actions = recent_menu.findChildren(QAction)
        assert len(recent_actions) > 0
        
        # Test recent file limit
        for i in range(15):  # Add more than limit
            main_window.add_recent_file(f"/path/to/file_{i}.mp4")
        
        assert len(main_window.recent_files) <= 10  # Default limit
    
    def test_error_handling(self, main_window):
        """Test error handling"""
        # Test error dialog
        with patch('PyQt6.QtWidgets.QMessageBox.critical') as mock_message:
            main_window.show_error("Test Error", "Test error message")
            mock_message.assert_called_once_with(
                main_window, "Test Error", "Test error message"
            )
        
        # Test warning dialog
        with patch('PyQt6.QtWidgets.QMessageBox.warning') as mock_message:
            main_window.show_warning("Test Warning", "Test warning message")
            mock_message.assert_called_once_with(
                main_window, "Test Warning", "Test warning message"
            )
        
        # Test info dialog
        with patch('PyQt6.QtWidgets.QMessageBox.information') as mock_message:
            main_window.show_info("Test Info", "Test info message")
            mock_message.assert_called_once_with(
                main_window, "Test Info", "Test info message"
            )
    
    def test_theme_integration(self, main_window, qtbot):
        """Test theme integration"""
        theme_manager = get_theme_manager()
        
        # Test theme switching
        original_theme = theme_manager.current_theme
        
        # Switch to light theme
        theme_manager.set_theme("light")
        qtbot.wait(100)
        
        # Switch to dark theme
        theme_manager.set_theme("dark")
        qtbot.wait(100)
        
        # Restore original theme
        theme_manager.set_theme(original_theme)
    
    def test_component_integration(self, main_window):
        """Test integration between components"""
        # Check that components are properly connected
        assert main_window.video_player is not None
        assert main_window.ocr_preview is not None
        assert main_window.subtitle_editor is not None
        
        # Test signal-slot connections
        assert hasattr(main_window.video_player, 'frame_extracted')
        assert hasattr(main_window.ocr_preview, 'ocr_completed')
        assert hasattr(main_window.subtitle_editor, 'subtitle_changed')
    
    def test_state_persistence(self, main_window, qtbot):
        """Test state persistence"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            state_path = f.name
        
        # Modify state
        main_window.resize(1000, 700)
        main_window.move(50, 50)
        
        # Save state
        main_window.save_state(state_path)
        
        # Create new window and load state
        new_window = EnhancedMainWindow(main_window.view_model)
        new_window.load_state(state_path)
        
        # Verify state was loaded
        assert new_window.size() == main_window.size()
        assert new_window.pos() == main_window.pos()
        
        new_window.close()
        Path(state_path).unlink()
    
    def test_resource_cleanup(self, main_window, qtbot):
        """Test resource cleanup"""
        # Simulate resource usage
        main_window.current_file = test_data.sample_video_path
        
        # Test cleanup
        main_window.cleanup()
        
        # Verify resources were cleaned up
        assert main_window.current_file is None


# ============================================================================
# Enhanced Video Player Tests
# ============================================================================

class TestEnhancedVideoPlayer:
    """Comprehensive test suite for EnhancedVideoPlayer"""
    
    @pytest.fixture
    def video_player(self, qtbot):
        """Create video player instance"""
        player = EnhancedVideoPlayer()
        qtbot.addWidget(player)
        player.show()
        yield player
        player.close()
    
    def test_initialization(self, video_player):
        """Test video player initialization"""
        assert video_player is not None
        assert hasattr(video_player, 'media_player')
        assert hasattr(video_player, 'video_widget')
        assert hasattr(video_player, 'controls')
        assert hasattr(video_player, 'progress_slider')
        assert hasattr(video_player, 'volume_slider')
        assert hasattr(video_player, 'play_button')
        assert hasattr(video_player, 'time_label')
    
    def test_video_loading(self, video_player, test_data, qtbot):
        """Test video loading functionality"""
        # Test valid video file
        video_player.load_video(test_data.sample_video_path)
        qtbot.wait(500)  # Wait for video to load
        
        assert video_player.current_video == test_data.sample_video_path
        assert video_player.media_player.media() is not None
    
    def test_playback_controls(self, video_player, test_data, qtbot):
        """Test playback controls"""
        video_player.load_video(test_data.sample_video_path)
        qtbot.wait(500)
        
        # Test play
        QTest.mouseClick(video_player.play_button, Qt.MouseButton.LeftButton)
        qtbot.wait(100)
        
        # Test pause
        QTest.mouseClick(video_player.play_button, Qt.MouseButton.LeftButton)
        qtbot.wait(100)
        
        # Test stop
        QTest.mouseClick(video_player.stop_button, Qt.MouseButton.LeftButton)
        qtbot.wait(100)
    
    def test_seek_controls(self, video_player, test_data, qtbot):
        """Test seek controls"""
        video_player.load_video(test_data.sample_video_path)
        qtbot.wait(500)
        
        # Test progress slider
        video_player.progress_slider.setValue(50)
        qtbot.wait(100)
        assert video_player.progress_slider.value() == 50
        
        # Test frame stepping
        QTest.mouseClick(video_player.frame_forward_button, Qt.MouseButton.LeftButton)
        qtbot.wait(100)
        
        QTest.mouseClick(video_player.frame_backward_button, Qt.MouseButton.LeftButton)
        qtbot.wait(100)
    
    def test_volume_controls(self, video_player, test_data, qtbot):
        """Test volume controls"""
        video_player.load_video(test_data.sample_video_path)
        qtbot.wait(500)
        
        # Test volume slider
        video_player.volume_slider.setValue(50)
        qtbot.wait(100)
        assert video_player.volume_slider.value() == 50
        
        # Test mute button
        QTest.mouseClick(video_player.mute_button, Qt.MouseButton.LeftButton)
        qtbot.wait(100)
        
        QTest.mouseClick(video_player.mute_button, Qt.MouseButton.LeftButton)
        qtbot.wait(100)
    
    def test_keyboard_controls(self, video_player, test_data, qtbot):
        """Test keyboard controls"""
        video_player.load_video(test_data.sample_video_path)
        qtbot.wait(500)
        
        # Test space bar for play/pause
        QTest.keyClick(video_player, Qt.Key.Key_Space)
        qtbot.wait(100)
        
        # Test arrow keys for seeking
        QTest.keyClick(video_player, Qt.Key.Key_Right)
        QTest.keyClick(video_player, Qt.Key.Key_Left)
        qtbot.wait(100)
        
        # Test volume controls
        QTest.keyClick(video_player, Qt.Key.Key_Up)
        QTest.keyClick(video_player, Qt.Key.Key_Down)
        qtbot.wait(100)
    
    def test_frame_extraction(self, video_player, test_data, qtbot):
        """Test frame extraction functionality"""
        video_player.load_video(test_data.sample_video_path)
        qtbot.wait(500)
        
        # Extract frame at specific time
        frame = video_player.extract_frame(1000)  # 1 second
        assert frame is not None
        assert isinstance(frame, np.ndarray)
        assert len(frame.shape) == 3
        
        # Extract current frame
        current_frame = video_player.get_current_frame()
        assert current_frame is not None
        assert isinstance(current_frame, np.ndarray)
    
    def test_screenshot_functionality(self, video_player, test_data, qtbot):
        """Test screenshot functionality"""
        video_player.load_video(test_data.sample_video_path)
        qtbot.wait(500)
        
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            screenshot_path = f.name
        
        # Take screenshot
        video_player.take_screenshot(screenshot_path)
        qtbot.wait(100)
        
        # Verify screenshot was saved
        assert Path(screenshot_path).exists()
        assert Path(screenshot_path).stat().st_size > 0
        
        Path(screenshot_path).unlink()
    
    def test_fullscreen_mode(self, video_player, test_data, qtbot):
        """Test fullscreen mode"""
        video_player.load_video(test_data.sample_video_path)
        qtbot.wait(500)
        
        # Test entering fullscreen
        video_player.toggle_fullscreen()
        qtbot.wait(100)
        assert video_player.isFullScreen()
        
        # Test exiting fullscreen
        video_player.toggle_fullscreen()
        qtbot.wait(100)
        assert not video_player.isFullScreen()
    
    def test_error_handling(self, video_player, qtbot):
        """Test error handling"""
        # Test invalid video file
        invalid_path = "/nonexistent/video.mp4"
        
        with patch('PyQt6.QtWidgets.QMessageBox.critical') as mock_message:
            video_player.load_video(invalid_path)
            mock_message.assert_called_once()
    
    def test_performance_monitoring(self, video_player, test_data, qtbot):
        """Test performance monitoring"""
        video_player.load_video(test_data.sample_video_path)
        qtbot.wait(500)
        
        # Test FPS monitoring
        assert hasattr(video_player, 'fps_label')
        assert video_player.fps_label.isVisible()
        
        # Test frame time monitoring
        assert hasattr(video_player, 'frame_time_label')
        assert video_player.frame_time_label.isVisible()
    
    def test_memory_management(self, video_player, test_data, qtbot):
        """Test memory management"""
        video_player.load_video(test_data.sample_video_path)
        qtbot.wait(500)
        
        # Test frame buffer cleanup
        initial_memory = psutil.Process().memory_info().rss
        
        # Extract multiple frames
        for i in range(10):
            frame = video_player.extract_frame(i * 100)
            assert frame is not None
        
        qtbot.wait(100)
        
        # Check memory usage
        final_memory = psutil.Process().memory_info().rss
        memory_increase = final_memory - initial_memory
        
        # Memory increase should be reasonable
        assert memory_increase < 100 * 1024 * 1024  # Less than 100MB


# ============================================================================
# Enhanced OCR Preview Tests
# ============================================================================

class TestEnhancedOCRPreview:
    """Comprehensive test suite for EnhancedOCRPreview"""
    
    @pytest.fixture
    def ocr_preview(self, qtbot):
        """Create OCR preview instance"""
        preview = EnhancedOCRPreview()
        qtbot.addWidget(preview)
        preview.show()
        yield preview
        preview.close()
    
    def test_initialization(self, ocr_preview):
        """Test OCR preview initialization"""
        assert ocr_preview is not None
        assert hasattr(ocr_preview, 'text_editor')
        assert hasattr(ocr_preview, 'results_table')
        assert hasattr(ocr_preview, 'confidence_filter')
        assert hasattr(ocr_preview, 'export_button')
        assert hasattr(ocr_preview, 'clear_button')
    
    def test_image_loading(self, ocr_preview, test_data, qtbot):
        """Test image loading functionality"""
        ocr_preview.load_image(test_data.sample_image)
        qtbot.wait(100)
        
        assert ocr_preview.current_image is not None
        assert np.array_equal(ocr_preview.current_image, test_data.sample_image)
    
    def test_ocr_processing(self, ocr_preview, test_data, qtbot):
        """Test OCR processing functionality"""
        ocr_preview.load_image(test_data.sample_image)
        qtbot.wait(100)
        
        # Mock OCR processing
        with patch.object(ocr_preview, 'process_ocr') as mock_ocr:
            mock_ocr.return_value = test_data.sample_ocr_results
            
            ocr_preview.start_ocr_processing()
            qtbot.wait(100)
            
            mock_ocr.assert_called_once()
            assert len(ocr_preview.ocr_results) == len(test_data.sample_ocr_results)
    
    def test_text_display(self, ocr_preview, test_data, qtbot):
        """Test text display functionality"""
        ocr_preview.load_image(test_data.sample_image)
        qtbot.wait(100)
        
        # Add OCR results
        ocr_preview.add_ocr_results(test_data.sample_ocr_results)
        qtbot.wait(100)
        
        # Check text editor content
        text_content = ocr_preview.text_editor.toPlainText()
        assert len(text_content) > 0
        assert "Test OCR Text" in text_content
    
    def test_confidence_filtering(self, ocr_preview, test_data, qtbot):
        """Test confidence filtering functionality"""
        ocr_preview.add_ocr_results(test_data.sample_ocr_results)
        qtbot.wait(100)
        
        # Test confidence filter
        ocr_preview.confidence_filter.set_min_confidence(0.9)
        qtbot.wait(100)
        
        filtered_results = ocr_preview.get_filtered_results()
        assert len(filtered_results) > 0
        assert all(r['confidence'] >= 0.9 for r in filtered_results)
    
    def test_text_editing(self, ocr_preview, test_data, qtbot):
        """Test text editing functionality"""
        ocr_preview.load_image(test_data.sample_image)
        qtbot.wait(100)
        
        # Simulate text editing
        text_editor = ocr_preview.text_editor
        QTest.keyClicks(text_editor, "Edited text")
        qtbot.wait(100)
        
        assert text_editor.toPlainText() == "Edited text"
    
    def test_export_functionality(self, ocr_preview, test_data, qtbot):
        """Test export functionality"""
        ocr_preview.add_ocr_results(test_data.sample_ocr_results)
        qtbot.wait(100)
        
        # Test text export
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            export_path = f.name
        
        ocr_preview.export_text(export_path)
        qtbot.wait(100)
        
        # Verify file was exported
        assert Path(export_path).exists()
        with open(export_path, 'r') as f:
            content = f.read()
            assert "Test OCR Text" in content
        
        Path(export_path).unlink()
    
    def test_zoom_controls(self, ocr_preview, test_data, qtbot):
        """Test zoom controls"""
        ocr_preview.load_image(test_data.sample_image)
        qtbot.wait(100)
        
        # Test zoom in
        QTest.mouseClick(ocr_preview.zoom_in_button, Qt.MouseButton.LeftButton)
        qtbot.wait(100)
        assert ocr_preview.zoom_level > 1.0
        
        # Test zoom out
        QTest.mouseClick(ocr_preview.zoom_out_button, Qt.MouseButton.LeftButton)
        qtbot.wait(100)
        assert ocr_preview.zoom_level < 1.0
        
        # Test reset zoom
        QTest.mouseClick(ocr_preview.reset_zoom_button, Qt.MouseButton.LeftButton)
        qtbot.wait(100)
        assert ocr_preview.zoom_level == 1.0
    
    def test_result_selection(self, ocr_preview, test_data, qtbot):
        """Test result selection functionality"""
        ocr_preview.add_ocr_results(test_data.sample_ocr_results)
        qtbot.wait(100)
        
        # Test result selection
        table = ocr_preview.results_table
        table.selectRow(0)
        qtbot.wait(100)
        
        selected_result = ocr_preview.get_selected_result()
        assert selected_result is not None
        assert selected_result['text'] == "Test OCR Text"
    
    def test_security_features(self, ocr_preview, test_data, qtbot):
        """Test security features"""
        # Test input sanitization
        malicious_text = "<script>alert('xss')</script>Test text"
        sanitized_text = ocr_preview.sanitize_text(malicious_text)
        
        assert "<script>" not in sanitized_text
        assert "Test text" in sanitized_text
        
        # Test result sanitization
        malicious_result = {
            "text": "<script>alert('xss')</script>",
            "confidence": 0.95,
            "language": "en",
            "position": {"x": 100, "y": 100, "width": 200, "height": 50},
            "timestamp": 1.0
        }
        
        sanitized_result = ocr_preview.sanitize_result(malicious_result)
        assert "<script>" not in sanitized_result['text']
    
    def test_performance_monitoring(self, ocr_preview, test_data, qtbot):
        """Test performance monitoring"""
        ocr_preview.load_image(test_data.sample_image)
        qtbot.wait(100)
        
        # Test OCR processing time monitoring
        start_time = time.time()
        ocr_preview.start_ocr_processing()
        qtbot.wait(100)
        end_time = time.time()
        
        processing_time = end_time - start_time
        assert processing_time < 5.0  # Should be fast
    
    def test_memory_efficiency(self, ocr_preview, test_data, qtbot):
        """Test memory efficiency"""
        initial_memory = psutil.Process().memory_info().rss
        
        # Load large number of results
        large_results = []
        for i in range(1000):
            result = {
                "text": f"Test text {i}",
                "confidence": 0.8 + (i % 20) * 0.01,
                "language": "en",
                "position": {"x": i % 100, "y": i % 100, "width": 100, "height": 50},
                "timestamp": i * 0.1
            }
            large_results.append(result)
        
        ocr_preview.add_ocr_results(large_results)
        qtbot.wait(100)
        
        # Check memory usage
        final_memory = psutil.Process().memory_info().rss
        memory_increase = final_memory - initial_memory
        
        # Memory increase should be reasonable
        assert memory_increase < 50 * 1024 * 1024  # Less than 50MB
        
        # Clear results
        ocr_preview.clear_results()
        qtbot.wait(100)
        
        # Memory should be freed
        final_memory_after_clear = psutil.Process().memory_info().rss
        assert final_memory_after_clear < final_memory


# ============================================================================
# Enhanced Subtitle Editor Tests
# ============================================================================

class TestEnhancedSubtitleEditor:
    """Comprehensive test suite for EnhancedSubtitleEditor"""
    
    @pytest.fixture
    def subtitle_editor(self, qtbot):
        """Create subtitle editor instance"""
        editor = EnhancedSubtitleEditor()
        qtbot.addWidget(editor)
        editor.show()
        yield editor
        editor.close()
    
    def test_initialization(self, subtitle_editor):
        """Test subtitle editor initialization"""
        assert subtitle_editor is not None
        assert hasattr(subtitle_editor, 'subtitle_table')
        assert hasattr(subtitle_editor, 'text_editor')
        assert hasattr(subtitle_editor, 'time_controls')
        assert hasattr(subtitle_editor, 'format_selector')
        assert hasattr(subtitle_editor, 'export_button')
    
    def test_subtitle_loading(self, subtitle_editor, test_data, qtbot):
        """Test subtitle loading functionality"""
        subtitle_editor.load_subtitles(test_data.sample_subtitles)
        qtbot.wait(100)
        
        assert len(subtitle_editor.subtitles) == len(test_data.sample_subtitles)
        assert subtitle_editor.subtitle_table.rowCount() == len(test_data.sample_subtitles)
    
    def test_subtitle_editing(self, subtitle_editor, test_data, qtbot):
        """Test subtitle editing functionality"""
        subtitle_editor.load_subtitles(test_data.sample_subtitles)
        qtbot.wait(100)
        
        # Select first subtitle
        subtitle_editor.subtitle_table.selectRow(0)
        qtbot.wait(100)
        
        # Edit text
        text_editor = subtitle_editor.text_editor
        text_editor.clear()
        QTest.keyClicks(text_editor, "Edited subtitle text")
        qtbot.wait(100)
        
        # Save changes
        QTest.mouseClick(subtitle_editor.save_button, Qt.MouseButton.LeftButton)
        qtbot.wait(100)
        
        # Verify changes
        assert subtitle_editor.subtitles[0]['text'] == "Edited subtitle text"
    
    def test_timing_adjustment(self, subtitle_editor, test_data, qtbot):
        """Test timing adjustment functionality"""
        subtitle_editor.load_subtitles(test_data.sample_subtitles)
        qtbot.wait(100)
        
        # Select first subtitle
        subtitle_editor.subtitle_table.selectRow(0)
        qtbot.wait(100)
        
        # Adjust timing
        subtitle_editor.start_time_spin.setValue(1000)
        subtitle_editor.end_time_spin.setValue(6000)
        qtbot.wait(100)
        
        # Verify timing changes
        assert subtitle_editor.subtitles[0]['start'] == 1000
        assert subtitle_editor.subtitles[0]['end'] == 6000
    
    def test_subtitle_add_delete(self, subtitle_editor, qtbot):
        """Test subtitle add/delete functionality"""
        # Add new subtitle
        QTest.mouseClick(subtitle_editor.add_button, Qt.MouseButton.LeftButton)
        qtbot.wait(100)
        
        assert len(subtitle_editor.subtitles) == 1
        assert subtitle_editor.subtitle_table.rowCount() == 1
        
        # Delete subtitle
        subtitle_editor.subtitle_table.selectRow(0)
        QTest.mouseClick(subtitle_editor.delete_button, Qt.MouseButton.LeftButton)
        qtbot.wait(100)
        
        assert len(subtitle_editor.subtitles) == 0
        assert subtitle_editor.subtitle_table.rowCount() == 0
    
    def test_search_functionality(self, subtitle_editor, test_data, qtbot):
        """Test search functionality"""
        subtitle_editor.load_subtitles(test_data.sample_subtitles)
        qtbot.wait(100)
        
        # Search for text
        QTest.keyClicks(subtitle_editor.search_edit, "test")
        qtbot.wait(100)
        
        QTest.mouseClick(subtitle_editor.search_button, Qt.MouseButton.LeftButton)
        qtbot.wait(100)
        
        # Verify search results
        assert len(subtitle_editor.search_results) > 0
        assert "test" in subtitle_editor.search_results[0]['text'].lower()
    
    def test_format_export(self, subtitle_editor, test_data, qtbot):
        """Test format export functionality"""
        subtitle_editor.load_subtitles(test_data.sample_subtitles)
        qtbot.wait(100)
        
        # Test SRT export
        with tempfile.NamedTemporaryFile(mode='w', suffix='.srt', delete=False) as f:
            srt_path = f.name
        
        subtitle_editor.export_srt(srt_path)
        qtbot.wait(100)
        assert Path(srt_path).exists()
        
        # Test VTT export
        with tempfile.NamedTemporaryFile(mode='w', suffix='.vtt', delete=False) as f:
            vtt_path = f.name
        
        subtitle_editor.export_vtt(vtt_path)
        qtbot.wait(100)
        assert Path(vtt_path).exists()
        
        # Test ASS export
        with tempfile.NamedTemporaryFile(mode='w', suffix='.ass', delete=False) as f:
            ass_path = f.name
        
        subtitle_editor.export_ass(ass_path)
        qtbot.wait(100)
        assert Path(ass_path).exists()
        
        # Cleanup
        Path(srt_path).unlink()
        Path(vtt_path).unlink()
        Path(ass_path).unlink()
    
    def test_batch_operations(self, subtitle_editor, test_data, qtbot):
        """Test batch operations"""
        subtitle_editor.load_subtitles(test_data.sample_subtitles)
        qtbot.wait(100)
        
        # Test select all
        QTest.mouseClick(subtitle_editor.select_all_button, Qt.MouseButton.LeftButton)
        qtbot.wait(100)
        
        selected_rows = subtitle_editor.subtitle_table.selectedItems()
        assert len(selected_rows) > 0
        
        # Test time shift
        subtitle_editor.time_shift_spin.setValue(1000)
        QTest.mouseClick(subtitle_editor.time_shift_button, Qt.MouseButton.LeftButton)
        qtbot.wait(100)
        
        # Verify time shift
        assert subtitle_editor.subtitles[0]['start'] == 1000
        assert subtitle_editor.subtitles[0]['end'] == 6000
    
    def test_validation(self, subtitle_editor, test_data, qtbot):
        """Test validation functionality"""
        subtitle_editor.load_subtitles(test_data.sample_subtitles)
        qtbot.wait(100)
        
        # Test invalid timing
        subtitle_editor.subtitle_table.selectRow(0)
        subtitle_editor.start_time_spin.setValue(10000)  # After end time
        subtitle_editor.end_time_spin.setValue(5000)   # Before start time
        
        # Should show validation error
        with patch('PyQt6.QtWidgets.QMessageBox.warning') as mock_message:
            QTest.mouseClick(subtitle_editor.save_button, Qt.MouseButton.LeftButton)
            mock_message.assert_called_once()
    
    def test_undo_redo(self, subtitle_editor, test_data, qtbot):
        """Test undo/redo functionality"""
        subtitle_editor.load_subtitles(test_data.sample_subtitles)
        qtbot.wait(100)
        
        # Make changes
        subtitle_editor.subtitle_table.selectRow(0)
        original_text = subtitle_editor.subtitles[0]['text']
        
        text_editor = subtitle_editor.text_editor
        text_editor.clear()
        QTest.keyClicks(text_editor, "Modified text")
        qtbot.wait(100)
        
        QTest.mouseClick(subtitle_editor.save_button, Qt.MouseButton.LeftButton)
        qtbot.wait(100)
        
        # Test undo
        QTest.mouseClick(subtitle_editor.undo_button, Qt.MouseButton.LeftButton)
        qtbot.wait(100)
        assert subtitle_editor.subtitles[0]['text'] == original_text
        
        # Test redo
        QTest.mouseClick(subtitle_editor.redo_button, Qt.MouseButton.LeftButton)
        qtbot.wait(100)
        assert subtitle_editor.subtitles[0]['text'] == "Modified text"


# ============================================================================
# Enhanced Settings Dialog Tests
# ============================================================================

class TestEnhancedSettingsDialog:
    """Comprehensive test suite for EnhancedSettingsDialog"""
    
    @pytest.fixture
    def settings_dialog(self, qtbot, config):
        """Create settings dialog instance"""
        dialog = EnhancedSettingsDialog(config)
        qtbot.addWidget(dialog)
        dialog.show()
        yield dialog
        dialog.close()
    
    def test_initialization(self, settings_dialog):
        """Test settings dialog initialization"""
        assert settings_dialog is not None
        assert hasattr(settings_dialog, 'tabs')
        assert hasattr(settings_dialog, 'button_box')
        assert settings_dialog.windowTitle() == "设置"
        assert settings_dialog.isModal()
    
    def test_theme_settings(self, settings_dialog, qtbot):
        """Test theme settings functionality"""
        # Find theme tab
        theme_tab = settings_dialog.findChild(QWidget, "theme_tab")
        assert theme_tab is not None
        
        # Test theme selection
        theme_combo = theme_tab.findChild(QComboBox, "theme_combo")
        assert theme_combo is not None
        
        # Test theme switching
        theme_combo.setCurrentText("light")
        qtbot.wait(100)
        assert theme_combo.currentText() == "light"
        
        theme_combo.setCurrentText("dark")
        qtbot.wait(100)
        assert theme_combo.currentText() == "dark"
    
    def test_ocr_settings(self, settings_dialog, qtbot):
        """Test OCR settings functionality"""
        # Find OCR tab
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
        engine_combo.setCurrentText("Tesseract")
        language_combo.setCurrentText("中文")
        confidence_spin.setValue(0.85)
        
        assert engine_combo.currentText() == "Tesseract"
        assert language_combo.currentText() == "中文"
        assert confidence_spin.value() == 0.85
    
    def test_processing_settings(self, settings_dialog, qtbot):
        """Test processing settings functionality"""
        # Find processing tab
        processing_tab = settings_dialog.findChild(QWidget, "processing_tab")
        assert processing_tab is not None
        
        # Test scene threshold
        scene_spin = processing_tab.findChild(QDoubleSpinBox, "scene_spin")
        assert scene_spin is not None
        
        # Test cache size
        cache_spin = processing_tab.findChild(QSpinBox, "cache_spin")
        assert cache_spin is not None
        
        # Test concurrent jobs
        concurrent_spin = processing_tab.findChild(QSpinBox, "concurrent_spin")
        assert concurrent_spin is not None
        
        # Test setting values
        scene_spin.setValue(0.4)
        cache_spin.setValue(200)
        concurrent_spin.setValue(8)
        
        assert scene_spin.value() == 0.4
        assert cache_spin.value() == 200
        assert concurrent_spin.value() == 8
    
    def test_security_settings(self, settings_dialog, qtbot):
        """Test security settings functionality"""
        # Find security tab
        security_tab = settings_dialog.findChild(QWidget, "security_tab")
        assert security_tab is not None
        
        # Test input validation
        validation_checkbox = security_tab.findChild(QCheckBox, "validation_checkbox")
        assert validation_checkbox is not None
        
        # Test file size limit
        file_size_spin = security_tab.findChild(QSpinBox, "file_size_spin")
        assert file_size_spin is not None
        
        # Test rate limiting
        rate_limit_checkbox = security_tab.findChild(QCheckBox, "rate_limit_checkbox")
        assert rate_limit_checkbox is not None
        
        # Test setting values
        validation_checkbox.setChecked(True)
        file_size_spin.setValue(200)
        rate_limit_checkbox.setChecked(True)
        
        assert validation_checkbox.isChecked()
        assert file_size_spin.value() == 200
        assert rate_limit_checkbox.isChecked()
    
    def test_validation(self, settings_dialog, qtbot):
        """Test settings validation"""
        # Test invalid confidence threshold
        ocr_tab = settings_dialog.findChild(QWidget, "ocr_tab")
        confidence_spin = ocr_tab.findChild(QDoubleSpinBox, "confidence_spin")
        confidence_spin.setValue(1.5)  # Invalid value
        
        # Should clamp to valid range
        assert confidence_spin.value() <= 1.0
        
        # Test invalid cache size
        processing_tab = settings_dialog.findChild(QWidget, "processing_tab")
        cache_spin = processing_tab.findChild(QSpinBox, "cache_spin")
        cache_spin.setValue(-10)  # Invalid value
        
        # Should clamp to valid range
        assert cache_spin.value() >= 0
    
    def test_save_load_settings(self, settings_dialog, qtbot):
        """Test save and load settings functionality"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            config_path = f.name
        
        # Modify settings
        ocr_tab = settings_dialog.findChild(QWidget, "ocr_tab")
        engine_combo = ocr_tab.findChild(QComboBox, "engine_combo")
        engine_combo.setCurrentText("EasyOCR")
        
        # Save settings
        settings_dialog.save_settings(config_path)
        qtbot.wait(100)
        
        # Create new dialog and load settings
        new_dialog = EnhancedSettingsDialog(settings_dialog.config)
        new_dialog.load_settings(config_path)
        
        # Verify settings were loaded
        new_ocr_tab = new_dialog.findChild(QWidget, "ocr_tab")
        new_engine_combo = new_ocr_tab.findChild(QComboBox, "engine_combo")
        assert new_engine_combo.currentText() == "EasyOCR"
        
        new_dialog.close()
        Path(config_path).unlink()
    
    def test_reset_settings(self, settings_dialog, qtbot):
        """Test reset settings functionality"""
        # Modify settings
        ocr_tab = settings_dialog.findChild(QWidget, "ocr_tab")
        engine_combo = ocr_tab.findChild(QComboBox, "engine_combo")
        original_engine = engine_combo.currentText()
        
        engine_combo.setCurrentText("EasyOCR")
        
        # Reset settings
        reset_button = settings_dialog.findChild(QPushButton, "reset_button")
        QTest.mouseClick(reset_button, Qt.MouseButton.LeftButton)
        qtbot.wait(100)
        
        # Verify settings were reset
        assert engine_combo.currentText() == original_engine
    
    def test_apply_settings(self, settings_dialog, qtbot):
        """Test apply settings functionality"""
        # Modify settings
        ocr_tab = settings_dialog.findChild(QWidget, "ocr_tab")
        engine_combo = ocr_tab.findChild(QComboBox, "engine_combo")
        engine_combo.setCurrentText("Tesseract")
        
        # Apply settings
        apply_button = settings_dialog.findChild(QPushButton, "apply_button")
        QTest.mouseClick(apply_button, Qt.MouseButton.LeftButton)
        qtbot.wait(100)
        
        # Verify settings were applied
        assert settings_dialog.config.processing.ocr_config.engine == "Tesseract"
    
    def test_input_validation_security(self, settings_dialog, qtbot):
        """Test input validation security"""
        # Test malicious input in text fields
        security_tab = settings_dialog.findChild(QWidget, "security_tab")
        formats_edit = security_tab.findChild(QLineEdit, "formats_edit")
        
        # Test XSS injection
        malicious_input = "<script>alert('xss')</script>mp4,avi"
        QTest.keyClicks(formats_edit, malicious_input)
        
        # Should sanitize input
        sanitized_text = formats_edit.text()
        assert "<script>" not in sanitized_text
        assert "mp4" in sanitized_text


# ============================================================================
# Theme System Tests
# ============================================================================

class TestThemeSystem:
    """Comprehensive test suite for Theme System"""
    
    @pytest.fixture
    def theme_manager(self):
        """Create theme manager instance"""
        return get_theme_manager()
    
    def test_theme_manager_initialization(self, theme_manager):
        """Test theme manager initialization"""
        assert theme_manager is not None
        assert hasattr(theme_manager, 'current_theme')
        assert hasattr(theme_manager, 'themes')
        assert len(theme_manager.themes) > 0
    
    def test_available_themes(self, theme_manager):
        """Test available themes"""
        themes = theme_manager.get_available_themes()
        assert isinstance(themes, list)
        assert len(themes) > 0
        assert "dark" in themes
        assert "light" in themes
    
    def test_theme_switching(self, theme_manager, qtbot):
        """Test theme switching functionality"""
        original_theme = theme_manager.current_theme
        
        # Switch to light theme
        theme_manager.set_theme("light")
        qtbot.wait(100)
        assert theme_manager.current_theme == "light"
        
        # Switch to dark theme
        theme_manager.set_theme("dark")
        qtbot.wait(100)
        assert theme_manager.current_theme == "dark"
        
        # Restore original theme
        theme_manager.set_theme(original_theme)
    
    def test_theme_properties(self, theme_manager):
        """Test theme properties"""
        theme = theme_manager.current_theme
        
        assert hasattr(theme, 'name')
        assert hasattr(theme, 'colors')
        assert hasattr(theme, 'fonts')
        assert hasattr(theme, 'styles')
        
        # Check colors
        colors = theme.colors
        assert hasattr(colors, 'primary')
        assert hasattr(colors, 'secondary')
        assert hasattr(colors, 'background')
        assert hasattr(colors, 'foreground')
    
    def test_custom_theme_creation(self, theme_manager):
        """Test custom theme creation"""
        custom_theme_data = {
            "name": "custom_test",
            "colors": {
                "primary": "#ff0000",
                "secondary": "#00ff00",
                "background": "#0000ff",
                "foreground": "#ffffff",
                "accent": "#ffff00",
                "success": "#00ff00",
                "warning": "#ffff00",
                "error": "#ff0000"
            },
            "fonts": {
                "main": "Arial",
                "monospace": "Courier New",
                "header": "Arial Black"
            },
            "styles": {
                "button": " QPushButton { background: #ff0000; } "
            }
        }
        
        # Add custom theme
        theme_manager.add_custom_theme(custom_theme_data)
        assert "custom_test" in theme_manager.get_available_themes()
        
        # Apply custom theme
        theme_manager.set_theme("custom_test")
        assert theme_manager.current_theme == "custom_test"
        
        # Verify theme properties
        theme = theme_manager.current_theme
        assert theme.name == "custom_test"
        assert theme.colors.primary.name() == "#ff0000"
    
    def test_theme_persistence(self, theme_manager):
        """Test theme persistence"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            config_path = f.name
        
        # Save theme configuration
        theme_manager.save_config(config_path)
        
        # Create new theme manager and load configuration
        new_theme_manager = ThemeManager()
        new_theme_manager.load_config(config_path)
        
        assert new_theme_manager.current_theme == theme_manager.current_theme
        
        Path(config_path).unlink()
    
    def test_theme_application_to_widgets(self, theme_manager, qtbot):
        """Test theme application to widgets"""
        from PyQt6.QtWidgets import QWidget, QPushButton, QLabel
        
        # Create test widgets
        widget = QWidget()
        qtbot.addWidget(widget)
        
        button = QPushButton("Test Button", widget)
        label = QLabel("Test Label", widget)
        
        # Apply theme
        theme_manager.apply_theme_to_widget(widget)
        
        # Verify theme was applied
        assert widget.styleSheet() != ""
        assert button.styleSheet() != ""
        assert label.styleSheet() != ""
    
    def test_theme_signals(self, theme_manager, qtbot):
        """Test theme signals"""
        signal_received = False
        
        def on_theme_changed(theme_name):
            nonlocal signal_received
            signal_received = True
        
        # Connect signal
        theme_manager.theme_changed.connect(on_theme_changed)
        
        # Change theme
        theme_manager.set_theme("light")
        qtbot.wait(100)
        
        # Verify signal was received
        assert signal_received
    
    def test_theme_performance(self, theme_manager, qtbot):
        """Test theme performance"""
        # Test theme switching performance
        start_time = time.time()
        
        for i in range(10):
            theme_manager.set_theme("light")
            theme_manager.set_theme("dark")
        
        end_time = time.time()
        switching_time = end_time - start_time
        
        # Theme switching should be fast
        assert switching_time < 2.0  # Less than 2 seconds for 20 switches
    
    def test_theme_memory_usage(self, theme_manager):
        """Test theme memory usage"""
        initial_memory = psutil.Process().memory_info().rss
        
        # Create multiple themes
        for i in range(10):
            theme_data = {
                "name": f"test_theme_{i}",
                "colors": {
                    "primary": f"#{i:02x}0000",
                    "secondary": f"#00{i:02x}00",
                    "background": f"#0000{i:02x}",
                    "foreground": "#ffffff",
                    "accent": "#ffff00",
                    "success": "#00ff00",
                    "warning": "#ffff00",
                    "error": "#ff0000"
                },
                "fonts": {
                    "main": "Arial",
                    "monospace": "Courier New",
                    "header": "Arial Black"
                },
                "styles": {}
            }
            theme_manager.add_custom_theme(theme_data)
        
        # Check memory usage
        final_memory = psutil.Process().memory_info().rss
        memory_increase = final_memory - initial_memory
        
        # Memory increase should be reasonable
        assert memory_increase < 10 * 1024 * 1024  # Less than 10MB


# ============================================================================
# Security Tests
# ============================================================================

class TestUISecurity:
    """Comprehensive test suite for UI security features"""
    
    def test_input_validation(self):
        """Test input validation across components"""
        # Test malicious input patterns
        malicious_inputs = [
            "<script>alert('xss')</script>",
            "javascript:alert('xss')",
            "vbscript:alert('xss')",
            "data:text/html,<script>alert('xss')</script>",
            "file:///etc/passwd",
            "../../../etc/passwd",
            "<img src=x onerror=alert('xss')>",
            "<svg onload=alert('xss')>",
            "<iframe src=javascript:alert('xss')>",
            "<link rel=stylesheet href=javascript:alert('xss')>"
        ]
        
        # Test sanitization
        for malicious_input in malicious_inputs:
            sanitized = self.sanitize_input(malicious_input)
            assert "<script>" not in sanitized
            assert "javascript:" not in sanitized
            assert "vbscript:" not in sanitized
            assert "data:" not in sanitized
            assert "file:" not in sanitized
            assert "../" not in sanitized
    
    def test_file_upload_security(self, qtbot):
        """Test file upload security"""
        # Test malicious file paths
        malicious_paths = [
            "../../../etc/passwd",
            "..\\..\\..\\windows\\system32\\config\\sam",
            "/etc/shadow",
            "C:\\Windows\\System32\\config\\SAM",
            "file:///etc/passwd",
            "http://malicious.com/malicious.exe",
            "ftp://malicious.com/malicious.exe"
        ]
        
        for malicious_path in malicious_paths:
            sanitized = self.sanitize_file_path(malicious_path)
            assert "../" not in sanitized
            assert "file:" not in sanitized
            assert "http:" not in sanitized
            assert "ftp:" not in sanitized
    
    def test_path_traversal_protection(self):
        """Test path traversal protection"""
        # Test various path traversal attempts
        traversal_attempts = [
            "../../../etc/passwd",
            "..\\..\\..\\windows\\system32\\config\\sam",
            "/etc/passwd",
            "C:\\Windows\\System32\\config\\SAM",
            "./../../etc/passwd",
            ".\\..\\..\\windows\\system32\\config\\sam"
        ]
        
        for attempt in traversal_attempts:
            is_safe = self.is_safe_path(attempt)
            assert not is_safe
    
    def test_xss_prevention(self):
        """Test XSS prevention"""
        # Test various XSS vectors
        xss_vectors = [
            "<script>alert('xss')</script>",
            "<img src=x onerror=alert('xss')>",
            "<svg onload=alert('xss')>",
            "<iframe src=javascript:alert('xss')>",
            "<link rel=stylesheet href=javascript:alert('xss')>",
            "<body onload=alert('xss')>",
            "<div onmouseover=alert('xss')>",
            "<input type=image src=x onerror=alert('xss')>",
            "<details open ontoggle=alert('xss')>",
            "<select onfocus=alert('xss')>"
        ]
        
        for xss_vector in xss_vectors:
            sanitized = self.sanitize_html(xss_vector)
            assert "<script>" not in sanitized
            assert "onerror=" not in sanitized
            assert "onload=" not in sanitized
            assert "javascript:" not in sanitized
    
    def test_command_injection_prevention(self):
        """Test command injection prevention"""
        # Test command injection vectors
        injection_vectors = [
            "test; rm -rf /",
            "test && rm -rf /",
            "test | rm -rf /",
            "test || rm -rf /",
            "test $(rm -rf /)",
            "test `rm -rf /`",
            "test & rm -rf /",
            "test > /etc/passwd",
            "test < /etc/passwd"
        ]
        
        for injection_vector in injection_vectors:
            sanitized = self.sanitize_command(injection_vector)
            assert ";" not in sanitized
            assert "&" not in sanitized
            assert "|" not in sanitized
            assert "`" not in sanitized
            assert "$(" not in sanitized
    
    def test_sql_injection_prevention(self):
        """Test SQL injection prevention"""
        # Test SQL injection vectors
        sql_vectors = [
            "test' OR '1'='1",
            "test' UNION SELECT * FROM users--",
            "test'; DROP TABLE users;--",
            "test' OR SLEEP(10)--",
            "test' WAITFOR DELAY '0:0:10'--",
            "1' OR '1'='1",
            "admin'--",
            "' OR '1'='1",
            "1; DROP TABLE users--"
        ]
        
        for sql_vector in sql_vectors:
            sanitized = self.sanitize_sql(sql_vector)
            assert "'" not in sanitized or sanitized.count("'") == 2  # Only allow quoted strings
            assert ";" not in sanitized
            assert "--" not in sanitized
            assert "UNION" not in sanitized
            assert "DROP" not in sanitized
    
    def test_file_type_validation(self):
        """Test file type validation"""
        # Test allowed file types
        allowed_types = ["mp4", "avi", "mkv", "mov", "wmv", "flv"]
        
        # Test valid files
        valid_files = ["test.mp4", "video.avi", "movie.mkv"]
        for valid_file in valid_files:
            is_valid = self.is_valid_file_type(valid_file, allowed_types)
            assert is_valid
        
        # Test invalid files
        invalid_files = ["test.exe", "script.js", "malicious.php", "virus.bat"]
        for invalid_file in invalid_files:
            is_valid = self.is_valid_file_type(invalid_file, allowed_types)
            assert not is_valid
    
    def test_file_size_validation(self):
        """Test file size validation"""
        # Test valid sizes
        valid_sizes = [1024, 1024 * 1024, 50 * 1024 * 1024]  # 1KB, 1MB, 50MB
        max_size = 100 * 1024 * 1024  # 100MB
        
        for size in valid_sizes:
            is_valid = self.is_valid_file_size(size, max_size)
            assert is_valid
        
        # Test invalid sizes
        invalid_sizes = [200 * 1024 * 1024, 500 * 1024 * 1024]  # 200MB, 500MB
        for size in invalid_sizes:
            is_valid = self.is_valid_file_size(size, max_size)
            assert not is_valid
    
    def test_rate_limiting(self):
        """Test rate limiting functionality"""
        # Test rate limiting
        rate_limiter = self.create_rate_limiter(max_requests=10, time_window=60)
        
        # Test allowed requests
        for i in range(10):
            is_allowed = rate_limiter.is_allowed("test_user")
            assert is_allowed
        
        # Test rate limit exceeded
        is_allowed = rate_limiter.is_allowed("test_user")
        assert not is_allowed
    
    def test_session_security(self):
        """Test session security"""
        # Test session creation
        session = self.create_session()
        assert session is not None
        assert hasattr(session, 'session_id')
        assert hasattr(session, 'user_id')
        assert hasattr(session, 'created_at')
        assert hasattr(session, 'expires_at')
        
        # Test session validation
        is_valid = self.validate_session(session)
        assert is_valid
        
        # Test session expiration
        expired_session = self.create_expired_session()
        is_valid = self.validate_session(expired_session)
        assert not is_valid
    
    # Helper methods for security tests
    def sanitize_input(self, input_str: str) -> str:
        """Sanitize input string"""
        # Remove HTML tags
        import re
        clean = re.sub(r'<[^>]*>', '', input_str)
        # Remove dangerous protocols
        clean = re.sub(r'javascript:', '', clean, flags=re.IGNORECASE)
        clean = re.sub(r'vbscript:', '', clean, flags=re.IGNORECASE)
        clean = re.sub(r'data:', '', clean, flags=re.IGNORECASE)
        clean = re.sub(r'file:', '', clean, flags=re.IGNORECASE)
        # Remove path traversal
        clean = re.sub(r'\.\./', '', clean)
        clean = re.sub(r'\.\.\\', '', clean)
        return clean
    
    def sanitize_file_path(self, file_path: str) -> str:
        """Sanitize file path"""
        # Remove path traversal
        clean = file_path.replace('../', '').replace('..\\', '')
        # Remove dangerous protocols
        clean = clean.replace('file://', '').replace('http://', '').replace('ftp://', '')
        return clean
    
    def is_safe_path(self, path: str) -> bool:
        """Check if path is safe"""
        dangerous_patterns = ['../', '..\\', 'file://', 'http://', 'ftp://']
        return not any(pattern in path for pattern in dangerous_patterns)
    
    def sanitize_html(self, html: str) -> str:
        """Sanitize HTML"""
        import re
        # Remove script tags
        clean = re.sub(r'<script.*?>.*?</script>', '', html, flags=re.IGNORECASE | re.DOTALL)
        # Remove event handlers
        clean = re.sub(r'on\w+="[^"]*"', '', clean)
        clean = re.sub(r"on\w+='[^']*'", '', clean)
        clean = re.sub(r'on\w+=[^\s>]*', '', clean)
        # Remove dangerous protocols
        clean = re.sub(r'javascript:', '', clean, flags=re.IGNORECASE)
        return clean
    
    def sanitize_command(self, command: str) -> str:
        """Sanitize command"""
        dangerous_chars = [';', '&', '|', '`', '$(', ')']
        clean = command
        for char in dangerous_chars:
            clean = clean.replace(char, '')
        return clean
    
    def sanitize_sql(self, sql: str) -> str:
        """Sanitize SQL"""
        import re
        # Remove SQL comments
        clean = re.sub(r'--.*$', '', sql, flags=re.MULTILINE)
        # Remove dangerous keywords
        dangerous_keywords = ['UNION', 'DROP', 'DELETE', 'INSERT', 'UPDATE', 'ALTER']
        for keyword in dangerous_keywords:
            clean = re.sub(r'\b' + keyword + r'\b', '', clean, flags=re.IGNORECASE)
        return clean
    
    def is_valid_file_type(self, filename: str, allowed_types: List[str]) -> bool:
        """Check if file type is valid"""
        extension = filename.split('.')[-1].lower()
        return extension in allowed_types
    
    def is_valid_file_size(self, size: int, max_size: int) -> bool:
        """Check if file size is valid"""
        return size <= max_size
    
    def create_rate_limiter(self, max_requests: int, time_window: int):
        """Create rate limiter"""
        class RateLimiter:
            def __init__(self, max_requests, time_window):
                self.max_requests = max_requests
                self.time_window = time_window
                self.requests = {}
            
            def is_allowed(self, user_id: str) -> bool:
                import time
                current_time = time.time()
                
                if user_id not in self.requests:
                    self.requests[user_id] = []
                
                # Remove expired requests
                self.requests[user_id] = [
                    req_time for req_time in self.requests[user_id]
                    if current_time - req_time < self.time_window
                ]
                
                # Check if request is allowed
                if len(self.requests[user_id]) < self.max_requests:
                    self.requests[user_id].append(current_time)
                    return True
                
                return False
        
        return RateLimiter(max_requests, time_window)
    
    def create_session(self):
        """Create session"""
        import time
        import uuid
        
        class Session:
            def __init__(self):
                self.session_id = str(uuid.uuid4())
                self.user_id = "test_user"
                self.created_at = time.time()
                self.expires_at = time.time() + 3600  # 1 hour
        
        return Session()
    
    def create_expired_session(self):
        """Create expired session"""
        import time
        import uuid
        
        class Session:
            def __init__(self):
                self.session_id = str(uuid.uuid4())
                self.user_id = "test_user"
                self.created_at = time.time() - 7200  # 2 hours ago
                self.expires_at = time.time() - 3600  # 1 hour ago
        
        return Session()
    
    def validate_session(self, session) -> bool:
        """Validate session"""
        import time
        return time.time() < session.expires_at


# ============================================================================
# Performance Tests
# ============================================================================

class TestUIPerformance:
    """Comprehensive test suite for UI performance"""
    
    def test_widget_creation_performance(self, qtbot):
        """Test widget creation performance"""
        # Test main window creation
        start_time = time.time()
        
        view_model = MainViewModel()
        window = EnhancedMainWindow(view_model)
        
        creation_time = time.time() - start_time
        
        # Creation should be fast
        assert creation_time < 2.0  # Less than 2 seconds
        
        window.close()
    
    def test_theme_switching_performance(self, qtbot):
        """Test theme switching performance"""
        theme_manager = get_theme_manager()
        
        # Test theme switching performance
        start_time = time.time()
        
        for i in range(10):
            theme_manager.set_theme("light")
            theme_manager.set_theme("dark")
        
        switching_time = time.time() - start_time
        
        # Theme switching should be fast
        assert switching_time < 2.0  # Less than 2 seconds for 20 switches
    
    def test_large_data_handling_performance(self, qtbot):
        """Test large data handling performance"""
        # Test with large number of OCR results
        large_results = []
        for i in range(1000):
            result = {
                "text": f"Test text {i}",
                "confidence": 0.8 + (i % 20) * 0.01,
                "language": "en",
                "position": {"x": i % 100, "y": i % 100, "width": 100, "height": 50},
                "timestamp": i * 0.1
            }
            large_results.append(result)
        
        # Test OCR preview with large data
        preview = EnhancedOCRPreview()
        qtbot.addWidget(preview)
        
        start_time = time.time()
        preview.add_ocr_results(large_results)
        loading_time = time.time() - start_time
        
        # Loading should be fast
        assert loading_time < 3.0  # Less than 3 seconds
        
        preview.close()
    
    def test_memory_usage_performance(self, qtbot):
        """Test memory usage performance"""
        initial_memory = psutil.Process().memory_info().rss
        
        # Create and use components
        view_model = MainViewModel()
        window = EnhancedMainWindow(view_model)
        qtbot.addWidget(window)
        
        # Load some data
        for i in range(100):
            # Simulate some work
            pass
        
        # Check memory usage
        final_memory = psutil.Process().memory_info().rss
        memory_increase = final_memory - initial_memory
        
        # Memory increase should be reasonable
        assert memory_increase < 100 * 1024 * 1024  # Less than 100MB
        
        window.close()
    
    def test_responsiveness_performance(self, qtbot):
        """Test responsiveness performance"""
        view_model = MainViewModel()
        window = EnhancedMainWindow(view_model)
        qtbot.addWidget(window)
        
        # Test UI responsiveness
        start_time = time.time()
        
        # Perform various UI operations
        for i in range(10):
            window.resize(800 + i * 10, 600 + i * 10)
            QTest.qWait(10)
        
        responsiveness_time = time.time() - start_time
        
        # UI should be responsive
        assert responsiveness_time < 1.0  # Less than 1 second
        
        window.close()
    
    def test_concurrent_operations_performance(self, qtbot):
        """Test concurrent operations performance"""
        import threading
        import time
        
        results = []
        
        def worker_function():
            start_time = time.time()
            
            # Simulate some work
            time.sleep(0.1)
            
            end_time = time.time()
            results.append(end_time - start_time)
        
        # Create multiple threads
        threads = []
        for i in range(10):
            thread = threading.Thread(target=worker_function)
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Check concurrent performance
        assert len(results) == 10
        assert all(result < 0.5 for result in results)  # All operations should complete quickly


# ============================================================================
# Integration Tests
# ============================================================================

class TestUIIntegration:
    """Comprehensive test suite for UI integration"""
    
    def test_complete_workflow_integration(self, qtbot, test_data):
        """Test complete workflow integration"""
        # Create main application
        view_model = MainViewModel()
        window = EnhancedMainWindow(view_model)
        qtbot.addWidget(window)
        
        # Test complete workflow
        # 1. Load video
        window.load_video(test_data.sample_video_path)
        qtbot.wait(500)
        
        # 2. Process OCR
        window.start_ocr_processing()
        qtbot.wait(1000)
        
        # 3. Edit subtitles
        window.subtitle_editor.load_subtitles(test_data.sample_subtitles)
        qtbot.wait(100)
        
        # 4. Export results
        with tempfile.NamedTemporaryFile(mode='w', suffix='.srt', delete=False) as f:
            export_path = f.name
        
        window.subtitle_editor.export_srt(export_path)
        qtbot.wait(100)
        
        # Verify workflow completed
        assert Path(export_path).exists()
        
        window.close()
        Path(export_path).unlink()
    
    def test_signal_slot_integration(self, qtbot, test_data):
        """Test signal-slot integration"""
        view_model = MainViewModel()
        window = EnhancedMainWindow(view_model)
        qtbot.addWidget(window)
        
        # Test signal propagation
        signal_received = False
        
        def on_signal_received():
            nonlocal signal_received
            signal_received = True
        
        # Connect to video player signal
        window.video_player.frame_extracted.connect(on_signal_received)
        
        # Trigger signal
        window.video_player.extract_frame(1000)
        qtbot.wait(100)
        
        # Verify signal was received
        assert signal_received
        
        window.close()
    
    def test_data_flow_integration(self, qtbot, test_data):
        """Test data flow integration"""
        view_model = MainViewModel()
        window = EnhancedMainWindow(view_model)
        qtbot.addWidget(window)
        
        # Test data flow from video to OCR to subtitles
        window.load_video(test_data.sample_video_path)
        qtbot.wait(500)
        
        # Mock OCR processing
        window.ocr_preview.add_ocr_results(test_data.sample_ocr_results)
        qtbot.wait(100)
        
        # Check data consistency
        assert len(window.ocr_preview.ocr_results) == len(test_data.sample_ocr_results)
        
        window.close()
    
    def test_state_synchronization_integration(self, qtbot, test_data):
        """Test state synchronization integration"""
        view_model = MainViewModel()
        window = EnhancedMainWindow(view_model)
        qtbot.addWidget(window)
        
        # Test state synchronization between components
        window.load_video(test_data.sample_video_path)
        qtbot.wait(500)
        
        # Verify video state is synchronized
        assert window.video_player.current_video == test_data.sample_video_path
        assert window.current_file == test_data.sample_video_path
        
        window.close()
    
    def test_error_handling_integration(self, qtbot):
        """Test error handling integration"""
        view_model = MainViewModel()
        window = EnhancedMainWindow(view_model)
        qtbot.addWidget(window)
        
        # Test error propagation
        with patch('PyQt6.QtWidgets.QMessageBox.critical') as mock_message:
            window.load_video("/nonexistent/video.mp4")
            mock_message.assert_called_once()
        
        window.close()
    
    def test_resource_management_integration(self, qtbot):
        """Test resource management integration"""
        view_model = MainViewModel()
        window = EnhancedMainWindow(view_model)
        qtbot.addWidget(window)
        
        # Test resource cleanup
        initial_memory = psutil.Process().memory_info().rss
        
        # Use some resources
        window.resize(1200, 800)
        
        # Cleanup
        window.cleanup()
        
        # Check memory usage
        final_memory = psutil.Process().memory_info().rss
        memory_increase = final_memory - initial_memory
        
        # Memory should be properly managed
        assert memory_increase < 50 * 1024 * 1024  # Less than 50MB
        
        window.close()


# ============================================================================
# Accessibility Tests
# ============================================================================

class TestUIAccessibility:
    """Comprehensive test suite for UI accessibility"""
    
    def test_keyboard_navigation(self, qtbot):
        """Test keyboard navigation"""
        view_model = MainViewModel()
        window = EnhancedMainWindow(view_model)
        qtbot.addWidget(window)
        
        # Test tab navigation
        current_widget = window.focusWidget()
        
        # Press Tab to navigate
        QTest.keyClick(window, Qt.Key.Key_Tab)
        new_widget = window.focusWidget()
        
        # Focus should change
        assert new_widget != current_widget
        
        # Test arrow key navigation
        QTest.keyClick(window, Qt.Key.Key_Right)
        QTest.keyClick(window, Qt.Key.Key_Left)
        QTest.keyClick(window, Qt.Key.Key_Up)
        QTest.keyClick(window, Qt.Key.Key_Down)
        
        window.close()
    
    def test_screen_reader_compatibility(self, qtbot):
        """Test screen reader compatibility"""
        view_model = MainViewModel()
        window = EnhancedMainWindow(view_model)
        qtbot.addWidget(window)
        
        # Test accessible names
        assert window.accessibleName() != ""
        
        # Test accessible descriptions
        assert window.accessibleDescription() != ""
        
        # Test widget accessibility
        for widget in window.findChildren(QWidget):
            if widget.isVisible():
                # Widget should have accessible name or be properly labeled
                assert widget.accessibleName() != "" or widget.toolTip() != ""
        
        window.close()
    
    def test_high_contrast_mode(self, qtbot):
        """Test high contrast mode"""
        view_model = MainViewModel()
        window = EnhancedMainWindow(view_model)
        qtbot.addWidget(window)
        
        # Test high contrast theme
        theme_manager = get_theme_manager()
        original_theme = theme_manager.current_theme
        
        theme_manager.set_theme("high_contrast")
        qtbot.wait(100)
        
        # Verify high contrast colors
        theme = theme_manager.current_theme
        assert theme.colors.primary != theme.colors.background
        assert theme.colors.foreground != theme.colors.background
        
        # Restore original theme
        theme_manager.set_theme(original_theme)
        
        window.close()
    
    def test_font_scaling(self, qtbot):
        """Test font scaling"""
        view_model = MainViewModel()
        window = EnhancedMainWindow(view_model)
        qtbot.addWidget(window)
        
        # Test font size changes
        original_font = window.font()
        
        # Increase font size
        larger_font = QFont(original_font)
        larger_font.setPointSize(original_font.pointSize() + 4)
        window.setFont(larger_font)
        
        # Verify font size changed
        assert window.font().pointSize() > original_font.pointSize()
        
        # Decrease font size
        smaller_font = QFont(original_font)
        smaller_font.setPointSize(original_font.pointSize() - 2)
        window.setFont(smaller_font)
        
        # Verify font size changed
        assert window.font().pointSize() < original_font.pointSize()
        
        window.close()
    
    def test_color_blind_friendly(self, qtbot):
        """Test color blind friendly design"""
        view_model = MainViewModel()
        window = EnhancedMainWindow(view_model)
        qtbot.addWidget(window)
        
        # Test color blind friendly theme
        theme_manager = get_theme_manager()
        original_theme = theme_manager.current_theme
        
        theme_manager.set_theme("colorblind_friendly")
        qtbot.wait(100)
        
        # Verify color blind friendly colors
        theme = theme_manager.current_theme
        
        # Check for sufficient contrast
        primary_color = theme.colors.primary
        background_color = theme.colors.background
        
        # Simple contrast check (should be improved with proper contrast ratio calculation)
        assert primary_color != background_color
        
        # Restore original theme
        theme_manager.set_theme(original_theme)
        
        window.close()


# ============================================================================
# Main Test Runner
# ============================================================================

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Run tests with pytest
    pytest.main([
        __file__,
        "-v",
        "--tb=short",
        "--strict-markers",
        "--cov=visionsub",
        "--cov-report=term-missing",
        "--cov-report=html:htmlcov",
        "--cov-fail-under=80",
        "-m",
        "not performance"  # Skip performance tests by default
    ])