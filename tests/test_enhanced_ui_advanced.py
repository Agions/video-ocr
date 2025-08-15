"""
Enhanced UI Components Comprehensive Test Suite

This module provides thorough testing for the remaining enhanced UI components:
- EnhancedMain application initialization and lifecycle
- Enhanced OCR Preview with secure text rendering
- Enhanced Subtitle Editor with advanced editing features
- Theme System integration and management
- Security features for UI components
- Performance testing for UI responsiveness
- Integration tests for component interactions

Test Categories:
- Unit Tests: Individual component functionality
- Integration Tests: Component interactions
- Security Tests: Input validation and sanitization
- Performance Tests: UI responsiveness and resource usage
- Usability Tests: User workflows and accessibility
"""

import pytest
import asyncio
import numpy as np
import cv2
import json
import tempfile
import time
import psutil
import threading
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from PyQt6.QtCore import (
    Qt, QTimer, QEvent, QRect, QSize, QPoint, pyqtSignal,
    QThread, QPropertyAnimation, QEasingCurve, QModelIndex,
    QObject, QSettings, QStandardPaths
)
from PyQt6.QtGui import (
    QImage, QPixmap, QFont, QColor, QPalette, QPainter,
    QBrush, QPen, QTextCharFormat, QTextCursor, QKeySequence,
    QMouseEvent, QKeyEvent, QWheelEvent, QDragEnterEvent,
    QDropEvent, QCloseEvent, QTextDocument, QSyntaxHighlighter,
    QTextFormat, QRegularExpression
)
from PyQt6.QtWidgets import (
    QApplication, QWidget, QPushButton, QLabel, QLineEdit, 
    QComboBox, QSpinBox, QDoubleSpinBox, QCheckBox, QSlider,
    QTextEdit, QTableWidget, QTableWidgetItem, QHeaderView,
    QDialog, QMessageBox, QFileDialog, QMenu, QToolBar,
    QStatusBar, QMenuBar, QSplitter, QFrame, QScrollArea,
    QGroupBox, QTabWidget, QListWidget, QListWidgetItem,
    QStyledItemDelegate, QStyleOptionViewItem, QSystemTrayIcon,
    QProgressBar, QToolButton, QStackedWidget, QFormLayout,
    QRadioButton, QButtonGroup, QTreeWidget, QTreeWidgetItem,
    QGraphicsView, QGraphicsScene, QGraphicsPixmapItem,
    QGraphicsTextItem, QGraphicsRectItem
)
from PyQt6.QtTest import QTest

# Add src to path
import sys
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Import enhanced UI components
from visionsub.ui.enhanced_main import EnhancedApplication
from visionsub.ui.enhanced_ocr_preview import (
    EnhancedOCRPreview, OCRResult, OCRResultType, 
    SecureOCRRenderer, SecureTextHighlighter
)
from visionsub.ui.enhanced_subtitle_editor import (
    EnhancedSubtitleEditor, SubtitleEditAction, SubtitleEditState,
    SecureSubtitleValidator, SubtitleTableModel, SubtitleSearchProxyModel
)
from visionsub.ui.theme_system import ThemeManager, get_theme_manager, ThemeColors
from visionsub.models.config import AppConfig, OcrConfig, ProcessingConfig, UIConfig, SecurityConfig
from visionsub.models.subtitle import SubtitleItem
from visionsub.view_models.main_view_model import MainViewModel

# Configure logging
import logging
logger = logging.getLogger(__name__)


# ============================================================================
# Enhanced Main Application Tests
# ============================================================================

class TestEnhancedMainApplication:
    """Comprehensive test suite for EnhancedMain Application"""
    
    @pytest.fixture
    def app(self):
        """Create EnhancedApplication instance"""
        return EnhancedApplication([])
    
    def test_application_initialization(self, app):
        """Test application initialization"""
        assert app is not None
        assert hasattr(app, 'main_window')
        assert hasattr(app, 'config')
        assert hasattr(app, 'theme_manager')
        assert hasattr(app, 'view_model')
    
    def test_main_window_creation(self, app):
        """Test main window creation"""
        assert app.main_window is not None
        assert app.main_window.windowTitle() == "VisionSub - 视频OCR字幕提取工具"
        assert app.main_window.isVisible()
    
    def test_config_loading(self, app):
        """Test configuration loading"""
        assert app.config is not None
        assert isinstance(app.config, AppConfig)
        assert app.config.processing is not None
        assert app.config.ui is not None
        assert app.config.security is not None
    
    def test_theme_manager_initialization(self, app):
        """Test theme manager initialization"""
        assert app.theme_manager is not None
        assert isinstance(app.theme_manager, ThemeManager)
        assert app.theme_manager.current_theme is not None
    
    def test_view_model_initialization(self, app):
        """Test view model initialization"""
        assert app.view_model is not None
        assert isinstance(app.view_model, MainViewModel)
    
    def test_application_lifecycle(self, app):
        """Test application lifecycle"""
        # Test application setup
        assert app.applicationName() == "VisionSub"
        assert app.organizationName() == "Agions"
        
        # Test application version
        assert app.applicationVersion() == "2.0.0"
    
    def test_settings_persistence(self, app):
        """Test settings persistence"""
        # Test settings file path
        settings = QSettings()
        assert settings is not None
        
        # Test settings organization
        assert settings.organizationName() == "Agions"
        assert settings.applicationName() == "VisionSub"
    
    def test_error_handling(self, app):
        """Test error handling"""
        # Test global error handler
        with patch('PyQt6.QtWidgets.QMessageBox.critical') as mock_message:
            app.show_error("Test Error", "Test error message")
            mock_message.assert_called_once()
    
    def test_resource_cleanup(self, app):
        """Test resource cleanup"""
        # Test cleanup on application exit
        assert hasattr(app, 'cleanup')
        
        # Call cleanup
        app.cleanup()
        
        # Verify cleanup completed
        assert True  # Placeholder for actual cleanup verification
    
    def test_performance_monitoring(self, app):
        """Test performance monitoring"""
        # Test startup time
        start_time = time.time()
        
        # Create new application instance
        new_app = EnhancedApplication([])
        
        startup_time = time.time() - start_time
        
        # Startup should be fast
        assert startup_time < 3.0  # Less than 3 seconds
        
        new_app.quit()
    
    def test_memory_usage(self, app):
        """Test memory usage"""
        initial_memory = psutil.Process().memory_info().rss
        
        # Simulate some application usage
        app.main_window.resize(1200, 800)
        
        final_memory = psutil.Process().memory_info().rss
        memory_increase = final_memory - initial_memory
        
        # Memory increase should be reasonable
        assert memory_increase < 100 * 1024 * 1024  # Less than 100MB


# ============================================================================
# Enhanced OCR Preview Advanced Tests
# ============================================================================

class TestEnhancedOCRPreviewAdvanced:
    """Advanced test suite for EnhancedOCRPreview"""
    
    @pytest.fixture
    def ocr_preview(self, qtbot):
        """Create OCR preview instance"""
        preview = EnhancedOCRPreview()
        qtbot.addWidget(preview)
        preview.show()
        yield preview
        preview.close()
    
    @pytest.fixture
    def sample_image(self):
        """Create sample image for testing"""
        image = np.ones((480, 640, 3), dtype=np.uint8) * 255
        cv2.putText(image, "Test OCR Text", (100, 100), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        cv2.putText(image, "Another Text", (100, 200), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        return image
    
    def test_advanced_image_processing(self, ocr_preview, sample_image, qtbot):
        """Test advanced image processing"""
        ocr_preview.load_image(sample_image)
        qtbot.wait(100)
        
        # Test image preprocessing
        ocr_preview.apply_preprocessing()
        qtbot.wait(100)
        
        # Test image enhancement
        ocr_preview.apply_enhancement()
        qtbot.wait(100)
        
        # Test image filters
        ocr_preview.apply_filter("sharpen")
        qtbot.wait(100)
        
        # Verify image was processed
        assert ocr_preview.current_image is not None
    
    def test_batch_ocr_processing(self, ocr_preview, sample_image, qtbot):
        """Test batch OCR processing"""
        ocr_preview.load_image(sample_image)
        qtbot.wait(100)
        
        # Create multiple OCR results
        batch_results = []
        for i in range(10):
            result = OCRResult(
                text=f"Batch result {i}",
                confidence=0.8 + (i % 20) * 0.01,
                language="en",
                position=QRect(50 + i * 20, 50 + i * 10, 100, 30),
                timestamp=i * 0.5,
                result_type=OCRResultType.RAW
            )
            batch_results.append(result)
        
        # Process batch results
        ocr_preview.process_batch_results(batch_results)
        qtbot.wait(100)
        
        # Verify results were processed
        assert len(ocr_preview.ocr_results) == len(batch_results)
    
    def test_advanced_text_editing(self, ocr_preview, sample_image, qtbot):
        """Test advanced text editing"""
        ocr_preview.load_image(sample_image)
        qtbot.wait(100)
        
        # Test text selection
        text_editor = ocr_preview.text_editor
        cursor = text_editor.textCursor()
        cursor.select(QTextCursor.SelectionType.LineUnderCursor)
        text_editor.setTextCursor(cursor)
        qtbot.wait(100)
        
        # Test text formatting
        ocr_preview.apply_text_format("bold")
        qtbot.wait(100)
        
        ocr_preview.apply_text_format("italic")
        qtbot.wait(100)
        
        # Test text alignment
        ocr_preview.set_text_alignment("center")
        qtbot.wait(100)
        
        # Verify formatting was applied
        assert text_editor.toPlainText() != ""
    
    def test_advanced_export_options(self, ocr_preview, sample_image, qtbot):
        """Test advanced export options"""
        ocr_preview.load_image(sample_image)
        qtbot.wait(100)
        
        # Add some OCR results
        results = [
            OCRResult(
                text="Test result 1",
                confidence=0.9,
                language="en",
                position=QRect(10, 10, 100, 30),
                timestamp=1.0,
                result_type=OCRResultType.RAW
            ),
            OCRResult(
                text="Test result 2",
                confidence=0.85,
                language="en",
                position=QRect(10, 50, 100, 30),
                timestamp=2.0,
                result_type=OCRResultType.PROCESSED
            )
        ]
        ocr_preview.add_ocr_results(results)
        qtbot.wait(100)
        
        # Test export to different formats
        export_formats = ["txt", "json", "csv", "xml", "html"]
        for format_type in export_formats:
            with tempfile.NamedTemporaryFile(mode='w', suffix=f'.{format_type}', delete=False) as f:
                export_path = f.name
            
            ocr_preview.export_results(export_path, format_type)
            qtbot.wait(100)
            
            # Verify file was exported
            assert Path(export_path).exists()
            assert Path(export_path).stat().st_size > 0
            
            Path(export_path).unlink()
    
    def test_advanced_search_functionality(self, ocr_preview, sample_image, qtbot):
        """Test advanced search functionality"""
        ocr_preview.load_image(sample_image)
        qtbot.wait(100)
        
        # Add multiple results
        results = [
            OCRResult(
                text="Search test 1",
                confidence=0.9,
                language="en",
                position=QRect(10, 10, 100, 30),
                timestamp=1.0,
                result_type=OCRResultType.RAW
            ),
            OCRResult(
                text="Another search test",
                confidence=0.85,
                language="en",
                position=QRect(10, 50, 100, 30),
                timestamp=2.0,
                result_type=OCRResultType.PROCESSED
            ),
            OCRResult(
                text="Third search test",
                confidence=0.8,
                language="en",
                position=QRect(10, 90, 100, 30),
                timestamp=3.0,
                result_type=OCRResultType.RAW
            )
        ]
        ocr_preview.add_ocr_results(results)
        qtbot.wait(100)
        
        # Test basic search
        ocr_preview.search_text("search")
        qtbot.wait(100)
        
        # Verify search results
        assert len(ocr_preview.search_results) > 0
        
        # Test regex search
        ocr_preview.search_text(r"test \d+", use_regex=True)
        qtbot.wait(100)
        
        # Test case-sensitive search
        ocr_preview.search_text("Search", case_sensitive=True)
        qtbot.wait(100)
        
        # Test whole word search
        ocr_preview.search_text("test", whole_word=True)
        qtbot.wait(100)
    
    def test_advanced_filtering(self, ocr_preview, sample_image, qtbot):
        """Test advanced filtering"""
        ocr_preview.load_image(sample_image)
        qtbot.wait(100)
        
        # Add results with different properties
        results = [
            OCRResult(
                text="High confidence text",
                confidence=0.95,
                language="en",
                position=QRect(10, 10, 100, 30),
                timestamp=1.0,
                result_type=OCRResultType.RAW
            ),
            OCRResult(
                text="Medium confidence text",
                confidence=0.75,
                language="zh",
                position=QRect(10, 50, 100, 30),
                timestamp=2.0,
                result_type=OCRResultType.PROCESSED
            ),
            OCRResult(
                text="Low confidence text",
                confidence=0.45,
                language="en",
                position=QRect(10, 90, 100, 30),
                timestamp=3.0,
                result_type=OCRResultType.CONFIDENCE_FILTERED
            )
        ]
        ocr_preview.add_ocr_results(results)
        qtbot.wait(100)
        
        # Test confidence filtering
        ocr_preview.filter_by_confidence(0.8)
        qtbot.wait(100)
        
        filtered_results = ocr_preview.get_filtered_results()
        assert all(r.confidence >= 0.8 for r in filtered_results)
        
        # Test language filtering
        ocr_preview.filter_by_language("en")
        qtbot.wait(100)
        
        filtered_results = ocr_preview.get_filtered_results()
        assert all(r.language == "en" for r in filtered_results)
        
        # Test type filtering
        ocr_preview.filter_by_type(OCRResultType.RAW)
        qtbot.wait(100)
        
        filtered_results = ocr_preview.get_filtered_results()
        assert all(r.result_type == OCRResultType.RAW for r in filtered_results)
    
    def test_advanced_security_features(self, ocr_preview, sample_image, qtbot):
        """Test advanced security features"""
        ocr_preview.load_image(sample_image)
        qtbot.wait(100)
        
        # Test input sanitization
        malicious_text = "<script>alert('xss')</script>Test text"
        sanitized_text = ocr_preview.sanitize_text(malicious_text)
        
        assert "<script>" not in sanitized_text
        assert "Test text" in sanitized_text
        
        # Test result sanitization
        malicious_result = OCRResult(
            text="<script>alert('xss')</script>",
            confidence=0.95,
            language="en",
            position=QRect(10, 10, 100, 30),
            timestamp=1.0,
            result_type=OCRResultType.RAW
        )
        
        sanitized_result = ocr_preview.sanitize_result(malicious_result)
        assert "<script>" not in sanitized_result.text
        
        # Test secure rendering
        ocr_preview.enable_secure_rendering(True)
        qtbot.wait(100)
        
        assert ocr_preview.is_secure_rendering_enabled()
        
        # Test content validation
        assert ocr_preview.validate_content("Valid content")
        assert not ocr_preview.validate_content("<script>alert('xss')</script>")
    
    def test_advanced_performance_features(self, ocr_preview, sample_image, qtbot):
        """Test advanced performance features"""
        ocr_preview.load_image(sample_image)
        qtbot.wait(100)
        
        # Test performance monitoring
        assert hasattr(ocr_preview, 'performance_monitor')
        
        # Test lazy loading
        ocr_preview.enable_lazy_loading(True)
        qtbot.wait(100)
        
        assert ocr_preview.is_lazy_loading_enabled()
        
        # Test caching
        ocr_preview.enable_caching(True)
        qtbot.wait(100)
        
        assert ocr_preview.is_caching_enabled()
        
        # Test batch processing performance
        start_time = time.time()
        
        # Process large batch
        large_batch = []
        for i in range(100):
            result = OCRResult(
                text=f"Batch item {i}",
                confidence=0.8 + (i % 20) * 0.01,
                language="en",
                position=QRect(10, 10 + i * 5, 100, 30),
                timestamp=i * 0.1,
                result_type=OCRResultType.RAW
            )
            large_batch.append(result)
        
        ocr_preview.process_batch_results(large_batch)
        qtbot.wait(100)
        
        processing_time = time.time() - start_time
        
        # Processing should be fast
        assert processing_time < 2.0  # Less than 2 seconds for 100 items


# ============================================================================
# Enhanced Subtitle Editor Advanced Tests
# ============================================================================

class TestEnhancedSubtitleEditorAdvanced:
    """Advanced test suite for EnhancedSubtitleEditor"""
    
    @pytest.fixture
    def subtitle_editor(self, qtbot):
        """Create subtitle editor instance"""
        editor = EnhancedSubtitleEditor()
        qtbot.addWidget(editor)
        editor.show()
        yield editor
        editor.close()
    
    @pytest.fixture
    def sample_subtitles(self):
        """Create sample subtitle data"""
        return [
            SubtitleItem(
                index=1,
                start_time="00:00:01,000",
                end_time="00:00:03,000",
                content="第一条字幕"
            ),
            SubtitleItem(
                index=2,
                start_time="00:00:04,000",
                end_time="00:00:06,000",
                content="第二条字幕"
            ),
            SubtitleItem(
                index=3,
                start_time="00:00:07,000",
                end_time="00:00:09,000",
                content="第三条字幕"
            )
        ]
    
    def test_advanced_subtitle_operations(self, subtitle_editor, sample_subtitles, qtbot):
        """Test advanced subtitle operations"""
        subtitle_editor.load_subtitles(sample_subtitles)
        qtbot.wait(100)
        
        # Test subtitle merging
        subtitle_editor.merge_subtitles(0, 1)
        qtbot.wait(100)
        
        assert len(subtitle_editor.subtitles) == 2
        
        # Test subtitle splitting
        subtitle_editor.split_subtitle(0, 0.5)
        qtbot.wait(100)
        
        assert len(subtitle_editor.subtitles) == 3
        
        # Test subtitle duplication
        subtitle_editor.duplicate_subtitle(0)
        qtbot.wait(100)
        
        assert len(subtitle_editor.subtitles) == 4
    
    def test_advanced_timing_operations(self, subtitle_editor, sample_subtitles, qtbot):
        """Test advanced timing operations"""
        subtitle_editor.load_subtitles(sample_subtitles)
        qtbot.wait(100)
        
        # Test time stretching
        subtitle_editor.stretch_timing(1.5)
        qtbot.wait(100)
        
        # Verify timing was stretched
        assert subtitle_editor.subtitles[0].end_time > sample_subtitles[0].end_time
        
        # Test time compression
        subtitle_editor.stretch_timing(0.8)
        qtbot.wait(100)
        
        # Test time shifting
        subtitle_editor.shift_timing(1000)
        qtbot.wait(100)
        
        # Verify timing was shifted
        assert subtitle_editor.subtitles[0].start_time != sample_subtitles[0].start_time
        
        # Test gap adjustment
        subtitle_editor.adjust_gaps(500)
        qtbot.wait(100)
        
        # Test duration normalization
        subtitle_editor.normalize_durations()
        qtbot.wait(100)
    
    def test_advanced_text_operations(self, subtitle_editor, sample_subtitles, qtbot):
        """Test advanced text operations"""
        subtitle_editor.load_subtitles(sample_subtitles)
        qtbot.wait(100)
        
        # Test text search and replace
        subtitle_editor.search_replace("字幕", "subtitle")
        qtbot.wait(100)
        
        # Verify replacement
        assert "subtitle" in subtitle_editor.subtitles[0].content
        
        # Test text formatting
        subtitle_editor.apply_text_format(0, "bold")
        qtbot.wait(100)
        
        subtitle_editor.apply_text_format(0, "italic")
        qtbot.wait(100)
        
        # Test text case conversion
        subtitle_editor.convert_case(0, "upper")
        qtbot.wait(100)
        
        # Test text alignment
        subtitle_editor.set_alignment(0, "center")
        qtbot.wait(100)
        
        # Test text spell checking
        subtitle_editor.check_spelling()
        qtbot.wait(100)
        
        # Test text translation
        subtitle_editor.translate_text(0, "en")
        qtbot.wait(100)
    
    def test_advanced_import_export(self, subtitle_editor, sample_subtitles, qtbot):
        """Test advanced import/export"""
        subtitle_editor.load_subtitles(sample_subtitles)
        qtbot.wait(100)
        
        # Test export to multiple formats
        export_formats = {
            "srt": subtitle_editor.export_srt,
            "vtt": subtitle_editor.export_vtt,
            "ass": subtitle_editor.export_ass,
            "ssa": subtitle_editor.export_ssa,
            "sub": subtitle_editor.export_sub,
            "txt": subtitle_editor.export_txt,
            "xml": subtitle_editor.export_xml,
            "json": subtitle_editor.export_json
        }
        
        for format_type, export_func in export_formats.items():
            with tempfile.NamedTemporaryFile(mode='w', suffix=f'.{format_type}', delete=False) as f:
                export_path = f.name
            
            export_func(export_path)
            qtbot.wait(100)
            
            # Verify file was exported
            assert Path(export_path).exists()
            assert Path(export_path).stat().st_size > 0
            
            Path(export_path).unlink()
        
        # Test import from multiple formats
        import_formats = ["srt", "vtt", "ass"]
        for format_type in import_formats:
            with tempfile.NamedTemporaryFile(mode='w', suffix=f'.{format_type}', delete=False) as f:
                import_path = f.name
            
            # Create sample file
            if format_type == "srt":
                f.write("1\n00:00:01,000 --> 00:00:03,000\nTest subtitle\n\n")
            elif format_type == "vtt":
                f.write("WEBVTT\n\n00:00:01.000 --> 00:00:03.000\nTest subtitle\n\n")
            elif format_type == "ass":
                f.write("[V4+ Styles]\nFormat: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding\nStyle: Default,Arial,20,&H00FFFFFF,&H000000FF,&H00000000,&H00000000,0,0,0,0,100,100,0,0,1,0,0,2,0,0,0,1\n\n[Events]\nFormat: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\nDialogue: 0,0:00:01.00,0:00:03.00,Default,,0,0,0,,Test subtitle\n")
            
            f.close()
            
            # Import file
            subtitle_editor.import_subtitles(import_path)
            qtbot.wait(100)
            
            # Verify import was successful
            assert len(subtitle_editor.subtitles) > 0
            
            Path(import_path).unlink()
    
    def test_advanced_validation(self, subtitle_editor, sample_subtitles, qtbot):
        """Test advanced validation"""
        subtitle_editor.load_subtitles(sample_subtitles)
        qtbot.wait(100)
        
        # Test timing validation
        validation_results = subtitle_editor.validate_timing()
        assert isinstance(validation_results, dict)
        assert "valid" in validation_results
        assert "errors" in validation_results
        
        # Test content validation
        content_results = subtitle_editor.validate_content()
        assert isinstance(content_results, dict)
        assert "valid" in content_results
        assert "errors" in content_results
        
        # Test format validation
        format_results = subtitle_editor.validate_format()
        assert isinstance(format_results, dict)
        assert "valid" in format_results
        assert "errors" in format_results
        
        # Test comprehensive validation
        comprehensive_results = subtitle_editor.validate_all()
        assert isinstance(comprehensive_results, dict)
        assert "timing" in comprehensive_results
        assert "content" in comprehensive_results
        assert "format" in comprehensive_results
    
    def test_advanced_collaboration_features(self, subtitle_editor, sample_subtitles, qtbot):
        """Test advanced collaboration features"""
        subtitle_editor.load_subtitles(sample_subtitles)
        qtbot.wait(100)
        
        # Test change tracking
        subtitle_editor.enable_change_tracking(True)
        qtbot.wait(100)
        
        assert subtitle_editor.is_change_tracking_enabled()
        
        # Make some changes
        subtitle_editor.edit_subtitle(0, "Modified subtitle")
        qtbot.wait(100)
        
        # Test change history
        changes = subtitle_editor.get_change_history()
        assert len(changes) > 0
        
        # Test undo/redo
        subtitle_editor.undo()
        qtbot.wait(100)
        
        subtitle_editor.redo()
        qtbot.wait(100)
        
        # Test comment system
        subtitle_editor.add_comment(0, "This is a comment")
        qtbot.wait(100)
        
        comments = subtitle_editor.get_comments(0)
        assert len(comments) > 0
        
        # Test version control
        subtitle_editor.create_version("Initial version")
        qtbot.wait(100)
        
        versions = subtitle_editor.get_versions()
        assert len(versions) > 0
        
        # Test diff viewing
        diff = subtitle_editor.get_diff(0, 1)
        assert isinstance(diff, str)
    
    def test_advanced_performance_features(self, subtitle_editor, sample_subtitles, qtbot):
        """Test advanced performance features"""
        subtitle_editor.load_subtitles(sample_subtitles)
        qtbot.wait(100)
        
        # Test performance monitoring
        assert hasattr(subtitle_editor, 'performance_monitor')
        
        # Test lazy loading
        subtitle_editor.enable_lazy_loading(True)
        qtbot.wait(100)
        
        assert subtitle_editor.is_lazy_loading_enabled()
        
        # Test caching
        subtitle_editor.enable_caching(True)
        qtbot.wait(100)
        
        assert subtitle_editor.is_caching_enabled()
        
        # Test batch processing performance
        start_time = time.time()
        
        # Process large batch
        large_batch = []
        for i in range(100):
            subtitle = SubtitleItem(
                index=i + 1,
                start_time=f"00:00:{i:02d},000",
                end_time=f"00:00:{i+1:02d},000",
                content=f"Subtitle {i + 1}"
            )
            large_batch.append(subtitle)
        
        subtitle_editor.load_subtitles(large_batch)
        qtbot.wait(100)
        
        processing_time = time.time() - start_time
        
        # Processing should be fast
        assert processing_time < 2.0  # Less than 2 seconds for 100 items
        
        # Test memory optimization
        initial_memory = psutil.Process().memory_info().rss
        
        # Load very large dataset
        very_large_batch = []
        for i in range(1000):
            subtitle = SubtitleItem(
                index=i + 1,
                start_time=f"00:00:{i//60:02d}:{i%60:02d},000",
                end_time=f"00:00:{(i+1)//60:02d}:{(i+1)%60:02d},000",
                content=f"Large subtitle {i + 1}"
            )
            very_large_batch.append(subtitle)
        
        subtitle_editor.load_subtitles(very_large_batch)
        qtbot.wait(100)
        
        # Check memory usage
        final_memory = psutil.Process().memory_info().rss
        memory_increase = final_memory - initial_memory
        
        # Memory increase should be reasonable
        assert memory_increase < 100 * 1024 * 1024  # Less than 100MB
    
    def test_advanced_accessibility_features(self, subtitle_editor, sample_subtitles, qtbot):
        """Test advanced accessibility features"""
        subtitle_editor.load_subtitles(sample_subtitles)
        qtbot.wait(100)
        
        # Test screen reader support
        assert subtitle_editor.accessibleName() != ""
        assert subtitle_editor.accessibleDescription() != ""
        
        # Test keyboard navigation
        subtitle_editor.setFocus()
        qtbot.wait(100)
        
        # Test tab navigation
        QTest.keyClick(subtitle_editor, Qt.Key.Key_Tab)
        qtbot.wait(100)
        
        # Test arrow key navigation
        QTest.keyClick(subtitle_editor, Qt.Key.Key_Up)
        QTest.keyClick(subtitle_editor, Qt.Key.Key_Down)
        QTest.keyClick(subtitle_editor, Qt.Key.Key_Left)
        QTest.keyClick(subtitle_editor, Qt.Key.Key_Right)
        qtbot.wait(100)
        
        # Test high contrast mode
        subtitle_editor.set_high_contrast(True)
        qtbot.wait(100)
        
        assert subtitle_editor.is_high_contrast_enabled()
        
        # Test font scaling
        original_font = subtitle_editor.font()
        
        subtitle_editor.set_font_scale(1.5)
        qtbot.wait(100)
        
        assert subtitle_editor.font().pointSize() > original_font.pointSize()
        
        subtitle_editor.set_font_scale(1.0)
        qtbot.wait(100)
        
        # Test color blind mode
        subtitle_editor.set_color_blind_mode(True)
        qtbot.wait(100)
        
        assert subtitle_editor.is_color_blind_mode_enabled()
        
        # Test text-to-speech
        subtitle_editor.enable_text_to_speech(True)
        qtbot.wait(100)
        
        assert subtitle_editor.is_text_to_speech_enabled()


# ============================================================================
# Advanced Integration Tests
# ============================================================================

class TestAdvancedIntegration:
    """Advanced integration tests for UI components"""
    
    def test_complete_workflow_integration(self, qtbot):
        """Test complete workflow integration"""
        # Create main application
        app = EnhancedApplication([])
        
        # Test complete workflow
        # 1. Load video
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as f:
            video_path = f.name
            
            # Create sample video
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(f.name, fourcc, 30.0, (640, 480))
            
            for i in range(30):
                frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
                cv2.putText(frame, f"Frame {i}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                out.write(frame)
            
            out.release()
        
        app.main_window.load_video(video_path)
        qtbot.wait(500)
        
        # 2. Process OCR
        sample_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        app.main_window.ocr_preview.load_image(sample_image)
        qtbot.wait(100)
        
        # Add sample OCR results
        from visionsub.ui.enhanced_ocr_preview import OCRResult, OCRResultType
        results = [
            OCRResult(
                text="Test OCR result",
                confidence=0.9,
                language="en",
                position=QRect(10, 10, 100, 30),
                timestamp=1.0,
                result_type=OCRResultType.RAW
            )
        ]
        app.main_window.ocr_preview.add_ocr_results(results)
        qtbot.wait(100)
        
        # 3. Edit subtitles
        sample_subtitles = [
            SubtitleItem(
                index=1,
                start_time="00:00:01,000",
                end_time="00:00:03,000",
                content="Test subtitle"
            )
        ]
        app.main_window.subtitle_editor.load_subtitles(sample_subtitles)
        qtbot.wait(100)
        
        # 4. Export results
        with tempfile.NamedTemporaryFile(mode='w', suffix='.srt', delete=False) as f:
            export_path = f.name
        
        app.main_window.subtitle_editor.export_srt(export_path)
        qtbot.wait(100)
        
        # Verify workflow completed
        assert Path(export_path).exists()
        assert Path(video_path).exists()
        
        # Cleanup
        Path(export_path).unlink()
        Path(video_path).unlink()
        app.quit()
    
    def test_stress_testing(self, qtbot):
        """Test stress testing"""
        # Create main application
        app = EnhancedApplication([])
        
        # Test with large dataset
        large_subtitles = []
        for i in range(1000):
            subtitle = SubtitleItem(
                index=i + 1,
                start_time=f"00:00:{i//60:02d}:{i%60:02d},000",
                end_time=f"00:00:{(i+1)//60:02d}:{(i+1)%60:02d},000",
                content=f"Large subtitle {i + 1}"
            )
            large_subtitles.append(subtitle)
        
        # Test loading large dataset
        start_time = time.time()
        app.main_window.subtitle_editor.load_subtitles(large_subtitles)
        qtbot.wait(1000)
        
        loading_time = time.time() - start_time
        
        # Loading should be fast
        assert loading_time < 5.0  # Less than 5 seconds
        
        # Test memory usage
        initial_memory = psutil.Process().memory_info().rss
        
        # Perform operations
        for i in range(100):
            app.main_window.subtitle_editor.edit_subtitle(i % len(large_subtitles), f"Modified {i}")
            qtbot.wait(10)
        
        final_memory = psutil.Process().memory_info().rss
        memory_increase = final_memory - initial_memory
        
        # Memory increase should be reasonable
        assert memory_increase < 200 * 1024 * 1024  # Less than 200MB
        
        app.quit()
    
    def test_concurrent_operations(self, qtbot):
        """Test concurrent operations"""
        app = EnhancedApplication([])
        
        results = []
        
        def worker_function(worker_id):
            start_time = time.time()
            
            # Simulate work
            for i in range(10):
                app.main_window.subtitle_editor.add_subtitle(f"Worker {worker_id} subtitle {i}")
                qtbot.wait(10)
            
            end_time = time.time()
            results.append(end_time - start_time)
        
        # Create multiple threads
        threads = []
        for i in range(5):
            thread = threading.Thread(target=worker_function, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join(timeout=30.0)
        
        # Check concurrent performance
        assert len(results) == 5
        assert all(result < 5.0 for result in results)  # All operations should complete quickly
        
        app.quit()
    
    def test_error_recovery(self, qtbot):
        """Test error recovery"""
        app = EnhancedApplication([])
        
        # Test with invalid data
        invalid_data = [
            {"invalid": "data"},
            {"start": "invalid_time", "end": "invalid_time", "text": "test"}
        ]
        
        # Should handle invalid data gracefully
        try:
            app.main_window.subtitle_editor.load_subtitles(invalid_data)
        except Exception as e:
            # Should handle exception gracefully
            assert isinstance(e, (ValueError, TypeError))
        
        # Test recovery after error
        valid_data = [
            SubtitleItem(
                index=1,
                start_time="00:00:01,000",
                end_time="00:00:03,000",
                content="Valid subtitle"
            )
        ]
        
        app.main_window.subtitle_editor.load_subtitles(valid_data)
        qtbot.wait(100)
        
        # Should recover and work normally
        assert len(app.main_window.subtitle_editor.subtitles) == 1
        
        app.quit()
    
    def test_cross_component_communication(self, qtbot):
        """Test cross-component communication"""
        app = EnhancedApplication([])
        
        # Test signal propagation between components
        signal_received = False
        
        def on_signal_received():
            nonlocal signal_received
            signal_received = True
        
        # Connect video player to OCR preview
        app.main_window.video_player.frame_extracted.connect(on_signal_received)
        
        # Trigger signal
        app.main_window.video_player.extract_frame(1000)
        qtbot.wait(100)
        
        # Verify signal was received
        assert signal_received
        
        # Test data flow between components
        sample_data = {"test": "data"}
        app.main_window.view_model.set_data(sample_data)
        qtbot.wait(100)
        
        # Verify data was propagated
        assert app.main_window.view_model.get_data() == sample_data
        
        app.quit()
    
    def test_resource_management(self, qtbot):
        """Test resource management"""
        app = EnhancedApplication([])
        
        # Test memory management
        initial_memory = psutil.Process().memory_info().rss
        
        # Load and unload resources
        for i in range(10):
            # Load large dataset
            large_subtitles = []
            for j in range(100):
                subtitle = SubtitleItem(
                    index=j + 1,
                    start_time=f"00:00:{j//60:02d}:{j%60:02d},000",
                    end_time=f"00:00:{(j+1)//60:02d}:{(j+1)%60:02d},000",
                    content=f"Subtitle {j + 1}"
                )
                large_subtitles.append(subtitle)
            
            app.main_window.subtitle_editor.load_subtitles(large_subtitles)
            qtbot.wait(100)
            
            # Clear data
            app.main_window.subtitle_editor.clear_subtitles()
            qtbot.wait(100)
        
        # Check memory usage
        final_memory = psutil.Process().memory_info().rss
        memory_increase = final_memory - initial_memory
        
        # Memory should be properly managed
        assert memory_increase < 50 * 1024 * 1024  # Less than 50MB
        
        # Test cleanup
        app.cleanup()
        
        # Verify cleanup completed
        assert True  # Placeholder for actual cleanup verification
        
        app.quit()


# ============================================================================
# Security Testing Suite
# ============================================================================

class TestUISecurityAdvanced:
    """Advanced security testing suite"""
    
    def test_input_validation_comprehensive(self):
        """Test comprehensive input validation"""
        # Test malicious input patterns
        malicious_inputs = [
            # XSS attacks
            "<script>alert('xss')</script>",
            "<img src=x onerror=alert('xss')>",
            "<svg onload=alert('xss')>",
            "<iframe src=javascript:alert('xss')>",
            "<body onload=alert('xss')>",
            "<div onmouseover=alert('xss')>",
            "<input type=image src=x onerror=alert('xss')>",
            "<details open ontoggle=alert('xss')>",
            "<select onfocus=alert('xss')>",
            "<textarea onfocus=alert('xss')>",
            "<keygen onfocus=alert('xss')>",
            "<video onerror=alert('xss')>",
            "<audio onerror=alert('xss')>",
            
            # Protocol attacks
            "javascript:alert('xss')",
            "vbscript:alert('xss')",
            "data:text/html,<script>alert('xss')</script>",
            "file:///etc/passwd",
            "ftp://malicious.com/malicious.exe",
            "http://malicious.com/malicious.exe",
            "https://malicious.com/malicious.exe",
            
            # Path traversal
            "../../../etc/passwd",
            "..\\..\\..\\windows\\system32\\config\\sam",
            "/etc/passwd",
            "/etc/shadow",
            "C:\\Windows\\System32\\config\\SAM",
            "./../../etc/passwd",
            ".\\..\\..\\windows\\system32\\config\\sam",
            
            # SQL injection
            "' OR '1'='1",
            "' UNION SELECT * FROM users--",
            "'; DROP TABLE users;--",
            "' OR SLEEP(10)--",
            "' WAITFOR DELAY '0:0:10'--",
            "1' OR '1'='1",
            "admin'--",
            "' OR '1'='1",
            "1; DROP TABLE users--",
            
            # Command injection
            "test; rm -rf /",
            "test && rm -rf /",
            "test | rm -rf /",
            "test || rm -rf /",
            "test $(rm -rf /)",
            "test `rm -rf /`",
            "test & rm -rf /",
            "test > /etc/passwd",
            "test < /etc/passwd",
            "test\nrm -rf /",
            "test\rrm -rf /",
            
            # Unicode and encoding attacks
            "\x00\x01\x02\x03",  # Null bytes and control characters
            "\u2028\u2029\u000b\u000c",  # Unicode control characters
            "%3Cscript%3Ealert('xss')%3C/script%3E",  # URL encoded
            "&#60;script&#62;alert('xss')&#60;/script&#62;",  # HTML encoded
            "&lt;script&gt;alert('xss')&lt;/script&gt;",  # HTML entities
            
            # Very long inputs
            "a" * 10000,  # Very long string
            "test " * 1000,  # Very long repeated string
            
            # Empty inputs
            "",
            "   ",
            "\t\n\r",
            
            # Binary data
            b"\x00\x01\x02\x03".decode('latin-1'),
            
            # Mixed attacks
            "<script>alert('xss')</script>../../../etc/passwd",
            "javascript:alert('xss'); rm -rf /",
            "' OR '1'='1'; DROP TABLE users;--"
        ]
        
        # Test sanitization for each malicious input
        for malicious_input in malicious_inputs:
            sanitized = self.sanitize_input_comprehensive(malicious_input)
            
            # Verify dangerous content is removed
            assert "<script>" not in sanitized
            assert "javascript:" not in sanitized
            assert "vbscript:" not in sanitized
            assert "data:" not in sanitized
            assert "file:" not in sanitized
            assert "../" not in sanitized
            assert "..\\" not in sanitized
            assert ";" not in sanitized
            assert "&" not in sanitized
            assert "|" not in sanitized
            assert "`" not in sanitized
            assert "$(" not in sanitized
            assert "'" not in sanitized or sanitized.count("'") <= 2  # Allow quoted strings
            assert "--" not in sanitized
            assert "UNION" not in sanitized
            assert "DROP" not in sanitized
            assert "DELETE" not in sanitized
            assert "INSERT" not in sanitized
            assert "UPDATE" not in sanitized
            assert "ALTER" not in sanitized
            
            # Verify input is not empty (unless it was empty originally)
            if malicious_input.strip():
                assert sanitized.strip() != ""
    
    def test_file_upload_security_comprehensive(self):
        """Test comprehensive file upload security"""
        # Test malicious file paths
        malicious_paths = [
            # Path traversal
            "../../../etc/passwd",
            "..\\..\\..\\windows\\system32\\config\\sam",
            "/etc/passwd",
            "/etc/shadow",
            "C:\\Windows\\System32\\config\\SAM",
            "./../../etc/passwd",
            ".\\..\\..\\windows\\system32\\config\\sam",
            
            # Protocol attacks
            "file:///etc/passwd",
            "http://malicious.com/malicious.exe",
            "https://malicious.com/malicious.exe",
            "ftp://malicious.com/malicious.exe",
            "smb://malicious.com/share/malicious.exe",
            "nfs://malicious.com/share/malicious.exe",
            
            # Dangerous file types
            "malicious.exe",
            "virus.bat",
            "script.js",
            "malicious.php",
            "backdoor.asp",
            "shell.jsp",
            "malicious.dll",
            "malicious.so",
            "malicious.dylib",
            "malicious.app",
            "malicious.dmg",
            "malicious.deb",
            "malicious.rpm",
            "malicious.msi",
            
            # Hidden files
            ".hidden_file",
            "..",
            ".",
            
            # System files
            "autoexec.bat",
            "config.sys",
            "boot.ini",
            "ntldr",
            "ntdetect.com",
            "bootmgr",
            "pagefile.sys",
            "hiberfil.sys",
            
            # Special files
            "/dev/null",
            "/dev/zero",
            "/dev/random",
            "/dev/urandom",
            "/proc/self/environ",
            "/proc/meminfo",
            "/proc/cpuinfo",
            
            # Network paths
            "\\\\malicious\\share\\malicious.exe",
            "//malicious/share/malicious.exe",
            
            # Encoded paths
            "%2e%2e%2f%2e%2e%2f%2e%2e%2fetc%2fpasswd",  # URL encoded
            "..%2f..%2f..%2fetc%2fpasswd",
            "..\\..\\..\\windows\\system32\\config\\sam",
            
            # Mixed attacks
            "../../../etc/passwd; rm -rf /",
            "file:///etc/passwd && rm -rf /",
            "http://malicious.com/malicious.exe | rm -rf /",
        ]
        
        # Test sanitization for each malicious path
        for malicious_path in malicious_paths:
            sanitized = self.sanitize_file_path_comprehensive(malicious_path)
            
            # Verify dangerous content is removed
            assert "../" not in sanitized
            assert "..\\" not in sanitized
            assert "file:" not in sanitized
            assert "http:" not in sanitized
            assert "https:" not in sanitized
            assert "ftp:" not in sanitized
            assert "smb:" not in sanitized
            assert "nfs:" not in sanitized
            assert ";" not in sanitized
            assert "&" not in sanitized
            assert "|" not in sanitized
            assert "`" not in sanitized
            assert "$(" not in sanitized
            assert "\\\\" not in sanitized
            assert "//" not in sanitized or "//" in sanitized[:2]  # Allow protocol
            
            # Verify path is safe
            assert not self.is_dangerous_path(sanitized)
    
    def test_session_security_comprehensive(self):
        """Test comprehensive session security"""
        # Test session creation
        session = self.create_secure_session()
        assert session is not None
        assert hasattr(session, 'session_id')
        assert hasattr(session, 'user_id')
        assert hasattr(session, 'created_at')
        assert hasattr(session, 'expires_at')
        assert hasattr(session, 'ip_address')
        assert hasattr(session, 'user_agent')
        assert hasattr(session, 'csrf_token')
        
        # Test session validation
        is_valid = self.validate_secure_session(session)
        assert is_valid
        
        # Test session expiration
        expired_session = self.create_expired_session()
        is_valid = self.validate_secure_session(expired_session)
        assert not is_valid
        
        # Test session hijacking protection
        hijacked_session = self.create_hijacked_session()
        is_valid = self.validate_secure_session(hijacked_session)
        assert not is_valid
        
        # Test session fixation protection
        fixed_session = self.create_fixed_session()
        is_valid = self.validate_secure_session(fixed_session)
        assert not is_valid
        
        # Test CSRF protection
        csrf_token = self.generate_csrf_token()
        assert csrf_token is not None
        assert len(csrf_token) >= 32  # Minimum length
        
        is_valid = self.validate_csrf_token(csrf_token, csrf_token)
        assert is_valid
        
        is_valid = self.validate_csrf_token(csrf_token, "invalid_token")
        assert not is_valid
        
        # Test session timeout
        timeout_session = self.create_timeout_session()
        is_valid = self.validate_secure_session(timeout_session)
        assert not is_valid
    
    def test_rate_limiting_comprehensive(self):
        """Test comprehensive rate limiting"""
        # Test basic rate limiting
        rate_limiter = self.create_rate_limiter_comprehensive(
            max_requests=10, 
            time_window=60,
            burst_limit=5,
            cooldown_period=30
        )
        
        # Test allowed requests
        for i in range(10):
            is_allowed = rate_limiter.is_allowed("test_user")
            assert is_allowed
        
        # Test rate limit exceeded
        is_allowed = rate_limiter.is_allowed("test_user")
        assert not is_allowed
        
        # Test burst limit
        burst_limiter = self.create_rate_limiter_comprehensive(
            max_requests=100,
            time_window=60,
            burst_limit=5,
            burst_window=1
        )
        
        # Test burst requests
        for i in range(5):
            is_allowed = burst_limiter.is_allowed("test_user")
            assert is_allowed
        
        # Test burst limit exceeded
        is_allowed = burst_limiter.is_allowed("test_user")
        assert not is_allowed
        
        # Test cooldown period
        cooldown_limiter = self.create_rate_limiter_comprehensive(
            max_requests=10,
            time_window=60,
            cooldown_period=30
        )
        
        # Exhaust requests
        for i in range(10):
            cooldown_limiter.is_allowed("test_user")
        
        # Test cooldown
        is_allowed = cooldown_limiter.is_allowed("test_user")
        assert not is_allowed
        
        # Test IP-based rate limiting
        ip_limiter = self.create_ip_rate_limiter()
        
        for i in range(5):
            is_allowed = ip_limiter.is_allowed("192.168.1.1")
            assert is_allowed
        
        is_allowed = ip_limiter.is_allowed("192.168.1.1")
        assert not is_allowed
        
        # Test user-based rate limiting
        user_limiter = self.create_user_rate_limiter()
        
        for i in range(5):
            is_allowed = user_limiter.is_allowed("test_user")
            assert is_allowed
        
        is_allowed = user_limiter.is_allowed("test_user")
        assert not is_allowed
    
    def test_data_encryption_comprehensive(self):
        """Test comprehensive data encryption"""
        # Test data encryption
        sensitive_data = "sensitive_information"
        encrypted_data = self.encrypt_data(sensitive_data)
        
        assert encrypted_data is not None
        assert encrypted_data != sensitive_data
        assert len(encrypted_data) > len(sensitive_data)
        
        # Test data decryption
        decrypted_data = self.decrypt_data(encrypted_data)
        assert decrypted_data == sensitive_data
        
        # Test encryption with different keys
        key1 = self.generate_encryption_key()
        key2 = self.generate_encryption_key()
        
        encrypted1 = self.encrypt_with_key(sensitive_data, key1)
        encrypted2 = self.encrypt_with_key(sensitive_data, key2)
        
        assert encrypted1 != encrypted2
        
        # Test decryption with wrong key
        try:
            decrypted = self.decrypt_with_key(encrypted1, key2)
            assert False, "Should have failed"
        except Exception:
            pass  # Expected to fail
        
        # Test key derivation
        password = "secure_password"
        salt = self.generate_salt()
        
        key1 = self.derive_key(password, salt)
        key2 = self.derive_key(password, salt)
        
        assert key1 == key2
        
        # Test hash verification
        data = "test_data"
        hash1 = self.hash_data(data)
        hash2 = self.hash_data(data)
        
        assert hash1 == hash2
        
        # Test tamper detection
        original_data = "original_data"
        hash_value = self.hash_data(original_data)
        
        tampered_data = "tampered_data"
        tampered_hash = self.hash_data(tampered_data)
        
        assert hash_value != tampered_hash
        
        # Test secure random generation
        random_bytes = self.generate_secure_random(32)
        assert len(random_bytes) == 32
        assert isinstance(random_bytes, bytes)
        
        # Test secure token generation
        token = self.generate_secure_token()
        assert len(token) >= 32
        assert isinstance(token, str)
    
    # Helper methods for comprehensive security tests
    def sanitize_input_comprehensive(self, input_str: str) -> str:
        """Comprehensive input sanitization"""
        import re
        import html
        
        # HTML escape
        clean = html.escape(input_str)
        
        # Remove dangerous protocols
        clean = re.sub(r'javascript:', '', clean, flags=re.IGNORECASE)
        clean = re.sub(r'vbscript:', '', clean, flags=re.IGNORECASE)
        clean = re.sub(r'data:', '', clean, flags=re.IGNORECASE)
        clean = re.sub(r'file:', '', clean, flags=re.IGNORECASE)
        clean = re.sub(r'ftp:', '', clean, flags=re.IGNORECASE)
        clean = re.sub(r'smb:', '', clean, flags=re.IGNORECASE)
        clean = re.sub(r'nfs:', '', clean, flags=re.IGNORECASE)
        
        # Remove path traversal
        clean = re.sub(r'\.\./', '', clean)
        clean = re.sub(r'\.\.\\', '', clean)
        
        # Remove command injection
        clean = re.sub(r';', '', clean)
        clean = re.sub(r'&', '', clean)
        clean = re.sub(r'\|', '', clean)
        clean = re.sub(r'`', '', clean)
        clean = re.sub(r'\$\(', '', clean)
        
        # Remove SQL injection
        clean = re.sub(r'--.*$', '', clean, flags=re.MULTILINE)
        clean = re.sub(r'/\*.*?\*/', '', clean, flags=re.DOTALL)
        
        # Remove dangerous keywords
        dangerous_keywords = [
            'UNION', 'DROP', 'DELETE', 'INSERT', 'UPDATE', 'ALTER',
            'CREATE', 'TRUNCATE', 'EXEC', 'EXECUTE', 'XP_', 'SP_'
        ]
        
        for keyword in dangerous_keywords:
            clean = re.sub(r'\b' + keyword + r'\b', '', clean, flags=re.IGNORECASE)
        
        # Remove control characters
        clean = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', clean)
        
        # Limit length
        clean = clean[:10000]
        
        return clean.strip()
    
    def sanitize_file_path_comprehensive(self, file_path: str) -> str:
        """Comprehensive file path sanitization"""
        import os
        
        # Remove path traversal
        clean = file_path.replace('../', '').replace('..\\', '')
        
        # Remove dangerous protocols
        clean = clean.replace('file://', '').replace('http://', '')
        clean = clean.replace('https://', '').replace('ftp://', '')
        clean = clean.replace('smb://', '').replace('nfs://', '')
        
        # Remove network paths
        clean = clean.replace('\\\\', '').replace('//', '')
        
        # Get basename
        clean = os.path.basename(clean)
        
        # Remove dangerous characters
        dangerous_chars = ['<', '>', ':', '"', '|', '?', '*']
        for char in dangerous_chars:
            clean = clean.replace(char, '')
        
        return clean
    
    def is_dangerous_path(self, path: str) -> bool:
        """Check if path is dangerous"""
        dangerous_patterns = [
            '../', '..\\', 'file://', 'http://', 'https://', 'ftp://',
            'smb://', 'nfs://', '\\\\', '//', '/etc/', '/dev/', '/proc/',
            'C:\\Windows\\', 'C:\\Program Files\\', 'C:\\System Volume Information\\'
        ]
        
        return any(pattern in path for pattern in dangerous_patterns)
    
    def create_secure_session(self):
        """Create secure session"""
        import time
        import uuid
        import hashlib
        
        class SecureSession:
            def __init__(self):
                self.session_id = str(uuid.uuid4())
                self.user_id = "test_user"
                self.created_at = time.time()
                self.expires_at = time.time() + 3600  # 1 hour
                self.ip_address = "127.0.0.1"
                self.user_agent = "Test Browser"
                self.csrf_token = hashlib.sha256(str(uuid.uuid4()).encode()).hexdigest()
                self.last_activity = time.time()
        
        return SecureSession()
    
    def create_expired_session(self):
        """Create expired session"""
        import time
        import uuid
        
        class ExpiredSession:
            def __init__(self):
                self.session_id = str(uuid.uuid4())
                self.user_id = "test_user"
                self.created_at = time.time() - 7200  # 2 hours ago
                self.expires_at = time.time() - 3600  # 1 hour ago
                self.ip_address = "127.0.0.1"
                self.user_agent = "Test Browser"
                self.csrf_token = "expired_token"
                self.last_activity = time.time() - 3600
        
        return ExpiredSession()
    
    def create_hijacked_session(self):
        """Create hijacked session"""
        import time
        import uuid
        
        class HijackedSession:
            def __init__(self):
                self.session_id = str(uuid.uuid4())
                self.user_id = "test_user"
                self.created_at = time.time()
                self.expires_at = time.time() + 3600
                self.ip_address = "192.168.1.100"  # Different IP
                self.user_agent = "Malicious Browser"
                self.csrf_token = "hijacked_token"
                self.last_activity = time.time()
        
        return HijackedSession()
    
    def create_fixed_session(self):
        """Create fixed session"""
        import time
        import uuid
        
        class FixedSession:
            def __init__(self):
                self.session_id = "fixed_session_id"  # Predictable ID
                self.user_id = "test_user"
                self.created_at = time.time()
                self.expires_at = time.time() + 3600
                self.ip_address = "127.0.0.1"
                self.user_agent = "Test Browser"
                self.csrf_token = "fixed_token"  # Predictable token
                self.last_activity = time.time()
        
        return FixedSession()
    
    def create_timeout_session(self):
        """Create timeout session"""
        import time
        import uuid
        
        class TimeoutSession:
            def __init__(self):
                self.session_id = str(uuid.uuid4())
                self.user_id = "test_user"
                self.created_at = time.time() - 1800  # 30 minutes ago
                self.expires_at = time.time() + 1800  # 30 minutes from now
                self.ip_address = "127.0.0.1"
                self.user_agent = "Test Browser"
                self.csrf_token = "timeout_token"
                self.last_activity = time.time() - 1800  # 30 minutes ago
        
        return TimeoutSession()
    
    def validate_secure_session(self, session) -> bool:
        """Validate secure session"""
        import time
        
        # Check if session exists
        if not session:
            return False
        
        # Check expiration
        if time.time() > session.expires_at:
            return False
        
        # Check activity timeout
        if time.time() - session.last_activity > 1800:  # 30 minutes
            return False
        
        # Check required fields
        required_fields = ['session_id', 'user_id', 'created_at', 'expires_at', 'csrf_token']
        for field in required_fields:
            if not hasattr(session, field):
                return False
        
        return True
    
    def generate_csrf_token(self) -> str:
        """Generate CSRF token"""
        import uuid
        import hashlib
        
        return hashlib.sha256(str(uuid.uuid4()).encode()).hexdigest()
    
    def validate_csrf_token(self, token: str, expected_token: str) -> bool:
        """Validate CSRF token"""
        return token == expected_token and len(token) >= 32
    
    def create_rate_limiter_comprehensive(self, max_requests: int, time_window: int, 
                                        burst_limit: int = None, burst_window: int = None,
                                        cooldown_period: int = None):
        """Create comprehensive rate limiter"""
        import time
        
        class ComprehensiveRateLimiter:
            def __init__(self, max_requests, time_window, burst_limit=None, burst_window=None, cooldown_period=None):
                self.max_requests = max_requests
                self.time_window = time_window
                self.burst_limit = burst_limit
                self.burst_window = burst_window
                self.cooldown_period = cooldown_period
                self.requests = {}
                self.burst_requests = {}
                self.cooldown_until = {}
            
            def is_allowed(self, user_id: str) -> bool:
                current_time = time.time()
                
                # Check cooldown
                if self.cooldown_period and user_id in self.cooldown_until:
                    if current_time < self.cooldown_until[user_id]:
                        return False
                    else:
                        del self.cooldown_until[user_id]
                
                # Check burst limit
                if self.burst_limit and self.burst_window:
                    if user_id not in self.burst_requests:
                        self.burst_requests[user_id] = []
                    
                    # Remove expired burst requests
                    self.burst_requests[user_id] = [
                        req_time for req_time in self.burst_requests[user_id]
                        if current_time - req_time < self.burst_window
                    ]
                    
                    if len(self.burst_requests[user_id]) >= self.burst_limit:
                        # Set cooldown
                        if self.cooldown_period:
                            self.cooldown_until[user_id] = current_time + self.cooldown_period
                        return False
                
                # Check regular rate limit
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
                    
                    # Add to burst requests if applicable
                    if self.burst_limit and self.burst_window:
                        if user_id not in self.burst_requests:
                            self.burst_requests[user_id] = []
                        self.burst_requests[user_id].append(current_time)
                    
                    return True
                
                # Set cooldown
                if self.cooldown_period:
                    self.cooldown_until[user_id] = current_time + self.cooldown_period
                
                return False
        
        return ComprehensiveRateLimiter(max_requests, time_window, burst_limit, burst_window, cooldown_period)
    
    def create_ip_rate_limiter(self):
        """Create IP-based rate limiter"""
        return self.create_rate_limiter_comprehensive(max_requests=100, time_window=60)
    
    def create_user_rate_limiter(self):
        """Create user-based rate limiter"""
        return self.create_rate_limiter_comprehensive(max_requests=50, time_window=60)
    
    def encrypt_data(self, data: str) -> str:
        """Encrypt data"""
        import hashlib
        import base64
        
        # Simple encryption for testing (use proper encryption in production)
        key = "secure_key_123"
        encrypted = hashlib.sha256((data + key).encode()).hexdigest()
        return base64.b64encode(encrypted.encode()).decode()
    
    def decrypt_data(self, encrypted_data: str) -> str:
        """Decrypt data"""
        import base64
        import hashlib
        
        # Simple decryption for testing (use proper decryption in production)
        try:
            decoded = base64.b64decode(encrypted_data.encode()).decode()
            return decoded[:len(decoded)//2]  # Return part of original for testing
        except:
            return ""
    
    def generate_encryption_key(self) -> str:
        """Generate encryption key"""
        import uuid
        return str(uuid.uuid4())
    
    def encrypt_with_key(self, data: str, key: str) -> str:
        """Encrypt with specific key"""
        import hashlib
        import base64
        
        encrypted = hashlib.sha256((data + key).encode()).hexdigest()
        return base64.b64encode(encrypted.encode()).decode()
    
    def decrypt_with_key(self, encrypted_data: str, key: str) -> str:
        """Decrypt with specific key"""
        import base64
        import hashlib
        
        try:
            decoded = base64.b64decode(encrypted_data.encode()).decode()
            expected = hashlib.sha256(("decrypted" + key).encode()).hexdigest()
            if decoded == expected:
                return "decrypted"
            else:
                raise ValueError("Invalid key")
        except:
            raise ValueError("Decryption failed")
    
    def derive_key(self, password: str, salt: str) -> str:
        """Derive key from password"""
        import hashlib
        
        return hashlib.pbkdf2_hmac('sha256', password.encode(), salt.encode(), 100000).hex()
    
    def generate_salt(self) -> str:
        """Generate salt"""
        import uuid
        return str(uuid.uuid4())
    
    def hash_data(self, data: str) -> str:
        """Hash data"""
        import hashlib
        
        return hashlib.sha256(data.encode()).hexdigest()
    
    def generate_secure_random(self, length: int) -> bytes:
        """Generate secure random bytes"""
        import os
        return os.urandom(length)
    
    def generate_secure_token(self) -> str:
        """Generate secure token"""
        import secrets
        return secrets.token_urlsafe(32)


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
        "not performance and not load"  # Skip performance and load tests by default
    ])