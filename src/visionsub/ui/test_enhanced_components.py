"""
Test script for enhanced UI components
"""
import sys
import time
from pathlib import Path
from typing import List
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from PyQt6.QtCore import Qt, QTimer, QSize, QRect
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, 
    QWidget, QPushButton, QLabel, QTabWidget
)
from PyQt6.QtGui import QImage, QPixmap

# Import enhanced components
from visionsub.ui.theme_system import get_theme_manager, initialize_theme_manager
from visionsub.ui.enhanced_main_window import EnhancedMainWindow
from visionsub.ui.enhanced_video_player import EnhancedVideoPlayer
from visionsub.ui.enhanced_settings_dialog import EnhancedSettingsDialog
from visionsub.ui.enhanced_ocr_preview import EnhancedOCRPreview, OCRResult
from visionsub.ui.enhanced_subtitle_editor import EnhancedSubtitleEditor
from visionsub.models.config import AppConfig
from visionsub.models.subtitle import SubtitleItem
from visionsub.view_models.main_view_model import MainViewModel


class TestWindow(QMainWindow):
    """Test window for enhanced UI components"""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Enhanced UI Components Test")
        self.setGeometry(100, 100, 1400, 900)
        
        self.setup_ui()
        
    def setup_ui(self):
        """Setup test UI"""
        central_widget = QWidget()
        layout = QVBoxLayout(central_widget)
        
        # Title
        title = QLabel("Enhanced UI Components Test")
        title.setStyleSheet("font-size: 24px; font-weight: bold; padding: 20px;")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)
        
        # Tab widget
        tabs = QTabWidget()
        layout.addWidget(tabs)
        
        # Main Window Tab
        main_window_tab = self.create_main_window_tab()
        tabs.addTab(main_window_tab, "Main Window")
        
        # Video Player Tab
        video_player_tab = self.create_video_player_tab()
        tabs.addTab(video_player_tab, "Video Player")
        
        # Settings Tab
        settings_tab = self.create_settings_tab()
        tabs.addTab(settings_tab, "Settings")
        
        # OCR Preview Tab
        ocr_preview_tab = self.create_ocr_preview_tab()
        tabs.addTab(ocr_preview_tab, "OCR Preview")
        
        # Subtitle Editor Tab
        subtitle_editor_tab = self.create_subtitle_editor_tab()
        tabs.addTab(subtitle_editor_tab, "Subtitle Editor")
        
        self.setCentralWidget(central_widget)
        
    def create_main_window_tab(self) -> QWidget:
        """Create main window test tab"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Description
        desc = QLabel("Enhanced Main Window with modern UI/UX and security features")
        desc.setStyleSheet("font-size: 16px; padding: 10px; color: #64748b;")
        layout.addWidget(desc)
        
        # Test button
        test_button = QPushButton("Open Enhanced Main Window")
        test_button.setStyleSheet("""
            QPushButton {
                background: #3b82f6;
                color: white;
                border: none;
                border-radius: 8px;
                padding: 12px 24px;
                font-size: 16px;
                font-weight: bold;
            }
            QPushButton:hover {
                background: #2563eb;
            }
            QPushButton:pressed {
                background: #1d4ed8;
            }
        """)
        test_button.clicked.connect(self.open_main_window)
        layout.addWidget(test_button)
        
        layout.addStretch()
        return widget
        
    def create_video_player_tab(self) -> QWidget:
        """Create video player test tab"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Description
        desc = QLabel("Enhanced Video Player with modern controls and security features")
        desc.setStyleSheet("font-size: 16px; padding: 10px; color: #64748b;")
        layout.addWidget(desc)
        
        # Create test frame
        test_frame = self.create_test_frame()
        
        # Video player
        self.video_player = EnhancedVideoPlayer()
        self.video_player.update_frame(test_frame)
        layout.addWidget(self.video_player)
        
        # Control buttons
        button_layout = QHBoxLayout()
        
        play_button = QPushButton("Play Animation")
        play_button.clicked.connect(self.play_video_animation)
        button_layout.addWidget(play_button)
        
        zoom_button = QPushButton("Test Zoom")
        zoom_button.clicked.connect(self.test_video_zoom)
        button_layout.addWidget(zoom_button)
        
        button_layout.addStretch()
        layout.addLayout(button_layout)
        
        return widget
        
    def create_settings_tab(self) -> QWidget:
        """Create settings test tab"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Description
        desc = QLabel("Enhanced Settings Dialog with modern UI/UX and security features")
        desc.setStyleSheet("font-size: 16px; padding: 10px; color: #64748b;")
        layout.addWidget(desc)
        
        # Test button
        test_button = QPushButton("Open Enhanced Settings")
        test_button.setStyleSheet("""
            QPushButton {
                background: #10b981;
                color: white;
                border: none;
                border-radius: 8px;
                padding: 12px 24px;
                font-size: 16px;
                font-weight: bold;
            }
            QPushButton:hover {
                background: #059669;
            }
            QPushButton:pressed {
                background: #047857;
            }
        """)
        test_button.clicked.connect(self.open_settings)
        layout.addWidget(test_button)
        
        layout.addStretch()
        return widget
        
    def create_ocr_preview_tab(self) -> QWidget:
        """Create OCR preview test tab"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Description
        desc = QLabel("Enhanced OCR Preview with secure rendering and modern design")
        desc.setStyleSheet("font-size: 16px; padding: 10px; color: #64748b;")
        layout.addWidget(desc)
        
        # OCR preview
        self.ocr_preview = EnhancedOCRPreview()
        layout.addWidget(self.ocr_preview)
        
        # Test buttons
        button_layout = QHBoxLayout()
        
        add_result_button = QPushButton("Add Test Results")
        add_result_button.clicked.connect(self.add_test_ocr_results)
        button_layout.addWidget(add_result_button)
        
        clear_button = QPushButton("Clear Results")
        clear_button.clicked.connect(self.clear_ocr_results)
        button_layout.addWidget(clear_button)
        
        button_layout.addStretch()
        layout.addLayout(button_layout)
        
        return widget
        
    def create_subtitle_editor_tab(self) -> QWidget:
        """Create subtitle editor test tab"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Description
        desc = QLabel("Enhanced Subtitle Editor with modern UI/UX and security features")
        desc.setStyleSheet("font-size: 16px; padding: 10px; color: #64748b;")
        layout.addWidget(desc)
        
        # Subtitle editor
        self.subtitle_editor = EnhancedSubtitleEditor()
        layout.addWidget(self.subtitle_editor)
        
        # Test buttons
        button_layout = QHBoxLayout()
        
        load_button = QPushButton("Load Test Subtitles")
        load_button.clicked.connect(self.load_test_subtitles)
        button_layout.addWidget(load_button)
        
        add_button = QPushButton("Add Test Subtitle")
        add_button.clicked.connect(self.add_test_subtitle)
        button_layout.addWidget(add_button)
        
        button_layout.addStretch()
        layout.addLayout(button_layout)
        
        return widget
        
    def create_test_frame(self) -> np.ndarray:
        """Create test video frame"""
        # Create a colorful test frame
        height, width = 480, 640
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Create gradient background
        for y in range(height):
            for x in range(width):
                frame[y, x] = [
                    int(255 * x / width),      # Red gradient
                    int(255 * y / height),     # Green gradient
                    int(255 * (x + y) / (width + height))  # Blue gradient
                ]
        
        # Add some text regions (simulating subtitles)
        cv2 = None
        try:
            import cv2
            # Add black text box at bottom
            cv2.rectangle(frame, (50, height - 100), (width - 50, height - 50), (0, 0, 0), -1)
            
            # Add text
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(frame, "Test Subtitle Text", (100, height - 70), font, 1, (255, 255, 255), 2)
            cv2.putText(frame, "Enhanced UI Components", (100, height - 40), font, 1, (255, 255, 255), 2)
            
        except ImportError:
            # Fallback without OpenCV
            pass
        
        return frame
        
    def open_main_window(self):
        """Open enhanced main window"""
        try:
            vm = MainViewModel()
            window = EnhancedMainWindow(vm)
            window.show()
        except Exception as e:
            print(f"Error opening main window: {e}")
            
    def play_video_animation(self):
        """Play video animation"""
        try:
            # Create animated frames
            frames = []
            for i in range(30):
                frame = self.create_test_frame()
                # Add moving element
                x = int(50 + i * 20)
                y = 50
                if hasattr(frame, 'shape'):
                    frame[y:y+30, x:x+30] = [255, 0, 0]  # Red square
                frames.append(frame)
            
            # Animate
            self.animation_frames = frames
            self.animation_index = 0
            
            self.animation_timer = QTimer()
            self.animation_timer.timeout.connect(self.next_animation_frame)
            self.animation_timer.start(100)  # 10 FPS
            
        except Exception as e:
            print(f"Error playing animation: {e}")
            
    def next_animation_frame(self):
        """Show next animation frame"""
        if hasattr(self, 'animation_frames') and hasattr(self, 'animation_index'):
            if self.animation_index < len(self.animation_frames):
                frame = self.animation_frames[self.animation_index]
                self.video_player.update_frame(frame)
                self.animation_index += 1
            else:
                self.animation_timer.stop()
                
    def test_video_zoom(self):
        """Test video zoom functionality"""
        try:
            # Test zoom levels
            zoom_levels = [1.0, 1.5, 2.0, 1.0]
            self.zoom_index = 0
            
            def next_zoom():
                if hasattr(self, 'zoom_index'):
                    if self.zoom_index < len(zoom_levels):
                        self.video_player.set_zoom(zoom_levels[self.zoom_index])
                        self.zoom_index += 1
                        QTimer.singleShot(1000, next_zoom)
            
            next_zoom()
            
        except Exception as e:
            print(f"Error testing zoom: {e}")
            
    def open_settings(self):
        """Open enhanced settings dialog"""
        try:
            config = AppConfig()
            dialog = EnhancedSettingsDialog(config, self)
            dialog.config_changed.connect(lambda c: print(f"Config changed: {c}"))
            dialog.show()
        except Exception as e:
            print(f"Error opening settings: {e}")
            
    def add_test_ocr_results(self):
        """Add test OCR results"""
        try:
            results = [
                OCRResult(
                    text="This is a high confidence result",
                    confidence=0.95,
                    language="en",
                    position=QRect(10, 10, 200, 30),
                    timestamp=1.0
                ),
                OCRResult(
                    text="Medium confidence text here",
                    confidence=0.75,
                    language="en",
                    position=QRect(10, 50, 180, 30),
                    timestamp=2.0
                ),
                OCRResult(
                    text="Low confidence recognition",
                    confidence=0.45,
                    language="en",
                    position=QRect(10, 90, 160, 30),
                    timestamp=3.0
                ),
                OCRResult(
                    text="Another high confidence result",
                    confidence=0.88,
                    language="en",
                    position=QRect(10, 130, 220, 30),
                    timestamp=4.0
                )
            ]
            
            self.ocr_preview.add_results(results)
            
        except Exception as e:
            print(f"Error adding OCR results: {e}")
            
    def clear_ocr_results(self):
        """Clear OCR results"""
        try:
            self.ocr_preview.clear_results()
        except Exception as e:
            print(f"Error clearing OCR results: {e}")
            
    def load_test_subtitles(self):
        """Load test subtitles"""
        try:
            subtitles = [
                SubtitleItem(
                    index=1,
                    start_time="00:00:01,000",
                    end_time="00:00:03,000",
                    content="This is the first test subtitle"
                ),
                SubtitleItem(
                    index=2,
                    start_time="00:00:04,000",
                    end_time="00:00:06,000",
                    content="This is the second test subtitle with longer text"
                ),
                SubtitleItem(
                    index=3,
                    start_time="00:00:07,000",
                    end_time="00:00:09,000",
                    content="Third subtitle for testing"
                )
            ]
            
            self.subtitle_editor.set_subtitles(subtitles)
            
        except Exception as e:
            print(f"Error loading test subtitles: {e}")
            
    def add_test_subtitle(self):
        """Add test subtitle"""
        try:
            subtitle = SubtitleItem(
                index=4,
                start_time="00:00:10,000",
                end_time="00:00:12,000",
                content="Dynamically added subtitle"
            )
            
            # Get current subtitles and add new one
            current_subtitles = self.subtitle_editor.get_modified_subtitles()
            current_subtitles.append(subtitle)
            self.subtitle_editor.set_subtitles(current_subtitles)
            
        except Exception as e:
            print(f"Error adding test subtitle: {e}")


def main():
    """Main test function"""
    try:
        # Create application
        app = QApplication(sys.argv)
        
        # Initialize theme manager
        theme_manager = initialize_theme_manager()
        
        # Create test window
        test_window = TestWindow()
        test_window.show()
        
        # Run application
        return_code = app.exec()
        
        return return_code
        
    except Exception as e:
        print(f"Error running test: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())