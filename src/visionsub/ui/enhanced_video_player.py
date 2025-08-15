"""
Enhanced Video Player Component with Modern Controls and Security Features
"""
import logging
from typing import Optional, Tuple, Any
from PyQt6.QtCore import (
    Qt, QPoint, QRect, QSize, QTimer, pyqtSignal, 
    QPropertyAnimation, QEasingCurve, QParallelAnimationGroup
)
from PyQt6.QtGui import (
    QImage, QPixmap, QPainter, QPen, QBrush, QColor,
    QFont, QFontMetrics, QMouseEvent, QWheelEvent, QKeyEvent
)
from PyQt6.QtWidgets import (
    QLabel, QRubberBand, QSlider, QPushButton, QProgressBar,
    QWidget, QVBoxLayout, QHBoxLayout, QFrame, QStyle,
    QSizePolicy, QSpacerItem, QGraphicsDropShadowEffect,
    QToolTip
)

from visionsub.ui.theme_system import (
    ThemeManager, get_theme_manager, StyledWidget, 
    Card, Button, ThemeColors
)

logger = logging.getLogger(__name__)


class VideoControls(StyledWidget):
    """Modern video controls with enhanced UX"""
    
    # Signals
    play_clicked = pyqtSignal()
    pause_clicked = pyqtSignal()
    stop_clicked = pyqtSignal()
    seek_forward = pyqtSignal()
    seek_backward = pyqtSignal()
    frame_changed = pyqtSignal(int)
    volume_changed = pyqtSignal(float)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.theme_manager = get_theme_manager()
        self.is_playing = False
        self.total_frames = 0
        self.current_frame = 0
        self.setup_ui()
        
    def setup_ui(self):
        """Setup modern video controls"""
        layout = QVBoxLayout(self)
        layout.setSpacing(8)
        layout.setContentsMargins(16, 16, 16, 16)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                background: #e2e8f0;
                border: none;
                border-radius: 4px;
                height: 6px;
            }
            QProgressBar::chunk {
                background: #3b82f6;
                border-radius: 4px;
            }
        """)
        layout.addWidget(self.progress_bar)
        
        # Control buttons
        controls_layout = QHBoxLayout()
        controls_layout.setSpacing(8)
        
        # Play/Pause button
        self.play_pause_button = QPushButton("▶️")
        self.play_pause_button.setFixedSize(40, 40)
        self.play_pause_button.setStyleSheet("""
            QPushButton {
                background: #3b82f6;
                color: white;
                border: none;
                border-radius: 20px;
                font-size: 18px;
                font-weight: bold;
            }
            QPushButton:hover {
                background: #2563eb;
            }
            QPushButton:pressed {
                background: #1d4ed8;
            }
        """)
        self.play_pause_button.clicked.connect(self.toggle_playback)
        controls_layout.addWidget(self.play_pause_button)
        
        # Stop button
        self.stop_button = QPushButton("⏹️")
        self.stop_button.setFixedSize(40, 40)
        self.stop_button.setStyleSheet("""
            QPushButton {
                background: #64748b;
                color: white;
                border: none;
                border-radius: 20px;
                font-size: 16px;
            }
            QPushButton:hover {
                background: #475569;
            }
            QPushButton:pressed {
                background: #334155;
            }
        """)
        self.stop_button.clicked.connect(self.stop_clicked.emit)
        controls_layout.addWidget(self.stop_button)
        
        # Seek backward
        self.seek_backward_button = QPushButton("⏪")
        self.seek_backward_button.setFixedSize(36, 36)
        self.seek_backward_button.setStyleSheet("""
            QPushButton {
                background: #64748b;
                color: white;
                border: none;
                border-radius: 18px;
                font-size: 14px;
            }
            QPushButton:hover {
                background: #475569;
            }
        """)
        self.seek_backward_button.clicked.connect(self.seek_backward.emit)
        controls_layout.addWidget(self.seek_backward_button)
        
        # Seek forward
        self.seek_forward_button = QPushButton("⏩")
        self.seek_forward_button.setFixedSize(36, 36)
        self.seek_forward_button.setStyleSheet("""
            QPushButton {
                background: #64748b;
                color: white;
                border: none;
                border-radius: 18px;
                font-size: 14px;
            }
            QPushButton:hover {
                background: #475569;
            }
        """)
        self.seek_forward_button.clicked.connect(self.seek_forward.emit)
        controls_layout.addWidget(self.seek_forward_button)
        
        controls_layout.addSpacing(16)
        
        # Time display
        self.time_label = QLabel("00:00 / 00:00")
        self.time_label.setStyleSheet("font-weight: bold; color: #64748b;")
        self.time_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        controls_layout.addWidget(self.time_label)
        
        controls_layout.addSpacing(16)
        
        # Volume control
        volume_layout = QHBoxLayout()
        volume_layout.setSpacing(4)
        
        self.volume_button = QPushButton("🔊")
        self.volume_button.setFixedSize(32, 32)
        self.volume_button.setStyleSheet("""
            QPushButton {
                background: transparent;
                border: none;
                font-size: 16px;
            }
            QPushButton:hover {
                background: rgba(59, 130, 246, 0.1);
                border-radius: 16px;
            }
        """)
        volume_layout.addWidget(self.volume_button)
        
        self.volume_slider = QSlider(Qt.Orientation.Horizontal)
        self.volume_slider.setRange(0, 100)
        self.volume_slider.setValue(70)
        self.volume_slider.setFixedWidth(80)
        self.volume_slider.setStyleSheet("""
            QSlider::groove:horizontal {
                height: 4px;
                background: #e2e8f0;
                border-radius: 2px;
            }
            QSlider::handle:horizontal {
                background: #64748b;
                border: none;
                width: 12px;
                margin: -4px 0;
                border-radius: 6px;
            }
            QSlider::handle:horizontal:hover {
                background: #475569;
            }
        """)
        self.volume_slider.valueChanged.connect(self.on_volume_changed)
        volume_layout.addWidget(self.volume_slider)
        
        controls_layout.addLayout(volume_layout)
        
        layout.addLayout(controls_layout)
        
    def toggle_playback(self):
        """Toggle between play and pause"""
        if self.is_playing:
            self.pause()
            self.pause_clicked.emit()
        else:
            self.play()
            self.play_clicked.emit()
            
    def play(self):
        """Set to playing state"""
        self.is_playing = True
        self.play_pause_button.setText("⏸️")
        
    def pause(self):
        """Set to paused state"""
        self.is_playing = False
        self.play_pause_button.setText("▶️")
        
    def stop(self):
        """Set to stopped state"""
        self.is_playing = False
        self.play_pause_button.setText("▶️")
        self.current_frame = 0
        self.update_progress()
        
    def set_total_frames(self, total: int):
        """Set total number of frames"""
        self.total_frames = total
        self.update_time_display()
        
    def set_current_frame(self, frame: int):
        """Set current frame"""
        self.current_frame = frame
        self.update_progress()
        self.update_time_display()
        
    def update_progress(self):
        """Update progress bar"""
        if self.total_frames > 0:
            progress = int((self.current_frame / self.total_frames) * 100)
            self.progress_bar.setValue(progress)
        else:
            self.progress_bar.setValue(0)
            
    def update_time_display(self):
        """Update time display"""
        if self.total_frames > 0:
            current_time = self.frames_to_time(self.current_frame)
            total_time = self.frames_to_time(self.total_frames)
            self.time_label.setText(f"{current_time} / {total_time}")
        else:
            self.time_label.setText("00:00 / 00:00")
            
    def frames_to_time(self, frames: int) -> str:
        """Convert frames to time string (assuming 30 FPS)"""
        seconds = frames // 30
        minutes = seconds // 60
        seconds = seconds % 60
        return f"{minutes:02d}:{seconds:02d}"
        
    def on_volume_changed(self, value: int):
        """Handle volume change"""
        volume = value / 100.0
        self.volume_changed.emit(volume)
        
        # Update volume button icon
        if value == 0:
            self.volume_button.setText("🔇")
        elif value < 30:
            self.volume_button.setText("🔈")
        elif value < 70:
            self.volume_button.setText("🔉")
        else:
            self.volume_button.setText("🔊")


class VideoOverlay(QWidget):
    """Video overlay for displaying information and controls"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.theme_manager = get_theme_manager()
        self.setup_ui()
        
    def setup_ui(self):
        """Setup overlay UI"""
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        
        # Information display
        self.info_label = QLabel()
        self.info_label.setStyleSheet("""
            QLabel {
                background: rgba(0, 0, 0, 0.7);
                color: white;
                padding: 8px 16px;
                border-radius: 8px;
                font-size: 14px;
                font-weight: bold;
            }
        """)
        self.info_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.info_label.setVisible(False)
        
    def show_info(self, text: str, duration: int = 2000):
        """Show information overlay"""
        self.info_label.setText(text)
        self.info_label.setVisible(True)
        self.info_label.adjustSize()
        
        # Center the label
        parent_rect = self.parent().rect()
        label_rect = self.info_label.rect()
        x = (parent_rect.width() - label_rect.width()) // 2
        y = parent_rect.height() - label_rect.height() - 50
        self.info_label.move(x, y)
        
        # Auto-hide after duration
        QTimer.singleShot(duration, self.info_label.hide)
        
    def paintEvent(self, event):
        """Paint overlay effects"""
        super().paintEvent(event)


class EnhancedVideoPlayer(StyledWidget):
    """Enhanced video player with modern controls and security features"""
    
    # Signals
    roi_changed = pyqtSignal(QRect)
    frame_double_clicked = pyqtSignal()
    key_pressed = pyqtSignal(str)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.theme_manager = get_theme_manager()
        
        # Video properties
        self._pixmap: Optional[QPixmap] = None
        self._current_frame: Optional[Any] = None
        self._frame_size: QSize = QSize(0, 0)
        self._aspect_ratio: float = 1.0
        
        # ROI selection
        self._roi_rect: QRect = QRect()
        self._is_selecting_roi: bool = False
        self._roi_start: QPoint = QPoint()
        self._roi_end: QPoint = QPoint()
        
        # Zoom and pan
        self._zoom_level: float = 1.0
        self._pan_offset: QPoint = QPoint()
        self._is_panning: bool = False
        self._pan_start: QPoint = QPoint()
        
        # Animation
        self._animation_timer = QTimer()
        self._animation_timer.timeout.connect(self.update_animation)
        self._target_zoom: float = 1.0
        self._animation_progress: float = 0.0
        
        self.setup_ui()
        self.setup_security_features()
        
    def setup_ui(self):
        """Setup enhanced video player UI"""
        layout = QVBoxLayout(self)
        layout.setSpacing(0)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Video display area
        self.video_display = QLabel()
        self.video_display.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_display.setStyleSheet("""
            QLabel {
                background: #0f172a;
                border: 2px solid #1e293b;
                border-radius: 12px;
            }
        """)
        self.video_display.setMouseTracking(True)
        layout.addWidget(self.video_display, 1)
        
        # Video controls
        self.controls = VideoControls()
        self.controls.play_clicked.connect(self.on_play_clicked)
        self.controls.pause_clicked.connect(self.on_pause_clicked)
        self.controls.stop_clicked.connect(self.on_stop_clicked)
        self.controls.seek_forward.connect(self.on_seek_forward)
        self.controls.seek_backward.connect(self.on_seek_backward)
        layout.addWidget(self.controls)
        
        # Video overlay
        self.overlay = VideoOverlay(self.video_display)
        self.overlay.setParent(self.video_display)
        self.overlay.resize(self.video_display.size())
        
        # Set up mouse tracking for video display
        self.video_display.installEventFilter(self)
        
    def setup_security_features(self):
        """Setup security features"""
        # Maximum zoom level to prevent excessive memory usage
        self._max_zoom_level: float = 5.0
        self._min_zoom_level: float = 0.1
        
        # Maximum frame size for security
        self._max_frame_width: int = 4096
        self._max_frame_height: int = 4096
        
    def update_frame(self, frame):
        """Update the displayed video frame with security validation"""
        try:
            # Validate frame data
            if frame is None:
                logger.warning("Received None frame")
                return
                
            # Check frame dimensions
            if hasattr(frame, 'shape'):
                height, width = frame.shape[:2]
                if width > self._max_frame_width or height > self._max_frame_height:
                    logger.error(f"Frame dimensions exceed security limits: {width}x{height}")
                    return
                    
            # Convert to QImage and validate
            if hasattr(frame, 'shape') and len(frame.shape) == 3:
                height, width, channel = frame.shape
                bytes_per_line = 3 * width
                
                # Validate memory access
                if width * height * channel > 100 * 1024 * 1024:  # 100MB limit
                    logger.error("Frame size exceeds memory limit")
                    return
                    
                q_image = QImage(frame.data, width, height, bytes_per_line, QImage.Format.Format_RGB888).rgbSwapped()
                
                # Validate image creation
                if q_image.isNull():
                    logger.error("Failed to create QImage from frame")
                    return
                    
                self._pixmap = QPixmap.fromImage(q_image)
                self._current_frame = frame
                self._frame_size = QSize(width, height)
                self._aspect_ratio = width / height if height > 0 else 1.0
                
                self._update_display()
                self.overlay.show_info("Frame updated", 1000)
                
        except Exception as e:
            logger.error(f"Error updating frame: {e}")
            
    def _update_display(self):
        """Update the video display"""
        if self._pixmap and not self._pixmap.isNull():
            # Calculate display size with zoom
            display_size = self._calculate_display_size()
            
            # Create scaled pixmap
            scaled_pixmap = self._pixmap.scaled(
                display_size,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )
            
            # Apply zoom and pan
            if self._zoom_level != 1.0:
                scaled_pixmap = self._apply_zoom_and_pan(scaled_pixmap)
                
            self.video_display.setPixmap(scaled_pixmap)
        else:
            self.video_display.setText("请打开视频文件。")
            
    def _calculate_display_size(self) -> QSize:
        """Calculate display size based on widget size and aspect ratio"""
        widget_size = self.video_display.size()
        
        if widget_size.width() <= 0 or widget_size.height() <= 0:
            return QSize(640, 480)
            
        # Calculate size maintaining aspect ratio
        width = widget_size.width()
        height = int(width / self._aspect_ratio)
        
        if height > widget_size.height():
            height = widget_size.height()
            width = int(height * self._aspect_ratio)
            
        return QSize(width, height)
        
    def _apply_zoom_and_pan(self, pixmap: QPixmap) -> QPixmap:
        """Apply zoom and pan transformations"""
        if self._zoom_level == 1.0:
            return pixmap
            
        # Create new pixmap for zoomed content
        original_size = pixmap.size()
        zoomed_size = QSize(
            int(original_size.width() * self._zoom_level),
            int(original_size.height() * self._zoom_level)
        )
        
        # Scale the pixmap
        zoomed_pixmap = pixmap.scaled(
            zoomed_size,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )
        
        return zoomed_pixmap
        
    def set_roi(self, roi: QRect):
        """Set ROI rectangle"""
        if self._roi_rect != roi:
            self._roi_rect = roi
            self.update()
            self.roi_changed.emit(roi)
            
    def get_roi(self) -> QRect:
        """Get current ROI rectangle"""
        return self._roi_rect
        
    def clear_roi(self):
        """Clear ROI selection"""
        self._roi_rect = QRect()
        self.update()
        
    def set_zoom(self, zoom_level: float):
        """Set zoom level with animation"""
        # Validate zoom level
        zoom_level = max(self._min_zoom_level, min(self._max_zoom_level, zoom_level))
        
        if zoom_level != self._zoom_level:
            self._target_zoom = zoom_level
            self._animation_progress = 0.0
            self._animation_timer.start(16)  # ~60 FPS
            
    def update_animation(self):
        """Update zoom animation"""
        self._animation_progress += 0.1
        
        if self._animation_progress >= 1.0:
            self._zoom_level = self._target_zoom
            self._animation_timer.stop()
        else:
            # Smooth easing
            t = self._animation_progress
            t = t * t * (3.0 - 2.0 * t)  # Smoothstep
            self._zoom_level = self._zoom_level + (self._target_zoom - self._zoom_level) * t
            
        self._update_display()
        
    def eventFilter(self, obj, event):
        """Filter events for video display"""
        if obj != self.video_display:
            return False
            
        if event.type() == event.Type.MouseButtonPress:
            return self.mousePressEvent(event)
        elif event.type() == event.Type.MouseMove:
            return self.mouseMoveEvent(event)
        elif event.type() == event.Type.MouseButtonRelease:
            return self.mouseReleaseEvent(event)
        elif event.type() == event.Type.MouseButtonDblClick:
            return self.mouseDoubleClickEvent(event)
        elif event.type() == event.Type.Wheel:
            return self.wheelEvent(event)
        elif event.type() == event.Type.KeyPress:
            return self.keyPressEvent(event)
            
        return False
        
    def mousePressEvent(self, event: QMouseEvent):
        """Handle mouse press events"""
        if event.button() == Qt.MouseButton.LeftButton:
            if event.modifiers() & Qt.KeyboardModifier.ControlModifier:
                # Start ROI selection
                self._is_selecting_roi = True
                self._roi_start = event.pos()
                self._roi_end = event.pos()
            else:
                # Start panning
                self._is_panning = True
                self._pan_start = event.pos()
                
        elif event.button() == Qt.MouseButton.RightButton:
            # Clear ROI
            self.clear_roi()
            
    def mouseMoveEvent(self, event: QMouseEvent):
        """Handle mouse move events"""
        if self._is_selecting_roi:
            self._roi_end = event.pos()
            self.update()
            
        elif self._is_panning:
            delta = event.pos() - self._pan_start
            self._pan_offset += delta
            self._pan_start = event.pos()
            self._update_display()
            
    def mouseReleaseEvent(self, event: QMouseEvent):
        """Handle mouse release events"""
        if event.button() == Qt.MouseButton.LeftButton:
            if self._is_selecting_roi:
                self._is_selecting_roi = False
                
                # Create ROI rectangle
                roi_rect = QRect(self._roi_start, self._roi_end).normalized()
                
                # Convert to pixmap coordinates
                if self._pixmap and not self._pixmap.isNull():
                    pixmap_rect = self._map_widget_to_pixmap(roi_rect)
                    self.set_roi(pixmap_rect)
                    
            elif self._is_panning:
                self._is_panning = False
                
    def mouseDoubleClickEvent(self, event: QMouseEvent):
        """Handle double click events"""
        if event.button() == Qt.MouseButton.LeftButton:
            self.frame_double_clicked.emit()
            
    def wheelEvent(self, event: QWheelEvent):
        """Handle wheel events for zooming"""
        if event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            # Zoom with Ctrl+Wheel
            delta = event.angleDelta().y()
            zoom_factor = 1.1 if delta > 0 else 0.9
            new_zoom = self._zoom_level * zoom_factor
            self.set_zoom(new_zoom)
            
    def keyPressEvent(self, event: QKeyEvent):
        """Handle key press events"""
        key_map = {
            Qt.Key.Key_Space: "space",
            Qt.Key.Key_Left: "left",
            Qt.Key.Key_Right: "right",
            Qt.Key.Key_Up: "up",
            Qt.Key.Key_Down: "down",
            Qt.Key.Key_Plus: "plus",
            Qt.Key.Key_Minus: "minus",
            Qt.Key.Key_Equal: "equal",
            Qt.Key.Key_0: "zero",
        }
        
        key_name = key_map.get(event.key(), str(event.key()))
        self.key_pressed.emit(key_name)
        
        # Handle zoom shortcuts
        if event.key() == Qt.Key.Key_Plus or event.key() == Qt.Key.Key_Equal:
            self.set_zoom(self._zoom_level * 1.2)
        elif event.key() == Qt.Key.Key_Minus:
            self.set_zoom(self._zoom_level * 0.8)
        elif event.key() == Qt.Key.Key_0:
            self.set_zoom(1.0)
            self._pan_offset = QPoint()
            
    def _map_widget_to_pixmap(self, widget_rect: QRect) -> QRect:
        """Map widget coordinates to pixmap coordinates"""
        if not self._pixmap or self._pixmap.isNull():
            return QRect()
            
        widget_size = self.video_display.size()
        pixmap_size = self._pixmap.size()
        
        # Calculate scaled pixmap size
        scaled_pixmap = self._pixmap.scaled(
            widget_size,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )
        scaled_size = scaled_pixmap.size()
        
        # Calculate offset
        offset_x = (widget_size.width() - scaled_size.width()) / 2
        offset_y = (widget_size.height() - scaled_size.height()) / 2
        
        # Adjust rectangle
        adjusted_rect = widget_rect.translated(-offset_x, -offset_y)
        
        # Scale to original pixmap size
        scale_x = pixmap_size.width() / scaled_size.width()
        scale_y = pixmap_size.height() / scaled_size.height()
        
        pixmap_x = int(adjusted_rect.x() * scale_x)
        pixmap_y = int(adjusted_rect.y() * scale_y)
        pixmap_width = int(adjusted_rect.width() * scale_x)
        pixmap_height = int(adjusted_rect.height() * scale_y)
        
        return QRect(pixmap_x, pixmap_y, pixmap_width, pixmap_height)
        
    def paintEvent(self, event):
        """Custom paint event"""
        super().paintEvent(event)
        
        # Draw ROI selection rectangle
        if self._is_selecting_roi:
            painter = QPainter(self)
            painter.setPen(QPen(QColor(59, 130, 246), 2, Qt.PenStyle.DashLine))
            painter.setBrush(QBrush(QColor(59, 130, 246, 50)))
            painter.drawRect(QRect(self._roi_start, self._roi_end).normalized())
            
    def on_play_clicked(self):
        """Handle play button click"""
        self.overlay.show_info("播放", 1000)
        
    def on_pause_clicked(self):
        """Handle pause button click"""
        self.overlay.show_info("暂停", 1000)
        
    def on_stop_clicked(self):
        """Handle stop button click"""
        self.overlay.show_info("停止", 1000)
        
    def on_seek_forward(self):
        """Handle seek forward"""
        self.overlay.show_info("快进", 1000)
        
    def on_seek_backward(self):
        """Handle seek backward"""
        self.overlay.show_info("快退", 1000)
        
    def resizeEvent(self, event):
        """Handle resize events"""
        super().resizeEvent(event)
        
        # Update overlay size
        if hasattr(self, 'overlay'):
            self.overlay.resize(self.video_display.size())
            
        # Update display
        self._update_display()


# Example usage
if __name__ == "__main__":
    import sys
    import numpy as np
    from PyQt6.QtWidgets import QApplication
    
    app = QApplication(sys.argv)
    
    # Initialize theme manager
    theme_manager = get_theme_manager()
    
    # Create test frame
    test_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    # Create and show video player
    player = EnhancedVideoPlayer()
    player.update_frame(test_frame)
    player.show()
    
    sys.exit(app.exec())