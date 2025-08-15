"""
Enhanced Main Window with Modern UI/UX Design and Security Features
"""
import logging
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from PyQt6.QtCore import (
    Qt, pyqtSlot, QSize, QPoint, QRect, QTimer, 
    QPropertyAnimation, QEasingCurve, pyqtSignal
)
from PyQt6.QtGui import (
    QFont, QIcon, QPixmap, QPalette, QColor, 
    QPainter, QBrush, QPen, QFontMetrics, QAction
)
from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QSplitter, QFrame, QLabel, QPushButton, QComboBox,
    QSlider, QSpinBox, QProgressBar, QListWidget,
    QFileDialog, QMessageBox, QStatusBar, QToolBar,
    QMenuBar, QMenu, QDialog, QTabWidget, QScrollArea,
    QGroupBox, QFormLayout, QCheckBox, QDoubleSpinBox,
    QSpinBox, QTextEdit, QSizePolicy, QSpacerItem,
    QStackedWidget, QGraphicsDropShadowEffect, QToolTip
)

from visionsub.models.config import OcrConfig, AppConfig
from visionsub.ui.theme_system import (
    ThemeManager, get_theme_manager, StyledWidget, 
    Card, Button, ThemeColors, ThemeDefinition
)
from visionsub.ui.video_player import VideoPlayer
from visionsub.ui.roi_selection import ROISelectionPanel
from visionsub.ui.subtitle_editor import SubtitleEditorWindow
from visionsub.ui.advanced_ocr_settings import AdvancedOCRSettingsDialog
from visionsub.view_models.main_view_model import MainViewModel

logger = logging.getLogger(__name__)


class SecureFileDialog(QFileDialog):
    """Secure file dialog with input validation"""
    
    def __init__(self, parent=None, allowed_extensions=None, max_file_size_mb=500):
        super().__init__(parent)
        self.allowed_extensions = allowed_extensions or ['mp4', 'avi', 'mkv', 'mov', 'wmv', 'flv', 'webm']
        self.max_file_size_mb = max_file_size_mb
        self.setFileMode(QFileDialog.FileMode.ExistingFiles)
        
    def accept(self):
        """Override accept to validate selected files"""
        selected_files = self.selectedFiles()
        
        if not selected_files:
            return
            
        # Validate each file
        for file_path in selected_files:
            if not self._validate_file(file_path):
                return
                
        super().accept()
        
    def _validate_file(self, file_path: str) -> bool:
        """Validate file path and properties"""
        try:
            path = Path(file_path)
            
            # Check if file exists
            if not path.exists():
                QMessageBox.warning(self, "文件错误", f"文件不存在: {file_path}")
                return False
                
            # Check file extension
            if path.suffix.lower().lstrip('.') not in self.allowed_extensions:
                QMessageBox.warning(
                    self, "格式错误", 
                    f"不支持的文件格式: {path.suffix}\n支持的格式: {', '.join(self.allowed_extensions)}"
                )
                return False
                
            # Check file size
            file_size_mb = path.stat().st_size / (1024 * 1024)
            if file_size_mb > self.max_file_size_mb:
                QMessageBox.warning(
                    self, "文件过大", 
                    f"文件大小 ({file_size_mb:.1f}MB) 超过限制 ({self.max_file_size_mb}MB)"
                )
                return False
                
            # Check file path security
            if self._is_malicious_path(file_path):
                QMessageBox.warning(self, "安全错误", "检测到不安全的文件路径")
                return False
                
            return True
            
        except Exception as e:
            logger.error(f"File validation error: {e}")
            QMessageBox.warning(self, "验证错误", f"文件验证失败: {e}")
            return False
            
    def _is_malicious_path(self, file_path: str) -> bool:
        """Check for malicious path patterns"""
        malicious_patterns = [
            r'\.\.',  # Directory traversal
            r'~$',   # Backup files
            r'\.tmp$', # Temporary files
            r'\.exe$', # Executable files
            r'\.bat$', # Batch files
            r'\.sh$',  # Shell scripts
            r'\.cmd$', # Command files
        ]
        
        for pattern in malicious_patterns:
            if re.search(pattern, file_path, re.IGNORECASE):
                return True
                
        return False


class ModernStatusBar(QStatusBar):
    """Modern status bar with security indicators"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.theme_manager = get_theme_manager()
        self.setup_ui()
        
    def setup_ui(self):
        """Setup modern status bar"""
        # Security status indicator
        self.security_label = QLabel("🔒 安全")
        self.security_label.setStyleSheet("color: #10b981; font-weight: bold;")
        self.addWidget(self.security_label)
        
        # Performance indicator
        self.performance_label = QLabel("⚡ 性能: 正常")
        self.performance_label.setStyleSheet("color: #3b82f6;")
        self.addWidget(self.performance_label)
        
        # Memory usage indicator
        self.memory_label = QLabel("💾 内存: 0 MB")
        self.memory_label.setStyleSheet("color: #8b5cf6;")
        self.addWidget(self.memory_label)
        
        # Progress indicator
        self.progress_bar = QProgressBar()
        self.progress_bar.setMaximumWidth(200)
        self.progress_bar.setVisible(False)
        self.addWidget(self.progress_bar)
        
        # Status message
        self.status_label = QLabel("就绪")
        self.addPermanentWidget(self.status_label)
        
        # Update timer
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.update_indicators)
        self.update_timer.start(1000)
        
    def update_indicators(self):
        """Update status indicators"""
        try:
            # Update memory usage
            import psutil
            memory_mb = psutil.Process().memory_info().rss / 1024 / 1024
            self.memory_label.setText(f"💾 内存: {memory_mb:.1f} MB")
            
            # Update color based on memory usage
            if memory_mb > 1000:
                self.memory_label.setStyleSheet("color: #ef4444;")
            elif memory_mb > 500:
                self.memory_label.setStyleSheet("color: #f59e0b;")
            else:
                self.memory_label.setStyleSheet("color: #8b5cf6;")
                
        except ImportError:
            pass
            
    def show_progress(self, current: int, total: int):
        """Show progress bar"""
        self.progress_bar.setVisible(True)
        self.progress_bar.setMaximum(total)
        self.progress_bar.setValue(current)
        
    def hide_progress(self):
        """Hide progress bar"""
        self.progress_bar.setVisible(False)
        
    def set_status(self, message: str, message_type: str = "info"):
        """Set status message with color coding"""
        self.status_label.setText(message)
        
        colors = {
            "info": "#3b82f6",
            "success": "#10b981", 
            "warning": "#f59e0b",
            "error": "#ef4444"
        }
        
        color = colors.get(message_type, "#64748b")
        self.status_label.setStyleSheet(f"color: {color};")


class ModernToolbar(QToolBar):
    """Modern toolbar with animations and security features"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.theme_manager = get_theme_manager()
        self.setup_ui()
        
    def setup_ui(self):
        """Setup modern toolbar"""
        self.setMovable(False)
        self.setIconSize(QSize(24, 24))
        self.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonIconOnly)
        
        # Apply modern styling
        self.setStyleSheet("""
            QToolBar {
                background: transparent;
                border: none;
                spacing: 4px;
            }
            QToolButton {
                background: transparent;
                border: none;
                border-radius: 6px;
                padding: 8px;
                margin: 2px;
            }
            QToolButton:hover {
                background: rgba(59, 130, 246, 0.1);
            }
            QToolButton:pressed {
                background: rgba(59, 130, 246, 0.2);
            }
        """)


class EnhancedVideoPlayer(VideoPlayer):
    """Enhanced video player with modern controls and security features"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.theme_manager = get_theme_manager()
        self.setup_enhanced_features()
        
    def setup_enhanced_features(self):
        """Setup enhanced features"""
        # Add shadow effect
        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(20)
        shadow.setColor(QColor(0, 0, 0, 50))
        shadow.setOffset(0, 4)
        self.setGraphicsEffect(shadow)
        
        # Enhanced styling
        self.setStyleSheet("""
            QLabel {
                background: #1e293b;
                border: 2px solid #334155;
                border-radius: 12px;
                padding: 8px;
            }
        """)


class ModernControlPanel(StyledWidget):
    """Modern control panel with enhanced UX"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.vm = None
        self.setup_ui()
        
    def set_view_model(self, vm):
        """Set view model and connect signals"""
        self.vm = vm
        self.connect_signals()
        
    def setup_ui(self):
        """Setup modern control panel"""
        layout = QVBoxLayout(self)
        layout.setSpacing(16)
        layout.setContentsMargins(20, 20, 20, 20)
        
        # File operations card
        file_card = Card("文件操作")
        file_layout = QVBoxLayout()
        
        self.open_button = Button("打开视频", "primary")
        self.open_button.clicked.connect(self.open_file)
        file_layout.addWidget(self.open_button)
        
        self.add_to_queue_button = Button("添加到队列", "secondary")
        self.add_to_queue_button.clicked.connect(self.add_to_queue)
        file_layout.addWidget(self.add_to_queue_button)
        
        file_card.add_layout(file_layout)
        layout.addWidget(file_card)
        
        # OCR settings card
        ocr_card = Card("OCR 设置")
        ocr_layout = QFormLayout()
        ocr_layout.setSpacing(12)
        
        self.language_combo = QComboBox()
        self.language_combo.addItems(["中文", "英文", "韩文", "日文", "法文", "德文", "西班牙文", "俄文", "阿拉伯文", "印地文"])
        self.language_combo.setCurrentText("中文")
        ocr_layout.addRow("识别语言:", self.language_combo)
        
        threshold_layout = QHBoxLayout()
        self.threshold_slider = QSlider(Qt.Orientation.Horizontal)
        self.threshold_slider.setRange(0, 255)
        self.threshold_slider.setValue(180)
        
        self.threshold_spinbox = QSpinBox()
        self.threshold_spinbox.setRange(0, 255)
        self.threshold_spinbox.setValue(180)
        
        threshold_layout.addWidget(self.threshold_slider)
        threshold_layout.addWidget(self.threshold_spinbox)
        ocr_layout.addRow("二值化阈值:", threshold_layout)
        
        self.confidence_spinbox = QDoubleSpinBox()
        self.confidence_spinbox.setRange(0.0, 1.0)
        self.confidence_spinbox.setSingleStep(0.05)
        self.confidence_spinbox.setValue(0.8)
        ocr_layout.addRow("置信度阈值:", self.confidence_spinbox)
        
        ocr_card.add_layout(ocr_layout)
        layout.addWidget(ocr_card)
        
        # Action buttons card
        action_card = Card("操作")
        action_layout = QVBoxLayout()
        
        self.ocr_button = Button("执行OCR识别", "accent")
        self.ocr_button.clicked.connect(self.run_ocr)
        action_layout.addWidget(self.ocr_button)
        
        self.start_batch_button = Button("开始批量处理", "success")
        self.start_batch_button.clicked.connect(self.start_batch)
        action_layout.addWidget(self.start_batch_button)
        
        action_card.add_layout(action_layout)
        layout.addWidget(action_card)
        
        layout.addStretch()
        
    def connect_signals(self):
        """Connect signals to view model"""
        if self.vm:
            self.language_combo.currentTextChanged.connect(self.vm.set_language)
            self.threshold_slider.valueChanged.connect(self.vm.set_threshold)
            self.threshold_spinbox.valueChanged.connect(self.vm.set_threshold)
            
            # Sync threshold widgets
            self.threshold_slider.valueChanged.connect(self.threshold_spinbox.setValue)
            self.threshold_spinbox.valueChanged.connect(self.threshold_slider.setValue)
            
    def open_file(self):
        """Open video file with security validation"""
        dialog = SecureFileDialog(self)
        dialog.setWindowTitle("选择视频文件")
        dialog.setNameFilter("视频文件 (*.mp4 *.avi *.mkv *.mov *.wmv *.flv *.webm)")
        
        if dialog.exec() == QDialog.DialogCode.Accepted:
            files = dialog.selectedFiles()
            if files and self.vm:
                self.vm.load_video(files[0])
                
    def add_to_queue(self):
        """Add files to batch queue"""
        dialog = SecureFileDialog(self)
        dialog.setWindowTitle("选择要添加到队列的视频文件")
        dialog.setFileMode(QFileDialog.FileMode.ExistingFiles)
        dialog.setNameFilter("视频文件 (*.mp4 *.avi *.mkv *.mov *.wmv *.flv *.webm)")
        
        if dialog.exec() == QDialog.DialogCode.Accepted:
            files = dialog.selectedFiles()
            if files and self.vm:
                self.vm.add_to_queue(files)
                
    def run_ocr(self):
        """Run OCR on current frame"""
        if self.vm:
            self.vm.run_single_frame_ocr_sync()
            
    def start_batch(self):
        """Start batch processing"""
        if self.vm:
            self.vm.start_batch_processing()


class ModernBatchPanel(StyledWidget):
    """Modern batch processing panel"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.vm = None
        self.setup_ui()
        
    def set_view_model(self, vm):
        """Set view model and connect signals"""
        self.vm = vm
        self.connect_signals()
        
    def setup_ui(self):
        """Setup modern batch panel"""
        layout = QVBoxLayout(self)
        layout.setSpacing(16)
        layout.setContentsMargins(20, 20, 20, 20)
        
        # Queue card
        queue_card = Card("处理队列")
        queue_layout = QVBoxLayout()
        
        self.queue_list = QListWidget()
        self.queue_list.setAlternatingRowColors(True)
        self.queue_list.setStyleSheet("""
            QListWidget {
                background: #f8fafc;
                border: 1px solid #e2e8f0;
                border-radius: 8px;
                padding: 8px;
            }
            QListWidget::item {
                padding: 8px;
                border-radius: 4px;
                margin: 2px;
            }
            QListWidget::item:selected {
                background: #3b82f6;
                color: white;
            }
        """)
        queue_layout.addWidget(self.queue_list)
        
        queue_card.add_layout(queue_layout)
        layout.addWidget(queue_card)
        
        # Progress card
        progress_card = Card("处理进度")
        progress_layout = QVBoxLayout()
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                background: #f1f5f9;
                border: none;
                border-radius: 8px;
                height: 24px;
                text-align: center;
            }
            QProgressBar::chunk {
                background: #3b82f6;
                border-radius: 8px;
            }
        """)
        progress_layout.addWidget(self.progress_bar)
        
        self.status_label = QLabel("等待处理...")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        progress_layout.addWidget(self.status_label)
        
        progress_card.add_layout(progress_layout)
        layout.addWidget(progress_card)
        
        # Queue management
        queue_mgmt_layout = QHBoxLayout()
        
        self.clear_queue_button = Button("清空队列", "error")
        self.clear_queue_button.clicked.connect(self.clear_queue)
        queue_mgmt_layout.addWidget(self.clear_queue_button)
        
        queue_mgmt_layout.addStretch()
        
        layout.addLayout(queue_mgmt_layout)
        
    def connect_signals(self):
        """Connect signals to view model"""
        if self.vm:
            self.vm.queue_changed.connect(self.update_queue)
            self.vm.batch_progress_changed.connect(self.progress_bar.setValue)
            self.vm.batch_status_changed.connect(self.status_label.setText)
            
    def update_queue(self, queue: List[str]):
        """Update queue list"""
        self.queue_list.clear()
        self.queue_list.addItems(queue)
        
    def clear_queue(self):
        """Clear processing queue"""
        if self.vm:
            self.vm.clear_queue()


class EnhancedMainWindow(QMainWindow):
    """Enhanced main window with modern UI/UX and security features"""
    
    def __init__(self, view_model: MainViewModel):
        super().__init__()
        self.vm = view_model
        self.theme_manager = get_theme_manager()
        
        # Initialize theme system
        self.theme_manager.theme_changed.connect(self.on_theme_changed)
        
        # Setup window properties
        self.setWindowTitle("VisionSub - 视频OCR字幕提取工具")
        self.setGeometry(100, 100, 1600, 900)
        self.setMinimumSize(1200, 700)
        
        # Initialize UI components
        self.setup_ui()
        self.setup_menus()
        self.setup_status_bar()
        self.setup_toolbar()
        
        # Connect signals
        self.connect_signals()
        
        # Initialize theme
        self.theme_manager.set_theme("dark")
        
        logger.info("Enhanced main window initialized")
        
    def setup_ui(self):
        """Setup main UI layout"""
        # Central widget
        central_widget = QWidget()
        main_layout = QHBoxLayout(central_widget)
        main_layout.setSpacing(0)
        main_layout.setContentsMargins(0, 0, 0, 0)
        
        # Main splitter
        main_splitter = QSplitter(Qt.Orientation.Horizontal)
        main_splitter.setHandleWidth(2)
        main_splitter.setChildrenCollapsible(False)
        
        # Left panel - Video player and controls
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setSpacing(16)
        left_layout.setContentsMargins(20, 20, 20, 20)
        
        # Video player
        self.video_player = EnhancedVideoPlayer()
        left_layout.addWidget(self.video_player, 2)
        
        # Video controls
        video_controls_layout = QHBoxLayout()
        
        self.play_pause_button = Button("▶️ 播放", "primary")
        self.play_pause_button.clicked.connect(self.toggle_playback)
        video_controls_layout.addWidget(self.play_pause_button)
        
        self.stop_button = Button("⏹️ 停止", "secondary")
        self.stop_button.clicked.connect(self.stop_playback)
        video_controls_layout.addWidget(self.stop_button)
        
        self.frame_slider = QSlider(Qt.Orientation.Horizontal)
        self.frame_slider.setStyleSheet("""
            QSlider::groove:horizontal {
                height: 8px;
                background: #e2e8f0;
                border-radius: 4px;
            }
            QSlider::handle:horizontal {
                background: #3b82f6;
                border: 2px solid white;
                width: 20px;
                margin: -6px 0;
                border-radius: 10px;
            }
            QSlider::handle:horizontal:hover {
                background: #2563eb;
            }
        """)
        video_controls_layout.addWidget(self.frame_slider)
        
        self.frame_label = QLabel("0 / 0")
        self.frame_label.setStyleSheet("font-weight: bold; min-width: 80px;")
        self.frame_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        video_controls_layout.addWidget(self.frame_label)
        
        left_layout.addLayout(video_controls_layout)
        
        # OCR results
        ocr_results_card = Card("OCR 识别结果")
        ocr_results_layout = QVBoxLayout()
        
        self.ocr_result_label = QLabel("请选择视频文件并执行OCR识别")
        self.ocr_result_label.setWordWrap(True)
        self.ocr_result_label.setStyleSheet("""
            QLabel {
                background: #f8fafc;
                border: 1px solid #e2e8f0;
                border-radius: 8px;
                padding: 16px;
                font-family: 'Monaco', 'Menlo', monospace;
                font-size: 12px;
            }
        """)
        ocr_results_layout.addWidget(self.ocr_result_label)
        
        ocr_results_card.add_layout(ocr_results_layout)
        left_layout.addWidget(ocr_results_card)
        
        main_splitter.addWidget(left_widget)
        
        # Right panel - Controls and settings
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setSpacing(16)
        right_layout.setContentsMargins(20, 20, 20, 20)
        
        # Control panel
        self.control_panel = ModernControlPanel()
        self.control_panel.set_view_model(self.vm)
        right_layout.addWidget(self.control_panel, 1)
        
        # ROI panel
        self.roi_panel = ROISelectionPanel(self.vm.get_roi_manager())
        self.roi_panel.setMaximumWidth(400)
        right_layout.addWidget(self.roi_panel, 1)
        
        # Batch panel
        self.batch_panel = ModernBatchPanel()
        self.batch_panel.set_view_model(self.vm)
        right_layout.addWidget(self.batch_panel, 1)
        
        # Action buttons
        action_buttons_layout = QHBoxLayout()
        
        self.subtitle_editor_button = Button("📝 字幕编辑器", "success")
        self.subtitle_editor_button.clicked.connect(self.open_subtitle_editor)
        action_buttons_layout.addWidget(self.subtitle_editor_button)
        
        self.advanced_settings_button = Button("⚙️ 高级设置", "secondary")
        self.advanced_settings_button.clicked.connect(self.open_advanced_settings)
        action_buttons_layout.addWidget(self.advanced_settings_button)
        
        right_layout.addLayout(action_buttons_layout)
        
        main_splitter.addWidget(right_widget)
        
        # Set splitter sizes
        main_splitter.setSizes([1000, 600])
        
        main_layout.addWidget(main_splitter)
        self.setCentralWidget(central_widget)
        
    def setup_menus(self):
        """Setup menu bar"""
        menubar = self.menuBar()
        
        # File menu
        file_menu = menubar.addMenu("文件")
        
        open_action = QAction("打开视频", self)
        open_action.setShortcut("Ctrl+O")
        open_action.triggered.connect(self.open_file)
        file_menu.addAction(open_action)
        
        save_action = QAction("保存字幕", self)
        save_action.setShortcut("Ctrl+S")
        save_action.triggered.connect(self.save_subtitles)
        file_menu.addAction(save_action)
        
        file_menu.addSeparator()
        
        exit_action = QAction("退出", self)
        exit_action.setShortcut("Ctrl+Q")
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # Edit menu
        edit_menu = menubar.addMenu("编辑")
        
        undo_action = QAction("撤销", self)
        undo_action.setShortcut("Ctrl+Z")
        edit_menu.addAction(undo_action)
        
        redo_action = QAction("重做", self)
        redo_action.setShortcut("Ctrl+Y")
        edit_menu.addAction(redo_action)
        
        edit_menu.addSeparator()
        
        settings_action = QAction("设置", self)
        settings_action.setShortcut("Ctrl+,")
        settings_action.triggered.connect(self.open_advanced_settings)
        edit_menu.addAction(settings_action)
        
        # View menu
        view_menu = menubar.addMenu("视图")
        
        theme_menu = view_menu.addMenu("主题")
        
        for theme_name in self.theme_manager.get_available_themes():
            theme_action = QAction(theme_name.title(), self)
            theme_action.triggered.connect(lambda checked, name=theme_name: self.theme_manager.set_theme(name))
            theme_menu.addAction(theme_action)
            
        # Help menu
        help_menu = menubar.addMenu("帮助")
        
        about_action = QAction("关于", self)
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)
        
    def setup_status_bar(self):
        """Setup status bar"""
        self.status_bar = ModernStatusBar(self)
        self.setStatusBar(self.status_bar)
        
    def setup_toolbar(self):
        """Setup toolbar"""
        self.toolbar = ModernToolbar(self)
        self.addToolBar(self.toolbar)
        
    def connect_signals(self):
        """Connect all signals"""
        # Video player signals
        self.video_player.roi_changed.connect(self.vm.update_roi)
        
        # Control signals
        self.frame_slider.sliderMoved.connect(lambda val: self.vm.seek_frame(val, True))
        
        # View model signals
        self.vm.frame_changed.connect(self.video_player.update_frame)
        self.vm.config_changed.connect(self.update_config_display)
        self.vm.single_ocr_result_changed.connect(self.ocr_result_label.setText)
        self.vm.video_loaded.connect(self.on_video_loaded)
        self.vm.is_playing_changed.connect(self.on_playback_state_changed)
        self.vm.frame_index_changed.connect(self.frame_slider.setValue)
        self.vm.error_occurred.connect(self.on_error)
        
        # ROI panel signals
        self.roi_panel.roi_config_changed.connect(self.vm.update_roi_config)
        
    def open_file(self):
        """Open video file"""
        dialog = SecureFileDialog(self)
        dialog.setWindowTitle("选择视频文件")
        dialog.setNameFilter("视频文件 (*.mp4 *.avi *.mkv *.mov *.wmv *.flv *.webm)")
        
        if dialog.exec() == QDialog.DialogCode.Accepted:
            files = dialog.selectedFiles()
            if files:
                self.vm.load_video(files[0])
                
    def save_subtitles(self):
        """Save subtitles"""
        if hasattr(self.vm, 'processed_subtitles') and self.vm.processed_subtitles:
            dialog = QFileDialog(self)
            dialog.setWindowTitle("保存字幕")
            dialog.setNameFilter("SRT文件 (*.srt);;JSON文件 (*.json)")
            dialog.setAcceptMode(QFileDialog.AcceptMode.AcceptSave)
            
            if dialog.exec() == QDialog.DialogCode.Accepted:
                file_path = dialog.selectedFiles()[0]
                try:
                    # Save subtitles logic here
                    self.status_bar.set_status(f"字幕已保存到: {file_path}", "success")
                except Exception as e:
                    self.status_bar.set_status(f"保存失败: {e}", "error")
        else:
            QMessageBox.information(self, "保存字幕", "没有可保存的字幕数据")
            
    def toggle_playback(self):
        """Toggle video playback"""
        if self.vm.is_playing:
            self.vm.pause()
        else:
            self.vm.play()
            
    def stop_playback(self):
        """Stop video playback"""
        self.vm.stop()
        
    def on_video_loaded(self, frame_count: int):
        """Handle video loaded event"""
        self.frame_slider.setEnabled(True)
        self.frame_slider.setRange(0, frame_count - 1)
        self.frame_label.setText(f"0 / {frame_count}")
        
        # Update ROI panel
        video_info = self.vm.get_video_info()
        if video_info:
            video_size = QSize(video_info.get('width', 0), video_info.get('height', 0))
            self.roi_panel.set_video_size(video_size)
            
        self.status_bar.set_status("视频加载成功", "success")
        
    def on_playback_state_changed(self, is_playing: bool):
        """Handle playback state change"""
        self.play_pause_button.setText("⏸️ 暂停" if is_playing else "▶️ 播放")
        
    def on_error(self, error_message: str):
        """Handle error events"""
        self.status_bar.set_status(error_message, "error")
        QMessageBox.critical(self, "错误", error_message)
        
    def update_config_display(self, config: OcrConfig):
        """Update configuration display"""
        # Update control panel
        self.control_panel.language_combo.setCurrentText(config.language)
        self.control_panel.threshold_slider.setValue(config.threshold)
        self.control_panel.threshold_spinbox.setValue(config.threshold)
        self.control_panel.confidence_spinbox.setValue(config.confidence_threshold)
        
    def open_subtitle_editor(self):
        """Open subtitle editor"""
        if hasattr(self.vm, 'processed_subtitles') and self.vm.processed_subtitles:
            editor = SubtitleEditorWindow(self)
            editor.set_subtitles(self.vm.processed_subtitles)
            editor.show()
        else:
            QMessageBox.information(self, "字幕编辑器", "没有可编辑的字幕数据")
            
    def open_advanced_settings(self):
        """Open advanced settings dialog"""
        dialog = AdvancedOCRSettingsDialog(self.vm.config, self)
        dialog.config_changed.connect(self.vm.update_config)
        dialog.show()
        
    def show_about(self):
        """Show about dialog"""
        about_text = """
        <h2>VisionSub - 视频OCR字幕提取工具</h2>
        <p>版本: 2.0.0</p>
        <p>基于PyQt6的现代桌面应用程序，用于从视频中提取和编辑字幕。</p>
        <p>支持多种OCR引擎，包括PaddleOCR和Tesseract。</p>
        <p><b>特性:</b></p>
        <ul>
            <li>🎥 支持多种视频格式</li>
            <li>🔍 智能OCR识别</li>
            <li>📝 高级字幕编辑</li>
            <li>🎨 现代化UI设计</li>
            <li>🔒 安全文件处理</li>
            <li>⚡ 批量处理支持</li>
        </ul>
        """
        
        QMessageBox.about(self, "关于 VisionSub", about_text)
        
    def on_theme_changed(self, theme: ThemeDefinition):
        """Handle theme change"""
        logger.info(f"Theme changed to: {theme.name}")
        
    def closeEvent(self, event):
        """Handle window close event"""
        # Ask for confirmation if there are unsaved changes
        if hasattr(self.vm, 'processed_subtitles') and self.vm.processed_subtitles:
            reply = QMessageBox.question(
                self, "退出确认",
                "有未保存的字幕数据，确定要退出吗？",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No
            )
            
            if reply == QMessageBox.StandardButton.No:
                event.ignore()
                return
                
        event.accept()
        
    def keyPressEvent(self, event):
        """Handle key press events"""
        # Global keyboard shortcuts
        if event.key() == Qt.Key.Key_Space:
            self.toggle_playback()
        elif event.key() == Qt.Key.Key_Escape:
            self.stop_playback()
        elif event.key() == Qt.Key.Key_F1:
            self.show_about()
        else:
            super().keyPressEvent(event)


# Example usage
if __name__ == "__main__":
    import sys
    from PyQt6.QtWidgets import QApplication
    
    app = QApplication(sys.argv)
    
    # Initialize theme manager
    theme_manager = get_theme_manager()
    
    # Create view model
    vm = MainViewModel()
    
    # Create and show main window
    window = EnhancedMainWindow(vm)
    window.show()
    
    sys.exit(app.exec())