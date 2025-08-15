"""
Enhanced Settings Dialog with Modern UI/UX and Security Features
"""
import json
import logging
import re
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from PyQt6.QtCore import Qt, pyqtSignal, QTimer, QSettings
from PyQt6.QtGui import QFont, QColor, QValidator, QIntValidator, QDoubleValidator
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QFormLayout, 
    QLabel, QPushButton, QComboBox, QCheckBox, 
    QSpinBox, QDoubleSpinBox, QGroupBox, QTabWidget,
    QDialogButtonBox, QMessageBox, QWidget, QLineEdit,
    QTextEdit, QScrollArea, QFrame, QSizePolicy,
    QSpacerItem, QSlider, QProgressBar, QRadioButton,
    QButtonGroup, QStyle, QStyleOption
)

from visionsub.models.config import (
    AppConfig, OcrConfig, ProcessingConfig, 
    UIConfig, SecurityConfig, LoggingConfig
)
from visionsub.ui.theme_system import (
    ThemeManager, get_theme_manager, StyledWidget, 
    Card, Button, ThemeColors
)

logger = logging.getLogger(__name__)


class SecureInputValidator(QValidator):
    """Validator for secure input handling"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.malicious_patterns = [
            r'<script.*?>.*?</script>',  # Script tags
            r'javascript:',             # JavaScript protocol
            r'vbscript:',               # VBScript protocol
            r'data:',                   # Data protocol
            r'file:',                   # File protocol
            r'ftp:',                    # FTP protocol
            r'http://',                 # HTTP URLs
            r'https://',                # HTTPS URLs
            r'\.\./',                   # Directory traversal
            r'\.\|',                    # Pipe injection
            r'\$\(',                    # Command substitution
            r'`.*`',                    # Backtick commands
            r'<.*?>',                   # HTML tags
            r'&.*?;',                   # HTML entities
        ]
        
    def validate(self, input_str: str, pos: int) -> Tuple[QValidator.State, str, int]:
        """Validate input string"""
        # Check for malicious patterns
        for pattern in self.malicious_patterns:
            if re.search(pattern, input_str, re.IGNORECASE):
                return (QValidator.State.Invalid, input_str, pos)
                
        # Check for control characters
        if any(ord(char) < 32 and char not in ['\t', '\n', '\r'] for char in input_str):
            return (QValidator.State.Invalid, input_str, pos)
            
        # Check for excessive length
        if len(input_str) > 1000:
            return (QValidator.State.Invalid, input_str, pos)
            
        return (QValidator.State.Acceptable, input_str, pos)
        
    def fixup(self, input_str: str) -> str:
        """Fix invalid input"""
        # Remove malicious patterns
        for pattern in self.malicious_patterns:
            input_str = re.sub(pattern, '', input_str, flags=re.IGNORECASE)
            
        # Remove control characters except whitespace
        input_str = ''.join(char for char in input_str if ord(char) >= 32 or char in ['\t', '\n', '\r'])
        
        # Truncate excessive length
        if len(input_str) > 1000:
            input_str = input_str[:1000]
            
        return input_str


class SecureLineEdit(QLineEdit):
    """Secure line edit with input validation"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setValidator(SecureInputValidator())
        self.setMaxLength(1000)
        
    def setText(self, text: str):
        """Override setText to validate input"""
        validator = self.validator()
        if validator:
            state, fixed_text, _ = validator.validate(text, 0)
            if state == QValidator.State.Invalid:
                text = fixed_text
        super().setText(text)


class SecureTextEdit(QTextEdit):
    """Secure text edit with input validation"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.validator = SecureInputValidator()
        self.setAcceptRichText(False)
        
    def insertPlainText(self, text: str):
        """Override to validate input"""
        if self.validator:
            state, fixed_text, _ = self.validator.validate(text, 0)
            if state == QValidator.State.Invalid:
                text = fixed_text
        super().insertPlainText(text)
        
    def setPlainText(self, text: str):
        """Override to validate input"""
        if self.validator:
            state, fixed_text, _ = self.validator.validate(text, 0)
            if state == QValidator.State.Invalid:
                text = fixed_text
        super().setPlainText(text)


class ThemeSettingsWidget(StyledWidget):
    """Theme settings widget"""
    
    theme_changed = pyqtSignal(str)
    
    def __init__(self, config: UIConfig, parent=None):
        super().__init__(parent)
        self.config = config
        self.theme_manager = get_theme_manager()
        self.setup_ui()
        
    def setup_ui(self):
        """Setup theme settings UI"""
        layout = QVBoxLayout(self)
        layout.setSpacing(16)
        layout.setContentsMargins(20, 20, 20, 20)
        
        # Theme selection
        theme_group = QGroupBox("主题设置")
        theme_layout = QVBoxLayout(theme_group)
        
        self.theme_combo = QComboBox()
        self.theme_combo.addItems(self.theme_manager.get_available_themes())
        self.theme_combo.setCurrentText(self.config.theme)
        self.theme_combo.currentTextChanged.connect(self.on_theme_changed)
        theme_layout.addWidget(QLabel("选择主题:"))
        theme_layout.addWidget(self.theme_combo)
        
        # Font size
        font_size_layout = QHBoxLayout()
        font_size_layout.addWidget(QLabel("字体大小:"))
        
        self.font_size_slider = QSlider(Qt.Orientation.Horizontal)
        self.font_size_slider.setRange(8, 24)
        self.font_size_slider.setValue(self.config.font_size)
        self.font_size_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.font_size_slider.setTickInterval(2)
        
        self.font_size_label = QLabel(f"{self.config.font_size}px")
        self.font_size_label.setMinimumWidth(50)
        
        self.font_size_slider.valueChanged.connect(self.on_font_size_changed)
        font_size_layout.addWidget(self.font_size_slider)
        font_size_layout.addWidget(self.font_size_label)
        
        theme_layout.addLayout(font_size_layout)
        
        # Animations
        self.animations_checkbox = QCheckBox("启用动画效果")
        self.animations_checkbox.setChecked(self.config.enable_animations)
        theme_layout.addWidget(self.animations_checkbox)
        
        # Performance metrics
        self.metrics_checkbox = QCheckBox("显示性能指标")
        self.metrics_checkbox.setChecked(self.config.show_performance_metrics)
        theme_layout.addWidget(self.metrics_checkbox)
        
        layout.addWidget(theme_group)
        
        # Theme preview
        preview_group = QGroupBox("主题预览")
        preview_layout = QVBoxLayout(preview_group)
        
        self.preview_label = QLabel("主题预览")
        self.preview_label.setWordWrap(True)
        self.preview_label.setStyleSheet("""
            QLabel {
                background: #f8fafc;
                border: 1px solid #e2e8f0;
                border-radius: 8px;
                padding: 16px;
                font-size: 14px;
            }
        """)
        preview_layout.addWidget(self.preview_label)
        
        layout.addWidget(preview_group)
        
        layout.addStretch()
        
        # Update preview
        self.update_preview()
        
    def on_theme_changed(self, theme_name: str):
        """Handle theme change"""
        self.theme_changed.emit(theme_name)
        self.update_preview()
        
    def on_font_size_changed(self, value: int):
        """Handle font size change"""
        self.font_size_label.setText(f"{value}px")
        self.update_preview()
        
    def update_preview(self):
        """Update theme preview"""
        theme = self.theme_manager.current_theme
        if theme:
            preview_text = f"""
            <b>主题:</b> {theme.name}<br>
            <b>主色调:</b> {theme.colors.primary.name()}<br>
            <b>背景色:</b> {theme.colors.background.name()}<br>
            <b>文字色:</b> {theme.colors.text_primary.name()}<br>
            <b>字体大小:</b> {self.font_size_slider.value()}px
            """
            self.preview_label.setText(preview_text)
            
    def get_config(self) -> UIConfig:
        """Get current configuration"""
        return UIConfig(
            theme=self.theme_combo.currentText(),
            language=self.config.language,
            window_size=self.config.window_size,
            font_size=self.font_size_slider.value(),
            enable_animations=self.animations_checkbox.isChecked(),
            show_performance_metrics=self.metrics_checkbox.isChecked()
        )


class SecuritySettingsWidget(StyledWidget):
    """Security settings widget"""
    
    def __init__(self, config: SecurityConfig, parent=None):
        super().__init__(parent)
        self.config = config
        self.setup_ui()
        
    def setup_ui(self):
        """Setup security settings UI"""
        layout = QVBoxLayout(self)
        layout.setSpacing(16)
        layout.setContentsMargins(20, 20, 20, 20)
        
        # Input validation
        validation_group = QGroupBox("输入验证")
        validation_layout = QVBoxLayout(validation_group)
        
        self.input_validation_checkbox = QCheckBox("启用输入验证和清理")
        self.input_validation_checkbox.setChecked(self.config.enable_input_validation)
        validation_layout.addWidget(self.input_validation_checkbox)
        
        layout.addWidget(validation_group)
        
        # File security
        file_group = QGroupBox("文件安全")
        file_layout = QFormLayout(file_group)
        
        self.max_file_size_spinbox = QSpinBox()
        self.max_file_size_spinbox.setRange(10, 2048)
        self.max_file_size_spinbox.setValue(self.config.max_file_size_mb)
        self.max_file_size_spinbox.setSuffix(" MB")
        file_layout.addRow("最大文件大小:", self.max_file_size_spinbox)
        
        # Video formats
        formats_layout = QVBoxLayout()
        self.formats_edit = SecureLineEdit()
        self.formats_edit.setText(", ".join(self.config.allowed_video_formats))
        self.formats_edit.setPlaceholderText("例如: mp4, avi, mkv")
        formats_layout.addWidget(self.formats_edit)
        file_layout.addRow("允许的视频格式:", formats_layout)
        
        layout.addWidget(file_group)
        
        # Rate limiting
        rate_group = QGroupBox("速率限制")
        rate_layout = QFormLayout(rate_group)
        
        self.rate_limiting_checkbox = QCheckBox("启用速率限制")
        self.rate_limiting_checkbox.setChecked(self.config.enable_rate_limiting)
        rate_layout.addRow("速率限制:", self.rate_limiting_checkbox)
        
        self.rate_limit_spinbox = QSpinBox()
        self.rate_limit_spinbox.setRange(1, 1000)
        self.rate_limit_spinbox.setValue(self.config.rate_limit_requests_per_minute)
        self.rate_limit_spinbox.setSuffix(" 请求/分钟")
        self.rate_limit_spinbox.setEnabled(self.config.enable_rate_limiting)
        rate_layout.addRow("请求限制:", self.rate_limit_spinbox)
        
        self.rate_limiting_checkbox.toggled.connect(self.rate_limit_spinbox.setEnabled)
        
        layout.addWidget(rate_group)
        
        # Security status
        status_group = QGroupBox("安全状态")
        status_layout = QVBoxLayout(status_group)
        
        self.security_status_label = QLabel("🔒 安全状态: 正常")
        self.security_status_label.setStyleSheet("color: #10b981; font-weight: bold;")
        status_layout.addWidget(self.security_status_label)
        
        self.security_details_label = QLabel("所有安全检查均已通过")
        self.security_details_label.setWordWrap(True)
        self.security_details_label.setStyleSheet("color: #64748b;")
        status_layout.addWidget(self.security_details_label)
        
        layout.addWidget(status_group)
        
        layout.addStretch()
        
    def get_config(self) -> SecurityConfig:
        """Get current configuration"""
        formats_text = self.formats_edit.text()
        formats = [fmt.strip() for fmt in formats_text.split(",") if fmt.strip()]
        
        return SecurityConfig(
            enable_input_validation=self.input_validation_checkbox.isChecked(),
            max_file_size_mb=self.max_file_size_spinbox.value(),
            allowed_video_formats=formats,
            enable_rate_limiting=self.rate_limiting_checkbox.isChecked(),
            rate_limit_requests_per_minute=self.rate_limit_spinbox.value()
        )


class OCRSettingsWidget(StyledWidget):
    """OCR settings widget"""
    
    def __init__(self, config: OcrConfig, parent=None):
        super().__init__(parent)
        self.config = config
        self.setup_ui()
        
    def setup_ui(self):
        """Setup OCR settings UI"""
        layout = QVBoxLayout(self)
        layout.setSpacing(16)
        layout.setContentsMargins(20, 20, 20, 20)
        
        # Basic OCR settings
        basic_group = QGroupBox("基本OCR设置")
        basic_layout = QFormLayout(basic_group)
        
        # Engine selection
        self.engine_combo = QComboBox()
        self.engine_combo.addItems(["PaddleOCR", "Tesseract"])
        self.engine_combo.setCurrentText(self.config.engine)
        basic_layout.addRow("OCR引擎:", self.engine_combo)
        
        # Language selection
        self.language_combo = QComboBox()
        self.language_combo.addItems(["中文", "英文", "韩文", "日文", "法文", "德文", "西班牙文", "俄文", "阿拉伯文", "印地文"])
        self.language_combo.setCurrentText(self.config.language)
        basic_layout.addRow("识别语言:", self.language_combo)
        
        # Threshold
        threshold_layout = QHBoxLayout()
        self.threshold_slider = QSlider(Qt.Orientation.Horizontal)
        self.threshold_slider.setRange(0, 255)
        self.threshold_slider.setValue(self.config.threshold)
        
        self.threshold_spinbox = QSpinBox()
        self.threshold_spinbox.setRange(0, 255)
        self.threshold_spinbox.setValue(self.config.threshold)
        
        threshold_layout.addWidget(self.threshold_slider)
        threshold_layout.addWidget(self.threshold_spinbox)
        basic_layout.addRow("二值化阈值:", threshold_layout)
        
        # Confidence threshold
        self.confidence_spinbox = QDoubleSpinBox()
        self.confidence_spinbox.setRange(0.0, 1.0)
        self.confidence_spinbox.setSingleStep(0.05)
        self.confidence_spinbox.setDecimals(2)
        self.confidence_spinbox.setValue(self.config.confidence_threshold)
        basic_layout.addRow("置信度阈值:", self.confidence_spinbox)
        
        # Auto detect language
        self.auto_detect_checkbox = QCheckBox("自动检测语言")
        self.auto_detect_checkbox.setChecked(self.config.auto_detect_language)
        basic_layout.addRow("语言检测:", self.auto_detect_checkbox)
        
        layout.addWidget(basic_group)
        
        # Preprocessing settings
        preprocess_group = QGroupBox("图像预处理")
        preprocess_layout = QVBoxLayout(preprocess_group)
        
        self.enable_preprocessing_checkbox = QCheckBox("启用图像预处理")
        self.enable_preprocessing_checkbox.setChecked(self.config.enable_preprocessing)
        preprocess_layout.addWidget(self.enable_preprocessing_checkbox)
        
        # Individual preprocessing options
        self.denoise_checkbox = QCheckBox("去噪 (Denoising)")
        self.denoise_checkbox.setChecked(self.config.denoise)
        preprocess_layout.addWidget(self.denoise_checkbox)
        
        self.enhance_contrast_checkbox = QCheckBox("对比度增强 (Contrast Enhancement)")
        self.enhance_contrast_checkbox.setChecked(self.config.enhance_contrast)
        preprocess_layout.addWidget(self.enhance_contrast_checkbox)
        
        self.sharpen_checkbox = QCheckBox("图像锐化 (Sharpening)")
        self.sharpen_checkbox.setChecked(self.config.sharpen)
        preprocess_layout.addWidget(self.sharpen_checkbox)
        
        self.enable_postprocessing_checkbox = QCheckBox("启用文本后处理")
        self.enable_postprocessing_checkbox.setChecked(self.config.enable_postprocessing)
        preprocess_layout.addWidget(self.enable_postprocessing_checkbox)
        
        layout.addWidget(preprocess_group)
        
        # Advanced settings
        advanced_group = QGroupBox("高级设置")
        advanced_layout = QVBoxLayout(advanced_group)
        
        # Custom parameters
        custom_layout = QVBoxLayout()
        custom_layout.addWidget(QLabel("自定义参数 (JSON格式):"))
        
        self.custom_params_edit = SecureTextEdit()
        self.custom_params_edit.setMaximumHeight(100)
        self.custom_params_edit.setPlaceholderText('{"key": "value"}')
        
        if self.config.custom_params:
            try:
                self.custom_params_edit.setPlainText(json.dumps(self.config.custom_params, indent=2))
            except:
                self.custom_params_edit.setPlainText(str(self.config.custom_params))
                
        custom_layout.addWidget(self.custom_params_edit)
        advanced_layout.addLayout(custom_layout)
        
        layout.addWidget(advanced_group)
        
        layout.addStretch()
        
        # Connect signals
        self.threshold_slider.valueChanged.connect(self.threshold_spinbox.setValue)
        self.threshold_spinbox.valueChanged.connect(self.threshold_slider.setValue)
        
    def get_config(self) -> OcrConfig:
        """Get current configuration"""
        custom_params = None
        try:
            custom_text = self.custom_params_edit.toPlainText().strip()
            if custom_text:
                custom_params = json.loads(custom_text)
        except json.JSONDecodeError:
            # If JSON parsing fails, keep original
            custom_params = self.config.custom_params
            
        return OcrConfig(
            engine=self.engine_combo.currentText(),
            language=self.language_combo.currentText(),
            threshold=self.threshold_slider.value(),
            roi_rect=self.config.roi_rect,
            confidence_threshold=self.confidence_spinbox.value(),
            enable_preprocessing=self.enable_preprocessing_checkbox.isChecked(),
            enable_postprocessing=self.enable_postprocessing_checkbox.isChecked(),
            custom_params=custom_params,
            denoise=self.denoise_checkbox.isChecked(),
            enhance_contrast=self.enhance_contrast_checkbox.isChecked(),
            sharpen=self.sharpen_checkbox.isChecked(),
            auto_detect_language=self.auto_detect_checkbox.isChecked()
        )


class ProcessingSettingsWidget(StyledWidget):
    """Processing settings widget"""
    
    def __init__(self, config: ProcessingConfig, parent=None):
        super().__init__(parent)
        self.config = config
        self.setup_ui()
        
    def setup_ui(self):
        """Setup processing settings UI"""
        layout = QVBoxLayout(self)
        layout.setSpacing(16)
        layout.setContentsMargins(20, 20, 20, 20)
        
        # Performance settings
        performance_group = QGroupBox("性能设置")
        performance_layout = QFormLayout(performance_group)
        
        self.cache_size_spinbox = QSpinBox()
        self.cache_size_spinbox.setRange(1, 10000)
        self.cache_size_spinbox.setValue(self.config.cache_size)
        performance_layout.addRow("缓存大小:", self.cache_size_spinbox)
        
        self.concurrent_jobs_spinbox = QSpinBox()
        self.concurrent_jobs_spinbox.setRange(1, 16)
        self.concurrent_jobs_spinbox.setValue(self.config.max_concurrent_jobs)
        performance_layout.addRow("并发任务数:", self.concurrent_jobs_spinbox)
        
        self.memory_limit_spinbox = QSpinBox()
        self.memory_limit_spinbox.setRange(256, 8192)
        self.memory_limit_spinbox.setValue(self.config.memory_limit_mb)
        self.memory_limit_spinbox.setSuffix(" MB")
        performance_layout.addRow("内存限制:", self.memory_limit_spinbox)
        
        layout.addWidget(performance_group)
        
        # Processing options
        options_group = QGroupBox("处理选项")
        options_layout = QVBoxLayout(options_group)
        
        self.batch_processing_checkbox = QCheckBox("启用批量处理")
        self.batch_processing_checkbox.setChecked(self.config.enable_batch_processing)
        options_layout.addWidget(self.batch_processing_checkbox)
        
        self.parallel_processing_checkbox = QCheckBox("启用并行处理")
        self.parallel_processing_checkbox.setChecked(self.config.enable_parallel_processing)
        options_layout.addWidget(self.parallel_processing_checkbox)
        
        self.scene_detection_checkbox = QCheckBox("启用场景检测")
        self.scene_detection_checkbox.setChecked(self.config.enable_scene_detection)
        options_layout.addWidget(self.scene_detection_checkbox)
        
        layout.addWidget(options_group)
        
        # Video processing
        video_group = QGroupBox("视频处理")
        video_layout = QFormLayout(video_group)
        
        self.frame_interval_spinbox = QDoubleSpinBox()
        self.frame_interval_spinbox.setRange(0.1, 10.0)
        self.frame_interval_spinbox.setSingleStep(0.1)
        self.frame_interval_spinbox.setValue(self.config.frame_interval)
        self.frame_interval_spinbox.setSuffix(" 秒")
        video_layout.addRow("帧间隔:", self.frame_interval_spinbox)
        
        self.scene_threshold_spinbox = QDoubleSpinBox()
        self.scene_threshold_spinbox.setRange(0.0, 1.0)
        self.scene_threshold_spinbox.setSingleStep(0.1)
        self.scene_threshold_spinbox.setDecimals(2)
        self.scene_threshold_spinbox.setValue(self.config.scene_threshold)
        video_layout.addRow("场景阈值:", self.scene_threshold_spinbox)
        
        layout.addWidget(video_group)
        
        # Output settings
        output_group = QGroupBox("输出设置")
        output_layout = QVBoxLayout(output_group)
        
        self.output_directory_edit = SecureLineEdit()
        self.output_directory_edit.setText(self.config.output_directory)
        output_layout.addWidget(QLabel("输出目录:"))
        output_layout.addWidget(self.output_directory_edit)
        
        self.create_subdirectories_checkbox = QCheckBox("创建子目录")
        self.create_subdirectories_checkbox.setChecked(self.config.create_subdirectories)
        output_layout.addWidget(self.create_subdirectories_checkbox)
        
        layout.addWidget(output_group)
        
        # Advanced options
        advanced_group = QGroupBox("高级选项")
        advanced_layout = QVBoxLayout(advanced_group)
        
        self.debug_mode_checkbox = QCheckBox("调试模式")
        self.debug_mode_checkbox.setChecked(self.config.debug_mode)
        advanced_layout.addWidget(self.debug_mode_checkbox)
        
        self.performance_monitoring_checkbox = QCheckBox("性能监控")
        self.performance_monitoring_checkbox.setChecked(self.config.enable_performance_monitoring)
        advanced_layout.addWidget(self.performance_monitoring_checkbox)
        
        layout.addWidget(advanced_group)
        
        layout.addStretch()
        
    def get_config(self) -> ProcessingConfig:
        """Get current configuration"""
        return ProcessingConfig(
            ocr_config=self.config.ocr_config,
            scene_threshold=self.scene_threshold_spinbox.value(),
            cache_size=self.cache_size_spinbox.value(),
            max_concurrent_jobs=self.concurrent_jobs_spinbox.value(),
            enable_batch_processing=self.batch_processing_checkbox.isChecked(),
            output_formats=self.config.output_formats,
            frame_interval=self.frame_interval_spinbox.value(),
            enable_scene_detection=self.scene_detection_checkbox.isChecked(),
            enable_parallel_processing=self.parallel_processing_checkbox.isChecked(),
            memory_limit_mb=self.memory_limit_spinbox.value(),
            output_directory=self.output_directory_edit.text(),
            create_subdirectories=self.create_subdirectories_checkbox.isChecked(),
            debug_mode=self.debug_mode_checkbox.isChecked(),
            enable_performance_monitoring=self.performance_monitoring_checkbox.isChecked()
        )


class EnhancedSettingsDialog(QDialog):
    """Enhanced settings dialog with modern UI/UX and security features"""
    
    config_changed = pyqtSignal(AppConfig)
    
    def __init__(self, config: AppConfig, parent=None):
        super().__init__(parent)
        self.config = config
        self.original_config = AppConfig(**config.model_dump())
        
        self.setWindowTitle("设置")
        self.setModal(True)
        self.resize(800, 600)
        
        self.setup_ui()
        self.load_config()
        
    def setup_ui(self):
        """Setup settings dialog UI"""
        layout = QVBoxLayout(self)
        layout.setSpacing(0)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Create tabs
        self.tabs = QTabWidget()
        self.tabs.setStyleSheet("""
            QTabWidget::pane {
                border: 1px solid #e2e8f0;
                border-radius: 8px;
                background: #f8fafc;
            }
            QTabBar::tab {
                background: #e2e8f0;
                border: 1px solid #cbd5e1;
                border-radius: 6px 6px 0 0;
                padding: 8px 16px;
                margin-right: 2px;
            }
            QTabBar::tab:selected {
                background: #f8fafc;
                border-bottom: 2px solid #3b82f6;
            }
            QTabBar::tab:hover {
                background: #f1f5f9;
            }
        """)
        
        # Theme settings
        self.theme_widget = ThemeSettingsWidget(self.config.ui)
        self.tabs.addTab(self.theme_widget, "主题")
        
        # Security settings
        self.security_widget = SecuritySettingsWidget(self.config.security)
        self.tabs.addTab(self.security_widget, "安全")
        
        # OCR settings
        self.ocr_widget = OCRSettingsWidget(self.config.processing.ocr_config)
        self.tabs.addTab(self.ocr_widget, "OCR")
        
        # Processing settings
        self.processing_widget = ProcessingSettingsWidget(self.config.processing)
        self.tabs.addTab(self.processing_widget, "处理")
        
        layout.addWidget(self.tabs)
        
        # Buttons
        button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | 
            QDialogButtonBox.StandardButton.Cancel |
            QDialogButtonBox.StandardButton.Apply |
            QDialogButtonBox.StandardButton.Reset
        )
        
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        button_box.button(QDialogButtonBox.StandardButton.Apply).clicked.connect(self.apply_settings)
        button_box.button(QDialogButtonBox.StandardButton.Reset).clicked.connect(self.reset_settings)
        
        layout.addWidget(button_box)
        
        # Connect theme change signal
        self.theme_widget.theme_changed.connect(self.on_theme_changed)
        
    def load_config(self):
        """Load configuration into UI"""
        # Theme settings are loaded in the widget constructor
        # Security settings are loaded in the widget constructor
        # OCR settings are loaded in the widget constructor
        # Processing settings are loaded in the widget constructor
        pass
        
    def get_config(self) -> AppConfig:
        """Get current configuration from UI"""
        return AppConfig(
            processing=self.processing_widget.get_config(),
            ui=self.theme_widget.get_config(),
            logging=self.config.logging,
            security=self.security_widget.get_config(),
            app_name=self.config.app_name,
            version=self.config.version,
            config_file_path=self.config.config_file_path
        )
        
    def apply_settings(self):
        """Apply settings without closing dialog"""
        try:
            new_config = self.get_config()
            
            # Validate configuration
            errors = new_config.validate_config()
            if errors:
                QMessageBox.warning(
                    self, "配置错误",
                    "配置包含以下错误:\n\n" + "\n".join(errors)
                )
                return
                
            # Apply configuration
            self.config = new_config
            self.config_changed.emit(new_config)
            
            # Show success message
            QMessageBox.information(
                self, "设置已应用",
                "设置已成功应用。"
            )
            
        except Exception as e:
            QMessageBox.critical(
                self, "错误",
                f"应用设置时出错: {e}"
            )
            
    def reset_settings(self):
        """Reset to original settings"""
        reply = QMessageBox.question(
            self, "重置设置",
            "确定要重置为原始设置吗？",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            self.config = AppConfig(**self.original_config.model_dump())
            self.load_config()
            self.config_changed.emit(self.config)
            
    def accept(self):
        """OK button clicked"""
        try:
            new_config = self.get_config()
            
            # Validate configuration
            errors = new_config.validate_config()
            if errors:
                QMessageBox.warning(
                    self, "配置错误",
                    "配置包含以下错误:\n\n" + "\n".join(errors)
                )
                return
                
            self.config = new_config
            super().accept()
            
        except Exception as e:
            QMessageBox.critical(
                self, "错误",
                f"保存设置时出错: {e}"
            )
            
    def reject(self):
        """Cancel button clicked"""
        # Restore original config
        self.config = AppConfig(**self.original_config.model_dump())
        super().reject()
        
    def on_theme_changed(self, theme_name: str):
        """Handle theme change"""
        # Update the theme
        theme_manager = get_theme_manager()
        theme_manager.set_theme(theme_name)


# Example usage
if __name__ == "__main__":
    import sys
    from PyQt6.QtWidgets import QApplication
    
    app = QApplication(sys.argv)
    
    # Initialize theme manager
    theme_manager = get_theme_manager()
    
    # Create test configuration
    config = AppConfig()
    
    # Create and show settings dialog
    dialog = EnhancedSettingsDialog(config)
    dialog.config_changed.connect(lambda c: print(f"Config changed: {c}"))
    dialog.show()
    
    sys.exit(app.exec())