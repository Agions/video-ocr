from typing import Dict, Any, Optional
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QFormLayout, 
    QLabel, QPushButton, QComboBox, QCheckBox, 
    QSpinBox, QDoubleSpinBox, QGroupBox, QTabWidget,
    QDialogButtonBox, QMessageBox, QSpinBox, QWidget
)

from visionsub.models.config import OcrConfig


class AdvancedOCRSettingsDialog(QDialog):
    """Advanced OCR settings dialog"""
    
    config_changed = pyqtSignal(OcrConfig)
    
    def __init__(self, config: OcrConfig, parent=None):
        super().__init__(parent)
        self.config = config
        self.original_config = OcrConfig(**config.model_dump())
        
        self.setWindowTitle("高级 OCR 设置")
        self.setModal(True)
        self.resize(600, 500)
        
        self._setup_ui()
        self._load_config()
    
    def _setup_ui(self):
        layout = QVBoxLayout(self)
        
        # Create tabs
        tabs = QTabWidget()
        
        # Basic settings tab
        basic_tab = self._create_basic_tab()
        tabs.addTab(basic_tab, "基本设置")
        
        # Preprocessing tab
        preprocessing_tab = self._create_preprocessing_tab()
        tabs.addTab(preprocessing_tab, "图像预处理")
        
        # Advanced tab
        advanced_tab = self._create_advanced_tab()
        tabs.addTab(advanced_tab, "高级设置")
        
        layout.addWidget(tabs)
        
        # Buttons
        button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | 
            QDialogButtonBox.StandardButton.Cancel |
            QDialogButtonBox.StandardButton.Apply |
            QDialogButtonBox.StandardButton.Reset
        )
        
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        button_box.button(QDialogButtonBox.StandardButton.Apply).clicked.connect(self._apply_settings)
        button_box.button(QDialogButtonBox.StandardButton.Reset).clicked.connect(self._reset_settings)
        
        layout.addWidget(button_box)
    
    def _create_basic_tab(self) -> QWidget:
        widget = QWidget()
        layout = QFormLayout(widget)
        
        # OCR Engine selection
        self.engine_combo = QComboBox()
        self.engine_combo.addItems(["PaddleOCR", "Tesseract"])
        layout.addRow("OCR 引擎:", self.engine_combo)
        
        # Language selection
        self.language_combo = QComboBox()
        self.language_combo.addItems(["中文", "英文", "韩文", "日文", "法文", "德文", "西班牙文", "俄文", "阿拉伯文", "印地文"])
        layout.addRow("识别语言:", self.language_combo)
        
        # Threshold
        threshold_layout = QHBoxLayout()
        self.threshold_slider = QSpinBox()
        self.threshold_slider.setRange(0, 255)
        self.threshold_slider.setSingleStep(1)
        
        self.threshold_spinbox = QSpinBox()
        self.threshold_spinbox.setRange(0, 255)
        self.threshold_spinbox.setSingleStep(1)
        
        threshold_layout.addWidget(self.threshold_slider)
        threshold_layout.addWidget(self.threshold_spinbox)
        layout.addRow("二值化阈值:", threshold_layout)
        
        # Confidence threshold
        self.confidence_spinbox = QDoubleSpinBox()
        self.confidence_spinbox.setRange(0.0, 1.0)
        self.confidence_spinbox.setSingleStep(0.05)
        self.confidence_spinbox.setDecimals(2)
        layout.addRow("置信度阈值:", self.confidence_spinbox)
        
        # Auto detect language
        self.auto_detect_checkbox = QCheckBox("自动检测语言")
        layout.addRow("语言检测:", self.auto_detect_checkbox)
        
        # Connect threshold widgets
        self.threshold_slider.valueChanged.connect(self.threshold_spinbox.setValue)
        self.threshold_spinbox.valueChanged.connect(self.threshold_slider.setValue)
        
        return widget
    
    def _create_preprocessing_tab(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Enable preprocessing group
        preprocessing_group = QGroupBox("图像预处理选项")
        preprocessing_layout = QVBoxLayout(preprocessing_group)
        
        self.enable_preprocessing_checkbox = QCheckBox("启用图像预处理")
        preprocessing_layout.addWidget(self.enable_preprocessing_checkbox)
        
        # Individual preprocessing options
        self.denoise_checkbox = QCheckBox("去噪 (Denoising)")
        preprocessing_layout.addWidget(self.denoise_checkbox)
        
        self.enhance_contrast_checkbox = QCheckBox("对比度增强 (Contrast Enhancement)")
        preprocessing_layout.addWidget(self.enhance_contrast_checkbox)
        
        self.sharpen_checkbox = QCheckBox("图像锐化 (Sharpening)")
        preprocessing_layout.addWidget(self.sharpen_checkbox)
        
        layout.addWidget(preprocessing_group)
        
        # Post-processing group
        postprocessing_group = QGroupBox("文本后处理选项")
        postprocessing_layout = QVBoxLayout(postprocessing_group)
        
        self.enable_postprocessing_checkbox = QCheckBox("启用文本后处理")
        postprocessing_layout.addWidget(self.enable_postprocessing_checkbox)
        
        layout.addWidget(postprocessing_group)
        
        layout.addStretch()
        
        return widget
    
    def _create_advanced_tab(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Custom parameters group
        custom_group = QGroupBox("自定义参数")
        custom_layout = QFormLayout(custom_group)
        
        self.custom_params_text = QLabel("自定义参数 (JSON格式)")
        self.custom_params_text.setWordWrap(True)
        custom_layout.addRow("参数格式:", self.custom_params_text)
        
        layout.addWidget(custom_group)
        
        # Performance tips
        tips_group = QGroupBox("性能提示")
        tips_layout = QVBoxLayout(tips_group)
        
        tips = [
            "• 启用预处理会提高识别准确率但增加处理时间",
            "• 对比度增强对低对比度视频效果显著",
            "• 去噪对有颗粒感的视频效果较好",
            "• 自动语言检测可能需要更多处理时间",
            "• 调整置信度阈值可以过滤低质量识别结果"
        ]
        
        for tip in tips:
            tip_label = QLabel(tip)
            tip_label.setWordWrap(True)
            tip_label.setStyleSheet("color: #666; font-size: 9pt;")
            tips_layout.addWidget(tip_label)
        
        layout.addWidget(tips_group)
        layout.addStretch()
        
        return widget
    
    def _load_config(self):
        """Load current configuration into UI"""
        # Basic settings
        self.engine_combo.setCurrentText(self.config.engine)
        self.language_combo.setCurrentText(self.config.language)
        self.threshold_slider.setValue(self.config.threshold)
        self.threshold_spinbox.setValue(self.config.threshold)
        self.confidence_spinbox.setValue(self.config.confidence_threshold)
        self.auto_detect_checkbox.setChecked(self.config.auto_detect_language)
        
        # Preprocessing settings
        self.enable_preprocessing_checkbox.setChecked(self.config.enable_preprocessing)
        self.denoise_checkbox.setChecked(self.config.denoise)
        self.enhance_contrast_checkbox.setChecked(self.config.enhance_contrast)
        self.sharpen_checkbox.setChecked(self.config.sharpen)
        self.enable_postprocessing_checkbox.setChecked(self.config.enable_postprocessing)
        
        # Update preprocessing options enabled state
        self._update_preprocessing_state()
    
    def _update_preprocessing_state(self):
        """Update preprocessing options enabled state"""
        enabled = self.enable_preprocessing_checkbox.isChecked()
        
        self.denoise_checkbox.setEnabled(enabled)
        self.enhance_contrast_checkbox.setEnabled(enabled)
        self.sharpen_checkbox.setEnabled(enabled)
    
    def _get_config_from_ui(self) -> OcrConfig:
        """Get configuration from UI"""
        config_data = self.config.model_dump()
        
        # Update basic settings
        config_data.update({
            'engine': self.engine_combo.currentText(),
            'language': self.language_combo.currentText(),
            'threshold': self.threshold_slider.value(),
            'confidence_threshold': self.confidence_spinbox.value(),
            'auto_detect_language': self.auto_detect_checkbox.isChecked()
        })
        
        # Update preprocessing settings
        config_data.update({
            'enable_preprocessing': self.enable_preprocessing_checkbox.isChecked(),
            'denoise': self.denoise_checkbox.isChecked(),
            'enhance_contrast': self.enhance_contrast_checkbox.isChecked(),
            'sharpen': self.sharpen_checkbox.isChecked(),
            'enable_postprocessing': self.enable_postprocessing_checkbox.isChecked()
        })
        
        return OcrConfig(**config_data)
    
    def _apply_settings(self):
        """Apply settings without closing dialog"""
        try:
            new_config = self._get_config_from_ui()
            self.config = new_config
            self.config_changed.emit(new_config)
            
            # Show confirmation
            QMessageBox.information(
                self, 
                "设置已应用", 
                "高级 OCR 设置已成功应用。"
            )
            
        except Exception as e:
            QMessageBox.critical(
                self,
                "错误",
                f"应用设置时出错: {e}"
            )
    
    def _reset_settings(self):
        """Reset to original settings"""
        reply = QMessageBox.question(
            self,
            "重置设置",
            "确定要重置为原始设置吗？",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            self.config = OcrConfig(**self.original_config.model_dump())
            self._load_config()
            self.config_changed.emit(self.config)
    
    def accept(self):
        """OK button clicked"""
        try:
            new_config = self._get_config_from_ui()
            self.config = new_config
            super().accept()
            
        except Exception as e:
            QMessageBox.critical(
                self,
                "错误",
                f"保存设置时出错: {e}"
            )
    
    def reject(self):
        """Cancel button clicked"""
        # Restore original config
        self.config = OcrConfig(**self.original_config.model_dump())
        super().reject()
    
    def get_config(self) -> OcrConfig:
        """Get current configuration"""
        return self.config