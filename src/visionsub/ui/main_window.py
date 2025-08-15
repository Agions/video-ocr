from typing import List

from PyQt6.QtCore import Qt, pyqtSlot, QSize
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QMainWindow,
    QProgressBar,
    QPushButton,
    QSlider,
    QSpinBox,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from visionsub.models.config import OcrConfig
from visionsub.ui.video_player import VideoPlayer
from visionsub.ui.roi_selection import ROISelectionPanel
from visionsub.ui.subtitle_editor import SubtitleEditorWindow
from visionsub.ui.advanced_ocr_settings import AdvancedOCRSettingsDialog
from visionsub.view_models.main_view_model import MainViewModel


class MainWindow(QMainWindow):
    def __init__(self, view_model: MainViewModel):
        super().__init__()
        self.vm = view_model
        self.setWindowTitle("VisionSub - 视频OCR字幕提取工具")
        self.setGeometry(100, 100, 1400, 800)

        # Load application style
        self._load_style_sheet()

        # 设置中文字体
        self._setup_chinese_font()

        # --- Widgets ---
        self.video_player = VideoPlayer()
        self.open_button = QPushButton("打开视频预览")
        self.ocr_button = QPushButton("执行OCR识别")
        self.single_result_label = QLabel("<OCR识别结果>")
        self.language_combo = QComboBox()
        self.threshold_slider = QSlider(Qt.Orientation.Horizontal)
        self.threshold_spinbox = QSpinBox()
        self.playback_slider = QSlider(Qt.Orientation.Horizontal)
        self.play_pause_button = QPushButton("播放")

        # --- Batch Processing Widgets ---
        self.queue_list_widget = QListWidget()
        self.add_to_queue_button = QPushButton("添加视频到队列")
        self.clear_queue_button = QPushButton("清空队列")
        self.start_batch_button = QPushButton("开始批量处理")
        self.batch_progress_bar = QProgressBar()
        self.batch_status_label = QLabel("批量处理空闲中")
        
        # --- ROI Selection Panel ---
        self.roi_panel = ROISelectionPanel(self.vm.get_roi_manager())
        self.roi_panel.setMaximumWidth(350)
        
        # --- Subtitle Editor Button ---
        self.subtitle_editor_button = QPushButton("打开字幕编辑器")
        self.subtitle_editor_button.setStyleSheet("QPushButton { background-color: #4CAF50; color: white; font-weight: bold; padding: 8px; }")
        self.subtitle_editor_window = None
        
        # --- Advanced OCR Settings Button ---
        self.advanced_settings_button = QPushButton("高级 OCR 设置")
        self.advanced_settings_button.setStyleSheet("QPushButton { background-color: #2196F3; color: white; font-weight: bold; padding: 8px; }")
        self.advanced_settings_dialog = None
        
        # --- Style OCR button ---
        self.ocr_button.setStyleSheet("QPushButton { background-color: #FF9800; color: white; font-weight: bold; padding: 8px; }")

        self._setup_ui()
        self._connect_vm()

    def _load_style_sheet(self):
        """Load application style sheet"""
        try:
            from pathlib import Path
            style_path = Path(__file__).parent / "style.qss"
            if style_path.exists():
                with open(style_path, 'r', encoding='utf-8') as f:
                    self.setStyleSheet(f.read())
        except Exception as e:
            print(f"Failed to load style sheet: {e}")

    def _setup_chinese_font(self):
        """设置中文字体支持"""
        # 尝试使用系统中文字体
        chinese_fonts = [
            "PingFang SC",  # macOS 中文字体
            "Microsoft YaHei",  # Windows 中文字体
            "Noto Sans CJK SC",  # Linux 中文字体
            "WenQuanYi Micro Hei",  # Linux 中文字体
            "SimHei",  # Windows 黑体
            "SimSun",  # Windows 宋体
            "Arial Unicode MS"  # 跨平台 Unicode 字体
        ]

        font = QFont()
        font.setPointSize(10)  # 设置基础字体大小

        # 查找可用的中文字体
        from PyQt6.QtGui import QFontDatabase
        available_families = QFontDatabase.families()
        for font_name in chinese_fonts:
            if font_name in available_families:
                font.setFamily(font_name)
                break

        self.setFont(font)

    def _setup_ui(self):
        # --- Central Widget with Tabs ---
        central_widget = QWidget()
        main_layout = QHBoxLayout(central_widget)
        
        # --- Left Panel (Video) ---
        video_widget = QWidget()
        video_layout = QVBoxLayout(video_widget)
        video_layout.addWidget(self.video_player, 1)
        playback_layout = QHBoxLayout()
        playback_layout.addWidget(self.play_pause_button)
        playback_layout.addWidget(self.playback_slider)
        video_layout.addLayout(playback_layout)
        video_layout.addWidget(self.ocr_button)
        video_layout.addWidget(self.single_result_label)
        video_layout.setStretch(0, 1)

        # --- Right Panel (Controls & ROI) ---
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        
        # Basic controls
        controls_group = QWidget()
        controls_layout = QVBoxLayout(controls_group)
        
        controls_layout.addWidget(self.open_button)
        param_form = QFormLayout()
        self.language_combo.addItems(["中文", "英文", "韩文", "日文"])
        self.language_combo.setCurrentText("中文")  # 默认选择中文
        param_form.addRow(QLabel("识别语言:"), self.language_combo)
        threshold_layout = QHBoxLayout()
        threshold_layout.addWidget(self.threshold_slider)
        threshold_layout.addWidget(self.threshold_spinbox)
        param_form.addRow(QLabel("阈值:"), threshold_layout)
        controls_layout.addLayout(param_form)
        
        right_layout.addWidget(controls_group)
        
        # ROI Panel
        right_layout.addWidget(self.roi_panel)
        
        # Batch controls
        batch_group = QWidget()
        batch_layout = QVBoxLayout(batch_group)
        
        batch_buttons_layout = QHBoxLayout()
        batch_buttons_layout.addWidget(self.add_to_queue_button)
        batch_buttons_layout.addWidget(self.clear_queue_button)

        batch_layout.addSpacing(10)
        batch_layout.addWidget(QLabel("批量处理队列:"))
        batch_layout.addWidget(self.queue_list_widget)
        batch_layout.addLayout(batch_buttons_layout)
        batch_layout.addWidget(self.start_batch_button)
        batch_layout.addWidget(self.batch_progress_bar)
        batch_layout.addWidget(self.batch_status_label)
        
        right_layout.addWidget(batch_group)
        
        # Subtitle Editor Button
        editor_layout = QHBoxLayout()
        editor_layout.addWidget(self.subtitle_editor_button)
        right_layout.addLayout(editor_layout)
        
        # Advanced Settings Button
        advanced_layout = QHBoxLayout()
        advanced_layout.addWidget(self.advanced_settings_button)
        right_layout.addLayout(advanced_layout)
        
        right_layout.addStretch()

        # --- Main Splitter ---
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(video_widget)
        splitter.addWidget(right_widget)
        splitter.setSizes([900, 500])
        
        main_layout.addWidget(splitter)
        self.setCentralWidget(central_widget)

    def _connect_vm(self):
        # --- Connections ---
        # Preview
        self.open_button.clicked.connect(self.select_file)
        self.video_player.roi_changed.connect(self.vm.update_roi)
        self.language_combo.currentTextChanged.connect(self.vm.set_language)
        self.threshold_slider.valueChanged.connect(self.vm.set_threshold)
        self.threshold_spinbox.valueChanged.connect(self.vm.set_threshold)
        self.play_pause_button.clicked.connect(self.vm.toggle_playback)
        self.playback_slider.sliderMoved.connect(lambda val: self.vm.seek_frame(val, True))
        self.ocr_button.clicked.connect(self.run_ocr)

        # Batch
        self.add_to_queue_button.clicked.connect(self.select_files_for_queue)
        self.clear_queue_button.clicked.connect(self.vm.clear_queue)
        self.start_batch_button.clicked.connect(self.vm.start_batch_processing)

        # ROI Panel
        self.roi_panel.roi_config_changed.connect(self.vm.update_roi_config)
        
        # Subtitle Editor
        self.subtitle_editor_button.clicked.connect(self.open_subtitle_editor)
        
        # Advanced Settings
        self.advanced_settings_button.clicked.connect(self.open_advanced_settings)

        # VM -> UI
        self.vm.frame_changed.connect(self.video_player.update_frame)
        self.vm.config_changed.connect(self.update_config_display)
        self.vm.single_ocr_result_changed.connect(self.single_result_label.setText)
        self.vm.video_loaded.connect(self.on_video_loaded)
        self.vm.is_playing_changed.connect(self.on_playback_state_changed)
        self.vm.frame_index_changed.connect(self.playback_slider.setValue)
        self.vm.queue_changed.connect(self.update_queue_list)
        self.vm.batch_progress_changed.connect(self.batch_progress_bar.setValue)
        self.vm.batch_status_changed.connect(self.batch_status_label.setText)

        # Sync widgets
        self.threshold_slider.valueChanged.connect(self.threshold_spinbox.setValue)
        self.threshold_spinbox.valueChanged.connect(self.threshold_slider.setValue)

        self.update_config_display(self.vm.config)

    @pyqtSlot()
    def select_file(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "选择视频文件", "", "视频文件 (*.mp4 *.avi *.mkv)")
        if file_path:
            self.vm.load_video(file_path)

    @pyqtSlot()
    def select_files_for_queue(self):
        files, _ = QFileDialog.getOpenFileNames(self, "选择要添加到队列的视频文件", "", "视频文件 (*.mp4 *.avi *.mkv)")
        if files:
            self.vm.add_to_queue(files)

    @pyqtSlot(list)
    def update_queue_list(self, queue: List[str]):
        self.queue_list_widget.clear()
        self.queue_list_widget.addItems(queue)

    @pyqtSlot(OcrConfig)
    def update_config_display(self, config: OcrConfig):
        # Block signals to prevent feedback loops
        self.language_combo.blockSignals(True)
        self.threshold_slider.blockSignals(True)
        self.threshold_spinbox.blockSignals(True)
        self.language_combo.setCurrentText(config.language)
        self.threshold_slider.setValue(config.threshold)
        self.threshold_spinbox.setValue(config.threshold)
        self.language_combo.blockSignals(False)
        self.threshold_slider.blockSignals(False)
        self.threshold_spinbox.blockSignals(False)

    @pyqtSlot(int)
    def on_video_loaded(self, frame_count: int):
        self.playback_slider.setEnabled(True)
        self.play_pause_button.setEnabled(True)
        self.playback_slider.setRange(0, frame_count - 1)
        
        # 设置ROI面板的视频尺寸
        video_info = self.vm.get_video_info()
        if video_info:
            from PyQt6.QtCore import QSize
            video_size = QSize(video_info.get('width', 0), video_info.get('height', 0))
            self.roi_panel.set_video_size(video_size)

    @pyqtSlot(bool)
    def on_playback_state_changed(self, is_playing: bool):
        self.play_pause_button.setText("暂停" if is_playing else "播放")
    
    def open_subtitle_editor(self):
        """打开字幕编辑器"""
        if self.subtitle_editor_window is None or not self.subtitle_editor_window.isVisible():
            self.subtitle_editor_window = SubtitleEditorWindow(self)
            self.subtitle_editor_window.show()
            # 如果有已处理的字幕，加载到编辑器中
            if hasattr(self.vm, 'processed_subtitles') and self.vm.processed_subtitles:
                self.subtitle_editor_window.set_subtitles(self.vm.processed_subtitles)
        else:
            self.subtitle_editor_window.raise_()
            self.subtitle_editor_window.activateWindow()
    
    def run_ocr(self):
        """Run OCR on the current frame"""
        self.vm.run_single_frame_ocr_sync()
    
    def open_advanced_settings(self):
        """打开高级OCR设置对话框"""
        if self.advanced_settings_dialog is None or not self.advanced_settings_dialog.isVisible():
            self.advanced_settings_dialog = AdvancedOCRSettingsDialog(self.vm.config, self)
            self.advanced_settings_dialog.config_changed.connect(self.vm.update_config)
            self.advanced_settings_dialog.show()
        else:
            self.advanced_settings_dialog.raise_()
            self.advanced_settings_dialog.activateWindow()
