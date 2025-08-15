"""
Subtitle Editor Component - Advanced subtitle editing and management
"""
import logging
from dataclasses import dataclass
from datetime import timedelta
from typing import Any, Dict, List, Optional, Tuple

from PyQt6.QtCore import QAbstractTableModel, QModelIndex, Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QAction, QColor, QFont, QKeySequence, QShortcut, QTextCharFormat
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDialog,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMenu,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QTimeEdit,
    QToolBar,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from visionsub.models.subtitle import SubtitleItem

logger = logging.getLogger(__name__)


@dataclass
class SubtitleEditState:
    """字幕编辑状态"""
    is_editing: bool = False
    current_index: int = -1
    undo_stack: List[Dict[str, Any]] = None
    redo_stack: List[Dict[str, Any]] = None
    modified_count: int = 0
    
    def __post_init__(self):
        if self.undo_stack is None:
            self.undo_stack = []
        if self.redo_stack is None:
            self.redo_stack = []


class SubtitleEditDialog(QDialog):
    """字幕编辑对话框"""
    
    def __init__(self, subtitle: SubtitleItem = None, parent=None):
        super().__init__(parent)
        self.subtitle = subtitle
        self.setWindowTitle("编辑字幕" if subtitle else "添加字幕")
        self.setModal(True)
        self.resize(500, 400)
        
        self.setup_ui()
        
        if subtitle:
            self.load_subtitle_data()
    
    def setup_ui(self):
        """设置UI"""
        layout = QFormLayout(self)
        
        # 基本信息
        self.index_edit = QLineEdit()
        self.index_edit.setReadOnly(True)
        layout.addRow("序号:", self.index_edit)
        
        # 时间编辑
        time_layout = QHBoxLayout()
        self.start_time_edit = QTimeEdit()
        self.end_time_edit = QTimeEdit()
        self.start_time_edit.setDisplayFormat("HH:mm:ss")
        self.end_time_edit.setDisplayFormat("HH:mm:ss")
        time_layout.addWidget(QLabel("开始时间:"))
        time_layout.addWidget(self.start_time_edit)
        time_layout.addWidget(QLabel("结束时间:"))
        time_layout.addWidget(self.end_time_edit)
        layout.addRow("时间:", time_layout)
        
        # 文本内容
        self.text_edit = QTextEdit()
        self.text_edit.setPlaceholderText("请输入字幕内容...")
        self.text_edit.setMinimumHeight(150)
        layout.addRow("内容:", self.text_edit)
        
        # 置信度
        self.confidence_spin = QComboBox()
        self.confidence_spin.setEditable(True)
        for i in range(0, 101, 10):
            self.confidence_spin.addItem(f"{i}%")
        layout.addRow("置信度:", self.confidence_spin)
        
        # 语言
        self.language_edit = QLineEdit()
        layout.addRow("语言:", self.language_edit)
        
        # 按钮
        button_layout = QHBoxLayout()
        self.ok_btn = QPushButton("确定")
        self.cancel_btn = QPushButton("取消")
        button_layout.addStretch()
        button_layout.addWidget(self.ok_btn)
        button_layout.addWidget(self.cancel_btn)
        layout.addRow("", button_layout)
        
        # 连接信号
        self.ok_btn.clicked.connect(self.accept)
        self.cancel_btn.clicked.connect(self.reject)
        
        # 设置快捷键
        QShortcut(QKeySequence("Ctrl+S"), self, self.accept)
        QShortcut(QKeySequence("Escape"), self, self.reject)
    
    def load_subtitle_data(self):
        """加载字幕数据"""
        if self.subtitle:
            self.index_edit.setText(str(self.subtitle.index))
            
            # 解析时间
            start_time = self._parse_srt_time(self.subtitle.start_time)
            end_time = self._parse_srt_time(self.subtitle.end_time)
            
            if start_time:
                self.start_time_edit.setTime(start_time)
            if end_time:
                self.end_time_edit.setTime(end_time)
            
            self.text_edit.setPlainText(self.subtitle.content)
            # 注释掉不存在的字段
            # self.confidence_spin.setCurrentText(f"{int(self.subtitle.confidence * 100)}%")
            # self.language_edit.setText(self.subtitle.language or "")
    
    def get_subtitle_data(self) -> Dict[str, Any]:
        """获取字幕数据"""
        start_time = self.start_time_edit.time()
        end_time = self.end_time_edit.time()
        
        return {
            "index": int(self.index_edit.text()) if self.index_edit.text() else 1,
            "start_time": self._format_srt_time(start_time),
            "end_time": self._format_srt_time(end_time),
            "text": self.text_edit.toPlainText(),
            "confidence": float(self.confidence_spin.currentText().replace("%", "")) / 100.0,
            "language": self.language_edit.text()
        }
    
    def _parse_srt_time(self, time_str: str):
        """解析SRT时间格式"""
        try:
            # 格式: HH:MM:SS,ms
            if "," in time_str:
                time_part, ms_part = time_str.split(",")
                h, m, s = time_part.split(":")
                ms = int(ms_part)
            else:
                h, m, s = time_str.split(":")
                ms = 0
            
            from PyQt6.QtCore import QTime
            return QTime(int(h), int(m), int(s), ms)
        except:
            return None
    
    def _format_srt_time(self, qtime):
        """格式化为SRT时间格式"""
        return f"{qtime.toString('HH:mm:ss')},000"


class SubtitleTableModel(QAbstractTableModel):
    """字幕表格模型"""
    
    def __init__(self, subtitles: List[SubtitleItem] = None):
        super().__init__()
        self.subtitles = subtitles or []
        self.headers = ["序号", "开始时间", "结束时间", "内容", "置信度", "语言"]
    
    def rowCount(self, parent=None):
        return len(self.subtitles)
    
    def columnCount(self, parent=None):
        return len(self.headers)
    
    def data(self, index: QModelIndex, role: int = Qt.ItemDataRole.DisplayRole):
        if not index.isValid():
            return None
        
        row = index.row()
        col = index.column()
        
        if row >= len(self.subtitles):
            return None
        
        subtitle = self.subtitles[row]
        
        if role == Qt.ItemDataRole.DisplayRole:
            if col == 0:  # 序号
                return str(subtitle.index)
            elif col == 1:  # 开始时间
                return str(subtitle.start_time)
            elif col == 2:  # 结束时间
                return str(subtitle.end_time)
            elif col == 3:  # 内容
                return subtitle.text[:50] + "..." if len(subtitle.text) > 50 else subtitle.text
            elif col == 4:  # 置信度
                return f"{subtitle.confidence:.1%}"
            elif col == 5:  # 语言
                return subtitle.language or ""
        
        elif role == Qt.ItemDataRole.TextAlignmentRole:
            if col in [0, 4]:  # 序号、置信度居中对齐
                return Qt.AlignmentFlag.AlignCenter
            elif col in [1, 2]:  # 时间居右对齐
                return Qt.AlignmentFlag.AlignRight
            return Qt.AlignmentFlag.AlignLeft
        
        elif role == Qt.ItemDataRole.BackgroundRole:
            # 根据置信度设置背景色
            if col == 4:  # 置信度列
                confidence = subtitle.confidence
                if confidence >= 0.8:
                    return QColor(144, 238, 144, 100)  # 浅绿色
                elif confidence >= 0.6:
                    return QColor(255, 255, 224, 100)  # 浅黄色
                else:
                    return QColor(255, 182, 193, 100)  # 浅红色
        
        return None
    
    def headerData(self, section: int, orientation: Qt.Orientation, role: int = Qt.ItemDataRole.DisplayRole):
        if orientation == Qt.Orientation.Horizontal and role == Qt.ItemDataRole.DisplayRole:
            return self.headers[section]
        return None
    
    def update_data(self, subtitles: List[SubtitleItem]):
        """更新数据"""
        self.beginResetModel()
        self.subtitles = subtitles
        self.endResetModel()
    
    def get_subtitle_at(self, row: int) -> Optional[SubtitleItem]:
        """获取指定行的字幕"""
        if 0 <= row < len(self.subtitles):
            return self.subtitles[row]
        return None
    
    def update_subtitle_at(self, row: int, subtitle: SubtitleItem):
        """更新指定行的字幕"""
        if 0 <= row < len(self.subtitles):
            self.subtitles[row] = subtitle
            self.dataChanged.emit(
                self.index(row, 0),
                self.index(row, self.columnCount() - 1)
            )


class SubtitleEditorWidget(QWidget):
    """字幕编辑器组件"""
    
    # 信号定义
    subtitle_modified = pyqtSignal(int, SubtitleItem)  # 字幕修改信号
    subtitle_added = pyqtSignal(SubtitleItem)         # 字幕添加信号
    subtitle_deleted = pyqtSignal(int)               # 字幕删除信号
    subtitles_saved = pyqtSignal(str)                 # 字幕保存信号
    subtitles_loaded = pyqtSignal(list)               # 字幕加载信号
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.state = SubtitleEditState()
        self.subtitles: List[SubtitleItem] = []
        
        self.setup_ui()
        self.setup_shortcuts()
        self.connect_signals()
    
    def setup_ui(self):
        """设置UI"""
        layout = QVBoxLayout(self)
        
        # 工具栏
        toolbar_layout = QHBoxLayout()
        
        # 编辑操作按钮
        self.add_btn = QPushButton("添加")
        self.edit_btn = QPushButton("编辑")
        self.delete_btn = QPushButton("删除")
        self.duplicate_btn = QPushButton("复制")
        
        # 撤销/重做按钮
        self.undo_btn = QPushButton("撤销")
        self.redo_btn = QPushButton("重做")
        
        # 保存/加载按钮
        self.save_btn = QPushButton("保存")
        self.load_btn = QPushButton("加载")
        
        toolbar_layout.addWidget(self.add_btn)
        toolbar_layout.addWidget(self.edit_btn)
        toolbar_layout.addWidget(self.delete_btn)
        toolbar_layout.addWidget(self.duplicate_btn)
        toolbar_layout.addSpacing(10)
        toolbar_layout.addWidget(self.undo_btn)
        toolbar_layout.addWidget(self.redo_btn)
        toolbar_layout.addSpacing(10)
        toolbar_layout.addWidget(self.save_btn)
        toolbar_layout.addWidget(self.load_btn)
        toolbar_layout.addStretch()
        
        layout.addLayout(toolbar_layout)
        
        # 搜索和过滤
        filter_layout = QHBoxLayout()
        self.search_edit = QLineEdit()
        self.search_edit.setPlaceholderText("搜索字幕内容...")
        self.language_filter = QComboBox()
        self.language_filter.addItem("所有语言")
        self.confidence_filter = QComboBox()
        self.confidence_filter.addItem("所有置信度")
        self.confidence_filter.addItems(["高 (>80%)", "中 (60-80%)", "低 (<60%)"])
        
        filter_layout.addWidget(QLabel("搜索:"))
        filter_layout.addWidget(self.search_edit)
        filter_layout.addWidget(QLabel("语言:"))
        filter_layout.addWidget(self.language_filter)
        filter_layout.addWidget(QLabel("置信度:"))
        filter_layout.addWidget(self.confidence_filter)
        
        layout.addLayout(filter_layout)
        
        # 分割器
        splitter = QSplitter(Qt.Orientation.Vertical)
        
        # 字幕表格
        self.subtitle_table = QTableWidget()
        self.subtitle_table.setColumnCount(6)
        self.subtitle_table.setHorizontalHeaderLabels([
            "序号", "开始时间", "结束时间", "内容", "置信度", "语言"
        ])
        self.subtitle_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.subtitle_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.subtitle_table.horizontalHeader().setStretchLastSection(True)
        self.subtitle_table.setAlternatingRowColors(True)
        
        # 设置列宽
        self.subtitle_table.setColumnWidth(0, 60)   # 序号
        self.subtitle_table.setColumnWidth(1, 100)  # 开始时间
        self.subtitle_table.setColumnWidth(2, 100)  # 结束时间
        self.subtitle_table.setColumnWidth(3, 300)  # 内容
        self.subtitle_table.setColumnWidth(4, 80)   # 置信度
        self.subtitle_table.setColumnWidth(5, 80)   # 语言
        
        splitter.addWidget(self.subtitle_table)
        
        # 预览区域
        preview_group = QGroupBox("字幕预览")
        preview_layout = QVBoxLayout(preview_group)
        
        self.preview_edit = QTextEdit()
        self.preview_edit.setReadOnly(True)
        self.preview_edit.setMaximumHeight(150)
        preview_layout.addWidget(self.preview_edit)
        
        # 预览控制
        preview_control_layout = QHBoxLayout()
        self.prev_btn = QPushButton("上一个")
        self.next_btn = QPushButton("下一个")
        self.play_btn = QPushButton("播放")
        self.stop_btn = QPushButton("停止")
        
        preview_control_layout.addWidget(self.prev_btn)
        preview_control_layout.addWidget(self.next_btn)
        preview_control_layout.addWidget(self.play_btn)
        preview_control_layout.addWidget(self.stop_btn)
        preview_control_layout.addStretch()
        
        preview_layout.addLayout(preview_control_layout)
        splitter.addWidget(preview_group)
        
        # 设置分割器比例
        splitter.setSizes([400, 150])
        
        layout.addWidget(splitter)
        
        # 状态栏
        self.status_label = QLabel("就绪")
        layout.addWidget(self.status_label)
        
        # 初始化按钮状态
        self.update_button_states()
    
    def setup_shortcuts(self):
        """设置快捷键"""
        # 编辑快捷键
        QShortcut(QKeySequence("Ctrl+N"), self, self.add_subtitle)
        QShortcut(QKeySequence("Ctrl+E"), self, self.edit_subtitle)
        QShortcut(QKeySequence("Delete"), self, self.delete_subtitle)
        QShortcut(QKeySequence("Ctrl+D"), self, self.duplicate_subtitle)
        
        # 撤销/重做快捷键
        QShortcut(QKeySequence("Ctrl+Z"), self, self.undo)
        QShortcut(QKeySequence("Ctrl+Y"), self, self.redo)
        
        # 保存/加载快捷键
        QShortcut(QKeySequence("Ctrl+S"), self, self.save_subtitles)
        QShortcut(QKeySequence("Ctrl+O"), self, self.load_subtitles)
        
        # 导航快捷键
        QShortcut(QKeySequence("Up"), self, lambda: self.navigate_subtitle(-1))
        QShortcut(QKeySequence("Down"), self, lambda: self.navigate_subtitle(1))
    
    def connect_signals(self):
        """连接信号"""
        # 按钮信号
        self.add_btn.clicked.connect(self.add_subtitle)
        self.edit_btn.clicked.connect(self.edit_subtitle)
        self.delete_btn.clicked.connect(self.delete_subtitle)
        self.duplicate_btn.clicked.connect(self.duplicate_subtitle)
        self.undo_btn.clicked.connect(self.undo)
        self.redo_btn.clicked.connect(self.redo)
        self.save_btn.clicked.connect(self.save_subtitles)
        self.load_btn.clicked.connect(self.load_subtitles)
        
        # 预览控制信号
        self.prev_btn.clicked.connect(lambda: self.navigate_subtitle(-1))
        self.next_btn.clicked.connect(lambda: self.navigate_subtitle(1))
        self.play_btn.clicked.connect(self.play_current_subtitle)
        self.stop_btn.clicked.connect(self.stop_preview)
        
        # 表格选择信号
        self.subtitle_table.itemSelectionChanged.connect(self.on_selection_changed)
        self.subtitle_table.doubleClicked.connect(self.edit_subtitle)
        
        # 搜索和过滤信号
        self.search_edit.textChanged.connect(self.filter_subtitles)
        self.language_filter.currentTextChanged.connect(self.filter_subtitles)
        self.confidence_filter.currentTextChanged.connect(self.filter_subtitles)
    
    def set_subtitles(self, subtitles: List[SubtitleItem]):
        """设置字幕列表"""
        self.subtitles = subtitles.copy()
        self.update_table()
        self.update_status(f"加载了 {len(subtitles)} 条字幕")
    
    def update_table(self):
        """更新表格"""
        self.subtitle_table.setRowCount(len(self.subtitles))
        
        for row, subtitle in enumerate(self.subtitles):
            # 序号
            item = QTableWidgetItem(str(subtitle.index))
            item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.subtitle_table.setItem(row, 0, item)
            
            # 开始时间
            item = QTableWidgetItem(str(subtitle.start_time))
            item.setTextAlignment(Qt.AlignmentFlag.AlignRight)
            self.subtitle_table.setItem(row, 1, item)
            
            # 结束时间
            item = QTableWidgetItem(str(subtitle.end_time))
            item.setTextAlignment(Qt.AlignmentFlag.AlignRight)
            self.subtitle_table.setItem(row, 2, item)
            
            # 内容
            content = subtitle.content
            if len(content) > 80:
                content = content[:77] + "..."
            item = QTableWidgetItem(content)
            self.subtitle_table.setItem(row, 3, item)
            
            # 置信度（暂时显示为空）
            item = QTableWidgetItem("")
            item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.subtitle_table.setItem(row, 4, item)
            
            # 语言（暂时显示为空）
            item = QTableWidgetItem("")
            item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.subtitle_table.setItem(row, 5, item)
        
        self.update_button_states()
    
    def on_selection_changed(self):
        """选择变化事件"""
        selected_items = self.subtitle_table.selectedItems()
        if selected_items:
            row = selected_items[0].row()
            self.state.current_index = row
            self.update_preview(row)
        else:
            self.state.current_index = -1
            self.preview_edit.clear()
        
        self.update_button_states()
    
    def update_preview(self, row: int):
        """更新预览"""
        if 0 <= row < len(self.subtitles):
            subtitle = self.subtitles[row]
            preview_text = f"序号: {subtitle.index}\n"
            preview_text += f"时间: {subtitle.start_time} --> {subtitle.end_time}\n"
            preview_text += f"内容: {subtitle.content}"
            
            self.preview_edit.setPlainText(preview_text)
    
    def update_button_states(self):
        """更新按钮状态"""
        has_selection = self.state.current_index >= 0
        can_undo = len(self.state.undo_stack) > 0
        can_redo = len(self.state.redo_stack) > 0
        
        self.edit_btn.setEnabled(has_selection)
        self.delete_btn.setEnabled(has_selection)
        self.duplicate_btn.setEnabled(has_selection)
        self.undo_btn.setEnabled(can_undo)
        self.redo_btn.setEnabled(can_redo)
        
        # 更新撤销/重做按钮文本
        self.undo_btn.setText(f"撤销 ({len(self.state.undo_stack)})")
        self.redo_btn.setText(f"重做 ({len(self.state.redo_stack)})")
    
    def add_subtitle(self):
        """添加字幕"""
        dialog = SubtitleEditDialog(parent=self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            data = dialog.get_subtitle_data()
            
            # 创建新字幕
            new_subtitle = SubtitleItem(
                index=data["index"],
                start_time=data["start_time"],
                end_time=data["end_time"],
                content=data["text"]
            )
            
            # 保存当前状态到撤销栈
            self.save_to_undo_stack()
            
            # 添加字幕
            self.subtitles.append(new_subtitle)
            self.update_table()
            
            # 发送信号
            self.subtitle_added.emit(new_subtitle)
            self.update_status(f"添加了字幕: {new_subtitle.text[:30]}...")
    
    def edit_subtitle(self):
        """编辑字幕"""
        if self.state.current_index < 0:
            return
        
        current_subtitle = self.subtitles[self.state.current_index]
        dialog = SubtitleEditDialog(current_subtitle, parent=self)
        
        if dialog.exec() == QDialog.DialogCode.Accepted:
            data = dialog.get_subtitle_data()
            
            # 保存当前状态到撤销栈
            self.save_to_undo_stack()
            
            # 更新字幕
            updated_subtitle = SubtitleItem(
                index=data["index"],
                start_time=data["start_time"],
                end_time=data["end_time"],
                content=data["text"]
            )
            
            self.subtitles[self.state.current_index] = updated_subtitle
            self.update_table()
            
            # 发送信号
            self.subtitle_modified.emit(self.state.current_index, updated_subtitle)
            self.update_status(f"修改了字幕: {updated_subtitle.text[:30]}...")
    
    def delete_subtitle(self):
        """删除字幕"""
        if self.state.current_index < 0:
            return
        
        reply = QMessageBox.question(
            self, "确认删除",
            f"确定要删除第 {self.state.current_index + 1} 条字幕吗？",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            # 保存当前状态到撤销栈
            self.save_to_undo_stack()
            
            deleted_subtitle = self.subtitles.pop(self.state.current_index)
            self.update_table()
            
            # 发送信号
            self.subtitle_deleted.emit(self.state.current_index)
            self.update_status(f"删除了字幕: {deleted_subtitle.text[:30]}...")
    
    def duplicate_subtitle(self):
        """复制字幕"""
        if self.state.current_index < 0:
            return
        
        original = self.subtitles[self.state.current_index]
        
        # 保存当前状态到撤销栈
        self.save_to_undo_stack()
        
        # 创建副本
        new_subtitle = SubtitleItem(
            index=len(self.subtitles) + 1,
            start_time=original.start_time,
            end_time=original.end_time,
            content=original.content
        )
        
        self.subtitles.append(new_subtitle)
        self.update_table()
        
        # 发送信号
        self.subtitle_added.emit(new_subtitle)
        self.update_status(f"复制了字幕: {new_subtitle.text[:30]}...")
    
    def save_to_undo_stack(self):
        """保存到撤销栈"""
        current_state = {
            "subtitles": [sub.model_dump() for sub in self.subtitles],
            "modified_count": self.state.modified_count
        }
        
        self.state.undo_stack.append(current_state)
        self.state.redo_stack.clear()
        
        # 限制撤销栈大小
        if len(self.state.undo_stack) > 50:
            self.state.undo_stack.pop(0)
        
        self.update_button_states()
    
    def undo(self):
        """撤销"""
        if not self.state.undo_stack:
            return
        
        # 保存当前状态到重做栈
        current_state = {
            "subtitles": [sub.to_srt_format() for sub in self.subtitles],
            "modified_count": self.state.modified_count
        }
        self.state.redo_stack.append(current_state)
        
        # 恢复撤销状态
        undo_state = self.state.undo_stack.pop()
        self.subtitles = self.parse_srt_list(undo_state["subtitles"])
        self.state.modified_count = undo_state["modified_count"]
        
        self.update_table()
        self.update_button_states()
        self.update_status("撤销了上一步操作")
    
    def redo(self):
        """重做"""
        if not self.state.redo_stack:
            return
        
        # 保存当前状态到撤销栈
        current_state = {
            "subtitles": [sub.to_srt_format() for sub in self.subtitles],
            "modified_count": self.state.modified_count
        }
        self.state.undo_stack.append(current_state)
        
        # 恢复重做状态
        redo_state = self.state.redo_stack.pop()
        self.subtitles = self.parse_srt_list(redo_state["subtitles"])
        self.state.modified_count = redo_state["modified_count"]
        
        self.update_table()
        self.update_button_states()
        self.update_status("重做了上一步操作")
    
    def parse_srt_list(self, srt_list: List[str]) -> List[SubtitleItem]:
        """解析SRT格式列表为字幕对象"""
        subtitles = []
        for srt_text in srt_list:
            try:
                lines = srt_text.strip().split('\n')
                if len(lines) >= 3:
                    index = int(lines[0])
                    time_line = lines[1]
                    content = '\n'.join(lines[2:])
                    
                    # 解析时间
                    time_parts = time_line.split(' --> ')
                    if len(time_parts) == 2:
                        start_time = time_parts[0].strip()
                        end_time = time_parts[1].strip()
                        
                        subtitle = SubtitleItem(
                            index=index,
                            start_time=start_time,
                            end_time=end_time,
                            content=content
                        )
                        subtitles.append(subtitle)
            except Exception as e:
                logger.warning(f"解析SRT时出错: {e}")
                continue
        
        return subtitles
    
    def save_subtitles(self):
        """保存字幕"""
        file_path, _ = QFileDialog.getSaveFileName(
            self, "保存字幕", "", "SRT文件 (*.srt);;JSON文件 (*.json)"
        )
        
        if file_path:
            try:
                if file_path.endswith('.srt'):
                    content = self.export_to_srt()
                elif file_path.endswith('.json'):
                    content = self.export_to_json()
                else:
                    content = self.export_to_srt()
                    file_path += '.srt'
                
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                
                self.update_status(f"字幕已保存到: {file_path}")
                self.subtitles_saved.emit(file_path)
                
            except Exception as e:
                QMessageBox.critical(self, "保存失败", f"保存字幕时发生错误: {e}")
    
    def load_subtitles(self):
        """加载字幕"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "加载字幕", "", "SRT文件 (*.srt);;JSON文件 (*.json)"
        )
        
        if file_path:
            try:
                if file_path.endswith('.srt'):
                    self.subtitles = self.import_from_srt(file_path)
                elif file_path.endswith('.json'):
                    self.subtitles = self.import_from_json(file_path)
                else:
                    raise ValueError("不支持的文件格式")
                
                self.update_table()
                self.update_status(f"从 {file_path} 加载了 {len(self.subtitles)} 条字幕")
                self.subtitles_loaded.emit(self.subtitles)
                
            except Exception as e:
                QMessageBox.critical(self, "加载失败", f"加载字幕时发生错误: {e}")
    
    def export_to_srt(self) -> str:
        """导出为SRT格式"""
        srt_content = ""
        for i, subtitle in enumerate(self.subtitles, 1):
            srt_content += f"{i}\n"
            srt_content += f"{subtitle.start_time} --> {subtitle.end_time}\n"
            srt_content += f"{subtitle.content}\n\n"
        return srt_content
    
    def export_to_json(self) -> str:
        """导出为JSON格式"""
        import json
        data = [sub.model_dump() for sub in self.subtitles]
        return json.dumps(data, indent=2, ensure_ascii=False)
    
    def import_from_srt(self, file_path: str) -> List[SubtitleItem]:
        """从SRT文件导入"""
        subtitles = []
        
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read().strip()
        
        # 解析SRT格式
        blocks = content.split('\n\n')
        
        for block in blocks:
            lines = block.strip().split('\n')
            if len(lines) >= 3:
                try:
                    index = int(lines[0])
                    time_line = lines[1]
                    text_lines = lines[2:]
                    
                    # 解析时间
                    time_parts = time_line.split(' --> ')
                    if len(time_parts) == 2:
                        start_time = time_parts[0].strip()
                        end_time = time_parts[1].strip()
                        
                        # 合并文本行
                        text = '\n'.join(text_lines)
                        
                        subtitle = SubtitleItem(
                            index=index,
                            start_time=start_time,
                            end_time=end_time,
                            content=text
                        )
                        subtitles.append(subtitle)
                        
                except Exception as e:
                    logger.warning(f"解析SRT块时出错: {e}")
                    continue
        
        return subtitles
    
    def import_from_json(self, file_path: str) -> List[SubtitleItem]:
        """从JSON文件导入"""
        import json
        
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        subtitles = []
        for item in data:
            try:
                # 手动创建字幕对象，处理不同的字段名
                subtitle = SubtitleItem(
                    index=item.get("index", 1),
                    start_time=item.get("start_time", "00:00:00,000"),
                    end_time=item.get("end_time", "00:00:00,000"),
                    content=item.get("content", item.get("text", ""))
                )
                subtitles.append(subtitle)
            except Exception as e:
                logger.warning(f"解析JSON字幕项时出错: {e}")
                continue
        
        return subtitles
    
    def filter_subtitles(self):
        """过滤字幕"""
        search_text = self.search_edit.text().lower()
        language_filter = self.language_filter.currentText()
        confidence_filter = self.confidence_filter.currentText()
        
        for row in range(self.subtitle_table.rowCount()):
            subtitle = self.subtitles[row]
            show_row = True
            
            # 搜索过滤
            if search_text:
                show_row = search_text in subtitle.content.lower()
            
            # 语言过滤（暂时跳过）
            # if show_row and language_filter != "所有语言":
            #     show_row = subtitle.language == language_filter
            
            # 置信度过滤（暂时跳过）
            # if show_row and confidence_filter != "所有置信度":
            #     confidence = subtitle.confidence
            #     if confidence_filter == "高 (>80%)":
            #         show_row = confidence > 0.8
            #     elif confidence_filter == "中 (60-80%)":
            #         show_row = 0.6 <= confidence <= 0.8
            #     elif confidence_filter == "低 (<60%)":
            #         show_row = confidence < 0.6
            
            self.subtitle_table.setRowHidden(row, not show_row)
    
    def navigate_subtitle(self, direction: int):
        """导航字幕"""
        if not self.subtitles:
            return
        
        new_index = self.state.current_index + direction
        
        if new_index < 0:
            new_index = len(self.subtitles) - 1
        elif new_index >= len(self.subtitles):
            new_index = 0
        
        if new_index != self.state.current_index:
            self.subtitle_table.selectRow(new_index)
            self.subtitle_table.scrollToItem(self.subtitle_table.item(new_index, 0))
    
    def play_current_subtitle(self):
        """播放当前字幕"""
        if self.state.current_index >= 0:
            subtitle = self.subtitles[self.state.current_index]
            self.update_status(f"播放字幕: {subtitle.content[:30]}...")
            # 这里可以集成视频播放功能
    
    def stop_preview(self):
        """停止预览"""
        self.update_status("停止预览")
        # 这里可以停止视频播放
    
    def update_status(self, message: str):
        """更新状态栏"""
        self.status_label.setText(message)
        logger.info(f"SubtitleEditor: {message}")
    
    def get_modified_subtitles(self) -> List[SubtitleItem]:
        """获取修改后的字幕列表"""
        return self.subtitles.copy()
    
    def has_unsaved_changes(self) -> bool:
        """检查是否有未保存的更改"""
        return self.state.modified_count > 0


class SubtitleEditorWindow(QMainWindow):
    """字幕编辑器窗口"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("VisionSub - 字幕编辑器")
        self.setGeometry(200, 200, 1000, 700)
        
        # 创建编辑器组件
        self.editor = SubtitleEditorWidget()
        self.setCentralWidget(self.editor)
        
        # 创建菜单栏
        self.create_menus()
        
        # 连接信号
        self.connect_signals()
    
    def create_menus(self):
        """创建菜单栏"""
        menubar = self.menuBar()
        
        # 文件菜单
        file_menu = menubar.addMenu("文件")
        
        save_action = QAction("保存", self)
        save_action.setShortcut(QKeySequence("Ctrl+S"))
        save_action.triggered.connect(self.editor.save_subtitles)
        
        load_action = QAction("加载", self)
        load_action.setShortcut(QKeySequence("Ctrl+O"))
        load_action.triggered.connect(self.editor.load_subtitles)
        
        exit_action = QAction("退出", self)
        exit_action.setShortcut(QKeySequence("Ctrl+Q"))
        exit_action.triggered.connect(self.close)
        
        file_menu.addAction(save_action)
        file_menu.addAction(load_action)
        file_menu.addSeparator()
        file_menu.addAction(exit_action)
        
        # 编辑菜单
        edit_menu = menubar.addMenu("编辑")
        
        add_action = QAction("添加字幕", self)
        add_action.setShortcut(QKeySequence("Ctrl+N"))
        add_action.triggered.connect(self.editor.add_subtitle)
        
        edit_action = QAction("编辑字幕", self)
        edit_action.setShortcut(QKeySequence("Ctrl+E"))
        edit_action.triggered.connect(self.editor.edit_subtitle)
        
        delete_action = QAction("删除字幕", self)
        delete_action.setShortcut(QKeySequence("Delete"))
        delete_action.triggered.connect(self.editor.delete_subtitle)
        
        duplicate_action = QAction("复制字幕", self)
        duplicate_action.setShortcut(QKeySequence("Ctrl+D"))
        duplicate_action.triggered.connect(self.editor.duplicate_subtitle)
        
        undo_action = QAction("撤销", self)
        undo_action.setShortcut(QKeySequence("Ctrl+Z"))
        undo_action.triggered.connect(self.editor.undo)
        
        redo_action = QAction("重做", self)
        redo_action.setShortcut(QKeySequence("Ctrl+Y"))
        redo_action.triggered.connect(self.editor.redo)
        
        edit_menu.addAction(add_action)
        edit_menu.addAction(edit_action)
        edit_menu.addAction(delete_action)
        edit_menu.addAction(duplicate_action)
        edit_menu.addSeparator()
        edit_menu.addAction(undo_action)
        edit_menu.addAction(redo_action)
    
    def connect_signals(self):
        """连接信号"""
        self.editor.subtitle_modified.connect(self.on_subtitle_modified)
        self.editor.subtitle_added.connect(self.on_subtitle_added)
        self.editor.subtitle_deleted.connect(self.on_subtitle_deleted)
    
    def on_subtitle_modified(self, index: int, subtitle: SubtitleItem):
        """字幕修改事件"""
        logger.info(f"Subtitle modified at index {index}: {subtitle.text[:30]}...")
    
    def on_subtitle_added(self, subtitle: SubtitleItem):
        """字幕添加事件"""
        logger.info(f"Subtitle added: {subtitle.text[:30]}...")
    
    def on_subtitle_deleted(self, index: int):
        """字幕删除事件"""
        logger.info(f"Subtitle deleted at index {index}")
    
    def set_subtitles(self, subtitles: List[SubtitleItem]):
        """设置字幕列表"""
        self.editor.set_subtitles(subtitles)
    
    def get_modified_subtitles(self) -> List[SubtitleItem]:
        """获取修改后的字幕列表"""
        return self.editor.get_modified_subtitles()
    
    def closeEvent(self, event):
        """关闭事件"""
        if self.editor.has_unsaved_changes():
            reply = QMessageBox.question(
                self, "未保存的更改",
                "有未保存的更改，是否要保存？",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No | QMessageBox.StandardButton.Cancel
            )
            
            if reply == QMessageBox.StandardButton.Yes:
                self.editor.save_subtitles()
            elif reply == QMessageBox.StandardButton.Cancel:
                event.ignore()
                return
        
        event.accept()