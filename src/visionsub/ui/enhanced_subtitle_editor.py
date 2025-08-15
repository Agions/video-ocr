"""
Enhanced Subtitle Editor with Modern UI/UX and Security Features
"""
import json
import logging
import re
from datetime import timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
from PyQt6.QtCore import (
    Qt, pyqtSignal, QSize, QPoint, QRect, QTimer, 
    QPropertyAnimation, QEasingCurve, QThread, pyqtSlot,
    QAbstractTableModel, QModelIndex, QSortFilterProxyModel
)
from PyQt6.QtGui import (
    QFont, QColor, QPalette, QPainter, QBrush, QPen,
    QTextCharFormat, QTextCursor, QImage, QPixmap,
    QSyntaxHighlighter, QTextDocument, QStandardItemModel,
    QStandardItem, QKeySequence, QAction, QIcon, QShortcut, QActionGroup
)
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QTextEdit, QTableWidget, QTableWidgetItem, QHeaderView,
    QSplitter, QFrame, QScrollArea, QGroupBox,
    QComboBox, QSpinBox, QDoubleSpinBox, QCheckBox,
    QProgressBar, QTabWidget, QListWidget, QListWidgetItem,
    QStyledItemDelegate, QStyleOptionViewItem, QApplication,
    QMainWindow, QDialog, QFileDialog, QMessageBox, QMenu,
    QToolBar, QStatusBar, QMenuBar,
    QTimeEdit, QLineEdit, QFormLayout, QDialogButtonBox,
    QAbstractItemView, QSlider
)

from visionsub.models.subtitle import SubtitleItem
from visionsub.ui.theme_system import (
    ThemeManager, get_theme_manager, StyledWidget, 
    Card, Button, ThemeColors
)

logger = logging.getLogger(__name__)


class SubtitleEditAction(Enum):
    """Subtitle edit actions"""
    ADD = "add"
    EDIT = "edit"
    DELETE = "delete"
    DUPLICATE = "duplicate"
    MERGE = "merge"
    SPLIT = "split"
    SHIFT_TIME = "shift_time"
    ADJUST_DURATION = "adjust_duration"


@dataclass
class SubtitleEditState:
    """Subtitle editing state"""
    is_editing: bool = False
    current_index: int = -1
    undo_stack: List[Dict[str, Any]] = field(default_factory=list)
    redo_stack: List[Dict[str, Any]] = field(default_factory=list)
    modified_count: int = 0
    last_action: Optional[SubtitleEditAction] = None
    selection_start: int = -1
    selection_end: int = -1


class SecureSubtitleValidator:
    """Validator for subtitle content"""
    
    def __init__(self):
        self.max_text_length = 5000
        self.max_subtitle_count = 10000
        self.max_duration_hours = 24
        self.allowed_chars = set(
            'abcdefghijklmnopqrstuvwxyz'
            'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
            '0123456789'
            ' .,!?;:()[]{}"\'-'
            'áéíóúÁÉÍÓÚñÑ'
            'äöüÄÖÜß'
            'àèìòùÀÈÌÒÙ'
            'âêîôûÂÊÎÔÛ'
            'çÇ'
            '¿¡'
            '€£¥₩'
            '\n\r\t'
        )
        
    def validate_text(self, text: str) -> str:
        """Validate and sanitize subtitle text"""
        if not text:
            return ""
            
        # Remove control characters except newlines and tabs
        text = ''.join(char for char in text if ord(char) >= 32 or char in '\n\r\t')
        
        # Remove potentially dangerous characters
        text = ''.join(char for char in text if char in self.allowed_chars)
        
        # Limit length
        if len(text) > self.max_text_length:
            text = text[:self.max_text_length] + "..."
            
        return text.strip()
        
    def validate_time_format(self, time_str: str) -> bool:
        """Validate SRT time format"""
        pattern = r'^\d{2}:\d{2}:\d{2},\d{3}$'
        return re.match(pattern, time_str) is not None
        
    def validate_subtitle_item(self, subtitle: SubtitleItem) -> bool:
        """Validate subtitle item"""
        try:
            # Validate text
            if not subtitle.content or len(subtitle.content) > self.max_text_length:
                return False
                
            # Validate time format
            if not self.validate_time_format(subtitle.start_time):
                return False
            if not self.validate_time_format(subtitle.end_time):
                return False
                
            # Validate time order
            start_ms = self.time_to_ms(subtitle.start_time)
            end_ms = self.time_to_ms(subtitle.end_time)
            
            if start_ms >= end_ms:
                return False
                
            # Validate duration
            duration_ms = end_ms - start_ms
            max_duration_ms = self.max_duration_hours * 3600 * 1000
            
            if duration_ms > max_duration_ms:
                return False
                
            return True
            
        except Exception:
            return False
            
    def time_to_ms(self, time_str: str) -> int:
        """Convert SRT time to milliseconds"""
        try:
            time_part, ms_part = time_str.split(',')
            h, m, s = time_part.split(':')
            ms = int(ms_part)
            return int(h) * 3600 * 1000 + int(m) * 60 * 1000 + int(s) * 1000 + ms
        except:
            return 0


class SubtitleTableModel(QAbstractTableModel):
    """Enhanced subtitle table model"""
    
    def __init__(self, subtitles: List[SubtitleItem] = None):
        super().__init__()
        self.subtitles = subtitles or []
        self.headers = ["序号", "开始时间", "结束时间", "内容", "时长"]
        self.validator = SecureSubtitleValidator()
        
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
                return subtitle.start_time
            elif col == 2:  # 结束时间
                return subtitle.end_time
            elif col == 3:  # 内容
                content = subtitle.content
                if len(content) > 80:
                    content = content[:77] + "..."
                return content
            elif col == 4:  # 时长
                start_ms = self.validator.time_to_ms(subtitle.start_time)
                end_ms = self.validator.time_to_ms(subtitle.end_time)
                duration_ms = end_ms - start_ms
                duration_sec = duration_ms / 1000
                return f"{duration_sec:.1f}s"
                
        elif role == Qt.ItemDataRole.TextAlignmentRole:
            if col in [0, 4]:  # 序号、时长居中对齐
                return Qt.AlignmentFlag.AlignCenter
            elif col in [1, 2]:  # 时间居右对齐
                return Qt.AlignmentFlag.AlignRight
            return Qt.AlignmentFlag.AlignLeft
            
        elif role == Qt.ItemDataRole.ToolTipRole:
            if col == 3:  # 内容显示完整tooltip
                return subtitle.content
                
        elif role == Qt.ItemDataRole.BackgroundRole:
            # Color based on duration
            if col == 4:  # 时长列
                start_ms = self.validator.time_to_ms(subtitle.start_time)
                end_ms = self.validator.time_to_ms(subtitle.end_time)
                duration_ms = end_ms - start_ms
                
                if duration_ms < 1000:  # Less than 1 second
                    return QColor(255, 182, 193, 100)  # Light red
                elif duration_ms > 10000:  # More than 10 seconds
                    return QColor(144, 238, 144, 100)  # Light green
                else:
                    return QColor(255, 255, 224, 100)  # Light yellow
                    
        return None
        
    def headerData(self, section: int, orientation: Qt.Orientation, role: int = Qt.ItemDataRole.DisplayRole):
        if orientation == Qt.Orientation.Horizontal and role == Qt.ItemDataRole.DisplayRole:
            return self.headers[section]
        return None
        
    def setData(self, index: QModelIndex, value: Any, role: int = Qt.ItemDataRole.EditRole) -> bool:
        if not index.isValid() or role != Qt.ItemDataRole.EditRole:
            return False
            
        row = index.row()
        col = index.column()
        
        if row >= len(self.subtitles):
            return False
            
        subtitle = self.subtitles[row]
        
        try:
            if col == 0:  # 序号
                new_index = int(value)
                if new_index > 0:
                    subtitle.index = new_index
            elif col == 1:  # 开始时间
                if self.validator.validate_time_format(value):
                    subtitle.start_time = value
            elif col == 2:  # 结束时间
                if self.validator.validate_time_format(value):
                    subtitle.end_time = value
            elif col == 3:  # 内容
                subtitle.content = self.validator.validate_text(value)
                
            self.dataChanged.emit(index, index)
            return True
            
        except Exception:
            return False
            
    def flags(self, index: QModelIndex):
        base_flags = Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable
        
        if index.column() in [0, 1, 2, 3]:  # Allow editing
            base_flags |= Qt.ItemFlag.ItemIsEditable
            
        return base_flags
        
    def insertRows(self, row: int, count: int, parent=None) -> bool:
        self.beginInsertRows(parent, row, row + count - 1)
        
        for i in range(count):
            new_subtitle = SubtitleItem(
                index=row + i + 1,
                start_time="00:00:00,000",
                end_time="00:00:01,000",
                content="新字幕"
            )
            self.subtitles.insert(row + i, new_subtitle)
            
        self.endInsertRows()
        return True
        
    def removeRows(self, row: int, count: int, parent=None) -> bool:
        self.beginRemoveRows(parent, row, row + count - 1)
        
        for _ in range(count):
            if row < len(self.subtitles):
                self.subtitles.pop(row)
                
        self.endRemoveRows()
        return True
        
    def update_data(self, subtitles: List[SubtitleItem]):
        """Update all data"""
        self.beginResetModel()
        self.subtitles = subtitles
        self.endResetModel()
        
    def get_subtitle_at(self, row: int) -> Optional[SubtitleItem]:
        """Get subtitle at row"""
        if 0 <= row < len(self.subtitles):
            return self.subtitles[row]
        return None
        
    def update_subtitle_at(self, row: int, subtitle: SubtitleItem):
        """Update subtitle at row"""
        if 0 <= row < len(self.subtitles):
            self.subtitles[row] = subtitle
            self.dataChanged.emit(
                self.index(row, 0),
                self.index(row, self.columnCount() - 1)
            )


class SubtitleSearchProxyModel(QSortFilterProxyModel):
    """Proxy model for searching and filtering subtitles"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.search_text = ""
        self.min_duration = 0
        self.max_duration = float('inf')
        
    def set_search_text(self, text: str):
        """Set search text"""
        self.search_text = text.lower()
        self.invalidateFilter()
        
    def set_duration_range(self, min_duration: float, max_duration: float):
        """Set duration range filter"""
        self.min_duration = min_duration
        self.max_duration = max_duration
        self.invalidateFilter()
        
    def filterAcceptsRow(self, source_row: int, source_parent: QModelIndex) -> bool:
        """Check if row should be accepted"""
        source_model = self.sourceModel()
        if not source_model:
            return False
            
        # Get subtitle
        subtitle = source_model.get_subtitle_at(source_row)
        if not subtitle:
            return False
            
        # Apply text search
        if self.search_text:
            if self.search_text not in subtitle.content.lower():
                return False
                
        # Apply duration filter
        if hasattr(source_model, 'validator'):
            validator = source_model.validator
            start_ms = validator.time_to_ms(subtitle.start_time)
            end_ms = validator.time_to_ms(subtitle.end_time)
            duration_ms = end_ms - start_ms
            duration_sec = duration_ms / 1000
            
            if not (self.min_duration <= duration_sec <= self.max_duration):
                return False
                
        return True


class EnhancedSubtitleEditDialog(QDialog):
    """Enhanced subtitle edit dialog"""
    
    def __init__(self, subtitle: SubtitleItem = None, parent=None):
        super().__init__(parent)
        self.subtitle = subtitle
        self.validator = SecureSubtitleValidator()
        self.original_subtitle = None
        
        self.setWindowTitle("编辑字幕" if subtitle else "添加字幕")
        self.setModal(True)
        self.resize(600, 500)
        
        self.setup_ui()
        
        if subtitle:
            self.original_subtitle = SubtitleItem(
                index=subtitle.index,
                start_time=subtitle.start_time,
                end_time=subtitle.end_time,
                content=subtitle.content
            )
            self.load_subtitle_data()
            
    def setup_ui(self):
        """Setup dialog UI"""
        layout = QVBoxLayout(self)
        layout.setSpacing(16)
        layout.setContentsMargins(20, 20, 20, 20)
        
        # Basic info
        basic_group = QGroupBox("基本信息")
        basic_layout = QFormLayout(basic_group)
        
        self.index_edit = QLineEdit()
        self.index_edit.setPlaceholderText("字幕序号")
        basic_layout.addRow("序号:", self.index_edit)
        
        # Time editing
        time_layout = QHBoxLayout()
        self.start_time_edit = QLineEdit()
        self.start_time_edit.setPlaceholderText("HH:MM:SS,mmm")
        self.start_time_edit.setMaximumWidth(120)
        
        self.end_time_edit = QLineEdit()
        self.end_time_edit.setPlaceholderText("HH:MM:SS,mmm")
        self.end_time_edit.setMaximumWidth(120)
        
        time_layout.addWidget(QLabel("开始:"))
        time_layout.addWidget(self.start_time_edit)
        time_layout.addWidget(QLabel("结束:"))
        time_layout.addWidget(self.end_time_edit)
        time_layout.addStretch()
        
        basic_layout.addRow("时间:", time_layout)
        
        # Duration display
        self.duration_label = QLabel("时长: 0.0s")
        self.duration_label.setStyleSheet("color: #64748b;")
        basic_layout.addRow("时长:", self.duration_label)
        
        layout.addWidget(basic_group)
        
        # Text content
        text_group = QGroupBox("字幕内容")
        text_layout = QVBoxLayout(text_group)
        
        self.text_edit = QTextEdit()
        self.text_edit.setPlaceholderText("请输入字幕内容...")
        self.text_edit.setMinimumHeight(150)
        self.text_edit.textChanged.connect(self.update_duration)
        text_layout.addWidget(self.text_edit)
        
        # Text statistics
        stats_layout = QHBoxLayout()
        self.char_count_label = QLabel("字符数: 0")
        self.word_count_label = QLabel("单词数: 0")
        self.line_count_label = QLabel("行数: 0")
        
        stats_layout.addWidget(self.char_count_label)
        stats_layout.addWidget(self.word_count_label)
        stats_layout.addWidget(self.line_count_label)
        stats_layout.addStretch()
        
        text_layout.addLayout(stats_layout)
        layout.addWidget(text_group)
        
        # Validation status
        self.validation_label = QLabel("✓ 输入有效")
        self.validation_label.setStyleSheet("color: #10b981; font-weight: bold;")
        layout.addWidget(self.validation_label)
        
        # Buttons
        button_layout = QHBoxLayout()
        
        self.preview_button = QPushButton("预览")
        self.preview_button.clicked.connect(self.preview_subtitle)
        button_layout.addWidget(self.preview_button)
        
        button_layout.addStretch()
        
        self.ok_button = QPushButton("确定")
        self.ok_button.clicked.connect(self.accept)
        self.ok_button.setDefault(True)
        
        self.cancel_button = QPushButton("取消")
        self.cancel_button.clicked.connect(self.reject)
        
        button_layout.addWidget(self.cancel_button)
        button_layout.addWidget(self.ok_button)
        
        layout.addLayout(button_layout)
        
        # Connect signals
        self.start_time_edit.textChanged.connect(self.validate_input)
        self.end_time_edit.textChanged.connect(self.validate_input)
        self.text_edit.textChanged.connect(self.validate_input)
        
        # Set up validation timer
        self.validation_timer = QTimer()
        self.validation_timer.timeout.connect(self.validate_input)
        self.validation_timer.start(500)
        
    def load_subtitle_data(self):
        """Load subtitle data"""
        if self.subtitle:
            self.index_edit.setText(str(self.subtitle.index))
            self.start_time_edit.setText(self.subtitle.start_time)
            self.end_time_edit.setText(self.subtitle.end_time)
            self.text_edit.setPlainText(self.subtitle.content)
            self.update_statistics()
            self.update_duration()
            
    def validate_input(self):
        """Validate input fields"""
        try:
            # Get data
            index = int(self.index_edit.text()) if self.index_edit.text() else 1
            start_time = self.start_time_edit.text()
            end_time = self.end_time_edit.text()
            content = self.text_edit.toPlainText()
            
            # Create temporary subtitle for validation
            temp_subtitle = SubtitleItem(
                index=index,
                start_time=start_time,
                end_time=end_time,
                content=content
            )
            
            # Validate
            if self.validator.validate_subtitle_item(temp_subtitle):
                self.validation_label.setText("✓ 输入有效")
                self.validation_label.setStyleSheet("color: #10b981; font-weight: bold;")
                self.ok_button.setEnabled(True)
            else:
                self.validation_label.setText("✗ 输入无效")
                self.validation_label.setStyleSheet("color: #ef4444; font-weight: bold;")
                self.ok_button.setEnabled(False)
                
        except Exception:
            self.validation_label.setText("✗ 输入无效")
            self.validation_label.setStyleSheet("color: #ef4444; font-weight: bold;")
            self.ok_button.setEnabled(False)
            
    def update_statistics(self):
        """Update text statistics"""
        text = self.text_edit.toPlainText()
        
        char_count = len(text)
        word_count = len(text.split())
        line_count = len(text.split('\n'))
        
        self.char_count_label.setText(f"字符数: {char_count}")
        self.word_count_label.setText(f"单词数: {word_count}")
        self.line_count_label.setText(f"行数: {line_count}")
        
    def update_duration(self):
        """Update duration display"""
        try:
            start_time = self.start_time_edit.text()
            end_time = self.end_time_edit.text()
            
            if self.validator.validate_time_format(start_time) and self.validator.validate_time_format(end_time):
                start_ms = self.validator.time_to_ms(start_time)
                end_ms = self.validator.time_to_ms(end_time)
                duration_ms = end_ms - start_ms
                duration_sec = duration_ms / 1000
                
                self.duration_label.setText(f"时长: {duration_sec:.1f}s")
                
                # Color code duration
                if duration_ms < 1000:
                    self.duration_label.setStyleSheet("color: #ef4444; font-weight: bold;")
                elif duration_ms > 10000:
                    self.duration_label.setStyleSheet("color: #10b981; font-weight: bold;")
                else:
                    self.duration_label.setStyleSheet("color: #64748b;")
            else:
                self.duration_label.setText("时长: --")
                self.duration_label.setStyleSheet("color: #64748b;")
                
        except Exception:
            self.duration_label.setText("时长: --")
            self.duration_label.setStyleSheet("color: #64748b;")
            
    def preview_subtitle(self):
        """Preview subtitle"""
        try:
            subtitle_data = self.get_subtitle_data()
            
            # Create preview dialog
            preview_text = f"""
            <b>序号:</b> {subtitle_data['index']}<br>
            <b>时间:</b> {subtitle_data['start_time']} --> {subtitle_data['end_time']}<br>
            <b>内容:</b> {subtitle_data['text']}
            """
            
            QMessageBox.information(self, "字幕预览", preview_text)
            
        except Exception as e:
            QMessageBox.warning(self, "预览错误", f"预览字幕时出错: {e}")
            
    def get_subtitle_data(self) -> Dict[str, Any]:
        """Get subtitle data from dialog"""
        return {
            'index': int(self.index_edit.text()) if self.index_edit.text() else 1,
            'start_time': self.start_time_edit.text(),
            'end_time': self.end_time_edit.text(),
            'text': self.text_edit.toPlainText()
        }
        
    def accept(self):
        """Accept dialog"""
        try:
            data = self.get_subtitle_data()
            
            # Validate data
            temp_subtitle = SubtitleItem(
                index=data['index'],
                start_time=data['start_time'],
                end_time=data['end_time'],
                content=data['text']
            )
            
            if not self.validator.validate_subtitle_item(temp_subtitle):
                QMessageBox.warning(self, "验证错误", "字幕数据无效，请检查输入")
                return
                
            super().accept()
            
        except Exception as e:
            QMessageBox.warning(self, "保存错误", f"保存字幕时出错: {e}")


class EnhancedSubtitleEditor(StyledWidget):
    """Enhanced subtitle editor widget"""
    
    # Signals
    subtitle_modified = pyqtSignal(int, SubtitleItem)
    subtitle_added = pyqtSignal(SubtitleItem)
    subtitle_deleted = pyqtSignal(int)
    subtitles_saved = pyqtSignal(str)
    subtitles_loaded = pyqtSignal(list)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.theme_manager = get_theme_manager()
        self.validator = SecureSubtitleValidator()
        self.state = SubtitleEditState()
        self.subtitles: List[SubtitleItem] = []
        
        self.setup_ui()
        self.setup_shortcuts()
        self.connect_signals()
        
    def setup_ui(self):
        """Setup enhanced subtitle editor UI"""
        layout = QVBoxLayout(self)
        layout.setSpacing(16)
        layout.setContentsMargins(20, 20, 20, 20)
        
        # Toolbar
        toolbar_layout = QHBoxLayout()
        
        # Edit actions
        self.add_button = Button("➕ 添加", "primary")
        self.add_button.clicked.connect(self.add_subtitle)
        toolbar_layout.addWidget(self.add_button)
        
        self.edit_button = Button("✏️ 编辑", "secondary")
        self.edit_button.clicked.connect(self.edit_subtitle)
        toolbar_layout.addWidget(self.edit_button)
        
        self.delete_button = Button("🗑️ 删除", "error")
        self.delete_button.clicked.connect(self.delete_subtitle)
        toolbar_layout.addWidget(self.delete_button)
        
        self.duplicate_button = Button("📋 复制", "secondary")
        self.duplicate_button.clicked.connect(self.duplicate_subtitle)
        toolbar_layout.addWidget(self.duplicate_button)
        
        toolbar_layout.addSpacing(10)
        
        # Undo/redo
        self.undo_button = Button("↶ 撤销", "secondary")
        self.undo_button.clicked.connect(self.undo)
        toolbar_layout.addWidget(self.undo_button)
        
        self.redo_button = Button("↷ 重做", "secondary")
        self.redo_button.clicked.connect(self.redo)
        toolbar_layout.addWidget(self.redo_button)
        
        toolbar_layout.addSpacing(10)
        
        # Save/Load
        self.save_button = Button("💾 保存", "success")
        self.save_button.clicked.connect(self.save_subtitles)
        toolbar_layout.addWidget(self.save_button)
        
        self.load_button = Button("📁 加载", "secondary")
        self.load_button.clicked.connect(self.load_subtitles)
        toolbar_layout.addWidget(self.load_button)
        
        toolbar_layout.addStretch()
        
        layout.addLayout(toolbar_layout)
        
        # Search and filter
        filter_card = Card("搜索和过滤")
        filter_layout = QVBoxLayout(filter_card)
        
        search_layout = QHBoxLayout()
        search_layout.addWidget(QLabel("搜索:"))
        
        self.search_edit = QLineEdit()
        self.search_edit.setPlaceholderText("搜索字幕内容...")
        search_layout.addWidget(self.search_edit)
        
        filter_layout.addLayout(search_layout)
        
        # Duration filter
        duration_layout = QHBoxLayout()
        duration_layout.addWidget(QLabel("时长范围:"))
        
        self.min_duration_slider = QSlider(Qt.Orientation.Horizontal)
        self.min_duration_slider.setRange(0, 300)  # 0-30 seconds
        self.min_duration_slider.setValue(0)
        self.min_duration_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        
        self.max_duration_slider = QSlider(Qt.Orientation.Horizontal)
        self.max_duration_slider.setRange(0, 300)  # 0-30 seconds
        self.max_duration_slider.setValue(300)
        self.max_duration_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        
        self.min_duration_label = QLabel("0s")
        self.max_duration_label = QLabel("30s")
        
        duration_layout.addWidget(self.min_duration_label)
        duration_layout.addWidget(self.min_duration_slider)
        duration_layout.addWidget(QLabel("-"))
        duration_layout.addWidget(self.max_duration_slider)
        duration_layout.addWidget(self.max_duration_label)
        
        filter_layout.addLayout(duration_layout)
        
        layout.addWidget(filter_card)
        
        # Main content
        content_splitter = QSplitter(Qt.Orientation.Vertical)
        
        # Table view
        table_group = QGroupBox("字幕列表")
        table_layout = QVBoxLayout(table_group)
        
        # Create model and proxy
        self.table_model = SubtitleTableModel()
        self.proxy_model = SubtitleSearchProxyModel()
        self.proxy_model.setSourceModel(self.table_model)
        
        self.table_view = QTableWidget()
        self.table_view.setModel(self.proxy_model)
        self.table_view.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table_view.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.table_view.setEditTriggers(QAbstractItemView.EditTrigger.DoubleClicked)
        self.table_view.setAlternatingRowColors(True)
        self.table_view.horizontalHeader().setStretchLastSection(True)
        
        table_layout.addWidget(self.table_view)
        content_splitter.addWidget(table_group)
        
        # Preview area
        preview_group = QGroupBox("预览")
        preview_layout = QVBoxLayout(preview_group)
        
        self.preview_edit = QTextEdit()
        self.preview_edit.setReadOnly(True)
        self.preview_edit.setMaximumHeight(150)
        self.preview_edit.setStyleSheet("""
            QTextEdit {
                background: #f8fafc;
                border: 1px solid #e2e8f0;
                border-radius: 8px;
                padding: 16px;
                font-family: 'Monaco', 'Menlo', 'Consolas', monospace;
                font-size: 14px;
            }
        """)
        preview_layout.addWidget(self.preview_edit)
        
        # Preview controls
        preview_control_layout = QHBoxLayout()
        self.prev_button = Button("⬆️ 上一个", "secondary")
        self.prev_button.clicked.connect(lambda: self.navigate_subtitle(-1))
        preview_control_layout.addWidget(self.prev_button)
        
        self.next_button = Button("⬇️ 下一个", "secondary")
        self.next_button.clicked.connect(lambda: self.navigate_subtitle(1))
        preview_control_layout.addWidget(self.next_button)
        
        preview_control_layout.addStretch()
        
        preview_layout.addLayout(preview_control_layout)
        content_splitter.addWidget(preview_group)
        
        content_splitter.setSizes([400, 150])
        layout.addWidget(content_splitter)
        
        # Status bar
        self.status_label = QLabel("就绪")
        layout.addWidget(self.status_label)
        
        # Initialize button states
        self.update_button_states()
        
    def setup_shortcuts(self):
        """Setup keyboard shortcuts"""
        from PyQt6.QtGui import QShortcut
        
        # Edit shortcuts
        QShortcut(QKeySequence("Ctrl+N"), self, self.add_subtitle)
        QShortcut(QKeySequence("Ctrl+E"), self, self.edit_subtitle)
        QShortcut(QKeySequence.Delete, self, self.delete_subtitle)
        QShortcut(QKeySequence("Ctrl+D"), self, self.duplicate_subtitle)
        
        # Undo/redo shortcuts
        QShortcut(QKeySequence("Ctrl+Z"), self, self.undo)
        QShortcut(QKeySequence("Ctrl+Y"), self, self.redo)
        
        # Save/load shortcuts
        QShortcut(QKeySequence("Ctrl+S"), self, self.save_subtitles)
        QShortcut(QKeySequence("Ctrl+O"), self, self.load_subtitles)
        
        # Navigation shortcuts
        QShortcut(QKeySequence.Up, self, lambda: self.navigate_subtitle(-1))
        QShortcut(QKeySequence.Down, self, lambda: self.navigate_subtitle(1))
        
    def connect_signals(self):
        """Connect signals"""
        # Table signals
        self.table_view.selectionModel().currentChanged.connect(self.on_selection_changed)
        self.table_view.doubleClicked.connect(self.edit_subtitle)
        
        # Search signals
        self.search_edit.textChanged.connect(self.on_search_changed)
        self.min_duration_slider.valueChanged.connect(self.on_duration_filter_changed)
        self.max_duration_slider.valueChanged.connect(self.on_duration_filter_changed)
        
    def set_subtitles(self, subtitles: List[SubtitleItem]):
        """Set subtitle list"""
        # Validate and sanitize subtitles
        validated_subtitles = []
        for subtitle in subtitles:
            if self.validator.validate_subtitle_item(subtitle):
                validated_subtitles.append(subtitle)
            else:
                logger.warning(f"Invalid subtitle skipped: {subtitle}")
                
        self.subtitles = validated_subtitles
        self.table_model.update_data(self.subtitles)
        self.update_status(f"加载了 {len(self.subtitles)} 条字幕")
        
    def on_selection_changed(self, current: QModelIndex, previous: QModelIndex):
        """Handle selection change"""
        if current.isValid():
            source_row = self.proxy_model.mapToSource(current).row()
            self.state.current_index = source_row
            self.update_preview(source_row)
        else:
            self.state.current_index = -1
            self.preview_edit.clear()
            
        self.update_button_states()
        
    def on_search_changed(self, text: str):
        """Handle search text change"""
        self.proxy_model.set_search_text(text)
        
    def on_duration_filter_changed(self):
        """Handle duration filter change"""
        min_duration = self.min_duration_slider.value() / 10.0  # Convert to seconds
        max_duration = self.max_duration_slider.value() / 10.0  # Convert to seconds
        
        self.min_duration_label.setText(f"{min_duration:.1f}s")
        self.max_duration_label.setText(f"{max_duration:.1f}s")
        
        self.proxy_model.set_duration_range(min_duration, max_duration)
        
    def update_preview(self, row: int):
        """Update preview for given row"""
        if 0 <= row < len(self.subtitles):
            subtitle = self.subtitles[row]
            preview_text = f"序号: {subtitle.index}\n"
            preview_text += f"时间: {subtitle.start_time} --> {subtitle.end_time}\n"
            preview_text += f"内容: {subtitle.content}"
            
            self.preview_edit.setPlainText(preview_text)
            
    def update_button_states(self):
        """Update button states"""
        has_selection = self.state.current_index >= 0
        can_undo = len(self.state.undo_stack) > 0
        can_redo = len(self.state.redo_stack) > 0
        
        self.edit_button.setEnabled(has_selection)
        self.delete_button.setEnabled(has_selection)
        self.duplicate_button.setEnabled(has_selection)
        self.undo_button.setEnabled(can_undo)
        self.redo_button.setEnabled(can_redo)
        
        # Update button text
        self.undo_button.setText(f"↶ 撤销 ({len(self.state.undo_stack)})")
        self.redo_button.setText(f"↷ 重做 ({len(self.state.redo_stack)})")
        
    def save_to_undo_stack(self):
        """Save current state to undo stack"""
        current_state = {
            'subtitles': [sub.model_dump() for sub in self.subtitles],
            'modified_count': self.state.modified_count
        }
        
        self.state.undo_stack.append(current_state)
        self.state.redo_stack.clear()
        
        # Limit stack size
        if len(self.state.undo_stack) > 50:
            self.state.undo_stack.pop(0)
            
        self.update_button_states()
        
    def add_subtitle(self):
        """Add new subtitle"""
        dialog = EnhancedSubtitleEditDialog(parent=self)
        
        if dialog.exec() == QDialog.DialogCode.Accepted:
            data = dialog.get_subtitle_data()
            
            # Save to undo stack
            self.save_to_undo_stack()
            
            # Create new subtitle
            new_subtitle = SubtitleItem(
                index=data['index'],
                start_time=data['start_time'],
                end_time=data['end_time'],
                content=data['text']
            )
            
            # Add subtitle
            self.subtitles.append(new_subtitle)
            self.table_model.update_data(self.subtitles)
            
            # Emit signal
            self.subtitle_added.emit(new_subtitle)
            self.update_status(f"添加了字幕: {new_subtitle.content[:30]}...")
            
    def edit_subtitle(self):
        """Edit selected subtitle"""
        if self.state.current_index < 0:
            return
            
        current_subtitle = self.subtitles[self.state.current_index]
        dialog = EnhancedSubtitleEditDialog(current_subtitle, parent=self)
        
        if dialog.exec() == QDialog.DialogCode.Accepted:
            data = dialog.get_subtitle_data()
            
            # Save to undo stack
            self.save_to_undo_stack()
            
            # Update subtitle
            updated_subtitle = SubtitleItem(
                index=data['index'],
                start_time=data['start_time'],
                end_time=data['end_time'],
                content=data['text']
            )
            
            self.subtitles[self.state.current_index] = updated_subtitle
            self.table_model.update_data(self.subtitles)
            
            # Emit signal
            self.subtitle_modified.emit(self.state.current_index, updated_subtitle)
            self.update_status(f"修改了字幕: {updated_subtitle.content[:30]}...")
            
    def delete_subtitle(self):
        """Delete selected subtitle"""
        if self.state.current_index < 0:
            return
            
        reply = QMessageBox.question(
            self, "确认删除",
            f"确定要删除第 {self.state.current_index + 1} 条字幕吗？",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            # Save to undo stack
            self.save_to_undo_stack()
            
            deleted_subtitle = self.subtitles.pop(self.state.current_index)
            self.table_model.update_data(self.subtitles)
            
            # Emit signal
            self.subtitle_deleted.emit(self.state.current_index)
            self.update_status(f"删除了字幕: {deleted_subtitle.content[:30]}...")
            
    def duplicate_subtitle(self):
        """Duplicate selected subtitle"""
        if self.state.current_index < 0:
            return
            
        original = self.subtitles[self.state.current_index]
        
        # Save to undo stack
        self.save_to_undo_stack()
        
        # Create duplicate
        new_subtitle = SubtitleItem(
            index=len(self.subtitles) + 1,
            start_time=original.start_time,
            end_time=original.end_time,
            content=original.content
        )
        
        self.subtitles.append(new_subtitle)
        self.table_model.update_data(self.subtitles)
        
        # Emit signal
        self.subtitle_added.emit(new_subtitle)
        self.update_status(f"复制了字幕: {new_subtitle.content[:30]}...")
        
    def undo(self):
        """Undo last action"""
        if not self.state.undo_stack:
            return
            
        # Save current state to redo stack
        current_state = {
            'subtitles': [sub.model_dump() for sub in self.subtitles],
            'modified_count': self.state.modified_count
        }
        self.state.redo_stack.append(current_state)
        
        # Restore undo state
        undo_state = self.state.undo_stack.pop()
        
        # Restore subtitles
        self.subtitles = []
        for sub_data in undo_state['subtitles']:
            subtitle = SubtitleItem(**sub_data)
            self.subtitles.append(subtitle)
            
        self.state.modified_count = undo_state['modified_count']
        
        self.table_model.update_data(self.subtitles)
        self.update_button_states()
        self.update_status("撤销了上一步操作")
        
    def redo(self):
        """Redo last undone action"""
        if not self.state.redo_stack:
            return
            
        # Save current state to undo stack
        current_state = {
            'subtitles': [sub.model_dump() for sub in self.subtitles],
            'modified_count': self.state.modified_count
        }
        self.state.undo_stack.append(current_state)
        
        # Restore redo state
        redo_state = self.state.redo_stack.pop()
        
        # Restore subtitles
        self.subtitles = []
        for sub_data in redo_state['subtitles']:
            subtitle = SubtitleItem(**sub_data)
            self.subtitles.append(subtitle)
            
        self.state.modified_count = redo_state['modified_count']
        
        self.table_model.update_data(self.subtitles)
        self.update_button_states()
        self.update_status("重做了上一步操作")
        
    def save_subtitles(self):
        """Save subtitles to file"""
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
        """Load subtitles from file"""
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
                    
                self.table_model.update_data(self.subtitles)
                self.update_status(f"从 {file_path} 加载了 {len(self.subtitles)} 条字幕")
                self.subtitles_loaded.emit(self.subtitles)
                
            except Exception as e:
                QMessageBox.critical(self, "加载失败", f"加载字幕时发生错误: {e}")
                
    def export_to_srt(self) -> str:
        """Export subtitles to SRT format"""
        srt_content = ""
        for i, subtitle in enumerate(self.subtitles, 1):
            srt_content += f"{i}\n"
            srt_content += f"{subtitle.start_time} --> {subtitle.end_time}\n"
            srt_content += f"{subtitle.content}\n\n"
        return srt_content
        
    def export_to_json(self) -> str:
        """Export subtitles to JSON format"""
        data = [sub.model_dump() for sub in self.subtitles]
        return json.dumps(data, indent=2, ensure_ascii=False)
        
    def import_from_srt(self, file_path: str) -> List[SubtitleItem]:
        """Import subtitles from SRT file"""
        subtitles = []
        
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read().strip()
            
        # Parse SRT format
        blocks = content.split('\n\n')
        
        for block in blocks:
            lines = block.strip().split('\n')
            if len(lines) >= 3:
                try:
                    index = int(lines[0])
                    time_line = lines[1]
                    text_lines = lines[2:]
                    
                    # Parse time
                    time_parts = time_line.split(' --> ')
                    if len(time_parts) == 2:
                        start_time = time_parts[0].strip()
                        end_time = time_parts[1].strip()
                        
                        # Merge text lines
                        text = '\n'.join(text_lines)
                        
                        subtitle = SubtitleItem(
                            index=index,
                            start_time=start_time,
                            end_time=end_time,
                            content=text
                        )
                        
                        if self.validator.validate_subtitle_item(subtitle):
                            subtitles.append(subtitle)
                            
                except Exception as e:
                    logger.warning(f"解析SRT块时出错: {e}")
                    continue
                    
        return subtitles
        
    def import_from_json(self, file_path: str) -> List[SubtitleItem]:
        """Import subtitles from JSON file"""
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        subtitles = []
        for item in data:
            try:
                subtitle = SubtitleItem(
                    index=item.get("index", 1),
                    start_time=item.get("start_time", "00:00:00,000"),
                    end_time=item.get("end_time", "00:00:00,000"),
                    content=item.get("content", item.get("text", ""))
                )
                
                if self.validator.validate_subtitle_item(subtitle):
                    subtitles.append(subtitle)
                    
            except Exception as e:
                logger.warning(f"解析JSON字幕项时出错: {e}")
                continue
                
        return subtitles
        
    def navigate_subtitle(self, direction: int):
        """Navigate through subtitles"""
        if not self.subtitles:
            return
            
        current_row = self.table_view.currentIndex().row()
        new_row = current_row + direction
        
        if new_row < 0:
            new_row = len(self.subtitles) - 1
        elif new_row >= len(self.subtitles):
            new_row = 0
            
        if new_row != current_row:
            index = self.proxy_model.index(new_row, 0)
            self.table_view.setCurrentIndex(index)
            self.table_view.scrollTo(index)
            
    def update_status(self, message: str):
        """Update status message"""
        self.status_label.setText(message)
        logger.info(f"EnhancedSubtitleEditor: {message}")
        
    def get_modified_subtitles(self) -> List[SubtitleItem]:
        """Get modified subtitles"""
        return self.subtitles.copy()
        
    def has_unsaved_changes(self) -> bool:
        """Check if there are unsaved changes"""
        return self.state.modified_count > 0


# Example usage
if __name__ == "__main__":
    import sys
    from PyQt6.QtWidgets import QApplication
    
    app = QApplication(sys.argv)
    
    # Initialize theme manager
    theme_manager = get_theme_manager()
    
    # Create test subtitles
    test_subtitles = [
        SubtitleItem(
            index=1,
            start_time="00:00:01,000",
            end_time="00:00:03,000",
            content="这是第一条字幕"
        ),
        SubtitleItem(
            index=2,
            start_time="00:00:04,000",
            end_time="00:00:06,000",
            content="这是第二条字幕"
        ),
        SubtitleItem(
            index=3,
            start_time="00:00:07,000",
            end_time="00:00:09,000",
            content="这是第三条字幕"
        )
    ]
    
    # Create and show subtitle editor
    editor = EnhancedSubtitleEditor()
    editor.set_subtitles(test_subtitles)
    editor.show()
    
    sys.exit(app.exec())