"""
Enhanced OCR Preview Component with Secure Rendering and Modern Design
"""
import logging
import re
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
from PyQt6.QtCore import (
    Qt, pyqtSignal, QSize, QPoint, QRect, QTimer, 
    QPropertyAnimation, QEasingCurve, QThread, pyqtSlot
)
from PyQt6.QtGui import (
    QFont, QColor, QPalette, QPainter, QBrush, QPen,
    QTextCharFormat, QTextCursor, QImage, QPixmap,
    QSyntaxHighlighter, QTextDocument, QStandardItemModel,
    QStandardItem
)
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QTextEdit, QTableWidget, QTableWidgetItem, QHeaderView,
    QSplitter, QFrame, QScrollArea, QGroupBox,
    QComboBox, QSpinBox, QDoubleSpinBox, QCheckBox,
    QProgressBar, QTabWidget, QListWidget, QListWidgetItem,
    QStyledItemDelegate, QStyleOptionViewItem, QApplication
)

from visionsub.models.config import OcrConfig
from visionsub.ui.theme_system import (
    ThemeManager, get_theme_manager, StyledWidget, 
    Card, Button, ThemeColors
)

logger = logging.getLogger(__name__)


class OCRResultType(Enum):
    """Types of OCR results"""
    RAW = "raw"
    PROCESSED = "processed"
    CONFIDENCE_FILTERED = "confidence_filtered"
    MANUAL_EDIT = "manual_edit"


@dataclass
class OCRResult:
    """OCR result data structure"""
    text: str
    confidence: float
    language: str
    position: QRect
    timestamp: float
    result_type: OCRResultType = OCRResultType.RAW
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class SecureTextHighlighter(QSyntaxHighlighter):
    """Secure text highlighter for OCR results"""
    
    def __init__(self, document):
        super().__init__(document)
        self.theme_manager = get_theme_manager()
        self.setup_formats()
        
    def setup_formats(self):
        """Setup text formats"""
        theme = self.theme_manager.current_theme
        colors = theme.colors
        
        # High confidence format
        self.high_confidence_format = QTextCharFormat()
        self.high_confidence_format.setBackground(QColor(colors.success.name()))
        self.high_confidence_format.setForeground(QColor(colors.text_primary.name()))
        
        # Medium confidence format
        self.medium_confidence_format = QTextCharFormat()
        self.medium_confidence_format.setBackground(QColor(colors.warning.name()))
        self.medium_confidence_format.setForeground(QColor(colors.text_primary.name()))
        
        # Low confidence format
        self.low_confidence_format = QTextCharFormat()
        self.low_confidence_format.setBackground(QColor(colors.error.name()))
        self.low_confidence_format.setForeground(QColor(colors.text_primary.name()))
        
        # Edited text format
        self.edited_format = QTextCharFormat()
        self.edited_format.setBackground(QColor(colors.accent.name()))
        self.edited_format.setForeground(QColor(colors.text_primary.name()))
        self.edited_format.setFontWeight(QFont.Weight.Bold)
        
    def highlightBlock(self, text: str):
        """Highlight text block"""
        # Simple highlighting based on confidence (if available)
        # This is a basic implementation - in practice, you'd parse the text
        # and apply formatting based on the OCR result data
        
        # Remove any existing formatting
        self.setFormat(0, len(text), QTextCharFormat())
        
        # Apply formatting based on text patterns or metadata
        if hasattr(self.document(), 'ocr_results'):
            results = getattr(self.document(), 'ocr_results', [])
            
            for result in results:
                if result.text in text:
                    start_pos = text.find(result.text)
                    if start_pos >= 0:
                        if result.result_type == OCRResultType.MANUAL_EDIT:
                            self.setFormat(start_pos, len(result.text), self.edited_format)
                        elif result.confidence >= 0.8:
                            self.setFormat(start_pos, len(result.text), self.high_confidence_format)
                        elif result.confidence >= 0.6:
                            self.setFormat(start_pos, len(result.text), self.medium_confidence_format)
                        else:
                            self.setFormat(start_pos, len(result.text), self.low_confidence_format)


class SecureOCRRenderer:
    """Secure OCR text renderer"""
    
    def __init__(self):
        self.max_text_length = 10000
        self.max_result_count = 1000
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
        )
        
    def sanitize_text(self, text: str) -> str:
        """Sanitize OCR text for secure display"""
        if not text:
            return ""
            
        # Remove control characters except newlines and tabs
        text = ''.join(char for char in text if ord(char) >= 32 or char in '\n\t')
        
        # Remove potentially dangerous characters
        text = ''.join(char for char in text if char in self.allowed_chars)
        
        # Limit length
        if len(text) > self.max_text_length:
            text = text[:self.max_text_length] + "..."
            
        return text.strip()
        
    def sanitize_results(self, results: List[OCRResult]) -> List[OCRResult]:
        """Sanitize OCR results"""
        if len(results) > self.max_result_count:
            results = results[:self.max_result_count]
            
        sanitized = []
        for result in results:
            sanitized_text = self.sanitize_text(result.text)
            if sanitized_text:  # Only include non-empty results
                sanitized_result = OCRResult(
                    text=sanitized_text,
                    confidence=min(1.0, max(0.0, result.confidence)),
                    language=result.language[:10] if result.language else "",  # Limit language code length
                    position=result.position,
                    timestamp=result.timestamp,
                    result_type=result.result_type,
                    metadata=result.metadata.copy() if result.metadata else {}
                )
                sanitized.append(sanitized_result)
                
        return sanitized


class OCRResultDelegate(QStyledItemDelegate):
    """Delegate for rendering OCR results in table"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.theme_manager = get_theme_manager()
        self.renderer = SecureOCRRenderer()
        
    def paint(self, painter, option, index):
        """Paint OCR result item"""
        if not index.isValid():
            return
            
        # Get data
        result = index.data(Qt.ItemDataRole.UserRole)
        if not isinstance(result, OCRResult):
            return
            
        # Prepare painter
        painter.save()
        
        # Draw background
        theme = self.theme_manager.current_theme
        colors = theme.colors
        
        if option.state & QStyle.StateFlag.State_Selected:
            painter.fillRect(option.rect, QColor(colors.primary.name()))
            text_color = QColor(colors.background.name())
        else:
            painter.fillRect(option.rect, QColor(colors.surface.name()))
            
            # Set background color based on confidence
            if result.confidence >= 0.8:
                painter.fillRect(option.rect, QColor(colors.success.name()).lighter(150))
            elif result.confidence >= 0.6:
                painter.fillRect(option.rect, QColor(colors.warning.name()).lighter(150))
            else:
                painter.fillRect(option.rect, QColor(colors.error.name()).lighter(150))
                
            text_color = QColor(colors.text_primary.name())
            
        painter.setPen(QPen(text_color))
        
        # Draw text
        text_rect = option.rect.adjusted(8, 4, -8, -4)
        
        # Draw confidence indicator
        confidence_text = f"{result.confidence:.1%}"
        confidence_rect = QRect(text_rect.right() - 50, text_rect.top(), 50, text_rect.height())
        painter.drawText(confidence_rect, Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter, confidence_text)
        
        # Draw main text
        text_rect.adjust(0, 0, -60, 0)
        painter.drawText(text_rect, Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, result.text)
        
        painter.restore()
        
    def sizeHint(self, option, index):
        """Get size hint for item"""
        return QSize(option.rect.width(), 30)


class OCRResultTable(QTableWidget):
    """Table widget for displaying OCR results"""
    
    result_selected = pyqtSignal(OCRResult)
    result_edited = pyqtSignal(OCRResult)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.theme_manager = get_theme_manager()
        self.renderer = SecureOCRRenderer()
        self.results: List[OCRResult] = []
        
        self.setup_ui()
        
    def setup_ui(self):
        """Setup table UI"""
        self.setColumnCount(4)
        self.setHorizontalHeaderLabels(["文本", "置信度", "语言", "时间戳"])
        
        # Set up header
        header = self.horizontalHeader()
        header.setStretchLastSection(True)
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
        
        # Set up table properties
        self.setAlternatingRowColors(True)
        self.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        
        # Apply styling
        self.setStyleSheet("""
            QTableWidget {
                background: #f8fafc;
                border: 1px solid #e2e8f0;
                border-radius: 8px;
                gridline-color: #e2e8f0;
            }
            QTableWidget::item {
                padding: 8px;
                border-bottom: 1px solid #e2e8f0;
            }
            QTableWidget::item:selected {
                background: #3b82f6;
                color: white;
            }
            QHeaderView::section {
                background: #f1f5f9;
                padding: 8px;
                border: 1px solid #e2e8f0;
                font-weight: bold;
            }
        """)
        
        # Connect signals
        self.itemSelectionChanged.connect(self.on_selection_changed)
        self.doubleClicked.connect(self.on_double_clicked)
        
    def set_results(self, results: List[OCRResult]):
        """Set OCR results"""
        # Sanitize results
        self.results = self.renderer.sanitize_results(results)
        
        # Update table
        self.setRowCount(len(self.results))
        
        for row, result in enumerate(self.results):
            # Text
            text_item = QTableWidgetItem(result.text)
            text_item.setData(Qt.ItemDataRole.UserRole, result)
            self.setItem(row, 0, text_item)
            
            # Confidence
            confidence_item = QTableWidgetItem(f"{result.confidence:.1%}")
            confidence_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.setItem(row, 1, confidence_item)
            
            # Language
            language_item = QTableWidgetItem(result.language)
            language_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.setItem(row, 2, language_item)
            
            # Timestamp
            timestamp_item = QTableWidgetItem(f"{result.timestamp:.2f}s")
            timestamp_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.setItem(row, 3, timestamp_item)
            
            # Set row color based on confidence
            self.set_row_color(row, result.confidence)
            
    def set_row_color(self, row: int, confidence: float):
        """Set row color based on confidence"""
        for col in range(self.columnCount()):
            item = self.item(row, col)
            if item:
                if confidence >= 0.8:
                    item.setBackground(QColor(144, 238, 144, 100))  # Light green
                elif confidence >= 0.6:
                    item.setBackground(QColor(255, 255, 224, 100))  # Light yellow
                else:
                    item.setBackground(QColor(255, 182, 193, 100))  # Light red
                    
    def on_selection_changed(self):
        """Handle selection change"""
        selected_items = self.selectedItems()
        if selected_items:
            row = selected_items[0].row()
            if 0 <= row < len(self.results):
                self.result_selected.emit(self.results[row])
                
    def on_double_clicked(self, index):
        """Handle double click"""
        row = index.row()
        if 0 <= row < len(self.results):
            self.result_edited.emit(self.results[row])
            
    def get_selected_result(self) -> Optional[OCRResult]:
        """Get selected OCR result"""
        selected_items = self.selectedItems()
        if selected_items:
            row = selected_items[0].row()
            if 0 <= row < len(self.results):
                return self.results[row]
        return None


class OCRTextEditor(QTextEdit):
    """Secure text editor for OCR results"""
    
    text_changed = pyqtSignal(str)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.theme_manager = get_theme_manager()
        self.renderer = SecureOCRRenderer()
        self.highlighter = SecureTextHighlighter(self.document())
        
        self.setup_ui()
        
    def setup_ui(self):
        """Setup editor UI"""
        self.setAcceptRichText(False)
        self.setLineWrapMode(QTextEdit.LineWrapMode.WidgetWidth)
        self.setPlaceholderText("OCR识别结果将在这里显示...")
        
        # Apply styling
        self.setStyleSheet("""
            QTextEdit {
                background: #ffffff;
                border: 1px solid #e2e8f0;
                border-radius: 8px;
                padding: 16px;
                font-family: 'Monaco', 'Menlo', 'Consolas', monospace;
                font-size: 14px;
                line-height: 1.5;
            }
            QTextEdit:focus {
                border: 2px solid #3b82f6;
            }
        """)
        
        # Connect signals
        self.textChanged.connect(self.on_text_changed)
        
    def set_text(self, text: str):
        """Set text with sanitization"""
        sanitized_text = self.renderer.sanitize_text(text)
        self.setPlainText(sanitized_text)
        
    def append_result(self, result: OCRResult):
        """Append OCR result to editor"""
        sanitized_text = self.renderer.sanitize_text(result.text)
        
        # Move cursor to end
        cursor = self.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        
        # Insert text with formatting
        cursor.insertText(sanitized_text + "\n")
        
        # Scroll to bottom
        self.verticalScrollBar().setValue(self.verticalScrollBar().maximum())
        
    def clear_results(self):
        """Clear all results"""
        self.clear()
        
    def on_text_changed(self):
        """Handle text change"""
        text = self.toPlainText()
        sanitized_text = self.renderer.sanitize_text(text)
        self.text_changed.emit(sanitized_text)


class ConfidenceFilterWidget(StyledWidget):
    """Widget for filtering results by confidence"""
    
    filter_changed = pyqtSignal(float, float)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()
        
    def setup_ui(self):
        """Setup confidence filter UI"""
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        layout.addWidget(QLabel("置信度过滤:"))
        
        self.min_slider = QSlider(Qt.Orientation.Horizontal)
        self.min_slider.setRange(0, 100)
        self.min_slider.setValue(0)
        self.min_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        
        self.max_slider = QSlider(Qt.Orientation.Horizontal)
        self.max_slider.setRange(0, 100)
        self.max_slider.setValue(100)
        self.max_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        
        self.min_label = QLabel("0%")
        self.min_label.setMinimumWidth(40)
        self.max_label = QLabel("100%")
        self.max_label.setMinimumWidth(40)
        
        layout.addWidget(self.min_label)
        layout.addWidget(self.min_slider)
        layout.addWidget(QLabel("-"))
        layout.addWidget(self.max_slider)
        layout.addWidget(self.max_label)
        
        # Connect signals
        self.min_slider.valueChanged.connect(self.on_filter_changed)
        self.max_slider.valueChanged.connect(self.on_filter_changed)
        
    def on_filter_changed(self):
        """Handle filter change"""
        min_val = self.min_slider.value() / 100.0
        max_val = self.max_slider.value() / 100.0
        
        # Ensure min <= max
        if min_val > max_val:
            if self.sender() == self.min_slider:
                self.max_slider.setValue(self.min_slider.value())
                max_val = min_val
            else:
                self.min_slider.setValue(self.max_slider.value())
                min_val = max_val
                
        self.min_label.setText(f"{int(min_val * 100)}%")
        self.max_label.setText(f"{int(max_val * 100)}%")
        
        self.filter_changed.emit(min_val, max_val)
        
    def get_range(self) -> Tuple[float, float]:
        """Get confidence range"""
        return (self.min_slider.value() / 100.0, self.max_slider.value() / 100.0)


class EnhancedOCRPreview(StyledWidget):
    """Enhanced OCR preview component with secure rendering and modern design"""
    
    # Signals
    result_selected = pyqtSignal(OCRResult)
    result_edited = pyqtSignal(OCRResult)
    text_exported = pyqtSignal(str)
    filter_applied = pyqtSignal(float, float)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.theme_manager = get_theme_manager()
        self.renderer = SecureOCRRenderer()
        
        self.results: List[OCRResult] = []
        self.filtered_results: List[OCRResult] = []
        
        self.setup_ui()
        self.connect_signals()
        
    def setup_ui(self):
        """Setup enhanced OCR preview UI"""
        layout = QVBoxLayout(self)
        layout.setSpacing(16)
        layout.setContentsMargins(20, 20, 20, 20)
        
        # Header
        header_layout = QHBoxLayout()
        
        self.title_label = QLabel("OCR 识别结果")
        self.title_label.setStyleSheet("font-size: 18px; font-weight: bold;")
        header_layout.addWidget(self.title_label)
        
        header_layout.addStretch()
        
        # Action buttons
        self.clear_button = Button("清空", "secondary")
        self.clear_button.clicked.connect(self.clear_results)
        header_layout.addWidget(self.clear_button)
        
        self.export_button = Button("导出", "primary")
        self.export_button.clicked.connect(self.export_text)
        header_layout.addWidget(self.export_button)
        
        layout.addLayout(header_layout)
        
        # Filter controls
        filter_card = Card("过滤选项")
        filter_layout = QVBoxLayout(filter_card)
        
        self.confidence_filter = ConfidenceFilterWidget()
        self.confidence_filter.filter_changed.connect(self.apply_filter)
        filter_layout.addWidget(self.confidence_filter)
        
        layout.addWidget(filter_card)
        
        # Main content area
        content_splitter = QSplitter(Qt.Orientation.Vertical)
        
        # Text editor
        text_group = QGroupBox("文本预览")
        text_layout = QVBoxLayout(text_group)
        
        self.text_editor = OCRTextEditor()
        self.text_editor.text_changed.connect(self.on_text_changed)
        text_layout.addWidget(self.text_editor)
        
        content_splitter.addWidget(text_group)
        
        # Results table
        table_group = QGroupBox("详细结果")
        table_layout = QVBoxLayout(table_group)
        
        self.results_table = OCRResultTable()
        self.results_table.result_selected.connect(self.on_result_selected)
        self.results_table.result_edited.connect(self.on_result_edited)
        table_layout.addWidget(self.results_table)
        
        content_splitter.addWidget(table_group)
        
        # Set splitter sizes
        content_splitter.setSizes([300, 200])
        
        layout.addWidget(content_splitter)
        
        # Statistics
        stats_card = Card("统计信息")
        stats_layout = QHBoxLayout(stats_card)
        
        self.total_results_label = QLabel("总计: 0")
        self.filtered_results_label = QLabel("已过滤: 0")
        self.avg_confidence_label = QLabel("平均置信度: 0%")
        
        stats_layout.addWidget(self.total_results_label)
        stats_layout.addWidget(self.filtered_results_label)
        stats_layout.addWidget(self.avg_confidence_label)
        stats_layout.addStretch()
        
        layout.addWidget(stats_card)
        
    def connect_signals(self):
        """Connect signals"""
        self.theme_manager.theme_changed.connect(self.on_theme_changed)
        
    def add_result(self, result: OCRResult):
        """Add OCR result"""
        sanitized_result = self.renderer.sanitize_result(result)
        self.results.append(sanitized_result)
        self.text_editor.append_result(sanitized_result)
        self.apply_filter()
        self.update_statistics()
        
    def add_results(self, results: List[OCRResult]):
        """Add multiple OCR results"""
        sanitized_results = self.renderer.sanitize_results(results)
        self.results.extend(sanitized_results)
        
        # Add to text editor
        for result in sanitized_results:
            self.text_editor.append_result(result)
            
        self.apply_filter()
        self.update_statistics()
        
    def set_results(self, results: List[OCRResult]):
        """Set OCR results"""
        self.results = self.renderer.sanitize_results(results)
        self.text_editor.clear_results()
        
        # Add to text editor
        for result in self.results:
            self.text_editor.append_result(result)
            
        self.apply_filter()
        self.update_statistics()
        
    def clear_results(self):
        """Clear all results"""
        self.results.clear()
        self.filtered_results.clear()
        self.text_editor.clear_results()
        self.results_table.set_results([])
        self.update_statistics()
        
    def apply_filter(self):
        """Apply confidence filter"""
        min_confidence, max_confidence = self.confidence_filter.get_range()
        
        self.filtered_results = [
            result for result in self.results
            if min_confidence <= result.confidence <= max_confidence
        ]
        
        self.results_table.set_results(self.filtered_results)
        self.filter_applied.emit(min_confidence, max_confidence)
        self.update_statistics()
        
    def update_statistics(self):
        """Update statistics display"""
        total_count = len(self.results)
        filtered_count = len(self.filtered_results)
        
        # Calculate average confidence
        if self.filtered_results:
            avg_confidence = sum(r.confidence for r in self.filtered_results) / len(self.filtered_results)
        else:
            avg_confidence = 0.0
            
        self.total_results_label.setText(f"总计: {total_count}")
        self.filtered_results_label.setText(f"已过滤: {filtered_count}")
        self.avg_confidence_label.setText(f"平均置信度: {avg_confidence:.1%}")
        
    def on_text_changed(self, text: str):
        """Handle text change"""
        # Update text editor's OCR results for highlighting
        if hasattr(self.text_editor.document(), 'ocr_results'):
            setattr(self.text_editor.document(), 'ocr_results', self.filtered_results)
            self.text_editor.highlighter.rehighlight()
            
    def on_result_selected(self, result: OCRResult):
        """Handle result selection"""
        self.result_selected.emit(result)
        
    def on_result_edited(self, result: OCRResult):
        """Handle result editing"""
        self.result_edited.emit(result)
        
    def export_text(self):
        """Export text to file"""
        if not self.filtered_results:
            return
            
        text = "\n".join(result.text for result in self.filtered_results)
        self.text_exported.emit(text)
        
    def on_theme_changed(self, theme):
        """Handle theme change"""
        # Update highlighter
        self.text_editor.highlighter.setup_formats()
        self.text_editor.highlighter.rehighlight()
        
        # Update table styling
        self.results_table.setStyleSheet("""
            QTableWidget {
                background: #f8fafc;
                border: 1px solid #e2e8f0;
                border-radius: 8px;
                gridline-color: #e2e8f0;
            }
            QTableWidget::item {
                padding: 8px;
                border-bottom: 1px solid #e2e8f0;
            }
            QTableWidget::item:selected {
                background: #3b82f6;
                color: white;
            }
            QHeaderView::section {
                background: #f1f5f9;
                padding: 8px;
                border: 1px solid #e2e8f0;
                font-weight: bold;
            }
        """)


# Example usage
if __name__ == "__main__":
    import sys
    import time
    from PyQt6.QtWidgets import QApplication
    
    app = QApplication(sys.argv)
    
    # Initialize theme manager
    theme_manager = get_theme_manager()
    
    # Create test results
    test_results = [
        OCRResult(
            text="这是一段测试文本",
            confidence=0.95,
            language="zh",
            position=QRect(10, 10, 100, 30),
            timestamp=1.0
        ),
        OCRResult(
            text="This is a test text",
            confidence=0.87,
            language="en",
            position=QRect(10, 50, 120, 30),
            timestamp=2.0
        ),
        OCRResult(
            text="低置信度文本",
            confidence=0.45,
            language="zh",
            position=QRect(10, 90, 80, 30),
            timestamp=3.0
        )
    ]
    
    # Create and show OCR preview
    preview = EnhancedOCRPreview()
    preview.set_results(test_results)
    preview.show()
    
    sys.exit(app.exec())