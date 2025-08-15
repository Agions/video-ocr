"""
Enhanced ROI Selection Widget - Advanced ROI selection and management
"""
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from PyQt6.QtCore import QPoint, QRect, QSize, Qt, pyqtSignal
from PyQt6.QtGui import QBrush, QColor, QFont, QMouseEvent, QPainter, QPen
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from visionsub.core.roi_manager import ROIInfo, ROIManager, ROIType

logger = logging.getLogger(__name__)


@dataclass
class ROISelectionState:
    """ROI选择状态"""
    is_selecting: bool = False
    start_point: QPoint = None
    current_point: QPoint = None
    current_rect: QRect = None
    preview_rect: QRect = None


class ROIOverlay(QWidget):
    """ROI叠加层，用于绘制ROI选择"""

    roi_selected = pyqtSignal(QRect)
    roi_changed = pyqtSignal(QRect)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMouseTracking(True)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, False)

        self.state = ROISelectionState()
        self.current_roi = QRect()
        self.roi_rects: List[QRect] = []
        self.active_roi_index = -1

        # 绘制样式
        self.selection_pen = QPen(QColor(255, 0, 0), 2, Qt.PenStyle.DashLine)
        self.selection_brush = QBrush(QColor(255, 0, 0, 30))
        self.active_pen = QPen(QColor(0, 255, 0), 2, Qt.PenStyle.SolidLine)
        self.inactive_pen = QPen(QColor(128, 128, 128), 1, Qt.PenStyle.DashLine)

        # 显示设置
        self.show_coordinates = True
        self.show_dimensions = True
        self.show_grid = False

    def set_current_roi(self, roi: QRect):
        """设置当前ROI"""
        self.current_roi = roi
        self.update()

    def set_roi_rects(self, rects: List[QRect], active_index: int = -1):
        """设置所有ROI矩形"""
        self.roi_rects = rects
        self.active_roi_index = active_index
        self.update()

    def mousePressEvent(self, event: QMouseEvent):
        """鼠标按下事件"""
        if event.button() == Qt.MouseButton.LeftButton:
            self.state.is_selecting = True
            self.state.start_point = event.pos()
            self.state.current_point = event.pos()
            self.state.current_rect = QRect(self.state.start_point, QSize())

    def mouseMoveEvent(self, event: QMouseEvent):
        """鼠标移动事件"""
        if self.state.is_selecting:
            self.state.current_point = event.pos()
            self.state.current_rect = QRect(self.state.start_point, self.state.current_point).normalized()
            self.update()

    def mouseReleaseEvent(self, event: QMouseEvent):
        """鼠标释放事件"""
        if event.button() == Qt.MouseButton.LeftButton and self.state.is_selecting:
            self.state.is_selecting = False

            if self.state.current_rect.isValid() and self.state.current_rect.width() > 5 and self.state.current_rect.height() > 5:
                self.current_roi = self.state.current_rect
                self.roi_selected.emit(self.current_roi)

            self.state.current_rect = None
            self.update()

    def paintEvent(self, event):
        """绘制事件"""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # 绘制网格
        if self.show_grid:
            self._draw_grid(painter)

        # 绘制所有ROI矩形
        for i, rect in enumerate(self.roi_rects):
            if rect.isValid():
                if i == self.active_roi_index:
                    painter.setPen(self.active_pen)
                    painter.setBrush(QBrush(QColor(0, 255, 0, 20)))
                else:
                    painter.setPen(self.inactive_pen)
                    painter.setBrush(QBrush(QColor(128, 128, 128, 10)))

                painter.drawRect(rect)

        # 绘制当前ROI
        if self.current_roi.isValid():
            painter.setPen(self.active_pen)
            painter.setBrush(QBrush(QColor(0, 255, 0, 30)))
            painter.drawRect(self.current_roi)

            # 绘制坐标和尺寸信息
            if self.show_coordinates or self.show_dimensions:
                self._draw_roi_info(painter, self.current_roi)

        # 绘制选择中的矩形
        if self.state.is_selecting and self.state.current_rect:
            painter.setPen(self.selection_pen)
            painter.setBrush(self.selection_brush)
            painter.drawRect(self.state.current_rect)

    def _draw_grid(self, painter: QPainter):
        """绘制网格"""
        painter.setPen(QPen(QColor(200, 200, 200), 1, Qt.PenStyle.DotLine))

        grid_size = 50
        width = self.width()
        height = self.height()

        # 垂直线
        for x in range(0, width, grid_size):
            painter.drawLine(x, 0, x, height)

        # 水平线
        for y in range(0, height, grid_size):
            painter.drawLine(0, y, width, y)

    def _draw_roi_info(self, painter: QPainter, rect: QRect):
        """绘制ROI信息"""
        info_text = ""

        if self.show_coordinates:
            info_text += f"({rect.x()}, {rect.y()})"

        if self.show_dimensions:
            if info_text:
                info_text += " "
            info_text += f"{rect.width()}×{rect.height()}"

        if info_text:
            painter.setPen(QPen(QColor(0, 0, 0)))
            painter.setFont(QFont("Arial", 8))

            # 调整文本位置，确保在ROI内部
            text_rect = painter.fontMetrics().boundingRect(info_text)
            text_x = rect.x() + 5
            text_y = rect.y() + 15

            if text_x + text_rect.width() > rect.x() + rect.width():
                text_x = rect.x() + rect.width() - text_rect.width() - 5

            if text_y > rect.y() + rect.height() - 5:
                text_y = rect.y() + rect.height() - 5

            # 绘制背景
            bg_rect = QRect(text_x - 2, text_y - text_rect.height() - 2,
                           text_rect.width() + 4, text_rect.height() + 4)
            painter.fillRect(bg_rect, QColor(255, 255, 255, 200))

            # 绘制文本
            painter.drawText(text_x, text_y, info_text)


class ROIManagementDialog(QDialog):
    """ROI管理对话框"""

    roi_edited = pyqtSignal()

    def __init__(self, roi_manager: ROIManager, parent=None):
        super().__init__(parent)
        self.roi_manager = roi_manager
        self.setWindowTitle("ROI管理器")
        self.setModal(True)
        self.resize(800, 600)

        self.setup_ui()
        self.load_roi_list()

        # 连接信号
        self.roi_manager.roi_added.connect(self.on_roi_added)
        self.roi_manager.roi_removed.connect(self.on_roi_removed)
        self.roi_manager.roi_updated.connect(self.on_roi_updated)

    def setup_ui(self):
        """设置UI"""
        layout = QVBoxLayout(self)

        # 工具栏
        toolbar_layout = QHBoxLayout()

        self.add_btn = QPushButton("添加ROI")
        self.edit_btn = QPushButton("编辑")
        self.delete_btn = QPushButton("删除")
        self.duplicate_btn = QPushButton("复制")
        self.import_btn = QPushButton("导入预设")
        self.export_btn = QPushButton("导出预设")

        toolbar_layout.addWidget(self.add_btn)
        toolbar_layout.addWidget(self.edit_btn)
        toolbar_layout.addWidget(self.delete_btn)
        toolbar_layout.addWidget(self.duplicate_btn)
        toolbar_layout.addStretch()
        toolbar_layout.addWidget(self.import_btn)
        toolbar_layout.addWidget(self.export_btn)

        layout.addLayout(toolbar_layout)

        # ROI列表
        self.roi_table = QTableWidget()
        self.roi_table.setColumnCount(6)
        self.roi_table.setHorizontalHeaderLabels(["名称", "类型", "区域", "启用", "置信度", "语言"])
        self.roi_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.roi_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.roi_table.horizontalHeader().setStretchLastSection(True)

        layout.addWidget(self.roi_table)

        # 按钮区域
        button_layout = QHBoxLayout()
        self.set_active_btn = QPushButton("设为活动")
        self.close_btn = QPushButton("关闭")

        button_layout.addStretch()
        button_layout.addWidget(self.set_active_btn)
        button_layout.addWidget(self.close_btn)

        layout.addLayout(button_layout)

        # 连接信号
        self.add_btn.clicked.connect(self.add_roi)
        self.edit_btn.clicked.connect(self.edit_roi)
        self.delete_btn.clicked.connect(self.delete_roi)
        self.duplicate_btn.clicked.connect(self.duplicate_roi)
        self.import_btn.clicked.connect(self.import_presets)
        self.export_btn.clicked.connect(self.export_presets)
        self.set_active_btn.clicked.connect(self.set_active_roi)
        self.close_btn.clicked.connect(self.accept)
        self.roi_table.itemSelectionChanged.connect(self.on_selection_changed)

    def load_roi_list(self):
        """加载ROI列表"""
        self.roi_table.setRowCount(0)

        for roi in self.roi_manager.get_all_rois():
            row = self.roi_table.rowCount()
            self.roi_table.insertRow(row)

            # 名称
            name_item = QTableWidgetItem(roi.name)
            name_item.setData(Qt.ItemDataRole.UserRole, roi.id)
            self.roi_table.setItem(row, 0, name_item)

            # 类型
            if hasattr(roi.type, 'value'):
                type_text = roi.type.value
            else:
                type_text = str(roi.type)
            type_item = QTableWidgetItem(type_text)
            self.roi_table.setItem(row, 1, type_item)

            # 区域
            rect_text = f"{roi.rect[0]},{roi.rect[1]},{roi.rect[2]},{roi.rect[3]}"
            rect_item = QTableWidgetItem(rect_text)
            self.roi_table.setItem(row, 2, rect_item)

            # 启用状态
            enabled_item = QTableWidgetItem("是" if roi.enabled else "否")
            self.roi_table.setItem(row, 3, enabled_item)

            # 置信度
            conf_item = QTableWidgetItem(f"{roi.confidence_threshold:.2f}" if roi.confidence_threshold > 0 else "")
            self.roi_table.setItem(row, 4, conf_item)

            # 语言
            lang_item = QTableWidgetItem(roi.language)
            self.roi_table.setItem(row, 5, lang_item)

            # 标记活动ROI
            if roi.id == self.roi_manager.active_roi_id:
                for col in range(6):
                    item = self.roi_table.item(row, col)
                    if item:
                        item.setBackground(QColor(0, 255, 0, 50))

    def on_selection_changed(self):
        """选择变化事件"""
        has_selection = len(self.roi_table.selectedItems()) > 0
        self.edit_btn.setEnabled(has_selection)
        self.delete_btn.setEnabled(has_selection)
        self.duplicate_btn.setEnabled(has_selection)
        self.set_active_btn.setEnabled(has_selection)

    def get_selected_roi_id(self) -> Optional[str]:
        """获取选中的ROI ID"""
        selected_items = self.roi_table.selectedItems()
        if selected_items:
            row = selected_items[0].row()
            name_item = self.roi_table.item(row, 0)
            if name_item:
                return name_item.data(Qt.ItemDataRole.UserRole)
        return None

    def add_roi(self):
        """添加ROI"""
        dialog = ROIEditDialog(self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            roi_data = dialog.get_roi_data()
            self.roi_manager.add_roi(**roi_data)

    def edit_roi(self):
        """编辑ROI"""
        roi_id = self.get_selected_roi_id()
        if not roi_id:
            return

        roi = self.roi_manager.get_roi(roi_id)
        if not roi:
            return

        dialog = ROIEditDialog(self, roi)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            roi_data = dialog.get_roi_data()
            self.roi_manager.update_roi(roi_id, **roi_data)

    def delete_roi(self):
        """删除ROI"""
        roi_id = self.get_selected_roi_id()
        if not roi_id:
            return

        reply = QMessageBox.question(
            self, "确认删除",
            "确定要删除选中的ROI吗？",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )

        if reply == QMessageBox.StandardButton.Yes:
            self.roi_manager.remove_roi(roi_id)

    def duplicate_roi(self):
        """复制ROI"""
        roi_id = self.get_selected_roi_id()
        if not roi_id:
            return

        roi = self.roi_manager.get_roi(roi_id)
        if not roi:
            return

        self.roi_manager.add_roi(
            name=f"{roi.name}_副本",
            roi_type=roi.type,
            rect=roi.rect,
            description=roi.description,
            confidence_threshold=roi.confidence_threshold,
            language=roi.language
        )

    def set_active_roi(self):
        """设置活动ROI"""
        roi_id = self.get_selected_roi_id()
        if roi_id:
            self.roi_manager.set_active_roi(roi_id)

    def import_presets(self):
        """导入预设"""
        # 这里可以实现从文件导入预设的逻辑
        QMessageBox.information(self, "导入预设", "导入预设功能待实现")

    def export_presets(self):
        """导出预设"""
        # 这里可以实现导出预设到文件的逻辑
        QMessageBox.information(self, "导出预设", "导出预设功能待实现")

    def on_roi_added(self, roi: ROIInfo):
        """ROI添加事件"""
        self.load_roi_list()
        self.roi_edited.emit()

    def on_roi_removed(self, roi_id: str):
        """ROI移除事件"""
        self.load_roi_list()
        self.roi_edited.emit()

    def on_roi_updated(self, roi: ROIInfo):
        """ROI更新事件"""
        self.load_roi_list()
        self.roi_edited.emit()


class ROIEditDialog(QDialog):
    """ROI编辑对话框"""

    def __init__(self, parent=None, roi: ROIInfo = None):
        super().__init__(parent)
        self.roi = roi
        self.setWindowTitle("编辑ROI" if roi else "添加ROI")
        self.setModal(True)

        self.setup_ui()

        if roi:
            self.load_roi_data()

    def setup_ui(self):
        """设置UI"""
        layout = QFormLayout(self)

        # 基本信息
        self.name_edit = QLineEdit()
        layout.addRow("名称:", self.name_edit)

        self.type_combo = QComboBox()
        for roi_type in ROIType:
            self.type_combo.addItem(roi_type.value, roi_type)
        layout.addRow("类型:", self.type_combo)

        # ROI区域
        rect_layout = QHBoxLayout()
        self.x_spin = QSpinBox()
        self.y_spin = QSpinBox()
        self.w_spin = QSpinBox()
        self.h_spin = QSpinBox()

        self.x_spin.setRange(0, 9999)
        self.y_spin.setRange(0, 9999)
        self.w_spin.setRange(0, 9999)
        self.h_spin.setRange(0, 9999)

        rect_layout.addWidget(QLabel("X:"))
        rect_layout.addWidget(self.x_spin)
        rect_layout.addWidget(QLabel("Y:"))
        rect_layout.addWidget(self.y_spin)
        rect_layout.addWidget(QLabel("W:"))
        rect_layout.addWidget(self.w_spin)
        rect_layout.addWidget(QLabel("H:"))
        rect_layout.addWidget(self.h_spin)

        layout.addRow("区域:", rect_layout)

        # 高级设置
        self.enabled_check = QCheckBox("启用")
        self.enabled_check.setChecked(True)
        layout.addRow("启用:", self.enabled_check)

        self.confidence_spin = QDoubleSpinBox()
        self.confidence_spin.setRange(0.0, 1.0)
        self.confidence_spin.setSingleStep(0.1)
        self.confidence_spin.setValue(0.0)
        layout.addRow("置信度阈值:", self.confidence_spin)

        self.language_edit = QLineEdit()
        layout.addRow("语言:", self.language_edit)

        self.description_edit = QLineEdit()
        layout.addRow("描述:", self.description_edit)

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

    def load_roi_data(self):
        """加载ROI数据"""
        if self.roi:
            self.name_edit.setText(self.roi.name)
            self.type_combo.setCurrentText(self.roi.type.value)

            x, y, w, h = self.roi.rect
            self.x_spin.setValue(x)
            self.y_spin.setValue(y)
            self.w_spin.setValue(w)
            self.h_spin.setValue(h)

            self.enabled_check.setChecked(self.roi.enabled)
            self.confidence_spin.setValue(self.roi.confidence_threshold)
            self.language_edit.setText(self.roi.language)
            self.description_edit.setText(self.roi.description)

    def get_roi_data(self) -> Dict[str, Any]:
        """获取ROI数据"""
        return {
            "name": self.name_edit.text(),
            "roi_type": self.type_combo.currentData(),
            "rect": (self.x_spin.value(), self.y_spin.value(),
                    self.w_spin.value(), self.h_spin.value()),
            "enabled": self.enabled_check.isChecked(),
            "confidence_threshold": self.confidence_spin.value(),
            "language": self.language_edit.text(),
            "description": self.description_edit.text()
        }


class ROISelectionPanel(QWidget):
    """ROI选择面板"""

    roi_config_changed = pyqtSignal(dict)

    def __init__(self, roi_manager: ROIManager, parent=None):
        super().__init__(parent)
        self.roi_manager = roi_manager
        self.current_video_size = QSize(0, 0)

        self.setup_ui()
        self.connect_signals()
        self.update_roi_list()

    def setup_ui(self):
        """设置UI"""
        layout = QVBoxLayout(self)

        # ROI选择
        roi_group = QGroupBox("ROI选择")
        roi_layout = QVBoxLayout(roi_group)

        # ROI下拉列表
        self.roi_combo = QComboBox()
        self.roi_combo.setMinimumWidth(200)
        roi_layout.addWidget(QLabel("选择ROI:"))
        roi_layout.addWidget(self.roi_combo)

        # ROI信息显示
        self.info_label = QLabel("未选择ROI")
        self.info_label.setWordWrap(True)
        roi_layout.addWidget(self.info_label)

        # 快速操作按钮
        quick_layout = QHBoxLayout()
        self.manage_btn = QPushButton("管理ROI")
        self.quick_add_btn = QPushButton("快速添加")
        self.clear_btn = QPushButton("清除选择")

        quick_layout.addWidget(self.manage_btn)
        quick_layout.addWidget(self.quick_add_btn)
        quick_layout.addWidget(self.clear_btn)

        roi_layout.addLayout(quick_layout)
        layout.addWidget(roi_group)

        # ROI设置
        settings_group = QGroupBox("ROI设置")
        settings_layout = QFormLayout(settings_group)

        # 启用ROI
        self.enable_check = QCheckBox()
        self.enable_check.setChecked(True)
        settings_layout.addRow("启用ROI:", self.enable_check)

        # 自定义置信度阈值
        self.custom_confidence_check = QCheckBox()
        settings_layout.addRow("自定义置信度:", self.custom_confidence_check)

        self.confidence_spin = QDoubleSpinBox()
        self.confidence_spin.setRange(0.0, 1.0)
        self.confidence_spin.setSingleStep(0.1)
        self.confidence_spin.setValue(0.8)
        settings_layout.addRow("置信度阈值:", self.confidence_spin)

        # 语言设置
        self.language_edit = QLineEdit()
        settings_layout.addRow("ROI语言:", self.language_edit)

        layout.addWidget(settings_group)

        # 预设ROI
        preset_group = QGroupBox("预设ROI")
        preset_layout = QVBoxLayout(preset_group)

        self.preset_combo = QComboBox()
        self.preset_combo.addItem("选择预设...")

        # 添加常见预设
        presets = [
            ("底部字幕", "subtitle_bottom"),
            ("顶部字幕", "subtitle_top"),
            ("左侧字幕", "subtitle_left"),
            ("右侧字幕", "subtitle_right"),
            ("中心区域", "center_area"),
            ("全屏", "full_screen")
        ]

        for name, value in presets:
            self.preset_combo.addItem(name, value)

        preset_layout.addWidget(QLabel("快速预设:"))
        preset_layout.addWidget(self.preset_combo)

        layout.addWidget(preset_group)

        layout.addStretch()

    def connect_signals(self):
        """连接信号"""
        self.roi_combo.currentIndexChanged.connect(self.on_roi_selected)
        self.manage_btn.clicked.connect(self.open_roi_manager)
        self.quick_add_btn.clicked.connect(self.quick_add_roi)
        self.clear_btn.clicked.connect(self.clear_roi_selection)
        self.enable_check.toggled.connect(self.on_roi_settings_changed)
        self.custom_confidence_check.toggled.connect(self.on_roi_settings_changed)
        self.confidence_spin.valueChanged.connect(self.on_roi_settings_changed)
        self.language_edit.textChanged.connect(self.on_roi_settings_changed)
        self.preset_combo.currentIndexChanged.connect(self.on_preset_selected)

        # 连接ROI管理器信号
        self.roi_manager.roi_added.connect(self.update_roi_list)
        self.roi_manager.roi_removed.connect(self.update_roi_list)
        self.roi_manager.roi_updated.connect(self.update_roi_list)
        self.roi_manager.active_roi_changed.connect(self.on_active_roi_changed)

    def update_roi_list(self):
        """更新ROI列表"""
        current_id = self.get_current_roi_id()

        self.roi_combo.blockSignals(True)
        self.roi_combo.clear()

        for roi in self.roi_manager.get_all_rois():
            self.roi_combo.addItem(roi.name, roi.id)

        # 恢复选择
        if current_id:
            index = self.roi_combo.findData(current_id)
            if index >= 0:
                self.roi_combo.setCurrentIndex(index)

        self.roi_combo.blockSignals(False)

        self.update_roi_info()

    def get_current_roi_id(self) -> Optional[str]:
        """获取当前选择的ROI ID"""
        index = self.roi_combo.currentIndex()
        if index >= 0:
            return self.roi_combo.itemData(index)
        return None

    def on_roi_selected(self, index: int):
        """ROI选择事件"""
        roi_id = self.get_current_roi_id()
        if roi_id:
            self.roi_manager.set_active_roi(roi_id)
        self.update_roi_info()
        self.emit_roi_config_changed()

    def on_active_roi_changed(self, roi: ROIInfo):
        """活动ROI变化事件"""
        index = self.roi_combo.findData(roi.id)
        if index >= 0:
            self.roi_combo.setCurrentIndex(index)
        self.update_roi_info()

    def update_roi_info(self):
        """更新ROI信息显示"""
        roi_id = self.get_current_roi_id()
        if roi_id:
            roi = self.roi_manager.get_roi(roi_id)
            if roi:
                info_text = f"名称: {roi.name}\n"
                info_text += f"类型: {roi.type.value if hasattr(roi.type, 'value') else roi.type}\n"
                info_text += f"区域: {roi.rect}\n"
                info_text += f"状态: {'启用' if roi.enabled else '禁用'}"
                if roi.description:
                    info_text += f"\n描述: {roi.description}"

                self.info_label.setText(info_text)
                return

        self.info_label.setText("未选择ROI")

    def open_roi_manager(self):
        """打开ROI管理器"""
        dialog = ROIManagementDialog(self.roi_manager, self)
        dialog.exec()

    def quick_add_roi(self):
        """快速添加ROI"""
        # 简单的快速添加对话框
        name, ok = QMessageBox.getText(
            self, "快速添加ROI",
            "请输入ROI名称:",
            QMessageBox.StandardButton.Ok | QMessageBox.StandardButton.Cancel
        )

        if ok and name:
            # 创建一个默认大小的ROI
            if self.current_video_size.width() > 0 and self.current_video_size.height() > 0:
                # 默认区域：底部1/4
                x = 0
                y = self.current_video_size.height() * 3 // 4
                w = self.current_video_size.width()
                h = self.current_video_size.height() // 4
            else:
                x, y, w, h = 0, 0, 640, 180  # 默认大小

            self.roi_manager.add_roi(
                name=name,
                roi_type=ROIType.CUSTOM,
                rect=(x, y, w, h),
                description="快速添加的ROI"
            )

    def clear_roi_selection(self):
        """清除ROI选择"""
        self.roi_combo.setCurrentIndex(-1)
        self.roi_manager.active_roi_id = None
        self.update_roi_info()
        self.emit_roi_config_changed()

    def on_preset_selected(self, index: int):
        """预设选择事件"""
        if index <= 0:  # "选择预设..."
            return

        preset_name = self.preset_combo.currentText()
        preset_value = self.preset_combo.currentData()

        # 根据预设创建ROI
        if self.current_video_size.width() > 0 and self.current_video_size.height() > 0:
            vw = self.current_video_size.width()
            vh = self.current_video_size.height()

            if preset_value == "subtitle_bottom":
                rect = (0, vh * 3 // 4, vw, vh // 4)
            elif preset_value == "subtitle_top":
                rect = (0, 0, vw, vh // 4)
            elif preset_value == "subtitle_left":
                rect = (0, 0, vw // 3, vh)
            elif preset_value == "subtitle_right":
                rect = (vw * 2 // 3, 0, vw // 3, vh)
            elif preset_value == "center_area":
                rect = (vw // 4, vh // 4, vw // 2, vh // 2)
            elif preset_value == "full_screen":
                rect = (0, 0, vw, vh)
            else:
                rect = (0, vh * 3 // 4, vw, vh // 4)  # 默认底部

            self.roi_manager.add_roi(
                name=preset_name,
                roi_type=ROIType.SUBTITLE,
                rect=rect,
                description=f"预设: {preset_name}"
            )

        # 重置选择
        self.preset_combo.setCurrentIndex(0)

    def on_roi_settings_changed(self):
        """ROI设置变化事件"""
        self.emit_roi_config_changed()

    def emit_roi_config_changed(self):
        """发送ROI配置变化信号"""
        config = self.get_roi_config()
        self.roi_config_changed.emit(config)

    def get_roi_config(self) -> Dict[str, Any]:
        """获取ROI配置"""
        roi_id = self.get_current_roi_id()
        config = self.roi_manager.get_roi_config(roi_id)

        # 添加面板特定的设置
        config["roi_enabled"] = self.enable_check.isChecked()

        if self.custom_confidence_check.isChecked():
            config["confidence_threshold"] = self.confidence_spin.value()

        if self.language_edit.text():
            config["language"] = self.language_edit.text()

        return config

    def set_video_size(self, size: QSize):
        """设置视频尺寸"""
        self.current_video_size = size
