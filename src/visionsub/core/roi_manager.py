"""
ROI Manager - Region of Interest management for video OCR
"""
import json
import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PyQt6.QtCore import QObject, pyqtSignal

logger = logging.getLogger(__name__)


class ROIType(Enum):
    """ROI类型枚举"""
    CUSTOM = "custom"           # 自定义ROI
    SUBTITLE = "subtitle"       # 字幕区域
    TITLE = "title"           # 标题区域
    TEXT_AREA = "text_area"   # 文本区域
    FULL_SCREEN = "full_screen" # 全屏


@dataclass
class ROIInfo:
    """ROI信息"""
    id: str
    name: str
    type: ROIType
    rect: Tuple[int, int, int, int]  # (x, y, width, height)
    enabled: bool = True
    description: str = ""
    confidence_threshold: float = 0.0
    language: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "id": self.id,
            "name": self.name,
            "type": self.type.value,
            "rect": self.rect,
            "enabled": self.enabled,
            "description": self.description,
            "confidence_threshold": self.confidence_threshold,
            "language": self.language
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ROIInfo':
        """从字典创建"""
        return cls(
            id=data["id"],
            name=data["name"],
            type=ROIType(data["type"]),
            rect=tuple(data["rect"]),
            enabled=data.get("enabled", True),
            description=data.get("description", ""),
            confidence_threshold=data.get("confidence_threshold", 0.0),
            language=data.get("language", "")
        )


class ROIManager(QObject):
    """ROI管理器"""

    # 信号定义
    roi_added = pyqtSignal(ROIInfo)          # ROI添加信号
    roi_removed = pyqtSignal(str)             # ROI移除信号
    roi_updated = pyqtSignal(ROIInfo)         # ROI更新信号
    roi_enabled = pyqtSignal(str, bool)       # ROI启用/禁用信号
    active_roi_changed = pyqtSignal(ROIInfo)  # 活动ROI变化信号

    def __init__(self, parent=None):
        super().__init__(parent)
        self.rois: Dict[str, ROIInfo] = {}
        self.active_roi_id: Optional[str] = None
        self.roi_counter = 0

        # 初始化默认ROI
        self._init_default_rois()

    def _init_default_rois(self):
        """初始化默认ROI"""
        # 全屏ROI
        self.add_roi(
            name="全屏",
            roi_type=ROIType.FULL_SCREEN,
            rect=(0, 0, 0, 0),  # 特殊值，表示全屏
            description="识别整个画面"
        )

        # 常见字幕区域预设 (基于1920x1080分辨率的比例)
        self._add_subtitle_presets()

    def _add_subtitle_presets(self):
        """添加常见字幕区域预设"""
        # 基于1920x1080分辨率的预设区域 (比例可以自动适应)
        presets = [
            # 底部字幕区域 (最常见的位置)
            {
                "name": "底部字幕",
                "rect": (0, 900, 1920, 180),  # 底部180像素高度
                "type": ROIType.SUBTITLE,
                "description": "屏幕底部的标准字幕区域",
                "confidence_threshold": 0.7
            },
            # 顶部标题区域
            {
                "name": "顶部标题",
                "rect": (0, 0, 1920, 120),  # 顶部120像素高度
                "type": ROIType.TITLE,
                "description": "屏幕顶部的标题或开场文字区域",
                "confidence_threshold": 0.8
            },
            # 左侧字幕区域
            {
                "name": "左侧字幕",
                "rect": (50, 200, 400, 680),  # 左侧垂直字幕区域
                "type": ROIType.TEXT_AREA,
                "description": "屏幕左侧的垂直字幕区域",
                "confidence_threshold": 0.6
            },
            # 右侧字幕区域
            {
                "name": "右侧字幕",
                "rect": (1470, 200, 400, 680),  # 右侧垂直字幕区域
                "type": ROIType.TEXT_AREA,
                "description": "屏幕右侧的垂直字幕区域",
                "confidence_threshold": 0.6
            },
            # 中央区域 (常用于重要文字)
            {
                "name": "中央区域",
                "rect": (320, 300, 1280, 480),  # 中央区域
                "type": ROIType.TEXT_AREA,
                "description": "屏幕中央的重要文字区域",
                "confidence_threshold": 0.9
            },
            # 底部小字幕 (弹幕风格)
            {
                "name": "底部小字幕",
                "rect": (0, 950, 1920, 130),  # 底部小字区域
                "type": ROIType.SUBTITLE,
                "description": "底部小字号字幕或弹幕区域",
                "confidence_threshold": 0.5
            },
            # 双行字幕区域 (上下排列)
            {
                "name": "双行字幕上",
                "rect": (0, 850, 1920, 80),  # 上行字幕
                "type": ROIType.SUBTITLE,
                "description": "双行字幕的上行区域",
                "confidence_threshold": 0.7
            },
            {
                "name": "双行字幕下",
                "rect": (0, 930, 1920, 80),  # 下行字幕
                "type": ROIType.SUBTITLE,
                "description": "双行字幕的下行区域",
                "confidence_threshold": 0.7
            }
        ]

        for preset in presets:
            self.add_roi(
                name=preset["name"],
                roi_type=preset["type"],
                rect=preset["rect"],
                description=preset["description"],
                confidence_threshold=preset["confidence_threshold"]
            )

    def add_roi(self, name: str, roi_type: ROIType, rect: Tuple[int, int, int, int],
                description: str = "", confidence_threshold: float = 0.0,
                language: str = "") -> str:
        """
        添加ROI
        
        Args:
            name: ROI名称
            roi_type: ROI类型
            rect: ROI矩形区域 (x, y, width, height)
            description: 描述
            confidence_threshold: 置信度阈值
            language: 语言设置
            
        Returns:
            str: ROI ID
        """
        roi_id = f"roi_{self.roi_counter}"
        self.roi_counter += 1

        roi_info = ROIInfo(
            id=roi_id,
            name=name,
            type=roi_type,
            rect=rect,
            description=description,
            confidence_threshold=confidence_threshold,
            language=language
        )

        self.rois[roi_id] = roi_info
        self.roi_added.emit(roi_info)

        # 如果是第一个ROI，设为活动ROI
        if self.active_roi_id is None:
            self.set_active_roi(roi_id)

        logger.info(f"Added ROI: {name} ({roi_id})")
        return roi_id

    def remove_roi(self, roi_id: str) -> bool:
        """
        移除ROI
        
        Args:
            roi_id: ROI ID
            
        Returns:
            bool: 是否成功移除
        """
        if roi_id not in self.rois:
            return False

        del self.rois[roi_id]
        self.roi_removed.emit(roi_id)

        # 如果移除的是活动ROI，选择新的活动ROI
        if self.active_roi_id == roi_id:
            self.active_roi_id = None
            if self.rois:
                # 选择第一个启用的ROI
                for roi in self.rois.values():
                    if roi.enabled:
                        self.set_active_roi(roi.id)
                        break

        logger.info(f"Removed ROI: {roi_id}")
        return True

    def update_roi(self, roi_id: str, **kwargs) -> bool:
        """
        更新ROI
        
        Args:
            roi_id: ROI ID
            **kwargs: 要更新的字段
            
        Returns:
            bool: 是否成功更新
        """
        if roi_id not in self.rois:
            return False

        roi = self.rois[roi_id]

        # 更新字段
        for key, value in kwargs.items():
            if hasattr(roi, key):
                setattr(roi, key, value)

        self.roi_updated.emit(roi)
        logger.info(f"Updated ROI: {roi_id}")
        return True

    def set_roi_enabled(self, roi_id: str, enabled: bool) -> bool:
        """
        设置ROI启用状态
        
        Args:
            roi_id: ROI ID
            enabled: 是否启用
            
        Returns:
            bool: 是否成功设置
        """
        if roi_id not in self.rois:
            return False

        self.rois[roi_id].enabled = enabled
        self.roi_enabled.emit(roi_id, enabled)
        return True

    def set_active_roi(self, roi_id: str) -> bool:
        """
        设置活动ROI
        
        Args:
            roi_id: ROI ID
            
        Returns:
            bool: 是否成功设置
        """
        if roi_id not in self.rois:
            return False

        self.active_roi_id = roi_id
        roi = self.rois[roi_id]
        self.active_roi_changed.emit(roi)

        logger.info(f"Set active ROI: {roi.name} ({roi_id})")
        return True

    def get_active_roi(self) -> Optional[ROIInfo]:
        """获取活动ROI"""
        if self.active_roi_id and self.active_roi_id in self.rois:
            return self.rois[self.active_roi_id]
        return None

    def get_roi(self, roi_id: str) -> Optional[ROIInfo]:
        """获取指定ROI"""
        return self.rois.get(roi_id)

    def get_all_rois(self) -> List[ROIInfo]:
        """获取所有ROI"""
        return list(self.rois.values())

    def get_enabled_rois(self) -> List[ROIInfo]:
        """获取启用的ROI"""
        return [roi for roi in self.rois.values() if roi.enabled]

    def apply_roi_to_frame(self, frame: np.ndarray, roi_id: str = None) -> np.ndarray:
        """
        将ROI应用到帧
        
        Args:
            frame: 输入帧
            roi_id: ROI ID，如果为None则使用活动ROI
            
        Returns:
            np.ndarray: 应用ROI后的帧
        """
        roi_info = self.get_roi(roi_id) if roi_id else self.get_active_roi()

        if not roi_info or not roi_info.enabled:
            return frame

        # 特殊处理全屏ROI
        if roi_info.type == ROIType.FULL_SCREEN or roi_info.rect == (0, 0, 0, 0):
            return frame

        x, y, w, h = roi_info.rect

        # 验证ROI范围
        if x < 0 or y < 0 or w <= 0 or h <= 0:
            logger.warning(f"Invalid ROI dimensions: {roi_info.rect}")
            return frame

        if x + w > frame.shape[1] or y + h > frame.shape[0]:
            logger.warning(f"ROI out of frame bounds: {roi_info.rect}")
            return frame

        # 提取ROI区域
        roi_frame = frame[y:y+h, x:x+w]

        if roi_frame.size == 0:
            logger.warning("Empty ROI frame extracted")
            return frame

        return roi_frame

    def get_roi_config(self, roi_id: str = None) -> Dict[str, Any]:
        """
        获取ROI配置
        
        Args:
            roi_id: ROI ID，如果为None则使用活动ROI
            
        Returns:
            Dict: ROI配置
        """
        roi_info = self.get_roi(roi_id) if roi_id else self.get_active_roi()

        if not roi_info:
            return {}

        config = {
            "roi_rect": roi_info.rect,
            "roi_enabled": roi_info.enabled,
            "roi_type": roi_info.type.value,
            "roi_id": roi_info.id
        }

        if roi_info.confidence_threshold > 0:
            config["confidence_threshold"] = roi_info.confidence_threshold

        if roi_info.language:
            config["language"] = roi_info.language

        return config

    def save_rois(self, file_path: str) -> bool:
        """
        保存ROI配置到文件
        
        Args:
            file_path: 文件路径
            
        Returns:
            bool: 是否成功保存
        """
        try:
            data = {
                "rois": [roi.to_dict() for roi in self.rois.values()],
                "active_roi_id": self.active_roi_id,
                "roi_counter": self.roi_counter
            }

            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

            logger.info(f"Saved ROI configuration to: {file_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to save ROI configuration: {e}")
            return False

    def load_rois(self, file_path: str) -> bool:
        """
        从文件加载ROI配置
        
        Args:
            file_path: 文件路径
            
        Returns:
            bool: 是否成功加载
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # 清除现有ROI
            self.rois.clear()

            # 加载ROI
            for roi_data in data.get("rois", []):
                roi_info = ROIInfo.from_dict(roi_data)
                self.rois[roi_info.id] = roi_info

            # 设置活动ROI
            self.active_roi_id = data.get("active_roi_id")
            self.roi_counter = data.get("roi_counter", len(self.rois))

            logger.info(f"Loaded ROI configuration from: {file_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to load ROI configuration: {e}")
            return False

    def import_roi_presets(self, presets: List[Dict[str, Any]]) -> int:
        """
        导入ROI预设
        
        Args:
            presets: 预设列表
            
        Returns:
            int: 导入的ROI数量
        """
        imported_count = 0

        for preset in presets:
            try:
                self.add_roi(
                    name=preset["name"],
                    roi_type=ROIType(preset.get("type", "custom")),
                    rect=tuple(preset["rect"]),
                    description=preset.get("description", ""),
                    confidence_threshold=preset.get("confidence_threshold", 0.0),
                    language=preset.get("language", "")
                )
                imported_count += 1
            except Exception as e:
                logger.warning(f"Failed to import ROI preset: {e}")

        logger.info(f"Imported {imported_count} ROI presets")
        return imported_count

    def export_roi_presets(self) -> List[Dict[str, Any]]:
        """
        导出ROI预设
        
        Returns:
            List[Dict]: 预设列表
        """
        return [roi.to_dict() for roi in self.rois.values()]

    def clear_all_rois(self):
        """清除所有ROI"""
        self.rois.clear()
        self.active_roi_id = None
        self.roi_counter = 0

        # 重新初始化默认ROI
        self._init_default_rois()

        logger.info("Cleared all ROIs and reinitialized defaults")

    def scale_roi_to_resolution(
        self, 
        roi_rect: Tuple[int, int, int, int], 
        original_resolution: Tuple[int, int], 
        target_resolution: Tuple[int, int]
    ) -> Tuple[int, int, int, int]:
        """
        将ROI缩放到目标分辨率
        
        Args:
            roi_rect: 原始ROI矩形 (x, y, width, height)
            original_resolution: 原始分辨率 (width, height)
            target_resolution: 目标分辨率 (width, height)
            
        Returns:
            缩放后的ROI矩形
        """
        orig_w, orig_h = original_resolution
        target_w, target_h = target_resolution
        
        x, y, w, h = roi_rect
        
        # 计算缩放比例
        scale_x = target_w / orig_w
        scale_y = target_h / orig_h
        
        # 缩放坐标和尺寸
        scaled_x = int(x * scale_x)
        scaled_y = int(y * scale_y)
        scaled_w = int(w * scale_x)
        scaled_h = int(h * scale_y)
        
        return (scaled_x, scaled_y, scaled_w, scaled_h)
    
    def adapt_presets_to_video(self, video_width: int, video_height: int):
        """
        根据视频分辨率调整预设ROI
        
        Args:
            video_width: 视频宽度
            video_height: 视频高度
        """
        # 标准分辨率 (1920x1080)
        standard_res = (1920, 1080)
        target_res = (video_width, video_height)
        
        # 如果已经是标准分辨率，不需要调整
        if target_res == standard_res:
            return
        
        # 调整所有预设ROI
        for roi in self.rois.values():
            if roi.type in [ROIType.SUBTITLE, ROIType.TITLE, ROIType.TEXT_AREA]:
                scaled_rect = self.scale_roi_to_resolution(
                    roi.rect, standard_res, target_res
                )
                roi.rect = scaled_rect
                logger.info(f"Adapted ROI '{roi.name}' to resolution {video_width}x{video_height}")
    
    def get_roi_statistics(self) -> Dict[str, Any]:
        """获取ROI统计信息"""
        total_rois = len(self.rois)
        enabled_rois = len(self.get_enabled_rois())

        type_counts = {}
        for roi in self.rois.values():
            roi_type = roi.type.value
            type_counts[roi_type] = type_counts.get(roi_type, 0) + 1

        return {
            "total_rois": total_rois,
            "enabled_rois": enabled_rois,
            "disabled_rois": total_rois - enabled_rois,
            "active_roi_id": self.active_roi_id,
            "type_distribution": type_counts
        }
