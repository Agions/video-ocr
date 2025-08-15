"""
Unified core engine for VisionSub - coordinates all processing components
"""
import logging
from typing import Any, Dict, List, Optional

import numpy as np

from ..models.config import ProcessingConfig
from ..models.subtitle import SubtitleItem
from ..security import validate_file_operation
from .ocr_engine import OCREngineFactory
from .roi_manager import ROIManager
from .subtitle_processor import ProcessingContext, SubtitleProcessor
from .video_processor import UnifiedVideoProcessor

logger = logging.getLogger(__name__)


class ProcessingEngine:
    """
    Main processing engine that orchestrates video processing, OCR, and subtitle generation
    """

    def __init__(self, config: ProcessingConfig):
        self.config = config
        self.video_processor = UnifiedVideoProcessor(config)

        # 创建OCR引擎
        ocr_config = config.ocr_config.model_dump()
        self.ocr_service = OCREngineFactory.create_engine(
            config.ocr_config.engine,
            ocr_config
        )

        # 创建ROI管理器
        self.roi_manager = ROIManager()
        self.subtitle_processor = SubtitleProcessor(config)
        self._is_processing = False
        
        # 初始化ROI设置
        self._initialize_roi_from_config()

    def _initialize_roi_from_config(self):
        """从配置初始化ROI设置"""
        roi_rect = self.config.ocr_config.roi_rect
        
        # 如果配置了有效的ROI区域，创建一个自定义ROI
        if roi_rect and roi_rect != (0, 0, 0, 0):
            self.roi_manager.add_roi(
                name="配置ROI",
                roi_type="custom",
                rect=roi_rect,
                description="从配置文件加载的ROI区域"
            )

    def set_roi_config(self, roi_config: Dict[str, Any]):
        """
        设置ROI配置
        
        Args:
            roi_config: ROI配置字典
        """
        if roi_config.get("roi_enabled", False):
            roi_rect = roi_config.get("roi_rect", (0, 0, 0, 0))
            
            # 更新或创建ROI
            if roi_rect != (0, 0, 0, 0):
                active_roi = self.roi_manager.get_active_roi()
                if active_roi and active_roi.type.value == "custom":
                    # 更新现有ROI
                    self.roi_manager.update_roi(
                        active_roi.id,
                        rect=roi_rect
                    )
                else:
                    # 创建新ROI
                    roi_id = self.roi_manager.add_roi(
                        name="自定义ROI",
                        roi_type="custom",
                        rect=roi_rect,
                        description="用户自定义的ROI区域"
                    )
                    self.roi_manager.set_active_roi(roi_id)
        
        # 更新OCR引擎配置
        if "confidence_threshold" in roi_config:
            ocr_config = self.ocr_service.get_config()
            ocr_config["confidence_threshold"] = roi_config["confidence_threshold"]
            self.ocr_service.update_config(ocr_config)
        
        if "language" in roi_config and roi_config["language"]:
            ocr_config = self.ocr_service.get_config()
            ocr_config["language"] = roi_config["language"]
            self.ocr_service.update_config(ocr_config)

    def get_roi_manager(self) -> ROIManager:
        """获取ROI管理器"""
        return self.roi_manager

    async def process_video(self, video_path: str) -> List[SubtitleItem]:
        """
        Process a video file and extract subtitles

        Args:
            video_path: Path to the video file

        Returns:
            List of extracted subtitle items

        Raises:
            VideoProcessingError: If video processing fails
            OCRError: If OCR processing fails
            SecurityError: If file validation fails
        """
        if self._is_processing:
            raise RuntimeError("Already processing another video")

        # Security validation
        if not validate_file_operation(video_path, 'video'):
            from ..core.errors import SecurityError
            raise SecurityError(f"Security validation failed for video file: {video_path}")

        self._is_processing = True
        try:
            logger.info(f"Starting video processing: {video_path}")

            # 获取视频信息
            video_info = self.video_processor.get_video_info(video_path)

            # 创建处理上下文
            context = ProcessingContext(
                video_duration=video_info.duration,
                fps=video_info.fps,
                frame_count=video_info.frame_count,
                video_width=video_info.width,
                video_height=video_info.height
            )

            # 提取视频帧并处理
            ocr_results = []
            frame_interval = self.config.frame_interval

            async for frame_data in self.video_processor.extract_frames(video_path, frame_interval):
                frame, timestamp = frame_data

                # 应用ROI到帧
                processed_frame = self.roi_manager.apply_roi_to_frame(frame)
                
                # 执行OCR
                ocr_result = await self.ocr_service.process_image(processed_frame)
                ocr_results.append((ocr_result, timestamp))

            # 处理OCR结果生成字幕
            subtitles = await self.subtitle_processor.process_ocr_results(ocr_results, context)

            logger.info(f"Successfully processed {len(subtitles)} subtitle items")
            return subtitles

        except Exception as e:
            logger.error(f"Video processing failed: {e}")
            raise
        finally:
            self._is_processing = False

    async def process_frame(
        self,
        frame: np.ndarray,
        timestamp: float,
        apply_roi: bool = True
    ) -> Dict[str, Any]:
        """
        Process a single frame for OCR

        Args:
            frame: Video frame as numpy array
            timestamp: Frame timestamp in seconds
            apply_roi: Whether to apply ROI processing

        Returns:
            Dictionary containing OCR results and metadata
        """
        try:
            # 应用ROI到帧
            processed_frame = frame
            if apply_roi:
                processed_frame = self.roi_manager.apply_roi_to_frame(frame)
            
            ocr_result = await self.ocr_service.process_image(processed_frame)
            
            # 获取ROI信息
            active_roi = self.roi_manager.get_active_roi()
            roi_info = None
            if active_roi:
                roi_info = {
                    'id': active_roi.id,
                    'name': active_roi.name,
                    'type': active_roi.type.value,
                    'rect': active_roi.rect,
                    'enabled': active_roi.enabled
                }
            
            return {
                'text': [item.text for item in ocr_result.items],
                'confidence': ocr_result.metadata.get('avg_confidence', 0.0),
                'timestamp': timestamp,
                'processing_time': ocr_result.processing_time,
                'items_count': len(ocr_result.items),
                'engine_info': ocr_result.engine_info,
                'roi_info': roi_info,
                'roi_applied': apply_roi and active_roi is not None
            }
        except Exception as e:
            logger.error(f"Frame processing failed: {e}")
            raise

    def update_config(self, config: ProcessingConfig):
        """Update processing configuration"""
        self.config = config
        self.video_processor.update_config(config)

        # 重新创建OCR引擎
        ocr_config = config.ocr_config.model_dump()
        self.ocr_service = OCREngineFactory.create_engine(
            config.ocr_config.engine,
            ocr_config
        )

        # 更新字幕处理器配置
        self.subtitle_processor = SubtitleProcessor(config)

    def get_processing_stats(self) -> Dict[str, Any]:
        """Get current processing statistics"""
        return {
            'is_processing': self._is_processing,
            'video_processor_stats': self.video_processor.get_stats(),
            'ocr_engine': self.config.ocr_config.engine,
            'available_engines': OCREngineFactory.list_available_engines()
        }
