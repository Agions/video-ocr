from typing import Optional, Dict, Any
import logging

import numpy as np
from PyQt6.QtCore import QObject, QRect, QTimer, pyqtSignal

from visionsub.core.engine import ProcessingEngine
from visionsub.core.ocr import get_ocr_engine
from visionsub.core.roi_manager import ROIManager
from visionsub.core.video import VideoReader
from visionsub.models.config import OcrConfig, ProcessingConfig
from visionsub.services.batch_service import BatchService

logger = logging.getLogger(__name__)


class SimpleProcessingEngine:
    """简单处理引擎，用于错误情况下的后备"""
    
    def __init__(self):
        self._roi_manager = SimpleROIManager()
    
    async def process_frame(self, frame, timestamp=None, apply_roi=False):
        """简单处理帧"""
        return {
            'timestamp': timestamp or 0,
            'frame_shape': frame.shape,
            'text': [],
            'confidence': 0.0,
            'roi_applied': False,
            'roi_info': None
        }
    
    def get_roi_manager(self):
        """获取ROI管理器"""
        return self._roi_manager


class SimpleROIManager:
    """简单ROI管理器"""
    
    def update_config(self, config):
        """更新配置"""
        pass
    
    def has_active_roi(self):
        """检查是否有活动ROI"""
        return False
    
    def get_active_roi_rect(self):
        """获取活动ROI矩形"""
        return None


class MainViewModel(QObject):
    """
    The central ViewModel for the application. It orchestrates the UI,
    manages application state, and handles user interactions.
    """
    # --- Signals for the View ---
    frame_changed = pyqtSignal(np.ndarray)
    config_changed = pyqtSignal(OcrConfig)
    single_ocr_result_changed = pyqtSignal(str)
    error_occurred = pyqtSignal(str)
    video_loaded = pyqtSignal(int)  # Carries total frame count
    is_playing_changed = pyqtSignal(bool)
    frame_index_changed = pyqtSignal(int)
    roi_config_changed = pyqtSignal(dict)  # ROI配置变化信号
    subtitles_processed = pyqtSignal(list)  # 字幕处理完成信号
    queue_changed = pyqtSignal(list)  # Batch queue changed signal
    batch_progress_changed = pyqtSignal(int)  # Batch processing progress signal
    batch_status_changed = pyqtSignal(str)  # Batch processing status signal

    def __init__(self):
        super().__init__()

        # --- State Management ---
        self.config = OcrConfig()
        self.video_reader: Optional[VideoReader] = None
        self.current_frame: Optional[np.ndarray] = None
        self.current_frame_index = 0
        self.is_playing = False
        self.timer = QTimer()
        self.timer.timeout.connect(self.advance_frame)
        
        # --- Subtitle Management ---
        self.processed_subtitles = []  # 存储已处理的字幕

        # --- Services ---
        # 暂时注释掉OCR引擎初始化，避免测试时的依赖问题
        # self.ocr_engine = get_ocr_engine(self.config.engine, self.config.language)
        self.batch_service = BatchService()
        
        # --- Processing Engine (延迟初始化) ---
        self._processing_engine = None
        self._roi_manager = None

    @property
    def processing_engine(self):
        """延迟初始化处理引擎"""
        if self._processing_engine is None:
            try:
                processing_config = ProcessingConfig(ocr_config=self.config)
                self._processing_engine = ProcessingEngine(processing_config)
            except Exception as e:
                self.error_occurred.emit(f"处理引擎初始化失败: {e}")
                # 创建一个简单的处理引擎作为后备
                self._processing_engine = SimpleProcessingEngine()
        return self._processing_engine
    
    @property
    def roi_manager(self):
        """延迟初始化ROI管理器"""
        if self._roi_manager is None:
            self._roi_manager = self.processing_engine.get_roi_manager()
        return self._roi_manager

    def load_video(self, file_path: str) -> bool:
        """Load a video file."""
        try:
            self.video_reader = VideoReader(file_path)
            self.current_frame_index = 0
            self.current_frame = self.video_reader.get_frame(0)
            self.frame_changed.emit(self.current_frame)
            self.video_loaded.emit(self.video_reader.metadata.frame_count)
            return True
        except Exception as e:
            self.error_occurred.emit(f"Failed to load video: {e}")
            return False

    def play(self):
        """Start video playback."""
        if self.video_reader and not self.is_playing:
            self.is_playing = True
            self.timer.start(33)  # ~30 FPS
            self.is_playing_changed.emit(True)

    def pause(self):
        """Pause video playback."""
        if self.is_playing:
            self.is_playing = False
            self.timer.stop()
            self.is_playing_changed.emit(False)

    def stop(self):
        """Stop video playback and reset to beginning."""
        self.pause()
        if self.video_reader:
            self.current_frame_index = 0
            self.current_frame = self.video_reader.get_frame(0)
            self.frame_changed.emit(self.current_frame)
            self.frame_index_changed.emit(0)

    def set_frame_index(self, frame_index: int):
        """Set the current frame index."""
        if self.video_reader and 0 <= frame_index < self.video_reader.metadata.frame_count:
            self.current_frame_index = frame_index
            self.current_frame = self.video_reader.get_frame(frame_index)
            self.frame_changed.emit(self.current_frame)
            self.frame_index_changed.emit(frame_index)

    def advance_frame(self):
        """Advance to the next frame."""
        if self.video_reader and self.current_frame_index < self.video_reader.metadata.frame_count - 1:
            self.set_frame_index(self.current_frame_index + 1)
        else:
            self.pause()

    def update_config(self, config: OcrConfig):
        """Update the OCR configuration."""
        self.config = config
        self.config_changed.emit(config)
        # Update OCR engine with new config
        self.ocr_engine = get_ocr_engine(self.config.engine)

    async def run_single_frame_ocr(self):
        """Run OCR on the current frame."""
        if self.current_frame is not None:
            try:
                # 使用处理引擎进行OCR，支持ROI
                result = await self.processing_engine.process_frame(
                    self.current_frame, 
                    timestamp=self.current_frame_index / 30.0,  # 估算时间戳
                    apply_roi=True
                )
                
                # 格式化OCR结果
                if result.get('roi_applied', False):
                    roi_info = result.get('roi_info', {})
                    roi_name = roi_info.get('name', 'Unknown')
                    text = f"[ROI: {roi_name}] " + " ".join(result.get('text', []))
                else:
                    text = " ".join(result.get('text', []))
                
                # 添加置信度信息
                confidence = result.get('confidence', 0.0)
                if confidence > 0:
                    text += f" (置信度: {confidence:.2f})"
                
                self.single_ocr_result_changed.emit(text)
            except Exception as e:
                self.error_occurred.emit(f"OCR failed: {e}")
                
    def run_single_frame_ocr_sync(self):
        """同步版本的单帧OCR处理"""
        import asyncio
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # 在事件循环运行时，创建任务
                asyncio.create_task(self.run_single_frame_ocr())
            else:
                # 没有事件循环时直接运行
                asyncio.run(self.run_single_frame_ocr())
        except Exception as e:
            self.error_occurred.emit(f"OCR failed: {e}")

    def update_roi(self, roi: QRect):
        """Update the region of interest."""
        if self.video_reader:
            # 转换QRect为元组格式
            roi_rect = (roi.x(), roi.y(), roi.width(), roi.height())
            
            # 更新配置中的ROI
            self.config.roi_rect = roi_rect
            
            # 更新处理引擎的ROI配置
            roi_config = {
                "roi_enabled": True,
                "roi_rect": roi_rect
            }
            self.processing_engine.set_roi_config(roi_config)
            
            # 发送配置变化信号
            self.config_changed.emit(self.config)

    def update_roi_config(self, roi_config: Dict[str, Any]):
        """更新ROI配置"""
        # 更新处理引擎的ROI配置
        self.processing_engine.set_roi_config(roi_config)
        
        # 如果需要，更新主配置
        if "roi_rect" in roi_config:
            self.config.roi_rect = roi_config["roi_rect"]
            self.config_changed.emit(self.config)
        
        # 发送ROI配置变化信号
        self.roi_config_changed.emit(roi_config)

    def get_roi_manager(self) -> ROIManager:
        """获取ROI管理器"""
        return self.roi_manager

    def get_video_info(self) -> Optional[Dict[str, Any]]:
        """获取视频信息"""
        if self.video_reader:
            metadata = self.video_reader.metadata
            return {
                'width': metadata.width,
                'height': metadata.height,
                'frame_count': metadata.frame_count,
                'fps': metadata.fps,
                'duration': metadata.duration
            }
        return None
    
    async def process_video_subtitles(self, video_path: str) -> bool:
        """处理视频并提取字幕"""
        try:
            # 使用处理引擎处理视频
            subtitles = await self.processing_engine.process_video(video_path)
            
            # 存储处理后的字幕
            self.processed_subtitles = subtitles
            
            # 发送字幕处理完成信号
            self.subtitles_processed.emit(subtitles)
            
            logger.info(f"成功处理视频字幕: {len(subtitles)} 条字幕")
            return True
            
        except Exception as e:
            logger.error(f"处理视频字幕失败: {e}")
            self.error_occurred.emit(f"处理视频字幕失败: {e}")
            return False
    
    def get_processed_subtitles(self) -> list:
        """获取已处理的字幕"""
        return self.processed_subtitles.copy()
    
    def set_processed_subtitles(self, subtitles: list):
        """设置已处理的字幕"""
        self.processed_subtitles = subtitles.copy()
        self.subtitles_processed.emit(subtitles)

    def set_threshold(self, threshold: float):
        """Set the OCR confidence threshold."""
        self.config.confidence_threshold = threshold
        self.config_changed.emit(self.config)

    def set_language(self, language: str):
        """Set the OCR language."""
        self.config.language = language
        self.config_changed.emit(self.config)

    def toggle_playback(self):
        """Toggle video play/pause state."""
        if self.is_playing:
            self.pause()
        else:
            self.play()

    def seek_frame(self, frame_index: int, is_user_action: bool = False):
        """Seek to a specific frame."""
        self.set_frame_index(frame_index)

    def add_to_queue(self, file_paths: list):
        """Add videos to batch processing queue."""
        for file_path in file_paths:
            self.batch_service.add_to_queue(file_path)
        self.update_queue_display()

    def clear_queue(self):
        """Clear the batch processing queue."""
        self.batch_service.clear_queue()
        self.update_queue_display()

    def start_batch_processing(self):
        """Start batch processing of videos in queue."""
        if self.batch_service.get_queue_size() > 0:
            import asyncio
            # Create async task for batch processing
            asyncio.create_task(self._process_batch_queue())

    async def _process_batch_queue(self):
        """Process the batch queue asynchronously."""
        try:
            await self.batch_service.process_queue()
        except Exception as e:
            self.error_occurred.emit(f"Batch processing failed: {e}")

    def update_queue_display(self):
        """Update the queue display (emits signal with current queue)."""
        queue_items = self.batch_service.get_queue_items()
        self.queue_changed.emit(queue_items)
