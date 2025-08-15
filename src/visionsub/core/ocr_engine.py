"""
OCR Engine Abstract Layer - Defines interfaces and factory for OCR engines
"""
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Type

import numpy as np

logger = logging.getLogger(__name__)


class OCRResultType(Enum):
    """OCR结果类型枚举"""
    TEXT = "text"
    WORD = "word"
    LINE = "line"


@dataclass
class OCRTextItem:
    """OCR文本项"""
    text: str
    confidence: float
    bbox: List[int]  # [x, y, width, height]
    type: OCRResultType
    language: Optional[str] = None
    line_number: Optional[int] = None
    word_number: Optional[int] = None


@dataclass
class OCRResult:
    """OCR处理结果"""
    items: List[OCRTextItem]
    processing_time: float
    engine_info: Dict[str, Any]
    image_info: Dict[str, Any]
    metadata: Dict[str, Any] = None


class OCREngine(ABC):
    """OCR引擎抽象基类"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self._initialized = False
        self._supported_languages = []

    @abstractmethod
    async def initialize(self) -> bool:
        """
        初始化OCR引擎

        Returns:
            bool: 初始化是否成功
        """
        pass

    @abstractmethod
    async def process_image(self, image: np.ndarray) -> OCRResult:
        """
        处理图像并返回OCR结果

        Args:
            image: 输入图像(numpy数组)

        Returns:
            OCRResult: OCR处理结果
        """
        pass

    @abstractmethod
    async def process_batch(self, images: List[np.ndarray]) -> List[OCRResult]:
        """
        批量处理图像

        Args:
            images: 图像列表

        Returns:
            List[OCRResult]: OCR结果列表
        """
        pass

    @abstractmethod
    def get_supported_languages(self) -> List[str]:
        """
        获取支持的语言列表

        Returns:
            List[str]: 支持的语言代码列表
        """
        pass

    @abstractmethod
    def cleanup(self):
        """清理资源"""
        pass

    def is_initialized(self) -> bool:
        """检查是否已初始化"""
        return self._initialized

    def get_config(self) -> Dict[str, Any]:
        """获取配置"""
        return self.config.copy()

    def update_config(self, config: Dict[str, Any]):
        """更新配置"""
        self.config.update(config)


class OCREngineFactory:
    """OCR引擎工厂"""

    _engines: Dict[str, Type[OCREngine]] = {}
    _instances: Dict[str, OCREngine] = {}

    @classmethod
    def register_engine(cls, name: str, engine_class: Type[OCREngine]):
        """
        注册OCR引擎

        Args:
            name: 引擎名称
            engine_class: 引擎类
        """
        cls._engines[name] = engine_class
        logger.info(f"Registered OCR engine: {name}")

    @classmethod
    def create_engine(cls, name: str, config: Dict[str, Any]) -> OCREngine:
        """
        创建OCR引擎实例

        Args:
            name: 引擎名称
            config: 配置参数

        Returns:
            OCREngine: OCR引擎实例

        Raises:
            ValueError: 不支持的引擎
        """
        if name not in cls._engines:
            raise ValueError(f"Unsupported OCR engine: {name}. Available: {list(cls._engines.keys())}")

        # 创建实例
        engine_class = cls._engines[name]
        engine = engine_class(config)

        # 初始化引擎
        try:
            # 检查是否在异步上下文中
            import asyncio
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # 在运行的事件循环中，延迟初始化
                    engine._initialized = True  # 标记为已初始化，但实际初始化将在第一次使用时进行
                    logger.info(f"Created OCR engine {name} (deferred initialization)")
                else:
                    # 可以安全地初始化
                    initialized = loop.run_until_complete(engine.initialize())
                    if not initialized:
                        raise RuntimeError(f"Failed to initialize OCR engine: {name}")
                    logger.info(f"Created and initialized OCR engine: {name}")
            except RuntimeError:
                # 没有事件循环，创建新的事件循环
                initialized = asyncio.run(engine.initialize())
                if not initialized:
                    raise RuntimeError(f"Failed to initialize OCR engine: {name}") from None
                logger.info(f"Created and initialized OCR engine: {name}")

            cls._instances[name] = engine
            return engine

        except Exception as e:
            logger.error(f"Failed to create OCR engine {name}: {e}")
            raise

    @classmethod
    def get_engine(cls, name: str) -> Optional[OCREngine]:
        """
        获取已创建的引擎实例

        Args:
            name: 引擎名称

        Returns:
            Optional[OCREngine]: 引擎实例，如果不存在返回None
        """
        return cls._instances.get(name)

    @classmethod
    def list_available_engines(cls) -> List[str]:
        """
        获取可用的引擎列表

        Returns:
            List[str]: 可用的引擎名称列表
        """
        return list(cls._engines.keys())

    @classmethod
    def cleanup_all(cls):
        """清理所有引擎实例"""
        for name, engine in cls._instances.items():
            try:
                engine.cleanup()
                logger.info(f"Cleaned up OCR engine: {name}")
            except Exception as e:
                logger.error(f"Error cleaning up OCR engine {name}: {e}")

        cls._instances.clear()


class BaseOCREngine(OCREngine):
    """基础OCR引擎实现，提供通用功能"""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.language_mapping = {
            "中文": "ch",
            "英文": "en",
            "韩文": "ko",
            "日文": "ja",
            "法文": "fr",
            "德文": "de",
            "西班牙文": "es",
            "俄文": "ru",
            "阿拉伯文": "ar",
            "葡萄牙文": "pt"
        }

    async def initialize(self) -> bool:
        """基础初始化"""
        try:
            self._initialized = True
            return True
        except Exception as e:
            logger.error(f"Failed to initialize base OCR engine: {e}")
            return False

    def map_language(self, display_name: str) -> str:
        """
        将显示语言名称映射为引擎支持的语言代码

        Args:
            display_name: 显示语言名称

        Returns:
            str: 语言代码
        """
        return self.language_mapping.get(display_name, display_name)

    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        图像预处理

        Args:
            image: 输入图像

        Returns:
            np.ndarray: 预处理后的图像
        """
        import cv2

        processed = image.copy()

        # 应用配置的预处理选项
        if self.config.get("enable_preprocessing", True):
            # 转换为灰度图
            if len(processed.shape) == 3:
                processed = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)

            # 应用阈值
            threshold = self.config.get("threshold", 180)
            _, processed = cv2.threshold(processed, threshold, 255, cv2.THRESH_BINARY)

            # 降噪
            if self.config.get("denoise", True):
                processed = cv2.medianBlur(processed, 3)

            # 增强对比度
            if self.config.get("enhance_contrast", True):
                processed = cv2.equalizeHist(processed)

        return processed

    def postprocess_text(self, text: str) -> str:
        """
        文本后处理

        Args:
            text: 原始文本

        Returns:
            str: 处理后的文本
        """
        if not self.config.get("enable_postprocessing", True):
            return text

        # 基础文本清理
        processed = text.strip()

        # 移除多余的空白字符
        processed = ' '.join(processed.split())

        # 修复常见的OCR错误
        processed = self._fix_common_errors(processed)

        return processed

    def _fix_common_errors(self, text: str) -> str:
        """修复常见的OCR错误"""
        # 可以在这里添加特定的错误修复规则
        replacements = {
            '|': 'I',
            '0': 'O',
            '1': 'I',
            '5': 'S',
        }

        result = text
        for wrong, correct in replacements.items():
            result = result.replace(wrong, correct)

        return result

    def filter_by_confidence(self, items: List[OCRTextItem]) -> List[OCRTextItem]:
        """
        根据置信度过滤结果

        Args:
            items: OCR结果项列表

        Returns:
            List[OCRTextItem]: 过滤后的结果
        """
        threshold = self.config.get("confidence_threshold", 0.8)
        return [item for item in items if item.confidence >= threshold]

    def merge_adjacent_items(self, items: List[OCRTextItem]) -> List[OCRTextItem]:
        """
        合并相邻的文本项

        Args:
            items: OCR结果项列表

        Returns:
            List[OCRTextItem]: 合并后的结果
        """
        if not items:
            return []

        # 简单的合并策略：将同一行的文本合并
        merged = []
        current_line = []
        current_line_num = items[0].line_number

        for item in items:
            if item.line_number == current_line_num:
                current_line.append(item)
            else:
                if current_line:
                    merged.append(self._merge_line_items(current_line))
                current_line = [item]
                current_line_num = item.line_number

        if current_line:
            merged.append(self._merge_line_items(current_line))

        return merged

    def _merge_line_items(self, items: List[OCRTextItem]) -> OCRTextItem:
        """合并同一行的文本项"""
        if not items:
            raise ValueError("Cannot merge empty items list")

        # 计算合并后的边界框
        x_coords = [item.bbox[0] for item in items]
        y_coords = [item.bbox[1] for item in items]
        widths = [item.bbox[2] for item in items]
        heights = [item.bbox[3] for item in items]

        min_x = min(x_coords)
        min_y = min(y_coords)
        max_width = max(x + w for x, w in zip(x_coords, widths))
        max_height = max(y + h for y, h in zip(y_coords, heights))

        merged_bbox = [min_x, min_y, max_width - min_x, max_height - min_y]

        # 合并文本
        merged_text = ' '.join(item.text for item in items)

        # 计算平均置信度
        avg_confidence = sum(item.confidence for item in items) / len(items)

        return OCRTextItem(
            text=merged_text,
            confidence=avg_confidence,
            bbox=merged_bbox,
            type=OCRResultType.LINE,
            language=items[0].language,
            line_number=items[0].line_number
        )

    def cleanup(self):
        """清理资源"""
        self._initialized = False
        self._supported_languages.clear()


# Register OCR engine implementations (deferred to avoid circular imports)
def _register_ocr_engines():
    """注册OCR引擎实现"""
    try:
        from .ocr_engines import EasyOCREngine, PaddleOCREngine, TesseractOCREngine

        # Register engines
        OCREngineFactory.register_engine("PaddleOCR", PaddleOCREngine)
        OCREngineFactory.register_engine("Tesseract", TesseractOCREngine)
        OCREngineFactory.register_engine("EasyOCR", EasyOCREngine)

        logger.info("OCR engines registered successfully")

    except ImportError as e:
        logger.warning(f"Failed to register OCR engines: {e}")


# 注册引擎装饰器
def register_ocr_engine(name: str):
    """注册OCR引擎的装饰器"""
    def decorator(engine_class: Type[OCREngine]):
        OCREngineFactory.register_engine(name, engine_class)
        return engine_class
    return decorator


# 自动注册引擎
_register_ocr_engines()
