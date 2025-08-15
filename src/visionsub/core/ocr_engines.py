"""
PaddleOCR Engine Implementation
"""
import logging
import time
from typing import Any, Dict, List

import numpy as np

from ..core.ocr_engine import BaseOCREngine, OCRResult, OCRTextItem, register_ocr_engine

logger = logging.getLogger(__name__)


@register_ocr_engine("PaddleOCR")
class PaddleOCREngine(BaseOCREngine):
    """PaddleOCR引擎实现"""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.ocr_engine = None
        self._supported_languages = [
            "ch", "en", "ko", "ja", "fr", "de", "es", "ru", "ar", "pt"
        ]

    async def initialize(self) -> bool:
        """初始化PaddleOCR引擎"""
        try:
            from paddleocr import PaddleOCR

            # 映射语言代码
            lang_code = self.map_language(self.config.get("language", "ch"))

            # 创建PaddleOCR实例
            ocr_config = {
                "use_angle_cls": True,
                "lang": lang_code,
                "use_gpu": self.config.get("use_gpu", False),
                "show_log": False,
                "det_model_dir": self.config.get("det_model_dir"),
                "rec_model_dir": self.config.get("rec_model_dir"),
                "cls_model_dir": self.config.get("cls_model_dir"),
            }

            # 移除None值
            ocr_config = {k: v for k, v in ocr_config.items() if v is not None}

            self.ocr_engine = PaddleOCR(**ocr_config)
            self._initialized = True

            logger.info(f"PaddleOCR engine initialized successfully with language: {lang_code}")
            return True

        except ImportError:
            logger.error("PaddleOCR not installed. Please install with: pip install paddleocr")
            return False
        except Exception as e:
            logger.error(f"Failed to initialize PaddleOCR: {e}")
            return False

    async def process_image(self, image: np.ndarray) -> OCRResult:
        """处理图像"""
        if not self.is_initialized():
            raise RuntimeError("PaddleOCR engine not initialized")

        start_time = time.time()

        try:
            # 预处理图像
            processed_image = self.preprocess_image(image)

            # 执行OCR
            result = self.ocr_engine.ocr(processed_image, cls=True)

            # 解析结果
            ocr_items = []
            if result and result[0]:
                for line in result[0]:
                    if len(line) >= 2:
                        box, (text, confidence) = line[0], line[1]

                        # 转换边界框格式
                        x_coords = [point[0] for point in box]
                        y_coords = [point[1] for point in box]
                        bbox = [
                            int(min(x_coords)),
                            int(min(y_coords)),
                            int(max(x_coords) - min(x_coords)),
                            int(max(y_coords) - min(y_coords))
                        ]

                        ocr_item = OCRTextItem(
                            text=self.postprocess_text(text),
                            confidence=float(confidence),
                            bbox=bbox,
                            type=OCRTextItem.TEXT,
                            language=self.config.get("language", "ch")
                        )
                        ocr_items.append(ocr_item)

            # 过滤低置信度结果
            ocr_items = self.filter_by_confidence(ocr_items)

            # 合并相邻项
            ocr_items = self.merge_adjacent_items(ocr_items)

            processing_time = time.time() - start_time

            return OCRResult(
                items=ocr_items,
                processing_time=processing_time,
                engine_info={
                    "engine": "PaddleOCR",
                    "language": self.config.get("language", "ch"),
                    "version": self._get_paddleocr_version()
                },
                image_info={
                    "shape": image.shape,
                    "dtype": str(image.dtype),
                    "preprocessing_applied": self.config.get("enable_preprocessing", True)
                },
                metadata={
                    "total_items": len(ocr_items),
                    "avg_confidence": np.mean([item.confidence for item in ocr_items]) if ocr_items else 0.0
                }
            )

        except Exception as e:
            logger.error(f"PaddleOCR processing failed: {e}")
            raise

    async def process_batch(self, images: List[np.ndarray]) -> List[OCRResult]:
        """批量处理图像"""
        results = []
        for image in images:
            result = await self.process_image(image)
            results.append(result)
        return results

    def get_supported_languages(self) -> List[str]:
        """获取支持的语言列表"""
        return self._supported_languages.copy()

    def _get_paddleocr_version(self) -> str:
        """获取PaddleOCR版本"""
        try:
            import paddleocr
            return getattr(paddleocr, "__version__", "unknown")
        except Exception:
            return "unknown"

    def cleanup(self):
        """清理资源"""
        if self.ocr_engine:
            try:
                del self.ocr_engine
            except Exception:
                pass
        super().cleanup()


@register_ocr_engine("Tesseract")
class TesseractOCREngine(BaseOCREngine):
    """Tesseract OCR引擎实现"""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.ocr_engine = None
        self._supported_languages = [
            "chi_sim", "chi_tra", "eng", "kor", "jpn", "fra", "deu", "spa", "rus", "ara", "por"
        ]

    async def initialize(self) -> bool:
        """初始化Tesseract引擎"""
        try:
            import pytesseract

            # 映射语言代码
            lang_mapping = {
                "中文": "chi_sim+chi_tra",
                "英文": "eng",
                "韩文": "kor",
                "日文": "jpn",
                "法文": "fra",
                "德文": "deu",
                "西班牙文": "spa",
                "俄文": "rus",
                "阿拉伯文": "ara",
                "葡萄牙文": "por"
            }

            lang_code = lang_mapping.get(self.config.get("language", "中文"), "eng")

            # 设置Tesseract配置
            tesseract_config = self.config.get("tesseract_config", "")

            self.ocr_engine = pytesseract
            self.tesseract_lang = lang_code
            self.tesseract_config = tesseract_config

            self._initialized = True

            logger.info(f"Tesseract engine initialized successfully with language: {lang_code}")
            return True

        except ImportError:
            logger.error("Tesseract not installed. Please install with: pip install pytesseract")
            return False
        except Exception as e:
            logger.error(f"Failed to initialize Tesseract: {e}")
            return False

    async def process_image(self, image: np.ndarray) -> OCRResult:
        """处理图像"""
        if not self.is_initialized():
            raise RuntimeError("Tesseract engine not initialized")

        start_time = time.time()

        try:
            # 预处理图像
            processed_image = self.preprocess_image(image)

            # 执行OCR
            import pytesseract

            # 获取详细数据
            data = self.ocr_engine.image_to_data(
                processed_image,
                lang=self.tesseract_lang,
                config=self.tesseract_config,
                output_type=pytesseract.Output.DICT
            )

            # 解析结果
            ocr_items = []
            for i in range(len(data['text'])):
                conf = int(data['conf'][i])
                if conf > 0:  # 过滤空结果
                    text = data['text'][i].strip()
                    if text:  # 过滤空文本
                        x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]

                        ocr_item = OCRTextItem(
                            text=self.postprocess_text(text),
                            confidence=conf / 100.0,  # 转换为0-1范围
                            bbox=[x, y, w, h],
                            type=OCRTextItem.TEXT,
                            language=self.config.get("language", "中文")
                        )
                        ocr_items.append(ocr_item)

            # 过滤低置信度结果
            ocr_items = self.filter_by_confidence(ocr_items)

            # 合并相邻项
            ocr_items = self.merge_adjacent_items(ocr_items)

            processing_time = time.time() - start_time

            return OCRResult(
                items=ocr_items,
                processing_time=processing_time,
                engine_info={
                    "engine": "Tesseract",
                    "language": self.tesseract_lang,
                    "version": self._get_tesseract_version()
                },
                image_info={
                    "shape": image.shape,
                    "dtype": str(image.dtype),
                    "preprocessing_applied": self.config.get("enable_preprocessing", True)
                },
                metadata={
                    "total_items": len(ocr_items),
                    "avg_confidence": np.mean([item.confidence for item in ocr_items]) if ocr_items else 0.0
                }
            )

        except Exception as e:
            logger.error(f"Tesseract processing failed: {e}")
            raise

    async def process_batch(self, images: List[np.ndarray]) -> List[OCRResult]:
        """批量处理图像"""
        results = []
        for image in images:
            result = await self.process_image(image)
            results.append(result)
        return results

    def get_supported_languages(self) -> List[str]:
        """获取支持的语言列表"""
        return self._supported_languages.copy()

    def _get_tesseract_version(self) -> str:
        """获取Tesseract版本"""
        try:
            return self.ocr_engine.get_tesseract_version()
        except Exception:
            return "unknown"

    def cleanup(self):
        """清理资源"""
        self.ocr_engine = None
        super().cleanup()


@register_ocr_engine("EasyOCR")
class EasyOCREngine(BaseOCREngine):
    """EasyOCR引擎实现"""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.ocr_engine = None
        self._supported_languages = [
            "ch_sim", "ch_tra", "en", "ko", "ja", "fr", "de", "es", "ru", "ar", "pt"
        ]

    async def initialize(self) -> bool:
        """初始化EasyOCR引擎"""
        try:
            import easyocr

            # 映射语言代码
            lang_mapping = {
                "中文": ["ch_sim", "ch_tra"],
                "英文": ["en"],
                "韩文": ["ko"],
                "日文": ["ja"],
                "法文": ["fr"],
                "德文": ["de"],
                "西班牙文": ["es"],
                "俄文": ["ru"],
                "阿拉伯文": ["ar"],
                "葡萄牙文": ["pt"]
            }

            lang_codes = lang_mapping.get(self.config.get("language", "中文"), ["en"])

            # 创建EasyOCR实例
            reader_config = {
                "gpu": self.config.get("use_gpu", False),
                "model_storage_directory": self.config.get("model_storage_directory"),
                "download_enabled": self.config.get("download_enabled", True)
            }

            # 移除None值
            reader_config = {k: v for k, v in reader_config.items() if v is not None}

            self.ocr_engine = easyocr.Reader(lang_codes, **reader_config)
            self.easyocr_langs = lang_codes

            self._initialized = True

            logger.info(f"EasyOCR engine initialized successfully with languages: {lang_codes}")
            return True

        except ImportError:
            logger.error("EasyOCR not installed. Please install with: pip install easyocr")
            return False
        except Exception as e:
            logger.error(f"Failed to initialize EasyOCR: {e}")
            return False

    async def process_image(self, image: np.ndarray) -> OCRResult:
        """处理图像"""
        if not self.is_initialized():
            raise RuntimeError("EasyOCR engine not initialized")

        start_time = time.time()

        try:
            # 预处理图像
            processed_image = self.preprocess_image(image)

            # 执行OCR
            results = self.ocr_engine.readtext(processed_image)

            # 解析结果
            ocr_items = []
            for (bbox, text, confidence) in results:
                # 转换边界框格式
                x_coords = [point[0] for point in bbox]
                y_coords = [point[1] for point in bbox]
                x = int(min(x_coords))
                y = int(min(y_coords))
                w = int(max(x_coords) - min(x_coords))
                h = int(max(y_coords) - min(y_coords))

                ocr_item = OCRTextItem(
                    text=self.postprocess_text(text),
                    confidence=float(confidence),
                    bbox=[x, y, w, h],
                    type=OCRTextItem.TEXT,
                    language=self.config.get("language", "中文")
                )
                ocr_items.append(ocr_item)

            # 过滤低置信度结果
            ocr_items = self.filter_by_confidence(ocr_items)

            # 合并相邻项
            ocr_items = self.merge_adjacent_items(ocr_items)

            processing_time = time.time() - start_time

            return OCRResult(
                items=ocr_items,
                processing_time=processing_time,
                engine_info={
                    "engine": "EasyOCR",
                    "languages": self.easyocr_langs,
                    "version": self._get_easyocr_version()
                },
                image_info={
                    "shape": image.shape,
                    "dtype": str(image.dtype),
                    "preprocessing_applied": self.config.get("enable_preprocessing", True)
                },
                metadata={
                    "total_items": len(ocr_items),
                    "avg_confidence": np.mean([item.confidence for item in ocr_items]) if ocr_items else 0.0
                }
            )

        except Exception as e:
            logger.error(f"EasyOCR processing failed: {e}")
            raise

    async def process_batch(self, images: List[np.ndarray]) -> List[OCRResult]:
        """批量处理图像"""
        results = []
        for image in images:
            result = await self.process_image(image)
            results.append(result)
        return results

    def get_supported_languages(self) -> List[str]:
        """获取支持的语言列表"""
        return self._supported_languages.copy()

    def _get_easyocr_version(self) -> str:
        """获取EasyOCR版本"""
        try:
            import easyocr
            return getattr(easyocr, "__version__", "unknown")
        except Exception:
            return "unknown"

    def cleanup(self):
        """清理资源"""
        if self.ocr_engine:
            try:
                del self.ocr_engine
            except Exception:
                pass
        super().cleanup()
