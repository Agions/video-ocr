from abc import abstractmethod
from typing import Protocol

import numpy as np


def map_language_code(lang_display_name: str) -> str:
    """
    将中文语言显示名称映射到PaddleOCR语言代码

    Args:
        lang_display_name: 中文语言显示名称

    Returns:
        PaddleOCR支持的语言代码
    """
    language_mapping = {
        "中文": "ch",
        "英文": "en",
        "韩文": "korean",
        "日文": "japan"
    }
    return language_mapping.get(lang_display_name, "ch")  # 默认使用中文

# Lazy load paddleocr to speed up initial application startup
_paddle_ocr_instance = None

class OcrEngine(Protocol):
    """
    A protocol defining the standard interface for any OCR engine.
    This allows for interchangeable OCR implementations.
    """
    @abstractmethod
    def recognize(self, image: np.ndarray) -> str:
        """
        Recognizes text from a given image.

        Args:
            image: The image (as a NumPy array) to process.

        Returns:
            The recognized text, concatenated into a single string.
        """
        ...

class PaddleOcrEngine(OcrEngine):
    """An OCR engine implementation using Baidu's PaddleOCR."""

    def __init__(self, lang: str = 'ch', use_gpu: bool = False):
        """
        Initializes the PaddleOCR engine.
        The model is loaded lazily on the first call to recognize().

        Args:
            lang: The language model to use (e.g., 'ch', 'en', 'korean').
            use_gpu: Whether to use the GPU for processing.
        """
        self.lang = lang
        self.use_gpu = use_gpu
        self.engine = self._get_instance()

    def _get_instance(self):
        """Lazily initializes and returns the PaddleOCR instance."""
        global _paddle_ocr_instance
        if _paddle_ocr_instance is None:
            try:
                from paddleocr import PaddleOCR
                print("Initializing PaddleOCR engine...")
                _paddle_ocr_instance = PaddleOCR(use_angle_cls=True, lang=self.lang, use_gpu=self.use_gpu)
                print("PaddleOCR engine initialized.")
            except ImportError as e:
                raise RuntimeError("PaddleOCR is not installed. Please run 'poetry install'.") from e
        return _paddle_ocr_instance

    def recognize(self, image: np.ndarray) -> str:
        """
        Performs OCR using PaddleOCR.

        Args:
            image: The preprocessed image to recognize.

        Returns:
            The extracted text, with lines joined by spaces.
        """
        if self.engine is None:
            return "Error: PaddleOCR not initialized."

        # PaddleOCR returns a list of lists, each containing bounding box, text, and confidence.
        # e.g., [[[[x1,y1], [x2,y2], [x3,y3], [x4,y4]], ('text', confidence)], ...]
        result = self.engine.ocr(image, cls=True)

        if not result or not result[0]:
            return ""

        # Extract just the text from the results
        lines = [line[1][0] for line in result[0]]
        return " ".join(lines)

# A simple factory to get the desired engine
def get_ocr_engine(name: str, lang: str, use_gpu: bool = False) -> OcrEngine:
    """
    Factory function to get an instance of a specific OCR engine.

    Args:
        name: The name of the engine ("PaddleOCR" or "Tesseract").
        lang: The language code for the engine.
        use_gpu: Whether to use GPU (applies to PaddleOCR).

    Returns:
        An instance of the requested OcrEngine.

    Raises:
        ValueError: If an unknown engine name is provided.
    """
    if name == "PaddleOCR":
        return PaddleOcrEngine(lang=lang, use_gpu=use_gpu)
    # elif name == "Tesseract":
    #     return TesseractEngine(lang=lang) # Tesseract can be added here
    else:
        raise ValueError(f"Unknown OCR engine: {name}")
