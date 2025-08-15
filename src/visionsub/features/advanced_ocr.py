"""
Advanced OCR processor with text enhancement and language detection
"""
import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from ..core.errors import ErrorContext, OCRError
from ..models.config import OCRConfig

logger = logging.getLogger(__name__)


@dataclass
class EnhancedOCRResult:
    """Enhanced OCR result with additional metadata"""
    text: str
    confidence: float
    timestamp: float
    language: str
    enhanced_text: str
    bounding_boxes: Optional[List[Tuple[int, int, int, int]]] = None
    processing_time: float = 0.0
    frame_number: int = 0
    enhancement_applied: List[str] = None

    def __post_init__(self):
        if self.enhancement_applied is None:
            self.enhancement_applied = []


class TextEnhancer:
    """Text enhancement for OCR results"""

    def __init__(self):
        self.common_corrections = {
            # Common OCR errors
            '0': 'O',  # Zero to O
            '1': 'I',  # One to I
            '5': 'S',  # Five to S
            '2': 'Z',  # Two to Z
            '|': 'I',  # Pipe to I
            '[': 'I',  # Bracket to I
            ']': 'I',  # Bracket to I
            '{': 'I',  # Brace to I
            '}': 'I',  # Brace to I
        }

        self.punctuation_fixes = {
            r'\s+\.': '.',  # Space before period
            r'\s+,': ',',  # Space before comma
            r'\s+!': '!',  # Space before exclamation
            r'\s+\?': '?',  # Space before question mark
            r'\.{2,}': '…',  # Multiple periods to ellipsis
            r'\s+': ' ',  # Multiple spaces to single space
        }

    def enhance_text(self, text: str) -> str:
        """
        Apply text quality improvements

        Args:
            text: Raw OCR text

        Returns:
            Enhanced text
        """
        if not text:
            return text

        enhanced = text

        # Apply common character corrections
        for wrong, correct in self.common_corrections.items():
            enhanced = enhanced.replace(wrong, correct)

        # Fix punctuation
        for pattern, replacement in self.punctuation_fixes.items():
            enhanced = re.sub(pattern, replacement, enhanced)

        # Fix spacing issues
        enhanced = self._fix_spacing(enhanced)

        # Apply spell checking if available
        enhanced = self._apply_spell_check(enhanced)

        return enhanced

    def _fix_spacing(self, text: str) -> str:
        """Fix spacing issues in text"""
        # Remove extra spaces between words
        text = ' '.join(text.split())

        # Ensure proper spacing after punctuation
        text = re.sub(r'([.!?])(?=[A-Z])', r'\1 ', text)

        return text

    def _apply_spell_check(self, text: str) -> str:
        """Apply spell checking (placeholder implementation)"""
        # This would integrate with a spell checking library
        # For now, just return the text as-is
        return text


class LanguageDetector:
    """Language detection for OCR text"""

    def __init__(self):
        # Simple language detection patterns
        self.language_patterns = {
            'zh': {
                'patterns': [r'[\u4e00-\u9fff]'],  # Chinese characters
                'threshold': 0.3
            },
            'en': {
                'patterns': [r'[a-zA-Z]'],  # English letters
                'threshold': 0.8
            },
            'ja': {
                'patterns': [r'[\u3040-\u309f\u30a0-\u30ff]'],  # Japanese
                'threshold': 0.3
            },
            'ko': {
                'patterns': [r'[\uac00-\ud7af]'],  # Korean
                'threshold': 0.3
            }
        }

    def detect(self, text: str) -> str:
        """
        Detect language of text

        Args:
            text: Text to analyze

        Returns:
            Detected language code
        """
        if not text:
            return 'en'  # Default to English

        scores = {}

        for lang, config in self.language_patterns.items():
            score = 0
            for pattern in config['patterns']:
                matches = len(re.findall(pattern, text))
                score += matches

            if len(text) > 0:
                score = score / len(text)

            scores[lang] = score

        # Find language with highest score
        best_lang = max(scores, key=scores.get)
        best_score = scores[best_lang]

        # Check if score meets threshold
        threshold = self.language_patterns[best_lang]['threshold']
        if best_score >= threshold:
            return best_lang

        return 'en'  # Default to English


class ImagePreprocessor:
    """Image preprocessing for better OCR results"""

    def __init__(self):
        self.preprocessing_steps = []

    def preprocess(
        self,
        image: np.ndarray,
        config: OCRConfig
    ) -> np.ndarray:
        """
        Apply preprocessing steps to image

        Args:
            image: Input image
            config: OCR configuration

        Returns:
            Preprocessed image
        """
        processed = image.copy()

        # Convert to grayscale if needed
        if len(processed.shape) == 3:
            processed = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)

        # Apply denoising
        if config.denoise:
            processed = cv2.fastNlMeansDenoising(processed)

        # Apply contrast enhancement
        if config.enhance_contrast:
            processed = self._enhance_contrast(processed)

        # Apply thresholding
        if config.threshold > 0:
            processed = self._apply_threshold(processed, config.threshold)

        # Apply sharpening
        if config.sharpen:
            processed = self._sharpen_image(processed)

        return processed

    def _enhance_contrast(self, image: np.ndarray) -> np.ndarray:
        """Enhance image contrast"""
        # CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return clahe.apply(image)

    def _apply_threshold(self, image: np.ndarray, threshold: int) -> np.ndarray:
        """Apply thresholding"""
        _, binary = cv2.threshold(
            image, threshold, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        return binary

    def _sharpen_image(self, image: np.ndarray) -> np.ndarray:
        """Apply image sharpening"""
        kernel = np.array([[-1, -1, -1],
                          [-1,  9, -1],
                          [-1, -1, -1]])
        return cv2.filter2D(image, -1, kernel)


class AdvancedOCRProcessor:
    """Advanced OCR processor with enhancement capabilities"""

    def __init__(self, config: OCRConfig):
        self.config = config
        self.text_enhancer = TextEnhancer()
        self.language_detector = LanguageDetector()
        self.image_preprocessor = ImagePreprocessor()

        # Initialize OCR engine (placeholder)
        self.ocr_engine = self._initialize_ocr_engine()

    def _initialize_ocr_engine(self):
        """Initialize OCR engine based on configuration"""
        try:
            if self.config.engine == "PaddleOCR":
                from paddleocr import PaddleOCR
                # Initialize PaddleOCR with optimized settings
                self.ocr_engine = PaddleOCR(
                    use_angle_cls=True,    # Enable angle classification
                    lang=self.config.get_paddle_lang_code(),  # Language setting
                    use_gpu=False,         # Use CPU by default
                    show_log=False,        # Disable logging
                    det_model_dir=None,    # Use default detection model
                    rec_model_dir=None,    # Use default recognition model
                    cls_model_dir=None,    # Use default classification model
                )
                logger.info("PaddleOCR engine initialized successfully")
            elif self.config.engine == "Tesseract":
                import pytesseract
                self.ocr_engine = pytesseract
                logger.info("Tesseract engine initialized successfully")
            else:
                raise ValueError(f"Unsupported OCR engine: {self.config.engine}")
        except Exception as e:
            logger.error(f"Failed to initialize OCR engine: {e}")
            # Fallback to basic implementation
            self.ocr_engine = None

    async def process_with_enhancement(
        self,
        frame: np.ndarray,
        timestamp: float,
        frame_number: int = 0
    ) -> EnhancedOCRResult:
        """
        Process frame with text enhancement and language detection

        Args:
            frame: Video frame
            timestamp: Frame timestamp
            frame_number: Frame index

        Returns:
            Enhanced OCR result
        """
        import time
        start_time = time.time()

        try:
            # Preprocess image
            processed_image = self.image_preprocessor.preprocess(frame, self.config)

            # Detect language if auto-detect enabled
            detected_lang = self.config.language
            if self.config.auto_detect_language:
                # Perform initial OCR for language detection
                temp_result = self._perform_ocr(processed_image)
                detected_lang = self.language_detector.detect(temp_result.text)

            # Perform OCR with detected language
            ocr_result = self._perform_ocr(processed_image, detected_lang)

            # Enhance text quality
            enhanced_text = self.text_enhancer.enhance_text(ocr_result.text)

            # Create enhanced result
            enhanced_result = EnhancedOCRResult(
                text=ocr_result.text,
                confidence=ocr_result.confidence,
                timestamp=timestamp,
                language=detected_lang,
                enhanced_text=enhanced_text,
                bounding_boxes=ocr_result.bounding_boxes,
                frame_number=frame_number,
                processing_time=time.time() - start_time,
                enhancement_applied=self._get_enhancement_steps()
            )

            return enhanced_result

        except Exception as e:
            error_context = ErrorContext(
                operation="advanced_ocr_processing",
                frame_number=frame_number,
                timestamp=timestamp
            )
            raise OCRError(f"Advanced OCR processing failed: {e}", error_context)

    def _perform_ocr(self, image: np.ndarray, language: str = None) -> Dict[str, Any]:
        """
        Perform OCR on image using configured OCR engine

        Args:
            image: Preprocessed image
            language: Language code

        Returns:
            OCR result dictionary
        """
        if self.ocr_engine is None:
            return {
                'text': '',
                'confidence': 0.0,
                'bounding_boxes': []
            }

        try:
            if self.config.engine == "PaddleOCR":
                return self._perform_paddle_ocr(image, language)
            elif self.config.engine == "Tesseract":
                return self._perform_tesseract_ocr(image, language)
            else:
                raise ValueError(f"Unsupported OCR engine: {self.config.engine}")
        except Exception as e:
            logger.error(f"OCR processing failed: {e}")
            return {
                'text': '',
                'confidence': 0.0,
                'bounding_boxes': []
            }

    def _perform_paddle_ocr(self, image: np.ndarray, language: str = None) -> Dict[str, Any]:
        """Perform OCR using PaddleOCR"""
        try:
            # Convert grayscale to RGB if needed
            if len(image.shape) == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            
            # Perform OCR
            result = self.ocr_engine.ocr(image, cls=True)
            
            if not result or not result[0]:
                return {
                    'text': '',
                    'confidence': 0.0,
                    'bounding_boxes': []
                }
            
            # Extract text and confidence
            texts = []
            confidences = []
            bounding_boxes = []
            
            for line in result[0]:
                box = line[0]  # Bounding box coordinates
                text = line[1][0]  # Text content
                confidence = line[1][1]  # Confidence score
                
                texts.append(text)
                confidences.append(confidence)
                
                # Convert box to (x, y, w, h) format
                x_coords = [point[0] for point in box]
                y_coords = [point[1] for point in box]
                x, y = min(x_coords), min(y_coords)
                w, h = max(x_coords) - x, max(y_coords) - y
                bounding_boxes.append((int(x), int(y), int(w), int(h)))
            
            # Calculate average confidence
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
            
            # Combine all texts
            combined_text = ' '.join(texts)
            
            return {
                'text': combined_text,
                'confidence': avg_confidence,
                'bounding_boxes': bounding_boxes
            }
            
        except Exception as e:
            logger.error(f"PaddleOCR processing failed: {e}")
            return {
                'text': '',
                'confidence': 0.0,
                'bounding_boxes': []
            }

    def _perform_tesseract_ocr(self, image: np.ndarray, language: str = None) -> Dict[str, Any]:
        """Perform OCR using Tesseract"""
        try:
            # Convert grayscale to RGB if needed
            if len(image.shape) == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            
            # Configure Tesseract
            lang_code = language if language else 'eng+chi_sim'  # Default to English and Chinese
            config = f'--oem 3 --psm 6 -l {lang_code}'
            
            # Perform OCR
            data = self.ocr_engine.image_to_data(image, config=config, output_type=pytesseract.Output.DICT)
            
            # Extract text and confidence
            texts = []
            confidences = []
            bounding_boxes = []
            
            for i in range(len(data['text'])):
                confidence = int(data['conf'][i])
                if confidence > 0:  # Skip empty/low confidence results
                    text = data['text'][i].strip()
                    if text:
                        texts.append(text)
                        confidences.append(confidence)
                        
                        # Get bounding box
                        x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
                        bounding_boxes.append((x, y, w, h))
            
            # Calculate average confidence
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
            
            # Combine all texts
            combined_text = ' '.join(texts)
            
            return {
                'text': combined_text,
                'confidence': avg_confidence / 100.0,  # Convert to 0-1 scale
                'bounding_boxes': bounding_boxes
            }
            
        except Exception as e:
            logger.error(f"Tesseract processing failed: {e}")
            return {
                'text': '',
                'confidence': 0.0,
                'bounding_boxes': []
            }

    def _get_enhancement_steps(self) -> List[str]:
        """Get list of applied enhancement steps"""
        steps = []
        if self.config.denoise:
            steps.append('denoise')
        if self.config.enhance_contrast:
            steps.append('contrast_enhancement')
        if self.config.threshold > 0:
            steps.append('thresholding')
        if self.config.sharpen:
            steps.append('sharpening')

        return steps

    def update_config(self, config: OCRConfig):
        """Update OCR configuration"""
        self.config = config
        self.ocr_engine = self._initialize_ocr_engine()

    def get_stats(self) -> Dict[str, Any]:
        """Get processing statistics"""
        return {
            'config': {
                'denoise': self.config.denoise,
                'enhance_contrast': self.config.enhance_contrast,
                'threshold': self.config.threshold,
                'sharpen': self.config.sharpen,
                'auto_detect_language': self.config.auto_detect_language
            }
        }
