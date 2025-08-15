"""
Enhanced OCR Engine System with Security and Performance Improvements
"""
import asyncio
import logging
import time
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, AsyncIterator
from typing import Tuple

import numpy as np
from pydantic import BaseModel, Field, validator

from ..core.errors import OCRError, SecurityError, ConfigurationError
from ..core.ocr_engine import OCREngine, OCRResult, OCRTextItem, OCRResultType
from ..core.ocr_engine import OCREngineFactory, BaseOCREngine
from ..core.memory_manager import MemoryManager
from ..security.validator import validate_file_operation
from ..utils.metrics import MetricsCollector
from ..utils.health_monitor import HealthCheck

logger = logging.getLogger(__name__)


class OCRSecurityLevel(Enum):
    """OCR security levels"""
    BASIC = "basic"          # Basic validation
    ENHANCED = "enhanced"    # Enhanced validation and sanitization
    STRICT = "strict"        # Strict validation with additional checks
    PARANOID = "paranoid"    # Maximum security with extensive validation


class OCRPerformanceMode(Enum):
    """OCR performance modes"""
    ACCURACY = "accuracy"    # Prioritize accuracy over speed
    BALANCED = "balanced"    # Balance between accuracy and speed
    SPEED = "speed"          # Prioritize speed over accuracy
    ADAPTIVE = "adaptive"    # Adaptive mode based on system resources


@dataclass
class OCRSecurityConfig:
    """OCR security configuration"""
    level: OCRSecurityLevel = OCRSecurityLevel.ENHANCED
    max_input_size: int = 10 * 1024 * 1024  # 10MB
    allowed_formats: List[str] = field(default_factory=lambda: [
        '.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'
    ])
    sanitize_output: bool = True
    validate_confidence: bool = True
    min_confidence_threshold: float = 0.5
    max_text_length: int = 10000
    enable_content_filter: bool = True
    blocked_patterns: List[str] = field(default_factory=lambda: [
        r'<script', r'javascript:', r'vbscript:', r'onload=', r'onerror='
    ])


@dataclass
class OCRPerformanceConfig:
    """OCR performance configuration"""
    mode: OCRPerformanceMode = OCRPerformanceMode.BALANCED
    max_workers: int = 4
    batch_size: int = 10
    enable_caching: bool = True
    cache_ttl: int = 3600  # 1 hour
    enable_gpu: bool = True
    gpu_memory_limit_mb: int = 2048
    enable_preprocessing: bool = True
    enable_postprocessing: bool = True
    adaptive_quality: bool = True


@dataclass
class EnhancedOCRResult:
    """Enhanced OCR result with additional metadata"""
    original_result: OCRResult
    security_validated: bool = False
    performance_metrics: Dict[str, Any] = field(default_factory=dict)
    quality_score: float = 0.0
    processing_time: float = 0.0
    memory_usage: float = 0.0
    security_info: Dict[str, Any] = field(default_factory=dict)
    cached: bool = False


class EnhancedOCREngine(OCREngine):
    """Enhanced OCR engine with security and performance improvements"""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.security_config = OCRSecurityConfig()
        self.performance_config = OCRPerformanceConfig()
        
        # Initialize components
        self._initialize_enhanced_components()
        
        # Performance monitoring
        self.metrics_collector = MetricsCollector()
        self.memory_manager = MemoryManager()
        
        # Thread pool for concurrent processing
        self._thread_pool = ThreadPoolExecutor(
            max_workers=self.performance_config.max_workers,
            thread_name_prefix="OCR-Engine"
        )
        
        # Health monitoring
        self._health_status = True
        self._last_health_check = 0

    def _initialize_enhanced_components(self):
        """Initialize enhanced components"""
        try:
            # Initialize security components
            self._initialize_security_components()
            
            # Initialize performance components
            self._initialize_performance_components()
            
            logger.info("Enhanced OCR components initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize enhanced components: {e}")
            raise ConfigurationError(f"Component initialization failed: {e}")

    def _initialize_security_components(self):
        """Initialize security components"""
        try:
            # Content filter for sensitive content
            if self.security_config.enable_content_filter:
                self.content_filter = ContentFilter(self.security_config)
            else:
                self.content_filter = None
            
            # Output sanitizer
            if self.security_config.sanitize_output:
                self.output_sanitizer = OutputSanitizer(self.security_config)
            else:
                self.output_sanitizer = None
            
            logger.debug("Security components initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize security components: {e}")
            raise

    def _initialize_performance_components(self):
        """Initialize performance components"""
        try:
            # Initialize cache if enabled
            if self.performance_config.enable_caching:
                from ..core.cache_manager import SmartCache
                self.cache = SmartCache(
                    max_size=1000,
                    ttl=self.performance_config.cache_ttl
                )
            else:
                self.cache = None
            
            # Initialize GPU if enabled
            if self.performance_config.enable_gpu:
                self._initialize_gpu_acceleration()
            
            logger.debug("Performance components initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize performance components: {e}")
            raise

    def _initialize_gpu_acceleration(self):
        """Initialize GPU acceleration"""
        try:
            # Check if GPU is available
            import torch
            if torch.cuda.is_available():
                self.gpu_available = True
                self.gpu_device = torch.cuda.current_device()
                logger.info(f"GPU acceleration enabled: {torch.cuda.get_device_name()}")
            else:
                self.gpu_available = False
                logger.info("GPU not available, using CPU")
                
        except ImportError:
            self.gpu_available = False
            logger.info("PyTorch not available, using CPU")
        except Exception as e:
            self.gpu_available = False
            logger.warning(f"GPU initialization failed: {e}")

    async def initialize(self) -> bool:
        """Initialize enhanced OCR engine"""
        try:
            # Initialize base engine
            base_initialized = await super().initialize()
            if not base_initialized:
                return False
            
            # Validate configuration
            if not self._validate_configuration():
                return False
            
            # Run health check
            if not await self._perform_health_check():
                return False
            
            self._initialized = True
            logger.info("Enhanced OCR engine initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize enhanced OCR engine: {e}")
            return False

    def _validate_configuration(self) -> bool:
        """Validate OCR engine configuration"""
        try:
            # Validate security configuration
            if self.security_config.min_confidence_threshold < 0 or \
               self.security_config.min_confidence_threshold > 1:
                logger.error("Invalid confidence threshold")
                return False
            
            # Validate performance configuration
            if self.performance_config.max_workers <= 0:
                logger.error("Invalid max workers")
                return False
            
            if self.performance_config.batch_size <= 0:
                logger.error("Invalid batch size")
                return False
            
            logger.debug("Configuration validation passed")
            return True
            
        except Exception as e:
            logger.error(f"Configuration validation failed: {e}")
            return False

    async def _perform_health_check(self) -> bool:
        """Perform health check on OCR engine"""
        try:
            # Simple health check with test image
            test_image = np.zeros((100, 100, 3), dtype=np.uint8)
            test_image.fill(128)  # Gray image
            
            result = await self.process_image_enhanced(test_image)
            
            self._health_status = result is not None
            self._last_health_check = time.time()
            
            return self._health_status
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            self._health_status = False
            return False

    async def process_image_enhanced(self, image: np.ndarray) -> Optional[EnhancedOCRResult]:
        """
        Process image with enhanced security and performance
        
        Args:
            image: Input image as numpy array
            
        Returns:
            EnhancedOCRResult: Enhanced OCR result or None if failed
        """
        try:
            start_time = time.time()
            
            # Security validation
            if not await self._validate_image_input(image):
                raise SecurityError("Image validation failed")
            
            # Check cache
            cache_key = self._generate_cache_key(image)
            if self.cache and self.performance_config.enable_caching:
                cached_result = self.cache.get(cache_key)
                if cached_result:
                    cached_result.cached = True
                    return cached_result
            
            # Preprocess image
            processed_image = await self._preprocess_image(image)
            
            # Process OCR
            ocr_result = await self._process_ocr_internal(processed_image)
            
            # Security validation of results
            if not await self._validate_ocr_result(ocr_result):
                raise SecurityError("OCR result validation failed")
            
            # Post-process results
            enhanced_result = await self._postprocess_ocr_result(ocr_result)
            
            # Cache result
            if self.cache and self.performance_config.enable_caching:
                self.cache.set(cache_key, enhanced_result)
            
            # Update metrics
            processing_time = time.time() - start_time
            self.metrics_collector.record_ocr_processing_time(processing_time)
            
            enhanced_result.processing_time = processing_time
            enhanced_result.memory_usage = image.nbytes
            
            return enhanced_result
            
        except SecurityError as e:
            logger.error(f"Security error in OCR processing: {e}")
            raise
        except Exception as e:
            logger.error(f"Enhanced OCR processing failed: {e}")
            raise OCRError(f"OCR processing failed: {e}")

    async def _validate_image_input(self, image: np.ndarray) -> bool:
        """Validate image input for security"""
        try:
            # Check image dimensions
            if image.size > self.security_config.max_input_size:
                logger.error(f"Image too large: {image.size} bytes")
                return False
            
            # Check image shape
            if len(image.shape) not in [2, 3]:
                logger.error(f"Invalid image shape: {image.shape}")
                return False
            
            # Check for empty image
            if image.size == 0:
                logger.error("Empty image")
                return False
            
            # Check data type
            if image.dtype not in [np.uint8, np.uint16, np.float32, np.float64]:
                logger.error(f"Invalid image dtype: {image.dtype}")
                return False
            
            logger.debug("Image input validation passed")
            return True
            
        except Exception as e:
            logger.error(f"Image validation failed: {e}")
            return False

    def _generate_cache_key(self, image: np.ndarray) -> str:
        """Generate cache key for image"""
        try:
            import hashlib
            image_hash = hashlib.md5(image.tobytes()).hexdigest()
            return f"ocr_{image_hash}"
        except Exception as e:
            logger.error(f"Cache key generation failed: {e}")
            return f"ocr_{time.time()}"

    async def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for OCR"""
        try:
            if not self.performance_config.enable_preprocessing:
                return image
            
            processed = image.copy()
            
            # Convert to grayscale if needed
            if len(processed.shape) == 3:
                processed = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)
            
            # Apply adaptive preprocessing based on performance mode
            if self.performance_config.mode == OCRPerformanceMode.ACCURACY:
                processed = self._preprocess_for_accuracy(processed)
            elif self.performance_config.mode == OCRPerformanceMode.SPEED:
                processed = self._preprocess_for_speed(processed)
            else:
                processed = self._preprocess_balanced(processed)
            
            return processed
            
        except Exception as e:
            logger.error(f"Image preprocessing failed: {e}")
            return image

    def _preprocess_for_accuracy(self, image: np.ndarray) -> np.ndarray:
        """Preprocess for maximum accuracy"""
        try:
            import cv2
            
            # Denoising
            processed = cv2.fastNlMeansDenoising(image, None, 10, 10, 7, 21)
            
            # Contrast enhancement
            processed = cv2.equalizeHist(processed)
            
            # Sharpening
            kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
            processed = cv2.filter2D(processed, -1, kernel)
            
            return processed
            
        except Exception as e:
            logger.error(f"Accuracy preprocessing failed: {e}")
            return image

    def _preprocess_for_speed(self, image: np.ndarray) -> np.ndarray:
        """Preprocess for maximum speed"""
        try:
            # Simple thresholding
            _, processed = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            return processed
            
        except Exception as e:
            logger.error(f"Speed preprocessing failed: {e}")
            return image

    def _preprocess_balanced(self, image: np.ndarray) -> np.ndarray:
        """Preprocess for balanced performance"""
        try:
            import cv2
            
            # Simple denoising
            processed = cv2.medianBlur(image, 3)
            
            # Contrast enhancement
            processed = cv2.equalizeHist(processed)
            
            return processed
            
        except Exception as e:
            logger.error(f"Balanced preprocessing failed: {e}")
            return image

    async def _process_ocr_internal(self, image: np.ndarray) -> OCRResult:
        """Internal OCR processing"""
        try:
            # Use thread pool for CPU-intensive OCR operations
            return await asyncio.get_event_loop().run_in_executor(
                self._thread_pool,
                self._process_ocr_sync,
                image
            )
            
        except Exception as e:
            logger.error(f"Internal OCR processing failed: {e}")
            raise OCRError(f"OCR processing failed: {e}")

    def _process_ocr_sync(self, image: np.ndarray) -> OCRResult:
        """Synchronous OCR processing"""
        try:
            # Call base OCR engine
            return super().process_image(image)
            
        except Exception as e:
            logger.error(f"Synchronous OCR processing failed: {e}")
            raise OCRError(f"OCR processing failed: {e}")

    async def _validate_ocr_result(self, result: OCRResult) -> bool:
        """Validate OCR result for security"""
        try:
            # Validate confidence scores
            if self.security_config.validate_confidence:
                for item in result.items:
                    if item.confidence < self.security_config.min_confidence_threshold:
                        logger.warning(f"Low confidence item: {item.confidence}")
                        # Don't fail validation, just warn
            
            # Validate text content
            if self.content_filter:
                for item in result.items:
                    if not self.content_filter.validate_text(item.text):
                        logger.warning(f"Filtered content: {item.text}")
                        return False
            
            # Validate text length
            total_text = ' '.join(item.text for item in result.items)
            if len(total_text) > self.security_config.max_text_length:
                logger.error(f"Text too long: {len(total_text)} characters")
                return False
            
            logger.debug("OCR result validation passed")
            return True
            
        except Exception as e:
            logger.error(f"OCR result validation failed: {e}")
            return False

    async def _postprocess_ocr_result(self, result: OCRResult) -> EnhancedOCRResult:
        """Post-process OCR result"""
        try:
            # Sanitize output if enabled
            if self.output_sanitizer:
                sanitized_items = []
                for item in result.items:
                    sanitized_text = self.output_sanitizer.sanitize_text(item.text)
                    if sanitized_text:
                        sanitized_items.append(OCRTextItem(
                            text=sanitized_text,
                            confidence=item.confidence,
                            bbox=item.bbox,
                            type=item.type,
                            language=item.language,
                            line_number=item.line_number,
                            word_number=item.word_number
                        ))
                result.items = sanitized_items
            
            # Calculate quality score
            quality_score = self._calculate_quality_score(result)
            
            # Create enhanced result
            enhanced_result = EnhancedOCRResult(
                original_result=result,
                security_validated=True,
                quality_score=quality_score,
                security_info={
                    'validation_passed': True,
                    'content_filtered': self.content_filter is not None,
                    'output_sanitized': self.output_sanitizer is not None
                }
            )
            
            return enhanced_result
            
        except Exception as e:
            logger.error(f"OCR result post-processing failed: {e}")
            raise OCRError(f"Result post-processing failed: {e}")

    def _calculate_quality_score(self, result: OCRResult) -> float:
        """Calculate quality score for OCR result"""
        try:
            if not result.items:
                return 0.0
            
            # Average confidence
            avg_confidence = sum(item.confidence for item in result.items) / len(result.items)
            
            # Text completeness (ratio of non-empty items)
            non_empty_items = [item for item in result.items if item.text.strip()]
            completeness = len(non_empty_items) / len(result.items)
            
            # Combined quality score
            quality_score = (avg_confidence * 0.7 + completeness * 0.3)
            
            return quality_score
            
        except Exception as e:
            logger.error(f"Quality score calculation failed: {e}")
            return 0.5

    async def process_batch_enhanced(
        self, 
        images: List[np.ndarray]
    ) -> List[Optional[EnhancedOCRResult]]:
        """
        Process batch of images with enhanced performance
        
        Args:
            images: List of images to process
            
        Returns:
            List of enhanced OCR results
        """
        try:
            if not images:
                return []
            
            # Validate batch size
            if len(images) > self.performance_config.batch_size:
                logger.warning(f"Batch size exceeds limit: {len(images)}")
                # Process in chunks
                results = []
                for i in range(0, len(images), self.performance_config.batch_size):
                    chunk = images[i:i + self.performance_config.batch_size]
                    chunk_results = await self._process_batch_chunk(chunk)
                    results.extend(chunk_results)
                return results
            
            return await self._process_batch_chunk(images)
            
        except Exception as e:
            logger.error(f"Batch processing failed: {e}")
            raise OCRError(f"Batch processing failed: {e}")

    async def _process_batch_chunk(
        self, 
        images: List[np.ndarray]
    ) -> List[Optional[EnhancedOCRResult]]:
        """Process a chunk of images"""
        try:
            # Create tasks for concurrent processing
            tasks = []
            for image in images:
                task = asyncio.create_task(self.process_image_enhanced(image))
                tasks.append(task)
            
            # Wait for all tasks to complete
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process results
            processed_results = []
            for result in results:
                if isinstance(result, Exception):
                    logger.error(f"Batch processing error: {result}")
                    processed_results.append(None)
                else:
                    processed_results.append(result)
            
            return processed_results
            
        except Exception as e:
            logger.error(f"Batch chunk processing failed: {e}")
            raise OCRError(f"Batch processing failed: {e}")

    # Implement abstract methods from base class
    async def process_image(self, image: np.ndarray) -> OCRResult:
        """Process single image (base method)"""
        enhanced_result = await self.process_image_enhanced(image)
        return enhanced_result.original_result if enhanced_result else OCRResult(
            items=[], processing_time=0.0, engine_info={}, image_info={}
        )

    async def process_batch(self, images: List[np.ndarray]) -> List[OCRResult]:
        """Process batch of images (base method)"""
        enhanced_results = await self.process_batch_enhanced(images)
        return [
            result.original_result if result else OCRResult(
                items=[], processing_time=0.0, engine_info={}, image_info={}
            )
            for result in enhanced_results
        ]

    def get_supported_languages(self) -> List[str]:
        """Get supported languages"""
        return self._supported_languages

    def cleanup(self):
        """Cleanup resources"""
        try:
            # Shutdown thread pool
            self._thread_pool.shutdown(wait=True)
            
            # Clear cache
            if self.cache:
                self.cache.clear()
            
            # Call base cleanup
            super().cleanup()
            
            logger.info("Enhanced OCR engine cleaned up")
            
        except Exception as e:
            logger.error(f"Cleanup failed: {e}")

    async def get_health_status(self) -> Dict[str, Any]:
        """Get health status of OCR engine"""
        try:
            # Perform quick health check if needed
            if time.time() - self._last_health_check > 300:  # 5 minutes
                await self._perform_health_check()
            
            return {
                'healthy': self._health_status,
                'last_health_check': self._last_health_check,
                'gpu_available': getattr(self, 'gpu_available', False),
                'cache_enabled': self.cache is not None,
                'metrics': self.metrics_collector.get_metrics()
            }
            
        except Exception as e:
            logger.error(f"Health status check failed: {e}")
            return {'healthy': False, 'error': str(e)}


class ContentFilter:
    """Content filter for sensitive content"""

    def __init__(self, config: OCRSecurityConfig):
        self.config = config
        self.blocked_patterns = [re.compile(pattern, re.IGNORECASE) 
                                for pattern in config.blocked_patterns]

    def validate_text(self, text: str) -> bool:
        """Validate text content"""
        try:
            for pattern in self.blocked_patterns:
                if pattern.search(text):
                    logger.warning(f"Blocked pattern found: {pattern.pattern}")
                    return False
            return True
        except Exception as e:
            logger.error(f"Content validation failed: {e}")
            return False


class OutputSanitizer:
    """Output sanitizer for OCR results"""

    def __init__(self, config: OCRSecurityConfig):
        self.config = config

    def sanitize_text(self, text: str) -> str:
        """Sanitize text output"""
        try:
            # Remove control characters
            sanitized = ''.join(char for char in text if ord(char) >= 32 or char in '\n\r\t')
            
            # Normalize whitespace
            sanitized = ' '.join(sanitized.split())
            
            # Truncate if too long
            if len(sanitized) > self.config.max_text_length:
                sanitized = sanitized[:self.config.max_text_length]
            
            return sanitized.strip()
            
        except Exception as e:
            logger.error(f"Text sanitization failed: {e}")
            return text


# Factory for enhanced OCR engines
class EnhancedOCREngineFactory:
    """Factory for creating enhanced OCR engines"""

    @staticmethod
    def create_enhanced_engine(
        engine_name: str, 
        config: Dict[str, Any],
        security_config: Optional[OCRSecurityConfig] = None,
        performance_config: Optional[OCRPerformanceConfig] = None
    ) -> EnhancedOCREngine:
        """Create enhanced OCR engine"""
        try:
            # Create base engine
            base_engine = OCREngineFactory.create_engine(engine_name, config)
            
            # Create enhanced engine
            enhanced_engine = EnhancedOCREngine(config)
            
            # Override configurations if provided
            if security_config:
                enhanced_engine.security_config = security_config
            if performance_config:
                enhanced_engine.performance_config = performance_config
            
            return enhanced_engine
            
        except Exception as e:
            logger.error(f"Failed to create enhanced OCR engine: {e}")
            raise OCRError(f"Engine creation failed: {e}")