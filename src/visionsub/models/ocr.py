"""
OCR result models for VisionSub
"""
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from dataclasses import dataclass


@dataclass
class OCRResult:
    """OCR processing result"""
    text: str = ""
    confidence: float = 0.0
    boxes: List[List[int]] = Field(default_factory=list)
    processing_time: float = 0.0
    language: str = "unknown"
    engine: str = "unknown"
    
    @classmethod
    def empty(cls) -> 'OCRResult':
        """Create empty OCR result"""
        return cls()
    
    def is_empty(self) -> bool:
        """Check if result is empty"""
        return not self.text.strip()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "text": self.text,
            "confidence": self.confidence,
            "boxes": self.boxes,
            "processing_time": self.processing_time,
            "language": self.language,
            "engine": self.engine
        }


class OCRBatchRequest(BaseModel):
    """Batch OCR request model"""
    requests: List[Dict[str, Any]]
    priority: int = Field(default=0, ge=0, le=10)
    config: Dict[str, Any] = Field(default_factory=dict)


class OCRBatchResponse(BaseModel):
    """Batch OCR response model"""
    batch_id: str
    job_ids: List[str]
    status: str
    total_requests: int
    timestamp: float


class OCRJobStatus(BaseModel):
    """OCR job status model"""
    job_id: str
    status: str  # "queued", "processing", "completed", "failed"
    progress: float = Field(default=0.0, ge=0.0, le=1.0)
    worker: Optional[str] = None
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    processing_time: Optional[float] = None
    error: Optional[str] = None
    result: Optional[OCRResult] = None