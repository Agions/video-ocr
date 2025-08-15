from pydantic import BaseModel, Field


class VideoMetadata(BaseModel):
    """
    Holds metadata about the source video file.
    """
    file_path: str
    duration: float = Field(..., ge=0)
    fps: float = Field(..., gt=0)
    width: int = Field(..., gt=0)
    height: int = Field(..., gt=0)
    frame_count: int = Field(..., ge=0)
