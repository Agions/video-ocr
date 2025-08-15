from pydantic import BaseModel, Field


class SubtitleItem(BaseModel):
    """
    Represents a single, processed subtitle entry.
    """
    index: int = Field(..., ge=1, description="The sequential index of the subtitle.")
    start_time: str = Field(..., pattern=r"\d{2}:\d{2}:\d{2},\d{3}", description="Start time in SRT format (HH:MM:SS,ms).")
    end_time: str = Field(..., pattern=r"\d{2}:\d{2}:\d{2},\d{3}", description="End time in SRT format (HH:MM:SS,ms).")
    content: str = Field(..., description="The recognized text content of the subtitle.")

    def to_srt_format(self) -> str:
        """Converts the subtitle item to its string representation in SRT format."""
        return f"{self.index}\n{self.start_time} --> {self.end_time}\n{self.content}\n"
