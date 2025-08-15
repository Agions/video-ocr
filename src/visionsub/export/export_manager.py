"""
Multiple export format support for VisionSub
"""
import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

from ..core.errors import ErrorContext, FileIOError
from ..models.subtitle import SubtitleItem

logger = logging.getLogger(__name__)


@dataclass
class ExportOptions:
    """Export options for different formats"""
    include_timestamps: bool = True
    include_index: bool = True
    encoding: str = 'utf-8'
    newline: str = '\n'
    indent_json: int = 2


class SubtitleExporter(ABC):
    """Abstract base class for subtitle exporters"""

    @abstractmethod
    async def export(
        self,
        subtitles: List[SubtitleItem],
        output_path: str,
        options: ExportOptions = None
    ) -> None:
        """
        Export subtitles to file

        Args:
            subtitles: List of subtitle items
            output_path: Output file path
            options: Export options
        """
        pass

    @abstractmethod
    def get_file_extension(self) -> str:
        """Get file extension for this format"""
        pass

    @abstractmethod
    def get_description(self) -> str:
        """Get format description"""
        pass


class SRTExporter(SubtitleExporter):
    """SRT (SubRip Text) format exporter"""

    async def export(
        self,
        subtitles: List[SubtitleItem],
        output_path: str,
        options: ExportOptions = None
    ) -> None:
        """Export subtitles in SRT format"""
        options = options or ExportOptions()

        try:
            content = []

            for i, subtitle in enumerate(subtitles):
                # Index
                if options.include_index:
                    content.append(str(subtitle.index))

                # Time range
                if options.include_timestamps:
                    content.append(f"{subtitle.start_time} --> {subtitle.end_time}")

                # Text content
                content.append(subtitle.content)
                
                # Empty line between entries (but not after last one)
                if i < len(subtitles) - 1 and (options.include_index or options.include_timestamps):
                    content.append("")

            # Write to file
            with open(output_path, 'w', encoding=options.encoding, newline='') as f:
                f.write(options.newline.join(content))

            logger.info(f"Exported {len(subtitles)} subtitles to SRT format: {output_path}")

        except Exception as e:
            error_context = ErrorContext(
                operation="srt_export",
                file_path=output_path
            )
            raise FileIOError(f"SRT export failed: {e}", error_context)

    def get_file_extension(self) -> str:
        return ".srt"

    def get_description(self) -> str:
        return "SubRip Text (SRT) format"


class WebVTTExporter(SubtitleExporter):
    """WebVTT (Web Video Text Tracks) format exporter"""

    async def export(
        self,
        subtitles: List[SubtitleItem],
        output_path: str,
        options: ExportOptions = None
    ) -> None:
        """Export subtitles in WebVTT format"""
        options = options or ExportOptions()

        try:
            content = ["WEBVTT", ""]

            for subtitle in subtitles:
                # Optional cue identifier
                if options.include_index:
                    content.append(f"cue-{subtitle.index}")

                # Time range (WebVTT format uses . instead of ,)
                if options.include_timestamps:
                    start_time = subtitle.start_time.replace(',', '.')
                    end_time = subtitle.end_time.replace(',', '.')
                    content.append(f"{start_time} --> {end_time}")

                # Text content
                content.append(subtitle.content)
                content.append("")

            # Write to file
            with open(output_path, 'w', encoding=options.encoding, newline=options.newline) as f:
                f.write(options.newline.join(content))

            logger.info(f"Exported {len(subtitles)} subtitles to WebVTT format: {output_path}")

        except Exception as e:
            error_context = ErrorContext(
                operation="webvtt_export",
                file_path=output_path
            )
            raise FileIOError(f"WebVTT export failed: {e}", error_context)

    def get_file_extension(self) -> str:
        return ".vtt"

    def get_description(self) -> str:
        return "Web Video Text Tracks (WebVTT) format"


class ASSExporter(SubtitleExporter):
    """ASS (Advanced SubStation Alpha) format exporter"""

    async def export(
        self,
        subtitles: List[SubtitleItem],
        output_path: str,
        options: ExportOptions = None
    ) -> None:
        """Export subtitles in ASS format"""
        options = options or ExportOptions()

        try:
            content = [
                "[Script Info]",
                "Title: VisionSub Export",
                "ScriptType: v4.00+",
                "WrapStyle: 0",
                "ScaledBorderAndShadow: yes",
                "YCbCr Matrix: None",
                "",
                "[V4+ Styles]",
                "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding",
                "Style: Default,Arial,20,&H00FFFFFF,&H000000FF,&H00000000,&H00000000,0,0,0,0,100,100,0,0,1,0,0,2,0,0,0,1",
                "",
                "[Events]",
                "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text"
            ]

            for subtitle in subtitles:
                # Convert SRT time to ASS time format
                start_time = self._srt_time_to_ass(subtitle.start_time)
                end_time = self._srt_time_to_ass(subtitle.end_time)

                # Format dialogue line
                dialogue = f"Dialogue: 0,{start_time},{end_time},Default,,0,0,0,,{subtitle.content}"
                content.append(dialogue)

            # Write to file
            with open(output_path, 'w', encoding=options.encoding, newline=options.newline) as f:
                f.write(options.newline.join(content))

            logger.info(f"Exported {len(subtitles)} subtitles to ASS format: {output_path}")

        except Exception as e:
            error_context = ErrorContext(
                operation="ass_export",
                file_path=output_path
            )
            raise FileIOError(f"ASS export failed: {e}", error_context)

    def _srt_time_to_ass(self, srt_time: str) -> str:
        """Convert SRT time format to ASS time format"""
        # SRT: 00:00:00,000 -> ASS: H:MM:SS.CC
        if ',' in srt_time:
            time_part, ms_part = srt_time.split(',')
        else:
            time_part, ms_part = srt_time, '000'

        h, m, s = time_part.split(':')
        centiseconds = int(ms_part) // 10

        return f"{h}:{m}:{s}.{centiseconds:02d}"

    def get_file_extension(self) -> str:
        return ".ass"

    def get_description(self) -> str:
        return "Advanced SubStation Alpha (ASS) format"


class PlainTextExporter(SubtitleExporter):
    """Plain text format exporter"""

    async def export(
        self,
        subtitles: List[SubtitleItem],
        output_path: str,
        options: ExportOptions = None
    ) -> None:
        """Export subtitles in plain text format"""
        options = options or ExportOptions()

        try:
            content = []

            for subtitle in subtitles:
                # Add timestamp if requested
                if options.include_timestamps:
                    content.append(f"[{subtitle.start_time}]")

                # Add text content
                content.append(subtitle.content)
                content.append("")  # Empty line between entries

            # Write to file
            with open(output_path, 'w', encoding=options.encoding, newline=options.newline) as f:
                f.write(options.newline.join(content))

            logger.info(f"Exported {len(subtitles)} subtitles to plain text format: {output_path}")

        except Exception as e:
            error_context = ErrorContext(
                operation="plaintext_export",
                file_path=output_path
            )
            raise FileIOError(f"Plain text export failed: {e}", error_context)

    def get_file_extension(self) -> str:
        return ".txt"

    def get_description(self) -> str:
        return "Plain text format"


class JSONExporter(SubtitleExporter):
    """JSON format exporter"""

    async def export(
        self,
        subtitles: List[SubtitleItem],
        output_path: str,
        options: ExportOptions = None
    ) -> None:
        """Export subtitles in JSON format"""
        options = options or ExportOptions()

        try:
            # Convert subtitles to dictionary format
            subtitles_data = []
            for subtitle in subtitles:
                subtitle_dict = {
                    'content': subtitle.content,
                    'start_time': subtitle.start_time,
                    'end_time': subtitle.end_time
                }

                if options.include_index:
                    subtitle_dict['index'] = subtitle.index

                subtitles_data.append(subtitle_dict)

            # Create export data structure
            export_data = {
                'format': 'VisionSub JSON Export',
                'version': '1.0',
                'total_subtitles': len(subtitles_data),
                'subtitles': subtitles_data
            }

            # Write to file
            with open(output_path, 'w', encoding=options.encoding) as f:
                json.dump(export_data, f, indent=options.indent_json, ensure_ascii=False)

            logger.info(f"Exported {len(subtitles)} subtitles to JSON format: {output_path}")

        except Exception as e:
            error_context = ErrorContext(
                operation="json_export",
                file_path=output_path
            )
            raise FileIOError(f"JSON export failed: {e}", error_context)

    def get_file_extension(self) -> str:
        return ".json"

    def get_description(self) -> str:
        return "JavaScript Object Notation (JSON) format"


class ExportManager:
    """Manages multiple export formats"""

    def __init__(self):
        self.exporters = {
            '.srt': SRTExporter(),
            '.vtt': WebVTTExporter(),
            '.ass': ASSExporter(),
            '.txt': PlainTextExporter(),
            '.json': JSONExporter(),
        }

    async def export_subtitles(
        self,
        subtitles: List[SubtitleItem],
        output_path: str,
        format: str = None,
        options: ExportOptions = None
    ) -> None:
        """
        Export subtitles in various formats

        Args:
            subtitles: List of subtitle items
            output_path: Output file path
            format: Export format (optional, inferred from file extension)
            options: Export options
        """
        # Determine format
        if format is None:
            format = Path(output_path).suffix.lower()

        # Get appropriate exporter
        exporter = self.exporters.get(format)
        if not exporter:
            raise ValueError(f"Unsupported export format: {format}")

        # If custom format is specified, adjust output path to include correct extension
        if format and Path(output_path).suffix.lower() != format:
            output_path = str(Path(output_path).with_suffix(format))

        # Export subtitles
        await exporter.export(subtitles, output_path, options)

    async def export_multiple_formats(
        self,
        subtitles: List[SubtitleItem],
        base_path: str,
        formats: List[str],
        options: ExportOptions = None
    ) -> List[str]:
        """
        Export subtitles in multiple formats

        Args:
            subtitles: List of subtitle items
            base_path: Base output path (without extension)
            formats: List of formats to export
            options: Export options

        Returns:
            List of exported file paths
        """
        exported_files = []

        for format in formats:
            output_path = f"{base_path}{format}"
            try:
                await self.export_subtitles(subtitles, output_path, format, options)
                exported_files.append(output_path)
            except Exception as e:
                logger.error(f"Failed to export {format}: {e}")

        return exported_files

    def get_supported_formats(self) -> Dict[str, str]:
        """Get supported export formats"""
        return {
            ext: exporter.get_description()
            for ext, exporter in self.exporters.items()
        }

    def is_format_supported(self, format: str) -> bool:
        """Check if format is supported"""
        return format.lower() in self.exporters

    def suggest_format(self, filename: str) -> str:
        """Suggest export format based on filename"""
        path = Path(filename)
        if path.suffix.lower() in self.exporters:
            return path.suffix.lower()
        return '.srt'  # Default to SRT


class BatchExporter:
    """Batch export functionality"""

    def __init__(self, export_manager: ExportManager):
        self.export_manager = export_manager
        self.export_queue = []
        self.is_processing = False

    async def add_to_queue(
        self,
        subtitles: List[SubtitleItem],
        output_path: str,
        format: str = None,
        options: ExportOptions = None
    ):
        """Add export task to queue"""
        self.export_queue.append({
            'subtitles': subtitles,
            'output_path': output_path,
            'format': format,
            'options': options
        })

    async def process_queue(self) -> List[str]:
        """Process all export tasks in queue"""
        if self.is_processing:
            raise RuntimeError("Already processing export queue")

        self.is_processing = True
        exported_files = []

        try:
            for task in self.export_queue:
                try:
                    await self.export_manager.export_subtitles(
                        task['subtitles'],
                        task['output_path'],
                        task['format'],
                        task['options']
                    )
                    exported_files.append(task['output_path'])
                    logger.info(f"Successfully exported: {task['output_path']}")
                except Exception as e:
                    logger.error(f"Export failed for {task['output_path']}: {e}")

            return exported_files

        finally:
            self.export_queue.clear()
            self.is_processing = False

    def get_queue_size(self) -> int:
        """Get current queue size"""
        return len(self.export_queue)

    def clear_queue(self):
        """Clear export queue"""
        self.export_queue.clear()
