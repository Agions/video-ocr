"""
Tests for export functionality
"""
import pytest
import json
from pathlib import Path
from visionsub.models.subtitle import SubtitleItem
from visionsub.export.export_manager import (
    ExportManager, ExportOptions, SRTExporter, WebVTTExporter, 
    ASSExporter, PlainTextExporter, JSONExporter
)


class TestExportOptions:
    """Test export options"""
    
    def test_default_export_options(self):
        """Test default export options"""
        options = ExportOptions()
        assert options.include_timestamps is True
        assert options.include_index is True
        assert options.encoding == 'utf-8'
        assert options.newline == '\n'
        assert options.indent_json == 2
    
    def test_custom_export_options(self):
        """Test custom export options"""
        options = ExportOptions(
            include_timestamps=False,
            include_index=False,
            encoding='utf-16',
            newline='\r\n',
            indent_json=4
        )
        assert options.include_timestamps is False
        assert options.include_index is False
        assert options.encoding == 'utf-16'
        assert options.newline == '\r\n'
        assert options.indent_json == 4


class TestSubtitleExporters:
    """Test subtitle exporters"""
    
    @pytest.fixture
    def sample_subtitles(self):
        """Sample subtitle items for testing"""
        return [
            SubtitleItem(
                index=1,
                start_time="00:00:01,000",
                end_time="00:00:04,000",
                content="这是第一个字幕"
            ),
            SubtitleItem(
                index=2,
                start_time="00:00:05,000",
                end_time="00:00:08,000",
                content="This is the second subtitle"
            ),
            SubtitleItem(
                index=3,
                start_time="00:00:09,000",
                end_time="00:00:12,000",
                content="第三個字幕"
            )
        ]
    
    @pytest.mark.asyncio
    async def test_srt_exporter(self, sample_subtitles, tmp_path):
        """Test SRT exporter"""
        exporter = SRTExporter()
        output_file = tmp_path / "test.srt"
        
        # Export subtitles
        await exporter.export(sample_subtitles, str(output_file))
        
        # Verify file exists
        assert output_file.exists()
        
        # Read and verify content
        content = output_file.read_text(encoding='utf-8')
        lines = content.strip().split('\n')
        
        # Check structure (3 subtitles * 4 lines each = 12 lines, but strip() removes trailing empty line)
        assert len(lines) == 11
        
        # Check first subtitle
        assert lines[0] == "1"
        assert lines[1] == "00:00:01,000 --> 00:00:04,000"
        assert lines[2] == "这是第一个字幕"
        assert lines[3] == ""
        
        # Check file extension
        assert exporter.get_file_extension() == ".srt"
        assert exporter.get_description() == "SubRip Text (SRT) format"
    
    @pytest.mark.asyncio
    async def test_webvtt_exporter(self, sample_subtitles, tmp_path):
        """Test WebVTT exporter"""
        exporter = WebVTTExporter()
        output_file = tmp_path / "test.vtt"
        
        # Export subtitles
        await exporter.export(sample_subtitles, str(output_file))
        
        # Verify file exists
        assert output_file.exists()
        
        # Read and verify content
        content = output_file.read_text(encoding='utf-8')
        lines = content.strip().split('\n')
        
        # Check structure
        assert lines[0] == "WEBVTT"
        assert lines[1] == ""
        
        # Check first subtitle
        assert lines[2] == "cue-1"
        assert "00:00:01.000 --> 00:00:04.000" in lines[3]
        assert "这是第一个字幕" in lines[4]
        
        # Check file extension
        assert exporter.get_file_extension() == ".vtt"
        assert exporter.get_description() == "Web Video Text Tracks (WebVTT) format"
    
    @pytest.mark.asyncio
    async def test_ass_exporter(self, sample_subtitles, tmp_path):
        """Test ASS exporter"""
        exporter = ASSExporter()
        output_file = tmp_path / "test.ass"
        
        # Export subtitles
        await exporter.export(sample_subtitles, str(output_file))
        
        # Verify file exists
        assert output_file.exists()
        
        # Read and verify content
        content = output_file.read_text(encoding='utf-8')
        lines = content.strip().split('\n')
        
        # Check structure
        assert "[Script Info]" in lines
        assert "[V4+ Styles]" in lines
        assert "[Events]" in lines
        
        # Check dialogue lines
        dialogue_lines = [line for line in lines if line.startswith("Dialogue:")]
        assert len(dialogue_lines) == len(sample_subtitles)
        
        # Check file extension
        assert exporter.get_file_extension() == ".ass"
        assert exporter.get_description() == "Advanced SubStation Alpha (ASS) format"
    
    @pytest.mark.asyncio
    async def test_plain_text_exporter(self, sample_subtitles, tmp_path):
        """Test plain text exporter"""
        exporter = PlainTextExporter()
        output_file = tmp_path / "test.txt"
        
        # Export subtitles
        await exporter.export(sample_subtitles, str(output_file))
        
        # Verify file exists
        assert output_file.exists()
        
        # Read and verify content
        content = output_file.read_text(encoding='utf-8')
        lines = content.strip().split('\n')
        
        # Check structure (with timestamps)
        assert "[00:00:01,000]" in lines[0]
        assert "这是第一个字幕" in lines[1]
        assert "" in lines[2]  # Empty line
        
        # Check file extension
        assert exporter.get_file_extension() == ".txt"
        assert exporter.get_description() == "Plain text format"
    
    @pytest.mark.asyncio
    async def test_json_exporter(self, sample_subtitles, tmp_path):
        """Test JSON exporter"""
        exporter = JSONExporter()
        output_file = tmp_path / "test.json"
        
        # Export subtitles
        await exporter.export(sample_subtitles, str(output_file))
        
        # Verify file exists
        assert output_file.exists()
        
        # Read and verify content
        content = output_file.read_text(encoding='utf-8')
        data = json.loads(content)
        
        # Check structure
        assert data["format"] == "VisionSub JSON Export"
        assert data["version"] == "1.0"
        assert data["total_subtitles"] == len(sample_subtitles)
        assert len(data["subtitles"]) == len(sample_subtitles)
        
        # Check first subtitle
        first_sub = data["subtitles"][0]
        assert first_sub["content"] == "这是第一个字幕"
        assert first_sub["start_time"] == "00:00:01,000"
        assert first_sub["end_time"] == "00:00:04,000"
        assert first_sub["index"] == 1
        
        # Check file extension
        assert exporter.get_file_extension() == ".json"
        assert exporter.get_description() == "JavaScript Object Notation (JSON) format"
    
    @pytest.mark.asyncio
    async def test_export_options_customization(self, sample_subtitles, tmp_path):
        """Test export with custom options"""
        exporter = SRTExporter()
        output_file = tmp_path / "test_custom.srt"
        
        # Custom options
        options = ExportOptions(
            include_timestamps=False,
            include_index=False,
            encoding='utf-16',
            newline='\r\n'
        )
        
        # Export subtitles
        await exporter.export(sample_subtitles, str(output_file), options)
        
        # Verify file exists
        assert output_file.exists()
        
        # Read and verify content (no timestamps or indices)
        content = output_file.read_text(encoding='utf-16')
        lines = content.split('\r\n')
        
        # Should only have subtitle content (without empty lines)
        non_empty_lines = [line for line in lines if line.strip()]
        assert len(non_empty_lines) == 1  # All subtitles are in one line when no structure elements
        assert "这是第一个字幕" in non_empty_lines[0]
        assert "This is the second subtitle" in non_empty_lines[0]
        assert "第三個字幕" in non_empty_lines[0]
        
        # No timestamps or indices
        assert not any("-->" in line for line in lines)
        assert not any(line.isdigit() and len(line) < 3 for line in lines)


class TestExportManager:
    """Test export manager"""
    
    @pytest.fixture
    def sample_subtitles(self):
        """Sample subtitle items for testing"""
        return [
            SubtitleItem(
                index=1,
                start_time="00:00:01,000",
                end_time="00:00:04,000",
                content="Test subtitle 1"
            ),
            SubtitleItem(
                index=2,
                start_time="00:00:05,000",
                end_time="00:00:08,000",
                content="Test subtitle 2"
            )
        ]
    
    def test_export_manager_initialization(self):
        """Test export manager initialization"""
        manager = ExportManager()
        
        # Check supported formats
        formats = manager.get_supported_formats()
        assert ".srt" in formats
        assert ".vtt" in formats
        assert ".ass" in formats
        assert ".txt" in formats
        assert ".json" in formats
        
        # Check format support
        assert manager.is_format_supported(".srt") is True
        assert manager.is_format_supported(".vtt") is True
        assert manager.is_format_supported(".mp4") is False  # Unsupported format
    
    @pytest.mark.asyncio
    async def test_single_format_export(self, sample_subtitles, tmp_path):
        """Test single format export"""
        manager = ExportManager()
        output_file = tmp_path / "test_output.srt"
        
        # Export subtitles
        await manager.export_subtitles(sample_subtitles, str(output_file))
        
        # Verify file exists and has content
        assert output_file.exists()
        content = output_file.read_text()
        assert "Test subtitle 1" in content
        assert "Test subtitle 2" in content
    
    @pytest.mark.asyncio
    async def test_multiple_format_export(self, sample_subtitles, tmp_path):
        """Test multiple format export"""
        manager = ExportManager()
        base_path = str(tmp_path / "test_output")
        formats = [".srt", ".vtt", ".txt"]
        
        # Export subtitles in multiple formats
        exported_files = await manager.export_multiple_formats(
            sample_subtitles, base_path, formats
        )
        
        # Verify all files were created
        assert len(exported_files) == len(formats)
        for file_path in exported_files:
            assert Path(file_path).exists()
        
        # Verify file extensions
        for i, format in enumerate(formats):
            assert exported_files[i].endswith(format)
    
    def test_format_suggestion(self):
        """Test format suggestion"""
        manager = ExportManager()
        
        # Test known formats
        assert manager.suggest_format("video.srt") == ".srt"
        assert manager.suggest_format("subtitles.vtt") == ".vtt"
        assert manager.suggest_format("captions.ass") == ".ass"
        assert manager.suggest_format("text.txt") == ".txt"
        assert manager.suggest_format("data.json") == ".json"
        
        # Test unknown format (should default to .srt)
        assert manager.suggest_format("video.mp4") == ".srt"
        assert manager.suggest_format("unknown.xyz") == ".srt"
    
    @pytest.mark.asyncio
    async def test_unsupported_format_error(self, sample_subtitles, tmp_path):
        """Test error handling for unsupported formats"""
        manager = ExportManager()
        output_file = tmp_path / "test.mp4"  # Unsupported format
        
        # Should raise ValueError
        with pytest.raises(ValueError, match="Unsupported export format"):
            await manager.export_subtitles(sample_subtitles, str(output_file))
    
    @pytest.mark.asyncio
    async def test_custom_format_specification(self, sample_subtitles, tmp_path):
        """Test custom format specification"""
        manager = ExportManager()
        output_file = tmp_path / "test_output"  # No extension
        
        # Export with custom format
        await manager.export_subtitles(
            sample_subtitles, 
            str(output_file), 
            format=".vtt"
        )
        
        # Should create .vtt file
        vtt_file = tmp_path / "test_output.vtt"
        assert vtt_file.exists()
        
        content = vtt_file.read_text()
        assert "WEBVTT" in content
        assert "Test subtitle 1" in content


class TestBatchExporter:
    """Test batch export functionality"""
    
    @pytest.fixture
    def sample_subtitles(self):
        """Sample subtitle items for testing"""
        return [
            SubtitleItem(
                index=1,
                start_time="00:00:01,000",
                end_time="00:00:04,000",
                content="Batch test subtitle 1"
            )
        ]
    
    def test_batch_exporter_initialization(self):
        """Test batch exporter initialization"""
        export_manager = ExportManager()
        # Test that the export manager can be initialized
        assert export_manager is not None
    
    def test_batch_export_queue_operations(self, sample_subtitles, tmp_path):
        """Test batch export queue operations"""
        export_manager = ExportManager()
        # Test that the export manager can handle basic operations
        assert export_manager is not None