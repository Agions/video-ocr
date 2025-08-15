#!/usr/bin/env python3
"""
VisionSub Command Line Interface
Professional video OCR subtitle extraction tool
"""
import asyncio
import argparse
import json
import logging
import sys
from pathlib import Path
from typing import List, Optional

from visionsub.core.engine import ProcessingEngine
from visionsub.export.export_manager import ExportManager, ExportOptions
from visionsub.models.config import AppConfig, ProcessingConfig, OcrConfig
from visionsub.core.roi_manager import ROIManager


def setup_logging(level: str = "INFO", log_file: Optional[str] = None):
    """Setup logging configuration"""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            *([logging.FileHandler(log_file)] if log_file else [])
        ]
    )


def create_config_from_args(args) -> ProcessingConfig:
    """Create processing configuration from command line arguments"""
    # OCR configuration
    ocr_config = OcrConfig(
        engine=args.engine,
        language=args.language,
        confidence_threshold=args.confidence_threshold,
        roi_rect=tuple(args.roi_rect) if args.roi_rect else (0, 0, 0, 0),
        denoise=args.denoise,
        enhance_contrast=args.enhance_contrast,
        threshold=args.threshold,
        sharpen=args.sharpen,
        auto_detect_language=args.auto_detect_language,
        enable_preprocessing=args.enable_preprocessing,
        enable_postprocessing=args.enable_postprocessing
    )

    # Processing configuration
    processing_config = ProcessingConfig(
        ocr_config=ocr_config,
        scene_threshold=args.scene_threshold,
        cache_size=args.cache_size,
        frame_interval=args.frame_interval,
        enable_scene_detection=args.enable_scene_detection,
        enable_parallel_processing=args.enable_parallel_processing,
        memory_limit_mb=args.memory_limit,
        output_formats=args.export_formats.split(',') if args.export_formats else ['srt'],
        output_directory=args.output_dir,
        create_subdirectories=args.create_subdirectories,
        enable_performance_monitoring=args.enable_performance_monitoring
    )

    return processing_config


async def process_video(args):
    """Process a single video file"""
    logger = logging.getLogger(__name__)
    
    try:
        # Create configuration
        config = create_config_from_args(args)
        
        # Initialize processing engine
        engine = ProcessingEngine(config)
        
        # Setup ROI if specified
        if args.roi_config:
            roi_manager = engine.get_roi_manager()
            roi_manager.load_rois(args.roi_config)
        
        # Process video
        logger.info(f"Processing video: {args.input}")
        subtitles = await engine.process_video(args.input)
        
        logger.info(f"Extracted {len(subtitles)} subtitle items")
        
        # Export subtitles
        export_manager = ExportManager()
        export_options = ExportOptions(
            include_timestamps=True,
            include_index=True,
            encoding=args.encoding,
            newline=args.newline
        )
        
        # Determine output formats
        formats = args.export_formats.split(',') if args.export_formats else ['srt']
        
        # Create base output path
        input_path = Path(args.input)
        base_name = input_path.stem
        
        if args.output:
            base_path = args.output
        else:
            output_dir = Path(args.output_dir or "./output")
            output_dir.mkdir(parents=True, exist_ok=True)
            base_path = str(output_dir / base_name)
        
        # Export in multiple formats
        exported_files = await export_manager.export_multiple_formats(
            subtitles, base_path, formats, export_options
        )
        
        logger.info(f"Exported files: {exported_files}")
        
        # Print statistics
        stats = engine.get_processing_stats()
        logger.info(f"Processing statistics: {json.dumps(stats, indent=2)}")
        
        return exported_files
        
    except Exception as e:
        logger.error(f"Video processing failed: {e}")
        raise


async def process_multiple_videos(args):
    """Process multiple video files"""
    logger = logging.getLogger(__name__)
    
    input_files = args.input.split(',') if isinstance(args.input, str) else args.input
    all_exported_files = []
    
    for video_file in input_files:
        logger.info(f"Processing video: {video_file}")
        
        # Update args for current file
        current_args = argparse.Namespace(**vars(args))
        current_args.input = video_file
        
        try:
            exported_files = await process_video(current_args)
            all_exported_files.extend(exported_files)
        except Exception as e:
            logger.error(f"Failed to process {video_file}: {e}")
            continue
    
    logger.info(f"Total exported files: {len(all_exported_files)}")
    return all_exported_files


def list_roi_presets():
    """List available ROI presets"""
    roi_manager = ROIManager()
    rois = roi_manager.get_all_rois()
    
    print("Available ROI presets:")
    print("-" * 50)
    for roi in rois:
        print(f"Name: {roi.name}")
        print(f"Type: {roi.type.value}")
        print(f"Rectangle: {roi.rect}")
        print(f"Description: {roi.description}")
        print(f"Confidence Threshold: {roi.confidence_threshold}")
        print("-" * 50)


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="VisionSub - Professional Video OCR Subtitle Extraction Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  visionsub-cli input.mp4 -o output.srt
  
  # Advanced usage with custom settings
  visionsub-cli input.mp4 \\
    --language ch \\
    --scene-threshold 0.3 \\
    --cache-size 100 \\
    --export-formats srt,vtt,json \\
    --output-dir ./exports
  
  # Use custom ROI
  visionsub-cli input.mp4 \\
    --roi-rect 0,900,1920,180 \\
    --confidence-threshold 0.7
  
  # Batch processing
  visionsub-cli video1.mp4,video2.mp4,video3.mp4 \\
    --output-dir ./batch_output
        """
    )
    
    # Input/Output arguments
    parser.add_argument('input', help='Input video file(s) (comma-separated for multiple)')
    parser.add_argument('-o', '--output', help='Output file path (single format)')
    parser.add_argument('--output-dir', default='./output', help='Output directory')
    parser.add_argument('--export-formats', default='srt', help='Export formats (comma-separated)')
    
    # OCR configuration
    parser.add_argument('--engine', choices=['PaddleOCR', 'Tesseract'], default='PaddleOCR',
                       help='OCR engine to use')
    parser.add_argument('--language', default='中文', help='OCR language display name')
    parser.add_argument('--confidence-threshold', type=float, default=0.8,
                       help='Minimum confidence threshold (0.0-1.0)')
    parser.add_argument('--roi-rect', type=int, nargs=4, metavar=('X', 'Y', 'W', 'H'),
                       help='ROI rectangle (x y width height)')
    parser.add_argument('--roi-config', help='Path to ROI configuration file')
    
    # Preprocessing options
    parser.add_argument('--denoise', action='store_true', default=True,
                       help='Enable image denoising')
    parser.add_argument('--no-denoise', action='store_false', dest='denoise',
                       help='Disable image denoising')
    parser.add_argument('--enhance-contrast', action='store_true', default=True,
                       help='Enable contrast enhancement')
    parser.add_argument('--no-enhance-contrast', action='store_false', dest='enhance_contrast',
                       help='Disable contrast enhancement')
    parser.add_argument('--threshold', type=int, default=180,
                       help='Binarization threshold (0-255)')
    parser.add_argument('--sharpen', action='store_true', default=True,
                       help='Enable image sharpening')
    parser.add_argument('--no-sharpen', action='store_false', dest='sharpen',
                       help='Disable image sharpening')
    parser.add_argument('--auto-detect-language', action='store_true', default=True,
                       help='Enable automatic language detection')
    parser.add_argument('--enable-preprocessing', action='store_true', default=True,
                       help='Enable image preprocessing')
    parser.add_argument('--enable-postprocessing', action='store_true', default=True,
                       help='Enable text postprocessing')
    
    # Processing options
    parser.add_argument('--scene-threshold', type=float, default=0.3,
                       help='Scene change detection threshold (0.0-1.0)')
    parser.add_argument('--cache-size', type=int, default=100,
                       help='Maximum cache size')
    parser.add_argument('--frame-interval', type=float, default=1.0,
                       help='Frame processing interval in seconds')
    parser.add_argument('--enable-scene-detection', action='store_true', default=True,
                       help='Enable scene change detection')
    parser.add_argument('--enable-parallel-processing', action='store_true', default=True,
                       help='Enable parallel processing')
    parser.add_argument('--memory-limit', type=int, default=1024,
                       help='Memory limit in MB')
    parser.add_argument('--create-subdirectories', action='store_true', default=True,
                       help='Create subdirectories for each video')
    parser.add_argument('--enable-performance-monitoring', action='store_true', default=True,
                       help='Enable performance monitoring')
    
    # Output options
    parser.add_argument('--encoding', default='utf-8', help='Output file encoding')
    parser.add_argument('--newline', default='\\n', help='Newline character')
    
    # Logging and debugging
    parser.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], 
                       default='INFO', help='Logging level')
    parser.add_argument('--log-file', help='Log file path')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    
    # Utility commands
    parser.add_argument('--list-roi-presets', action='store_true',
                       help='List available ROI presets')
    
    args = parser.parse_args()
    
    # Setup logging
    if args.debug:
        args.log_level = 'DEBUG'
    
    setup_logging(args.log_level, args.log_file)
    logger = logging.getLogger(__name__)
    
    # Handle utility commands
    if args.list_roi_presets:
        list_roi_presets()
        return 0
    
    try:
        # Run processing
        if ',' in args.input:
            # Multiple files
            exported_files = asyncio.run(process_multiple_videos(args))
        else:
            # Single file
            exported_files = asyncio.run(process_video(args))
        
        logger.info(f"Processing completed successfully!")
        return 0
        
    except KeyboardInterrupt:
        logger.info("Processing interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"Processing failed: {e}")
        return 1


if __name__ == '__main__':
    sys.exit(main())