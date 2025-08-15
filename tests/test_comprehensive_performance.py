"""
Comprehensive performance tests for VisionSub components
"""

import pytest
import asyncio
import time
import psutil
import numpy as np
import cv2
import tempfile
import os
from unittest.mock import patch
from pathlib import Path

from visionsub.core.video_processor import VideoProcessor
from visionsub.core.async_ocr_engine import AsyncOCREngine
from visionsub.core.memory_manager import MemoryManager
from visionsub.core.logging_system import StructuredLogger
from visionsub.core.health_check import HealthCheck


class TestVideoProcessingPerformance:
    """Performance tests for video processing"""
    
    @pytest.fixture
    def video_processor(self):
        """Create video processor instance"""
        return VideoProcessor()
    
    @pytest.fixture
    def large_test_video(self):
        """Create a large test video for performance testing"""
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as f:
            fourcc = cv2.VideoWriter_fourcc(*'MP4V')
            out = cv2.VideoWriter(f.name, fourcc, 30.0, (1920, 1080))
            
            # Create 300 frames (10 seconds at 30 fps)
            for i in range(300):
                frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
                # Add text content
                cv2.putText(frame, f"Performance Test Frame {i}", (100, 100), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                out.write(frame)
            
            out.release()
            return f.name
    
    @pytest.mark.performance
    def test_video_info_extraction_performance(self, video_processor, large_test_video):
        """Test video info extraction performance"""
        start_time = time.time()
        
        # Extract video info multiple times
        for _ in range(10):
            info = video_processor.get_video_info(large_test_video)
            assert info['frame_count'] == 300
            assert info['fps'] == 30.0
        
        end_time = time.time()
        total_time = end_time - start_time
        
        print(f"Video info extraction (10 iterations): {total_time:.3f}s")
        print(f"Average time per extraction: {total_time/10:.3f}s")
        
        # Should complete within reasonable time
        assert total_time < 5.0  # 5 seconds for 10 extractions
    
    @pytest.mark.performance
    def test_frame_extraction_performance(self, video_processor, large_test_video):
        """Test frame extraction performance"""
        start_time = time.time()
        
        # Extract frames with different intervals
        intervals = [0.5, 1.0, 2.0]
        frame_counts = []
        
        for interval in intervals:
            frames = video_processor.extract_frames(large_test_video, interval=interval)
            frame_counts.append(len(frames))
            print(f"Interval {interval}s: {len(frames)} frames extracted")
        
        end_time = time.time()
        total_time = end_time - start_time
        
        print(f"Frame extraction (3 intervals): {total_time:.3f}s")
        
        # Verify frame counts are correct
        expected_counts = [int(300 / (interval * 30)) for interval in intervals]
        for actual, expected in zip(frame_counts, expected_counts):
            assert abs(actual - expected) <= 1  # Allow for rounding
        
        # Should complete within reasonable time
        assert total_time < 10.0  # 10 seconds for all extractions
    
    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_async_frame_processing_performance(self, video_processor, large_test_video):
        """Test async frame processing performance"""
        start_time = time.time()
        
        # Process frames asynchronously
        frames = await video_processor._process_frames_async(large_test_video, 1.0)
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        print(f"Async frame processing: {processing_time:.3f}s")
        print(f"Frames processed: {len(frames)}")
        print(f"Average time per frame: {processing_time/len(frames):.3f}s")
        
        # Should process frames efficiently
        assert len(frames) > 0
        assert processing_time < 15.0  # 15 seconds for async processing
    
    @pytest.mark.performance
    def test_memory_usage_during_processing(self, video_processor, large_test_video):
        """Test memory usage during video processing"""
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Process video
        frames = video_processor.extract_frames(large_test_video, interval=1.0)
        
        peak_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = peak_memory - initial_memory
        
        print(f"Initial memory: {initial_memory:.1f} MB")
        print(f"Peak memory: {peak_memory:.1f} MB")
        print(f"Memory increase: {memory_increase:.1f} MB")
        print(f"Frames extracted: {len(frames)}")
        
        # Memory increase should be reasonable
        assert memory_increase < 500  # Less than 500MB increase
        
        # Clean up frames to free memory
        del frames
    
    finally:
        if 'large_test_video' in locals():
            os.unlink(large_test_video)


class TestOCREnginePerformance:
    """Performance tests for OCR engine"""
    
    @pytest.fixture
    def ocr_engine(self):
        """Create OCR engine instance"""
        return AsyncOCREngine()
    
    @pytest.fixture
    def test_images(self):
        """Create test images for OCR performance testing"""
        images = []
        for i in range(20):
            # Create images with different text content
            img = np.zeros((400, 600, 3), dtype=np.uint8)
            texts = [
                f"Performance Test Text {i}",
                f"OCR Engine Benchmark {i}",
                f"VisionSub Performance {i}",
                f"Async Processing Test {i}",
                f"Text Recognition Speed {i}"
            ]
            
            text = texts[i % len(texts)]
            cv2.putText(img, text, (50, 200), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            images.append(img)
        
        return images
    
    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_single_image_ocr_performance(self, ocr_engine, test_images):
        """Test single image OCR performance"""
        times = []
        
        for img in test_images:
            start_time = time.time()
            result = await ocr_engine.process_image(img)
            end_time = time.time()
            
            times.append(end_time - start_time)
            
            # Verify result
            assert 'text' in result
            assert 'confidence' in result
            assert 'boxes' in result
        
        avg_time = sum(times) / len(times)
        max_time = max(times)
        min_time = min(times)
        
        print(f"Single image OCR performance:")
        print(f"  Average time: {avg_time:.3f}s")
        print(f"  Min time: {min_time:.3f}s")
        print(f"  Max time: {max_time:.3f}s")
        print(f"  Total images: {len(test_images)}")
        
        # Should process images efficiently
        assert avg_time < 2.0  # Average less than 2 seconds per image
        assert max_time < 5.0  # Maximum less than 5 seconds per image
    
    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_batch_ocr_performance(self, ocr_engine, test_images):
        """Test batch OCR performance"""
        batch_sizes = [1, 5, 10, 20]
        
        for batch_size in batch_sizes:
            start_time = time.time()
            
            # Process batch
            results = await ocr_engine.process_batch(test_images[:batch_size])
            
            end_time = time.time()
            batch_time = end_time - start_time
            
            print(f"Batch size {batch_size}: {batch_time:.3f}s ({batch_time/batch_size:.3f}s per image)")
            
            # Verify results
            assert len(results) == batch_size
            for result in results:
                assert 'text' in result
                assert 'confidence' in result
            
            # Batch processing should be efficient
            assert batch_time < batch_size * 3.0  # Less than 3 seconds per image
    
    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_concurrent_ocr_performance(self, ocr_engine, test_images):
        """Test concurrent OCR processing performance"""
        # Create multiple concurrent tasks
        tasks = []
        for i in range(0, len(test_images), 5):
            batch = test_images[i:i+5]
            tasks.append(ocr_engine.process_batch(batch))
        
        start_time = time.time()
        results = await asyncio.gather(*tasks)
        end_time = time.time()
        
        total_time = end_time - start_time
        total_images = sum(len(batch) for batch in results)
        
        print(f"Concurrent OCR processing:")
        print(f"  Total time: {total_time:.3f}s")
        print(f"  Total images: {total_images}")
        print(f"  Average time per image: {total_time/total_images:.3f}s")
        print(f"  Concurrent tasks: {len(tasks)}")
        
        # Concurrent processing should be efficient
        assert total_time < total_images * 2.0  # Less than 2 seconds per image
    
    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_ocr_engine_memory_usage(self, ocr_engine, test_images):
        """Test OCR engine memory usage"""
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Process images
        results = await ocr_engine.process_batch(test_images)
        
        peak_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = peak_memory - initial_memory
        
        print(f"OCR engine memory usage:")
        print(f"  Initial memory: {initial_memory:.1f} MB")
        print(f"  Peak memory: {peak_memory:.1f} MB")
        print(f"  Memory increase: {memory_increase:.1f} MB")
        print(f"  Images processed: {len(test_images)}")
        
        # Memory usage should be reasonable
        assert memory_increase < 1000  # Less than 1GB increase for 20 images


class TestMemoryManagerPerformance:
    """Performance tests for memory manager"""
    
    @pytest.fixture
    def memory_manager(self):
        """Create memory manager instance"""
        return MemoryManager(max_size="50MB")
    
    @pytest.mark.performance
    def test_cache_operations_performance(self, memory_manager):
        """Test cache operations performance"""
        # Test cache set performance
        set_times = []
        for i in range(100):
            img = np.zeros((100, 100, 3), dtype=np.uint8)
            
            start_time = time.time()
            memory_manager.cache.set(f"key_{i}", img)
            end_time = time.time()
            
            set_times.append(end_time - start_time)
        
        avg_set_time = sum(set_times) / len(set_times)
        print(f"Cache set average time: {avg_set_time:.6f}s")
        
        # Test cache get performance
        get_times = []
        for i in range(100):
            start_time = time.time()
            result = memory_manager.cache.get(f"key_{i}")
            end_time = time.time()
            
            get_times.append(end_time - start_time)
            assert result is not None
        
        avg_get_time = sum(get_times) / len(get_times)
        print(f"Cache get average time: {avg_get_time:.6f}s")
        
        # Cache operations should be fast
        assert avg_set_time < 0.001  # Less than 1ms
        assert avg_get_time < 0.001  # Less than 1ms
    
    @pytest.mark.performance
    def test_cache_eviction_performance(self, memory_manager):
        """Test cache eviction performance"""
        # Add large images to trigger eviction
        large_images = []
        for i in range(20):
            img = np.zeros((1000, 1000, 3), dtype=np.uint8)  # ~3MB each
            large_images.append(img)
        
        start_time = time.time()
        
        for i, img in enumerate(large_images):
            memory_manager.cache.set(f"large_img_{i}", img)
        
        end_time = time.time()
        eviction_time = end_time - start_time
        
        print(f"Cache eviction time: {eviction_time:.3f}s")
        print(f"Images processed: {len(large_images)}")
        
        # Check memory usage
        usage = memory_manager.get_memory_usage()
        print(f"Final memory usage: {usage['percentage']:.1f}%")
        
        # Eviction should maintain memory limit
        assert usage['percentage'] <= 100
        
        # Eviction should be efficient
        assert eviction_time < 5.0  # Less than 5 seconds for 20 images
    
    @pytest.mark.performance
    def test_memory_usage_tracking_performance(self, memory_manager):
        """Test memory usage tracking performance"""
        # Test memory usage calculation performance
        times = []
        
        for _ in range(100):
            start_time = time.time()
            usage = memory_manager.get_memory_usage()
            end_time = time.time()
            
            times.append(end_time - start_time)
            
            # Verify usage data
            assert 'total' in usage
            assert 'used' in usage
            assert 'available' in usage
            assert 'percentage' in usage
        
        avg_time = sum(times) / len(times)
        print(f"Memory usage tracking average time: {avg_time:.6f}s")
        
        # Memory tracking should be very fast
        assert avg_time < 0.0001  # Less than 0.1ms


class TestSystemPerformance:
    """System-wide performance tests"""
    
    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_end_to_end_performance(self):
        """Test end-to-end system performance"""
        # Create test video
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as f:
            fourcc = cv2.VideoWriter_fourcc(*'MP4V')
            out = cv2.VideoWriter(f.name, fourcc, 30.0, (640, 480))
            
            for i in range(60):  # 2 seconds at 30 fps
                frame = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(frame, f"E2E Test Frame {i}", (50, 240), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                out.write(frame)
            
            out.release()
            video_path = f.name
        
        try:
            # Initialize components
            video_processor = VideoProcessor()
            ocr_engine = AsyncOCREngine()
            memory_manager = MemoryManager(max_size="25MB")
            
            # Monitor performance
            process = psutil.Process()
            initial_memory = process.memory_info().rss / 1024 / 1024  # MB
            start_time = time.time()
            
            # Process video
            frames = video_processor.extract_frames(video_path, interval=0.5)
            
            # Process frames with OCR
            ocr_results = await ocr_engine.process_batch(frames)
            
            # End performance monitoring
            end_time = time.time()
            peak_memory = process.memory_info().rss / 1024 / 1024  # MB
            
            total_time = end_time - start_time
            memory_increase = peak_memory - initial_memory
            
            print(f"End-to-end performance:")
            print(f"  Total time: {total_time:.3f}s")
            print(f"  Frames processed: {len(frames)}")
            print(f"  OCR results: {len(ocr_results)}")
            print(f"  Memory increase: {memory_increase:.1f} MB")
            print(f"  Average time per frame: {total_time/len(frames):.3f}s")
            
            # Performance should be reasonable
            assert total_time < 30.0  # Complete within 30 seconds
            assert memory_increase < 200  # Less than 200MB memory increase
            assert len(ocr_results) == len(frames)
            
        finally:
            os.unlink(video_path)
    
    @pytest.mark.performance
    def test_system_resource_usage(self):
        """Test system resource usage under load"""
        process = psutil.Process()
        
        # Monitor resource usage
        initial_cpu = process.cpu_percent()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Create load by processing data
        memory_manager = MemoryManager(max_size="100MB")
        
        for i in range(100):
            img = np.zeros((500, 500, 3), dtype=np.uint8)
            memory_manager.cache.set(f"load_test_{i}", img)
        
        # Measure resource usage after load
        final_cpu = process.cpu_percent()
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        print(f"System resource usage:")
        print(f"  Initial CPU: {initial_cpu:.1f}%")
        print(f"  Final CPU: {final_cpu:.1f}%")
        print(f"  Initial memory: {initial_memory:.1f} MB")
        print(f"  Final memory: {final_memory:.1f} MB")
        print(f"  Memory increase: {final_memory - initial_memory:.1f} MB")
        
        # Resource usage should be reasonable
        assert final_memory - initial_memory < 150  # Less than 150MB increase
        assert final_cpu < 50  # CPU usage should be reasonable


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])