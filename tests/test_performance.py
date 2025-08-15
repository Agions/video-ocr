"""
Performance Testing Suite for VisionSub Application

This module provides comprehensive performance testing including:
- Load testing and stress testing
- Memory usage and leak detection
- Processing speed benchmarks
- Scalability testing
- Resource utilization analysis
"""

import pytest
import time
import psutil
import threading
import multiprocessing
import numpy as np
import cv2
from unittest.mock import Mock, patch
from pathlib import Path
import tempfile
import json
import statistics
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from memory_profiler import profile
import tracemalloc

# Add src to path
import sys
sys.path.insert(0, str(Path(__file__).parent / "src"))

from visionsub.video_utils import VideoProcessor, FrameExtractor
from visionsub.ocr_utils import OCRProcessor, OCRResult
from visionsub.subtitle_utils import SubtitleProcessor
from visionsub.core.cache_manager import CacheManager
from visionsub.models.config import AppConfig, ProcessingConfig, OcrConfig


class TestPerformanceMetrics:
    """Test suite for performance metrics collection"""
    
    @pytest.fixture
    def performance_monitor(self):
        """Create performance monitor instance"""
        return PerformanceMonitor()
    
    def test_cpu_usage_monitoring(self, performance_monitor):
        """Test CPU usage monitoring"""
        # Monitor CPU usage
        cpu_usage = performance_monitor.get_cpu_usage()
        
        assert isinstance(cpu_usage, float)
        assert 0 <= cpu_usage <= 100
    
    def test_memory_usage_monitoring(self, performance_monitor):
        """Test memory usage monitoring"""
        # Monitor memory usage
        memory_usage = performance_monitor.get_memory_usage()
        
        assert isinstance(memory_usage, dict)
        assert 'total' in memory_usage
        assert 'available' in memory_usage
        assert 'used' in memory_usage
        assert 'percent' in memory_usage
        
        assert 0 <= memory_usage['percent'] <= 100
    
    def test_disk_usage_monitoring(self, performance_monitor):
        """Test disk usage monitoring"""
        # Monitor disk usage
        disk_usage = performance_monitor.get_disk_usage()
        
        assert isinstance(disk_usage, dict)
        assert 'total' in disk_usage
        assert 'used' in disk_usage
        assert 'free' in disk_usage
        assert 'percent' in disk_usage
        
        assert 0 <= disk_usage['percent'] <= 100
    
    def test_network_monitoring(self, performance_monitor):
        """Test network monitoring"""
        # Monitor network activity
        network_stats = performance_monitor.get_network_stats()
        
        assert isinstance(network_stats, dict)
        assert 'bytes_sent' in network_stats
        assert 'bytes_recv' in network_stats
        assert 'packets_sent' in network_stats
        assert 'packets_recv' in network_stats


class TestVideoProcessingPerformance:
    """Test suite for video processing performance"""
    
    @pytest.fixture
    def video_processor(self):
        """Create video processor instance"""
        config = ProcessingConfig(
            ocr_config=OcrConfig(
                engine="PaddleOCR",
                language="中文",
                confidence_threshold=0.8
            ),
            scene_threshold=0.3,
            cache_size=100
        )
        return VideoProcessor(config)
    
    @pytest.fixture
    def sample_video_path(self):
        """Create a sample video file for testing"""
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as f:
            # Create a video with multiple frames
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(f.name, fourcc, 30.0, (640, 480))
            
            # Create 100 frames
            for i in range(100):
                frame = np.full((480, 640, 3), [i % 256, 100, 200], dtype=np.uint8)
                cv2.putText(frame, f"Frame {i}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                out.write(frame)
            
            out.release()
            yield f.name
        
        Path(f.name).unlink()
    
    def test_video_info_extraction_performance(self, video_processor, sample_video_path, benchmark):
        """Test video info extraction performance"""
        def extract_info():
            return video_processor.get_video_info(sample_video_path)
        
        result = benchmark(extract_info)
        
        assert 'fps' in result
        assert 'frame_count' in result
        assert 'duration' in result
        assert benchmark.stats.stats.mean < 1.0  # Should complete in less than 1 second
    
    def test_frame_extraction_performance(self, video_processor, sample_video_path, benchmark):
        """Test frame extraction performance"""
        def extract_frame():
            return video_processor.extract_frame(sample_video_path, 1000)
        
        frame = benchmark(extract_frame)
        
        assert frame is not None
        assert isinstance(frame, np.ndarray)
        assert benchmark.stats.stats.mean < 0.5  # Should complete in less than 0.5 seconds
    
    def test_batch_frame_extraction_performance(self, video_processor, sample_video_path, benchmark):
        """Test batch frame extraction performance"""
        timestamps = list(range(0, 10000, 1000))  # Every second
        
        def extract_frames():
            return video_processor.extract_frames_batch(sample_video_path, timestamps)
        
        frames = benchmark(extract_frames)
        
        assert len(frames) == len(timestamps)
        assert benchmark.stats.stats.mean < 2.0  # Should complete in less than 2 seconds
    
    def test_scene_detection_performance(self, video_processor, sample_video_path, benchmark):
        """Test scene detection performance"""
        def detect_scenes():
            return video_processor.detect_scenes(sample_video_path)
        
        scenes = benchmark(detect_scenes)
        
        assert isinstance(scenes, list)
        assert benchmark.stats.stats.mean < 3.0  # Should complete in less than 3 seconds
    
    def test_concurrent_video_processing(self, video_processor, sample_video_path):
        """Test concurrent video processing"""
        num_threads = 4
        timestamps = list(range(0, 5000, 500))
        
        def process_frames(thread_id):
            return video_processor.extract_frames_batch(sample_video_path, timestamps)
        
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(process_frames, i) for i in range(num_threads)]
            results = [future.result() for future in futures]
        
        assert len(results) == num_threads
        for result in results:
            assert len(result) == len(timestamps)
    
    def test_memory_usage_during_processing(self, video_processor, sample_video_path):
        """Test memory usage during video processing"""
        tracemalloc.start()
        
        # Get initial memory usage
        snapshot1 = tracemalloc.take_snapshot()
        
        # Process video
        video_processor.get_video_info(sample_video_path)
        frames = video_processor.extract_frames_batch(sample_video_path, list(range(0, 5000, 1000)))
        
        # Get final memory usage
        snapshot2 = tracemalloc.take_snapshot()
        
        # Calculate memory difference
        stats = snapshot2.compare_to(snapshot1, 'lineno')
        
        # Check memory usage is reasonable
        total_memory = sum(stat.size for stat in stats)
        assert total_memory < 100 * 1024 * 1024  # Less than 100MB
        
        tracemalloc.stop()


class TestOCRProcessingPerformance:
    """Test suite for OCR processing performance"""
    
    @pytest.fixture
    def ocr_processor(self):
        """Create OCR processor instance"""
        config = OcrConfig(
            engine="PaddleOCR",
            language="中文",
            confidence_threshold=0.8
        )
        return OCRProcessor(config)
    
    @pytest.fixture
    def sample_images(self):
        """Create sample images for testing"""
        images = []
        for i in range(10):
            image = np.full((200, 400, 3), 255, dtype=np.uint8)
            cv2.putText(image, f"Test text {i}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
            cv2.putText(image, f"测试文本 {i}", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
            images.append(image)
        return images
    
    def test_single_image_ocr_performance(self, ocr_processor, sample_images, benchmark):
        """Test single image OCR performance"""
        def process_image():
            return ocr_processor.process_image(sample_images[0])
        
        results = benchmark(process_image)
        
        assert isinstance(results, list)
        assert benchmark.stats.stats.mean < 2.0  # Should complete in less than 2 seconds
    
    def test_batch_ocr_performance(self, ocr_processor, sample_images, benchmark):
        """Test batch OCR performance"""
        def process_batch():
            return ocr_processor.process_batch(sample_images)
        
        results = benchmark(process_batch)
        
        assert len(results) == len(sample_images)
        assert benchmark.stats.stats.mean < 10.0  # Should complete in less than 10 seconds
    
    def test_concurrent_ocr_processing(self, ocr_processor, sample_images):
        """Test concurrent OCR processing"""
        num_threads = 4
        
        def process_images(thread_id):
            start_idx = thread_id * len(sample_images) // num_threads
            end_idx = (thread_id + 1) * len(sample_images) // num_threads
            return ocr_processor.process_batch(sample_images[start_idx:end_idx])
        
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(process_images, i) for i in range(num_threads)]
            results = [future.result() for future in futures]
        
        assert len(results) == num_threads
        total_results = sum(len(result) for result in results)
        assert total_results == len(sample_images)
    
    def test_ocr_memory_usage(self, ocr_processor, sample_images):
        """Test OCR memory usage"""
        tracemalloc.start()
        
        # Get initial memory usage
        snapshot1 = tracemalloc.take_snapshot()
        
        # Process images
        results = ocr_processor.process_batch(sample_images)
        
        # Get final memory usage
        snapshot2 = tracemalloc.take_snapshot()
        
        # Calculate memory difference
        stats = snapshot2.compare_to(snapshot1, 'lineno')
        
        # Check memory usage is reasonable
        total_memory = sum(stat.size for stat in stats)
        assert total_memory < 200 * 1024 * 1024  # Less than 200MB
        
        tracemalloc.stop()
    
    def test_ocr_engine_comparison(self, sample_images):
        """Test different OCR engines performance"""
        engines = ["PaddleOCR", "TesseractOCR", "EasyOCR"]
        performance_results = {}
        
        for engine in engines:
            config = OcrConfig(engine=engine, language="中文", confidence_threshold=0.8)
            ocr_processor = OCRProcessor(config)
            
            start_time = time.time()
            results = ocr_processor.process_batch(sample_images)
            end_time = time.time()
            
            performance_results[engine] = {
                'time': end_time - start_time,
                'results': len(results)
            }
        
        # Compare performance
        for engine, results in performance_results.items():
            assert results['time'] > 0
            assert results['results'] == len(sample_images)


class TestCachePerformance:
    """Test suite for cache performance"""
    
    @pytest.fixture
    def cache_manager(self):
        """Create cache manager instance"""
        with tempfile.TemporaryDirectory() as temp_dir:
            return CacheManager(cache_dir=temp_dir, max_size=1000)
    
    def test_cache_write_performance(self, cache_manager, benchmark):
        """Test cache write performance"""
        test_data = {"test": "data", "number": 123, "array": [1, 2, 3, 4, 5]}
        
        def cache_write():
            return cache_manager.store(f"key_{time.time()}", test_data)
        
        benchmark(cache_write)
        
        assert benchmark.stats.stats.mean < 0.01  # Should complete in less than 0.01 seconds
    
    def test_cache_read_performance(self, cache_manager, benchmark):
        """Test cache read performance"""
        # Pre-populate cache
        test_data = {"test": "data", "number": 123, "array": [1, 2, 3, 4, 5]}
        cache_manager.store("benchmark_key", test_data)
        
        def cache_read():
            return cache_manager.retrieve("benchmark_key")
        
        result = benchmark(cache_read)
        
        assert result == test_data
        assert benchmark.stats.stats.mean < 0.01  # Should complete in less than 0.01 seconds
    
    def test_cache_concurrent_access(self, cache_manager):
        """Test cache concurrent access"""
        num_threads = 10
        num_operations = 100
        
        def cache_operations(thread_id):
            for i in range(num_operations):
                key = f"thread_{thread_id}_key_{i}"
                data = {"thread": thread_id, "operation": i, "data": "test"}
                cache_manager.store(key, data)
                result = cache_manager.retrieve(key)
                assert result == data
        
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(cache_operations, i) for i in range(num_threads)]
            results = [future.result() for future in futures]
        
        assert len(results) == num_threads
        assert len(cache_manager.cache) == num_threads * num_operations
    
    def test_cache_memory_usage(self, cache_manager):
        """Test cache memory usage"""
        tracemalloc.start()
        
        # Get initial memory usage
        snapshot1 = tracemalloc.take_snapshot()
        
        # Populate cache
        for i in range(1000):
            data = {"key": i, "data": "x" * 1000}  # 1KB of data
            cache_manager.store(f"key_{i}", data)
        
        # Get final memory usage
        snapshot2 = tracemalloc.take_snapshot()
        
        # Calculate memory difference
        stats = snapshot2.compare_to(snapshot1, 'lineno')
        
        # Check memory usage is reasonable
        total_memory = sum(stat.size for stat in stats)
        assert total_memory < 10 * 1024 * 1024  # Less than 10MB for cache
        
        tracemalloc.stop()


class TestLoadTesting:
    """Test suite for load testing"""
    
    @pytest.fixture
    def load_tester(self):
        """Create load tester instance"""
        return LoadTester()
    
    def test_simultaneous_video_processing(self, load_tester):
        """Test simultaneous video processing"""
        num_processes = 4
        videos_per_process = 2
        
        def process_videos(process_id):
            results = []
            for i in range(videos_per_process):
                video_path = load_tester.create_sample_video(f"video_{process_id}_{i}.mp4")
                processor = VideoProcessor()
                
                start_time = time.time()
                info = processor.get_video_info(video_path)
                end_time = time.time()
                
                results.append({
                    'video': f"video_{process_id}_{i}",
                    'processing_time': end_time - start_time,
                    'success': info is not None
                })
                
                Path(video_path).unlink()
            
            return results
        
        with ProcessPoolExecutor(max_workers=num_processes) as executor:
            futures = [executor.submit(process_videos, i) for i in range(num_processes)]
            all_results = [future.result() for future in futures]
        
        # Analyze results
        total_videos = sum(len(results) for results in all_results)
        successful_videos = sum(len([r for r in results if r['success']]) for results in all_results)
        
        assert total_videos == num_processes * videos_per_process
        assert successful_videos == total_videos
        
        # Check processing times are reasonable
        all_times = [r['processing_time'] for results in all_results for r in results]
        avg_time = statistics.mean(all_times)
        assert avg_time < 5.0  # Average processing time less than 5 seconds
    
    def test_high_frequency_operations(self, load_tester):
        """Test high frequency operations"""
        num_operations = 1000
        operation_times = []
        
        def high_frequency_operation():
            cache_manager = CacheManager()
            data = {"timestamp": time.time(), "data": "test"}
            
            start_time = time.time()
            cache_manager.store(f"key_{time.time()}", data)
            result = cache_manager.retrieve(f"key_{time.time()}")
            end_time = time.time()
            
            return end_time - start_time
        
        for _ in range(num_operations):
            operation_time = high_frequency_operation()
            operation_times.append(operation_time)
        
        # Analyze performance
        avg_time = statistics.mean(operation_times)
        max_time = max(operation_times)
        min_time = min(operation_times)
        
        assert avg_time < 0.1  # Average operation time less than 0.1 seconds
        assert max_time < 1.0  # Maximum operation time less than 1 second
    
    def test_memory_leak_detection(self, load_tester):
        """Test memory leak detection"""
        psutil.Process().memory_info()
        
        initial_memory = psutil.Process().memory_info().rss
        
        # Run memory-intensive operations
        for i in range(100):
            video_path = load_tester.create_sample_video(f"leak_test_{i}.mp4")
            processor = VideoProcessor()
            
            # Process video
            info = processor.get_video_info(video_path)
            frames = processor.extract_frames_batch(video_path, [0, 1000, 2000])
            
            Path(video_path).unlink()
        
        final_memory = psutil.Process().memory_info().rss
        memory_increase = final_memory - initial_memory
        
        # Check for memory leaks
        assert memory_increase < 50 * 1024 * 1024  # Less than 50MB increase


class TestScalabilityTesting:
    """Test suite for scalability testing"""
    
    def test_linear_scaling(self):
        """Test linear scaling performance"""
        dataset_sizes = [10, 50, 100, 500, 1000]
        processing_times = []
        
        for size in dataset_sizes:
            # Create test dataset
            images = []
            for i in range(size):
                image = np.full((100, 200, 3), 255, dtype=np.uint8)
                cv2.putText(image, f"Test {i}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
                images.append(image)
            
            # Process dataset
            config = OcrConfig(engine="PaddleOCR", language="中文", confidence_threshold=0.8)
            ocr_processor = OCRProcessor(config)
            
            start_time = time.time()
            results = ocr_processor.process_batch(images)
            end_time = time.time()
            
            processing_times.append(end_time - start_time)
        
        # Check linear scaling (time should scale roughly linearly with size)
        for i in range(1, len(processing_times)):
            expected_ratio = dataset_sizes[i] / dataset_sizes[i-1]
            actual_ratio = processing_times[i] / processing_times[i-1]
            
            # Allow for some overhead
            assert actual_ratio <= expected_ratio * 1.5
    
    def test_concurrent_scaling(self):
        """Test concurrent scaling"""
        thread_counts = [1, 2, 4, 8, 16]
        processing_times = []
        
        for num_threads in thread_counts:
            # Create test dataset
            images = []
            for i in range(100):
                image = np.full((100, 200, 3), 255, dtype=np.uint8)
                cv2.putText(image, f"Test {i}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
                images.append(image)
            
            # Process with different thread counts
            config = OcrConfig(engine="PaddleOCR", language="中文", confidence_threshold=0.8)
            ocr_processor = OCRProcessor(config)
            
            start_time = time.time()
            
            with ThreadPoolExecutor(max_workers=num_threads) as executor:
                chunk_size = len(images) // num_threads
                chunks = [images[i:i + chunk_size] for i in range(0, len(images), chunk_size)]
                
                futures = [executor.submit(ocr_processor.process_batch, chunk) for chunk in chunks]
                results = [future.result() for future in futures]
            
            end_time = time.time()
            processing_times.append(end_time - start_time)
        
        # Check that more threads improve performance (up to a point)
        assert processing_times[1] < processing_times[0]  # 2 threads faster than 1
        assert processing_times[2] < processing_times[1]  # 4 threads faster than 2


class TestResourceUtilization:
    """Test suite for resource utilization"""
    
    def test_cpu_utilization(self):
        """Test CPU utilization during processing"""
        process = psutil.Process()
        
        # Create test data
        images = []
        for i in range(100):
            image = np.full((200, 400, 3), 255, dtype=np.uint8)
            cv2.putText(image, f"Test {i}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
            images.append(image)
        
        # Monitor CPU usage
        cpu_usage = []
        
        def monitor_cpu():
            while running:
                cpu_usage.append(process.cpu_percent())
                time.sleep(0.1)
        
        running = True
        monitor_thread = threading.Thread(target=monitor_cpu)
        monitor_thread.start()
        
        # Process images
        config = OcrConfig(engine="PaddleOCR", language="中文", confidence_threshold=0.8)
        ocr_processor = OCRProcessor(config)
        results = ocr_processor.process_batch(images)
        
        running = False
        monitor_thread.join()
        
        # Analyze CPU usage
        avg_cpu = statistics.mean(cpu_usage)
        max_cpu = max(cpu_usage)
        
        assert avg_cpu < 80.0  # Average CPU usage less than 80%
        assert max_cpu < 95.0  # Maximum CPU usage less than 95%
    
    def test_memory_utilization(self):
        """Test memory utilization during processing"""
        process = psutil.Process()
        
        # Create test data
        images = []
        for i in range(100):
            image = np.full((200, 400, 3), 255, dtype=np.uint8)
            cv2.putText(image, f"Test {i}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
            images.append(image)
        
        # Monitor memory usage
        memory_usage = []
        
        def monitor_memory():
            while running:
                memory_usage.append(process.memory_info().rss)
                time.sleep(0.1)
        
        running = True
        monitor_thread = threading.Thread(target=monitor_memory)
        monitor_thread.start()
        
        # Process images
        config = OcrConfig(engine="PaddleOCR", language="中文", confidence_threshold=0.8)
        ocr_processor = OCRProcessor(config)
        results = ocr_processor.process_batch(images)
        
        running = False
        monitor_thread.join()
        
        # Analyze memory usage
        max_memory = max(memory_usage)
        memory_increase = max_memory - memory_usage[0]
        
        assert memory_increase < 500 * 1024 * 1024  # Memory increase less than 500MB


class PerformanceMonitor:
    """Performance monitoring utility class"""
    
    def __init__(self):
        self.process = psutil.Process()
    
    def get_cpu_usage(self):
        """Get current CPU usage"""
        return self.process.cpu_percent()
    
    def get_memory_usage(self):
        """Get current memory usage"""
        memory = self.process.memory_info()
        return {
            'total': psutil.virtual_memory().total,
            'available': psutil.virtual_memory().available,
            'used': memory.rss,
            'percent': self.process.memory_percent()
        }
    
    def get_disk_usage(self):
        """Get disk usage"""
        disk = psutil.disk_usage('/')
        return {
            'total': disk.total,
            'used': disk.used,
            'free': disk.free,
            'percent': disk.percent
        }
    
    def get_network_stats(self):
        """Get network statistics"""
        net = psutil.net_io_counters()
        return {
            'bytes_sent': net.bytes_sent,
            'bytes_recv': net.bytes_recv,
            'packets_sent': net.packets_sent,
            'packets_recv': net.packets_recv
        }


class LoadTester:
    """Load testing utility class"""
    
    def create_sample_video(self, filename):
        """Create a sample video for testing"""
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as f:
            video_path = f.name
            
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(video_path, fourcc, 30.0, (640, 480))
        
        # Create 30 frames
        for i in range(30):
            frame = np.full((480, 640, 3), [i % 256, 100, 200], dtype=np.uint8)
            cv2.putText(frame, f"Frame {i}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            out.write(frame)
        
        out.release()
        return video_path


if __name__ == "__main__":
    pytest.main([__file__, "-v"])