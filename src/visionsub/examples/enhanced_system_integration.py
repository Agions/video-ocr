"""
Integration Example: Enhanced Video OCR Processing System
This example demonstrates how to use the enhanced backend components
"""
import asyncio
import logging
import time
from pathlib import Path
from typing import List, Dict, Any

from src.visionsub.core.enhanced_engine import EnhancedProcessingEngine, ProcessingPriority
from src.visionsub.core.enhanced_video_processor import EnhancedVideoProcessor, ProcessingOptions, ProcessingMode
from src.visionsub.core.enhanced_ocr_engine import EnhancedOCREngine, OCRSecurityConfig, OCRPerformanceConfig
from src.visionsub.security.enhanced_security_manager import EnhancedSecurityManager, SecurityPolicy, SecurityLevel
from src.visionsub.services.enhanced_services import ServiceManager, OCRService, VideoProcessingService, ProcessingRequest, ProcessingResponse
from src.visionsub.models.config import ProcessingConfig
from src.visionsub.models.subtitle import SubtitleItem
from src.visionsub.utils.metrics import MetricsCollector, PerformanceMonitor
from src.visionsub.utils.audit_logger import AuditLogger
from src.visionsub.utils.rate_limiter import RateLimiter


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class EnhancedVideoOCRSystem:
    """Enhanced Video OCR Processing System Integration"""

    def __init__(self):
        self.security_manager: Optional[EnhancedSecurityManager] = None
        self.service_manager: Optional[ServiceManager] = None
        self.processing_engine: Optional[EnhancedProcessingEngine] = None
        self.metrics_collector: Optional[MetricsCollector] = None
        self.audit_logger: Optional[AuditLogger] = None
        self.rate_limiter: Optional[RateLimiter] = None
        self._initialized = False

    async def initialize(self) -> bool:
        """Initialize the enhanced system"""
        try:
            logger.info("Initializing Enhanced Video OCR System...")
            
            # Step 1: Initialize Security Manager
            await self._initialize_security()
            
            # Step 2: Initialize Metrics and Monitoring
            await self._initialize_monitoring()
            
            # Step 3: Initialize Service Manager
            await self._initialize_services()
            
            # Step 4: Initialize Processing Engine
            await self._initialize_processing_engine()
            
            self._initialized = True
            logger.info("✓ Enhanced Video OCR System initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"✗ System initialization failed: {e}")
            return False

    async def _initialize_security(self):
        """Initialize security components"""
        logger.info("Initializing security components...")
        
        # Create enhanced security policy
        security_policy = SecurityPolicy(
            level=SecurityLevel.ENHANCED,
            enable_encryption=True,
            enable_audit_logging=True,
            enable_rate_limiting=True,
            max_file_size=1024 * 1024 * 1024,  # 1GB
            session_timeout=3600,  # 1 hour
            rate_limit_requests=100,
            rate_limit_window=60
        )
        
        # Initialize security manager
        self.security_manager = EnhancedSecurityManager(security_policy)
        await self.security_manager.initialize()
        
        # Initialize audit logger
        self.audit_logger = AuditLogger()
        await self.audit_logger.start()
        
        # Initialize rate limiter
        self.rate_limiter = RateLimiter(
            max_requests=100,
            window_seconds=60,
            enable_burst=True
        )
        
        logger.info("✓ Security components initialized")

    async def _initialize_monitoring(self):
        """Initialize monitoring components"""
        logger.info("Initializing monitoring components...")
        
        # Initialize metrics collector
        self.metrics_collector = MetricsCollector()
        
        logger.info("✓ Monitoring components initialized")

    async def _initialize_services(self):
        """Initialize service layer"""
        logger.info("Initializing service layer...")
        
        # Initialize service manager
        self.service_manager = ServiceManager(self.security_manager)
        
        # Register and start services
        await self.service_manager.register_service(OCRService(self.security_manager))
        await self.service_manager.register_service(VideoProcessingService(self.security_manager))
        
        success = await self.service_manager.start_all_services()
        if not success:
            raise Exception("Failed to start services")
        
        logger.info("✓ Service layer initialized")

    async def _initialize_processing_engine(self):
        """Initialize processing engine"""
        logger.info("Initializing processing engine...")
        
        # Create processing configuration
        processing_config = ProcessingConfig()
        
        # Initialize enhanced processing engine
        self.processing_engine = EnhancedProcessingEngine(processing_config)
        
        logger.info("✓ Processing engine initialized")

    async def process_video_secure(
        self,
        video_path: str,
        user_id: str = "demo_user",
        priority: ProcessingPriority = ProcessingPriority.NORMAL
    ) -> Dict[str, Any]:
        """
        Process video with enhanced security and monitoring
        
        Args:
            video_path: Path to video file
            user_id: User identifier
            priority: Processing priority
            
        Returns:
            Dictionary containing processing results and metadata
        """
        if not self._initialized:
            raise Exception("System not initialized")
        
        start_time = time.time()
        logger.info(f"Processing video: {video_path}")
        
        try:
            # Step 1: Create security context
            permissions = {"video:process", "ocr:process", "result:access"}
            roles = {"user"}
            
            async with self.security_manager.security_context(user_id, permissions, roles) as context:
                # Step 2: Rate limiting check
                if not await self.rate_limiter.is_allowed(user_id):
                    raise Exception("Rate limit exceeded")
                
                # Step 3: Process video through service manager
                processing_request = ProcessingRequest(
                    id=f"req_{int(time.time() * 1000)}",
                    video_path=video_path,
                    config=ProcessingConfig(),
                    priority=priority,
                    user_id=user_id,
                    session_id=context.session_id
                )
                
                # Step 4: Process through video processing service
                response = await self.service_manager.process_request(
                    "video_processing_service", processing_request
                )
                
                # Step 5: Record metrics
                processing_time = time.time() - start_time
                await self.metrics_collector.record_processing_time(processing_time)
                
                # Step 6: Prepare result
                result = {
                    "success": response.status == "success",
                    "processing_time": processing_time,
                    "subtitles_count": len(response.result) if response.result else 0,
                    "subtitles": [subtitle.dict() for subtitle in response.result] if response.result else [],
                    "error": response.error,
                    "metadata": {
                        "request_id": response.request_id,
                        "video_path": video_path,
                        "user_id": user_id,
                        "priority": priority.value,
                        "processing_metadata": response.metadata
                    }
                }
                
                if response.status == "success":
                    logger.info(f"✓ Video processed successfully: {len(response.result)} subtitles extracted")
                else:
                    logger.error(f"✗ Video processing failed: {response.error}")
                
                return result
                
        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"✗ Video processing failed: {e}")
            
            return {
                "success": False,
                "processing_time": processing_time,
                "error": str(e),
                "subtitles_count": 0,
                "subtitles": []
            }

    async def process_image_ocr_secure(
        self,
        image_data: bytes,
        image_shape: tuple,
        user_id: str = "demo_user"
    ) -> Dict[str, Any]:
        """
        Process image OCR with enhanced security
        
        Args:
            image_data: Raw image data as bytes
            image_shape: Image shape tuple (height, width, channels)
            user_id: User identifier
            
        Returns:
            Dictionary containing OCR results and metadata
        """
        if not self._initialized:
            raise Exception("System not initialized")
        
        start_time = time.time()
        logger.info("Processing image OCR...")
        
        try:
            # Step 1: Create security context
            permissions = {"ocr:process", "result:access"}
            roles = {"user"}
            
            async with self.security_manager.security_context(user_id, permissions, roles) as context:
                # Step 2: Rate limiting check
                if not await self.rate_limiter.is_allowed(user_id):
                    raise Exception("Rate limit exceeded")
                
                # Step 3: Create OCR request
                ocr_request = ProcessingRequest(
                    id=f"ocr_{int(time.time() * 1000)}",
                    video_path="image_data",  # Placeholder
                    config=ProcessingConfig(),
                    user_id=user_id,
                    session_id=context.session_id,
                    metadata={
                        "image_data": image_data,
                        "image_shape": image_shape,
                        "timestamp": time.time()
                    }
                )
                
                # Step 4: Process through OCR service
                response = await self.service_manager.process_request(
                    "ocr_service", ocr_request
                )
                
                # Step 5: Record metrics
                processing_time = time.time() - start_time
                await self.metrics_collector.record_ocr_processing_time(processing_time)
                
                # Step 6: Prepare result
                result = {
                    "success": response.status == "success",
                    "processing_time": processing_time,
                    "text_count": len(response.result) if response.result else 0,
                    "texts": [subtitle.text for subtitle in response.result] if response.result else [],
                    "confidences": [subtitle.confidence for subtitle in response.result] if response.result else [],
                    "error": response.error,
                    "metadata": {
                        "request_id": response.request_id,
                        "user_id": user_id,
                        "ocr_metadata": response.metadata
                    }
                }
                
                if response.status == "success":
                    logger.info(f"✓ Image OCR processed successfully: {len(response.result)} text items found")
                else:
                    logger.error(f"✗ Image OCR processing failed: {response.error}")
                
                return result
                
        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"✗ Image OCR processing failed: {e}")
            
            return {
                "success": False,
                "processing_time": processing_time,
                "error": str(e),
                "text_count": 0,
                "texts": [],
                "confidences": []
            }

    async def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        if not self._initialized:
            return {"error": "System not initialized"}
        
        try:
            # Get service status
            services_status = await self.service_manager.get_all_services_status()
            
            # Get security status
            security_status = await self.security_manager.get_security_status()
            
            # Get metrics
            metrics = await self.metrics_collector.get_metrics()
            
            # Get rate limiter stats
            rate_limiter_stats = await self.rate_limiter.get_global_stats()
            
            # Get audit logs stats
            audit_stats = await self.audit_logger.get_statistics()
            
            return {
                "system_initialized": True,
                "services": services_status,
                "security": security_status,
                "metrics": metrics,
                "rate_limiter": rate_limiter_stats,
                "audit_logs": audit_stats,
                "timestamp": time.time()
            }
            
        except Exception as e:
            logger.error(f"Failed to get system status: {e}")
            return {"error": str(e)}

    async def cleanup(self):
        """Cleanup system resources"""
        try:
            logger.info("Cleaning up Enhanced Video OCR System...")
            
            # Stop services
            if self.service_manager:
                await self.service_manager.stop_all_services()
            
            # Cleanup security manager
            if self.security_manager:
                await self.security_manager.cleanup()
            
            # Cleanup audit logger
            if self.audit_logger:
                await self.audit_logger.cleanup()
            
            # Cleanup processing engine
            if self.processing_engine:
                await self.processing_engine.cleanup()
            
            logger.info("✓ System cleanup completed")
            
        except Exception as e:
            logger.error(f"✗ System cleanup failed: {e}")


async def demo_enhanced_system():
    """Demonstrate the enhanced system capabilities"""
    logger.info("🚀 Starting Enhanced Video OCR System Demo")
    
    # Initialize system
    system = EnhancedVideoOCRSystem()
    if not await system.initialize():
        logger.error("❌ Failed to initialize system")
        return
    
    try:
        # Display system status
        logger.info("📊 Getting system status...")
        status = await system.get_system_status()
        logger.info(f"System Status: {status['system_initialized']}")
        logger.info(f"Services: {list(status['services'].keys())}")
        logger.info(f"Security Level: {status['security']['policy_level']}")
        
        # Example 1: Process video (placeholder - would need actual video file)
        logger.info("🎥 Example 1: Video Processing")
        video_result = await system.process_video_secure(
            video_path="example_video.mp4",  # Replace with actual video path
            user_id="demo_user_1"
        )
        logger.info(f"Video processing result: {video_result['success']}")
        
        # Example 2: Process image OCR (placeholder - would need actual image data)
        logger.info("🖼️ Example 2: Image OCR Processing")
        # This would require actual image data and shape
        # image_result = await system.process_image_ocr_secure(...)
        logger.info("Image OCR processing would be demonstrated here")
        
        # Example 3: Show monitoring capabilities
        logger.info("📈 Example 3: System Monitoring")
        final_status = await system.get_system_status()
        logger.info(f"Total requests processed: {final_status['rate_limiter']['total_requests']}")
        logger.info(f"Active services: {len([s for s in final_status['services'].values() if s['status'] == 'running'])}")
        
        logger.info("✅ Enhanced Video OCR System Demo completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Demo failed: {e}")
    
    finally:
        # Cleanup
        await system.cleanup()


async def performance_test():
    """Performance testing of the enhanced system"""
    logger.info("⚡ Starting Performance Test")
    
    system = EnhancedVideoOCRSystem()
    if not await system.initialize():
        logger.error("❌ Failed to initialize system for performance test")
        return
    
    try:
        # Simulate multiple concurrent requests
        logger.info("🔄 Testing concurrent processing...")
        
        tasks = []
        for i in range(5):
            task = asyncio.create_task(
                system.process_video_secure(
                    video_path=f"test_video_{i}.mp4",
                    user_id=f"test_user_{i}"
                )
            )
            tasks.append(task)
        
        # Wait for all tasks to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Analyze results
        successful = sum(1 for r in results if isinstance(r, dict) and r.get('success', False))
        failed = len(results) - successful
        
        logger.info(f"📊 Performance Test Results:")
        logger.info(f"  Total requests: {len(results)}")
        logger.info(f"  Successful: {successful}")
        logger.info(f"  Failed: {failed}")
        
        # Get final system status
        final_status = await system.get_system_status()
        logger.info(f"  Final system metrics: {final_status['metrics']}")
        
    except Exception as e:
        logger.error(f"❌ Performance test failed: {e}")
    
    finally:
        await system.cleanup()


if __name__ == "__main__":
    # Run the demo
    asyncio.run(demo_enhanced_system())
    
    # Uncomment to run performance test
    # asyncio.run(performance_test())