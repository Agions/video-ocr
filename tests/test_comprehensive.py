#!/usr/bin/env python3
"""
Comprehensive test for VisionSub microservices with actual API testing
"""
import sys
import os
import asyncio
import logging
import json
import time
from pathlib import Path
import aiohttp
import aiohttp.web

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

try:
    from visionsub.microservices import OCRService, UIService, ServiceConfig, ServiceOrchestrator
    from visionsub.core.config_manager import ConfigManager
    
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    async def start_services():
        """Start the microservices"""
        logger.info("Starting VisionSub microservices...")
        
        orchestrator = ServiceOrchestrator()
        
        # Create OCR service
        ocr_config = ServiceConfig(
            name="ocr_service",
            host="localhost",
            port=8081,
            version="1.0.0",
            description="OCR Processing Service"
        )
        ocr_service = OCRService(ocr_config)
        orchestrator.add_service(ocr_service)
        
        # Create UI service
        ui_config = ServiceConfig(
            name="ui_service",
            host="localhost",
            port=8080,
            version="1.0.0",
            description="UI Service"
        )
        ui_service = UIService(ui_config)
        orchestrator.add_service(ui_service)
        
        # Configure inter-service communication
        ui_service.set_ocr_service("http://localhost:8081")
        await ui_service.setup_session()
        
        # Setup OCR workers
        await ocr_service.setup_ocr_engine({})
        await ocr_service.start_workers(2)
        
        # Start all services
        await orchestrator.start_all_services()
        
        logger.info("Services started successfully")
        logger.info("OCR Service: http://localhost:8081")
        logger.info("UI Service: http://localhost:8080")
        
        return orchestrator, ocr_service, ui_service
    
    async def test_health_checks():
        """Test health check endpoints"""
        logger.info("Testing health check endpoints...")
        
        async with aiohttp.ClientSession() as session:
            # Test OCR service health
            try:
                async with session.get("http://localhost:8081/health") as response:
                    if response.status == 200:
                        health = await response.json()
                        logger.info(f"OCR Service Health: {health}")
                    else:
                        logger.error(f"OCR Service Health Check Failed: {response.status}")
            except Exception as e:
                logger.error(f"OCR Service Health Check Error: {e}")
            
            # Test UI service health
            try:
                async with session.get("http://localhost:8080/health") as response:
                    if response.status == 200:
                        health = await response.json()
                        logger.info(f"UI Service Health: {health}")
                    else:
                        logger.error(f"UI Service Health Check Failed: {response.status}")
            except Exception as e:
                logger.error(f"UI Service Health Check Error: {e}")
    
    async def test_ocr_processing():
        """Test OCR processing functionality"""
        logger.info("Testing OCR processing...")
        
        # Create a simple test OCR request
        test_request = {
            "image_id": "test_image_001",
            "image_data": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg==", # 1x1 pixel PNG
            "config": {"language": "zh"}
        }
        
        async with aiohttp.ClientSession() as session:
            try:
                # Submit OCR request
                async with session.post(
                    "http://localhost:8081/ocr/process",
                    json=test_request
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        logger.info(f"OCR Request Submitted: {result}")
                        
                        # Check job status
                        job_id = result["job_id"]
                        await asyncio.sleep(2)  # Wait for processing
                        
                        async with session.get(f"http://localhost:8081/ocr/status/{job_id}") as status_response:
                            if status_response.status == 200:
                                status = await status_response.json()
                                logger.info(f"OCR Job Status: {status}")
                            else:
                                logger.error(f"OCR Job Status Check Failed: {status_response.status}")
                    else:
                        logger.error(f"OCR Request Failed: {response.status}")
            except Exception as e:
                logger.error(f"OCR Processing Test Error: {e}")
    
    async def test_ui_api():
        """Test UI service API endpoints"""
        logger.info("Testing UI service API...")
        
        async with aiohttp.ClientSession() as session:
            try:
                # Test API status
                async with session.get("http://localhost:8080/api/status") as response:
                    if response.status == 200:
                        status = await response.json()
                        logger.info(f"UI API Status: {status}")
                    else:
                        logger.error(f"UI API Status Check Failed: {response.status}")
                
                # Test main page
                async with session.get("http://localhost:8080/") as response:
                    if response.status == 200:
                        logger.info("UI Main Page accessible")
                    else:
                        logger.error(f"UI Main Page Failed: {response.status}")
            except Exception as e:
                logger.error(f"UI API Test Error: {e}")
    
    async def test_configuration_hot_reload():
        """Test configuration hot reload functionality"""
        logger.info("Testing configuration hot reload...")
        
        try:
            # Test configuration manager
            config_manager = ConfigManager()
            config = config_manager.get_config()
            
            logger.info(f"Current environment: {config.environment}")
            logger.info(f"Log level: {config.log_level}")
            logger.info(f"Hot reload enabled: {config.hot_reload.enabled}")
            
            # Test hot reload by starting it briefly
            await config_manager.start_hot_reload()
            await asyncio.sleep(1)
            await config_manager.stop_hot_reload()
            
            logger.info("Configuration hot reload test completed")
            
        except Exception as e:
            logger.error(f"Configuration Hot Reload Test Error: {e}")
    
    async def comprehensive_test():
        """Run comprehensive tests"""
        logger.info("Starting comprehensive microservices test...")
        
        # Start services
        orchestrator, ocr_service, ui_service = await start_services()
        
        # Wait for services to fully start
        await asyncio.sleep(2)
        
        # Run tests
        await test_health_checks()
        await test_ocr_processing()
        await test_ui_api()
        await test_configuration_hot_reload()
        
        logger.info("Comprehensive test completed!")
        
        # Stop services
        await orchestrator.stop_all_services()
        logger.info("Services stopped")
        
        return True
    
    if __name__ == "__main__":
        try:
            result = asyncio.run(comprehensive_test())
            sys.exit(0 if result else 1)
        except KeyboardInterrupt:
            logger.info("Test interrupted by user")
            sys.exit(1)
        except Exception as e:
            logger.error(f"Comprehensive test failed: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
        
except ImportError as e:
    logger = logging.getLogger(__name__)
    logger.error(f"Import error: {e}")
    logger.error("Please ensure all dependencies are installed")
    sys.exit(1)
except Exception as e:
    logger = logging.getLogger(__name__)
    logger.error(f"Test setup failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)