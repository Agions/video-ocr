#!/usr/bin/env python3
"""
Simple test script for VisionSub microservices
"""
import sys
import os
import asyncio
import logging
from pathlib import Path

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

try:
    from visionsub.microservices import OCRService, UIService, ServiceConfig
    from visionsub.core.config_manager import ConfigManager
    
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    async def test_services():
        """Test basic service functionality"""
        logger.info("Starting VisionSub microservices test...")
        
        # Test OCR Service
        logger.info("Creating OCR Service...")
        ocr_config = ServiceConfig(
            name="ocr_service",
            host="localhost",
            port=8081,
            version="1.0.0",
            description="OCR Processing Service"
        )
        
        ocr_service = OCRService(ocr_config)
        logger.info("OCR Service created successfully")
        
        # Test UI Service
        logger.info("Creating UI Service...")
        ui_config = ServiceConfig(
            name="ui_service",
            host="localhost",
            port=8080,
            version="1.0.0",
            description="UI Service"
        )
        
        ui_service = UIService(ui_config)
        logger.info("UI Service created successfully")
        
        # Test configuration manager
        logger.info("Testing configuration manager...")
        config_manager = ConfigManager()
        config = config_manager.get_config()
        logger.info(f"Configuration loaded: {config.environment}")
        
        # Test health checks
        logger.info("Testing health checks...")
        ocr_health = await ocr_service.health_check()
        ui_health = await ui_service.health_check()
        
        logger.info(f"OCR Service Health: {ocr_health.status}")
        logger.info(f"UI Service Health: {ui_health.status}")
        
        logger.info("All tests completed successfully!")
        return True
        
    if __name__ == "__main__":
        result = asyncio.run(test_services())
        sys.exit(0 if result else 1)
        
except ImportError as e:
    logger = logging.getLogger(__name__)
    logger.error(f"Import error: {e}")
    logger.error("Please ensure all dependencies are installed")
    sys.exit(1)
except Exception as e:
    logger = logging.getLogger(__name__)
    logger.error(f"Test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)