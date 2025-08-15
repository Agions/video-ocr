"""
Health check module for VisionSub application
"""

import os
import sys
import psutil
import sqlite3
import redis
from typing import Dict, Any, Optional
from datetime import datetime
import json


class HealthCheck:
    """Health check utilities for VisionSub application"""
    
    @staticmethod
    def check_basic() -> Dict[str, Any]:
        """Basic health check - application is running"""
        return {
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "version": "1.0.0",
            "service": "visionsub",
            "checks": {
                "process": HealthCheck._check_process(),
                "memory": HealthCheck._check_memory(),
                "disk": HealthCheck._check_disk()
            }
        }
    
    @staticmethod
    def check_detailed() -> Dict[str, Any]:
        """Detailed health check including all services"""
        checks = HealthCheck.check_basic()
        
        # Add detailed checks
        checks["checks"].update({
            "database": HealthCheck._check_database(),
            "redis": HealthCheck._check_redis(),
            "ocr_engines": HealthCheck._check_ocr_engines(),
            "dependencies": HealthCheck._check_dependencies()
        })
        
        # Determine overall status
        overall_status = "healthy"
        for check_name, result in checks["checks"].items():
            if result.get("status") == "unhealthy":
                overall_status = "unhealthy"
                break
            elif result.get("status") == "degraded" and overall_status == "healthy":
                overall_status = "degraded"
        
        checks["status"] = overall_status
        return checks
    
    @staticmethod
    def _check_process() -> Dict[str, Any]:
        """Check if the process is running"""
        try:
            process = psutil.Process()
            return {
                "status": "healthy",
                "pid": process.pid,
                "cpu_percent": process.cpu_percent(),
                "memory_percent": process.memory_percent(),
                "create_time": datetime.fromtimestamp(process.create_time()).isoformat()
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e)
            }
    
    @staticmethod
    def _check_memory() -> Dict[str, Any]:
        """Check memory usage"""
        try:
            memory = psutil.virtual_memory()
            swap = psutil.swap_memory()
            
            memory_status = "healthy"
            if memory.percent > 90:
                memory_status = "unhealthy"
            elif memory.percent > 75:
                memory_status = "degraded"
            
            return {
                "status": memory_status,
                "total": memory.total,
                "available": memory.available,
                "used": memory.used,
                "percent": memory.percent,
                "swap_total": swap.total,
                "swap_used": swap.used,
                "swap_percent": swap.percent
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e)
            }
    
    @staticmethod
    def _check_disk() -> Dict[str, Any]:
        """Check disk usage"""
        try:
            disk = psutil.disk_usage('/')
            
            disk_status = "healthy"
            if disk.percent > 90:
                disk_status = "unhealthy"
            elif disk.percent > 80:
                disk_status = "degraded"
            
            return {
                "status": disk_status,
                "total": disk.total,
                "used": disk.used,
                "free": disk.free,
                "percent": disk.percent
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e)
            }
    
    @staticmethod
    def _check_database() -> Dict[str, Any]:
        """Check database connectivity"""
        try:
            # Try SQLite first
            db_path = os.environ.get('DATABASE_URL', '').replace('sqlite:///', '')
            if db_path and os.path.exists(db_path):
                conn = sqlite3.connect(db_path)
                cursor = conn.cursor()
                cursor.execute("SELECT 1")
                cursor.fetchone()
                conn.close()
                
                # Get database size
                db_size = os.path.getsize(db_path)
                
                return {
                    "status": "healthy",
                    "type": "sqlite",
                    "path": db_path,
                    "size": db_size
                }
            
            # Try PostgreSQL
            import psycopg2
            db_url = os.environ.get('DATABASE_URL', '')
            if db_url.startswith('postgresql://'):
                conn = psycopg2.connect(db_url)
                cursor = conn.cursor()
                cursor.execute("SELECT 1")
                cursor.fetchone()
                conn.close()
                
                return {
                    "status": "healthy",
                    "type": "postgresql",
                    "url": db_url.split('@')[-1] if '@' in db_url else db_url
                }
            
            return {
                "status": "degraded",
                "message": "No database configured"
            }
            
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e)
            }
    
    @staticmethod
    def _check_redis() -> Dict[str, Any]:
        """Check Redis connectivity"""
        try:
            redis_url = os.environ.get('REDIS_URL', 'redis://localhost:6379/0')
            r = redis.from_url(redis_url)
            
            # Test Redis connection
            r.ping()
            
            # Get Redis info
            info = r.info()
            
            return {
                "status": "healthy",
                "url": redis_url,
                "version": info.get('redis_version'),
                "connected_clients": info.get('connected_clients'),
                "used_memory": info.get('used_memory'),
                "used_memory_human": info.get('used_memory_human')
            }
            
        except Exception as e:
            return {
                "status": "degraded",
                "message": f"Redis not available: {str(e)}"
            }
    
    @staticmethod
    def _check_ocr_engines() -> Dict[str, Any]:
        """Check OCR engine availability"""
        results = {}
        
        # Check PaddleOCR
        try:
            import paddleocr
            results["paddle"] = {
                "status": "healthy",
                "version": getattr(paddleocr, '__version__', 'unknown')
            }
        except ImportError:
            results["paddle"] = {
                "status": "unhealthy",
                "message": "PaddleOCR not installed"
            }
        except Exception as e:
            results["paddle"] = {
                "status": "unhealthy",
                "error": str(e)
            }
        
        # Check Tesseract
        try:
            import pytesseract
            results["tesseract"] = {
                "status": "healthy",
                "version": pytesseract.get_tesseract_version()
            }
        except ImportError:
            results["tesseract"] = {
                "status": "unhealthy",
                "message": "pytesseract not installed"
            }
        except Exception as e:
            results["tesseract"] = {
                "status": "unhealthy",
                "error": str(e)
            }
        
        # Determine overall status
        overall_status = "healthy"
        for engine, result in results.items():
            if result.get("status") == "unhealthy":
                overall_status = "degraded"
        
        return {
            "status": overall_status,
            "engines": results
        }
    
    @staticmethod
    def _check_dependencies() -> Dict[str, Any]:
        """Check critical dependencies"""
        dependencies = {
            "opencv": False,
            "numpy": False,
            "pillow": False,
            "pyqt6": False,
            "sqlalchemy": False
        }
        
        try:
            import cv2
            dependencies["opencv"] = True
        except ImportError:
            pass
        
        try:
            import numpy
            dependencies["numpy"] = True
        except ImportError:
            pass
        
        try:
            from PIL import Image
            dependencies["pillow"] = True
        except ImportError:
            pass
        
        try:
            from PyQt6 import QtCore
            dependencies["pyqt6"] = True
        except ImportError:
            pass
        
        try:
            import sqlalchemy
            dependencies["sqlalchemy"] = True
        except ImportError:
            pass
        
        # Determine status
        missing_deps = [dep for dep, available in dependencies.items() if not available]
        if missing_deps:
            return {
                "status": "unhealthy",
                "message": f"Missing dependencies: {', '.join(missing_deps)}",
                "dependencies": dependencies
            }
        else:
            return {
                "status": "healthy",
                "dependencies": dependencies
            }


def main():
    """Command line interface for health checks"""
    import argparse
    
    parser = argparse.ArgumentParser(description='VisionSub Health Check')
    parser.add_argument('--format', choices=['json', 'text'], default='text',
                       help='Output format')
    parser.add_argument('--detailed', action='store_true',
                       help='Run detailed health check')
    
    args = parser.parse_args()
    
    try:
        if args.detailed:
            result = HealthCheck.check_detailed()
        else:
            result = HealthCheck.check_basic()
        
        if args.format == 'json':
            print(json.dumps(result, indent=2))
        else:
            print(f"VisionSub Health Status: {result['status'].upper()}")
            print(f"Timestamp: {result['timestamp']}")
            print(f"Version: {result['version']}")
            print("\nComponent Status:")
            
            for component, check in result['checks'].items():
                status = check.get('status', 'unknown')
                status_symbol = "✓" if status == "healthy" else "⚠" if status == "degraded" else "✗"
                print(f"  {status_symbol} {component}: {status.upper()}")
                
                if 'error' in check:
                    print(f"    Error: {check['error']}")
                elif 'message' in check:
                    print(f"    Message: {check['message']}")
            
            # Exit with appropriate code
            if result['status'] == 'unhealthy':
                sys.exit(1)
            elif result['status'] == 'degraded':
                sys.exit(2)
            else:
                sys.exit(0)
                
    except Exception as e:
        print(f"Health check failed: {str(e)}", file=sys.stderr)
        sys.exit(3)


if __name__ == "__main__":
    main()