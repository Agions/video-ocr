"""
安全输入验证模块
提供全面的输入验证和安全清理功能
"""

import os
import re
import mimetypes
from typing import Any, Dict, List, Optional, Union
from pathlib import Path
import magic
import logging

logger = logging.getLogger(__name__)

class InputValidator:
    """输入验证器类"""
    
    # 危险字符模式
    DANGEROUS_PATTERNS = [
        r'<script[^>]*>.*?</script>',
        r'javascript:',
        r'vbscript:',
        r'onload\s*=',
        r'onerror\s*=',
        r'eval\s*\(',
        r'document\.',
        r'window\.',
        r'alert\s*\(',
        r'exec\s*\(',
        r'system\s*\(',
        r'__import__\s*\(',
        r'subprocess\.',
        r'os\.',
    ]
    
    # 允许的文件扩展名
    ALLOWED_VIDEO_EXTENSIONS = {
        '.mp4', '.avi', '.mkv', '.mov', '.wmv', '.flv', '.webm', '.m4v', '.3gp'
    }
    
    # 允许的MIME类型
    ALLOWED_VIDEO_MIMES = {
        'video/mp4', 'video/avi', 'video/x-matroska', 'video/quicktime',
        'video/x-ms-wmv', 'video/x-flv', 'video/webm'
    }
    
    def __init__(self, max_file_size_mb: int = 500):
        self.max_file_size_bytes = max_file_size_mb * 1024 * 1024
    
    def validate_file_path(self, file_path: str) -> bool:
        """验证文件路径安全性"""
        try:
            path = Path(file_path)
            
            # 检查路径遍历攻击
            if '..' in str(path) or str(path).startswith('/'):
                logger.warning(f"检测到路径遍历攻击尝试: {file_path}")
                return False
            
            # 检查文件是否存在
            if not path.exists():
                logger.warning(f"文件不存在: {file_path}")
                return False
            
            # 检查是否为文件
            if not path.is_file():
                logger.warning(f"路径不是文件: {file_path}")
                return False
            
            # 检查文件大小
            file_size = path.stat().st_size
            if file_size > self.max_file_size_bytes:
                logger.warning(f"文件大小超过限制: {file_size} bytes")
                return False
            
            # 检查文件扩展名
            if path.suffix.lower() not in self.ALLOWED_VIDEO_EXTENSIONS:
                logger.warning(f"不允许的文件扩展名: {path.suffix}")
                return False
            
            # 检查文件MIME类型
            mime_type = self._get_file_mime_type(file_path)
            if mime_type not in self.ALLOWED_VIDEO_MIMES:
                logger.warning(f"不允许的MIME类型: {mime_type}")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"文件路径验证失败: {e}")
            return False
    
    def _get_file_mime_type(self, file_path: str) -> str:
        """获取文件MIME类型"""
        try:
            # 使用python-magic库检测文件类型
            mime = magic.Magic(mime=True)
            return mime.from_file(file_path)
        except Exception as e:
            logger.warning(f"MIME类型检测失败，使用路径推断: {e}")
            # 回退到基于扩展名的MIME类型推断
            mime_type, _ = mimetypes.guess_type(file_path)
            return mime_type or 'application/octet-stream'
    
    def sanitize_user_input(self, input_string: str) -> str:
        """清理用户输入"""
        if not isinstance(input_string, str):
            return str(input_string)
        
        # 移除危险字符和模式
        sanitized = input_string
        
        for pattern in self.DANGEROUS_PATTERNS:
            sanitized = re.sub(pattern, '', sanitized, flags=re.IGNORECASE | re.DOTALL)
        
        # 移除控制字符（保留换行符和制表符）
        sanitized = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', sanitized)
        
        # 限制字符串长度
        if len(sanitized) > 1000:
            sanitized = sanitized[:1000]
        
        return sanitized.strip()
    
    def validate_language_code(self, language_code: str) -> bool:
        """验证语言代码"""
        valid_codes = {
            'ch', 'en', 'ko', 'ja', 'fr', 'de', 'es', 'ru', 'ar', 'hi',
            '中文', '英文', '韩文', '日文', '法文', '德文', '西班牙文', '俄文', '阿拉伯文', '印地文'
        }
        return language_code in valid_codes
    
    def validate_threshold(self, threshold: int) -> bool:
        """验证阈值参数"""
        return isinstance(threshold, int) and 0 <= threshold <= 255
    
    def validate_roi_rect(self, roi_rect: tuple) -> bool:
        """验证ROI矩形参数"""
        if not isinstance(roi_rect, tuple) or len(roi_rect) != 4:
            return False
        
        x, y, w, h = roi_rect
        return all(isinstance(val, int) and val >= 0 for val in (x, y, w, h))
    
    def validate_config_dict(self, config_dict: Dict[str, Any]) -> bool:
        """验证配置字典"""
        try:
            # 检查必需字段
            required_fields = ['engine', 'language', 'threshold']
            for field in required_fields:
                if field not in config_dict:
                    logger.warning(f"配置缺少必需字段: {field}")
                    return False
            
            # 验证字段值
            if not self.validate_language_code(config_dict['language']):
                logger.warning("无效的语言代码")
                return False
            
            if not self.validate_threshold(config_dict['threshold']):
                logger.warning("无效的阈值")
                return False
            
            if config_dict['engine'] not in ['PaddleOCR', 'Tesseract']:
                logger.warning("无效的OCR引擎")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"配置验证失败: {e}")
            return False
    
    def is_safe_filename(self, filename: str) -> bool:
        """检查文件名是否安全"""
        if not filename:
            return False
        
        # 检查危险字符
        dangerous_chars = ['<', '>', ':', '"', '|', '?', '*', '\\', '/']
        if any(char in filename for char in dangerous_chars):
            return False
        
        # 检查长度
        if len(filename) > 255:
            return False
        
        # 检查保留名称
        reserved_names = {
            'CON', 'PRN', 'AUX', 'NUL',
            'COM1', 'COM2', 'COM3', 'COM4', 'COM5', 'COM6', 'COM7', 'COM8', 'COM9',
            'LPT1', 'LPT2', 'LPT3', 'LPT4', 'LPT5', 'LPT6', 'LPT7', 'LPT8', 'LPT9'
        }
        
        base_name = os.path.splitext(filename)[0].upper()
        if base_name in reserved_names:
            return False
        
        return True
    
    def get_safe_filename(self, filename: str) -> str:
        """获取安全的文件名"""
        if not filename:
            return "unnamed"
        
        # 移除危险字符
        safe_name = re.sub(r'[<>:"|?*\\/]$', '', filename)
        
        # 限制长度
        if len(safe_name) > 200:
            name, ext = os.path.splitext(safe_name)
            safe_name = name[:200-len(ext)] + ext
        
        return safe_name or "unnamed"

class SecurityConfig:
    """安全配置类"""
    
    def __init__(self):
        self.max_file_size_mb = 500
        self.max_concurrent_operations = 4
        self.enable_input_validation = True
        self.enable_rate_limiting = True
        self.rate_limit_requests_per_minute = 60
        self.allowed_video_formats = list(InputValidator.ALLOWED_VIDEO_EXTENSIONS)
        self.enable_file_sandbox = True
        self.temp_directory = "/tmp/visionsub"
        self.enable_audit_logging = True
        self.audit_log_file = "security_audit.log"
    
    def validate_config(self) -> List[str]:
        """验证安全配置"""
        errors = []
        
        if self.max_file_size_mb < 10 or self.max_file_size_mb > 2048:
            errors.append("文件大小限制必须在10MB到2048MB之间")
        
        if self.max_concurrent_operations < 1 or self.max_concurrent_operations > 16:
            errors.append("并发操作数必须在1到16之间")
        
        if self.rate_limit_requests_per_minute < 1 or self.rate_limit_requests_per_minute > 1000:
            errors.append("速率限制必须在1到1000请求/分钟之间")
        
        if not self.allowed_video_formats:
            errors.append("至少需要允许一种视频格式")
        
        return errors

# 全局输入验证器实例
input_validator = InputValidator()