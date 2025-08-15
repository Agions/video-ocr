"""
安全文件处理模块
提供安全的文件操作和临时文件管理
"""

import os
import tempfile
import hashlib
import shutil
import stat
import logging
from pathlib import Path
from typing import Optional, Dict, Any
from contextlib import contextmanager
import uuid

logger = logging.getLogger(__name__)

class SecureFileHandler:
    """安全文件处理器"""
    
    def __init__(self, temp_dir: Optional[str] = None):
        self.temp_dir = temp_dir or tempfile.gettempdir()
        self.app_temp_dir = os.path.join(self.temp_dir, "visionsub")
        self._ensure_temp_dir()
        
    def _ensure_temp_dir(self):
        """确保临时目录存在且权限正确"""
        try:
            os.makedirs(self.app_temp_dir, mode=0o700, exist_ok=True)
            # 设置目录权限为仅当前用户可读写执行
            os.chmod(self.app_temp_dir, 0o700)
        except Exception as e:
            logger.error(f"创建临时目录失败: {e}")
            raise
    
    def create_secure_temp_file(self, prefix: str = "visionsub_", 
                              suffix: str = ".tmp") -> str:
        """创建安全的临时文件"""
        try:
            # 使用UUID生成唯一文件名
            unique_id = str(uuid.uuid4())
            temp_filename = f"{prefix}{unique_id}{suffix}"
            temp_path = os.path.join(self.app_temp_dir, temp_filename)
            
            # 创建文件并设置权限
            with open(temp_path, 'wb') as f:
                pass
            
            # 设置文件权限为仅当前用户可读写
            os.chmod(temp_path, 0o600)
            
            logger.debug(f"创建安全临时文件: {temp_path}")
            return temp_path
            
        except Exception as e:
            logger.error(f"创建临时文件失败: {e}")
            raise
    
    def create_secure_temp_dir(self, prefix: str = "visionsub_") -> str:
        """创建安全的临时目录"""
        try:
            unique_id = str(uuid.uuid4())
            temp_dirname = f"{prefix}{unique_id}"
            temp_path = os.path.join(self.app_temp_dir, temp_dirname)
            
            # 创建目录并设置权限
            os.makedirs(temp_path, mode=0o700)
            os.chmod(temp_path, 0o700)
            
            logger.debug(f"创建安全临时目录: {temp_path}")
            return temp_path
            
        except Exception as e:
            logger.error(f"创建临时目录失败: {e}")
            raise
    
    @contextmanager
    def secure_file_context(self, file_path: str):
        """安全文件操作上下文管理器"""
        temp_path = None
        try:
            # 创建临时文件
            temp_path = self.create_secure_temp_file()
            
            # 提供临时文件路径
            yield temp_path
            
        except Exception as e:
            logger.error(f"文件操作失败: {e}")
            raise
        finally:
            # 清理临时文件
            if temp_path and os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                    logger.debug(f"清理临时文件: {temp_path}")
                except Exception as e:
                    logger.warning(f"清理临时文件失败: {e}")
    
    def calculate_file_hash(self, file_path: str, algorithm: str = 'sha256') -> str:
        """计算文件哈希值"""
        try:
            hash_func = hashlib.new(algorithm)
            
            with open(file_path, 'rb') as f:
                for chunk in iter(lambda: f.read(8192), b''):
                    hash_func.update(chunk)
            
            file_hash = hash_func.hexdigest()
            logger.debug(f"文件哈希计算完成: {file_path} -> {file_hash}")
            return file_hash
            
        except Exception as e:
            logger.error(f"计算文件哈希失败: {e}")
            raise
    
    def verify_file_integrity(self, file_path: str, expected_hash: str, 
                            algorithm: str = 'sha256') -> bool:
        """验证文件完整性"""
        try:
            actual_hash = self.calculate_file_hash(file_path, algorithm)
            is_valid = actual_hash.lower() == expected_hash.lower()
            
            if not is_valid:
                logger.warning(f"文件完整性验证失败: {file_path}")
                logger.warning(f"期望哈希: {expected_hash}")
                logger.warning(f"实际哈希: {actual_hash}")
            else:
                logger.debug(f"文件完整性验证通过: {file_path}")
            
            return is_valid
            
        except Exception as e:
            logger.error(f"文件完整性验证失败: {e}")
            return False
    
    def secure_copy_file(self, src_path: str, dst_path: str) -> bool:
        """安全复制文件"""
        try:
            # 验证源文件
            if not os.path.exists(src_path):
                logger.error(f"源文件不存在: {src_path}")
                return False
            
            # 创建目标目录
            dst_dir = os.path.dirname(dst_path)
            if dst_dir and not os.path.exists(dst_dir):
                os.makedirs(dst_dir, mode=0o700)
            
            # 复制文件
            shutil.copy2(src_path, dst_path)
            
            # 设置文件权限
            os.chmod(dst_path, 0o600)
            
            logger.debug(f"安全复制文件: {src_path} -> {dst_path}")
            return True
            
        except Exception as e:
            logger.error(f"文件复制失败: {e}")
            return False
    
    def secure_delete_file(self, file_path: str) -> bool:
        """安全删除文件"""
        try:
            if not os.path.exists(file_path):
                logger.warning(f"文件不存在，无需删除: {file_path}")
                return True
            
            # 多次覆写文件内容（安全删除）
            if os.path.isfile(file_path):
                file_size = os.path.getsize(file_path)
                
                # 用随机数据覆写3次
                import random
                for _ in range(3):
                    with open(file_path, 'r+b') as f:
                        random_data = bytes([random.randint(0, 255) for _ in range(file_size)])
                        f.write(random_data)
                        f.flush()
                        os.fsync(f.fileno())
            
            # 删除文件
            os.remove(file_path)
            
            logger.debug(f"安全删除文件: {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"文件删除失败: {e}")
            return False
    
    def get_file_metadata(self, file_path: str) -> Dict[str, Any]:
        """获取文件元数据"""
        try:
            if not os.path.exists(file_path):
                return {}
            
            stat_info = os.stat(file_path)
            
            metadata = {
                'size': stat_info.st_size,
                'modified_time': stat_info.st_mtime,
                'created_time': stat_info.st_ctime,
                'mode': stat_info.st_mode,
                'is_file': os.path.isfile(file_path),
                'is_dir': os.path.isdir(file_path),
                'permissions': oct(stat_info.st_mode & 0o777),
            }
            
            return metadata
            
        except Exception as e:
            logger.error(f"获取文件元数据失败: {e}")
            return {}
    
    def cleanup_temp_files(self, max_age_hours: int = 24):
        """清理过期的临时文件"""
        try:
            import time
            current_time = time.time()
            max_age_seconds = max_age_hours * 3600
            
            cleaned_count = 0
            
            for item in os.listdir(self.app_temp_dir):
                item_path = os.path.join(self.app_temp_dir, item)
                
                try:
                    # 检查文件/目录年龄
                    if os.path.isfile(item_path):
                        file_age = current_time - os.path.getmtime(item_path)
                        if file_age > max_age_seconds:
                            if self.secure_delete_file(item_path):
                                cleaned_count += 1
                    
                    elif os.path.isdir(item_path):
                        dir_age = current_time - os.path.getctime(item_path)
                        if dir_age > max_age_seconds:
                            shutil.rmtree(item_path)
                            cleaned_count += 1
                
                except Exception as e:
                    logger.warning(f"清理临时文件失败: {item_path} - {e}")
            
            logger.info(f"清理临时文件完成，共清理 {cleaned_count} 个文件/目录")
            
        except Exception as e:
            logger.error(f"清理临时文件失败: {e}")
    
    def validate_file_permissions(self, file_path: str) -> bool:
        """验证文件权限"""
        try:
            if not os.path.exists(file_path):
                return False
            
            stat_info = os.stat(file_path)
            mode = stat_info.st_mode
            
            # 检查是否为组或其他用户可写
            if mode & stat.S_IWGRP or mode & stat.S_IWOTH:
                logger.warning(f"文件权限过于宽松: {file_path}")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"验证文件权限失败: {e}")
            return False

class FileSandbox:
    """文件沙箱类，提供隔离的文件操作环境"""
    
    def __init__(self, sandbox_dir: Optional[str] = None):
        self.sandbox_dir = sandbox_dir or os.path.join(tempfile.gettempdir(), "visionsub_sandbox")
        self._ensure_sandbox_dir()
        self.file_handler = SecureFileHandler(self.sandbox_dir)
    
    def _ensure_sandbox_dir(self):
        """确保沙箱目录存在"""
        try:
            os.makedirs(self.sandbox_dir, mode=0o700, exist_ok=True)
            os.chmod(self.sandbox_dir, 0o700)
        except Exception as e:
            logger.error(f"创建沙箱目录失败: {e}")
            raise
    
    def is_path_in_sandbox(self, file_path: str) -> bool:
        """检查路径是否在沙箱内"""
        try:
            abs_path = os.path.abspath(file_path)
            sandbox_abs_path = os.path.abspath(self.sandbox_dir)
            return abs_path.startswith(sandbox_abs_path)
        except Exception:
            return False
    
    def sanitize_path(self, file_path: str) -> Optional[str]:
        """清理路径，确保在沙箱内"""
        try:
            # 移除路径中的危险字符
            clean_path = file_path.replace('..', '').replace('//', '/')
            
            # 构建沙箱内路径
            sandbox_path = os.path.join(self.sandbox_dir, clean_path.lstrip('/'))
            
            # 确保路径在沙箱内
            if self.is_path_in_sandbox(sandbox_path):
                return sandbox_path
            else:
                logger.warning(f"路径超出沙箱范围: {file_path}")
                return None
                
        except Exception as e:
            logger.error(f"路径清理失败: {e}")
            return None
    
    def create_file_in_sandbox(self, filename: str, content: bytes = b'') -> Optional[str]:
        """在沙箱内创建文件"""
        try:
            safe_path = self.sanitize_path(filename)
            if not safe_path:
                return None
            
            # 确保目录存在
            os.makedirs(os.path.dirname(safe_path), mode=0o700, exist_ok=True)
            
            # 写入文件
            with open(safe_path, 'wb') as f:
                f.write(content)
            
            # 设置权限
            os.chmod(safe_path, 0o600)
            
            return safe_path
            
        except Exception as e:
            logger.error(f"沙箱内文件创建失败: {e}")
            return None

# 全局文件处理器实例
file_handler = SecureFileHandler()
file_sandbox = FileSandbox()