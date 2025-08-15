"""
安全数据存储模块
提供加密的数据存储和配置管理
"""

import os
import json
import base64
import hashlib
import logging
from typing import Any, Dict, Optional, Union
from pathlib import Path
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import yaml

logger = logging.getLogger(__name__)

class SecureStorage:
    """安全存储管理器"""
    
    def __init__(self, storage_dir: Optional[str] = None):
        self.storage_dir = storage_dir or os.path.expanduser("~/.visionsub")
        self.config_file = os.path.join(self.storage_dir, "config.enc")
        self.key_file = os.path.join(self.storage_dir, "master.key")
        self._ensure_storage_dir()
        self._encryption_key = self._load_or_create_key()
        
    def _ensure_storage_dir(self):
        """确保存储目录存在"""
        try:
            os.makedirs(self.storage_dir, mode=0o700, exist_ok=True)
            os.chmod(self.storage_dir, 0o700)
        except Exception as e:
            logger.error(f"创建存储目录失败: {e}")
            raise
    
    def _load_or_create_key(self) -> bytes:
        """加载或创建加密密钥"""
        try:
            if os.path.exists(self.key_file):
                # 加载现有密钥
                with open(self.key_file, 'rb') as f:
                    key = f.read()
                logger.debug("加载现有加密密钥")
            else:
                # 创建新密钥
                key = Fernet.generate_key()
                with open(self.key_file, 'wb') as f:
                    f.write(key)
                os.chmod(self.key_file, 0o600)
                logger.info("创建新的加密密钥")
            
            return key
            
        except Exception as e:
            logger.error(f"加载/创建加密密钥失败: {e}")
            raise
    
    def _derive_key_from_password(self, password: str, salt: bytes) -> bytes:
        """从密码派生密钥"""
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        return base64.urlsafe_b64encode(kdf.derive(password.encode()))
    
    def encrypt_data(self, data: Union[str, bytes, Dict]) -> str:
        """加密数据"""
        try:
            if isinstance(data, dict):
                data = json.dumps(data, ensure_ascii=False)
            elif isinstance(data, str):
                data = data.encode('utf-8')
            elif not isinstance(data, bytes):
                data = str(data).encode('utf-8')
            
            fernet = Fernet(self._encryption_key)
            encrypted_data = fernet.encrypt(data)
            
            return base64.b64encode(encrypted_data).decode('utf-8')
            
        except Exception as e:
            logger.error(f"数据加密失败: {e}")
            raise
    
    def decrypt_data(self, encrypted_data: str) -> bytes:
        """解密数据"""
        try:
            encrypted_bytes = base64.b64decode(encrypted_data.encode('utf-8'))
            fernet = Fernet(self._encryption_key)
            decrypted_data = fernet.decrypt(encrypted_bytes)
            
            return decrypted_data
            
        except Exception as e:
            logger.error(f"数据解密失败: {e}")
            raise
    
    def save_config(self, config: Dict[str, Any], password: Optional[str] = None):
        """保存配置到加密文件"""
        try:
            # 添加时间戳和版本信息
            enhanced_config = {
                'version': '1.0',
                'timestamp': self._get_timestamp(),
                'data': config
            }
            
            # 序列化配置
            config_json = json.dumps(enhanced_config, ensure_ascii=False, indent=2)
            
            if password:
                # 使用密码加密
                salt = os.urandom(16)
                key = self._derive_key_from_password(password, salt)
                fernet = Fernet(key)
                encrypted_config = fernet.encrypt(config_json.encode('utf-8'))
                
                # 保存盐值和加密数据
                final_data = {
                    'salt': base64.b64encode(salt).decode('utf-8'),
                    'data': base64.b64encode(encrypted_config).decode('utf-8')
                }
                
                with open(self.config_file, 'w', encoding='utf-8') as f:
                    json.dump(final_data, f, indent=2, ensure_ascii=False)
            else:
                # 使用主密钥加密
                encrypted_config = self.encrypt_data(config_json)
                with open(self.config_file, 'w', encoding='utf-8') as f:
                    f.write(encrypted_config)
            
            # 设置文件权限
            os.chmod(self.config_file, 0o600)
            
            logger.info("配置保存成功")
            
        except Exception as e:
            logger.error(f"配置保存失败: {e}")
            raise
    
    def load_config(self, password: Optional[str] = None) -> Dict[str, Any]:
        """从加密文件加载配置"""
        try:
            if not os.path.exists(self.config_file):
                return {}
            
            with open(self.config_file, 'r', encoding='utf-8') as f:
                content = f.read().strip()
            
            if password:
                # 使用密码解密
                try:
                    data = json.loads(content)
                    salt = base64.b64decode(data['salt'].encode('utf-8'))
                    encrypted_data = base64.b64decode(data['data'].encode('utf-8'))
                    
                    key = self._derive_key_from_password(password, salt)
                    fernet = Fernet(key)
                    decrypted_config = fernet.decrypt(encrypted_data).decode('utf-8')
                except json.JSONDecodeError:
                    # 回退到主密钥解密
                    decrypted_config = self.decrypt_data(content).decode('utf-8')
            else:
                # 使用主密钥解密
                decrypted_config = self.decrypt_data(content).decode('utf-8')
            
            # 解析配置
            config_data = json.loads(decrypted_config)
            
            # 验证配置格式
            if 'version' not in config_data or 'data' not in config_data:
                raise ValueError("配置格式无效")
            
            logger.info("配置加载成功")
            return config_data['data']
            
        except Exception as e:
            logger.error(f"配置加载失败: {e}")
            return {}
    
    def backup_config(self, backup_path: Optional[str] = None) -> str:
        """备份配置文件"""
        try:
            if not os.path.exists(self.config_file):
                raise FileNotFoundError("配置文件不存在")
            
            if not backup_path:
                timestamp = self._get_timestamp().replace(':', '-')
                backup_path = os.path.join(self.storage_dir, f"config_backup_{timestamp}.enc")
            
            # 创建备份
            import shutil
            shutil.copy2(self.config_file, backup_path)
            os.chmod(backup_path, 0o600)
            
            logger.info(f"配置备份成功: {backup_path}")
            return backup_path
            
        except Exception as e:
            logger.error(f"配置备份失败: {e}")
            raise
    
    def restore_config(self, backup_path: str):
        """从备份恢复配置"""
        try:
            if not os.path.exists(backup_path):
                raise FileNotFoundError("备份文件不存在")
            
            # 验证备份文件
            if not self._validate_backup_file(backup_path):
                raise ValueError("备份文件验证失败")
            
            # 恢复配置
            import shutil
            shutil.copy2(backup_path, self.config_file)
            os.chmod(self.config_file, 0o600)
            
            logger.info(f"配置恢复成功: {backup_path}")
            
        except Exception as e:
            logger.error(f"配置恢复失败: {e}")
            raise
    
    def _validate_backup_file(self, backup_path: str) -> bool:
        """验证备份文件"""
        try:
            # 检查文件大小
            file_size = os.path.getsize(backup_path)
            if file_size == 0 or file_size > 10 * 1024 * 1024:  # 10MB限制
                return False
            
            # 检查文件权限
            import stat
            file_stat = os.stat(backup_path)
            if file_stat.st_mode & stat.S_IWGRP or file_stat.st_mode & stat.S_IWOTH:
                return False
            
            # 尝试解析文件内容
            with open(backup_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 如果是JSON格式，验证是否可以解析
            if content.strip().startswith('{'):
                try:
                    json.loads(content)
                except json.JSONDecodeError:
                    return False
            
            return True
            
        except Exception:
            return False
    
    def get_config_info(self) -> Dict[str, Any]:
        """获取配置信息"""
        try:
            info = {
                'storage_dir': self.storage_dir,
                'config_file': self.config_file,
                'key_file': self.key_file,
                'config_exists': os.path.exists(self.config_file),
                'key_exists': os.path.exists(self.key_file),
            }
            
            if os.path.exists(self.config_file):
                stat_info = os.stat(self.config_file)
                info.update({
                    'config_size': stat_info.st_size,
                    'config_modified': stat_info.st_mtime,
                    'config_permissions': oct(stat_info.st_mode & 0o777),
                })
            
            return info
            
        except Exception as e:
            logger.error(f"获取配置信息失败: {e}")
            return {}
    
    def _get_timestamp(self) -> str:
        """获取当前时间戳"""
        from datetime import datetime
        return datetime.now().isoformat()
    
    def change_password(self, old_password: Optional[str], new_password: str):
        """更改配置密码"""
        try:
            # 使用旧密码加载配置
            config = self.load_config(old_password)
            
            # 使用新密码保存配置
            self.save_config(config, new_password)
            
            logger.info("配置密码更改成功")
            
        except Exception as e:
            logger.error(f"配置密码更改失败: {e}")
            raise
    
    def export_config(self, export_path: str, password: Optional[str] = None):
        """导出配置"""
        try:
            config = self.load_config(password)
            
            # 创建导出数据
            export_data = {
                'version': '1.0',
                'exported_at': self._get_timestamp(),
                'application': 'VisionSub',
                'config': config
            }
            
            # 保存到文件
            with open(export_path, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)
            
            os.chmod(export_path, 0o600)
            logger.info(f"配置导出成功: {export_path}")
            
        except Exception as e:
            logger.error(f"配置导出失败: {e}")
            raise
    
    def import_config(self, import_path: str, password: Optional[str] = None):
        """导入配置"""
        try:
            # 验证导入文件
            if not os.path.exists(import_path):
                raise FileNotFoundError("导入文件不存在")
            
            # 读取导入数据
            with open(import_path, 'r', encoding='utf-8') as f:
                import_data = json.load(f)
            
            # 验证导入数据格式
            if 'config' not in import_data:
                raise ValueError("导入文件格式无效")
            
            # 导入配置
            config = import_data['config']
            self.save_config(config, password)
            
            logger.info(f"配置导入成功: {import_path}")
            
        except Exception as e:
            logger.error(f"配置导入失败: {e}")
            raise

class SecureCache:
    """安全缓存管理器"""
    
    def __init__(self, cache_dir: Optional[str] = None, max_size_mb: int = 100):
        self.cache_dir = cache_dir or os.path.join(os.path.expanduser("~/.visionsub"), "cache")
        self.max_size_bytes = max_size_mb * 1024 * 1024
        self._ensure_cache_dir()
        
    def _ensure_cache_dir(self):
        """确保缓存目录存在"""
        try:
            os.makedirs(self.cache_dir, mode=0o700, exist_ok=True)
            os.chmod(self.cache_dir, 0o700)
        except Exception as e:
            logger.error(f"创建缓存目录失败: {e}")
            raise
    
    def store_data(self, key: str, data: Any, ttl_hours: int = 24) -> bool:
        """存储数据到缓存"""
        try:
            import pickle
            import time
            
            # 创建缓存条目
            cache_entry = {
                'data': data,
                'created_at': time.time(),
                'ttl': ttl_hours * 3600
            }
            
            # 序列化数据
            serialized_data = pickle.dumps(cache_entry)
            
            # 创建缓存文件
            cache_file = os.path.join(self.cache_dir, f"{hashlib.md5(key.encode()).hexdigest()}.cache")
            
            with open(cache_file, 'wb') as f:
                f.write(serialized_data)
            
            os.chmod(cache_file, 0o600)
            
            # 检查缓存大小
            self._check_cache_size()
            
            logger.debug(f"数据缓存成功: {key}")
            return True
            
        except Exception as e:
            logger.error(f"数据缓存失败: {e}")
            return False
    
    def retrieve_data(self, key: str) -> Optional[Any]:
        """从缓存检索数据"""
        try:
            import pickle
            import time
            
            cache_file = os.path.join(self.cache_dir, f"{hashlib.md5(key.encode()).hexdigest()}.cache")
            
            if not os.path.exists(cache_file):
                return None
            
            # 读取缓存文件
            with open(cache_file, 'rb') as f:
                cache_entry = pickle.load(f)
            
            # 检查是否过期
            if time.time() - cache_entry['created_at'] > cache_entry['ttl']:
                os.remove(cache_file)
                return None
            
            logger.debug(f"缓存命中: {key}")
            return cache_entry['data']
            
        except Exception as e:
            logger.error(f"缓存检索失败: {e}")
            return None
    
    def clear_cache(self):
        """清空缓存"""
        try:
            import shutil
            if os.path.exists(self.cache_dir):
                shutil.rmtree(self.cache_dir)
                self._ensure_cache_dir()
            
            logger.info("缓存清空成功")
            
        except Exception as e:
            logger.error(f"缓存清空失败: {e}")
    
    def _check_cache_size(self):
        """检查并清理过期或过大的缓存"""
        try:
            import time
            
            total_size = 0
            cache_files = []
            
            # 统计缓存大小
            for filename in os.listdir(self.cache_dir):
                if filename.endswith('.cache'):
                    file_path = os.path.join(self.cache_dir, filename)
                    file_size = os.path.getsize(file_path)
                    total_size += file_size
                    
                    # 检查文件是否过期
                    with open(file_path, 'rb') as f:
                        import pickle
                        cache_entry = pickle.load(f)
                    
                    if time.time() - cache_entry['created_at'] > cache_entry['ttl']:
                        cache_files.append((file_path, file_size, True))  # 标记为过期
                    else:
                        cache_files.append((file_path, file_size, False))
            
            # 如果缓存过大，删除最旧的文件
            if total_size > self.max_size_bytes:
                # 按创建时间排序
                cache_files.sort(key=lambda x: os.path.getctime(x[0]))
                
                for file_path, file_size, is_expired in cache_files:
                    if total_size > self.max_size_bytes or is_expired:
                        os.remove(file_path)
                        total_size -= file_size
                        logger.debug(f"清理缓存文件: {file_path}")
            
        except Exception as e:
            logger.error(f"缓存大小检查失败: {e}")

# 全局存储实例
secure_storage = SecureStorage()
secure_cache = SecureCache()