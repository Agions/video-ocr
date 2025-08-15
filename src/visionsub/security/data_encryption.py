"""
数据加密模块
提供全面的数据加密和安全存储功能
"""

import os
import json
import base64
import hashlib
import logging
from typing import Any, Dict, List, Optional, Union, Tuple
from pathlib import Path
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives import padding
from cryptography.hazmat.backends import default_backend
import secrets

logger = logging.getLogger(__name__)

class DataEncryptor:
    """数据加密器"""
    
    def __init__(self, key: Optional[bytes] = None):
        self.key = key or self._generate_key()
        self.fernet = Fernet(self.key)
        
    def _generate_key(self) -> bytes:
        """生成加密密钥"""
        return Fernet.generate_key()
    
    def encrypt(self, data: Union[str, bytes, Dict]) -> str:
        """加密数据"""
        try:
            if isinstance(data, dict):
                data = json.dumps(data, ensure_ascii=False).encode('utf-8')
            elif isinstance(data, str):
                data = data.encode('utf-8')
            elif not isinstance(data, bytes):
                data = str(data).encode('utf-8')
            
            encrypted_data = self.fernet.encrypt(data)
            return base64.b64encode(encrypted_data).decode('utf-8')
            
        except Exception as e:
            logger.error(f"数据加密失败: {e}")
            raise
    
    def decrypt(self, encrypted_data: str) -> bytes:
        """解密数据"""
        try:
            encrypted_bytes = base64.b64decode(encrypted_data.encode('utf-8'))
            decrypted_data = self.fernet.decrypt(encrypted_bytes)
            return decrypted_data
            
        except Exception as e:
            logger.error(f"数据解密失败: {e}")
            raise
    
    def encrypt_json(self, data: Dict) -> str:
        """加密JSON数据"""
        return self.encrypt(data)
    
    def decrypt_json(self, encrypted_data: str) -> Dict:
        """解密JSON数据"""
        decrypted_bytes = self.decrypt(encrypted_data)
        return json.loads(decrypted_bytes.decode('utf-8'))

class PasswordManager:
    """密码管理器"""
    
    def __init__(self):
        self pepper = self._load_or_create_pepper()
        
    def _load_or_create_pepper(self) -> bytes:
        """加载或创建密码pepper"""
        pepper_file = os.path.join(os.path.expanduser("~/.visionsub"), "pepper.key")
        
        if os.path.exists(pepper_file):
            with open(pepper_file, 'rb') as f:
                return f.read()
        else:
            pepper = secrets.token_bytes(32)
            os.makedirs(os.path.dirname(pepper_file), exist_ok=True)
            with open(pepper_file, 'wb') as f:
                f.write(pepper)
            os.chmod(pepper_file, 0o600)
            return pepper
    
    def hash_password(self, password: str, salt: Optional[bytes] = None) -> Tuple[str, bytes]:
        """哈希密码"""
        if salt is None:
            salt = secrets.token_bytes(16)
        
        # 使用pepper增强安全性
        salted_password = password.encode('utf-8') + self.pepper
        
        # 使用PBKDF2进行密码哈希
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
            backend=default_backend()
        )
        
        password_hash = kdf.derive(salted_password)
        return base64.b64encode(password_hash).decode('utf-8'), salt
    
    def verify_password(self, password: str, password_hash: str, salt: bytes) -> bool:
        """验证密码"""
        try:
            salted_password = password.encode('utf-8') + self.pepper
            
            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=32,
                salt=salt,
                iterations=100000,
                backend=default_backend()
            )
            
            kdf.verify(salted_password, base64.b64decode(password_hash))
            return True
            
        except Exception:
            return False
    
    def generate_secure_password(self, length: int = 16) -> str:
        """生成安全密码"""
        alphabet = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789!@#$%^&*"
        return ''.join(secrets.choice(alphabet) for _ in range(length))

class SecureTokenManager:
    """安全令牌管理器"""
    
    def __init__(self):
        self.tokens = {}
        self.token_expiry = {}
        
    def generate_token(self, user_id: str, expiry_hours: int = 24) -> str:
        """生成令牌"""
        token = secrets.token_urlsafe(32)
        self.tokens[token] = user_id
        self.token_expiry[token] = time.time() + expiry_hours * 3600
        return token
    
    def validate_token(self, token: str) -> Optional[str]:
        """验证令牌"""
        if token not in self.tokens:
            return None
        
        # 检查是否过期
        if time.time() > self.token_expiry[token]:
            del self.tokens[token]
            del self.token_expiry[token]
            return None
        
        return self.tokens[token]
    
    def revoke_token(self, token: str) -> bool:
        """撤销令牌"""
        if token in self.tokens:
            del self.tokens[token]
            del self.token_expiry[token]
            return True
        return False
    
    def cleanup_expired_tokens(self):
        """清理过期令牌"""
        current_time = time.time()
        expired_tokens = [
            token for token, expiry in self.token_expiry.items()
            if current_time > expiry
        ]
        
        for token in expired_tokens:
            del self.tokens[token]
            del self.token_expiry[token]

class SecureConfigManager:
    """安全配置管理器"""
    
    def __init__(self, config_dir: Optional[str] = None):
        self.config_dir = config_dir or os.path.join(os.path.expanduser("~/.visionsub"), "config")
        self.encryptor = DataEncryptor()
        self.password_manager = PasswordManager()
        self._ensure_config_dir()
        
    def _ensure_config_dir(self):
        """确保配置目录存在"""
        os.makedirs(self.config_dir, mode=0o700, exist_ok=True)
        os.chmod(self.config_dir, 0o700)
    
    def save_config(self, config: Dict, config_name: str, password: Optional[str] = None):
        """保存配置"""
        try:
            config_file = os.path.join(self.config_dir, f"{config_name}.enc")
            
            if password:
                # 使用密码加密
                password_hash, salt = self.password_manager.hash_password(password)
                encrypted_config = self.encryptor.encrypt(config)
                
                secure_config = {
                    'password_hash': password_hash,
                    'salt': base64.b64encode(salt).decode('utf-8'),
                    'config': encrypted_config,
                    'version': '1.0'
                }
            else:
                # 使用默认密钥加密
                secure_config = {
                    'config': self.encryptor.encrypt(config),
                    'version': '1.0'
                }
            
            with open(config_file, 'w', encoding='utf-8') as f:
                json.dump(secure_config, f, indent=2, ensure_ascii=False)
            
            os.chmod(config_file, 0o600)
            logger.info(f"配置保存成功: {config_name}")
            
        except Exception as e:
            logger.error(f"配置保存失败: {e}")
            raise
    
    def load_config(self, config_name: str, password: Optional[str] = None) -> Optional[Dict]:
        """加载配置"""
        try:
            config_file = os.path.join(self.config_dir, f"{config_name}.enc")
            
            if not os.path.exists(config_file):
                return None
            
            with open(config_file, 'r', encoding='utf-8') as f:
                secure_config = json.load(f)
            
            # 验证配置版本
            if secure_config.get('version') != '1.0':
                logger.error("配置版本不兼容")
                return None
            
            # 验证密码（如果提供）
            if password and 'password_hash' in secure_config:
                salt = base64.b64decode(secure_config['salt'].encode('utf-8'))
                if not self.password_manager.verify_password(password, secure_config['password_hash'], salt):
                    logger.error("密码验证失败")
                    return None
            
            # 解密配置
            encrypted_config = secure_config['config']
            return self.encryptor.decrypt_json(encrypted_config)
            
        except Exception as e:
            logger.error(f"配置加载失败: {e}")
            return None
    
    def delete_config(self, config_name: str) -> bool:
        """删除配置"""
        try:
            config_file = os.path.join(self.config_dir, f"{config_name}.enc")
            
            if os.path.exists(config_file):
                os.remove(config_file)
                logger.info(f"配置删除成功: {config_name}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"配置删除失败: {e}")
            return False
    
    def list_configs(self) -> List[str]:
        """列出配置"""
        try:
            configs = []
            for file in os.listdir(self.config_dir):
                if file.endswith('.enc'):
                    config_name = file[:-4]  # 移除.enc后缀
                    configs.append(config_name)
            return configs
        except Exception as e:
            logger.error(f"列出配置失败: {e}")
            return []

class SecureDatabase:
    """安全数据库"""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.encryptor = DataEncryptor()
        self._ensure_db_directory()
        
    def _ensure_db_directory(self):
        """确保数据库目录存在"""
        os.makedirs(os.path.dirname(self.db_path), mode=0o700, exist_ok=True)
    
    def save_record(self, key: str, data: Dict):
        """保存记录"""
        try:
            # 加载数据库
            db_data = self._load_database()
            
            # 加密数据
            encrypted_data = self.encryptor.encrypt_json(data)
            
            # 保存记录
            db_data[key] = {
                'data': encrypted_data,
                'timestamp': time.time(),
                'version': '1.0'
            }
            
            # 保存数据库
            self._save_database(db_data)
            
            logger.debug(f"记录保存成功: {key}")
            
        except Exception as e:
            logger.error(f"记录保存失败: {e}")
            raise
    
    def load_record(self, key: str) -> Optional[Dict]:
        """加载记录"""
        try:
            db_data = self._load_database()
            
            if key not in db_data:
                return None
            
            record = db_data[key]
            
            # 验证版本
            if record.get('version') != '1.0':
                logger.error(f"记录版本不兼容: {key}")
                return None
            
            # 解密数据
            return self.encryptor.decrypt_json(record['data'])
            
        except Exception as e:
            logger.error(f"记录加载失败: {e}")
            return None
    
    def delete_record(self, key: str) -> bool:
        """删除记录"""
        try:
            db_data = self._load_database()
            
            if key in db_data:
                del db_data[key]
                self._save_database(db_data)
                logger.debug(f"记录删除成功: {key}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"记录删除失败: {e}")
            return False
    
    def list_records(self) -> List[str]:
        """列出记录"""
        try:
            db_data = self._load_database()
            return list(db_data.keys())
        except Exception as e:
            logger.error(f"列出记录失败: {e}")
            return []
    
    def _load_database(self) -> Dict:
        """加载数据库"""
        if not os.path.exists(self.db_path):
            return {}
        
        with open(self.db_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def _save_database(self, db_data: Dict):
        """保存数据库"""
        with open(self.db_path, 'w', encoding='utf-8') as f:
            json.dump(db_data, f, indent=2, ensure_ascii=False)
        
        os.chmod(self.db_path, 0o600)

class SecureFileSystem:
    """安全文件系统"""
    
    def __init__(self, base_path: str):
        self.base_path = Path(base_path)
        self.encryptor = DataEncryptor()
        self._ensure_base_directory()
        
    def _ensure_base_directory(self):
        """确保基础目录存在"""
        self.base_path.mkdir(mode=0o700, parents=True, exist_ok=True)
    
    def save_file(self, file_path: str, data: Union[str, bytes], encrypt: bool = True):
        """保存文件"""
        try:
            full_path = self.base_path / file_path
            full_path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            
            if encrypt:
                if isinstance(data, str):
                    encrypted_data = self.encryptor.encrypt(data)
                else:
                    encrypted_data = self.encryptor.encrypt(data)
                
                with open(full_path, 'w', encoding='utf-8') as f:
                    f.write(encrypted_data)
            else:
                if isinstance(data, str):
                    with open(full_path, 'w', encoding='utf-8') as f:
                        f.write(data)
                else:
                    with open(full_path, 'wb') as f:
                        f.write(data)
            
            os.chmod(full_path, 0o600)
            logger.debug(f"文件保存成功: {file_path}")
            
        except Exception as e:
            logger.error(f"文件保存失败: {e}")
            raise
    
    def load_file(self, file_path: str, encrypted: bool = True) -> Optional[Union[str, bytes]]:
        """加载文件"""
        try:
            full_path = self.base_path / file_path
            
            if not full_path.exists():
                return None
            
            if encrypted:
                with open(full_path, 'r', encoding='utf-8') as f:
                    encrypted_data = f.read()
                
                return self.encryptor.decrypt(encrypted_data)
            else:
                with open(full_path, 'rb') as f:
                    return f.read()
                    
        except Exception as e:
            logger.error(f"文件加载失败: {e}")
            return None
    
    def delete_file(self, file_path: str) -> bool:
        """删除文件"""
        try:
            full_path = self.base_path / file_path
            
            if full_path.exists():
                full_path.unlink()
                logger.debug(f"文件删除成功: {file_path}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"文件删除失败: {e}")
            return False
    
    def list_files(self, directory: str = "") -> List[str]:
        """列出文件"""
        try:
            dir_path = self.base_path / directory
            
            if not dir_path.exists():
                return []
            
            files = []
            for file_path in dir_path.rglob("*"):
                if file_path.is_file():
                    relative_path = file_path.relative_to(self.base_path)
                    files.append(str(relative_path))
            
            return files
            
        except Exception as e:
            logger.error(f"列出文件失败: {e}")
            return []

# 全局实例
data_encryptor = DataEncryptor()
password_manager = PasswordManager()
token_manager = SecureTokenManager()
config_manager = SecureConfigManager()

# 时间函数（需要导入）
import time

# 便捷函数
def encrypt_data(data: Any) -> str:
    """加密数据"""
    return data_encryptor.encrypt(data)

def decrypt_data(encrypted_data: str) -> bytes:
    """解密数据"""
    return data_encryptor.decrypt(encrypted_data)

def hash_password(password: str) -> Tuple[str, bytes]:
    """哈希密码"""
    return password_manager.hash_password(password)

def verify_password(password: str, password_hash: str, salt: bytes) -> bool:
    """验证密码"""
    return password_manager.verify_password(password, password_hash, salt)

def generate_token(user_id: str, expiry_hours: int = 24) -> str:
    """生成令牌"""
    return token_manager.generate_token(user_id, expiry_hours)

def validate_token(token: str) -> Optional[str]:
    """验证令牌"""
    return token_manager.validate_token(token)