"""
网络安全模块
提供安全的网络通信和API调用功能
"""

import ssl
import socket
import hashlib
import logging
from typing import Dict, Any, Optional, List
from urllib.parse import urlparse
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import certifi

logger = logging.getLogger(__name__)

class SecureHTTPClient:
    """安全HTTP客户端"""
    
    def __init__(self, timeout: int = 30, max_retries: int = 3):
        self.timeout = timeout
        self.max_retries = max_retries
        self.session = self._create_secure_session()
        
    def _create_secure_session(self) -> requests.Session:
        """创建安全的HTTP会话"""
        session = requests.Session()
        
        # 配置重试策略
        retry_strategy = Retry(
            total=self.max_retries,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        
        # 创建适配器
        adapter = HTTPAdapter(
            max_retries=retry_strategy,
            pool_connections=10,
            pool_maxsize=100
        )
        
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        
        # 设置安全头部
        session.headers.update({
            'User-Agent': 'VisionSub/2.0.0',
            'Accept': 'application/json',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
        })
        
        return session
    
    def get_ssl_context(self) -> ssl.SSLContext:
        """获取SSL上下文"""
        context = ssl.create_default_context()
        context.load_verify_locations(certifi.where())
        
        # 禁用不安全的协议和密码套件
        context.minimum_version = ssl.TLSVersion.TLSv1_2
        context.set_ciphers('ECDHE-ECDSA-AES256-GCM-SHA384:ECDHE-RSA-AES256-GCM-SHA384')
        
        # 启用证书验证
        context.verify_mode = ssl.CERT_REQUIRED
        context.check_hostname = True
        
        return context
    
    def secure_request(self, method: str, url: str, **kwargs) -> requests.Response:
        """发送安全HTTP请求"""
        try:
            # 验证URL
            if not self._is_safe_url(url):
                raise ValueError(f"不安全的URL: {url}")
            
            # 设置超时
            kwargs.setdefault('timeout', self.timeout)
            
            # 验证证书
            kwargs.setdefault('verify', certifi.where())
            
            # 发送请求
            response = self.session.request(method, url, **kwargs)
            
            # 验证响应
            self._validate_response(response)
            
            return response
            
        except requests.exceptions.SSLError as e:
            logger.error(f"SSL错误: {e}")
            raise
        except requests.exceptions.RequestException as e:
            logger.error(f"请求失败: {e}")
            raise
    
    def _is_safe_url(self, url: str) -> bool:
        """检查URL是否安全"""
        try:
            parsed = urlparse(url)
            
            # 检查协议
            if parsed.scheme not in ['https', 'http']:
                return False
            
            # 检查是否为本地地址
            if parsed.hostname in ['localhost', '127.0.0.1', '::1']:
                return False
            
            # 检查私有IP地址
            if self._is_private_ip(parsed.hostname):
                return False
            
            return True
            
        except Exception:
            return False
    
    def _is_private_ip(self, hostname: str) -> bool:
        """检查是否为私有IP地址"""
        try:
            # 解析主机名
            addrinfo = socket.getaddrinfo(hostname, None)
            for addr in addrinfo:
                ip = addr[4][0]
                
                # 检查IPv4私有地址
                if '.' in ip:
                    parts = ip.split('.')
                    if (parts[0] == '10' or 
                        (parts[0] == '172' and 16 <= int(parts[1]) <= 31) or
                        (parts[0] == '192' and parts[1] == '168')):
                        return True
                
                # 检查IPv6私有地址
                if ':' in ip:
                    if ip.startswith('fc00::') or ip.startswith('fd00::'):
                        return True
            
            return False
            
        except Exception:
            return False
    
    def _validate_response(self, response: requests.Response):
        """验证HTTP响应"""
        # 检查状态码
        if response.status_code >= 400:
            logger.warning(f"HTTP错误状态码: {response.status_code}")
        
        # 检查内容类型
        content_type = response.headers.get('content-type', '')
        if 'application/json' in content_type:
            try:
                response.json()
            except ValueError:
                logger.warning("响应不是有效的JSON格式")
        
        # 检查内容大小
        content_length = response.headers.get('content-length')
        if content_length and int(content_length) > 10 * 1024 * 1024:  # 10MB
            logger.warning("响应内容过大")

class APIRateLimiter:
    """API速率限制器"""
    
    def __init__(self, max_requests_per_minute: int = 60):
        self.max_requests = max_requests_per_minute
        self.requests = []
        self.window_size = 60  # 60秒窗口
        
    def can_make_request(self) -> bool:
        """检查是否可以发出请求"""
        import time
        
        current_time = time.time()
        
        # 清理过期的请求记录
        self.requests = [req_time for req_time in self.requests 
                        if current_time - req_time < self.window_size]
        
        # 检查是否超过限制
        if len(self.requests) >= self.max_requests:
            return False
        
        # 记录新请求
        self.requests.append(current_time)
        return True
    
    def get_wait_time(self) -> float:
        """获取需要等待的时间"""
        import time
        
        if len(self.requests) < self.max_requests:
            return 0.0
        
        current_time = time.time()
        oldest_request = min(self.requests)
        return max(0, self.window_size - (current_time - oldest_request))

class SecurityHeaders:
    """安全头部管理"""
    
    @staticmethod
    def get_secure_headers() -> Dict[str, str]:
        """获取安全HTTP头部"""
        return {
            'X-Content-Type-Options': 'nosniff',
            'X-Frame-Options': 'DENY',
            'X-XSS-Protection': '1; mode=block',
            'Strict-Transport-Security': 'max-age=31536000; includeSubDomains',
            'Content-Security-Policy': "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline';",
            'Referrer-Policy': 'strict-origin-when-cross-origin',
            'Permissions-Policy': 'geolocation=(), microphone=(), camera=()',
        }
    
    @staticmethod
    def validate_headers(headers: Dict[str, str]) -> List[str]:
        """验证响应头部"""
        issues = []
        
        # 检查安全头部
        required_headers = [
            'X-Content-Type-Options',
            'X-Frame-Options',
            'X-XSS-Protection'
        ]
        
        for header in required_headers:
            if header not in headers:
                issues.append(f"缺少安全头部: {header}")
        
        # 检查内容类型
        content_type = headers.get('content-type', '')
        if 'text/html' in content_type and 'X-Content-Type-Options' not in headers:
            issues.append("HTML响应应包含X-Content-Type-Options头部")
        
        return issues

class NetworkSecurity:
    """网络安全工具类"""
    
    @staticmethod
    def validate_certificate(cert_path: str) -> bool:
        """验证证书文件"""
        try:
            with open(cert_path, 'rb') as f:
                cert_data = f.read()
            
            # 创建SSL上下文并验证证书
            context = ssl.create_default_context()
            context.load_verify_locations(cert_path)
            
            return True
            
        except Exception as e:
            logger.error(f"证书验证失败: {e}")
            return False
    
    @staticmethod
    def check_ssl_vulnerabilities(hostname: str, port: int = 443) -> Dict[str, Any]:
        """检查SSL漏洞"""
        results = {
            'hostname': hostname,
            'port': port,
            'vulnerabilities': [],
            'recommendations': []
        }
        
        try:
            # 创建SSL连接
            context = ssl.create_default_context()
            context.check_hostname = False
            context.verify_mode = ssl.CERT_NONE
            
            with socket.create_connection((hostname, port), timeout=10) as sock:
                with context.wrap_socket(sock, server_hostname=hostname) as ssock:
                    # 获取证书信息
                    cert = ssock.getpeercert()
                    if cert:
                        results['certificate'] = {
                            'subject': cert.get('subject', []),
                            'issuer': cert.get('issuer', []),
                            'version': cert.get('version'),
                            'notAfter': cert.get('notAfter'),
                        }
                    
                    # 检查协议版本
                    protocol = ssock.version()
                    if protocol in ['SSLv2', 'SSLv3', 'TLSv1', 'TLSv1.1']:
                        results['vulnerabilities'].append(f"使用不安全的协议: {protocol}")
                        results['recommendations'].append("升级到TLSv1.2或更高版本")
                    
                    # 检查密码套件
                    cipher = ssock.cipher()
                    if cipher:
                        cipher_name = cipher[0]
                        if any(weak in cipher_name for weak in ['RC4', 'DES', 'MD5', 'NULL']):
                            results['vulnerabilities'].append(f"使用弱密码套件: {cipher_name}")
                            results['recommendations'].append("使用强密码套件")
            
        except Exception as e:
            logger.error(f"SSL漏洞检查失败: {e}")
            results['error'] = str(e)
        
        return results
    
    @staticmethod
    def generate_fingerprint(data: str) -> str:
        """生成数据指纹"""
        return hashlib.sha256(data.encode('utf-8')).hexdigest()
    
    @staticmethod
    def validate_integrity(data: str, expected_fingerprint: str) -> bool:
        """验证数据完整性"""
        actual_fingerprint = NetworkSecurity.generate_fingerprint(data)
        return actual_fingerprint == expected_fingerprint

# 全局网络客户端实例
http_client = SecureHTTPClient()
rate_limiter = APIRateLimiter()

# 便捷函数
def secure_get(url: str, **kwargs) -> requests.Response:
    """安全的GET请求"""
    if not rate_limiter.can_make_request():
        wait_time = rate_limiter.get_wait_time()
        raise Exception(f"速率限制，请等待 {wait_time:.1f} 秒")
    
    return http_client.secure_request('GET', url, **kwargs)

def secure_post(url: str, **kwargs) -> requests.Response:
    """安全的POST请求"""
    if not rate_limiter.can_make_request():
        wait_time = rate_limiter.get_wait_time()
        raise Exception(f"速率限制，请等待 {wait_time:.1f} 秒")
    
    return http_client.secure_request('POST', url, **kwargs)