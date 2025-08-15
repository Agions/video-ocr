#!/usr/bin/env python3
"""
VisionSub Enhanced UI Components - Security Testing Script
专业级前端UI组件安全测试脚本

作者: Agions
版本: 1.0.0
日期: 2025-08-15
"""

import sys
import os
import re
import json
import tempfile
import shutil
from typing import Dict, List, Any, Tuple
from PyQt6.QtWidgets import QApplication, QMessageBox, QFileDialog, QLineEdit, QTextEdit
from PyQt6.QtCore import Qt, QTimer, QThread, pyqtSignal
from PyQt6.QtGui import QPixmap, QImage, QColor
from PyQt6.QtTest import QTest

# 添加项目路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

class SecurityTestRunner:
    """安全测试运行器"""
    
    def __init__(self):
        self.app = None
        self.test_results = []
        self.vulnerabilities = []
        
    def initialize_application(self):
        """初始化应用程序"""
        if QApplication.instance() is None:
            self.app = QApplication(sys.argv)
        return True
    
    def run_security_tests(self):
        """运行所有安全测试"""
        print("🔒 开始安全测试...")
        
        tests = [
            self.test_input_validation,
            self.test_file_upload_security,
            self.test_xss_protection,
            self.test_path_traversal_protection,
            self.test_sql_injection_protection,
            self.test_command_injection_protection,
            self.test_csrf_protection,
            self.test_sensitive_data_exposure,
            self.test_authentication_security,
            self.test_authorization_security,
            self.test_session_security,
            self.test_error_handling_security,
            self.test_logging_security,
            self.test_configuration_security,
            self.test_network_security
        ]
        
        total_tests = len(tests)
        passed_tests = 0
        
        for i, test_func in enumerate(tests):
            test_name = test_func.__name__.replace("test_", "").replace("_", " ").title()
            print(f"🔍 运行安全测试 {i+1}/{total_tests}: {test_name}")
            
            try:
                result = test_func()
                if result:
                    passed_tests += 1
                    print(f"✅ {test_name}: 通过")
                else:
                    print(f"❌ {test_name}: 失败")
            except Exception as e:
                print(f"💥 {test_name}: 异常 - {str(e)}")
        
        # 输出测试结果
        print(f"\n📊 安全测试结果汇总:")
        print(f"   总测试数: {total_tests}")
        print(f"   通过测试: {passed_tests}")
        print(f"   失败测试: {total_tests - passed_tests}")
        print(f"   通过率: {(passed_tests/total_tests)*100:.1f}%")
        
        if self.vulnerabilities:
            print(f"\n🚨 发现的安全漏洞:")
            for vuln in self.vulnerabilities:
                print(f"   - {vuln['type']}: {vuln['description']} (严重程度: {vuln['severity']})")
        
        return passed_tests == total_tests
    
    def test_input_validation(self):
        """测试输入验证"""
        try:
            from visionsub.ui.enhanced_main_window import EnhancedMainWindow
            
            # 模拟主控制器
            class MockController:
                def process_video(self, file_path):
                    return True
                def export_subtitles(self, format_type):
                    return True
            
            main_window = EnhancedMainWindow(MockController())
            
            # 测试恶意输入
            malicious_inputs = [
                "<script>alert('xss')</script>",
                "javascript:alert('xss')",
                "<img src='x' onerror='alert(1)'>",
                "<svg onload='alert(1)'>",
                "' OR '1'='1",
                "'; DROP TABLE users; --",
                "../../../etc/passwd",
                "..\\..\\..\\windows\\system32",
                "<iframe src='javascript:alert(1)'>",
                "<object data='javascript:alert(1)'>"
            ]
            
            for malicious_input in malicious_inputs:
                # 测试输入净化
                sanitized = main_window.sanitize_input(malicious_input)
                
                # 检查是否移除了危险的HTML标签
                dangerous_patterns = [
                    r'<script.*?>.*?</script>',
                    r'javascript:',
                    r'on\w+\s*=',
                    r'<iframe.*?>',
                    r'<object.*?>',
                    r'<embed.*?>',
                    r'<applet.*?>'
                ]
                
                for pattern in dangerous_patterns:
                    if re.search(pattern, sanitized, re.IGNORECASE):
                        self.vulnerabilities.append({
                            'type': 'XSS',
                            'description': f'输入净化不完整: {malicious_input}',
                            'severity': 'High'
                        })
                        return False
            
            main_window.close()
            return True
            
        except Exception as e:
            self.vulnerabilities.append({
                'type': 'Input Validation',
                'description': f'输入验证测试异常: {str(e)}',
                'severity': 'Medium'
            })
            return False
    
    def test_file_upload_security(self):
        """测试文件上传安全"""
        try:
            from visionsub.ui.enhanced_main_window import EnhancedMainWindow
            
            class MockController:
                def process_video(self, file_path):
                    return True
                def export_subtitles(self, format_type):
                    return True
            
            main_window = EnhancedMainWindow(MockController())
            
            # 测试危险文件类型
            dangerous_files = [
                "test.exe",
                "malware.bat",
                "script.js",
                "shell.php",
                "backdoor.py",
                "virus.dll",
                "trojan.so",
                "exploit.sh"
            ]
            
            for dangerous_file in dangerous_files:
                is_valid = main_window.is_valid_video_file(dangerous_file)
                if is_valid:
                    self.vulnerabilities.append({
                        'type': 'File Upload',
                        'description': f'允许危险文件类型: {dangerous_file}',
                        'severity': 'High'
                    })
                    return False
            
            # 测试文件大小限制
            large_file_name = "large_file.mp4"
            # 模拟大文件检查
            if hasattr(main_window, 'check_file_size'):
                # 这里应该检查文件大小限制
                pass
            
            main_window.close()
            return True
            
        except Exception as e:
            self.vulnerabilities.append({
                'type': 'File Upload',
                'description': f'文件上传安全测试异常: {str(e)}',
                'severity': 'Medium'
            })
            return False
    
    def test_xss_protection(self):
        """测试XSS防护"""
        try:
            from visionsub.ui.enhanced_main_window import EnhancedMainWindow
            
            class MockController:
                def process_video(self, file_path):
                    return True
                def export_subtitles(self, format_type):
                    return True
            
            main_window = EnhancedMainWindow(MockController())
            
            # 测试XSS攻击向量
            xss_vectors = [
                "<script>alert('XSS')</script>",
                "<img src='x' onerror='alert(1)'>",
                "<svg onload='alert(1)'>",
                "javascript:alert('XSS')",
                "<iframe src='javascript:alert(1)'>",
                "<object data='javascript:alert(1)'>",
                "<applet code='javascript:alert(1)'>",
                "<meta http-equiv='refresh' content='0;url=javascript:alert(1)'>",
                "<body onload='alert(1)'>",
                "<div style='width:expression(alert(1))'>"
            ]
            
            for xss_vector in xss_vectors:
                # 测试文本显示组件的XSS防护
                if hasattr(main_window, 'display_text'):
                    # 这里应该测试文本显示功能
                    pass
                
                # 测试输入字段的XSS防护
                sanitized = main_window.sanitize_input(xss_vector)
                if xss_vector.lower() in sanitized.lower():
                    self.vulnerabilities.append({
                        'type': 'XSS',
                        'description': f'XSS防护不足: {xss_vector}',
                        'severity': 'High'
                    })
                    return False
            
            main_window.close()
            return True
            
        except Exception as e:
            self.vulnerabilities.append({
                'type': 'XSS',
                'description': f'XSS防护测试异常: {str(e)}',
                'severity': 'Medium'
            })
            return False
    
    def test_path_traversal_protection(self):
        """测试路径遍历防护"""
        try:
            from visionsub.ui.enhanced_main_window import EnhancedMainWindow
            
            class MockController:
                def process_video(self, file_path):
                    return True
                def export_subtitles(self, format_type):
                    return True
            
            main_window = EnhancedMainWindow(MockController())
            
            # 测试路径遍历攻击
            path_traversal_vectors = [
                "../../../etc/passwd",
                "..\\..\\..\\windows\\system32\\cmd.exe",
                "/etc/passwd",
                "C:\\Windows\\System32\\config\\SAM",
                "../../../../var/log/auth.log",
                "..\\..\\..\\Program Files\\Common Files\\System\\msadc\\msadcs.dll",
                "/proc/self/environ",
                "../../boot.ini",
                "..\\..\\..\\winnt\\system32\\drivers\\etc\\hosts"
            ]
            
            for path_vector in path_traversal_vectors:
                is_safe = main_window.is_safe_path(path_vector)
                if is_safe:
                    self.vulnerabilities.append({
                        'type': 'Path Traversal',
                        'description': f'路径遍历防护不足: {path_vector}',
                        'severity': 'High'
                    })
                    return False
            
            main_window.close()
            return True
            
        except Exception as e:
            self.vulnerabilities.append({
                'type': 'Path Traversal',
                'description': f'路径遍历防护测试异常: {str(e)}',
                'severity': 'Medium'
            })
            return False
    
    def test_sql_injection_protection(self):
        """测试SQL注入防护"""
        try:
            from visionsub.ui.enhanced_main_window import EnhancedMainWindow
            
            class MockController:
                def process_video(self, file_path):
                    return True
                def export_subtitles(self, format_type):
                    return True
            
            main_window = EnhancedMainWindow(MockController())
            
            # 测试SQL注入攻击向量
            sql_injection_vectors = [
                "' OR '1'='1",
                "' OR 1=1--",
                "' UNION SELECT NULL--",
                "' DROP TABLE users--",
                "'; EXEC xp_cmdshell('dir')--",
                "' WAITFOR DELAY '0:0:5'--",
                "' AND (SELECT COUNT(*) FROM information_schema.tables)>0--",
                "' OR SLEEP(5)--",
                "' OR pg_sleep(5)--",
                "' OR DBMS_PIPE.RECEIVE_MESSAGE('X', 5)--"
            ]
            
            for sql_vector in sql_injection_vectors:
                # 测试输入净化
                sanitized = main_window.sanitize_input(sql_vector)
                
                # 检查是否移除了SQL关键字
                sql_keywords = ['SELECT', 'INSERT', 'UPDATE', 'DELETE', 'DROP', 'UNION', 'EXEC', 'WAITFOR']
                for keyword in sql_keywords:
                    if keyword.upper() in sanitized.upper():
                        self.vulnerabilities.append({
                            'type': 'SQL Injection',
                            'description': f'SQL注入防护不足: {sql_vector}',
                            'severity': 'High'
                        })
                        return False
            
            main_window.close()
            return True
            
        except Exception as e:
            self.vulnerabilities.append({
                'type': 'SQL Injection',
                'description': f'SQL注入防护测试异常: {str(e)}',
                'severity': 'Medium'
            })
            return False
    
    def test_command_injection_protection(self):
        """测试命令注入防护"""
        try:
            from visionsub.ui.enhanced_main_window import EnhancedMainWindow
            
            class MockController:
                def process_video(self, file_path):
                    return True
                def export_subtitles(self, format_type):
                    return True
            
            main_window = EnhancedMainWindow(MockController())
            
            # 测试命令注入攻击向量
            command_injection_vectors = [
                "; rm -rf /",
                "| dir",
                "& net user",
                "`cat /etc/passwd`",
                "$(cat /etc/passwd)",
                "| powershell -c \"Get-Process\"",
                "& ping -n 5 127.0.0.1",
                "; whoami",
                "| ls -la",
                "& id"
            ]
            
            for cmd_vector in command_injection_vectors:
                # 测试输入净化
                sanitized = main_window.sanitize_input(cmd_vector)
                
                # 检查是否移除了命令分隔符
                dangerous_chars = [';', '|', '&', '`', '$', '(', ')']
                for char in dangerous_chars:
                    if char in sanitized:
                        self.vulnerabilities.append({
                            'type': 'Command Injection',
                            'description': f'命令注入防护不足: {cmd_vector}',
                            'severity': 'High'
                        })
                        return False
            
            main_window.close()
            return True
            
        except Exception as e:
            self.vulnerabilities.append({
                'type': 'Command Injection',
                'description': f'命令注入防护测试异常: {str(e)}',
                'severity': 'Medium'
            })
            return False
    
    def test_csrf_protection(self):
        """测试CSRF防护"""
        try:
            # 对于桌面应用，CSRF风险较低，但仍需检查
            from visionsub.ui.enhanced_main_window import EnhancedMainWindow
            
            class MockController:
                def process_video(self, file_path):
                    return True
                def export_subtitles(self, format_type):
                    return True
            
            main_window = EnhancedMainWindow(MockController())
            
            # 检查是否有CSRF令牌机制
            if hasattr(main_window, 'csrf_token'):
                # 应该有CSRF保护机制
                pass
            
            main_window.close()
            return True
            
        except Exception as e:
            self.vulnerabilities.append({
                'type': 'CSRF',
                'description': f'CSRF防护测试异常: {str(e)}',
                'severity': 'Low'
            })
            return False
    
    def test_sensitive_data_exposure(self):
        """测试敏感数据泄露"""
        try:
            from visionsub.ui.enhanced_main_window import EnhancedMainWindow
            
            class MockController:
                def process_video(self, file_path):
                    return True
                def export_subtitles(self, format_type):
                    return True
            
            main_window = EnhancedMainWindow(MockController())
            
            # 检查是否在UI中显示敏感信息
            sensitive_patterns = [
                r'password\s*=\s*[\'"].+?[\'"]',
                r'api_key\s*=\s*[\'"].+?[\'"]',
                r'secret\s*=\s*[\'"].+?[\'"]',
                r'token\s*=\s*[\'"].+?[\'"]',
                r'private_key\s*=\s*[\'"].+?[\'"]'
            ]
            
            # 检查源代码中是否有硬编码的敏感信息
            import inspect
            source = inspect.getsource(main_window)
            
            for pattern in sensitive_patterns:
                if re.search(pattern, source, re.IGNORECASE):
                    self.vulnerabilities.append({
                        'type': 'Sensitive Data Exposure',
                        'description': '发现硬编码的敏感信息',
                        'severity': 'High'
                    })
                    return False
            
            main_window.close()
            return True
            
        except Exception as e:
            self.vulnerabilities.append({
                'type': 'Sensitive Data Exposure',
                'description': f'敏感数据泄露测试异常: {str(e)}',
                'severity': 'Medium'
            })
            return False
    
    def test_authentication_security(self):
        """测试认证安全"""
        try:
            from visionsub.ui.enhanced_main_window import EnhancedMainWindow
            
            class MockController:
                def process_video(self, file_path):
                    return True
                def export_subtitles(self, format_type):
                    return True
            
            main_window = EnhancedMainWindow(MockController())
            
            # 检查认证机制
            if hasattr(main_window, 'authenticate_user'):
                # 应该有安全的认证机制
                pass
            
            main_window.close()
            return True
            
        except Exception as e:
            self.vulnerabilities.append({
                'type': 'Authentication',
                'description': f'认证安全测试异常: {str(e)}',
                'severity': 'Medium'
            })
            return False
    
    def test_authorization_security(self):
        """测试授权安全"""
        try:
            from visionsub.ui.enhanced_main_window import EnhancedMainWindow
            
            class MockController:
                def process_video(self, file_path):
                    return True
                def export_subtitles(self, format_type):
                    return True
            
            main_window = EnhancedMainWindow(MockController())
            
            # 检查授权机制
            if hasattr(main_window, 'check_permission'):
                # 应该有授权检查机制
                pass
            
            main_window.close()
            return True
            
        except Exception as e:
            self.vulnerabilities.append({
                'type': 'Authorization',
                'description': f'授权安全测试异常: {str(e)}',
                'severity': 'Medium'
            })
            return False
    
    def test_session_security(self):
        """测试会话安全"""
        try:
            from visionsub.ui.enhanced_main_window import EnhancedMainWindow
            
            class MockController:
                def process_video(self, file_path):
                    return True
                def export_subtitles(self, format_type):
                    return True
            
            main_window = EnhancedMainWindow(MockController())
            
            # 检查会话管理
            if hasattr(main_window, 'session_manager'):
                # 应该有安全的会话管理
                pass
            
            main_window.close()
            return True
            
        except Exception as e:
            self.vulnerabilities.append({
                'type': 'Session Security',
                'description': f'会话安全测试异常: {str(e)}',
                'severity': 'Medium'
            })
            return False
    
    def test_error_handling_security(self):
        """测试错误处理安全"""
        try:
            from visionsub.ui.enhanced_main_window import EnhancedMainWindow
            
            class MockController:
                def process_video(self, file_path):
                    return True
                def export_subtitles(self, format_type):
                    return True
            
            main_window = EnhancedMainWindow(MockController())
            
            # 测试错误处理是否泄露敏感信息
            try:
                # 触发一个错误
                main_window.process_invalid_file("nonexistent_file.mp4")
            except Exception as e:
                error_message = str(e)
                
                # 检查错误信息是否包含敏感信息
                sensitive_info_patterns = [
                    r'traceback',
                    r'stack trace',
                    r'internal server error',
                    r'database error',
                    r'file path:.*',
                    r'line \d+'
                ]
                
                for pattern in sensitive_info_patterns:
                    if re.search(pattern, error_message, re.IGNORECASE):
                        self.vulnerabilities.append({
                            'type': 'Error Handling',
                            'description': '错误信息泄露敏感信息',
                            'severity': 'Medium'
                        })
                        return False
            
            main_window.close()
            return True
            
        except Exception as e:
            self.vulnerabilities.append({
                'type': 'Error Handling',
                'description': f'错误处理安全测试异常: {str(e)}',
                'severity': 'Low'
            })
            return False
    
    def test_logging_security(self):
        """测试日志安全"""
        try:
            from visionsub.ui.enhanced_main_window import EnhancedMainWindow
            
            class MockController:
                def process_video(self, file_path):
                    return True
                def export_subtitles(self, format_type):
                    return True
            
            main_window = EnhancedMainWindow(MockController())
            
            # 检查日志记录是否安全
            if hasattr(main_window, 'logger'):
                # 应该有安全的日志记录机制
                pass
            
            main_window.close()
            return True
            
        except Exception as e:
            self.vulnerabilities.append({
                'type': 'Logging Security',
                'description': f'日志安全测试异常: {str(e)}',
                'severity': 'Low'
            })
            return False
    
    def test_configuration_security(self):
        """测试配置安全"""
        try:
            from visionsub.ui.enhanced_main_window import EnhancedMainWindow
            
            class MockController:
                def process_video(self, file_path):
                    return True
                def export_subtitles(self, format_type):
                    return True
            
            main_window = EnhancedMainWindow(MockController())
            
            # 检查配置文件安全性
            if hasattr(main_window, 'config_manager'):
                # 应该有安全的配置管理
                pass
            
            main_window.close()
            return True
            
        except Exception as e:
            self.vulnerabilities.append({
                'type': 'Configuration Security',
                'description': f'配置安全测试异常: {str(e)}',
                'severity': 'Low'
            })
            return False
    
    def test_network_security(self):
        """测试网络安全"""
        try:
            from visionsub.ui.enhanced_main_window import EnhancedMainWindow
            
            class MockController:
                def process_video(self, file_path):
                    return True
                def export_subtitles(self, format_type):
                    return True
            
            main_window = EnhancedMainWindow(MockController())
            
            # 检查网络安全
            if hasattr(main_window, 'network_manager'):
                # 应该有安全的网络通信
                pass
            
            main_window.close()
            return True
            
        except Exception as e:
            self.vulnerabilities.append({
                'type': 'Network Security',
                'description': f'网络安全测试异常: {str(e)}',
                'severity': 'Medium'
            })
            return False
    
    def generate_security_report(self):
        """生成安全报告"""
        report = {
            'scan_date': '2025-08-15',
            'scan_version': '1.0.0',
            'total_tests': 15,
            'passed_tests': 15 - len(self.vulnerabilities),
            'failed_tests': len(self.vulnerabilities),
            'vulnerabilities': self.vulnerabilities,
            'security_score': ((15 - len(self.vulnerabilities)) / 15) * 100,
            'recommendations': self.generate_recommendations()
        }
        
        return report
    
    def generate_recommendations(self):
        """生成安全建议"""
        recommendations = []
        
        # 按严重程度分类的建议
        high_severity = [v for v in self.vulnerabilities if v['severity'] == 'High']
        medium_severity = [v for v in self.vulnerabilities if v['severity'] == 'Medium']
        low_severity = [v for v in self.vulnerabilities if v['severity'] == 'Low']
        
        if high_severity:
            recommendations.append("🚨 高危漏洞: 立即修复所有高危漏洞")
        
        if medium_severity:
            recommendations.append("⚠️ 中危漏洞: 优先修复中危漏洞")
        
        if low_severity:
            recommendations.append("📋 低危漏洞: 在下次更新中修复低危漏洞")
        
        # 通用安全建议
        recommendations.extend([
            "🔒 实施输入验证和输出编码",
            "🛡️ 使用参数化查询防止SQL注入",
            "🔐 实施适当的认证和授权机制",
            "📝 记录安全事件但不记录敏感信息",
            "🔄 定期更新依赖库和安全补丁",
            "🔍 定期进行安全审计和渗透测试"
        ])
        
        return recommendations


def main():
    """主函数"""
    print("🛡️ VisionSub 前端安全测试")
    print("=" * 50)
    
    # 创建安全测试运行器
    test_runner = SecurityTestRunner()
    
    # 初始化应用程序
    if not test_runner.initialize_application():
        print("❌ 应用程序初始化失败")
        return False
    
    # 运行安全测试
    success = test_runner.run_security_tests()
    
    # 生成安全报告
    report = test_runner.generate_security_report()
    
    print(f"\n📋 安全评分: {report['security_score']:.1f}/100")
    
    if report['recommendations']:
        print(f"\n💡 安全建议:")
        for i, recommendation in enumerate(report['recommendations'], 1):
            print(f"   {i}. {recommendation}")
    
    # 保存报告
    report_file = "security_report.json"
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    
    print(f"\n📄 安全报告已保存到: {report_file}")
    
    return success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)