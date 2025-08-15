"""
代码注入防护模块
提供防止代码注入攻击的安全功能
"""

import ast
import re
import logging
from typing import Any, Dict, List, Optional, Set
import subprocess
import os
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

class CodeInjectionDetector:
    """代码注入检测器"""
    
    def __init__(self):
        self.dangerous_functions = {
            'eval', 'exec', 'compile', '__import__', 'getattr', 'setattr',
            'delattr', 'globals', 'locals', 'vars', 'dir', 'help',
            'input', 'open', 'file', 'exit', 'quit', 'reload'
        }
        
        self.dangerous_modules = {
            'os', 'sys', 'subprocess', 'shutil', 'glob', 'tempfile',
            'ctypes', 'marshal', 'pickle', 'shelve', 'dbm', 'sqlite3',
            'socket', 'urllib', 'requests', 'httplib', 'ftplib',
            'cmd', 'code', 'codeop', 'dis', 'parser', 'symbol', 'token',
            'tokenize', 'keyword', 'tabnanny', 'py_compile', 'compileall'
        }
        
        self.dangerous_patterns = [
            r'__import__\s*\(',
            r'eval\s*\(',
            r'exec\s*\(',
            r'compile\s*\(',
            r'subprocess\.',
            r'os\.(system|popen|spawn|exec)',
            r'sys\.(executable|path)',
            r'open\s*\(',
            r'file\s*\(',
            r'input\s*\(',
            r'getattr\s*\(',
            r'setattr\s*\(',
            r'delattr\s*\(',
            r'globals\s*\(',
            r'locals\s*\(',
            r'vars\s*\(',
            r'dir\s*\(',
            r'help\s*\(',
            r'exit\s*\(',
            r'quit\s*\(',
            r'reload\s*\(',
            r'marshal\.',
            r'pickle\.',
            r'shelve\.',
            r'dbm\.',
            r'sqlite3\.',
            r'socket\.',
            r'urllib\.',
            r'requests\.',
            r'httplib\.',
            r'ftplib\.',
            r'cmd\.',
            r'code\.',
            r'codeop\.',
            r'dis\.',
            r'parser\.',
            r'symbol\.',
            r'token\.',
            r'tokenize\.',
            r'keyword\.',
            r'tabnanny\.',
            r'py_compile\.',
            r'compileall\.',
            r'ctypes\.',
        ]
        
        self.whitelist_functions = {
            'print', 'len', 'str', 'int', 'float', 'bool', 'list', 'dict',
            'tuple', 'set', 'frozenset', 'range', 'enumerate', 'zip',
            'map', 'filter', 'sorted', 'reversed', 'any', 'all', 'sum',
            'min', 'max', 'abs', 'round', 'pow', 'divmod', 'chr', 'ord',
            'bin', 'hex', 'oct', 'format', 'bytes', 'bytearray', 'memoryview'
        }
    
    def scan_code(self, code: str, filename: str = "unknown") -> Dict[str, Any]:
        """扫描代码中的安全风险"""
        try:
            result = {
                'filename': filename,
                'is_safe': True,
                'warnings': [],
                'errors': [],
                'suggestions': []
            }
            
            # 语法检查
            try:
                ast.parse(code)
            except SyntaxError as e:
                result['errors'].append(f"语法错误: {e}")
                result['is_safe'] = False
                return result
            
            # 模式匹配检测
            pattern_issues = self._detect_dangerous_patterns(code)
            result['warnings'].extend(pattern_issues)
            
            # AST 分析
            ast_issues = self._analyze_ast(code)
            result['warnings'].extend(ast_issues)
            
            # 动态分析（如果安全的话）
            if not result['errors'] and not result['warnings']:
                dynamic_issues = self._dynamic_analysis(code)
                result['warnings'].extend(dynamic_issues)
            
            # 生成建议
            result['suggestions'] = self._generate_suggestions(result)
            
            # 判断安全性
            if result['errors'] or result['warnings']:
                result['is_safe'] = False
            
            return result
            
        except Exception as e:
            logger.error(f"代码扫描失败: {e}")
            return {
                'filename': filename,
                'is_safe': False,
                'errors': [f"扫描失败: {e}"],
                'warnings': [],
                'suggestions': []
            }
    
    def _detect_dangerous_patterns(self, code: str) -> List[str]:
        """检测危险模式"""
        issues = []
        lines = code.split('\n')
        
        for i, line in enumerate(lines, 1):
            for pattern in self.dangerous_patterns:
                if re.search(pattern, line, re.IGNORECASE):
                    # 检查是否在注释中
                    stripped_line = line.strip()
                    if stripped_line.startswith('#'):
                        continue
                    
                    # 检查是否在字符串中
                    if self._is_in_string(line, pattern):
                        continue
                    
                    issues.append(f"第{i}行: 检测到危险模式 '{pattern}'")
        
        return issues
    
    def _is_in_string(self, line: str, pattern: str) -> bool:
        """检查模式是否在字符串中"""
        try:
            # 简单的字符串检查
            quote_positions = []
            in_string = False
            quote_char = None
            
            for i, char in enumerate(line):
                if char in ['"', "'"] and (i == 0 or line[i-1] != '\\'):
                    if not in_string:
                        in_string = True
                        quote_char = char
                        quote_positions.append(i)
                    elif char == quote_char:
                        in_string = False
                        quote_char = None
                        quote_positions.append(i)
            
            # 检查模式是否在字符串中
            pattern_pos = line.find(pattern)
            for i in range(0, len(quote_positions), 2):
                if i + 1 < len(quote_positions):
                    start, end = quote_positions[i], quote_positions[i+1]
                    if start <= pattern_pos <= end:
                        return True
            
            return False
            
        except Exception:
            return False
    
    def _analyze_ast(self, code: str) -> List[str]:
        """AST 分析"""
        issues = []
        
        try:
            tree = ast.parse(code)
            
            for node in ast.walk(tree):
                # 检查函数调用
                if isinstance(node, ast.Call):
                    func_name = self._get_function_name(node.func)
                    
                    if func_name in self.dangerous_functions:
                        issues.append(f"检测到危险函数调用: {func_name}")
                    
                    # 检查模块访问
                    if isinstance(node.func, ast.Attribute):
                        module_name = self._get_module_name(node.func)
                        if module_name in self.dangerous_modules:
                            issues.append(f"检测到危险模块访问: {module_name}.{func_name}")
                
                # 检查导入
                elif isinstance(node, ast.Import):
                    for alias in node.names:
                        if alias.name in self.dangerous_modules:
                            issues.append(f"检测到危险模块导入: {alias.name}")
                
                elif isinstance(node, ast.ImportFrom):
                    if node.module in self.dangerous_modules:
                        issues.append(f"检测到危险模块导入: from {node.module} import ...")
                
                # 检查属性访问
                elif isinstance(node, ast.Attribute):
                    if node.attr in self.dangerous_functions:
                        issues.append(f"检测到危险属性访问: {node.attr}")
                
                # 检查名称
                elif isinstance(node, ast.Name):
                    if node.id in self.dangerous_functions:
                        issues.append(f"检测到危险名称: {node.id}")
            
        except Exception as e:
            issues.append(f"AST 分析失败: {e}")
        
        return issues
    
    def _get_function_name(self, node) -> str:
        """获取函数名称"""
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            return node.attr
        elif isinstance(node, ast.Call):
            return self._get_function_name(node.func)
        return "unknown"
    
    def _get_module_name(self, node) -> str:
        """获取模块名称"""
        if isinstance(node, ast.Attribute):
            value = node.value
            if isinstance(value, ast.Name):
                return value.id
            elif isinstance(value, ast.Attribute):
                return self._get_module_name(value)
        return "unknown"
    
    def _dynamic_analysis(self, code: str) -> List[str]:
        """动态分析（安全执行）"""
        issues = []
        
        try:
            # 创建安全的执行环境
            safe_globals = {
                '__builtins__': {
                    name: getattr(__builtins__, name) 
                    for name in self.whitelist_functions 
                    if hasattr(__builtins__, name)
                }
            }
            
            # 尝试执行代码（限制时间）
            import signal
            
            def timeout_handler(signum, frame):
                raise TimeoutError("代码执行超时")
            
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(5)  # 5秒超时
            
            try:
                exec(code, safe_globals, {})
            except Exception as e:
                issues.append(f"动态执行失败: {e}")
            finally:
                signal.alarm(0)  # 取消超时
            
        except Exception as e:
            issues.append(f"动态分析失败: {e}")
        
        return issues
    
    def _generate_suggestions(self, result: Dict) -> List[str]:
        """生成安全建议"""
        suggestions = []
        
        if result['errors']:
            suggestions.append("请修复代码中的错误")
        
        if result['warnings']:
            suggestions.append("请移除或替换危险的函数调用")
            suggestions.append("考虑使用白名单中的安全函数")
            suggestions.append("对用户输入进行严格的验证和清理")
        
        if any('eval' in warning for warning in result['warnings']):
            suggestions.append("避免使用 eval() 函数，考虑使用 ast.literal_eval() 替代")
        
        if any('exec' in warning for warning in result['warnings']):
            suggestions.append("避免使用 exec() 函数，考虑使用其他安全的替代方案")
        
        if any('subprocess' in warning for warning in result['warnings']):
            suggestions.append("避免使用 subprocess 模块，考虑使用更安全的替代方案")
        
        return suggestions

class SafeEvaluator:
    """安全代码评估器"""
    
    def __init__(self):
        self.detector = CodeInjectionDetector()
        self.safe_functions = {
            'abs', 'all', 'any', 'ascii', 'bin', 'bool', 'bytearray', 'bytes',
            'chr', 'complex', 'dict', 'dir', 'divmod', 'enumerate', 'filter',
            'float', 'format', 'frozenset', 'hash', 'hex', 'int', 'iter',
            'len', 'list', 'map', 'max', 'memoryview', 'min', 'next',
            'oct', 'ord', 'pow', 'range', 'repr', 'reversed', 'round',
            'set', 'slice', 'sorted', 'str', 'sum', 'tuple', 'zip'
        }
    
    def safe_eval(self, expression: str, context: Optional[Dict] = None) -> Any:
        """安全地评估表达式"""
        try:
            # 扫描代码
            scan_result = self.detector.scan_code(expression)
            if not scan_result['is_safe']:
                raise ValueError(f"表达式不安全: {scan_result['warnings']}")
            
            # 创建安全环境
            safe_globals = {
                '__builtins__': {
                    name: getattr(__builtins__, name) 
                    for name in self.safe_functions 
                    if hasattr(__builtins__, name)
                }
            }
            
            if context:
                safe_globals.update(context)
            
            # 使用 ast.literal_eval 如果可能
            try:
                return ast.literal_eval(expression)
            except (ValueError, SyntaxError):
                # 回退到安全的 eval
                return eval(expression, safe_globals, {})
            
        except Exception as e:
            logger.error(f"安全评估失败: {e}")
            raise ValueError(f"安全评估失败: {e}")
    
    def safe_exec(self, code: str, context: Optional[Dict] = None) -> Dict[str, Any]:
        """安全地执行代码"""
        try:
            # 扫描代码
            scan_result = self.detector.scan_code(code)
            if not scan_result['is_safe']:
                raise ValueError(f"代码不安全: {scan_result['warnings']}")
            
            # 创建安全环境
            safe_globals = {
                '__builtins__': {
                    name: getattr(__builtins__, name) 
                    for name in self.safe_functions 
                    if hasattr(__builtins__, name)
                }
            }
            
            if context:
                safe_globals.update(context)
            
            # 执行代码
            safe_locals = {}
            exec(code, safe_globals, safe_locals)
            
            return safe_locals
            
        except Exception as e:
            logger.error(f"安全执行失败: {e}")
            raise ValueError(f"安全执行失败: {e}")

class InputSanitizer:
    """输入清理器"""
    
    def __init__(self):
        self.html_escape_map = {
            '&': '&amp;',
            '<': '&lt;',
            '>': '&gt;',
            '"': '&quot;',
            "'": '&#x27;',
            '/': '&#x2F;'
        }
        
        self.sql_injection_patterns = [
            r'(union|select|insert|update|delete|drop|alter|create|exec)\s+',
            r'(\s|^)(or|and)\s+\w+\s*=',
            r'(\s|^)(or|and)\s+\d+\s*[<>=]',
            r'(\s|^)(or|and)\s+[\'"][^\'"]*[\'"]\s*=',
            r';\s*(drop|delete|update|insert)',
            r'--\s*$',
            r'/\*.*\*/',
            r'xp_cmdshell',
            r'exec\s*\(',
            r'waitfor\s+delay',
            r'sleep\s*\(',
        ]
        
        self.command_injection_patterns = [
            r'[;&|`\$]',
            r'\|\|',
            r'&&',
            r';\s*',
            r'`\s*.*\s*`',
            r'\$\s*\(',
            r'<\s*[^>]*>',
            r'>\s*[^>]*',
            r'2>\s*[^>]*',
            r'>>\s*[^>]*',
        ]
    
    def sanitize_html(self, text: str) -> str:
        """清理 HTML 输入"""
        for char, replacement in self.html_escape_map.items():
            text = text.replace(char, replacement)
        return text
    
    def sanitize_sql(self, text: str) -> str:
        """清理 SQL 输入"""
        # 移除危险的 SQL 模式
        for pattern in self.sql_injection_patterns:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE)
        
        # 转义单引号
        text = text.replace("'", "''")
        
        return text
    
    def sanitize_command(self, text: str) -> str:
        """清理命令行输入"""
        # 移除危险的命令注入模式
        for pattern in self.command_injection_patterns:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE)
        
        # 移除控制字符
        text = re.sub(r'[\x00-\x1f\x7f]', '', text)
        
        return text
    
    def sanitize_path(self, path: str) -> str:
        """清理路径输入"""
        # 移除路径遍历
        path = path.replace('..', '').replace('//', '/')
        
        # 移除危险字符
        dangerous_chars = ['<', '>', ':', '"', '|', '?', '*']
        for char in dangerous_chars:
            path = path.replace(char, '')
        
        return path
    
    def sanitize_filename(self, filename: str) -> str:
        """清理文件名"""
        # 移除路径分隔符
        filename = filename.replace('/', '').replace('\\', '')
        
        # 移除危险字符
        dangerous_chars = ['<', '>', ':', '"', '|', '?', '*']
        for char in dangerous_chars:
            filename = filename.replace(char, '')
        
        # 限制长度
        if len(filename) > 255:
            name, ext = os.path.splitext(filename)
            filename = name[:255-len(ext)] + ext
        
        return filename
    
    def validate_json(self, json_str: str) -> bool:
        """验证 JSON 字符串"""
        try:
            import json
            json.loads(json_str)
            return True
        except (ValueError, TypeError):
            return False
    
    def sanitize_xml(self, xml_str: str) -> str:
        """清理 XML 输入"""
        # 移除 XML 注释
        xml_str = re.sub(r'<!--.*?-->', '', xml_str, flags=re.DOTALL)
        
        # 移除 CDATA 部分
        xml_str = re.sub(r'<!\[CDATA\[.*?\]\]>', '', xml_str, flags=re.DOTALL)
        
        # 移除 DOCTYPE 声明
        xml_str = re.sub(r'<!DOCTYPE.*?>', '', xml_str, flags=re.DOTALL)
        
        return xml_str

# 全局实例
code_injection_detector = CodeInjectionDetector()
safe_evaluator = SafeEvaluator()
input_sanitizer = InputSanitizer()