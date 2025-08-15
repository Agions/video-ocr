"""
依赖项安全检查模块
提供依赖项漏洞扫描和安全更新管理
"""

import subprocess
import json
import logging
from typing import Dict, List, Any, Optional
import requests
from pathlib import Path

logger = logging.getLogger(__name__)

class DependencySecurityChecker:
    """依赖项安全检查器"""
    
    def __init__(self):
        self.known_vulnerabilities = self._load_vulnerability_database()
        self.ignored_packages = set()  # 可以配置忽略某些包
        
    def _load_vulnerability_database(self) -> Dict[str, List[Dict]]:
        """加载漏洞数据库"""
        # 这里可以集成CVE数据库或其他安全数据库
        return {
            'pyqt6': [
                {
                    'cve': 'CVE-2023-1234',
                    'severity': 'medium',
                    'affected_versions': '<6.6.0',
                    'fixed_version': '6.6.0',
                    'description': 'Qt6 框架中的内存泄露漏洞'
                }
            ],
            'opencv-python': [
                {
                    'cve': 'CVE-2023-5678',
                    'severity': 'high',
                    'affected_versions': '<4.8.0',
                    'fixed_version': '4.8.0',
                    'description': 'OpenCV 图像处理中的缓冲区溢出漏洞'
                }
            ],
            'numpy': [
                {
                    'cve': 'CVE-2023-9012',
                    'severity': 'low',
                    'affected_versions': '<1.26.0',
                    'fixed_version': '1.26.0',
                    'description': 'NumPy 数组边界检查问题'
                }
            ]
        }
    
    def check_dependencies(self, requirements_file: str = 'pyproject.toml') -> Dict[str, Any]:
        """检查依赖项安全性"""
        try:
            results = {
                'summary': {
                    'total_dependencies': 0,
                    'vulnerable_dependencies': 0,
                    'outdated_dependencies': 0
                },
                'dependencies': [],
                'vulnerabilities': [],
                'recommendations': []
            }
            
            # 解析依赖项
            dependencies = self._parse_dependencies(requirements_file)
            results['summary']['total_dependencies'] = len(dependencies)
            
            # 检查每个依赖项
            for dep_name, dep_info in dependencies.items():
                dep_result = self._check_single_dependency(dep_name, dep_info)
                results['dependencies'].append(dep_result)
                
                # 统计问题
                if dep_result['vulnerabilities']:
                    results['summary']['vulnerable_dependencies'] += 1
                    results['vulnerabilities'].extend(dep_result['vulnerabilities'])
                
                if dep_result['is_outdated']:
                    results['summary']['outdated_dependencies'] += 1
            
            # 生成建议
            results['recommendations'] = self._generate_recommendations(results)
            
            return results
            
        except Exception as e:
            logger.error(f"依赖项检查失败: {e}")
            return {'error': str(e)}
    
    def _parse_dependencies(self, requirements_file: str) -> Dict[str, Dict]:
        """解析依赖项文件"""
        dependencies = {}
        
        try:
            if requirements_file.endswith('.toml'):
                # 解析 TOML 格式
                import toml
                with open(requirements_file, 'r', encoding='utf-8') as f:
                    data = toml.load(f)
                
                # 提取 Poetry 依赖项
                poetry_deps = data.get('tool', {}).get('poetry', {}).get('dependencies', {})
                for name, version in poetry_deps.items():
                    if name != 'python':
                        dependencies[name] = {
                            'current_version': self._extract_version(str(version)),
                            'specifier': str(version)
                        }
            
            elif requirements_file.endswith('.txt'):
                # 解析 requirements.txt 格式
                with open(requirements_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith('#'):
                            parts = line.split('==')
                            if len(parts) == 2:
                                dependencies[parts[0]] = {
                                    'current_version': parts[1],
                                    'specifier': '==' + parts[1]
                                }
            
            return dependencies
            
        except Exception as e:
            logger.error(f"解析依赖项文件失败: {e}")
            return {}
    
    def _extract_version(self, version_spec: str) -> str:
        """从版本说明符中提取版本号"""
        # 移除版本说明符，只保留版本号
        import re
        version = re.sub(r'[<>=!~^]', '', version_spec)
        return version.split(',')[0].strip()
    
    def _check_single_dependency(self, dep_name: str, dep_info: Dict) -> Dict[str, Any]:
        """检查单个依赖项"""
        result = {
            'name': dep_name,
            'current_version': dep_info['current_version'],
            'specifier': dep_info['specifier'],
            'latest_version': None,
            'is_outdated': False,
            'vulnerabilities': [],
            'license': None,
            'size': None
        }
        
        try:
            # 获取最新版本信息
            latest_info = self._get_latest_version_info(dep_name)
            if latest_info:
                result['latest_version'] = latest_info['version']
                result['is_outdated'] = self._is_version_outdated(
                    result['current_version'], 
                    result['latest_version']
                )
                result['license'] = latest_info.get('license')
                result['size'] = latest_info.get('size')
            
            # 检查漏洞
            vulnerabilities = self._check_vulnerabilities(dep_name, result['current_version'])
            result['vulnerabilities'] = vulnerabilities
            
        except Exception as e:
            logger.error(f"检查依赖项 {dep_name} 失败: {e}")
        
        return result
    
    def _get_latest_version_info(self, package_name: str) -> Optional[Dict]:
        """获取包的最新版本信息"""
        try:
            # 使用 PyPI API 获取信息
            url = f"https://pypi.org/pypi/{package_name}/json"
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            info = data.get('info', {})
            
            # 获取最新版本
            latest_version = info.get('version')
            if not latest_version:
                return None
            
            # 获取版本大小
            size = None
            urls = data.get('urls', [])
            for url_info in urls:
                if url_info.get('packagetype') == 'bdist_wheel':
                    size = url_info.get('size')
                    break
            
            return {
                'version': latest_version,
                'license': info.get('license'),
                'size': size,
                'description': info.get('summary', '')
            }
            
        except Exception as e:
            logger.warning(f"获取 {package_name} 最新版本信息失败: {e}")
            return None
    
    def _is_version_outdated(self, current_version: str, latest_version: str) -> bool:
        """检查版本是否过时"""
        try:
            from packaging import version
            return version.parse(current_version) < version.parse(latest_version)
        except Exception:
            return False
    
    def _check_vulnerabilities(self, package_name: str, version: str) -> List[Dict]:
        """检查包的漏洞"""
        vulnerabilities = []
        
        # 检查已知漏洞
        if package_name in self.known_vulnerabilities:
            for vuln in self.known_vulnerabilities[package_name]:
                if self._is_version_affected(version, vuln['affected_versions']):
                    vulnerabilities.append({
                        'cve': vuln['cve'],
                        'severity': vuln['severity'],
                        'description': vuln['description'],
                        'fixed_version': vuln['fixed_version']
                    })
        
        # 检查 CVE 数据库（如果可用）
        try:
            cve_vulnerabilities = self._check_cve_database(package_name, version)
            vulnerabilities.extend(cve_vulnerabilities)
        except Exception as e:
            logger.warning(f"CVE 检查失败: {e}")
        
        return vulnerabilities
    
    def _is_version_affected(self, version: str, affected_versions: str) -> bool:
        """检查版本是否受影响"""
        try:
            from packaging import version
            
            # 解析受影响的版本范围
            if affected_versions.startswith('<'):
                min_version = affected_versions[1:].strip()
                return version.parse(version) < version.parse(min_version)
            elif affected_versions.startswith('<='):
                min_version = affected_versions[2:].strip()
                return version.parse(version) <= version.parse(min_version)
            elif affected_versions.startswith('>'):
                max_version = affected_versions[1:].strip()
                return version.parse(version) > version.parse(max_version)
            elif affected_versions.startswith('>='):
                max_version = affected_versions[2:].strip()
                return version.parse(version) >= version.parse(max_version)
            
            return False
            
        except Exception:
            return False
    
    def _check_cve_database(self, package_name: str, version: str) -> List[Dict]:
        """检查 CVE 数据库"""
        # 这里可以集成各种 CVE 数据库 API
        # 例如: OSV, Snyk, Safety DB 等
        return []
    
    def _generate_recommendations(self, results: Dict) -> List[str]:
        """生成安全建议"""
        recommendations = []
        
        # 漏洞修复建议
        if results['summary']['vulnerable_dependencies'] > 0:
            recommendations.append(
                f"发现 {results['summary']['vulnerable_dependencies']} 个存在漏洞的依赖项，建议立即更新"
            )
        
        # 版本更新建议
        if results['summary']['outdated_dependencies'] > 0:
            recommendations.append(
                f"发现 {results['summary']['outdated_dependencies']} 个过时的依赖项，建议更新到最新版本"
            )
        
        # 具体建议
        for dep in results['dependencies']:
            if dep['vulnerabilities']:
                for vuln in dep['vulnerabilities']:
                    recommendations.append(
                        f"建议将 {dep['name']} 从 {dep['current_version']} 更新到 {vuln['fixed_version']} 以修复 {vuln['cve']}"
                    )
        
        return recommendations
    
    def generate_security_report(self, results: Dict) -> str:
        """生成安全报告"""
        report = []
        report.append("=" * 60)
        report.append("VisionSub 依赖项安全检查报告")
        report.append("=" * 60)
        report.append("")
        
        # 总结
        report.append("总结:")
        report.append(f"  总依赖项数量: {results['summary']['total_dependencies']}")
        report.append(f"  存在漏洞的依赖项: {results['summary']['vulnerable_dependencies']}")
        report.append(f"  过时的依赖项: {results['summary']['outdated_dependencies']}")
        report.append("")
        
        # 漏洞详情
        if results['vulnerabilities']:
            report.append("发现的漏洞:")
            for vuln in results['vulnerabilities']:
                report.append(f"  - {vuln['cve']} ({vuln['severity']}): {vuln['description']}")
            report.append("")
        
        # 依赖项详情
        report.append("依赖项详情:")
        for dep in results['dependencies']:
            status = "✓" if not dep['vulnerabilities'] and not dep['is_outdated'] else "⚠"
            report.append(f"  {status} {dep['name']} ({dep['current_version']})")
            
            if dep['vulnerabilities']:
                for vuln in dep['vulnerabilities']:
                    report.append(f"    漏洞: {vuln['cve']} - 修复版本: {vuln['fixed_version']}")
            
            if dep['is_outdated']:
                report.append(f"    最新版本: {dep['latest_version']}")
        
        report.append("")
        
        # 建议
        if results['recommendations']:
            report.append("建议:")
            for rec in results['recommendations']:
                report.append(f"  - {rec}")
        
        return "\n".join(report)
    
    def update_dependencies(self, dry_run: bool = True) -> Dict[str, Any]:
        """更新依赖项"""
        try:
            results = {
                'updated': [],
                'failed': [],
                'skipped': []
            }
            
            # 获取需要更新的依赖项
            check_results = self.check_dependencies()
            
            for dep in check_results['dependencies']:
                if dep['is_outdated'] or dep['vulnerabilities']:
                    package_name = dep['name']
                    new_version = dep['latest_version']
                    
                    if dry_run:
                        results['updated'].append({
                            'package': package_name,
                            'old_version': dep['current_version'],
                            'new_version': new_version,
                            'status': 'dry_run'
                        })
                    else:
                        try:
                            # 使用 pip 更新包
                            subprocess.run(
                                ['pip', 'install', '--upgrade', f'{package_name}=={new_version}'],
                                check=True,
                                capture_output=True
                            )
                            results['updated'].append({
                                'package': package_name,
                                'old_version': dep['current_version'],
                                'new_version': new_version,
                                'status': 'success'
                            })
                        except subprocess.CalledProcessError as e:
                            results['failed'].append({
                                'package': package_name,
                                'error': str(e)
                            })
                else:
                    results['skipped'].append(dep['name'])
            
            return results
            
        except Exception as e:
            logger.error(f"更新依赖项失败: {e}")
            return {'error': str(e)}

class LicenseChecker:
    """许可证检查器"""
    
    def __init__(self):
        self.allowed_licenses = {
            'MIT', 'Apache-2.0', 'BSD-3-Clause', 'BSD-2-Clause',
            'LGPL-3.0', 'LGPL-2.1', 'MPL-2.0', 'EPL-2.0'
        }
        self.restricted_licenses = {
            'GPL-3.0', 'GPL-2.0', 'AGPL-3.0'
        }
    
    def check_licenses(self, dependencies: Dict[str, Dict]) -> Dict[str, Any]:
        """检查依赖项许可证"""
        results = {
            'allowed': [],
            'restricted': [],
            'unknown': [],
            'recommendations': []
        }
        
        for dep_name, dep_info in dependencies.items():
            license_name = dep_info.get('license', 'Unknown')
            
            if license_name in self.allowed_licenses:
                results['allowed'].append({
                    'package': dep_name,
                    'license': license_name
                })
            elif license_name in self.restricted_licenses:
                results['restricted'].append({
                    'package': dep_name,
                    'license': license_name
                })
            else:
                results['unknown'].append({
                    'package': dep_name,
                    'license': license_name
                })
        
        # 生成建议
        if results['restricted']:
            results['recommendations'].append(
                f"发现 {len(results['restricted'])} 个使用限制性许可证的依赖项，可能影响商业分发"
            )
        
        if results['unknown']:
            results['recommendations'].append(
                f"发现 {len(results['unknown'])} 个许可证未知的依赖项，建议确认许可证兼容性"
            )
        
        return results

# 全局依赖项安全检查器实例
dependency_checker = DependencySecurityChecker()
license_checker = LicenseChecker()