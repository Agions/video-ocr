"""
资源泄露防护模块
提供防止资源泄露的安全功能
"""

import os
import sys
import gc
import threading
import time
import logging
import psutil
import resource
from typing import Dict, List, Optional, Set, Any, Callable
from contextlib import contextmanager
from pathlib import Path
import weakref
import tracemalloc

logger = logging.getLogger(__name__)

class ResourceMonitor:
    """资源监控器"""
    
    def __init__(self):
        self.process = psutil.Process()
        self.initial_memory = None
        self.peak_memory = None
        self.start_time = time.time()
        self.monitoring = False
        self.monitor_thread = None
        self.thresholds = {
            'memory_mb': 1024,  # 1GB
            'cpu_percent': 80,   # 80%
            'open_files': 1000,  # 1000 files
            'threads': 50,       # 50 threads
        }
        
    def start_monitoring(self):
        """开始监控"""
        if self.monitoring:
            return
        
        self.initial_memory = self.process.memory_info().rss / 1024 / 1024
        self.peak_memory = self.initial_memory
        self.monitoring = True
        
        # 启动监控线程
        self.monitor_thread = threading.Thread(target=self._monitor_resources)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        
        # 启动内存跟踪
        tracemalloc.start()
        
        logger.info("资源监控已启动")
    
    def stop_monitoring(self):
        """停止监控"""
        self.monitoring = False
        
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        
        tracemalloc.stop()
        logger.info("资源监控已停止")
    
    def _monitor_resources(self):
        """监控资源使用"""
        while self.monitoring:
            try:
                # 监控内存
                memory_info = self.process.memory_info()
                current_memory = memory_info.rss / 1024 / 1024
                self.peak_memory = max(self.peak_memory, current_memory)
                
                # 检查内存阈值
                if current_memory > self.thresholds['memory_mb']:
                    logger.warning(f"内存使用超过阈值: {current_memory:.1f}MB > {self.thresholds['memory_mb']}MB")
                
                # 监控CPU
                cpu_percent = self.process.cpu_percent()
                if cpu_percent > self.thresholds['cpu_percent']:
                    logger.warning(f"CPU使用超过阈值: {cpu_percent}% > {self.thresholds['cpu_percent']}%")
                
                # 监控打开的文件
                try:
                    open_files = len(self.process.open_files())
                    if open_files > self.thresholds['open_files']:
                        logger.warning(f"打开文件数超过阈值: {open_files} > {self.thresholds['open_files']}")
                except Exception:
                    pass
                
                # 监控线程数
                thread_count = self.process.num_threads()
                if thread_count > self.thresholds['threads']:
                    logger.warning(f"线程数超过阈值: {thread_count} > {self.thresholds['threads']}")
                
                time.sleep(5)  # 每5秒检查一次
                
            except Exception as e:
                logger.error(f"资源监控失败: {e}")
                time.sleep(10)
    
    def get_resource_usage(self) -> Dict[str, Any]:
        """获取资源使用情况"""
        try:
            memory_info = self.process.memory_info()
            cpu_percent = self.process.cpu_percent()
            
            usage = {
                'memory_current_mb': memory_info.rss / 1024 / 1024,
                'memory_peak_mb': self.peak_memory,
                'memory_initial_mb': self.initial_memory,
                'cpu_percent': cpu_percent,
                'open_files': len(self.process.open_files()),
                'threads': self.process.num_threads(),
                'uptime_seconds': time.time() - self.start_time,
                'handle_count': self.process.num_handles() if hasattr(self.process, 'num_handles') else 0,
            }
            
            # 获取内存分配统计
            if tracemalloc.is_tracing():
                current, peak = tracemalloc.get_traced_memory()
                usage['memory_allocated_mb'] = current / 1024 / 1024
                usage['memory_peak_allocated_mb'] = peak / 1024 / 1024
            
            return usage
            
        except Exception as e:
            logger.error(f"获取资源使用情况失败: {e}")
            return {}
    
    def set_threshold(self, resource: str, value: Any):
        """设置资源阈值"""
        if resource in self.thresholds:
            self.thresholds[resource] = value
            logger.info(f"设置 {resource} 阈值为: {value}")
        else:
            logger.warning(f"未知的资源类型: {resource}")
    
    def generate_memory_report(self) -> str:
        """生成内存报告"""
        if not tracemalloc.is_tracing():
            return "内存跟踪未启用"
        
        try:
            snapshot = tracemalloc.take_snapshot()
            top_stats = snapshot.statistics('lineno')
            
            report = []
            report.append("=" * 60)
            report.append("VisionSub 内存使用报告")
            report.append("=" * 60)
            report.append("")
            
            # 总体统计
            current, peak = tracemalloc.get_traced_memory()
            report.append(f"当前内存分配: {current / 1024 / 1024:.1f} MB")
            report.append(f"峰值内存分配: {peak / 1024 / 1024:.1f} MB")
            report.append("")
            
            # 前10个内存分配
            report.append("Top 10 内存分配:")
            for i, stat in enumerate(top_stats[:10], 1):
                report.append(f"{i:2d}. {stat}")
            
            return "\n".join(report)
            
        except Exception as e:
            logger.error(f"生成内存报告失败: {e}")
            return f"生成内存报告失败: {e}"

class ResourceManager:
    """资源管理器"""
    
    def __init__(self):
        self.resources = {}
        self.weak_refs = {}
        self.cleanup_callbacks = {}
        self.monitor = ResourceMonitor()
        self.lock = threading.RLock()
        
    def register_resource(self, resource_id: str, resource: Any, 
                         cleanup_callback: Optional[Callable] = None):
        """注册资源"""
        with self.lock:
            self.resources[resource_id] = resource
            
            # 创建弱引用
            if cleanup_callback:
                def callback(ref):
                    self.cleanup_resource(resource_id)
                
                weak_ref = weakref.ref(resource, callback)
                self.weak_refs[resource_id] = weak_ref
            
            # 注册清理回调
            if cleanup_callback:
                self.cleanup_callbacks[resource_id] = cleanup_callback
            
            logger.debug(f"注册资源: {resource_id}")
    
    def cleanup_resource(self, resource_id: str):
        """清理资源"""
        with self.lock:
            if resource_id in self.resources:
                resource = self.resources.pop(resource_id, None)
                
                # 执行清理回调
                if resource_id in self.cleanup_callbacks:
                    try:
                        self.cleanup_callbacks[resource_id](resource)
                        del self.cleanup_callbacks[resource_id]
                    except Exception as e:
                        logger.error(f"资源清理回调失败: {e}")
                
                # 移除弱引用
                if resource_id in self.weak_refs:
                    del self.weak_refs[resource_id]
                
                logger.debug(f"清理资源: {resource_id}")
    
    def cleanup_all_resources(self):
        """清理所有资源"""
        with self.lock:
            for resource_id in list(self.resources.keys()):
                self.cleanup_resource(resource_id)
            
            # 强制垃圾回收
            gc.collect()
            
            logger.info("清理所有资源完成")
    
    def get_resource_info(self) -> Dict[str, Any]:
        """获取资源信息"""
        with self.lock:
            return {
                'total_resources': len(self.resources),
                'resource_ids': list(self.resources.keys()),
                'weak_refs': len(self.weak_refs),
                'cleanup_callbacks': len(self.cleanup_callbacks),
            }
    
    def is_resource_registered(self, resource_id: str) -> bool:
        """检查资源是否已注册"""
        return resource_id in self.resources
    
    @contextmanager
    def resource_context(self, resource_id: str, resource_factory: Callable, 
                        cleanup_callback: Optional[Callable] = None):
        """资源上下文管理器"""
        resource = None
        try:
            # 创建资源
            resource = resource_factory()
            
            # 注册资源
            self.register_resource(resource_id, resource, cleanup_callback)
            
            yield resource
            
        except Exception as e:
            logger.error(f"资源上下文错误: {e}")
            raise
        finally:
            # 清理资源
            if resource:
                self.cleanup_resource(resource_id)

class FileResourceTracker:
    """文件资源跟踪器"""
    
    def __init__(self):
        self.open_files = {}
        self.file_limits = {
            'max_open_files': 1000,
            'max_file_size_mb': 100,
            'max_total_size_mb': 1000,
        }
        self.lock = threading.RLock()
        
    def track_file(self, file_path: str, file_obj: Any):
        """跟踪文件"""
        with self.lock:
            file_id = id(file_obj)
            self.open_files[file_id] = {
                'path': file_path,
                'obj': file_obj,
                'open_time': time.time(),
                'size': self._get_file_size(file_path),
            }
            
            # 检查限制
            self._check_file_limits()
            
            logger.debug(f"跟踪文件: {file_path}")
    
    def untrack_file(self, file_obj: Any):
        """取消跟踪文件"""
        with self.lock:
            file_id = id(file_obj)
            if file_id in self.open_files:
                file_info = self.open_files.pop(file_id)
                logger.debug(f"取消跟踪文件: {file_info['path']}")
    
    def _get_file_size(self, file_path: str) -> int:
        """获取文件大小"""
        try:
            return os.path.getsize(file_path)
        except Exception:
            return 0
    
    def _check_file_limits(self):
        """检查文件限制"""
        # 检查打开文件数量
        if len(self.open_files) > self.file_limits['max_open_files']:
            logger.warning(f"打开文件数超过限制: {len(self.open_files)} > {self.file_limits['max_open_files']}")
        
        # 检查单个文件大小
        for file_id, file_info in self.open_files.items():
            if file_info['size'] > self.file_limits['max_file_size_mb'] * 1024 * 1024:
                logger.warning(f"文件大小超过限制: {file_info['path']} ({file_info['size']} bytes)")
        
        # 检查总文件大小
        total_size = sum(info['size'] for info in self.open_files.values())
        if total_size > self.file_limits['max_total_size_mb'] * 1024 * 1024:
            logger.warning(f"总文件大小超过限制: {total_size} bytes")
    
    def cleanup_old_files(self, max_age_seconds: int = 3600):
        """清理旧文件"""
        with self.lock:
            current_time = time.time()
            cleaned_files = []
            
            for file_id, file_info in list(self.open_files.items()):
                if current_time - file_info['open_time'] > max_age_seconds:
                    # 尝试关闭文件
                    try:
                        if hasattr(file_info['obj'], 'close'):
                            file_info['obj'].close()
                        cleaned_files.append(file_info['path'])
                        del self.open_files[file_id]
                    except Exception as e:
                        logger.error(f"关闭文件失败: {file_info['path']} - {e}")
            
            if cleaned_files:
                logger.info(f"清理旧文件: {len(cleaned_files)} 个")
    
    def get_file_info(self) -> Dict[str, Any]:
        """获取文件信息"""
        with self.lock:
            return {
                'open_files_count': len(self.open_files),
                'total_size_mb': sum(info['size'] for info in self.open_files.values()) / 1024 / 1024,
                'oldest_file_age': min((time.time() - info['open_time'] for info in self.open_files.values()), default=0),
                'file_list': [info['path'] for info in self.open_files.values()],
            }

class ThreadResourceTracker:
    """线程资源跟踪器"""
    
    def __init__(self):
        self.threads = {}
        self.thread_limits = {
            'max_threads': 50,
            'max_thread_lifetime': 3600,  # 1小时
        }
        self.lock = threading.RLock()
        
    def track_thread(self, thread_id: int, thread: threading.Thread):
        """跟踪线程"""
        with self.lock:
            self.threads[thread_id] = {
                'thread': thread,
                'start_time': time.time(),
                'name': thread.name,
                'daemon': thread.daemon,
            }
            
            # 检查限制
            self._check_thread_limits()
            
            logger.debug(f"跟踪线程: {thread.name}")
    
    def untrack_thread(self, thread_id: int):
        """取消跟踪线程"""
        with self.lock:
            if thread_id in self.threads:
                thread_info = self.threads.pop(thread_id)
                logger.debug(f"取消跟踪线程: {thread_info['name']}")
    
    def _check_thread_limits(self):
        """检查线程限制"""
        # 检查线程数量
        if len(self.threads) > self.thread_limits['max_threads']:
            logger.warning(f"线程数超过限制: {len(self.threads)} > {self.thread_limits['max_threads']}")
        
        # 检查线程生命周期
        current_time = time.time()
        for thread_id, thread_info in list(self.threads.items()):
            if current_time - thread_info['start_time'] > self.thread_limits['max_thread_lifetime']:
                logger.warning(f"线程运行时间过长: {thread_info['name']} ({current_time - thread_info['start_time']:.1f}s)")
    
    def cleanup_dead_threads(self):
        """清理死亡线程"""
        with self.lock:
            dead_threads = []
            
            for thread_id, thread_info in list(self.threads.items()):
                if not thread_info['thread'].is_alive():
                    dead_threads.append(thread_id)
                    del self.threads[thread_id]
            
            if dead_threads:
                logger.info(f"清理死亡线程: {len(dead_threads)} 个")
    
    def get_thread_info(self) -> Dict[str, Any]:
        """获取线程信息"""
        with self.lock:
            return {
                'active_threads': len(self.threads),
                'thread_names': [info['name'] for info in self.threads.values()],
                'daemon_threads': sum(1 for info in self.threads.values() if info['daemon']),
                'oldest_thread_age': min((time.time() - info['start_time'] for info in self.threads.values()), default=0),
            }

class ResourceLeakDetector:
    """资源泄露检测器"""
    
    def __init__(self):
        self.resource_manager = ResourceManager()
        self.file_tracker = FileResourceTracker()
        self.thread_tracker = ThreadResourceTracker()
        self.monitor = ResourceMonitor()
        self.detection_enabled = False
        
    def start_detection(self):
        """开始泄露检测"""
        if self.detection_enabled:
            return
        
        self.detection_enabled = True
        self.monitor.start_monitoring()
        
        # 启动定期检测
        detection_thread = threading.Thread(target=self._periodic_detection)
        detection_thread.daemon = True
        detection_thread.start()
        
        logger.info("资源泄露检测已启动")
    
    def stop_detection(self):
        """停止泄露检测"""
        self.detection_enabled = False
        self.monitor.stop_monitoring()
        logger.info("资源泄露检测已停止")
    
    def _periodic_detection(self):
        """定期检测"""
        while self.detection_enabled:
            try:
                # 检测文件泄露
                self.file_tracker.cleanup_old_files()
                
                # 检测线程泄露
                self.thread_tracker.cleanup_dead_threads()
                
                # 检测内存泄露
                self._detect_memory_leak()
                
                time.sleep(60)  # 每分钟检测一次
                
            except Exception as e:
                logger.error(f"资源泄露检测失败: {e}")
                time.sleep(120)
    
    def _detect_memory_leak(self):
        """检测内存泄露"""
        try:
            usage = self.monitor.get_resource_usage()
            
            # 检查内存增长
            if usage.get('memory_current_mb', 0) > usage.get('memory_initial_mb', 0) * 1.5:
                logger.warning(f"检测到内存增长: {usage['memory_current_mb']:.1f}MB > {usage['memory_initial_mb']:.1f}MB * 1.5")
                
                # 强制垃圾回收
                collected = gc.collect()
                if collected > 0:
                    logger.info(f"垃圾回收: {collected} 个对象")
                
                # 生成内存报告
                memory_report = self.monitor.generate_memory_report()
                logger.debug(f"内存报告:\n{memory_report}")
                
        except Exception as e:
            logger.error(f"内存泄露检测失败: {e}")
    
    def generate_leak_report(self) -> str:
        """生成泄露报告"""
        report = []
        report.append("=" * 60)
        report.append("VisionSub 资源泄露检测报告")
        report.append("=" * 60)
        report.append("")
        
        # 资源使用情况
        resource_usage = self.monitor.get_resource_usage()
        report.append("资源使用情况:")
        for key, value in resource_usage.items():
            report.append(f"  {key}: {value}")
        report.append("")
        
        # 资源管理器信息
        resource_info = self.resource_manager.get_resource_info()
        report.append("资源管理器信息:")
        for key, value in resource_info.items():
            report.append(f"  {key}: {value}")
        report.append("")
        
        # 文件跟踪信息
        file_info = self.file_tracker.get_file_info()
        report.append("文件跟踪信息:")
        for key, value in file_info.items():
            report.append(f"  {key}: {value}")
        report.append("")
        
        # 线程跟踪信息
        thread_info = self.thread_tracker.get_thread_info()
        report.append("线程跟踪信息:")
        for key, value in thread_info.items():
            report.append(f"  {key}: {value}")
        report.append("")
        
        # 内存报告
        memory_report = self.monitor.generate_memory_report()
        report.append("内存报告:")
        report.append(memory_report)
        
        return "\n".join(report)

# 全局实例
resource_manager = ResourceManager()
file_tracker = FileResourceTracker()
thread_tracker = ThreadResourceTracker()
resource_monitor = ResourceMonitor()
leak_detector = ResourceLeakDetector()

# 便捷函数
def track_file(file_path: str, file_obj: Any):
    """跟踪文件"""
    file_tracker.track_file(file_path, file_obj)

def untrack_file(file_obj: Any):
    """取消跟踪文件"""
    file_tracker.untrack_file(file_obj)

def track_thread(thread: threading.Thread):
    """跟踪线程"""
    thread_tracker.track_thread(thread.ident, thread)

def untrack_thread(thread: threading.Thread):
    """取消跟踪线程"""
    thread_tracker.untrack_thread(thread.ident)

@contextmanager
def file_context(file_path: str, mode: str = 'r'):
    """文件上下文管理器"""
    file_obj = None
    try:
        file_obj = open(file_path, mode)
        track_file(file_path, file_obj)
        yield file_obj
    except Exception as e:
        logger.error(f"文件操作失败: {e}")
        raise
    finally:
        if file_obj:
            try:
                file_obj.close()
            except Exception:
                pass
            untrack_file(file_obj)

@contextmanager
def thread_context(target: Callable, *args, **kwargs):
    """线程上下文管理器"""
    thread = None
    try:
        thread = threading.Thread(target=target, args=args, kwargs=kwargs)
        track_thread(thread)
        thread.start()
        yield thread
    except Exception as e:
        logger.error(f"线程操作失败: {e}")
        raise
    finally:
        if thread:
            try:
                thread.join(timeout=5)
            except Exception:
                pass
            untrack_thread(thread)