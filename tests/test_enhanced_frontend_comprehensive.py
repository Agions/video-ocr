#!/usr/bin/env python3
"""
VisionSub Enhanced UI Components - Quality Assurance Test Suite
专业级前端UI组件质量保证测试套件

作者: Agions
版本: 1.0.0
日期: 2025-08-15
"""

import sys
import os
import unittest
import time
import threading
import tempfile
import shutil
from unittest.mock import Mock, patch, MagicMock
from PyQt6.QtWidgets import QApplication, QMessageBox, QFileDialog
from PyQt6.QtCore import Qt, QTimer, QThread, pyqtSignal
from PyQt6.QtGui import QPixmap, QImage, QColor
from PyQt6.QtTest import QTest
from PyQt6 import QtCore

# 添加项目路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

class TestEnhancedMainWindow(unittest.TestCase):
    """增强主窗口测试套件"""
    
    @classmethod
    def setUpClass(cls):
        """测试类初始化"""
        cls.app = QApplication.instance()
        if cls.app is None:
            cls.app = QApplication(sys.argv)
    
    def setUp(self):
        """每个测试用例初始化"""
        from visionsub.ui.enhanced_main_window import EnhancedMainWindow
        
        # 模拟主控制器
        self.mock_controller = Mock()
        self.mock_controller.process_video.return_value = True
        self.mock_controller.export_subtitles.return_value = True
        
        # 创建主窗口
        self.window = EnhancedMainWindow(self.mock_controller)
    
    def tearDown(self):
        """清理测试环境"""
        self.window.close()
    
    def test_initialization(self):
        """测试主窗口初始化"""
        self.assertIsNotNone(self.window)
        self.assertEqual(self.window.windowTitle(), "VisionSub - 视频OCR字幕提取工具")
        self.assertTrue(self.window.isVisible())
    
    def test_menu_creation(self):
        """测试菜单创建"""
        # 验证文件菜单
        file_menu = self.window.findChild(objectName="file_menu")
        self.assertIsNotNone(file_menu)
        
        # 验证编辑菜单
        edit_menu = self.window.findChild(objectName="edit_menu")
        self.assertIsNotNone(edit_menu)
        
        # 验证帮助菜单
        help_menu = self.window.findChild(objectName="help_menu")
        self.assertIsNotNone(help_menu)
    
    def test_toolbar_creation(self):
        """测试工具栏创建"""
        toolbar = self.window.findChild(objectName="main_toolbar")
        self.assertIsNotNone(toolbar)
        self.assertTrue(toolbar.isVisible())
    
    def test_status_bar_creation(self):
        """测试状态栏创建"""
        status_bar = self.window.statusBar()
        self.assertIsNotNone(status_bar)
        self.assertTrue(status_bar.isVisible())
    
    def test_file_open_action(self):
        """测试文件打开功能"""
        # 模拟文件选择
        with patch('PyQt6.QtWidgets.QFileDialog.getOpenFileName') as mock_dialog:
            mock_dialog.return_value = ("/test/video.mp4", "Video Files (*.mp4 *.avi *.mkv)")
            
            # 触发文件打开动作
            self.window.open_file()
            
            # 验证文件选择对话框被调用
            mock_dialog.assert_called_once()
    
    def test_theme_switching(self):
        """测试主题切换功能"""
        # 测试切换到深色主题
        self.window.switch_theme("dark")
        self.assertEqual(self.window.current_theme, "dark")
        
        # 测试切换到浅色主题
        self.window.switch_theme("light")
        self.assertEqual(self.window.current_theme, "light")
    
    def test_security_status_display(self):
        """测试安全状态显示"""
        # 设置安全状态
        self.window.update_security_status("secure")
        
        # 验证状态栏更新
        status_bar = self.window.statusBar()
        self.assertIn("安全", status_bar.currentMessage())
    
    def test_window_resize(self):
        """测试窗口调整大小"""
        # 测试最小尺寸
        min_size = self.window.minimumSize()
        self.assertGreaterEqual(min_size.width(), 800)
        self.assertGreaterEqual(min_size.height(), 600)
        
        # 测试窗口调整
        self.window.resize(1200, 800)
        self.assertEqual(self.window.size().width(), 1200)
        self.assertEqual(self.window.size().height(), 800)

class TestEnhancedVideoPlayer(unittest.TestCase):
    """增强视频播放器测试套件"""
    
    @classmethod
    def setUpClass(cls):
        """测试类初始化"""
        cls.app = QApplication.instance()
        if cls.app is None:
            cls.app = QApplication(sys.argv)
    
    def setUp(self):
        """每个测试用例初始化"""
        from visionsub.ui.enhanced_video_player import EnhancedVideoPlayer
        
        self.player = EnhancedVideoPlayer()
    
    def tearDown(self):
        """清理测试环境"""
        self.player.close()
    
    def test_player_initialization(self):
        """测试播放器初始化"""
        self.assertIsNotNone(self.player)
        self.assertTrue(self.player.isVisible())
    
    def test_video_controls(self):
        """测试视频控件"""
        # 验证播放/暂停按钮
        play_button = self.player.findChild(objectName="play_button")
        self.assertIsNotNone(play_button)
        
        # 验证进度条
        progress_bar = self.player.findChild(objectName="progress_bar")
        self.assertIsNotNone(progress_bar)
        
        # 验证音量控制
        volume_slider = self.player.findChild(objectName="volume_slider")
        self.assertIsNotNone(volume_slider)
    
    def test_roi_selection(self):
        """测试ROI选择功能"""
        # 模拟ROI选择
        self.player.enable_roi_selection(True)
        self.assertTrue(self.player.roi_selection_enabled)
        
        # 测试ROI设置
        roi_rect = QtCore.QRect(100, 100, 200, 150)
        self.player.set_roi(roi_rect)
        self.assertEqual(self.player.current_roi, roi_rect)
    
    def test_zoom_functionality(self):
        """测试缩放功能"""
        # 测试放大
        self.player.zoom_in()
        self.assertGreater(self.player.zoom_level, 1.0)
        
        # 测试缩小
        self.player.zoom_out()
        self.assertLess(self.player.zoom_level, 1.0)
        
        # 测试重置缩放
        self.player.reset_zoom()
        self.assertEqual(self.player.zoom_level, 1.0)
    
    def test_performance_overlay(self):
        """测试性能覆盖层"""
        # 启用性能覆盖层
        self.player.show_performance_overlay(True)
        self.assertTrue(self.player.performance_overlay_visible)
        
        # 更新性能数据
        perf_data = {
            "fps": 30.0,
            "cpu_usage": 25.5,
            "memory_usage": 512.0,
            "frame_time": 33.3
        }
        self.player.update_performance_data(perf_data)
    
    def test_keyboard_shortcuts(self):
        """测试键盘快捷键"""
        # 测试空格键播放/暂停
        QTest.keyClick(self.player, Qt.Key.Key_Space)
        # 验证播放状态变化
        
        # 测试左右箭头快进/快退
        QTest.keyClick(self.player, Qt.Key.Key_Right)
        QTest.keyClick(self.player, Qt.Key.Key_Left)
        # 验证时间变化

class TestEnhancedSettingsDialog(unittest.TestCase):
    """增强设置对话框测试套件"""
    
    @classmethod
    def setUpClass(cls):
        """测试类初始化"""
        cls.app = QApplication.instance()
        if cls.app is None:
            cls.app = QApplication(sys.argv)
    
    def setUp(self):
        """每个测试用例初始化"""
        from visionsub.ui.enhanced_settings_dialog import EnhancedSettingsDialog
        
        self.dialog = EnhancedSettingsDialog()
    
    def tearDown(self):
        """清理测试环境"""
        self.dialog.close()
    
    def test_dialog_initialization(self):
        """测试对话框初始化"""
        self.assertIsNotNone(self.dialog)
        self.assertEqual(self.dialog.windowTitle(), "设置")
    
    def test_tab_creation(self):
        """测试选项卡创建"""
        # 验证常规选项卡
        general_tab = self.dialog.findChild(objectName="general_tab")
        self.assertIsNotNone(general_tab)
        
        # 验证OCR选项卡
        ocr_tab = self.dialog.findChild(objectName="ocr_tab")
        self.assertIsNotNone(ocr_tab)
        
        # 验证安全选项卡
        security_tab = self.dialog.findChild(objectName="security_tab")
        self.assertIsNotNone(security_tab)
    
    def test_settings_validation(self):
        """测试设置验证"""
        # 测试无效输入
        self.dialog.set_setting("max_file_size", "invalid")
        validation_result = self.dialog.validate_settings()
        self.assertFalse(validation_result["valid"])
        
        # 测试有效输入
        self.dialog.set_setting("max_file_size", "100")
        validation_result = self.dialog.validate_settings()
        self.assertTrue(validation_result["valid"])
    
    def test_theme_preview(self):
        """测试主题预览"""
        # 测试深色主题预览
        self.dialog.preview_theme("dark")
        self.assertEqual(self.dialog.current_preview_theme, "dark")
        
        # 测试浅色主题预览
        self.dialog.preview_theme("light")
        self.assertEqual(self.dialog.current_preview_theme, "light")
    
    def test_settings_export_import(self):
        """测试设置导出导入"""
        # 创建临时文件
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as tmp_file:
            tmp_path = tmp_file.name
        
        try:
            # 导出设置
            self.dialog.export_settings(tmp_path)
            self.assertTrue(os.path.exists(tmp_path))
            
            # 导入设置
            self.dialog.import_settings(tmp_path)
            # 验证设置已更新
            
        finally:
            # 清理临时文件
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
    
    def test_security_indicators(self):
        """测试安全指示器"""
        # 设置安全状态
        self.dialog.update_security_status("secure")
        
        # 验证安全指示器显示
        security_indicator = self.dialog.findChild(objectName="security_indicator")
        self.assertIsNotNone(security_indicator)
        self.assertIn("安全", security_indicator.text())

class TestEnhancedOCRPreview(unittest.TestCase):
    """增强OCR预览测试套件"""
    
    @classmethod
    def setUpClass(cls):
        """测试类初始化"""
        cls.app = QApplication.instance()
        if cls.app is None:
            cls.app = QApplication(sys.argv)
    
    def setUp(self):
        """每个测试用例初始化"""
        from visionsub.ui.enhanced_ocr_preview import EnhancedOCRPreview
        
        self.preview = EnhancedOCRPreview()
    
    def tearDown(self):
        """清理测试环境"""
        self.preview.close()
    
    def test_preview_initialization(self):
        """测试预览初始化"""
        self.assertIsNotNone(self.preview)
        self.assertTrue(self.preview.isVisible())
    
    def test_text_highlighting(self):
        """测试文本高亮显示"""
        # 模拟OCR结果
        ocr_results = [
            {"text": "测试文本1", "confidence": 0.95, "bbox": [10, 10, 100, 30]},
            {"text": "测试文本2", "confidence": 0.75, "bbox": [10, 50, 100, 30]},
            {"text": "测试文本3", "confidence": 0.45, "bbox": [10, 90, 100, 30]}
        ]
        
        # 显示OCR结果
        self.preview.display_ocr_results(ocr_results)
        
        # 验证结果已显示
        self.assertEqual(len(self.preview.ocr_results), 3)
    
    def test_confidence_filtering(self):
        """测试置信度过滤"""
        # 设置置信度阈值
        self.preview.set_confidence_threshold(0.8)
        
        # 验证过滤结果
        filtered_results = self.preview.get_filtered_results()
        for result in filtered_results:
            self.assertGreaterEqual(result["confidence"], 0.8)
    
    def test_search_functionality(self):
        """测试搜索功能"""
        # 模拟OCR结果
        ocr_results = [
            {"text": "搜索测试1", "confidence": 0.9, "bbox": [10, 10, 100, 30]},
            {"text": "搜索测试2", "confidence": 0.8, "bbox": [10, 50, 100, 30]},
            {"text": "其他文本", "confidence": 0.7, "bbox": [10, 90, 100, 30]}
        ]
        self.preview.display_ocr_results(ocr_results)
        
        # 搜索文本
        self.preview.search_text("搜索")
        
        # 验证搜索结果
        search_results = self.preview.get_search_results()
        self.assertEqual(len(search_results), 2)
    
    def test_export_functionality(self):
        """测试导出功能"""
        # 创建临时文件
        with tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as tmp_file:
            tmp_path = tmp_file.name
        
        try:
            # 导出文本
            self.preview.export_text(tmp_path)
            self.assertTrue(os.path.exists(tmp_path))
            
            # 验证文件内容
            with open(tmp_path, 'r', encoding='utf-8') as f:
                content = f.read()
                self.assertIn("OCR结果", content)
                
        finally:
            # 清理临时文件
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
    
    def test_statistics_display(self):
        """测试统计信息显示"""
        # 更新统计信息
        stats = {
            "total_words": 150,
            "average_confidence": 0.85,
            "processing_time": 2.5,
            "language_detected": "zh-CN"
        }
        self.preview.update_statistics(stats)
        
        # 验证统计信息显示
        self.assertEqual(self.preview.statistics, stats)

class TestEnhancedSubtitleEditor(unittest.TestCase):
    """增强字幕编辑器测试套件"""
    
    @classmethod
    def setUpClass(cls):
        """测试类初始化"""
        cls.app = QApplication.instance()
        if cls.app is None:
            cls.app = QApplication(sys.argv)
    
    def setUp(self):
        """每个测试用例初始化"""
        from visionsub.ui.enhanced_subtitle_editor import EnhancedSubtitleEditor
        
        self.editor = EnhancedSubtitleEditor()
    
    def tearDown(self):
        """清理测试环境"""
        self.editor.close()
    
    def test_editor_initialization(self):
        """测试编辑器初始化"""
        self.assertIsNotNone(self.editor)
        self.assertTrue(self.editor.isVisible())
    
    def test_table_model(self):
        """测试表格模型"""
        # 模拟字幕数据
        subtitle_data = [
            {"index": 1, "start_time": "00:00:01", "end_time": "00:00:03", "text": "第一句字幕"},
            {"index": 2, "start_time": "00:00:04", "end_time": "00:00:06", "text": "第二句字幕"}
        ]
        
        # 加载数据
        self.editor.load_subtitles(subtitle_data)
        
        # 验证数据加载
        self.assertEqual(len(self.editor.subtitle_data), 2)
    
    def test_real_time_validation(self):
        """测试实时验证"""
        # 测试无效时间格式
        self.editor.validate_time_format("invalid_time")
        # 验证错误提示
        
        # 测试有效时间格式
        self.editor.validate_time_format("00:01:30")
        # 验证通过
    
    def test_undo_redo_functionality(self):
        """测试撤销/重做功能"""
        # 执行编辑操作
        self.editor.edit_subtitle(0, "text", "修改后的文本")
        
        # 撤销操作
        self.editor.undo()
        # 验证撤销成功
        
        # 重做操作
        self.editor.redo()
        # 验证重做成功
    
    def test_batch_operations(self):
        """测试批量操作"""
        # 选择多个字幕
        indices = [0, 1, 2]
        self.editor.select_subtitles(indices)
        
        # 批量调整时间
        self.editor.batch_adjust_time(1.0)  # 延后1秒
        
        # 验证时间调整
        for i in indices:
            subtitle = self.editor.get_subtitle(i)
            # 验证时间已调整
    
    def test_import_export(self):
        """测试导入导出"""
        # 创建临时文件
        with tempfile.NamedTemporaryFile(suffix='.srt', delete=False) as tmp_file:
            tmp_path = tmp_file.name
        
        try:
            # 导出SRT
            self.editor.export_srt(tmp_path)
            self.assertTrue(os.path.exists(tmp_path))
            
            # 导入SRT
            self.editor.import_srt(tmp_path)
            # 验证导入成功
            
        finally:
            # 清理临时文件
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

class TestPerformanceBenchmark(unittest.TestCase):
    """性能基准测试套件"""
    
    @classmethod
    def setUpClass(cls):
        """测试类初始化"""
        cls.app = QApplication.instance()
        if cls.app is None:
            cls.app = QApplication(sys.argv)
    
    def test_ui_rendering_performance(self):
        """测试UI渲染性能"""
        from visionsub.ui.enhanced_main_window import EnhancedMainWindow
        
        # 创建主窗口
        window = EnhancedMainWindow(Mock())
        
        # 测量渲染时间
        start_time = time.time()
        window.show()
        window.repaint()
        end_time = time.time()
        
        # 验证渲染时间在合理范围内
        render_time = end_time - start_time
        self.assertLess(render_time, 0.1)  # 应该在100ms内完成
        
        window.close()
    
    def test_memory_usage(self):
        """测试内存使用"""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
        
        # 创建多个UI组件
        from visionsub.ui.enhanced_main_window import EnhancedMainWindow
        from visionsub.ui.enhanced_video_player import EnhancedVideoPlayer
        from visionsub.ui.enhanced_settings_dialog import EnhancedSettingsDialog
        
        windows = []
        for _ in range(5):
            window = EnhancedMainWindow(Mock())
            windows.append(window)
        
        # 测量内存增长
        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory
        
        # 验证内存增长在合理范围内
        self.assertLess(memory_increase, 100 * 1024 * 1024)  # 应该小于100MB
        
        # 清理
        for window in windows:
            window.close()
    
    def test_concurrent_operations(self):
        """测试并发操作"""
        from visionsub.ui.enhanced_main_window import EnhancedMainWindow
        
        window = EnhancedMainWindow(Mock())
        
        def simulate_user_interaction():
            """模拟用户交互"""
            for i in range(100):
                window.update_status(f"操作 {i}")
                time.sleep(0.01)
        
        # 创建多个线程模拟并发操作
        threads = []
        for _ in range(3):
            thread = threading.Thread(target=simulate_user_interaction)
            threads.append(thread)
            thread.start()
        
        # 等待所有线程完成
        for thread in threads:
            thread.join()
        
        window.close()

class TestSecurityValidation(unittest.TestCase):
    """安全验证测试套件"""
    
    def test_input_sanitization(self):
        """测试输入净化"""
        from visionsub.ui.enhanced_main_window import EnhancedMainWindow
        
        # 测试恶意输入
        malicious_input = "<script>alert('xss')</script>"
        sanitized = EnhancedMainWindow.sanitize_input(malicious_input)
        
        # 验证脚本标签被移除
        self.assertNotIn("<script>", sanitized)
        self.assertNotIn("</script>", sanitized)
    
    def test_file_validation(self):
        """测试文件验证"""
        from visionsub.ui.enhanced_main_window import EnhancedMainWindow
        
        # 测试允许的文件类型
        valid_files = ["test.mp4", "video.avi", "movie.mkv"]
        for file_path in valid_files:
            self.assertTrue(EnhancedMainWindow.is_valid_video_file(file_path))
        
        # 测试不允许的文件类型
        invalid_files = ["test.exe", "script.js", "malware.py"]
        for file_path in invalid_files:
            self.assertFalse(EnhancedMainWindow.is_valid_video_file(file_path))
    
    def test_path_traversal_protection(self):
        """测试路径遍历保护"""
        from visionsub.ui.enhanced_main_window import EnhancedMainWindow
        
        # 测试路径遍历攻击
        malicious_paths = [
            "../../../etc/passwd",
            "..\\..\\..\\windows\\system32",
            "/etc/passwd",
            "C:\\Windows\\System32"
        ]
        
        for path in malicious_paths:
            is_safe = EnhancedMainWindow.is_safe_path(path)
            self.assertFalse(is_safe)

def run_comprehensive_tests():
    """运行全面测试"""
    # 创建测试套件
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # 添加测试类
    test_classes = [
        TestEnhancedMainWindow,
        TestEnhancedVideoPlayer,
        TestEnhancedSettingsDialog,
        TestEnhancedOCRPreview,
        TestEnhancedSubtitleEditor,
        TestPerformanceBenchmark,
        TestSecurityValidation
    ]
    
    for test_class in test_classes:
        tests = loader.loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    # 运行测试
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # 输出测试结果
    print(f"\n=== 测试结果汇总 ===")
    print(f"运行测试: {result.testsRun}")
    print(f"失败测试: {len(result.failures)}")
    print(f"错误测试: {len(result.errors)}")
    print(f"成功率: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    
    if result.failures:
        print(f"\n=== 失败的测试 ===")
        for test, traceback in result.failures:
            print(f"- {test}: {traceback}")
    
    if result.errors:
        print(f"\n=== 错误的测试 ===")
        for test, traceback in result.errors:
            print(f"- {test}: {traceback}")
    
    return result.wasSuccessful()

if __name__ == "__main__":
    # 运行测试
    success = run_comprehensive_tests()
    sys.exit(0 if success else 1)