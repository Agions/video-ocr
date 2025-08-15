#!/usr/bin/env python3
"""
VisionSub Splash Screen
启动画面，改善应用程序启动体验

作者: Agions
版本: 1.0.0
日期: 2025-08-15
"""

import sys
import time
from pathlib import Path
from PyQt6.QtWidgets import QApplication, QSplashScreen, QLabel, QProgressBar, QVBoxLayout, QWidget
from PyQt6.QtCore import QTimer, Qt, QThread, pyqtSignal
from PyQt6.QtGui import QPixmap, QPainter, QColor, QFont

class SplashScreen(QSplashScreen):
    """自定义启动画面"""
    
    def __init__(self):
        # 创建启动画面图像
        self.pixmap = self.create_splash_pixmap()
        super().__init__(self.pixmap)
        
        # 设置属性
        self.setFixedSize(400, 300)
        self.setWindowFlags(Qt.WindowType.SplashScreen | Qt.WindowType.FramelessWindowHint | Qt.WindowType.WindowStaysOnTopHint)
        
        # 进度信息
        self.progress = 0
        self.status_message = "正在启动 VisionSub..."
        
        # 创建进度条
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setFixedHeight(4)
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: none;
                background: rgba(255, 255, 255, 20);
                border-radius: 2px;
            }
            QProgressBar::chunk {
                background: #007ACC;
                border-radius: 2px;
            }
        """)
        
        # 状态标签
        self.status_label = QLabel(self.status_message)
        self.status_label.setStyleSheet("color: white; font-size: 12px;")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        # 定位控件
        self.status_label.setGeometry(50, 220, 300, 20)
        self.progress_bar.setGeometry(50, 250, 300, 4)
    
    def create_splash_pixmap(self) -> QPixmap:
        """创建启动画面图像"""
        pixmap = QPixmap(400, 300)
        pixmap.fill(QColor(30, 30, 30))  # 深色背景
        
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        # 绘制Logo
        center_x = 200
        center_y = 120
        
        # 外圆
        painter.setPen(QColor(0, 122, 204))
        painter.setBrush(QColor(0, 122, 204))
        painter.drawEllipse(center_x - 50, center_y - 50, 100, 100)
        
        # 内圆（深色）
        painter.setPen(QColor(30, 30, 30))
        painter.setBrush(QColor(30, 30, 30))
        painter.drawEllipse(center_x - 40, center_y - 40, 80, 80)
        
        # 播放按钮
        painter.setPen(QColor(86, 156, 214))
        painter.setBrush(QColor(86, 156, 214))
        triangle_points = [
            (center_x + 10, center_y),
            (center_x + 30, center_y + 15),
            (center_x + 10, center_y + 30)
        ]
        painter.drawPolygon(triangle_points)
        
        # 应用名称
        font = QFont("Arial", 16, QFont.Weight.Bold)
        painter.setFont(font)
        painter.setPen(QColor(255, 255, 255))
        painter.drawText(center_x - 80, center_y + 80, 160, 30, Qt.AlignmentFlag.AlignCenter, "VisionSub")
        
        # 版本信息
        font = QFont("Arial", 10)
        painter.setFont(font)
        painter.setPen(QColor(150, 150, 150))
        painter.drawText(center_x - 50, center_y + 100, 100, 20, Qt.AlignmentFlag.AlignCenter, "版本 2.0.0")
        
        painter.end()
        return pixmap
    
    def update_progress(self, value: int, message: str = None):
        """更新进度"""
        self.progress = max(0, min(100, value))
        self.progress_bar.setValue(self.progress)
        
        if message:
            self.status_message = message
            self.status_label.setText(message)
        
        # 重绘
        self.update()
    
    def show_message(self, message: str):
        """显示状态消息"""
        self.status_message = message
        self.status_label.setText(message)
        self.update()


class AppInitializer(QThread):
    """应用程序初始化线程"""
    
    progress_updated = pyqtSignal(int, str)
    initialization_complete = pyqtSignal()
    initialization_failed = pyqtSignal(str)
    
    def __init__(self):
        super().__init__()
        self.app = None
    
    def run(self):
        """运行初始化"""
        try:
            self.progress_updated.emit(10, "初始化应用程序...")
            
            # 创建QApplication（如果还没有）
            if QApplication.instance() is None:
                self.app = QApplication(sys.argv)
                self.app.setApplicationName("VisionSub")
                self.app.setApplicationVersion("2.0.0")
                self.app.setOrganizationName("VisionSub")
            
            self.progress_updated.emit(20, "设置应用程序样式...")
            self.app.setStyle('Fusion')
            
            self.progress_updated.emit(30, "导入模块...")
            # 延迟导入，避免启动时的阻塞
            from visionsub.view_models.main_view_model import MainViewModel
            from visionsub.ui.main_window import MainWindow
            
            self.progress_updated.emit(50, "创建视图模型...")
            view_model = MainViewModel()
            
            self.progress_updated.emit(70, "创建主窗口...")
            main_window = MainWindow(view_model)
            
            self.progress_updated.emit(90, "完成初始化...")
            
            # 存储创建的对象
            self.view_model = view_model
            self.main_window = main_window
            
            self.progress_updated.emit(100, "启动完成!")
            
            # 发送完成信号
            self.initialization_complete.emit()
            
        except Exception as e:
            self.initialization_failed.emit(f"初始化失败: {e}")


def launch_with_splash():
    """使用启动画面启动应用程序"""
    import sys
    
    # 创建QApplication
    app = QApplication(sys.argv)
    
    # 创建启动画面
    splash = SplashScreen()
    splash.show()
    
    # 创建初始化线程
    initializer = AppInitializer()
    
    def on_progress_updated(value, message):
        splash.update_progress(value, message)
        QApplication.processEvents()
    
    def on_initialization_complete():
        # 隐藏启动画面
        splash.finish(initializer.main_window)
        
        # 显示主窗口
        initializer.main_window.show()
        
        # 启动事件循环
        sys.exit(app.exec())
    
    def on_initialization_failed(error_message):
        splash.show_message(f"错误: {error_message}")
        QTimer.singleShot(3000, lambda: splash.close())
        QTimer.singleShot(3000, lambda: sys.exit(1))
    
    # 连接信号
    initializer.progress_updated.connect(on_progress_updated)
    initializer.initialization_complete.connect(on_initialization_complete)
    initializer.initialization_failed.connect(on_initialization_failed)
    
    # 启动初始化线程
    initializer.start()
    
    # 启动事件循环
    return app.exec()


if __name__ == "__main__":
    sys.exit(launch_with_splash())