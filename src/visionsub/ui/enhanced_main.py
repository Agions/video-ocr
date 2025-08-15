"""
Enhanced Main Application Entry Point with Modern UI/UX and Security Features
"""
import logging
import sys
from pathlib import Path
from typing import Optional

from PyQt6.QtCore import Qt, QTimer, QSettings
from PyQt6.QtGui import QIcon, QFont
from PyQt6.QtWidgets import (
    QApplication, QSplashScreen, QMessageBox, 
    QStyleFactory, QSystemTrayIcon, QMenu, QWidget
)

# Add the src directory to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from visionsub.models.config import AppConfig
from visionsub.ui.theme_system import ThemeManager, get_theme_manager, initialize_theme_manager
from visionsub.ui.enhanced_main_window import EnhancedMainWindow
from visionsub.view_models.main_view_model import MainViewModel
from visionsub.core.logging_system import initialize_logging
from visionsub.core.config_manager import ConfigManager

# Configure logging
logger = logging.getLogger(__name__)


class EnhancedApplication(QApplication):
    """Enhanced application with modern features and security"""
    
    def __init__(self, argv):
        super().__init__(argv)
        
        # Set application properties
        self.setApplicationName("VisionSub")
        self.setApplicationVersion("2.0.0")
        self.setOrganizationName("VisionSub")
        self.setOrganizationDomain("visionsub.com")
        
        # Initialize components
        self.theme_manager = None
        self.config_manager = None
        self.main_window = None
        self.view_model = None
        self.splash_screen = None
        self.system_tray_icon = None
        
        # Setup application
        self.setup_application()
        
    def setup_application(self):
        """Setup application components"""
        try:
            # Show splash screen
            self.show_splash_screen()
            
            # Initialize logging
            self.setup_logging()
            
            # Initialize configuration
            self.setup_configuration()
            
            # Initialize theme system
            self.setup_theme_system()
            
            # Initialize view model
            self.setup_view_model()
            
            # Initialize main window
            self.setup_main_window()
            
            # Initialize system tray
            self.setup_system_tray()
            
            # Setup global event handling
            self.setup_event_handling()
            
            # Close splash screen
            self.close_splash_screen()
            
            logger.info("Enhanced application initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to setup application: {e}")
            self.show_error_dialog(f"初始化失败: {e}")
            sys.exit(1)
            
    def show_splash_screen(self):
        """Show splash screen"""
        try:
            splash_pixmap = self.style().standardIcon(
                QStyle.StandardPixmap.SP_ComputerIcon
            ).pixmap(400, 200)
            
            self.splash_screen = QSplashScreen(splash_pixmap)
            self.splash_screen.showMessage(
                "正在启动 VisionSub...",
                Qt.AlignmentFlag.AlignBottom | Qt.AlignmentFlag.AlignCenter,
                Qt.GlobalColor.white
            )
            self.splash_screen.show()
            
            # Process events to show splash screen
            self.processEvents()
            
        except Exception as e:
            logger.warning(f"Failed to show splash screen: {e}")
            
    def close_splash_screen(self):
        """Close splash screen"""
        if self.splash_screen:
            self.splash_screen.finish(self.main_window)
            self.splash_screen = None
            
    def setup_logging(self):
        """Setup logging system"""
        try:
            # Get logging config from app config
            logging_config = getattr(self.config, 'logging', None)
            if logging_config:
                initialize_logging(logging_config)
            else:
                # Use default logging
                logging.basicConfig(level=logging.INFO)
            logger.info("Logging system initialized")
            
        except Exception as e:
            print(f"Failed to setup logging: {e}")
            
    def setup_configuration(self):
        """Setup configuration system"""
        try:
            self.config_manager = ConfigManager()
            self.config = self.config_manager.load_config()
            
            logger.info("Configuration system initialized")
            
        except Exception as e:
            logger.error(f"Failed to setup configuration: {e}")
            # Use default configuration
            self.config = AppConfig()
            
    def setup_theme_system(self):
        """Setup theme system"""
        try:
            self.theme_manager = initialize_theme_manager()
            
            # Set application theme from config
            theme_name = self.config.ui.theme
            self.theme_manager.set_theme(theme_name)
            
            # Set application font
            font = QFont(self.config.ui.font_family)
            font.setPointSize(self.config.ui.font_size)
            self.setFont(font)
            
            logger.info(f"Theme system initialized with theme: {theme_name}")
            
        except Exception as e:
            logger.error(f"Failed to setup theme system: {e}")
            
    def setup_view_model(self):
        """Setup view model"""
        try:
            self.view_model = MainViewModel()
            
            # Configure view model with app settings
            self.view_model.set_config(self.config.processing)
            
            logger.info("View model initialized")
            
        except Exception as e:
            logger.error(f"Failed to setup view model: {e}")
            raise
            
    def setup_main_window(self):
        """Setup main window"""
        try:
            self.main_window = EnhancedMainWindow(self.view_model)
            
            # Set window properties from config
            self.main_window.setWindowTitle("VisionSub - 视频OCR字幕提取工具")
            self.main_window.setGeometry(
                100, 100,
                self.config.ui.window_size[0],
                self.config.ui.window_size[1]
            )
            
            # Show main window
            self.main_window.show()
            
            logger.info("Main window initialized")
            
        except Exception as e:
            logger.error(f"Failed to setup main window: {e}")
            raise
            
    def setup_system_tray(self):
        """Setup system tray icon"""
        try:
            if QSystemTrayIcon.isSystemTrayAvailable():
                self.system_tray_icon = QSystemTrayIcon(self)
                
                # Set icon
                icon = self.style().standardIcon(QStyle.StandardPixmap.SP_ComputerIcon)
                self.system_tray_icon.setIcon(icon)
                
                # Create context menu
                tray_menu = QMenu()
                
                show_action = tray_menu.addAction("显示窗口")
                show_action.triggered.connect(self.show_main_window)
                
                hide_action = tray_menu.addAction("隐藏窗口")
                hide_action.triggered.connect(self.hide_main_window)
                
                tray_menu.addSeparator()
                
                exit_action = tray_menu.addAction("退出")
                exit_action.triggered.connect(self.quit_application)
                
                self.system_tray_icon.setContextMenu(tray_menu)
                
                # Connect double-click
                self.system_tray_icon.activated.connect(self.on_tray_activated)
                
                # Show tray icon
                self.system_tray_icon.show()
                
                logger.info("System tray initialized")
                
        except Exception as e:
            logger.warning(f"Failed to setup system tray: {e}")
            
    def setup_event_handling(self):
        """Setup global event handling"""
        try:
            # Handle application focus changes
            self.focusChanged.connect(self.on_focus_changed)
            
            # Setup cleanup timer
            self.cleanup_timer = QTimer()
            self.cleanup_timer.timeout.connect(self.perform_cleanup)
            self.cleanup_timer.start(300000)  # 5 minutes
            
            logger.info("Event handling initialized")
            
        except Exception as e:
            logger.warning(f"Failed to setup event handling: {e}")
            
    def on_focus_changed(self, old: Optional[QWidget], new: Optional[QWidget]):
        """Handle application focus changes"""
        if new is None:
            # Application lost focus
            logger.debug("Application lost focus")
        else:
            # Application gained focus
            logger.debug("Application gained focus")
            
    def on_tray_activated(self, reason):
        """Handle system tray activation"""
        if reason == QSystemTrayIcon.ActivationReason.DoubleClick:
            if self.main_window.isVisible():
                self.main_window.hide()
            else:
                self.main_window.show()
                self.main_window.raise_()
                self.main_window.activateWindow()
                
    def show_main_window(self):
        """Show main window"""
        if self.main_window:
            self.main_window.show()
            self.main_window.raise_()
            self.main_window.activateWindow()
            
    def hide_main_window(self):
        """Hide main window"""
        if self.main_window:
            self.main_window.hide()
            
    def perform_cleanup(self):
        """Perform periodic cleanup"""
        try:
            # Clean up temporary files
            self.cleanup_temp_files()
            
            # Optimize memory usage
            self.optimize_memory()
            
            logger.debug("Cleanup completed")
            
        except Exception as e:
            logger.warning(f"Cleanup failed: {e}")
            
    def cleanup_temp_files(self):
        """Clean up temporary files"""
        try:
            import tempfile
            import os
            
            temp_dir = tempfile.gettempdir()
            app_temp_pattern = "visionsub_*"
            
            for item in Path(temp_dir).glob(app_temp_pattern):
                try:
                    if item.is_file():
                        item.unlink()
                    elif item.is_dir():
                        import shutil
                        shutil.rmtree(item)
                except Exception as e:
                    logger.warning(f"Failed to cleanup temp item {item}: {e}")
                    
        except Exception as e:
            logger.warning(f"Temp file cleanup failed: {e}")
            
    def optimize_memory(self):
        """Optimize memory usage"""
        try:
            import gc
            
            # Force garbage collection
            gc.collect()
            
            # Clear unused caches
            if hasattr(self.view_model, 'clear_cache'):
                self.view_model.clear_cache()
                
        except Exception as e:
            logger.warning(f"Memory optimization failed: {e}")
            
    def show_error_dialog(self, message: str):
        """Show error dialog"""
        QMessageBox.critical(
            None,
            "错误",
            message,
            QMessageBox.StandardButton.Ok
        )
        
    def quit_application(self):
        """Quit application safely"""
        try:
            logger.info("Shutting down application...")
            
            # Save configuration
            if self.config_manager and self.config:
                self.config_manager.save_config(self.config)
                
            # Close main window
            if self.main_window:
                self.main_window.close()
                
            # Hide system tray
            if self.system_tray_icon:
                self.system_tray_icon.hide()
                
            # Quit application
            self.quit()
            
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")
            self.quit()
            
    def event(self, event):
        """Handle application events"""
        try:
            # Handle file open events
            if event.type() == event.Type.FileOpen:
                file_path = event.file()
                if file_path and self.main_window:
                    self.main_window.open_file(file_path)
                return True
                
        except Exception as e:
            logger.warning(f"Failed to handle file open event: {e}")
            
        return super().event(event)


def main():
    """Main application entry point"""
    try:
        # Create application
        app = EnhancedApplication(sys.argv)
        
        # Run application
        return_code = app.exec()
        
        logger.info(f"Application exited with code: {return_code}")
        return return_code
        
    except KeyboardInterrupt:
        logger.info("Application interrupted by user")
        return 0
        
    except Exception as e:
        logger.error(f"Application crashed: {e}")
        return 1


def run_enhanced_gui():
    """Run enhanced GUI application"""
    """Run enhanced GUI application"""
    return main()


if __name__ == "__main__":
    sys.exit(main())