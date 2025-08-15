#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
VisionSub GUI Application Entry Point
Professional video OCR subtitle extraction tool with graphical user interface
"""

import sys
import logging
from pathlib import Path

# Add the src directory to Python path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import QTimer

from visionsub.ui.main_window import MainWindow
from visionsub.view_models.main_view_model import MainViewModel


def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('visionsub_gui.log'),
            logging.StreamHandler()
        ]
    )


def main():
    """Main entry point for the GUI application"""
    # Setup logging
    setup_logging()
    logger = logging.getLogger(__name__)
    
    try:
        # Create Qt Application
        app = QApplication(sys.argv)
        app.setApplicationName("VisionSub")
        app.setApplicationVersion("2.0.0")
        app.setOrganizationName("VisionSub")
        
        # Set application style
        app.setStyle('Fusion')
        
        # Create view model and main window
        view_model = MainViewModel()
        main_window = MainWindow(view_model)
        
        # Show main window
        main_window.show()
        
        logger.info("VisionSub GUI application started successfully")
        
        # Start the event loop
        sys.exit(app.exec())
        
    except Exception as e:
        logger.error(f"Failed to start GUI application: {e}")
        print(f"Error starting application: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()