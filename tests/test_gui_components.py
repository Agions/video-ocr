#!/usr/bin/env python3
"""
Test script for VisionSub GUI components
"""
import sys
import os
from pathlib import Path

# Add src to Python path
project_root = Path(__file__).parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

def test_gui_imports():
    """Test that all GUI components can be imported"""
    print("Testing GUI component imports...")
    
    try:
        # Test core UI imports
        from visionsub.ui.main_window import MainWindow
        print("✓ MainWindow imported successfully")
        
        from visionsub.ui.video_player import VideoPlayer
        print("✓ VideoPlayer imported successfully")
        
        from visionsub.ui.subtitle_editor import SubtitleEditorWidget
        print("✓ SubtitleEditorWidget imported successfully")
        
        from visionsub.ui.roi_selection import ROISelection
        print("✓ ROISelection imported successfully")
        
        from visionsub.view_models.main_view_model import MainViewModel
        print("✓ MainViewModel imported successfully")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def test_core_components():
    """Test core components that GUI depends on"""
    print("\nTesting core components...")
    
    try:
        # Test OCR engine
        from visionsub.core.ocr_engine import OCREngineFactory
        print("✓ OCR Engine Factory imported successfully")
        
        # Test ROI manager
        from visionsub.core.roi_manager import ROIManager, ROIType
        print("✓ ROI Manager imported successfully")
        
        # Test text processor
        from visionsub.core.text_processor import TextProcessor
        print("✓ Text Processor imported successfully")
        
        # Test configuration
        from visionsub.core.config_manager import ConfigManager
        print("✓ Config Manager imported successfully")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def test_qt_availability():
    """Test if Qt is available and working"""
    print("\nTesting Qt availability...")
    
    try:
        from PyQt6.QtWidgets import QApplication
        from PyQt6.QtCore import QTimer
        
        # Create a minimal Qt application without showing it
        app = QApplication([])
        print("✓ Qt Application created successfully")
        
        # Test basic Qt functionality
        timer = QTimer()
        timer.setSingleShot(True)
        print("✓ Qt Timer created successfully")
        
        # Clean up
        app.quit()
        print("✓ Qt Application closed successfully")
        
        return True
        
    except ImportError as e:
        print(f"❌ Qt import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Qt error: {e}")
        return False

def main():
    """Run all GUI tests"""
    print("🚀 Starting VisionSub GUI Component Tests")
    print("=" * 50)
    
    tests = [
        test_core_components,
        test_gui_imports,
        test_qt_availability
    ]
    
    results = []
    for test in tests:
        result = test()
        results.append(result)
        print()
    
    if all(results):
        print("🎉 All GUI component tests passed!")
        print("\nGUI Status:")
        print("- ✓ All core components imported successfully")
        print("- ✓ All GUI components imported successfully")
        print("- ✓ Qt framework is working correctly")
        print("- ✓ Ready to launch GUI application")
        
        print("\nTo launch the GUI application:")
        print("  python3 run_gui.py")
        
        return True
    else:
        print("❌ Some GUI component tests failed")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)