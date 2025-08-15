#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
VisionSub GUI Launcher
Simple script to launch the VisionSub GUI application
"""

import sys
import os
from pathlib import Path

# Add the src directory to Python path
project_root = Path(__file__).parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

def main():
    """Launch the VisionSub GUI application"""
    print("🚀 Starting VisionSub GUI Application...")
    print("=" * 50)
    
    try:
        # Import and run the GUI application with splash screen
        from splash_launcher import launch_with_splash
        return launch_with_splash()
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Please ensure all dependencies are installed:")
        print("  pip install -e .")
        sys.exit(1)
        
    except Exception as e:
        print(f"❌ Failed to start GUI: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()