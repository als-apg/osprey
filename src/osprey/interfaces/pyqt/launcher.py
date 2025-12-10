#!/usr/bin/env python3
"""
Launcher script for Osprey Framework PyQt GUI

This script provides a simple entry point for launching the GUI with
proper environment setup and dependency checking.
"""

import sys
import os
from pathlib import Path


def check_dependencies():
    """Check if required dependencies are available."""
    missing_deps = []
    
    try:
        import PyQt5
    except ImportError:
        missing_deps.append("PyQt5")
    
    try:
        import dotenv
    except ImportError:
        missing_deps.append("python-dotenv")
    
    try:
        import osprey
    except ImportError:
        missing_deps.append("osprey-framework")
    
    if missing_deps:
        print("❌ Missing dependencies:")
        for dep in missing_deps:
            print(f"  - {dep}")
        print("\nPlease install missing dependencies:")
        if "PyQt5" in missing_deps:
            print("  pip install PyQt5")
        if "python-dotenv" in missing_deps:
            print("  pip install python-dotenv")
        if "osprey-framework" in missing_deps:
            print("  pip install -e .")
        return False
    
    return True


def check_display():
    """Check if DISPLAY is set for GUI applications (Unix-like systems)."""
    if sys.platform.startswith('linux') or sys.platform == 'darwin':
        if "DISPLAY" not in os.environ and sys.platform.startswith('linux'):
            print("⚠️  Warning: DISPLAY environment variable not set.")
            print("   If you're using SSH, try: ssh -X username@hostname")
            print("   Or set DISPLAY manually: export DISPLAY=:0")
            return False
    return True


def main():
    """Main launcher function."""
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    # Check display for GUI (warning only, not fatal)
    check_display()
    
    # Import and run the GUI
    # The GUI will set up output redirection immediately in its __init__
    try:
        from osprey.interfaces.pyqt.gui import main as gui_main
        
        # Check for config path argument
        config_path = None
        if len(sys.argv) > 1:
            config_path = sys.argv[1]
            
            # Verify the config file exists
            if not Path(config_path).exists():
                print(f"❌ Error: Config file not found: {config_path}")
                print(f"   Please provide a valid config file path")
                sys.exit(1)
        
        # Launch GUI - it sets up output redirection in __init__ before any logging
        gui_main(config_path=config_path)
        
    except Exception as e:
        print(f"❌ Failed to launch GUI: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        # Restore stdout/stderr if they were redirected
        # (in case of early exit before GUI cleanup)
        pass


if __name__ == "__main__":
    main()