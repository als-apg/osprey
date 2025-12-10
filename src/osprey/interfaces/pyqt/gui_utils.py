"""
Shared utilities for PyQt GUI

This module contains common functions used across the GUI to reduce code duplication.
"""

from PyQt5.QtGui import QPalette, QColor
from pathlib import Path
import yaml
from typing import Optional, Dict, Any
from osprey.utils.logger import get_logger

logger = get_logger("gui_utils")


def create_dark_palette() -> QPalette:
    """
    Create a consistent dark color palette for the GUI.
    
    Returns:
        QPalette configured with dark theme colors
    """
    palette = QPalette()
    palette.setColor(QPalette.Window, QColor(45, 45, 48))
    palette.setColor(QPalette.WindowText, QColor(255, 255, 255))
    palette.setColor(QPalette.Base, QColor(30, 30, 30))
    palette.setColor(QPalette.AlternateBase, QColor(45, 45, 48))
    palette.setColor(QPalette.Text, QColor(255, 255, 255))
    palette.setColor(QPalette.Button, QColor(60, 60, 60))
    palette.setColor(QPalette.ButtonText, QColor(255, 255, 255))
    palette.setColor(QPalette.Highlight, QColor(0, 120, 215))
    palette.setColor(QPalette.HighlightedText, QColor(255, 255, 255))
    return palette


def load_config_safe(config_path: str) -> Optional[Dict[str, Any]]:
    """
    Safely load a YAML configuration file with error handling.
    
    Args:
        config_path: Path to the YAML config file
        
    Returns:
        Dictionary containing config data, or None if loading fails
    """
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            return config if config else {}
    except FileNotFoundError:
        logger.warning(f"Config file not found: {config_path}")
        return None
    except yaml.YAMLError as e:
        logger.error(f"YAML parsing error in {config_path}: {e}")
        return None
    except Exception as e:
        logger.error(f"Error loading config from {config_path}: {e}")
        return None


# Shared CSS for help dialog tabs
HELP_DIALOG_CSS = """
<style>
    body { color: #FFFFFF; }
    h2 { color: #00FFFF; }
    h3 { color: #FFD700; }
    h4 { color: #00FF00; }
    a { color: #0078D4; }
    code { background-color: #2D2D30; padding: 2px 4px; color: #00FFFF; }
    li { margin: 5px 0; }
</style>
"""