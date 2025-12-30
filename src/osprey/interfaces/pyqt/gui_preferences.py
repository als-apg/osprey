"""
GUI Preferences Manager

Manages GUI-specific user preferences that should persist across sessions
but should NOT be written to project configuration files.

This separates user interface preferences from project configuration,
preventing version control pollution and multi-project conflicts.
"""

from pathlib import Path
from typing import Any

import yaml

from osprey.utils.logger import get_logger

logger = get_logger("gui_preferences")


class GUIPreferences:
    """
    Manages GUI-specific preferences stored in user's home directory.

    These preferences are user-specific and persist across GUI sessions,
    but are completely separate from project configurations.

    Storage location: ~/.osprey/gui_preferences.yml
    """

    def __init__(self):
        """Initialize GUI preferences manager."""
        # Store preferences in user's home directory
        self.preferences_dir = Path.home() / ".osprey"
        self.preferences_path = self.preferences_dir / "gui_preferences.yml"

        # Default preferences
        self.defaults = {
            # GUI Display Settings
            "use_persistent_conversations": True,
            "conversation_storage_mode": "json",
            "redirect_output_to_gui": True,
            "suppress_terminal_output": False,
            "group_system_messages": True,
            "enable_routing_feedback": True,
            # Memory Monitoring Settings
            "memory_monitor_enabled": True,
            "memory_warning_threshold_mb": 500,
            "memory_critical_threshold_mb": 1000,
            "memory_check_interval_seconds": 5,
        }

        # Load preferences from file
        self.preferences = self._load_preferences()

    def _load_preferences(self) -> dict[str, Any]:
        """
        Load GUI preferences from user's home directory.

        Returns:
            Dictionary of preferences (uses defaults if file doesn't exist)
        """
        try:
            if not self.preferences_path.exists():
                logger.info("No GUI preferences file found, using defaults")
                return self.defaults.copy()

            with open(self.preferences_path) as f:
                loaded = yaml.safe_load(f) or {}

            # Merge with defaults (in case new preferences were added)
            preferences = self.defaults.copy()
            preferences.update(loaded)

            logger.info(f"Loaded GUI preferences from {self.preferences_path}")
            return preferences

        except Exception as e:
            logger.error(f"Failed to load GUI preferences: {e}, using defaults")
            return self.defaults.copy()

    def save_preferences(self) -> bool:
        """
        Save GUI preferences to user's home directory.

        Returns:
            True if successful, False otherwise
        """
        try:
            # Ensure directory exists
            self.preferences_dir.mkdir(parents=True, exist_ok=True)

            # Write preferences
            with open(self.preferences_path, "w") as f:
                yaml.dump(self.preferences, f, default_flow_style=False, sort_keys=False, indent=2)

            logger.info(f"Saved GUI preferences to {self.preferences_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to save GUI preferences: {e}")
            return False

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a preference value.

        Args:
            key: Preference key
            default: Default value if key not found

        Returns:
            Preference value or default
        """
        return self.preferences.get(key, default)

    def set(self, key: str, value: Any) -> None:
        """
        Set a preference value.

        Args:
            key: Preference key
            value: Preference value
        """
        self.preferences[key] = value

    def update(self, updates: dict[str, Any]) -> None:
        """
        Update multiple preferences at once.

        Args:
            updates: Dictionary of preference updates
        """
        self.preferences.update(updates)

    def get_all(self) -> dict[str, Any]:
        """
        Get all preferences.

        Returns:
            Dictionary of all preferences
        """
        return self.preferences.copy()

    def reset_to_defaults(self) -> None:
        """Reset all preferences to defaults."""
        self.preferences = self.defaults.copy()
        logger.info("Reset GUI preferences to defaults")
