"""GUI Tabs for Osprey Framework.

This package contains all tab widgets for the main GUI window.
"""

from osprey.interfaces.pyqt.gui_components.tabs.analytics_tab import AnalyticsTab
from osprey.interfaces.pyqt.gui_components.tabs.llm_details_tab import LLMDetailsTab
from osprey.interfaces.pyqt.gui_components.tabs.memory_tab import MemoryTab
from osprey.interfaces.pyqt.gui_components.tabs.projects_tab import ProjectsTab
from osprey.interfaces.pyqt.gui_components.tabs.system_info_tab import SystemInfoTab
from osprey.interfaces.pyqt.gui_components.tabs.tool_usage_tab import ToolUsageTab

__all__ = [
    "SystemInfoTab",
    "AnalyticsTab",
    "LLMDetailsTab",
    "ToolUsageTab",
    "ProjectsTab",
    "MemoryTab",
]
