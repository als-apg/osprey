"""OSPREY Type Registry — single source of truth for artifact, data, and tool types.

Provides canonical definitions (key, label, color) for every type used in
the artifact gallery.  Python tools, ``submit_response`` validation, and the
``/api/type-registry`` endpoint all reference this module.  The gallery JS
fetches the registry at startup and overlays its lookup tables, so adding a
new type here is all that's needed — no JS or CSS changes required.

Icons (SVGs) remain in ``gallery.js`` where they belong; this module only
stores display metadata (label + color).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class TypeDef:
    """A single type definition: display label + badge colour."""

    key: str
    label: str
    color: str  # hex colour for badges


# Artifact types
ARTIFACT_TYPES: dict[str, TypeDef] = {
    "plot_html": TypeDef("plot_html", "Plotly", "#8b5cf6"),
    "plot_png": TypeDef("plot_png", "Matplotlib", "#e8c9a0"),
    "table_html": TypeDef("table_html", "Table", "#22c55e"),
    "html": TypeDef("html", "HTML", "#c084fc"),
    "markdown": TypeDef("markdown", "Markdown", "#fbbf24"),
    "json": TypeDef("json", "JSON", "#3b82f6"),
    "image": TypeDef("image", "Image", "#a78bfa"),
    "text": TypeDef("text", "Text", "#8b9ab5"),
    "file": TypeDef("file", "File", "#506380"),
    "notebook": TypeDef("notebook", "Notebook", "#e879f9"),
    "dashboard_html": TypeDef("dashboard_html", "Dashboard", "#06b6d4"),
}

# Categories
CATEGORIES: dict[str, TypeDef] = {
    "archiver_data": TypeDef("archiver_data", "Archiver Data", "#2563eb"),
    "channel_values": TypeDef("channel_values", "Channel Values", "#14b8a6"),
    "write_results": TypeDef("write_results", "Write Results", "#e8c9a0"),
    "code_output": TypeDef("code_output", "Code Output", "#c084fc"),
    "visualization": TypeDef("visualization", "Visualization", "#fb923c"),
    "dashboard": TypeDef("dashboard", "Dashboard", "#06b6d4"),
    "document": TypeDef("document", "Document", "#a3e635"),
    "screenshot": TypeDef("screenshot", "Screenshot", "#a78bfa"),
    "agent_response": TypeDef("agent_response", "Agent Response", "#f472b6"),
    "channel_addresses": TypeDef("channel_addresses", "Channel Addresses", "#2dd4bf"),
    "channel_finder": TypeDef("channel_finder", "Channel Finder", "#10b981"),
    "logbook_research": TypeDef("logbook_research", "Logbook Research", "#e879f9"),
    "search_results": TypeDef("search_results", "Search Results", "#fb7185"),
    "notebook": TypeDef("notebook", "Notebook", "#d946ef"),
    "user_artifact": TypeDef("user_artifact", "User Artifact", "#94a3b8"),
    "diagnostic_report": TypeDef("diagnostic_report", "Diagnostic Report", "#ef4444"),
}

# Tool types
TOOL_TYPES: dict[str, TypeDef] = {
    "channel_read": TypeDef("channel_read", "Channel Read", "#4fd1c5"),
    "channel_write": TypeDef("channel_write", "Channel Write", "#e8c9a0"),
    "archiver_read": TypeDef("archiver_read", "Archiver Read", "#3b82f6"),
    "execute": TypeDef("execute", "Python Execute", "#c084fc"),
    "channel_find": TypeDef("channel_find", "Channel Find", "#22c55e"),
    # memory_save and memory_recall removed — replaced by Claude Code native memory
    "ariel_search": TypeDef("ariel_search", "ARIEL Search", "#e879f9"),
    "screenshot_capture": TypeDef("screenshot_capture", "Screenshot Capture", "#a78bfa"),
    "facility_description": TypeDef("facility_description", "Facility Description", "#fbbf24"),
    "artifact_save": TypeDef("artifact_save", "Artifact Save", "#94a3b8"),
    "artifact_delete": TypeDef("artifact_delete", "Artifact Delete", "#94a3b8"),
    "artifact_export": TypeDef("artifact_export", "Artifact Export", "#94a3b8"),
    "artifact_focus": TypeDef("artifact_focus", "Artifact Focus", "#60a5fa"),
    "submit_response": TypeDef("submit_response", "Submit Response", "#f472b6"),
    "channel-finder": TypeDef("channel-finder", "Channel Finder", "#2dd4bf"),
    "logbook-deep-research": TypeDef("logbook-deep-research", "Logbook Deep Research", "#e879f9"),
    "logbook-search": TypeDef("logbook-search", "Logbook Search", "#f0abfc"),
    "create_static_plot": TypeDef("create_static_plot", "Static Plot", "#fb923c"),
    "create_interactive_plot": TypeDef("create_interactive_plot", "Interactive Plot", "#38bdf8"),
    "create_dashboard": TypeDef("create_dashboard", "Dashboard", "#06b6d4"),
    "create_document": TypeDef("create_document", "Document", "#a3e635"),
}


def get_artifact_types() -> dict[str, TypeDef]:
    """Return the canonical artifact type definitions."""
    return dict(ARTIFACT_TYPES)


def get_tool_types() -> dict[str, TypeDef]:
    """Return the canonical tool type definitions."""
    return dict(TOOL_TYPES)


def get_categories() -> dict[str, TypeDef]:
    """Return the canonical category definitions for the unified artifact system."""
    return dict(CATEGORIES)


def valid_category_keys() -> set[str]:
    """Return the set of valid category strings for validation."""
    return set(CATEGORIES)


def _typedef_to_dict(td: TypeDef) -> dict[str, str]:
    """Serialise a TypeDef to a plain dict (label + color only)."""
    return {"label": td.label, "color": td.color}


def registry_to_api_dict() -> dict[str, Any]:
    """Return the full registry as a JSON-serialisable dict for the API."""
    return {
        "artifact_types": {k: _typedef_to_dict(v) for k, v in ARTIFACT_TYPES.items()},
        "tool_types": {k: _typedef_to_dict(v) for k, v in TOOL_TYPES.items()},
        "categories": {k: _typedef_to_dict(v) for k, v in CATEGORIES.items()},
    }
