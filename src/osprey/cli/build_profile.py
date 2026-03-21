"""Build profile data model and loader.

A build profile is a YAML file that describes how to assemble a
facility-specific assistant project from an OSPREY template plus
overlay files, config overrides, and MCP server definitions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from osprey.errors import BuildProfileError


@dataclass
class McpServerDef:
    """Definition of an MCP server to inject into a built project."""

    command: str
    args: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)
    permissions: dict[str, list[str]] = field(default_factory=dict)
    # permissions: {"allow": ["tool1"], "ask": ["tool2"]}


@dataclass
class BuildProfile:
    """Complete build profile parsed from YAML."""

    name: str
    base_template: str = "control_assistant"
    provider: str | None = None
    model: str | None = None
    channel_finder_mode: str | None = None
    config: dict[str, Any] = field(default_factory=dict)
    overlay: dict[str, str] = field(default_factory=dict)
    mcp_servers: dict[str, McpServerDef] = field(default_factory=dict)

    def validate(self, profile_dir: Path) -> None:
        """Validate profile consistency. Raises BuildProfileError with all issues."""
        errors: list[str] = []

        if not self.name:
            errors.append("Profile 'name' is required")

        # Validate overlay source paths exist
        for src, _dst in self.overlay.items():
            src_path = profile_dir / src
            if not src_path.exists():
                errors.append(f"Overlay source not found: {src} (resolved: {src_path})")

        # Path traversal guard on overlay destinations
        for _src, dst in self.overlay.items():
            normalized = Path(dst)
            if normalized.is_absolute() or ".." in normalized.parts:
                errors.append(f"Overlay destination must be relative without '..': {dst}")

        # Validate MCP server definitions
        for name, server in self.mcp_servers.items():
            if not server.command:
                errors.append(f"MCP server '{name}' missing 'command'")

        if errors:
            raise BuildProfileError(
                "Build profile validation failed:\n  - " + "\n  - ".join(errors)
            )


def load_profile(path: Path) -> BuildProfile:
    """Load a build profile from YAML.

    Args:
        path: Path to the profile YAML file.

    Returns:
        Parsed and validated BuildProfile.

    Raises:
        BuildProfileError: If the file is invalid or validation fails.
    """
    if not path.exists():
        raise BuildProfileError(f"Profile not found: {path}")

    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as e:
        raise BuildProfileError(f"Invalid YAML in {path}: {e}") from e

    if not isinstance(raw, dict):
        raise BuildProfileError(f"Profile must be a YAML mapping, got {type(raw).__name__}")

    profile = _parse_profile(raw)
    profile_dir = path.parent

    profile.validate(profile_dir)
    return profile


def _parse_profile(raw: dict[str, Any]) -> BuildProfile:
    """Parse raw YAML dict into a BuildProfile."""
    mcp_servers: dict[str, McpServerDef] = {}
    for name, sdef in raw.get("mcp_servers", {}).items():
        if not isinstance(sdef, dict):
            raise BuildProfileError(f"MCP server '{name}' must be a mapping")
        perms = sdef.get("permissions", {})
        mcp_servers[name] = McpServerDef(
            command=sdef.get("command", ""),
            args=sdef.get("args", []),
            env=sdef.get("env", {}),
            permissions={
                "allow": perms.get("allow", []),
                "ask": perms.get("ask", []),
            },
        )

    return BuildProfile(
        name=raw.get("name", ""),
        base_template=raw.get("base_template", "control_assistant"),
        provider=raw.get("provider"),
        model=raw.get("model"),
        channel_finder_mode=raw.get("channel_finder_mode"),
        config=raw.get("config", {}),
        overlay=raw.get("overlay", {}),
        mcp_servers=mcp_servers,
    )
