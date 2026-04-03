"""Command-line interface for Osprey Framework.

This package provides the unified CLI interface for the framework,
organizing all commands under a single 'osprey' entry point.

Commands:
    - init: Create new projects from templates
    - config: Manage project configuration (show, export, set)
    - deploy: Manage Docker/Podman services
    - claude: Claude Code integration (install skills, configure)
    - web: Launch web terminal interface
    - health: Check system health
    - channel-finder: Interactive channel search
Architecture:
    Uses Click for command-line parsing with a group-based structure.
    Each command is implemented in its own module for maintainability.
    Commands are lazy-loaded for fast startup time.
"""

from .main import cli, main

__all__ = ["cli", "main"]
