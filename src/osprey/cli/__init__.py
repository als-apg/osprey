"""Command-line interface for Osprey Framework.

This package provides the unified CLI interface for the framework,
organizing all commands under a single 'osprey' entry point.

Commands:
    - init: Create new projects from templates
    - config: Manage project configuration (show, export, set)
    - deploy: Manage Docker/Podman services
    - chat: Interactive conversation interface
    - generate: Generate capabilities and servers
    - remove: Remove components from project
    - health: Check system health

Architecture:
    Uses Click for command-line parsing with a group-based structure.
    Each command is implemented in its own module for maintainability.
    Commands are lazy-loaded for fast startup time.
"""

from .main import cli, main

__all__ = ["cli", "main"]
