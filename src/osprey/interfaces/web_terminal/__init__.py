"""OSPREY Web Terminal Interface.

A browser-based split-pane interface with a real terminal (running Claude Code
via PTY) on the left and a live workspace file viewer on the right.
"""

from osprey.interfaces.web_terminal.app import run_web

__all__ = ["run_web"]
