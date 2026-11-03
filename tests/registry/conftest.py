"""Fixtures shared by the registry suites."""

from __future__ import annotations

import pytest


@pytest.fixture
def template_manager():
    """A fresh :class:`~osprey.cli.templates.manager.TemplateManager`."""
    from osprey.cli.templates.manager import TemplateManager

    return TemplateManager()
