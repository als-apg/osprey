"""Fixtures shared by the web-terminal e2e modules."""

from __future__ import annotations

import pytest

from tests.e2e.sdk_helpers import init_project
from tests.e2e.web_terminals.render_projects import deny_swing_tool


@pytest.fixture(scope="module")
def render_a_denied_repo(tmp_path_factory):
    """Render A: a minimal deployment repo whose render denies the swing tool."""
    tmp = tmp_path_factory.mktemp("render-a")
    repo = init_project(tmp, "render-a-denied", provider="als-apg")
    deny_swing_tool(repo)
    return repo


@pytest.fixture(scope="module")
def render_b_permitted_repo(tmp_path_factory):
    """Render B: the same repo with the default config, where the swing tool is allowed.

    The one ``config.yml`` difference from render A is the swing tool's entry
    in render A's ``deny`` list.
    """
    tmp = tmp_path_factory.mktemp("render-b")
    return init_project(tmp, "render-b-permitted", provider="als-apg")
