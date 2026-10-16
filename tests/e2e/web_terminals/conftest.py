"""Fixtures shared by the web-terminal e2e modules."""

from __future__ import annotations

import pytest

from tests.e2e.sdk_helpers import init_project
from tests.e2e.web_terminals.persona_projects import deny_swing_tool


@pytest.fixture(scope="module")
def persona_a_denied_repo(tmp_path_factory):
    """Persona A: a minimal deployment repo whose render denies the swing tool."""
    tmp = tmp_path_factory.mktemp("persona-a")
    repo = init_project(tmp, "persona-a-denied", provider="als-apg")
    deny_swing_tool(repo)
    return repo


@pytest.fixture(scope="module")
def persona_b_permitted_repo(tmp_path_factory):
    """Persona B: the same repo with the default config, where the swing tool is allowed.

    The one ``config.yml`` difference from persona A is the swing tool's entry
    in persona A's ``deny`` list.
    """
    tmp = tmp_path_factory.mktemp("persona-b")
    return init_project(tmp, "persona-b-permitted", provider="als-apg")
