"""Claude Code SDK E2E test fixtures for safety scenarios.

Provides module-scoped project fixtures and overrides the parent conftest's
autouse registry-reset fixture (not needed for subprocess-based SDK tests).
"""

import pytest
import yaml

from tests.e2e.sdk_helpers import (
    HAS_SDK,
    has_anthropic_api_key,
    init_project,
    is_claude_code_available,
)


# Override parent conftest's autouse fixture (no-op for subprocess-based tests)
@pytest.fixture(autouse=True, scope="function")
def reset_registry_between_tests():
    """No-op override — SDK tests use subprocess isolation."""
    yield


# Module-level skip conditions applied to all tests in this directory
pytestmark = [
    pytest.mark.e2e,
    pytest.mark.skipif(not HAS_SDK, reason="claude_agent_sdk not installed"),
    pytest.mark.skipif(not is_claude_code_available(), reason="Claude Code CLI not installed"),
    pytest.mark.skipif(not has_anthropic_api_key(), reason="ANTHROPIC_API_KEY not set"),
]


@pytest.fixture(scope="module")
def safety_project(tmp_path_factory):
    """Module-scoped initialized project for safety tests.

    Creates a control_assistant project once per test file and reuses it
    across all tests in that file. Writes are enabled (default).
    """
    tmp = tmp_path_factory.mktemp("safety")
    return init_project(tmp, "safety-test-project")


@pytest.fixture(scope="module")
def safety_project_writes_off(tmp_path_factory):
    """Module-scoped project with writes_enabled: false.

    Used by kill-switch tests to verify that the writes_check hook
    blocks all write operations when the master kill switch is off.
    """
    tmp = tmp_path_factory.mktemp("safety-writes-off")
    project_dir = init_project(tmp, "safety-writes-off")
    config_path = project_dir / "config.yml"
    config = yaml.safe_load(config_path.read_text())
    config["control_system"]["writes_enabled"] = False
    config_path.write_text(yaml.dump(config, default_flow_style=False))
    return project_dir


@pytest.fixture(scope="module")
def safety_project_selective(tmp_path_factory):
    """Module-scoped project with approval.global_mode: selective.

    Used by approval flow tests to verify that write operations trigger
    the approval callback while reads pass through silently.
    """
    tmp = tmp_path_factory.mktemp("safety-selective")
    project_dir = init_project(tmp, "safety-selective")
    config_path = project_dir / "config.yml"
    config = yaml.safe_load(config_path.read_text())
    config["approval"]["global_mode"] = "selective"
    config["control_system"]["writes_enabled"] = True
    config_path.write_text(yaml.dump(config, default_flow_style=False))
    return project_dir


@pytest.fixture(scope="module")
def safety_project_all_capabilities(tmp_path_factory):
    """Module-scoped project with approval.global_mode: all_capabilities.

    Used by approval flow tests to verify that ALL tool calls (including
    reads) trigger the approval callback.
    """
    tmp = tmp_path_factory.mktemp("safety-all-caps")
    project_dir = init_project(tmp, "safety-all-caps")
    config_path = project_dir / "config.yml"
    config = yaml.safe_load(config_path.read_text())
    config["approval"]["global_mode"] = "all_capabilities"
    config["control_system"]["writes_enabled"] = True
    config_path.write_text(yaml.dump(config, default_flow_style=False))
    return project_dir
