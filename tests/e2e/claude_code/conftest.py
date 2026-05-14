"""Claude Code SDK E2E test fixtures for safety scenarios.

Provides module-scoped project fixtures and overrides the parent conftest's
autouse registry-reset fixture (not needed for subprocess-based SDK tests).
"""

import pytest
import yaml

from tests.e2e.sdk_helpers import (
    HAS_SDK,
    init_project,
    is_claude_code_available,
)


# Override parent conftest's autouse fixture (no-op for subprocess-based tests)
@pytest.fixture(autouse=True, scope="function")
def reset_registry_between_tests():
    """No-op override — SDK tests use subprocess isolation."""
    yield


# Module-level prerequisites. The ALS_APG_API_KEY gate lives on each test
# via `@pytest.mark.requires_als_apg` (auto-enforced by the root
# `tests/conftest.py` hook) rather than here, so the gating travels with
# the test rather than the directory.
pytestmark = [
    pytest.mark.e2e,
    pytest.mark.skipif(not HAS_SDK, reason="claude_agent_sdk not installed"),
    pytest.mark.skipif(not is_claude_code_available(), reason="Claude Code CLI not installed"),
]


@pytest.fixture(scope="module")
def safety_project(tmp_path_factory):
    """Module-scoped initialized project for safety tests.

    Creates a control_assistant project once per test file and reuses it
    across all tests in that file. Writes are enabled (default).
    """
    tmp = tmp_path_factory.mktemp("safety")
    return init_project(tmp, "safety-test-project", provider="als-apg")


@pytest.fixture(scope="module")
def safety_project_writes_off(tmp_path_factory):
    """Module-scoped project with writes_enabled: false.

    Used by kill-switch tests to verify that the writes_check hook
    blocks all write operations when the master kill switch is off.

    Calls ``osprey claude regen`` after flipping ``writes_enabled`` so the
    rendered ``settings.json`` reflects the new flag. Without regen, the
    renderer's writes-aware permissions.deny augmentation (which moves
    pure-write tools out of permissions.ask when writes are off) is bypassed,
    and Claude Code's permissions.ask layer fires ``can_use_tool`` for
    channel_write even though the writes_check hook denies it in parallel.
    """
    import subprocess
    import sys

    tmp = tmp_path_factory.mktemp("safety-writes-off")
    project_dir = init_project(tmp, "safety-writes-off", provider="als-apg")
    config_path = project_dir / "config.yml"
    config = yaml.safe_load(config_path.read_text())
    config["control_system"]["writes_enabled"] = False
    config_path.write_text(yaml.dump(config, default_flow_style=False))

    regen = subprocess.run(
        [sys.executable, "-m", "osprey.cli.main", "claude", "regen",
         "--project", str(project_dir)],
        capture_output=True, text=True, timeout=60,
    )
    assert regen.returncode == 0, (
        f"osprey claude regen failed:\n--- stdout ---\n{regen.stdout}\n"
        f"--- stderr ---\n{regen.stderr}"
    )
    return project_dir


@pytest.fixture(scope="module")
def safety_project_selective(tmp_path_factory):
    """Module-scoped project mirroring the production per-tool approval default.

    Used by approval flow tests to verify that write operations trigger
    the approval callback while reads pass through silently. Mirrors the
    rendered ``control_assistant/config.yml.j2`` defaults: channel reads
    skip approval, writes always ask, ``execute`` is content-aware.
    """
    tmp = tmp_path_factory.mktemp("safety-selective")
    project_dir = init_project(tmp, "safety-selective", provider="als-apg")
    config_path = project_dir / "config.yml"
    config = yaml.safe_load(config_path.read_text())
    config["approval"] = {
        "enabled": True,
        "default_policy": "always",
        "tools": {
            "channel_read": "skip",
            "archiver_read": "skip",
            "channel_limits": "skip",
            "channel_write": "always",
            "execute": "selective",
        },
    }
    config["control_system"]["writes_enabled"] = True
    config_path.write_text(yaml.dump(config, default_flow_style=False))
    return project_dir


@pytest.fixture(scope="module")
def safety_project_all_capabilities(tmp_path_factory):
    """Module-scoped project where every tool requires approval.

    Used by approval flow tests to verify that ALL tool calls (including
    reads) trigger the approval callback. With ``tools`` absent and
    ``default_policy: always``, every tool falls through to the always-ask
    path — equivalent to the legacy ``global_mode: all_capabilities``.
    """
    tmp = tmp_path_factory.mktemp("safety-all-caps")
    project_dir = init_project(tmp, "safety-all-caps", provider="als-apg")
    config_path = project_dir / "config.yml"
    config = yaml.safe_load(config_path.read_text())
    config["approval"] = {"enabled": True, "default_policy": "always"}
    config["control_system"]["writes_enabled"] = True
    config_path.write_text(yaml.dump(config, default_flow_style=False))
    return project_dir
