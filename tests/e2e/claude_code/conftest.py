"""Claude Code SDK E2E test fixtures for safety scenarios.

Provides module-scoped project fixtures and overrides the parent conftest's
autouse registry-reset fixture (not needed for subprocess-based SDK tests).
"""

from pathlib import Path

import pytest
import yaml

from tests.e2e.sdk_helpers import (
    HAS_SDK,
    init_project,
    is_claude_code_available,
)

# Dedicated, preset-decoupled limits DB for the write-safety scenarios. The
# generic safety e2e must not depend on any preset's production
# channel_limits.json (which is a pure projection of the VA manifest and
# carries no example read-only/bounded channels). This fixture supplies exactly
# the two channels those tests need: a bounded writable setpoint and a
# read-only readback.
SAFETY_LIMITS_DB = Path(__file__).parent / "fixtures" / "safety_limits.json"


def _point_at_safety_limits_db(project_dir: Path) -> None:
    """Repoint a rendered project's limits database at the safety fixture.

    ``control_system.limits_checking.database_path`` is read live by both the
    limits PreToolUse hook and the channel_write MCP tool via
    ``LimitsValidator.from_config()``; the path is resolved at runtime rather
    than baked into settings.json, so no ``osprey claude regen`` is needed.
    """
    config_path = project_dir / "config.yml"
    config = yaml.safe_load(config_path.read_text())
    config["control_system"]["limits_checking"]["database_path"] = str(SAFETY_LIMITS_DB)
    config_path.write_text(yaml.dump(config, default_flow_style=False))


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
    project_dir = init_project(tmp, "safety-test-project", provider="als-apg")
    _point_at_safety_limits_db(project_dir)
    return project_dir


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
        [sys.executable, "-m", "osprey.cli.main", "claude", "regen", "--project", str(project_dir)],
        capture_output=True,
        text=True,
        timeout=60,
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
    config["control_system"]["limits_checking"]["database_path"] = str(SAFETY_LIMITS_DB)
    config_path.write_text(yaml.dump(config, default_flow_style=False))
    return project_dir


@pytest.fixture(scope="module")
def safety_project_default_policy_always(tmp_path_factory):
    """Module-scoped project where every tool requires approval.

    Used by approval flow tests to verify that ALL tool calls (including
    reads) trigger the approval callback. With ``tools`` absent and
    ``default_policy: always``, every tool falls through to the always-ask
    path.
    """
    tmp = tmp_path_factory.mktemp("safety-default-always")
    project_dir = init_project(tmp, "safety-default-always", provider="als-apg")
    config_path = project_dir / "config.yml"
    config = yaml.safe_load(config_path.read_text())
    config["approval"] = {"enabled": True, "default_policy": "always"}
    config["control_system"]["writes_enabled"] = True
    config_path.write_text(yaml.dump(config, default_flow_style=False))
    return project_dir
