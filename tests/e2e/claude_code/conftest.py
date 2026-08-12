"""Claude Code SDK E2E test fixtures for safety scenarios.

Provides module-scoped deployment-repo fixtures and overrides the parent
conftest's autouse registry-reset fixture (not needed for subprocess-based SDK
tests).

Every fixture here yields the REPO ROOT — the handle ``sdk_helpers`` takes — and
resolves the render (``<repo>/build``) itself for the config edits, since the
rendered ``config.yml`` and ``.claude/`` are what a running agent session reads.
"""

from pathlib import Path

import pytest
import yaml

from tests.e2e.sdk_helpers import (
    HAS_SDK,
    init_project,
    is_claude_code_available,
    render_dir,
)

# Dedicated, preset-decoupled limits DB for the write-safety scenarios. The
# generic safety e2e must not depend on any preset's production
# channel_limits.json (which is a pure projection of the VA manifest and
# carries no example read-only/bounded channels). This fixture supplies exactly
# the two channels those tests need: a bounded writable setpoint and a
# read-only readback.
SAFETY_LIMITS_DB = Path(__file__).parent / "fixtures" / "safety_limits.json"


def _point_at_safety_limits_db(repo: Path) -> None:
    """Repoint the render's limits database at the safety fixture.

    ``control_system.limits_checking.database_path`` is read live by both the
    limits PreToolUse hook and the channel_write MCP tool via
    ``LimitsValidator.from_config()``; the path is resolved at runtime rather
    than baked into settings.json, so no re-render is needed. The value written
    is absolute, so it bypasses the relative-path resolution against
    ``CONFIG_FILE``'s directory entirely.
    """
    config_path = render_dir(repo) / "config.yml"
    config = yaml.safe_load(config_path.read_text())
    config["control_system"]["limits_checking"]["database_path"] = str(SAFETY_LIMITS_DB)
    config_path.write_text(yaml.dump(config, default_flow_style=False))


def _rerender_claude_artifacts(repo: Path) -> None:
    """Re-render the render's Claude Code artifacts after a ``config.yml`` edit.

    ``settings.json`` is rendered from ``config.yml`` at build time, so a
    runtime config edit that changes a permission decision leaves it stale. This
    is the same re-render step ``osprey build`` performs, run on its own so the
    edit above is not thrown away by a full rebuild (which would re-derive
    ``config.yml`` from ``profile.yml``).

    ``regen_if_drift`` returns ``[]`` and writes nothing when the render has no
    ``.claude/settings.json`` — it is a re-sync, never a bootstrap. That is the
    right behaviour for production and a trap here: the fixtures below flip a
    safety flag and then rely on this call to bake it into ``settings.json``, so
    a silent no-op would leave the kill-switch tests asserting against artifacts
    that never saw the flag — provider-metered tests passing for no reason. The
    assertion below pins the one condition under which that happens, so the
    fixture fails loudly at setup instead of the tests passing vacuously.
    """
    from osprey.cli.templates.manager import TemplateManager

    render = render_dir(repo)
    settings = render / ".claude" / "settings.json"
    assert settings.is_file(), (
        f"no rendered {settings} to re-sync — regen_if_drift would no-op silently "
        "and this fixture's config edit would never reach the agent"
    )
    TemplateManager().regen_if_drift(render)


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
    """Module-scoped deployment repo for safety tests.

    Builds a control_assistant deployment once per test file and reuses it
    across all tests in that file. Writes are enabled (default).
    """
    tmp = tmp_path_factory.mktemp("safety")
    repo = init_project(tmp, "safety-test-project", provider="als-apg")
    _point_at_safety_limits_db(repo)
    return repo


@pytest.fixture(scope="module")
def safety_project_writes_off(tmp_path_factory):
    """Module-scoped deployment with writes_enabled: false.

    Used by kill-switch tests to verify that the writes_check hook
    blocks all write operations when the master kill switch is off.

    Re-renders the Claude Code artifacts after flipping ``writes_enabled`` so the
    rendered ``settings.json`` reflects the new flag. Without that, the
    renderer's writes-aware permissions.deny augmentation (which moves
    pure-write tools out of permissions.ask when writes are off) is bypassed,
    and Claude Code's permissions.ask layer fires ``can_use_tool`` for
    channel_write even though the writes_check hook denies it in parallel.
    """
    tmp = tmp_path_factory.mktemp("safety-writes-off")
    repo = init_project(tmp, "safety-writes-off", provider="als-apg")
    config_path = render_dir(repo) / "config.yml"
    config = yaml.safe_load(config_path.read_text())
    config["control_system"]["writes_enabled"] = False
    config_path.write_text(yaml.dump(config, default_flow_style=False))
    _rerender_claude_artifacts(repo)
    return repo


@pytest.fixture(scope="module")
def safety_project_selective(tmp_path_factory):
    """Module-scoped deployment mirroring the production per-tool approval default.

    Used by approval flow tests to verify that write operations trigger
    the approval callback while reads pass through silently. Mirrors the
    rendered ``control_assistant/config.yml.j2`` defaults: channel reads
    skip approval, writes always ask, ``execute`` is content-aware.
    """
    tmp = tmp_path_factory.mktemp("safety-selective")
    repo = init_project(tmp, "safety-selective", provider="als-apg")
    config_path = render_dir(repo) / "config.yml"
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
    return repo


@pytest.fixture(scope="module")
def safety_project_default_policy_always(tmp_path_factory):
    """Module-scoped deployment where every tool requires approval.

    Used by approval flow tests to verify that ALL tool calls (including
    reads) trigger the approval callback. With ``tools`` absent and
    ``default_policy: always``, every tool falls through to the always-ask
    path.
    """
    tmp = tmp_path_factory.mktemp("safety-default-always")
    repo = init_project(tmp, "safety-default-always", provider="als-apg")
    config_path = render_dir(repo) / "config.yml"
    config = yaml.safe_load(config_path.read_text())
    config["approval"] = {"enabled": True, "default_policy": "always"}
    config["control_system"]["writes_enabled"] = True
    config_path.write_text(yaml.dump(config, default_flow_style=False))
    return repo
