"""Tests for the generalized kill-switch deny/remove_ask extension covering scan.

``build_claude_code_context``'s writes-off kill-switch block (previously
hardcoded to ``controls``/``python`` by name) now walks ``FRAMEWORK_SERVERS``
for any hooks_pre rule gated by ``_WRITES_CHECK``, so a new write server (e.g.
``scan``'s ``launch_run``) is covered automatically with no per-server code
change. These tests pin: scan's launch_run is hard-denied when writes are
off, stop_run (approval-only, no writes-check) is NEVER denied or
removed-from-ask (the kill switch must not block stopping a scan), the
existing controls/python behavior is preserved, and an extends clone gets the
rewritten-prefix matcher.

Also pins the task-2.3 authoring tools (``write_plan``,
``validate_plan``): both reach no hardware regardless of
``writes_enabled`` (write only emits a file, validate only dry-runs mock
devices), so they carry ``_APPROVAL`` only — same as ``stop_run`` — and must
never be denied or removed-from-ask by the kill switch.
"""

import yaml

from osprey.cli.templates import claude_code
from osprey.cli.templates.manager import TemplateManager

_PROJECT_COUNTER = 0


def _build_ctx(tmp_path, *, writes_enabled: bool, claude_code_overrides: dict | None = None):
    """Create a project, apply config overrides, and return the built context.

    Each call gets a unique project name/output dir (TemplateManager refuses
    to create a project in an already-existing directory), so callers can
    invoke this helper more than once per test.
    """
    global _PROJECT_COUNTER
    _PROJECT_COUNTER += 1
    manager = TemplateManager()
    project_dir = manager.create_project(
        project_name=f"killswitch-scan-{_PROJECT_COUNTER}",
        output_dir=tmp_path,
        data_bundle="control_assistant",
        context={"channel_finder_mode": "hierarchical"},
    )
    config = yaml.safe_load((project_dir / "config.yml").read_text())
    config["control_system"]["writes_enabled"] = writes_enabled
    if claude_code_overrides is not None:
        config["claude_code"] = claude_code_overrides
    (project_dir / "config.yml").write_text(yaml.dump(config))

    return claude_code.build_claude_code_context(
        manager.template_root, manager.jinja_env, project_dir, config
    )


# ---------------------------------------------------------------------------
# scan.launch_run — hard deny when writes are off
# ---------------------------------------------------------------------------


def test_scan_launch_denied_when_writes_off(tmp_path):
    ctx = _build_ctx(
        tmp_path,
        writes_enabled=False,
        claude_code_overrides={"servers": {"bluesky": {"enabled": True}}},
    )
    perms = ctx["facility_permissions"]
    assert "mcp__bluesky__launch_run" in perms["deny"]


def test_scan_launch_not_denied_when_writes_on(tmp_path):
    ctx = _build_ctx(
        tmp_path,
        writes_enabled=True,
        claude_code_overrides={"servers": {"bluesky": {"enabled": True}}},
    )
    perms = ctx["facility_permissions"]
    assert "mcp__bluesky__launch_run" not in perms.get("deny", [])
    assert "mcp__bluesky__launch_run" not in perms.get("remove_ask", [])


def test_scan_disabled_server_contributes_nothing(tmp_path):
    """scan is opt-in (default_enabled=False) — an un-enabled scan server must
    not contribute a deny entry even when writes are off."""
    ctx = _build_ctx(tmp_path, writes_enabled=False)
    perms = ctx["facility_permissions"]
    assert "mcp__bluesky__launch_run" not in perms.get("deny", [])
    assert "mcp__bluesky__launch_run" not in perms.get("remove_ask", [])


# ---------------------------------------------------------------------------
# scan.stop_run — never denied or removed-from-ask (safe direction)
# ---------------------------------------------------------------------------


def test_scan_stop_never_denied_or_removed(tmp_path):
    """stop_run carries approval only (no _WRITES_CHECK) — the kill switch
    must never block stopping a scan, regardless of writes_enabled."""
    for writes_enabled in (True, False):
        ctx = _build_ctx(
            tmp_path,
            writes_enabled=writes_enabled,
            claude_code_overrides={"servers": {"bluesky": {"enabled": True}}},
        )
        perms = ctx["facility_permissions"]
        assert "mcp__bluesky__stop_run" not in perms.get("deny", [])
        assert "mcp__bluesky__stop_run" not in perms.get("remove_ask", [])


# ---------------------------------------------------------------------------
# scan.write_plan / scan.validate_plan — task 2.3 authoring
# tools; never denied or removed-from-ask (neither reaches hardware)
# ---------------------------------------------------------------------------


def test_scan_authoring_tools_never_denied_or_removed(tmp_path):
    """write_plan/validate_plan carry approval only (no
    _WRITES_CHECK) — like stop_run, the kill switch must never block them,
    regardless of writes_enabled, since neither reaches hardware."""
    for writes_enabled in (True, False):
        ctx = _build_ctx(
            tmp_path,
            writes_enabled=writes_enabled,
            claude_code_overrides={"servers": {"bluesky": {"enabled": True}}},
        )
        perms = ctx["facility_permissions"]
        for tool in ("write_plan", "validate_plan"):
            matcher = f"mcp__bluesky__{tool}"
            assert matcher not in perms.get("deny", [])
            assert matcher not in perms.get("remove_ask", [])


# ---------------------------------------------------------------------------
# Regression parity: controls/python behavior preserved after generalizing
# ---------------------------------------------------------------------------


def test_controls_channel_write_still_denied_when_writes_off(tmp_path):
    ctx = _build_ctx(tmp_path, writes_enabled=False)
    perms = ctx["facility_permissions"]
    assert "mcp__controls__channel_write" in perms["deny"]


def test_python_execute_still_removed_from_ask_when_writes_off(tmp_path):
    ctx = _build_ctx(tmp_path, writes_enabled=False)
    perms = ctx["facility_permissions"]
    assert "mcp__python__execute" in perms["remove_ask"]
    # Must not ALSO be hard-denied — python's execute has a legitimate
    # read-only path and is handled via remove_ask, not deny.
    assert "mcp__python__execute" not in perms.get("deny", [])


def test_nothing_added_when_writes_enabled(tmp_path):
    ctx = _build_ctx(tmp_path, writes_enabled=True)
    perms = ctx["facility_permissions"]
    assert "mcp__controls__channel_write" not in perms.get("deny", [])
    assert "mcp__python__execute" not in perms.get("remove_ask", [])


# ---------------------------------------------------------------------------
# Extends clone: rewritten-prefix matcher
# ---------------------------------------------------------------------------


def test_extends_clone_of_scan_denied_with_rewritten_prefix(tmp_path):
    ctx = _build_ctx(
        tmp_path,
        writes_enabled=False,
        claude_code_overrides={"servers": {"bluesky2": {"extends": "bluesky"}}},
    )
    perms = ctx["facility_permissions"]
    assert "mcp__bluesky2__launch_run" in perms["deny"]
    # The template name itself must not leak into the clone's deny entry.
    assert "mcp__bluesky__launch_run" not in perms["deny"]
    # The clone's authoring tools (approval-only, no _WRITES_CHECK) are never
    # denied under the rewritten prefix either.
    assert "mcp__bluesky2__write_plan" not in perms.get("deny", [])
    assert "mcp__bluesky2__validate_plan" not in perms.get("deny", [])
