"""Drift guard: the rendered settings.json kill-switch stays in sync with
FRAMEWORK_SERVERS' write-gated tools.

Task 1.11 generalized the kill-switch's writes-off deny/remove_ask block in
``build_claude_code_context`` to walk ``FRAMEWORK_SERVERS`` for any
``hooks_pre`` rule gated by ``_WRITES_CHECK``, so a new write server (e.g.
scan's ``launch_run``) is covered automatically with no per-server code
change. ``test_killswitch_scan_deny.py`` pins that behavior at the
``build_claude_code_context`` context-dict level.

This module is the drift guard proper: it exercises the FULL render pipeline
(``TemplateManager.create_project`` + ``regenerate_claude_code`` ->
``settings.json.j2``) and reads the actual on-disk ``.claude/settings.json``,
and it computes the expected write-gated tool set *dynamically* from
``FRAMEWORK_SERVERS`` rather than hardcoding tool names. So it catches both
kinds of drift: a future change that decouples the template from
``facility_permissions.deny``/``remove_ask``, and a new
``_WRITES_CHECK``-gated tool added to ``FRAMEWORK_SERVERS`` with no matching
kill-switch coverage — including ``mcp__bluesky__launch_run`` specifically, but
not limited to it.
"""

from __future__ import annotations

import json

import yaml

from osprey.cli.templates import claude_code
from osprey.cli.templates.claude_code import _MIXED_READ_WRITE_TEMPLATES
from osprey.cli.templates.manager import TemplateManager
from osprey.registry.mcp import _WRITES_CHECK, FRAMEWORK_SERVERS

_PROJECT_COUNTER = 0


def _write_gated_matchers() -> dict[str, str]:
    """Map {template_name: matcher} for every FRAMEWORK_SERVERS hooks_pre rule
    gated by _WRITES_CHECK -- the full set the kill switch must cover."""
    matchers: dict[str, str] = {}
    for template_name, template_def in FRAMEWORK_SERVERS.items():
        for rule in template_def.hooks_pre:
            if _WRITES_CHECK in rule.hooks:
                matchers[template_name] = rule.matcher
    return matchers


def _build_project(tmp_path, *, writes_enabled: bool):
    """Create a real project on disk with every write-gated server enabled.

    Each call gets a unique project name (TemplateManager refuses to create a
    project in an already-existing directory).
    """
    global _PROJECT_COUNTER
    _PROJECT_COUNTER += 1
    manager = TemplateManager()
    project_dir = manager.create_project(
        project_name=f"killswitch-driftguard-{_PROJECT_COUNTER}",
        output_dir=tmp_path,
        data_bundle="control_assistant",
        context={"channel_finder_mode": "hierarchical"},
    )
    config = yaml.safe_load((project_dir / "config.yml").read_text())
    config["control_system"]["writes_enabled"] = writes_enabled
    # Force-enable every write-gated template (some, like scan, are opt-in) so
    # the rendered settings.json actually exercises the full write-gated set,
    # not just whatever happens to be on by default.
    config.setdefault("claude_code", {})["servers"] = {
        template_name: {"enabled": True} for template_name in _write_gated_matchers()
    }
    (project_dir / "config.yml").write_text(yaml.dump(config))

    # Re-render .claude/settings.json (and the rest of the Claude Code
    # integration) from the just-edited config.yml -- create_project() already
    # rendered once with the *original* config, before writes_enabled/servers
    # were overridden above.
    claude_code.regenerate_claude_code(manager.template_root, manager.jinja_env, project_dir)

    return project_dir


def _rendered_permissions(project_dir) -> dict:
    settings_path = project_dir / ".claude" / "settings.json"
    settings = json.loads(settings_path.read_text())
    return settings["permissions"]


def test_every_write_gated_tool_is_covered_when_writes_off(tmp_path):
    """Drift guard: every _WRITES_CHECK-gated matcher across FRAMEWORK_SERVERS
    ends up hard-denied (or, for the documented read/write-mixed exception,
    pulled from ask) in the rendered settings.json when writes are off.

    A future write-gated tool added to FRAMEWORK_SERVERS with no matching
    kill-switch coverage fails this test with no code change required here.
    """
    project_dir = _build_project(tmp_path, writes_enabled=False)
    perms = _rendered_permissions(project_dir)
    deny = set(perms["deny"])
    ask = set(perms["ask"])

    matchers = _write_gated_matchers()
    assert matchers, "no _WRITES_CHECK-gated tool found in FRAMEWORK_SERVERS at all"

    for template_name, matcher in matchers.items():
        if template_name in _MIXED_READ_WRITE_TEMPLATES:
            assert matcher not in ask, (
                f"{matcher!r} ({template_name}) is documented read/write-mixed and "
                f"must be pulled from ask, but is still present"
            )
        else:
            assert matcher in deny, (
                f"{matcher!r} ({template_name}) is _WRITES_CHECK-gated but missing "
                f"from the rendered settings.json deny list with writes disabled"
            )


def test_scan_launch_run_specifically_hard_denied_when_writes_off(tmp_path):
    """The concrete case this drift guard exists for: scan's launch_run."""
    project_dir = _build_project(tmp_path, writes_enabled=False)
    perms = _rendered_permissions(project_dir)
    assert "mcp__bluesky__launch_run" in perms["deny"]


def test_scan_launch_run_not_denied_when_writes_on(tmp_path):
    project_dir = _build_project(tmp_path, writes_enabled=True)
    perms = _rendered_permissions(project_dir)
    assert "mcp__bluesky__launch_run" not in perms["deny"]
    assert "mcp__bluesky__launch_run" not in perms.get("remove_ask", [])


def test_scan_stop_run_never_denied_regardless_of_writes_enabled(tmp_path):
    """stop_run carries approval only (no _WRITES_CHECK) -- the kill switch
    must never block stopping a scan, in either direction."""
    for writes_enabled in (True, False):
        project_dir = _build_project(tmp_path, writes_enabled=writes_enabled)
        perms = _rendered_permissions(project_dir)
        assert "mcp__bluesky__stop_run" not in perms["deny"]
