"""Drift guard: the rendered settings.json kill-switch stays in sync with
FRAMEWORK_SERVERS' write-gated tools.

Task 1.11 generalized the kill-switch's writes-off deny/remove_ask block in
``build_claude_code_context`` to walk ``FRAMEWORK_SERVERS`` for any
``hooks_pre`` rule gated by ``_WRITES_CHECK``, so a new write server (e.g. the
bluesky queue's arming tools) is covered automatically with no per-server code
change. ``test_killswitch_bluesky_deny.py`` pins that behavior at the
``build_claude_code_context`` context-dict level.

This module is the drift guard proper: it exercises the FULL render pipeline
(``TemplateManager.create_project`` + ``regenerate_claude_code`` ->
``settings.json.j2``) and reads the actual on-disk ``.claude/settings.json``,
and it computes the expected write-gated tool set *dynamically* from
``FRAMEWORK_SERVERS`` rather than hardcoding tool names. So it catches both
kinds of drift: a future change that decouples the template from
``facility_permissions.deny``/``remove_ask``, and a new
``_WRITES_CHECK``-gated tool added to ``FRAMEWORK_SERVERS`` with no matching
kill-switch coverage — including the bluesky queue's arming tools
specifically, but not limited to them.

The same sweep runs against the third render the per-target write posture
introduces: when the deployment's two targets DISAGREE, one settings.json
cannot deny a tool that is legal on the armed target, so the whole gated set is
pulled from ``ask`` and denied nowhere, and the runtime hooks decide per call.
The last section covers that render and pins that a posture spelled out per
connector, but agreeing with the deployment-wide key, renders byte-identically
to the deployment-wide key alone.
"""

from __future__ import annotations

import json

import yaml

from osprey import bluesky_tool_names as bsky
from osprey.cli.templates import claude_code
from osprey.cli.templates.manager import TemplateManager
from osprey.registry.mcp import (
    _WRITES_CHECK,
    FRAMEWORK_SERVERS,
    MIXED_READ_WRITE_TEMPLATES,
)

_PROJECT_COUNTER = 0


def _write_gated_matchers() -> dict[str, list[str]]:
    """Map {template_name: [matcher, ...]} for every FRAMEWORK_SERVERS hooks_pre
    rule gated by _WRITES_CHECK -- the full set the kill switch must cover.

    A LIST per template, not a single matcher: bluesky alone contributes two
    write-gated rules (the queue's arming pair), and keeping only the last one
    would silently leave the other unchecked by the drift guard below — the
    exact class of gap this module exists to catch.
    """
    matchers: dict[str, list[str]] = {}
    for template_name, template_def in FRAMEWORK_SERVERS.items():
        for rule in template_def.hooks_pre:
            if _WRITES_CHECK in rule.hooks:
                matchers.setdefault(template_name, []).append(rule.matcher)
    return matchers


def _build_project(
    tmp_path,
    *,
    writes_enabled: bool,
    connector_writes: dict | None = None,
    control_system_type: str | None = None,
):
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
    _rerender(
        manager,
        project_dir,
        writes_enabled,
        connector_writes,
        control_system_type=control_system_type,
    )
    return project_dir


def _rerender(
    manager,
    project_dir,
    writes_enabled: bool,
    connector_writes: dict | None = None,
    *,
    control_system_type: str | None = None,
):
    """Apply the write posture to config.yml and re-render the Claude Code artifacts.

    *connector_writes* sets ``control_system.connector.<type>.writes_enabled``
    per connector type and *control_system_type* sets the type the deployment
    builds. Both matter: the render counts only the targets a session here can
    be pointed at, and a deployment has two of those only when it renders the
    switch — its own type is one of the targets and both have a configured
    block. The control-assistant preset builds a ``mock`` and already carries an
    ``epics`` block for ``live`` and a ``virtual_accelerator`` block for ``va``,
    so naming ``epics`` as the type is what makes both selectable.
    """
    config = yaml.safe_load((project_dir / "config.yml").read_text())
    config["control_system"]["writes_enabled"] = writes_enabled
    if control_system_type is not None:
        config["control_system"]["type"] = control_system_type
    for connector_type, armed in (connector_writes or {}).items():
        config["control_system"]["connector"][connector_type]["writes_enabled"] = armed
    # Force-enable every write-gated template (some, like bluesky, are opt-in) so
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

    for template_name, template_matchers in matchers.items():
        for matcher in template_matchers:
            if template_name in MIXED_READ_WRITE_TEMPLATES:
                assert matcher not in ask, (
                    f"{matcher!r} ({template_name}) is documented read/write-mixed and "
                    f"must be pulled from ask, but is still present"
                )
            else:
                assert matcher in deny, (
                    f"{matcher!r} ({template_name}) is _WRITES_CHECK-gated but missing "
                    f"from the rendered settings.json deny list with writes disabled"
                )


def test_bluesky_arming_tools_specifically_hard_denied_when_writes_off(tmp_path):
    """The concrete case this drift guard exists for: the queue's arming pair.

    Named explicitly, on top of the dynamic sweep above, so the sweep going
    vacuous (an empty matcher map, a bluesky server that stopped rendering)
    cannot quietly take this coverage with it.
    """
    project_dir = _build_project(tmp_path, writes_enabled=False)
    perms = _rendered_permissions(project_dir)
    for tool in bsky.ARMING_TOOLS:
        assert f"mcp__bluesky__{tool}" in perms["deny"]


def test_bluesky_arming_tools_not_denied_when_writes_on(tmp_path):
    project_dir = _build_project(tmp_path, writes_enabled=True)
    perms = _rendered_permissions(project_dir)
    for tool in bsky.ARMING_TOOLS:
        assert f"mcp__bluesky__{tool}" not in perms["deny"]
        assert f"mcp__bluesky__{tool}" not in perms.get("remove_ask", [])


def test_bluesky_queue_stop_never_denied_even_with_writes_off(tmp_path):
    """Negative control at the rendered-settings level: halting stays reachable.

    Proves the deny list above is selecting on the arming hook rather than
    denying the bluesky server wholesale — and that a kill switch which took
    away the queue's stop would fail here.
    """
    project_dir = _build_project(tmp_path, writes_enabled=False)
    perms = _rendered_permissions(project_dir)
    assert f"mcp__bluesky__{bsky.QUEUE_STOP}" not in perms["deny"]
    assert f"mcp__bluesky__{bsky.QUEUE_STOP}" not in perms.get("remove_ask", [])


def test_bluesky_stop_run_never_denied_regardless_of_writes_enabled(tmp_path):
    """stop_run carries approval only (no _WRITES_CHECK) -- the kill switch
    must never block stopping a run, in either direction.

    This is load-bearing rather than merely tidy: stop_run is the only
    surface that ABORTS a plan already moving hardware (POST /queue/abort).
    Denying it under writes-off would take the emergency halt away at exactly
    the moment writes were disabled because something was wrong.
    """
    for writes_enabled in (True, False):
        project_dir = _build_project(tmp_path, writes_enabled=writes_enabled)
        perms = _rendered_permissions(project_dir)
        assert "mcp__bluesky__stop_run" not in perms["deny"]


# ---------------------------------------------------------------------------
# Mixed posture: the two targets disagree, so nothing is denied and nothing asks
# ---------------------------------------------------------------------------

#: The connector block backing each session target once the deployment renders
#: the switch. Named rather than resolved so a preset that stopped configuring
#: both targets fails these tests instead of quietly turning them into a second
#: copy of the all-off case.
_LIVE_CONNECTOR = "epics"
_VA_CONNECTOR = "virtual_accelerator"


def _settings_text(project_dir) -> str:
    return (project_dir / ".claude" / "settings.json").read_text()


def test_every_write_gated_tool_is_pulled_from_ask_on_a_mixed_render(tmp_path):
    """Drift guard for the mixed case: no deny, no ask, for the whole gated set.

    Computed from ``FRAMEWORK_SERVERS`` the same way as the writes-off sweep, so
    a new write-gated tool is covered here too with no change to this file. The
    ask half is the load-bearing one: a matcher left in ``permissions.ask``
    would reach the SDK's can_use_tool prompt, which the per-target writes-check
    hook cannot suppress, and an operator would be asked to approve a write the
    active target's posture forbids.
    """
    project_dir = _build_project(
        tmp_path,
        writes_enabled=False,
        control_system_type=_LIVE_CONNECTOR,
        connector_writes={_VA_CONNECTOR: True},
    )
    perms = _rendered_permissions(project_dir)
    deny = set(perms["deny"])
    ask = set(perms["ask"])

    matchers = _write_gated_matchers()
    assert matchers, "no _WRITES_CHECK-gated tool found in FRAMEWORK_SERVERS at all"

    for template_name, template_matchers in matchers.items():
        for matcher in template_matchers:
            assert matcher not in deny, (
                f"{matcher!r} ({template_name}) is legal on the armed target and must "
                f"not be statically denied on a mixed render"
            )
            assert matcher not in ask, (
                f"{matcher!r} ({template_name}) must be pulled from ask on a mixed "
                f"render so the writes-check hook decides it per call"
            )


def test_mixed_render_never_allows_a_pure_write_tool(tmp_path):
    """Pulling the gated set from ask must not push any of it into allow.

    ``allow`` is auto-approval at the permission layer. It is where the
    required-tool rescue re-grants python's read/write-mixed ``execute``, and it
    is where a rescue keyed on the wrong set would deposit ``channel_write`` and
    the queue's arming tools.
    """
    project_dir = _build_project(
        tmp_path,
        writes_enabled=True,
        control_system_type=_LIVE_CONNECTOR,
        connector_writes={_LIVE_CONNECTOR: False},
    )
    allow = set(_rendered_permissions(project_dir)["allow"])

    for template_name, template_matchers in _write_gated_matchers().items():
        if template_name in MIXED_READ_WRITE_TEMPLATES:
            continue
        for matcher in template_matchers:
            assert matcher not in allow, (
                f"{matcher!r} ({template_name}) is pure-write and must never be "
                f"auto-approved on a mixed render"
            )
    assert "mcp__controls__channel_write" not in allow
    for tool in bsky.ARMING_TOOLS:
        assert f"mcp__bluesky__{tool}" not in allow


def test_agreeing_per_connector_keys_render_byte_identically_with_writes_off(tmp_path):
    """Spelling the writes-off posture out per connector changes no byte.

    Rendered twice into the SAME project so the comparison is of the posture
    keying alone — project name, paths and interpreter are held fixed. The
    three-way render reads the resolved per-target postures, and per-connector
    keys that merely restate the deployment-wide one must resolve to the render
    that deployment already had.
    """
    manager = TemplateManager()
    project_dir = manager.create_project(
        project_name="killswitch-identity-off",
        output_dir=tmp_path,
        data_bundle="control_assistant",
        context={"channel_finder_mode": "hierarchical"},
    )
    _rerender(manager, project_dir, False, control_system_type=_LIVE_CONNECTOR)
    from_global_key_only = _settings_text(project_dir)

    _rerender(
        manager,
        project_dir,
        False,
        {_LIVE_CONNECTOR: False, _VA_CONNECTOR: False},
        control_system_type=_LIVE_CONNECTOR,
    )
    assert _settings_text(project_dir) == from_global_key_only


def test_agreeing_per_connector_keys_render_byte_identically_with_writes_on(tmp_path):
    """The same identity in the armed direction."""
    manager = TemplateManager()
    project_dir = manager.create_project(
        project_name="killswitch-identity-on",
        output_dir=tmp_path,
        data_bundle="control_assistant",
        context={"channel_finder_mode": "hierarchical"},
    )
    _rerender(manager, project_dir, True, control_system_type=_LIVE_CONNECTOR)
    from_global_key_only = _settings_text(project_dir)

    _rerender(
        manager,
        project_dir,
        True,
        {_LIVE_CONNECTOR: True, _VA_CONNECTOR: True},
        control_system_type=_LIVE_CONNECTOR,
    )
    assert _settings_text(project_dir) == from_global_key_only


def test_an_armed_block_for_an_unreachable_machine_still_renders_the_deny(tmp_path):
    """The rendered-settings pin for a one-target deployment.

    This project builds the ``mock`` connector and does not render the switch,
    so the armed ``epics`` block below describes a machine no session here
    reaches. Counting it as a second target would call the render mixed and
    drop the hard deny — writes off, and yet ``channel_write`` in neither deny
    nor ask.
    """
    project_dir = _build_project(
        tmp_path, writes_enabled=False, connector_writes={_LIVE_CONNECTOR: True}
    )
    deny = _rendered_permissions(project_dir)["deny"]
    assert "mcp__controls__channel_write" in deny
    for tool in bsky.ARMING_TOOLS:
        assert f"mcp__bluesky__{tool}" in deny
