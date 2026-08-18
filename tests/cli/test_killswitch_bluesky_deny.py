"""Tests for the generalized kill-switch deny/remove_ask extension covering bluesky.

``build_claude_code_context``'s writes-off kill-switch block walks
``FRAMEWORK_SERVERS`` for any hooks_pre rule gated by ``_WRITES_CHECK`` rather
than naming ``controls``/``python`` by hand, so an added write server (e.g.
the bluesky queue's arming tools) is covered automatically with no per-server
code change. These tests pin: every ``bsky.ARMING_TOOLS`` entry is hard-denied
when writes are off, the approval-only tools (``queue_stop``, ``stop_run``) are
NEVER denied or removed-from-ask (the kill switch must not block halting),
controls/python stay covered, and an extends clone gets the
rewritten-prefix matcher.

Tool names resolve from ``osprey.bluesky_tool_names`` so a rename carries
through; the ``mcp__<server>__`` prefix stays literal here because applying
that prefix is precisely what this code under test does.

Also pins the task-2.3 authoring tools (``write_plan``,
``validate_plan``): both reach no hardware regardless of
``writes_enabled`` (write only emits a file, validate only dry-runs mock
devices), so they carry ``_APPROVAL`` only — same as ``stop_run`` — and must
never be denied or removed-from-ask by the kill switch.
"""

import yaml

from osprey import bluesky_tool_names as bsky
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
        project_name=f"killswitch-bluesky-{_PROJECT_COUNTER}",
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
# The arming tools (queue_add / queue_start) — hard deny when writes are off
# ---------------------------------------------------------------------------


def test_bluesky_arming_tools_denied_when_writes_off(tmp_path):
    ctx = _build_ctx(
        tmp_path,
        writes_enabled=False,
        claude_code_overrides={"servers": {"bluesky": {"enabled": True}}},
    )
    perms = ctx["facility_permissions"]
    for tool in bsky.ARMING_TOOLS:
        assert f"mcp__bluesky__{tool}" in perms["deny"], (
            f"{tool!r} arms hardware motion and must be hard-denied when writes are off"
        )


def test_bluesky_arming_tools_not_denied_when_writes_on(tmp_path):
    ctx = _build_ctx(
        tmp_path,
        writes_enabled=True,
        claude_code_overrides={"servers": {"bluesky": {"enabled": True}}},
    )
    perms = ctx["facility_permissions"]
    for tool in bsky.ARMING_TOOLS:
        assert f"mcp__bluesky__{tool}" not in perms.get("deny", [])
        assert f"mcp__bluesky__{tool}" not in perms.get("remove_ask", [])


def test_bluesky_disabled_server_contributes_nothing(tmp_path):
    """bluesky is opt-in (default_enabled=False) — an un-enabled bluesky server
    must not contribute a deny entry even when writes are off."""
    ctx = _build_ctx(tmp_path, writes_enabled=False)
    perms = ctx["facility_permissions"]
    for tool in bsky.ARMING_TOOLS:
        assert f"mcp__bluesky__{tool}" not in perms.get("deny", [])
        assert f"mcp__bluesky__{tool}" not in perms.get("remove_ask", [])


# ---------------------------------------------------------------------------
# queue_stop / stop_run — never denied or removed-from-ask (safe direction)
# ---------------------------------------------------------------------------


def test_bluesky_stop_tools_never_denied_or_removed(tmp_path):
    """queue_stop and stop_run carry approval only (no _WRITES_CHECK).

    The kill switch must never block halting, regardless of writes_enabled —
    denying ``queue_stop`` under writes-off would take the queue's halt away at
    exactly the moment an operator is most likely to reach for it, and denying
    ``stop_run`` would take away the EMERGENCY halt (the only surface that
    aborts a plan already moving hardware). This is the negative control for
    the deny test above: it proves the kill switch is selecting on the arming
    hook, not blanket-denying the whole server.
    """
    for writes_enabled in (True, False):
        ctx = _build_ctx(
            tmp_path,
            writes_enabled=writes_enabled,
            claude_code_overrides={"servers": {"bluesky": {"enabled": True}}},
        )
        perms = ctx["facility_permissions"]
        for tool in (bsky.QUEUE_STOP, bsky.STOP_RUN):
            assert f"mcp__bluesky__{tool}" not in perms.get("deny", [])
            assert f"mcp__bluesky__{tool}" not in perms.get("remove_ask", [])


def test_bluesky_queue_read_tools_never_denied(tmp_path):
    """queue_list / queue_status are reads and stay reachable with writes off.

    Losing ``queue_status`` under writes-off would blind the agent to the very
    fact that the deployment cannot execute.
    """
    ctx = _build_ctx(
        tmp_path,
        writes_enabled=False,
        claude_code_overrides={"servers": {"bluesky": {"enabled": True}}},
    )
    perms = ctx["facility_permissions"]
    for tool in bsky.QUEUE_READ_TOOLS:
        assert f"mcp__bluesky__{tool}" not in perms.get("deny", [])
        assert f"mcp__bluesky__{tool}" not in perms.get("remove_ask", [])


# ---------------------------------------------------------------------------
# bluesky.write_plan / bluesky.validate_plan — task 2.3 authoring
# tools; never denied or removed-from-ask (neither reaches hardware)
# ---------------------------------------------------------------------------


def test_bluesky_authoring_tools_never_denied_or_removed(tmp_path):
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


def test_python_execute_regranted_via_allow_for_requiring_agent_when_writes_off(tmp_path):
    """A writes-off persona keeps a python-requiring agent's compute path.

    ``mcp__python__execute`` is pulled from ``ask`` under writes-off (the kill
    switch that stops a read-write execute from reaching the ``can_use_tool``
    prompt), but the pyat-specialist agent hard-requires it. It must therefore
    be re-granted via ``allow`` — reachable by the agent, off the approval-prompt
    path, still hook-guarded against write-access kernels. This is the readonly
    persona case where an inherited compute agent meets writes-off.
    """
    ctx = _build_ctx(tmp_path, writes_enabled=False)
    assert any(a["name"] == "pyat-specialist" and a["enabled"] for a in ctx["agents"]), (
        "guard assumes pyat-specialist (a python-requiring agent) is enabled"
    )
    perms = ctx["facility_permissions"]
    # Off the ask/can_use_tool path (kill-switch preserved) …
    assert "mcp__python__execute" in perms["remove_ask"]
    assert "mcp__python__execute" not in perms.get("ask", [])
    # … but re-granted via allow so the agent still has it, and never denied.
    assert "mcp__python__execute" in perms.get("allow", [])
    assert "mcp__python__execute" not in perms.get("deny", [])


def test_nothing_added_when_writes_enabled(tmp_path):
    ctx = _build_ctx(tmp_path, writes_enabled=True)
    perms = ctx["facility_permissions"]
    assert "mcp__controls__channel_write" not in perms.get("deny", [])
    assert "mcp__python__execute" not in perms.get("remove_ask", [])


# ---------------------------------------------------------------------------
# Extends clone: rewritten-prefix matcher
# ---------------------------------------------------------------------------


def test_extends_clone_of_bluesky_denied_with_rewritten_prefix(tmp_path):
    ctx = _build_ctx(
        tmp_path,
        writes_enabled=False,
        claude_code_overrides={"servers": {"bluesky2": {"extends": "bluesky"}}},
    )
    perms = ctx["facility_permissions"]
    for tool in bsky.ARMING_TOOLS:
        assert f"mcp__bluesky2__{tool}" in perms["deny"]
        # The template name itself must not leak into the clone's deny entry.
        assert f"mcp__bluesky__{tool}" not in perms["deny"]
    # The clone's authoring tools (approval-only, no _WRITES_CHECK) are never
    # denied under the rewritten prefix either.
    assert "mcp__bluesky2__write_plan" not in perms.get("deny", [])
    assert "mcp__bluesky2__validate_plan" not in perms.get("deny", [])
