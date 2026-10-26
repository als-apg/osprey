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
tier (write only emits a file, validate only dry-runs mock
devices), so they carry ``_APPROVAL`` only — same as ``stop_run`` — and must
never be denied or removed-from-ask by the kill switch.

The kill switch reads the tier, ``control_system.writes``, alone: ``never``
is the hard deny above, ``allowed`` denies nothing and pulls from ``ask`` only
the tools whose rule carries both writes-check and an approval hook (the arming
pair's preflight approval included). Only the literal ``allowed`` arms, and the
retired per-connector ``writes_enabled`` leaves no longer move the render once
the tier is spelled. ``test_killswitch_on_tier.py`` covers the facility half.
"""

from pathlib import Path

import yaml

from osprey import bluesky_tool_names as bsky
from osprey.cli.templates import claude_code
from osprey.cli.templates.manager import TemplateManager
from tests._preset_data import bundle_data_root


def _create_project(manager: TemplateManager, **kwargs) -> Path:
    """``create_project`` plus the steps a real build takes next.

    A build renders the framework template, overlays the resolved profile's
    ``config:`` block onto the result, stamps ``.osprey-manifest.json``, writes
    the facility views, and regenerates ``.claude/`` from the finished config. The template carries
    only derived and profile-field-derived keys, so a fixture that stops after
    the render holds half a config — the declarative half is the preset's, and
    the artifacts rendered before it landed do not know about the deployment's
    control system, services or servers. These fixtures render from a bundle
    rather than from a profile, so they overlay the preset ``osprey init``
    pairs with that bundle.
    """
    from osprey.cli.build_profile import resolve_build_profile
    from osprey.utils.config_writer import config_update_fields
    from tests._facility_file import write_facility_views

    bundle = kwargs.setdefault("data_bundle", "control_assistant")
    preset = bundle.replace("_", "-")
    kwargs.setdefault("data_root", bundle_data_root(bundle))
    project = manager.create_project(**kwargs)
    profile, _preset_dir = resolve_build_profile(None, preset=preset)
    config_update_fields(project / "config.yml", profile.config)
    manager.generate_manifest(
        project, kwargs["project_name"], preset, {}, artifacts=kwargs.get("artifacts")
    )
    # The build writes the facility views before its last render, so the
    # agent facts name the models the render serves.
    write_facility_views(project, bundle)
    # The build's last render, and the one that ships: `create_project` wrote
    # `.claude/` from a config.yml that did not yet carry the preset's block.
    manager.regenerate_claude_code(project)
    return project


_PROJECT_COUNTER = 0


def _build_ctx(
    tmp_path,
    *,
    writes,
    claude_code_overrides: dict | None = None,
    connector_writes: dict[str, bool] | None = None,
):
    """Create a project, apply config overrides, and return the built context.

    Each call gets a unique project name/output dir (TemplateManager refuses
    to create a project in an already-existing directory), so callers can
    invoke this helper more than once per test.

    *writes* is the tier, ``control_system.writes``, the one key the kill
    switch reads. *connector_writes* sets the retired per-connector
    ``writes_enabled`` leaves, which the render must ignore once the tier is
    spelled.
    """
    global _PROJECT_COUNTER
    _PROJECT_COUNTER += 1
    manager = TemplateManager()
    project_dir = _create_project(
        manager,
        project_name=f"killswitch-bluesky-{_PROJECT_COUNTER}",
        output_dir=tmp_path,
        data_bundle="control_assistant",
        context={"channel_finder_mode": "hierarchical"},
    )
    config = yaml.safe_load((project_dir / "config.yml").read_text())
    config["control_system"]["writes"] = writes
    for connector_type, armed in (connector_writes or {}).items():
        config["control_system"]["connector"][connector_type]["writes_enabled"] = armed
    if claude_code_overrides is not None:
        config["claude_code"] = claude_code_overrides
    (project_dir / "config.yml").write_text(yaml.dump(config))

    return claude_code.build_claude_code_context(
        manager.template_root, manager.jinja_env, project_dir, config
    )


def _ks_deny(ctx: dict) -> list[str]:
    """The kill switch's own deny entries.

    They render through the dedicated ``killswitch_deny`` context key rather
    than ``facility_permissions['deny']``, so that ``remove_deny`` — which a
    profile authors — can never subtract a writes-off deny.
    """
    return ctx["killswitch_deny"]


# ---------------------------------------------------------------------------
# The arming tools (queue_add / queue_start) — hard deny when writes are off
# ---------------------------------------------------------------------------


def test_bluesky_arming_tools_denied_when_writes_off(tmp_path):
    ctx = _build_ctx(
        tmp_path,
        writes="never",
        claude_code_overrides={"servers": {"bluesky": {"enabled": True}}},
    )
    for tool in bsky.ARMING_TOOLS:
        assert f"mcp__bluesky__{tool}" in _ks_deny(ctx), (
            f"{tool!r} arms hardware motion and must be hard-denied when writes are off"
        )


def test_bluesky_arming_tools_not_denied_when_writes_on(tmp_path):
    """On ``allowed`` the arming pair is not denied, and leaves ``ask``.

    Its rule carries writes-check and the preflight approval variant: the
    approval hook raises the prompt when the switch is on, and writes-check
    refuses the call before any prompt when it is off.
    """
    ctx = _build_ctx(
        tmp_path,
        writes="allowed",
        claude_code_overrides={"servers": {"bluesky": {"enabled": True}}},
    )
    perms = ctx["facility_permissions"]
    for tool in bsky.ARMING_TOOLS:
        assert f"mcp__bluesky__{tool}" not in _ks_deny(ctx)
        assert f"mcp__bluesky__{tool}" in perms.get("remove_ask", [])
        assert f"mcp__bluesky__{tool}" not in perms.get("allow", [])


def test_bluesky_disabled_server_contributes_nothing(tmp_path):
    """bluesky is opt-in (default_enabled=False) — an un-enabled bluesky server
    must not contribute a deny entry even when writes are off.

    Switched off explicitly: the preset this fixture builds on enables the
    server, so leaving the key unstated would test the armed deployment.
    """
    ctx = _build_ctx(
        tmp_path,
        writes="never",
        claude_code_overrides={"servers": {"bluesky": {"enabled": False}}},
    )
    perms = ctx["facility_permissions"]
    for tool in bsky.ARMING_TOOLS:
        assert f"mcp__bluesky__{tool}" not in _ks_deny(ctx)
        assert f"mcp__bluesky__{tool}" not in perms.get("remove_ask", [])


# ---------------------------------------------------------------------------
# queue_stop / stop_run — never denied or removed-from-ask (safe direction)
# ---------------------------------------------------------------------------


def test_bluesky_stop_tools_never_denied_or_removed(tmp_path):
    """queue_stop and stop_run carry approval only (no _WRITES_CHECK).

    The kill switch must never block halting, on either tier —
    denying ``queue_stop`` under writes-off would take the queue's halt away at
    exactly the moment an operator is most likely to reach for it, and denying
    ``stop_run`` would take away the EMERGENCY halt (the only surface that
    aborts a plan already moving hardware). This is the negative control for
    the deny test above: it proves the kill switch is selecting on the arming
    hook, not blanket-denying the whole server.
    """
    for writes in ("allowed", "never"):
        ctx = _build_ctx(
            tmp_path,
            writes=writes,
            claude_code_overrides={"servers": {"bluesky": {"enabled": True}}},
        )
        perms = ctx["facility_permissions"]
        # queue_remove sits in the same tier: it only discards pending work and
        # is the sole way past the interrupted-item start refusal, so a deny
        # here would trap a wedged queue exactly when writes are off.
        for tool in (bsky.QUEUE_STOP, bsky.QUEUE_REMOVE, bsky.STOP_RUN):
            assert f"mcp__bluesky__{tool}" not in _ks_deny(ctx)
            assert f"mcp__bluesky__{tool}" not in perms.get("remove_ask", [])


def test_bluesky_queue_read_tools_never_denied(tmp_path):
    """queue_list / queue_status are reads and stay reachable with writes off.

    Losing ``queue_status`` under writes-off would blind the agent to the very
    fact that the deployment cannot execute.
    """
    ctx = _build_ctx(
        tmp_path,
        writes="never",
        claude_code_overrides={"servers": {"bluesky": {"enabled": True}}},
    )
    perms = ctx["facility_permissions"]
    for tool in bsky.QUEUE_READ_TOOLS:
        assert f"mcp__bluesky__{tool}" not in _ks_deny(ctx)
        assert f"mcp__bluesky__{tool}" not in perms.get("remove_ask", [])


# ---------------------------------------------------------------------------
# bluesky.write_plan / bluesky.validate_plan — task 2.3 authoring
# tools; never denied or removed-from-ask (neither reaches hardware)
# ---------------------------------------------------------------------------


def test_bluesky_authoring_tools_never_denied_or_removed(tmp_path):
    """write_plan/validate_plan carry approval only (no
    _WRITES_CHECK) — like stop_run, the kill switch must never block them,
    on either tier, since neither reaches hardware."""
    for writes in ("allowed", "never"):
        ctx = _build_ctx(
            tmp_path,
            writes=writes,
            claude_code_overrides={"servers": {"bluesky": {"enabled": True}}},
        )
        perms = ctx["facility_permissions"]
        for tool in ("write_plan", "validate_plan"):
            matcher = f"mcp__bluesky__{tool}"
            assert matcher not in _ks_deny(ctx)
            assert matcher not in perms.get("remove_ask", [])


# ---------------------------------------------------------------------------
# Regression parity: controls/python behavior preserved after generalizing
# ---------------------------------------------------------------------------


def test_controls_channel_write_still_denied_when_writes_off(tmp_path):
    ctx = _build_ctx(tmp_path, writes="never")
    assert "mcp__controls__channel_write" in _ks_deny(ctx)


def test_python_execute_still_removed_from_ask_when_writes_off(tmp_path):
    ctx = _build_ctx(tmp_path, writes="never")
    perms = ctx["facility_permissions"]
    assert "mcp__python__execute" in perms["remove_ask"]
    # Must not ALSO be hard-denied — python's execute has a legitimate
    # read-only path and is handled via remove_ask, not deny.
    assert "mcp__python__execute" not in _ks_deny(ctx)


def test_python_execute_regranted_via_allow_for_requiring_agent_when_writes_off(tmp_path):
    """A writes-off render keeps a python-requiring agent's compute path.

    ``mcp__python__execute`` is pulled from ``ask`` under writes-off (the kill
    switch that stops a read-write execute from reaching the ``can_use_tool``
    prompt), but the pyat-specialist agent hard-requires it. It must therefore
    be re-granted via ``allow`` — reachable by the agent, off the approval-prompt
    path, still hook-guarded against write-access kernels. This is the readonly
    render case where an inherited compute agent meets writes-off.
    """
    ctx = _build_ctx(tmp_path, writes="never")
    assert any(a["name"] == "pyat-specialist" and a["enabled"] for a in ctx["agents"]), (
        "guard assumes pyat-specialist (a python-requiring agent) is enabled"
    )
    perms = ctx["facility_permissions"]
    # Off the ask/can_use_tool path (kill-switch preserved) …
    assert "mcp__python__execute" in perms["remove_ask"]
    assert "mcp__python__execute" not in perms.get("ask", [])
    # … but re-granted via allow so the agent still has it, and never denied.
    assert "mcp__python__execute" in perms.get("allow", [])
    assert "mcp__python__execute" not in _ks_deny(ctx)


def test_nothing_denied_when_writes_allowed(tmp_path):
    ctx = _build_ctx(tmp_path, writes="allowed")
    assert _ks_deny(ctx) == []
    # execute carries writes-check and approval, so it leaves `ask` and the
    # agent that requires it keeps it through `allow`.
    perms = ctx["facility_permissions"]
    assert "mcp__python__execute" in perms["remove_ask"]
    assert "mcp__python__execute" in perms["allow"]
    assert "mcp__controls__channel_write" not in perms["allow"]


# ---------------------------------------------------------------------------
# Extends clone: rewritten-prefix matcher
# ---------------------------------------------------------------------------


def test_extends_clone_of_bluesky_denied_with_rewritten_prefix(tmp_path):
    ctx = _build_ctx(
        tmp_path,
        writes="never",
        claude_code_overrides={"servers": {"bluesky2": {"extends": "bluesky"}}},
    )
    for tool in bsky.ARMING_TOOLS:
        assert f"mcp__bluesky2__{tool}" in _ks_deny(ctx)
        # The template name itself must not leak into the clone's deny entry.
        assert f"mcp__bluesky__{tool}" not in _ks_deny(ctx)
    # The clone's authoring tools (approval-only, no _WRITES_CHECK) are never
    # denied under the rewritten prefix either.
    assert "mcp__bluesky2__write_plan" not in _ks_deny(ctx)
    assert "mcp__bluesky2__validate_plan" not in _ks_deny(ctx)


# ---------------------------------------------------------------------------
# The tier alone: retired per-connector leaves no longer move the render
# ---------------------------------------------------------------------------

_LIVE_CONNECTOR = "epics"
_VA_CONNECTOR = "virtual_accelerator"


def test_never_denies_even_with_every_connector_leaf_armed(tmp_path):
    """The tier is a ceiling: armed retired leaves cannot lift a ``never``."""
    ctx = _build_ctx(
        tmp_path,
        writes="never",
        connector_writes={_LIVE_CONNECTOR: True, _VA_CONNECTOR: True},
        claude_code_overrides={"servers": {"bluesky": {"enabled": True}}},
    )
    assert "mcp__controls__channel_write" in _ks_deny(ctx)
    for tool in bsky.ARMING_TOOLS:
        assert f"mcp__bluesky__{tool}" in _ks_deny(ctx)


def test_allowed_denies_nothing_with_every_connector_leaf_disarmed(tmp_path):
    """The per-call decision is the runtime switch's, not a retired leaf's."""
    ctx = _build_ctx(
        tmp_path,
        writes="allowed",
        connector_writes={_LIVE_CONNECTOR: False, _VA_CONNECTOR: False},
        claude_code_overrides={"servers": {"bluesky": {"enabled": True}}},
    )
    assert _ks_deny(ctx) == []


def test_disagreeing_connector_leaves_render_exactly_like_their_tier(tmp_path):
    """No three-way render: disagreeing leaves change nothing the tier decides."""
    servers = {"servers": {"bluesky": {"enabled": True}}}
    for writes in ("allowed", "never"):
        plain = _build_ctx(tmp_path, writes=writes, claude_code_overrides=servers)
        split = _build_ctx(
            tmp_path,
            writes=writes,
            connector_writes={_LIVE_CONNECTOR: False, _VA_CONNECTOR: True},
            claude_code_overrides=servers,
        )
        assert _ks_deny(split) == _ks_deny(plain)
        assert split["facility_permissions"] == plain["facility_permissions"]


# ---------------------------------------------------------------------------
# Only the literal `allowed` arms writes
# ---------------------------------------------------------------------------


def test_a_value_other_than_allowed_renders_the_kill_switch(tmp_path):
    """``'Allowed'``, ``'true'``, ``True`` and ``1`` are not ``allowed``.

    A value nobody can be sure of lands on the side that costs an operator a
    config edit rather than a machine.
    """
    for value in ("Allowed", "true", True, 1):
        ctx = _build_ctx(
            tmp_path,
            writes=value,
            claude_code_overrides={"servers": {"bluesky": {"enabled": True}}},
        )
        assert "mcp__controls__channel_write" in _ks_deny(ctx), (
            f"control_system.writes={value!r} must not arm writes"
        )
