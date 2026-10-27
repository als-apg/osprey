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

Write posture is per target, so the render is three-way: no target armed is the
hard deny above, every target armed renders nothing, and targets that DISAGREE
render neither — one settings.json cannot say "denied on live, allowed on va",
so the gated tools are pulled from ``ask`` and the runtime hooks decide each
call. The targets counted are the ones a session here can actually be pointed
at, not both names unconditionally: a deployment that does not render the
switch has one, and an armed connector block for a machine it never reaches
must not soften its kill switch. The last three sections pin all of that, plus
the rule that only a literal ``true`` arms writes at either level.
"""

from collections.abc import Iterator
from pathlib import Path

import pytest
import yaml

from osprey import bluesky_tool_names as bsky
from osprey.cli.templates import claude_code
from osprey.cli.templates.manager import TemplateManager
from tests._preset_data import create_project_as_built

#: The project every ``_build_ctx`` call reads, built once per module.
_BASE_PROJECT: Path | None = None


@pytest.fixture(scope="module", autouse=True)
def _base_project(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Path]:
    """The render every test reads, made once; ``_build_ctx`` edits a copy of its config only."""
    global _BASE_PROJECT
    _BASE_PROJECT = create_project_as_built(
        TemplateManager(),
        facility_views=True,
        project_name="killswitch-bluesky-base",
        output_dir=tmp_path_factory.mktemp("killswitch-base"),
        data_bundle="control_assistant",
        context={"channel_finder_mode": "hierarchical"},
    )
    yield _BASE_PROJECT
    _BASE_PROJECT = None


def _build_ctx(
    *,
    writes_enabled: bool,
    claude_code_overrides: dict | None = None,
    connector_writes: dict[str, bool] | None = None,
    control_system_type: str | None = None,
    drop_connectors: tuple[str, ...] = (),
):
    """Apply config overrides to the base project's config and return the built context.

    The context is built from the config dict it is handed, so each call edits
    a fresh read of the base config and the project itself is never written;
    callers can invoke this helper more than once per test.

    *connector_writes* sets ``control_system.connector.<type>.writes_enabled``
    per connector type; *control_system_type* sets the type the deployment
    actually builds and *drop_connectors* removes connector blocks. Together
    they decide which targets a session here can be pointed at, which is what
    the render reads: a deployment is two-target only when it renders the
    switch (its own type is one of the targets and both have a configured
    block), and otherwise a session sits on the built type alone.
    """
    assert _BASE_PROJECT is not None, "the module's base project is built by _base_project"
    manager = TemplateManager()
    config = yaml.safe_load((_BASE_PROJECT / "config.yml").read_text())
    config["control_system"]["writes_enabled"] = writes_enabled
    if control_system_type is not None:
        config["control_system"]["type"] = control_system_type
    for connector_type in drop_connectors:
        del config["control_system"]["connector"][connector_type]
    for connector_type, armed in (connector_writes or {}).items():
        config["control_system"]["connector"][connector_type]["writes_enabled"] = armed
    if claude_code_overrides is not None:
        config["claude_code"] = claude_code_overrides

    return claude_code.build_claude_code_context(
        manager.template_root, manager.jinja_env, _BASE_PROJECT, config
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


def test_bluesky_arming_tools_denied_when_writes_off():
    ctx = _build_ctx(
        writes_enabled=False,
        claude_code_overrides={"servers": {"bluesky": {"enabled": True}}},
    )
    for tool in bsky.ARMING_TOOLS:
        assert f"mcp__bluesky__{tool}" in _ks_deny(ctx), (
            f"{tool!r} arms hardware motion and must be hard-denied when writes are off"
        )


def test_bluesky_arming_tools_not_denied_when_writes_on():
    ctx = _build_ctx(
        writes_enabled=True,
        claude_code_overrides={"servers": {"bluesky": {"enabled": True}}},
    )
    perms = ctx["facility_permissions"]
    for tool in bsky.ARMING_TOOLS:
        assert f"mcp__bluesky__{tool}" not in _ks_deny(ctx)
        assert f"mcp__bluesky__{tool}" not in perms.get("remove_ask", [])


def test_bluesky_disabled_server_contributes_nothing():
    """bluesky is opt-in (default_enabled=False) — an un-enabled bluesky server
    must not contribute a deny entry even when writes are off.

    Switched off explicitly: the preset this fixture builds on enables the
    server, so leaving the key unstated would test the armed deployment.
    """
    ctx = _build_ctx(
        writes_enabled=False,
        claude_code_overrides={"servers": {"bluesky": {"enabled": False}}},
    )
    perms = ctx["facility_permissions"]
    for tool in bsky.ARMING_TOOLS:
        assert f"mcp__bluesky__{tool}" not in _ks_deny(ctx)
        assert f"mcp__bluesky__{tool}" not in perms.get("remove_ask", [])


# ---------------------------------------------------------------------------
# queue_stop / stop_run — never denied or removed-from-ask (safe direction)
# ---------------------------------------------------------------------------


def test_bluesky_stop_tools_never_denied_or_removed():
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
            writes_enabled=writes_enabled,
            claude_code_overrides={"servers": {"bluesky": {"enabled": True}}},
        )
        perms = ctx["facility_permissions"]
        # queue_remove sits in the same tier: it only discards pending work and
        # is the sole way past the interrupted-item start refusal, so a deny
        # here would trap a wedged queue exactly when writes are off.
        for tool in (bsky.QUEUE_STOP, bsky.QUEUE_REMOVE, bsky.STOP_RUN):
            assert f"mcp__bluesky__{tool}" not in _ks_deny(ctx)
            assert f"mcp__bluesky__{tool}" not in perms.get("remove_ask", [])


def test_bluesky_queue_read_tools_never_denied():
    """queue_list / queue_status are reads and stay reachable with writes off.

    Losing ``queue_status`` under writes-off would blind the agent to the very
    fact that the deployment cannot execute.
    """
    ctx = _build_ctx(
        writes_enabled=False,
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


def test_bluesky_authoring_tools_never_denied_or_removed():
    """write_plan/validate_plan carry approval only (no
    _WRITES_CHECK) — like stop_run, the kill switch must never block them,
    regardless of writes_enabled, since neither reaches hardware."""
    for writes_enabled in (True, False):
        ctx = _build_ctx(
            writes_enabled=writes_enabled,
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


def test_controls_channel_write_still_denied_when_writes_off():
    ctx = _build_ctx(writes_enabled=False)
    assert "mcp__controls__channel_write" in _ks_deny(ctx)


def test_python_execute_still_removed_from_ask_when_writes_off():
    ctx = _build_ctx(writes_enabled=False)
    perms = ctx["facility_permissions"]
    assert "mcp__python__execute" in perms["remove_ask"]
    # Must not ALSO be hard-denied — python's execute has a legitimate
    # read-only path and is handled via remove_ask, not deny.
    assert "mcp__python__execute" not in _ks_deny(ctx)


def test_python_execute_regranted_via_allow_for_requiring_agent_when_writes_off():
    """A writes-off render keeps a python-requiring agent's compute path.

    ``mcp__python__execute`` is pulled from ``ask`` under writes-off (the kill
    switch that stops a read-write execute from reaching the ``can_use_tool``
    prompt), but the pyat-specialist agent hard-requires it. It must therefore
    be re-granted via ``allow`` — reachable by the agent, off the approval-prompt
    path, still hook-guarded against write-access kernels. This is the readonly
    render case where an inherited compute agent meets writes-off.
    """
    ctx = _build_ctx(writes_enabled=False)
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


def test_nothing_added_when_writes_enabled():
    ctx = _build_ctx(writes_enabled=True)
    perms = ctx["facility_permissions"]
    assert "mcp__controls__channel_write" not in _ks_deny(ctx)
    assert "mcp__python__execute" not in perms.get("remove_ask", [])


# ---------------------------------------------------------------------------
# Extends clone: rewritten-prefix matcher
# ---------------------------------------------------------------------------


def test_extends_clone_of_bluesky_denied_with_rewritten_prefix():
    ctx = _build_ctx(
        writes_enabled=False,
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
# Per-target posture: which targets a session here can actually be pointed at
# ---------------------------------------------------------------------------

#: The connector block backing each control target on a switch-rendering
#: deployment. Spelled out rather than resolved, so a preset that stopped
#: configuring both targets fails these tests instead of quietly making them
#: assert the all-off render.
_LIVE_CONNECTOR = "epics"
_VA_CONNECTOR = "virtual_accelerator"


def _switchable(**kwargs):
    """A two-target deployment: the built type is the live one and both blocks exist.

    The control-assistant preset builds a ``mock`` and carries the two blocks
    already, so naming ``epics`` as the type is all it takes to make ``live``
    and ``va`` both selectable.
    """
    return _build_ctx(control_system_type=_LIVE_CONNECTOR, **kwargs)


def test_mixed_posture_denies_nothing_and_pulls_the_write_tools_from_ask():
    """Global writes off, VA armed: no kill-switch deny, every gated tool unasked.

    settings.json is rendered once, before any session picks a target, so a
    static deny would be wrong on the armed target and a static ask would be
    wrong on the unarmed one — the ask reopens the SDK's can_use_tool prompt,
    which the writes-check hook cannot suppress. The render therefore steps
    aside on both counts and leaves the per-call decision to the hooks.
    """
    ctx = _switchable(
        writes_enabled=False,
        connector_writes={_VA_CONNECTOR: True},
        claude_code_overrides={"servers": {"bluesky": {"enabled": True}}},
    )
    perms = ctx["facility_permissions"]
    remove_ask = perms["remove_ask"]

    assert _ks_deny(ctx) == []
    assert "mcp__controls__channel_write" in remove_ask
    assert "mcp__python__execute" in remove_ask
    for tool in bsky.ARMING_TOOLS:
        assert f"mcp__bluesky__{tool}" in remove_ask


def test_mixed_posture_from_a_live_block_that_disarms_a_global_true():
    """The other direction: global writes on, the live machine's block says no.

    Reaches the same render as the case above — the two targets disagree — and
    must, since which key carries the disagreement is not something the
    permissions layer can act on differently.
    """
    ctx = _switchable(
        writes_enabled=True,
        connector_writes={_LIVE_CONNECTOR: False},
        claude_code_overrides={"servers": {"bluesky": {"enabled": True}}},
    )
    perms = ctx["facility_permissions"]
    remove_ask = perms["remove_ask"]

    assert _ks_deny(ctx) == []
    assert "mcp__controls__channel_write" in remove_ask
    for tool in bsky.ARMING_TOOLS:
        assert f"mcp__bluesky__{tool}" in remove_ask


def test_mixed_posture_never_puts_a_pure_write_tool_in_allow():
    """The rescue may re-grant read/write-mixed tools only, never pure writes.

    On a mixed render every writes-check-gated matcher is pulled from ``ask``,
    so a rescue keyed on ``remove_ask`` would be free to promote
    ``channel_write`` to ``allow`` — an auto-approved control-system write on a
    deployment that just said one of its targets must not be written to. The
    rescue is keyed on the read/write-mixed set instead.
    """
    ctx = _switchable(
        writes_enabled=False,
        connector_writes={_VA_CONNECTOR: True},
        claude_code_overrides={"servers": {"bluesky": {"enabled": True}}},
    )
    allow = ctx["facility_permissions"].get("allow", [])
    assert "mcp__controls__channel_write" not in allow
    for tool in bsky.ARMING_TOOLS:
        assert f"mcp__bluesky__{tool}" not in allow


def test_mixed_posture_regrants_python_execute_only_for_a_requiring_agent():
    """python's execute reaches ``allow`` through the required-tool rescue alone.

    With ``pyat-specialist`` enabled the tool is re-granted, exactly as on an
    all-off render: it is the agent's only compute path, and an agent declaring
    a tool that is in none of the permission lists fails build validation. With
    that agent off, the mixed render leaves ``execute`` in no list at all and
    the writes-check hook decides it per call.
    """
    with_agent = _switchable(
        writes_enabled=False,
        connector_writes={_VA_CONNECTOR: True},
    )
    assert any(a["name"] == "pyat-specialist" and a["enabled"] for a in with_agent["agents"])
    assert "mcp__python__execute" in with_agent["facility_permissions"]["allow"]
    assert "mcp__python__execute" not in _ks_deny(with_agent)

    without_agent = _switchable(
        writes_enabled=False,
        connector_writes={_VA_CONNECTOR: True},
        claude_code_overrides={"agents": {"pyat-specialist": {"enabled": False}}},
    )
    perms = without_agent["facility_permissions"]
    assert "mcp__python__execute" in perms["remove_ask"]
    assert "mcp__python__execute" not in perms.get("allow", [])
    assert "mcp__python__execute" not in _ks_deny(without_agent)


def test_agreeing_targets_still_render_the_all_off_kill_switch():
    """Per-connector keys that AGREE with the global one change nothing.

    The three-way render is keyed on the resolved per-target postures, so a
    deployment that spells its writes-off posture out per connector must land on
    the same hard deny as one that only sets the deployment-wide key.
    """
    ctx = _switchable(
        writes_enabled=False,
        connector_writes={_LIVE_CONNECTOR: False, _VA_CONNECTOR: False},
        claude_code_overrides={"servers": {"bluesky": {"enabled": True}}},
    )
    assert "mcp__controls__channel_write" in _ks_deny(ctx)
    for tool in bsky.ARMING_TOOLS:
        assert f"mcp__bluesky__{tool}" in _ks_deny(ctx)
    assert "mcp__python__execute" in ctx["facility_permissions"]["remove_ask"]


def test_both_targets_armed_renders_nothing_even_with_the_global_key_off():
    """Per-connector ``true`` on both targets is an all-on render.

    The mirror of the case above, and the one that proves the render reads the
    resolved postures rather than the deployment-wide key: writes are off
    globally here, yet neither target inherits that, so there is nothing to
    take away.
    """
    ctx = _switchable(
        writes_enabled=False,
        connector_writes={_LIVE_CONNECTOR: True, _VA_CONNECTOR: True},
        claude_code_overrides={"servers": {"bluesky": {"enabled": True}}},
    )
    perms = ctx["facility_permissions"]
    assert _ks_deny(ctx) == []
    assert "mcp__controls__channel_write" not in perms.get("remove_ask", [])
    assert "mcp__python__execute" not in perms.get("remove_ask", [])


# ---------------------------------------------------------------------------
# One-target deployments: an unreachable target must not soften the kill switch
# ---------------------------------------------------------------------------


def test_an_armed_block_for_an_unreachable_machine_keeps_the_hard_deny():
    """A doocs deployment carrying an armed epics block is NOT mixed.

    The session runs the doocs connector; the epics block names a second real
    machine no target reaches, and with no simulator block the deployment does
    not render the switch, so there is no second target to disagree with.
    Reading the stray block anyway would drop the hard deny — writes off, yet
    channel_write neither denied nor pulled from ask.
    """
    ctx = _build_ctx(
        writes_enabled=False,
        control_system_type="doocs",
        drop_connectors=(_VA_CONNECTOR,),
        connector_writes={_LIVE_CONNECTOR: True},
        claude_code_overrides={"servers": {"bluesky": {"enabled": True}}},
    )
    assert "mcp__controls__channel_write" in _ks_deny(ctx)
    for tool in bsky.ARMING_TOOLS:
        assert f"mcp__bluesky__{tool}" in _ks_deny(ctx)


def test_a_single_target_deployment_that_disarms_its_own_machine_is_denied():
    """An epics deployment with no VA block and its own block disarmed.

    Its one reachable target is unarmed, so the deployment-wide ``true`` arms
    nothing a session here can reach and the kill switch must fire. This is the
    case a two-target loop gets backwards in the dangerous direction: it would
    read an armed ``va`` the deployment does not configure and call the render
    mixed.
    """
    ctx = _build_ctx(
        writes_enabled=True,
        control_system_type=_LIVE_CONNECTOR,
        drop_connectors=(_VA_CONNECTOR,),
        connector_writes={_LIVE_CONNECTOR: False},
        claude_code_overrides={"servers": {"bluesky": {"enabled": True}}},
    )
    assert "mcp__controls__channel_write" in _ks_deny(ctx)


def test_a_single_target_deployment_that_arms_its_own_machine_renders_nothing():
    """The negative control: one reachable target, armed, is an all-on render."""
    ctx = _build_ctx(
        writes_enabled=False,
        control_system_type=_LIVE_CONNECTOR,
        drop_connectors=(_VA_CONNECTOR,),
        connector_writes={_LIVE_CONNECTOR: True},
        claude_code_overrides={"servers": {"bluesky": {"enabled": True}}},
    )
    assert _ks_deny(ctx) == []
    assert "mcp__python__execute" not in ctx["facility_permissions"].get("remove_ask", [])


# ---------------------------------------------------------------------------
# Only a literal `true` arms writes
# ---------------------------------------------------------------------------


def test_a_non_boolean_writes_enabled_renders_the_kill_switch():
    """``'true'`` and ``1`` are not ``true``, and arm nothing.

    A tightening: the render used to test the deployment-wide key for
    truthiness, so a quoted or numeric value skipped the kill switch entirely.
    Both levels of the posture now read a literal boolean and nothing else, so
    a value nobody can be sure of lands on the side that costs an operator a
    config edit rather than a machine.
    """
    for value in ("true", 1):
        ctx = _build_ctx(
            writes_enabled=value,
            claude_code_overrides={"servers": {"bluesky": {"enabled": True}}},
        )
        assert "mcp__controls__channel_write" in _ks_deny(ctx), (
            f"writes_enabled={value!r} must not arm writes"
        )
