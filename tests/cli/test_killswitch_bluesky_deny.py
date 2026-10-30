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
is the hard deny above, ``asked`` and ``allowed`` render alike: they deny nothing
and pull from ``ask`` only the tools whose rule carries both writes-check and an
approval hook (the arming pair's preflight approval included). Only those two
literal words arm, and the
retired per-connector ``writes_enabled`` leaves no longer move the render once
the tier is spelled — on a two-target deployment, on a one-target deployment,
and on one carrying an armed block for a machine it never reaches. The tier is
a ceiling for every machine, so which targets a session here can be pointed at
never changes the render. ``test_killswitch_on_tier.py`` covers the facility
half.
"""

from collections.abc import Iterator
from pathlib import Path

import pytest
import yaml

from osprey import bluesky_tool_names as bsky
from osprey.cli.templates import claude_code
from osprey.cli.templates.manager import TemplateManager
from tests._preset_data import create_project_as_built

#: The tier words that arm writes; the render does not tell them apart.
ARMING = ("asked", "allowed")

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
    writes: str = "allowed",
    claude_code_overrides: dict | None = None,
    connector_writes: dict[str, bool] | None = None,
    control_system_type: str | None = None,
    drop_connectors: tuple[str, ...] = (),
):
    """Apply config overrides to the base project's config and return the built context.

    The context is built from the config dict it is handed, so each call edits
    a fresh read of the base config and the project itself is never written;
    callers can invoke this helper more than once per test.

    *writes* is the tier, ``control_system.writes``, the one key the kill
    switch reads. *connector_writes* sets the retired per-connector
    ``writes_enabled`` leaves, which the render must ignore once the tier is
    spelled. *control_system_type* sets the type the deployment actually builds
    and *drop_connectors* removes connector blocks: together they shape which
    targets a session here could be pointed at, and the render must not care.
    """
    assert _BASE_PROJECT is not None, "the module's base project is built by _base_project"
    manager = TemplateManager()
    config = yaml.safe_load((_BASE_PROJECT / "config.yml").read_text())
    config["control_system"]["writes"] = writes
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
        writes="never",
        claude_code_overrides={"servers": {"bluesky": {"enabled": True}}},
    )
    for tool in bsky.ARMING_TOOLS:
        assert f"mcp__bluesky__{tool}" in _ks_deny(ctx), (
            f"{tool!r} arms hardware motion and must be hard-denied when writes are off"
        )


@pytest.mark.parametrize("word", ARMING)
def test_bluesky_arming_tools_not_denied_when_writes_on(word):
    """On an arming tier the arming pair is not denied, and leaves ``ask``.

    Its rule carries writes-check and the preflight approval variant: the
    approval hook decides the prompt from the switch's position, and writes-check
    refuses the call before any prompt when the switch is off.
    """
    ctx = _build_ctx(
        writes=word,
        claude_code_overrides={"servers": {"bluesky": {"enabled": True}}},
    )
    perms = ctx["facility_permissions"]
    for tool in bsky.ARMING_TOOLS:
        assert f"mcp__bluesky__{tool}" not in _ks_deny(ctx)
        assert f"mcp__bluesky__{tool}" in perms.get("remove_ask", [])
        assert f"mcp__bluesky__{tool}" not in perms.get("allow", [])


def test_bluesky_disabled_server_contributes_nothing():
    """bluesky is opt-in (default_enabled=False) — an un-enabled bluesky server
    must not contribute a deny entry even when writes are off.

    Switched off explicitly: the preset this fixture builds on enables the
    server, so leaving the key unstated would test the armed deployment.
    """
    ctx = _build_ctx(
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


def test_bluesky_stop_tools_never_denied_or_removed():
    """queue_stop and stop_run carry approval only (no _WRITES_CHECK).

    The kill switch must never block halting, on either tier —
    denying ``queue_stop`` under writes-off would take the queue's halt away at
    exactly the moment an operator is most likely to reach for it, and denying
    ``stop_run`` would take away the EMERGENCY halt (the only surface that
    aborts a plan already moving hardware). This is the negative control for
    the deny test above: it proves the kill switch is selecting on the arming
    hook, not blanket-denying the whole server.
    """
    for writes in ("asked", "allowed", "never"):
        ctx = _build_ctx(
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


def test_bluesky_queue_read_tools_never_denied():
    """queue_list / queue_status are reads and stay reachable with writes off.

    Losing ``queue_status`` under writes-off would blind the agent to the very
    fact that the deployment cannot execute.
    """
    ctx = _build_ctx(
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


def test_bluesky_authoring_tools_never_denied_or_removed():
    """write_plan/validate_plan carry approval only (no
    _WRITES_CHECK) — like stop_run, the kill switch must never block them,
    on either tier, since neither reaches hardware."""
    for writes in ("asked", "allowed", "never"):
        ctx = _build_ctx(
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


def test_controls_channel_write_still_denied_when_writes_off():
    ctx = _build_ctx(writes="never")
    assert "mcp__controls__channel_write" in _ks_deny(ctx)


def test_python_execute_still_removed_from_ask_when_writes_off():
    ctx = _build_ctx(writes="never")
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
    ctx = _build_ctx(writes="never")
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


@pytest.mark.parametrize("word", ARMING)
def test_nothing_denied_on_an_arming_tier(word):
    ctx = _build_ctx(writes=word)
    assert _ks_deny(ctx) == []
    # execute carries writes-check and approval, so it leaves `ask` and the
    # agent that requires it keeps it through `allow`.
    perms = ctx["facility_permissions"]
    assert "mcp__python__execute" in perms["remove_ask"]
    assert "mcp__python__execute" in perms["allow"]
    assert "mcp__controls__channel_write" not in perms["allow"]


@pytest.mark.parametrize("word", ARMING)
def test_an_arming_tier_never_puts_a_pure_write_tool_in_allow(word):
    """The rescue may re-grant read/write-mixed tools only, never pure writes.

    On an arming render every matcher carrying writes-check and approval
    is pulled from ``ask``, so a rescue keyed on ``remove_ask`` would be free to
    promote ``channel_write`` or the arming pair to ``allow`` — an
    auto-approved control-system write with no prompt in front of it. The
    rescue is keyed on the read/write-mixed set instead.
    """
    ctx = _build_ctx(
        writes=word,
        claude_code_overrides={"servers": {"bluesky": {"enabled": True}}},
    )
    allow = ctx["facility_permissions"].get("allow", [])
    assert "mcp__controls__channel_write" not in allow
    for tool in bsky.ARMING_TOOLS:
        assert f"mcp__bluesky__{tool}" not in allow


@pytest.mark.parametrize("word", ARMING)
def test_an_arming_tier_regrants_python_execute_only_for_a_requiring_agent(word):
    """python's execute reaches ``allow`` through the required-tool rescue alone.

    With ``pyat-specialist`` enabled the tool is re-granted, exactly as on a
    ``never`` render: it is the agent's only compute path, and an agent
    declaring a tool that is in none of the permission lists fails build
    validation. With that agent off, an arming render leaves ``execute``
    in no list at all and the two runtime hooks decide it per call.
    """
    with_agent = _build_ctx(writes=word)
    assert any(a["name"] == "pyat-specialist" and a["enabled"] for a in with_agent["agents"])
    assert "mcp__python__execute" in with_agent["facility_permissions"]["allow"]
    assert "mcp__python__execute" not in _ks_deny(with_agent)

    without_agent = _build_ctx(
        writes=word,
        claude_code_overrides={"agents": {"pyat-specialist": {"enabled": False}}},
    )
    perms = without_agent["facility_permissions"]
    assert "mcp__python__execute" in perms["remove_ask"]
    assert "mcp__python__execute" not in perms.get("allow", [])
    assert "mcp__python__execute" not in _ks_deny(without_agent)


# ---------------------------------------------------------------------------
# Extends clone: rewritten-prefix matcher
# ---------------------------------------------------------------------------


def test_extends_clone_of_bluesky_denied_with_rewritten_prefix():
    ctx = _build_ctx(
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

#: The connector block backing each control target on a switch-rendering
#: deployment. Spelled out rather than resolved, so a preset that stopped
#: configuring both targets fails these tests instead of quietly making them
#: assert a render with one block fewer than they claim.
_LIVE_CONNECTOR = "epics"
_VA_CONNECTOR = "virtual_accelerator"


def _switchable(**kwargs):
    """A two-target deployment: the built type is the live one and both blocks exist.

    The control-assistant preset builds a ``mock`` and carries the two blocks
    already, so naming ``epics`` as the type is all it takes to make ``live``
    and ``va`` both selectable — the deployment that used to render three ways.
    """
    return _build_ctx(control_system_type=_LIVE_CONNECTOR, **kwargs)


def test_never_denies_even_with_every_connector_leaf_armed():
    """The tier is a ceiling: armed retired leaves cannot lift a ``never``."""
    ctx = _switchable(
        writes="never",
        connector_writes={_LIVE_CONNECTOR: True, _VA_CONNECTOR: True},
        claude_code_overrides={"servers": {"bluesky": {"enabled": True}}},
    )
    assert "mcp__controls__channel_write" in _ks_deny(ctx)
    for tool in bsky.ARMING_TOOLS:
        assert f"mcp__bluesky__{tool}" in _ks_deny(ctx)


def test_never_denies_with_every_connector_leaf_disarmed():
    """Retired leaves that AGREE with a ``never`` tier change nothing either.

    The same hard deny as a deployment that spells the tier alone, with
    ``execute`` pulled from ``ask`` as on every ``never`` render.
    """
    ctx = _switchable(
        writes="never",
        connector_writes={_LIVE_CONNECTOR: False, _VA_CONNECTOR: False},
        claude_code_overrides={"servers": {"bluesky": {"enabled": True}}},
    )
    assert "mcp__controls__channel_write" in _ks_deny(ctx)
    for tool in bsky.ARMING_TOOLS:
        assert f"mcp__bluesky__{tool}" in _ks_deny(ctx)
    assert "mcp__python__execute" in ctx["facility_permissions"]["remove_ask"]


@pytest.mark.parametrize("word", ARMING)
def test_an_arming_tier_denies_nothing_with_every_connector_leaf_disarmed(word):
    """The per-call decision is the runtime switch's, not a retired leaf's."""
    ctx = _switchable(
        writes=word,
        connector_writes={_LIVE_CONNECTOR: False, _VA_CONNECTOR: False},
        claude_code_overrides={"servers": {"bluesky": {"enabled": True}}},
    )
    assert _ks_deny(ctx) == []


def test_disagreeing_connector_leaves_render_exactly_like_their_tier():
    """No three-way render: disagreeing leaves change nothing the tier decides."""
    servers = {"servers": {"bluesky": {"enabled": True}}}
    for writes in ("asked", "allowed", "never"):
        plain = _switchable(writes=writes, claude_code_overrides=servers)
        split = _switchable(
            writes=writes,
            connector_writes={_LIVE_CONNECTOR: False, _VA_CONNECTOR: True},
            claude_code_overrides=servers,
        )
        assert _ks_deny(split) == _ks_deny(plain)
        assert split["facility_permissions"] == plain["facility_permissions"]


# ---------------------------------------------------------------------------
# One-target deployments: the tier is a ceiling for every machine
# ---------------------------------------------------------------------------


def test_an_armed_block_for_an_unreachable_machine_keeps_the_hard_deny():
    """A doocs deployment carrying an armed epics block stays on its ``never``.

    The session runs the doocs connector; the epics block names a second real
    machine no target reaches, and with no simulator block the deployment does
    not render the switch. The tier is target-blind, so the stray armed leaf
    is not a second opinion the render could weigh: writes are ``never``, and
    ``channel_write`` and the arming pair are denied.
    """
    ctx = _build_ctx(
        writes="never",
        control_system_type="doocs",
        drop_connectors=(_VA_CONNECTOR,),
        connector_writes={_LIVE_CONNECTOR: True},
        claude_code_overrides={"servers": {"bluesky": {"enabled": True}}},
    )
    assert "mcp__controls__channel_write" in _ks_deny(ctx)
    for tool in bsky.ARMING_TOOLS:
        assert f"mcp__bluesky__{tool}" in _ks_deny(ctx)


def test_a_single_target_deployment_renders_exactly_its_tier():
    """An epics deployment with no VA block reads the tier and nothing else.

    One reachable target, its own retired leaf set against the tier each way:
    ``never`` with the leaf armed is the hard deny, ``allowed`` with the leaf
    disarmed renders nothing. A render that still counted targets would get
    both backwards, and the dangerous one — an armed leaf lifting a ``never`` —
    is the direction this pins first.
    """
    denied = _build_ctx(
        writes="never",
        control_system_type=_LIVE_CONNECTOR,
        drop_connectors=(_VA_CONNECTOR,),
        connector_writes={_LIVE_CONNECTOR: True},
        claude_code_overrides={"servers": {"bluesky": {"enabled": True}}},
    )
    assert "mcp__controls__channel_write" in _ks_deny(denied)
    for tool in bsky.ARMING_TOOLS:
        assert f"mcp__bluesky__{tool}" in _ks_deny(denied)

    armed = _build_ctx(
        writes="allowed",
        control_system_type=_LIVE_CONNECTOR,
        drop_connectors=(_VA_CONNECTOR,),
        connector_writes={_LIVE_CONNECTOR: False},
        claude_code_overrides={"servers": {"bluesky": {"enabled": True}}},
    )
    assert _ks_deny(armed) == []
    assert "mcp__controls__channel_write" not in armed["facility_permissions"].get("allow", [])


# ---------------------------------------------------------------------------
# Only the literal `asked` and `allowed` arm writes
# ---------------------------------------------------------------------------


def test_a_value_other_than_an_arming_word_renders_the_kill_switch():
    """``'Allowed'``, ``'Asked'``, ``'on'``, ``'true'``, ``True`` and ``1`` arm nothing.

    A value nobody can be sure of lands on the side that costs an operator a
    config edit rather than a machine.
    """
    for value in ("Allowed", "Asked", "on", "true", True, 1):
        ctx = _build_ctx(
            writes=value,
            claude_code_overrides={"servers": {"bluesky": {"enabled": True}}},
        )
        assert "mcp__controls__channel_write" in _ks_deny(ctx), (
            f"control_system.writes={value!r} must not arm writes"
        )
