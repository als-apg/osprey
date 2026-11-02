"""The composition card ``osprey init`` prints under its report.

What is pinned here, and kept apart on purpose: what the card SAYS for the
exemplar preset (derived from the resolved profile and its layers,
through the plain-text renderer), that the printed card and the plain-text
twin cannot disagree (the parity :mod:`osprey.cli.summary_card`'s tests pin
for the closing card), that a profile with none of a group's facts prints no
such group, and that the card is advisory — a derivation failure must never
fail the ``init`` that has already created the repo.

Content assertions read :func:`~osprey.cli.profile_card.format_profile_card`
rather than a CliRunner's captured stdout: the card's widest row (the MCP
server list) is longer than a default console, and an assertion on captured
output would be an assertion about wrapping. The one end-to-end assertion
against ``init``'s stdout sticks to lines that fit any width.
"""

from __future__ import annotations

import io
import json
import re
from pathlib import Path

import click
import pytest
import yaml
from click.testing import CliRunner
from rich.console import Console

from osprey.cli.build_profile import (
    GChatBridgeProfileConfig,
    NextcloudBridgeProfileConfig,
    TeamsBridgeProfileConfig,
    resolve_build_profile,
)
from osprey.cli.build_profile_model import BuildProfile
from osprey.cli.main import cli
from osprey.cli.profile_card import card_rows, format_profile_card, print_profile_card
from osprey.cli.profile_cmd import (
    _layer_profile_texts,
    _parsed_layer_deltas,
    read_layer_deltas,
)
from osprey.cli.styles import osprey_theme
from osprey.port_layout import DEFAULT_PORT_BASE, default_port, layout_ports
from tests.cli._reporters import ConsoleReporter, installed

#: Every port the exemplar lands on, at the base a profile with no
#: ``deployment.port_base`` resolves. Spelled through the layout rather than as
#: literals: the card's whole claim is that these numbers ARE the block, and a
#: literal here would keep passing after the block moved under it.
_PORTS = layout_ports(DEFAULT_PORT_BASE)

# ---------------------------------------------------------------------------
# The exemplar: the control-assistant preset, resolved the way `init` resolves
# it, with the layers the materializer parses. Session-scoped: the
# preset is packaged data, identical for every test here.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def exemplar() -> tuple[BuildProfile, dict]:
    resolved, _preset_dir = resolve_build_profile(None, "control-assistant", (), ())
    texts = _layer_profile_texts(resolved, "Exemplar", "")
    return resolved, _parsed_layer_deltas(texts)


@pytest.fixture(scope="session")
def exemplar_lines(exemplar: tuple[BuildProfile, dict]) -> list[str]:
    profile, deltas = exemplar
    return format_profile_card(profile, deltas)


def line_with(lines: list[str], *needles: str) -> str:
    """The one line carrying every needle — asserting there is exactly one."""
    hits = [line for line in lines if all(needle in line for needle in needles)]
    assert len(hits) == 1, f"expected one line with {needles!r}, got {hits!r}"
    return hits[0]


# ---------------------------------------------------------------------------
# What the card says for the exemplar
# ---------------------------------------------------------------------------


def test_the_groups_come_in_the_fixed_order(exemplar_lines: list[str]) -> None:
    titles = [line.strip() for line in exemplar_lines if line and not line.startswith("    ")]
    assert titles == [f"web terminal  :{_PORTS['nginx']}", "agent", "machine", "services"]


def test_a_blank_line_stands_before_every_group(exemplar_lines: list[str]) -> None:
    assert exemplar_lines[0] == ""
    for index, line in enumerate(exemplar_lines):
        if line and not line.startswith("    "):
            assert exemplar_lines[index - 1] == ""


def test_each_user_row_carries_rights_auth_and_port(exemplar_lines: list[str]) -> None:
    """The port a user opens is their index into the web family's band.

    Allocated by the render's own allocator, so the index reads off the port —
    roster position 1 is one port above position 0, and both are inside the web
    band rather than at whatever the profile happened to spell.
    """
    alice = line_with(exemplar_lines, "alice")
    assert "readwrite · rights asked" in alice
    assert "password" in alice
    assert alice.rstrip().endswith(f":{_PORTS['web']}")

    bob = line_with(exemplar_lines, "bob")
    assert "readonly" in bob
    assert "rights" not in bob
    assert bob.rstrip().endswith(f":{_PORTS['web'] + 1}")


def test_the_card_shows_every_family_the_deployment_publishes(exemplar_lines: list[str]) -> None:
    """The panels row says WHAT a user gets; this one says where each answers.

    Each family is a hundred-port band, and user 0 takes the first port of every
    one — so a row read at index 0 names the bands themselves.
    """
    ports = line_with(exemplar_lines, "ports (user 0)")

    for family in ("web", "artifact", "ariel", "lattice", "channel finder", "okf", "system health"):
        assert family in ports
    assert f"web :{_PORTS['web']}" in ports
    assert f"artifact :{_PORTS['artifact']}" in ports
    assert f"system health :{_PORTS['system_health']}" in ports
    # Ascending, so the row reads as the stretch of the block it is.
    numbers = [int(port) for port in re.findall(r":(\d+)", ports)]
    assert numbers == sorted(numbers)


def test_a_single_target_render_keeps_one_unqualified_write_right() -> None:
    # A render that reaches one machine reads as it always has: one item when
    # its tier allows writes, nothing when it does not.
    from types import SimpleNamespace

    from osprey.cli.profile_card import _write_rights

    armed = SimpleNamespace(
        config={
            "control_system.type": "virtual_accelerator",
            "control_system.connector.virtual_accelerator.serving": "in_process",
            "control_system.writes": "allowed",
        }
    )
    cold = SimpleNamespace(
        config={
            "control_system.type": "virtual_accelerator",
            "control_system.connector.virtual_accelerator.serving": "in_process",
        }
    )
    assert _write_rights(armed, {}, "readwrite") == ["rights allowed"]
    assert _write_rights(cold, {}, "readonly") == []


_TIER_KEY = "control_system.writes"


@pytest.mark.parametrize(
    ("word", "rights"),
    [("never", []), ("asked", ["rights asked"]), ("allowed", ["rights allowed"])],
)
def test_the_right_prints_the_ceiling_word_on_every_tier(word: str, rights: list[str]) -> None:
    """The word is the tier's own: asked and allowed cap the slider differently."""
    from types import SimpleNamespace

    from osprey.cli.profile_card import _write_rights

    host = SimpleNamespace(config={_TIER_KEY: word})
    assert _write_rights(host, {}, "plain") == rights

    # A layer's word beats the host's, in either direction.
    for other in ("never", "asked", "allowed"):
        over = SimpleNamespace(config={_TIER_KEY: other})
        assert _write_rights(over, {"r": {"config": {_TIER_KEY: word}}}, "r") == rights


@pytest.mark.parametrize("value", ["Asked", "on", True, None])
def test_a_tier_value_outside_the_three_words_prints_no_right(value: object) -> None:
    from types import SimpleNamespace

    from osprey.cli.profile_card import _write_rights

    assert _write_rights(SimpleNamespace(config={_TIER_KEY: value}), {}, "plain") == []


def test_a_retired_spelling_that_armed_prints_allowed_never_asked() -> None:
    """The rewrite of a retired armed leaf is ``allowed``; only the tier key says ``asked``."""
    from types import SimpleNamespace

    from osprey.cli.profile_card import _write_rights

    host = SimpleNamespace(config={"control_system.writes_enabled": True})
    assert _write_rights(host, {}, "plain") == ["rights allowed"]


def test_the_roster_row_prints_each_tier_word() -> None:
    profile = BuildProfile(
        name="three-tiers",
        config={
            "modules.web_terminals.enabled": True,
            "modules.web_terminals.users": [
                {"name": "dana", "index": 0, "tier": "gated"},
                {"name": "evan", "index": 1, "tier": "free"},
                {"name": "finn", "index": 2, "tier": "cold"},
            ],
        },
    )
    deltas = {
        "gated": {"config": {_TIER_KEY: "asked"}},
        "free": {"config": {_TIER_KEY: "allowed"}},
        "cold": {"config": {_TIER_KEY: "never"}},
    }

    lines = format_profile_card(profile, deltas)

    assert "gated · rights asked" in line_with(lines, "dana")
    assert "free · rights allowed" in line_with(lines, "evan")
    assert "rights" not in line_with(lines, "finn")


def test_a_switch_capable_render_answers_once_for_its_tier(exemplar_lines: list[str]) -> None:
    # The tier is target-blind (PR-3): a login whose tier allows writes may
    # write to every machine its render reaches, and the person's switch picks
    # which one. So a render that reaches two machines still answers once, and
    # the readonly tier's `never` answers with nothing.
    bob = line_with(exemplar_lines, "bob")
    assert "readonly" in bob
    assert "rights" not in bob
    assert "read-only" not in bob


def test_a_render_arming_one_block_is_allowed_on_every_machine() -> None:
    # A retired spelling that armed only the simulator's block is rewritten to
    # `control_system.writes: allowed` (PR-3), and the tier allows writes for
    # every machine — so the card no longer splits the login by target.
    profile = BuildProfile(
        name="mixed",
        config={
            "modules.web_terminals.enabled": True,
            "modules.web_terminals.users": [{"name": "dana", "index": 0, "tier": "va-write"}],
            "control_system.type": "virtual_accelerator",
            "control_system.connector.epics.gateways": ["gw.example:5064"],
            "control_system.connector.virtual_accelerator.port": 5064,
        },
    )
    deltas = {
        "va-write": {
            "config": {
                "control_system.writes_enabled": False,
                "control_system.connector.virtual_accelerator.writes_enabled": True,
            }
        }
    }

    dana = line_with(format_profile_card(profile, deltas), "dana")

    assert "va-write · rights allowed" in dana
    assert "read-only" not in dana


def test_a_shared_app_is_not_a_roster_row(exemplar_lines: list[str]) -> None:
    """The flagship's apps are `shared:` cards, not logins: the roster rows are
    its people, none of them shared, and no app takes a user's row or port."""
    for app in ("logbook", "knowledge", "channel-finder"):
        assert not [line for line in exemplar_lines if line.strip().startswith(f"{app} ")]
    for person in ("alice", "bob", "carol"):
        assert "shared" not in line_with(exemplar_lines, person, "password")


def _roster_card(access: object) -> list[str]:
    """The card for a one-user password deployment whose entry carries ``access``."""
    user: dict[str, object] = {"name": "ops", "index": 0}
    if access is not None:
        user["access"] = access
    profile = BuildProfile(
        name="roster",
        config={
            "modules.web_terminals.enabled": True,
            "modules.web_terminals.auth.method": "password",
            "modules.web_terminals.users": [user],
        },
    )
    return format_profile_card(profile, {})


def _default_tier_card(tiers: dict[str, str]) -> list[str]:
    """The card for a one-user deployment whose entry pins no render."""
    config: dict[str, object] = {
        "modules.web_terminals.enabled": True,
        "modules.web_terminals.users": [{"name": "ops", "index": 0}],
    }
    config.update({f"modules.web_terminals.{key}": value for key, value in tiers.items()})
    return format_profile_card(BuildProfile(name="roster", config=config), {})


def test_an_unpinned_user_row_shows_default_tier() -> None:
    """`default_tier` names the render an unpinned entry lands on, and it is the
    only key the card reads for it: a retired spelling is rewritten at the parse
    point, before any card exists, so no other key names a default here."""
    row = line_with(_default_tier_card({"default_tier": "readonly"}), "ops")
    assert "readonly" in row

    other_key_only = _default_tier_card({"default_render": "readwrite"})
    assert "readwrite" not in line_with(other_key_only, "ops")


def test_a_card_admitting_a_domain_says_shared() -> None:
    """`any` is not the only spelling that shares a card. The card reads the
    entry through the same predicate the deployment does, so a principal list
    carries the marker too — it used to compare the raw key against `"any"` and
    show a domain-admitting card as though it were the operator's own."""
    assert "password · shared" in line_with(_roster_card(["domain:example.com"]), "ops")
    assert "password · shared" in line_with(_roster_card(["user:alice@example.com"]), "ops")
    assert "password · shared" in line_with(_roster_card("any"), "ops")


def test_an_owner_only_card_carries_no_shared_marker() -> None:
    """`own`, an unwritten key and `[self]` are the owner-only set: the cell is
    the method alone."""
    for access in ("own", None, ["self"]):
        row = line_with(_roster_card(access), "ops")
        assert "password" in row
        assert "shared" not in row


def test_an_unreadable_access_is_named_rather_than_shown_as_unshared() -> None:
    """The card cannot say whether the entry is shared, and the plain method
    cell reads as "not shared" — the one answer that is certainly wrong for a
    value lint refuses. The card must not disagree with the refusal the
    operator is about to get from `osprey up`."""
    row = line_with(_roster_card("ANY"), "ops")

    assert "access invalid" in row
    assert "shared" not in row


def test_an_unreadable_access_never_fails_the_card() -> None:
    """The card is advisory: a roster it cannot read must still print, with the
    other users' rows intact."""
    profile = BuildProfile(
        name="roster",
        config={
            "modules.web_terminals.enabled": True,
            "modules.web_terminals.auth.method": "password",
            "modules.web_terminals.users": [
                {"name": "ops", "index": 0, "access": ["group:operators"]},
                {"name": "alice", "index": 1},
            ],
        },
    )

    lines = format_profile_card(profile, {})

    assert "access invalid" in line_with(lines, "ops")
    assert "access invalid" not in line_with(lines, "alice")


def test_the_panels_row_is_the_union_across_renders(exemplar_lines: list[str]) -> None:
    # EVENTS and BLUESKY are declared by the readwrite render, not the host
    # profile; a card that read only the host would miss them. Their labels
    # come from `web.panels.<id>.label` — they are not built-ins.
    # The order is declaration order: the host profile's `web_panels` first, in
    # the order it lists them, then whatever the layers add. JUPYTER
    # follows SYSTEM because the profile lists `jupyter` after `system-health`.
    panels = line_with(exemplar_lines, "panels")
    assert "ARIEL · CHANNELS · LATTICE · KNOWLEDGE · SYSTEM · JUPYTER · EVENTS · BLUESKY" in panels


def test_the_agent_group_names_servers_and_counts_its_toolkit(
    exemplar_lines: list[str],
) -> None:
    mcp = line_with(exemplar_lines, "mcp ")
    # The registry defaults, plus the two servers the preset switches on.
    for server in ("controls", "python", "bluesky", "health", "channel-finder"):
        assert server in mcp
    toolkit = line_with(exemplar_lines, "toolkit")
    assert re.search(r"\d+ hooks", toolkit)
    assert re.search(r"\d+ agents", toolkit)


def mcp_servers_on(lines: list[str]) -> list[str]:
    """The server names on the agent group's ``mcp`` row, in card order."""
    return line_with(lines, "mcp ").split("mcp", 1)[1].strip().split(" · ")


def test_the_mcp_row_lists_the_graph_and_dispatcher_servers_the_renders_ship(
    exemplar_lines: list[str],
) -> None:
    """The host declares a graph store; the write-armed tiers add the EVENTS
    panel. The render gates `graph` and `event_dispatcher` on exactly those, so
    the card lists both (P1.3)."""
    servers = mcp_servers_on(exemplar_lines)
    assert "graph" in servers
    assert "event_dispatcher" in servers
    assert len(servers) == len(set(servers))


def test_the_dispatcher_rides_on_the_tier_that_adds_the_events_panel(
    exemplar: tuple[BuildProfile, dict],
) -> None:
    """The host alone ships the graph server but no dispatcher entry; the
    readonly tier adds no panel, so it adds nothing; the admin tier's EVENTS
    panel is the entitlement that wires the dispatcher."""
    profile, deltas = exemplar
    host = mcp_servers_on(format_profile_card(profile, {}))
    assert "graph" in host
    assert "event_dispatcher" not in host

    readonly = mcp_servers_on(format_profile_card(profile, {"readonly": deltas["readonly"]}))
    assert readonly == host

    admin = mcp_servers_on(format_profile_card(profile, {"admin": deltas["admin"]}))
    assert admin == [*host, "event_dispatcher"]


def _gated_card(config: dict, deltas: dict | None = None) -> list[str]:
    """The servers a bare profile's card lists, for a ``config:`` bag."""
    profile = BuildProfile(name="gates", config=config)
    return mcp_servers_on(format_profile_card(profile, deltas or {}))


def test_a_declared_graph_store_is_what_ships_the_graph_server() -> None:
    assert "graph" not in _gated_card({})
    # The fully-defaulted block counts, dotted or nested, as the render reads it.
    assert "graph" in _gated_card({"services.graphdb": {}})
    assert "graph" in _gated_card({"services": {"graphdb": {"image": "neo4j:5"}}})
    # The bare `graphdb:` YAML reads as None: no store.
    assert "graph" not in _gated_card({"services.graphdb": None})


def test_a_tier_can_declare_the_graph_store_its_host_does_not() -> None:
    servers = _gated_card({}, {"curious": {"config": {"services.graphdb.image": "neo4j:5"}}})
    assert "graph" in servers


def test_a_malformed_graph_store_drops_the_server_but_not_the_card() -> None:
    """The render warns and leaves the graph server out; the card agrees."""
    servers = _gated_card({"services.graphdb.port_host": "not-a-port"})
    assert "graph" not in servers
    assert "controls" in servers


def test_an_events_block_no_layer_selects_wires_no_dispatcher() -> None:
    """The build switches off every block for a tab its profile does not
    select, so a spelled `web.panels.events` block alone grants nothing."""
    assert "event_dispatcher" not in _gated_card({"web.panels.events": {"label": "Events"}})
    assert "event_dispatcher" in _gated_card({}, {"ops": {"web_panels": ["events"]}})
    # Selected, the block's own facts stay and the panel reads as on.
    selected = _gated_card(
        {"web.panels.events": {"label": "Events"}}, {"ops": {"web_panels": ["events"]}}
    )
    assert "event_dispatcher" in selected


def test_the_card_reads_the_renders_own_gate_predicates(
    exemplar: tuple[BuildProfile, dict], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The two gates are the render's functions, not a restatement of them:
    change what the render's predicate answers and the card follows."""
    import osprey.cli.templates.claude_code as claude_code
    import osprey.deployment.web_terminals.roster as renders

    profile, deltas = exemplar
    monkeypatch.setattr(claude_code, "_graphdb_configured", lambda config: False)
    monkeypatch.setattr(renders, "config_needs_dispatcher_token", lambda config: False)

    servers = mcp_servers_on(format_profile_card(profile, deltas))
    assert "graph" not in servers
    assert "event_dispatcher" not in servers


def test_the_model_row_names_the_provider_default_when_the_profile_names_none(
    exemplar_lines: list[str],
) -> None:
    """The preset names no model, so the card says which one answers and why."""
    model = line_with(exemplar_lines, "model ")
    assert "anthropic" in model
    assert "(provider default) claude-sonnet-5" in model


def test_the_model_row_shows_a_named_model_without_its_vendor_prefix() -> None:
    lines = format_profile_card(
        BuildProfile(name="named", provider="als-apg", model="claude-haiku-4-5-20251001"), {}
    )
    model = line_with(lines, "model ")
    assert "Haiku 4.5 (claude-haiku-4-5-20251001)" in model
    assert "Claude" not in model


def test_the_machine_group_reads_connector_archiver_and_channels(
    exemplar_lines: list[str],
) -> None:
    control = line_with(exemplar_lines, "control ")
    # The baseline connector type, spelled the way the card spells one
    # (underscores to spaces), then the two simulator ports the preset
    # declares: the sandbox on 5064, which the baseline names, and the
    # stand-in, which `live_standin: true` places at the layout's stand-in slot.
    assert control.split()[1:3] == ["virtual", "accelerator"]
    assert "EPICS :5064" in control
    assert f"live stand-in :{default_port('va_standin')}" in control
    archiver = line_with(exemplar_lines, "archiver")
    assert "mongodb · 30 d retention" in archiver
    channels = line_with(exemplar_lines, "channels")
    assert "graph finder" in channels
    assert "tier" not in channels


def test_the_services_group_names_the_injected_stack(exemplar_lines: list[str]) -> None:
    bluesky = line_with(exemplar_lines, "bluesky ", f":{_PORTS['bluesky']}")
    assert f"tiled :{_PORTS['tiled']}" in bluesky
    assert f"web :{_PORTS['bluesky_web']}" in bluesky
    # The trailing space keeps the needle off the agent row's `event_dispatcher`.
    dispatch = line_with(exemplar_lines, "dispatch ")
    assert "1 worker · triggers " in dispatch


def test_the_services_group_names_every_chat_bridge_the_profile_carries() -> None:
    # One row per bridge block, each naming its platform the way its
    # documentation does — the card is where an operator confirms the bridge
    # they authored is the bridge the build will deploy.
    profile = BuildProfile(
        name="bridged",
        nextcloud_bridge=NextcloudBridgeProfileConfig(),
        gchat_bridge=GChatBridgeProfileConfig(),
        teams_bridge=TeamsBridgeProfileConfig(),
    )
    named = [
        row["value"]
        for row in card_rows(profile, {})
        if row["group"] == "services" and row["label"] == "bridge"
    ]
    assert named == ["Nextcloud Talk", "Google Chat", "Microsoft Teams"]


def test_a_profile_without_a_teams_block_gets_no_teams_row() -> None:
    lines = format_profile_card(BuildProfile(name="unbridged"), {})
    assert not any("Microsoft Teams" in line for line in lines)


# ---------------------------------------------------------------------------
# What the card leaves out
# ---------------------------------------------------------------------------


def test_a_bare_profile_gets_no_web_machine_or_services_group() -> None:
    lines = format_profile_card(BuildProfile(name="bare"), {})
    text = "\n".join(lines)
    assert "web terminal" not in text
    assert "machine" not in text
    assert "services" not in text
    # The agent group still stands: the registry's default servers are what a
    # bare profile's render would get, and saying so is the card's job.
    assert "  agent" in lines
    assert any("controls" in line for line in lines)


def test_a_profile_with_nothing_to_say_prints_nothing() -> None:
    # No web tier, no model, no services — and the one row a bare profile
    # would still get (the registry's default servers) suppressed the same way
    # a facility profile would do it.
    profile = BuildProfile(
        name="silent",
        config={
            "claude_code.servers.controls.enabled": False,
            "claude_code.servers.python.enabled": False,
            "claude_code.servers.osprey_workspace.enabled": False,
            "claude_code.servers.ariel.enabled": False,
            "claude_code.servers.osprey_facility_knowledge.enabled": False,
        },
    )
    assert format_profile_card(profile, {}) == []


# ---------------------------------------------------------------------------
# Parity, and the card's altitude
# ---------------------------------------------------------------------------


def recording_console(*, terminal: bool) -> tuple[Console, io.StringIO]:
    buffer = io.StringIO()
    return (
        Console(
            file=buffer,
            theme=osprey_theme,
            force_terminal=terminal,
            color_system="standard" if terminal else None,
            no_color=not terminal,
            width=300,
        ),
        buffer,
    )


def test_the_printed_card_is_the_plain_renderer_byte_for_byte(
    exemplar: tuple[BuildProfile, dict],
) -> None:
    profile, deltas = exemplar
    console, buffer = recording_console(terminal=False)
    with installed(ConsoleReporter(console, color=False)):
        print_profile_card(profile, deltas)
    assert buffer.getvalue().splitlines() == format_profile_card(profile, deltas)


def test_the_styled_card_strips_to_the_plain_renderer(
    exemplar: tuple[BuildProfile, dict],
) -> None:
    profile, deltas = exemplar
    console, buffer = recording_console(terminal=True)
    with installed(ConsoleReporter(console, color=True)):
        print_profile_card(profile, deltas)
    styled = buffer.getvalue()
    assert "\x1b[" in styled  # it really was styled
    stripped = [line.rstrip() for line in re.sub(r"\x1b\[[0-9;]*m", "", styled).splitlines()]
    assert stripped == [line.rstrip() for line in format_profile_card(profile, deltas)]


def test_a_derivation_failure_is_swallowed() -> None:
    # The card is advisory: whatever it meets, `init` has already succeeded.
    print_profile_card(None, {})  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# End to end: `osprey init` prints the card under its report
# ---------------------------------------------------------------------------


def test_init_prints_the_card_after_its_report(tmp_path: Path) -> None:
    runner = CliRunner()
    result = runner.invoke(
        cli,
        ["init", str(tmp_path / "exemplar"), "--preset", "control-assistant", "--no-git"],
        catch_exceptions=False,
    )
    assert result.exit_code == 0, result.output
    # Short lines only — the wide rows are the plain renderer's tests' business.
    assert f"  web terminal  :{_PORTS['nginx']}" in result.output
    assert "shared" in result.output
    report_at = result.output.index("✓ Created")
    assert result.output.index(f"  web terminal  :{_PORTS['nginx']}") > report_at


# ---------------------------------------------------------------------------
# read_layer_deltas: the render facts the card derives from, read off disk
# ---------------------------------------------------------------------------


@pytest.fixture
def exemplar_config(lifecycle_repo: Path) -> dict:
    """The exemplar repo's ``config:`` block, straight off its ``profile.yml``."""
    document = yaml.safe_load((lifecycle_repo / "profile.yml").read_text(encoding="utf-8"))
    return document["config"]


#: The layers the exemplar repo carries: its three tiers, then its three apps.
_EXEMPLAR_LAYERS = ["admin", "readonly", "readwrite", "channel-finder", "knowledge", "logbook"]


def test_read_layer_deltas_reads_one_delta_per_layer_file(lifecycle_repo: Path) -> None:
    deltas = read_layer_deltas(lifecycle_repo)

    # Keyed by file stem, tiers before apps, sorted within each.
    assert list(deltas) == _EXEMPLAR_LAYERS
    # Every one was really read, and it is the file's own content.
    assert all(delta for delta in deltas.values())
    assert deltas["logbook"] == yaml.safe_load(
        (lifecycle_repo / "apps" / "logbook.yml").read_text(encoding="utf-8")
    )
    assert deltas["admin"] == yaml.safe_load(
        (lifecycle_repo / "tiers" / "admin.yml").read_text(encoding="utf-8")
    )


def test_read_layer_deltas_never_consults_the_profile(lifecycle_repo: Path) -> None:
    # The layer set is the files on disk: a profile whose roster names no tier
    # and no app changes nothing.
    before = read_layer_deltas(lifecycle_repo)
    profile_path = lifecycle_repo / "profile.yml"
    document = yaml.safe_load(profile_path.read_text(encoding="utf-8"))
    document["config"] = {}
    profile_path.write_text(yaml.safe_dump(document), encoding="utf-8")

    assert read_layer_deltas(lifecycle_repo) == before
    assert list(before) == _EXEMPLAR_LAYERS


def test_read_layer_deltas_reads_a_layer_file_added_to_the_directory(
    lifecycle_repo: Path,
) -> None:
    # A file in a layer directory IS a layer the build renders, so the reader
    # reports it as well; a dot-file or a nested tree is not a layer.
    (lifecycle_repo / "tiers" / "operator.yml").write_text("provider: cborg\n", encoding="utf-8")
    (lifecycle_repo / "tiers" / ".draft.yml").write_text("provider: x\n", encoding="utf-8")
    (lifecycle_repo / "apps" / "nested").mkdir()
    (lifecycle_repo / "apps" / "nested" / "deep.yml").write_text("provider: x\n", encoding="utf-8")

    deltas = read_layer_deltas(lifecycle_repo)

    assert deltas["operator"] == {"provider": "cborg"}
    assert set(deltas) == {*_EXEMPLAR_LAYERS, "operator"}


def test_read_layer_deltas_drops_a_removed_layer(lifecycle_repo: Path) -> None:
    (lifecycle_repo / "apps" / "logbook.yml").unlink()

    deltas = read_layer_deltas(lifecycle_repo)

    assert "logbook" not in deltas
    assert len(deltas) == len(_EXEMPLAR_LAYERS) - 1


def test_read_layer_deltas_is_empty_for_a_repo_without_layers(tmp_path: Path) -> None:
    assert read_layer_deltas(tmp_path) == {}


def test_read_layer_deltas_refuses_an_unparsable_file(lifecycle_repo: Path) -> None:
    (lifecycle_repo / "tiers" / "admin.yml").write_text("config:\n  - [\n", encoding="utf-8")

    with pytest.raises(click.ClickException) as excinfo:
        read_layer_deltas(lifecycle_repo)

    message = str(excinfo.value)
    # The path named is the one the operator opens, and the parser's own
    # complaint rides along with it.
    assert "tiers/admin.yml" in message
    assert "not valid YAML" in message


def test_read_layer_deltas_refuses_a_file_that_is_not_a_mapping(lifecycle_repo: Path) -> None:
    # Valid YAML, but nothing a delta can be merged from. Carried as an empty
    # delta it would read as a layer that overrides nothing.
    (lifecycle_repo / "apps" / "knowledge.yml").write_text("- readonly\n", encoding="utf-8")

    with pytest.raises(click.ClickException) as excinfo:
        read_layer_deltas(lifecycle_repo)

    assert "apps/knowledge.yml is not a YAML mapping" in str(excinfo.value)


# ---------------------------------------------------------------------------
# card_rows: the same cells, flattened for a machine reader
# ---------------------------------------------------------------------------


def group_and_label_pairs(lines: list[str]) -> list[tuple[str, str]]:
    """``(group, label)`` per row of a plain card, read off the card itself.

    Against the real layout rather than an assumed one: a group title is the
    line indented once, a row is indented twice, and a row's label is its first
    column — everything up to the padding that separates it from the next one,
    which is never narrower than the gutter.
    """
    pairs: list[tuple[str, str]] = []
    group = ""
    for line in lines:
        if not line.strip():
            continue
        if not line.startswith("    "):
            # The title, without the address the web tier hangs beside it.
            group = re.split(r" {2,}", line.strip())[0]
            continue
        pairs.append((group, re.split(r" {3,}", line[4:], maxsplit=1)[0]))
    return pairs


def test_card_rows_carry_exactly_the_three_string_keys(
    exemplar: tuple[BuildProfile, dict],
) -> None:
    profile, deltas = exemplar
    rows = card_rows(profile, deltas)

    assert rows
    for row in rows:
        assert set(row) == {"group", "label", "value"}
        assert all(isinstance(value, str) for value in row.values())


def test_card_rows_follow_the_printed_card_row_for_row(
    exemplar: tuple[BuildProfile, dict], exemplar_lines: list[str]
) -> None:
    """The rows ARE the card: same groups, same labels, same order.

    The one assertion that keeps a JSON reader and an operator reading the same
    deployment — a row added, dropped or reordered on either side breaks here.
    """
    profile, deltas = exemplar
    rows = card_rows(profile, deltas)

    assert [(row["group"], row["label"]) for row in rows] == group_and_label_pairs(exemplar_lines)


def test_card_rows_say_what_their_printed_line_says(
    exemplar: tuple[BuildProfile, dict], exemplar_lines: list[str]
) -> None:
    # Label plus value is the whole line, once the column padding that only
    # exists to line the card up is collapsed.
    profile, deltas = exemplar
    printed = [line for line in exemplar_lines if line.startswith("    ")]
    rows = card_rows(profile, deltas)

    assert len(rows) == len(printed)
    for row, line in zip(rows, printed, strict=True):
        assert " ".join(f"{row['label']} {row['value']}".split()) == " ".join(line.split())


def test_card_rows_are_empty_when_the_card_is() -> None:
    # The profile with nothing to say: no groups, so no rows either.
    profile = BuildProfile(
        name="silent",
        config={
            "claude_code.servers.controls.enabled": False,
            "claude_code.servers.python.enabled": False,
            "claude_code.servers.osprey_workspace.enabled": False,
            "claude_code.servers.ariel.enabled": False,
            "claude_code.servers.osprey_facility_knowledge.enabled": False,
        },
    )

    assert format_profile_card(profile, {}) == []
    assert card_rows(profile, {}) == []


def test_card_rows_carry_the_layer_deltas_read_off_disk(
    lifecycle_repo: Path, exemplar_config: dict
) -> None:
    """A deployed repo end to end: its own profile, its own tier files.

    The rights on a roster row come from the render's own file, not from the
    host profile — so the same profile read with no deltas says something
    different about the same login.
    """
    profile = BuildProfile(name="als-exemplar", config=exemplar_config)
    deltas = read_layer_deltas(lifecycle_repo)

    rows = card_rows(profile, deltas)

    roster = {row["label"]: row["value"] for row in rows if row["group"] == "web terminal"}
    assert {"alice", "bob"} <= set(roster)
    assert roster["alice"].startswith("readwrite")
    assert roster["bob"].startswith("readonly")
    # The readonly tier states `never` in its own file; the host it is merged
    # over asks for writes, which is what bob reads as with no delta.
    assert "rights" not in roster["bob"]
    bare = {row["label"]: row["value"] for row in card_rows(profile, {})}
    assert "rights asked" in bare["bob"]
    assert bare["bob"] != roster["bob"]


# ---------------------------------------------------------------------------
# `osprey profile card`: the same card, on demand, for a repo that already exists
# ---------------------------------------------------------------------------


@pytest.fixture
def deployed(lifecycle_repo: Path) -> tuple[BuildProfile, dict]:
    """The exemplar repo's own profile, and the deltas its tier files hold.

    Resolved the way the command resolves them, so a test asserting on what the
    command printed is comparing against the same two inputs it printed from.
    """
    profile, _profile_dir = resolve_build_profile(lifecycle_repo / "profile.yml", None)
    return profile, read_layer_deltas(lifecycle_repo)


def break_the_card(repo: Path) -> None:
    """Leave the repo with a profile that resolves and a card that cannot.

    ``claude_code.servers`` as a string: a value the profile loader accepts —
    the block is the operator's to write — and the card's MCP reader walks as a
    mapping. The deeper ``claude_code.servers.<name>`` keys go first, because a
    profile spelling one subtree both ways is refused before any card is
    derived, and that refusal is a different failure from this one.
    """
    path = repo / "profile.yml"
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    config = document["config"]
    for key in [key for key in config if key.startswith("claude_code.servers.")]:
        del config[key]
    config["claude_code.servers"] = "all"
    path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")


def test_the_card_command_prints_the_plain_renderer_line_for_line(
    lifecycle_repo: Path, deployed: tuple[BuildProfile, dict], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Run from inside a repo, the verb prints the card and nothing else.

    The same equality the ``init`` card's parity test pins, asserted through
    the command: what an operator reads here is the plain renderer's output,
    line for line, with no report above it and no summary below.
    """
    profile, deltas = deployed
    # One delta per layer the exemplar deploys — the facts the roster rows
    # are made of, so an empty read would print a card that says less.
    assert list(deltas) == _EXEMPLAR_LAYERS
    monkeypatch.chdir(lifecycle_repo)

    result = CliRunner().invoke(cli, ["profile", "card"], catch_exceptions=False)

    assert result.exit_code == 0, result.stderr
    assert result.stdout.splitlines() == format_profile_card(profile, deltas)


def test_the_card_command_reads_the_repo_the_repo_flag_names(
    lifecycle_repo: Path,
    deployed: tuple[BuildProfile, dict],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Standing nowhere near the deployment: --repo moves the search, and the
    # card is the one that repo's profile derives.
    profile, deltas = deployed
    monkeypatch.chdir(tmp_path)

    result = CliRunner().invoke(
        cli, ["profile", "card", "--repo", str(lifecycle_repo)], catch_exceptions=False
    )

    assert result.exit_code == 0, result.stderr
    assert result.stdout.splitlines() == format_profile_card(profile, deltas)


def test_the_json_card_is_one_document_of_group_label_value_rows(
    lifecycle_repo: Path, deployed: tuple[BuildProfile, dict], monkeypatch: pytest.MonkeyPatch
) -> None:
    profile, deltas = deployed
    monkeypatch.chdir(lifecycle_repo)

    result = CliRunner().invoke(cli, ["profile", "card", "--json"], catch_exceptions=False)

    assert result.exit_code == 0, result.stderr
    rows = json.loads(result.stdout)
    assert rows == card_rows(profile, deltas)
    for row in rows:
        assert set(row) == {"group", "label", "value"}
    # The document is the whole of stdout: a reader parses it without stripping
    # anything off either end.
    assert result.stdout.strip() == json.dumps(rows)


def test_the_json_card_lists_graph_and_the_dispatcher_on_the_agent_row(
    lifecycle_repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The init repo's card, through the command: the host's graph store and the
    write-armed tiers' EVENTS panel both reach the agent row (P1.3)."""
    monkeypatch.chdir(lifecycle_repo)

    result = CliRunner().invoke(cli, ["profile", "card", "--json"], catch_exceptions=False)

    assert result.exit_code == 0, result.stderr
    agent = {
        row["label"]: row["value"] for row in json.loads(result.stdout) if row["group"] == "agent"
    }
    servers = agent["mcp"].split(" · ")
    assert "graph" in servers
    assert "event_dispatcher" in servers


def test_a_broken_layer_file_never_reaches_the_json_document(
    lifecycle_repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An unparsable layer refuses, and the refusal lands on stderr alone."""
    (lifecycle_repo / "apps" / "logbook.yml").write_text("- not a mapping\n", encoding="utf-8")
    monkeypatch.chdir(lifecycle_repo)

    result = CliRunner().invoke(cli, ["profile", "card", "--json"], catch_exceptions=False)

    assert result.exit_code == 1
    assert result.stdout == ""
    assert "apps/logbook.yml is not a YAML mapping" in result.stderr


@pytest.mark.parametrize("mode", [[], ["--json"]], ids=["plain", "json"])
def test_an_underivable_card_refuses_in_both_modes(
    lifecycle_repo: Path, monkeypatch: pytest.MonkeyPatch, mode: list[str]
) -> None:
    """The card IS the output here, so a shape it cannot read is a failure.

    Unlike ``init``'s card, which is advisory and swallows what it cannot
    derive: there the repo is already made, here there is nothing else to
    print. Both modes fail the same way, and neither prints half a card first.
    """
    break_the_card(lifecycle_repo)
    monkeypatch.chdir(lifecycle_repo)

    result = CliRunner().invoke(cli, ["profile", "card", *mode], catch_exceptions=False)

    assert result.exit_code == 1
    assert "Cannot derive the card" in result.stderr
    assert result.stdout == ""


def test_the_card_command_outside_a_repo_says_where_it_looked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The shared resolver's refusal, not one of this command's own.
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)

    result = CliRunner().invoke(cli, ["profile", "card"], catch_exceptions=False)

    assert result.exit_code == 1
    assert "No OSPREY deployment repo found" in result.stderr
    assert result.stdout == ""
