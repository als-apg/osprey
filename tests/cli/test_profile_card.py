"""The composition card ``osprey init`` prints under its report.

What is pinned here, and kept apart on purpose: what the card SAYS for the
exemplar preset (derived from the resolved profile and its persona deltas,
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

from osprey.cli.build_profile import resolve_build_profile
from osprey.cli.build_profile_model import BuildProfile
from osprey.cli.main import cli
from osprey.cli.phase_reporter import PhaseReporter, install_reporter
from osprey.cli.profile_card import card_rows, format_profile_card, print_profile_card
from osprey.cli.profile_cmd import (
    _parsed_persona_deltas,
    _persona_profile_texts,
    read_persona_deltas,
)
from osprey.cli.styles import osprey_theme
from osprey.port_layout import DEFAULT_PORT_BASE, default_port, layout_ports

#: Every port the exemplar lands on, at the base a profile with no
#: ``deployment.port_base`` resolves. Spelled through the layout rather than as
#: literals: the card's whole claim is that these numbers ARE the block, and a
#: literal here would keep passing after the block moved under it.
_PORTS = layout_ports(DEFAULT_PORT_BASE)

# ---------------------------------------------------------------------------
# The exemplar: the control-assistant preset, resolved the way `init` resolves
# it, with the persona deltas the materializer parses. Session-scoped: the
# preset is packaged data, identical for every test here.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def exemplar() -> tuple[BuildProfile, dict]:
    resolved, _preset_dir = resolve_build_profile(None, "control-assistant", (), ())
    texts = _persona_profile_texts(resolved, "Exemplar", "", "control-assistant")
    return resolved, _parsed_persona_deltas(texts)


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
    assert "readwrite · va rights approval-gated · standin rights approval-gated" in alice
    assert "password" in alice
    assert alice.rstrip().endswith(f":{_PORTS['web']}")

    bob = line_with(exemplar_lines, "bob")
    assert "readonly" in bob
    assert "rights approval-gated" not in bob
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
    # A render that reaches one machine names no target on the rights item —
    # a target-by-target posture needs a second half the render does not have.
    # Every one of the exemplar's write tiers now reaches two machines, so the
    # one-target reading is pinned at the renderer's own seam instead.
    from types import SimpleNamespace

    from osprey.cli.profile_card import _write_rights

    armed = SimpleNamespace(
        config={"control_system.type": "mock", "control_system.writes_enabled": True}
    )
    cold = SimpleNamespace(config={"control_system.type": "mock"})
    assert _write_rights(armed, {}, "readwrite") == ["rights approval-gated"]
    assert _write_rights(cold, {}, "readonly") == []


def test_a_switch_capable_render_answers_per_target(exemplar_lines: list[str]) -> None:
    # The readonly tier pins BOTH connector types off by name, which is also
    # what makes its render switch-capable: two machines a session could be
    # pointed at, so the card answers for each rather than once for the login.
    bob = line_with(exemplar_lines, "bob")
    assert "readonly · live read-only · va read-only" in bob


def test_a_mixed_persona_arms_only_the_target_its_own_block_names() -> None:
    # Write posture is per connector type: `control_system.writes_enabled`
    # answers only for a type whose own block says nothing. A persona that
    # says `false` deployment-wide and `true` under the simulator's block is
    # armed on the simulator and read-only on the live machine — and the card
    # has to show both halves, because which one carries the write path is
    # the whole point of saying it.
    profile = BuildProfile(
        name="mixed",
        config={
            "modules.web_terminals.enabled": True,
            "modules.web_terminals.users": [{"name": "dana", "index": 0, "persona": "va-write"}],
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

    assert "va-write · live read-only · va rights approval-gated" in dana


def test_a_shared_card_says_so(exemplar_lines: list[str]) -> None:
    """The standalone card is opened with any roster login's password, so the
    card says both: the method, and that the entry is shared."""
    logbook = line_with(exemplar_lines, "logbook", f":{_PORTS['web'] + 2}")
    assert "password · shared" in logbook
    assert "no login" not in logbook


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


def test_a_card_admitting_a_domain_says_shared() -> None:
    """`any` is not the only spelling that shares a card. The card reads the
    entry through the same predicate the deployment does, so a principal list
    carries the marker too — it used to compare the raw key against `"any"` and
    show a domain-admitting card as though it were the operator's own."""
    assert "password · shared" in line_with(_roster_card(["domain:lbl.gov"]), "ops")
    assert "password · shared" in line_with(_roster_card(["user:alice@lbl.gov"]), "ops")
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


def test_the_panels_row_is_the_union_across_personas(exemplar_lines: list[str]) -> None:
    # EVENTS and BLUESKY are declared by the readwrite persona, not the host
    # profile; a card that read only the host would miss them. Their labels
    # come from `web.panels.<id>.label` — they are not built-ins.
    # The order is declaration order: the host profile's `web_panels` first, in
    # the order it lists them, then whatever the persona deltas add. JUPYTER
    # follows SYSTEM because the profile lists `jupyter` after `system-health`.
    panels = line_with(exemplar_lines, "panels")
    assert "ARIEL · CHANNELS · KNOWLEDGE · SYSTEM · JUPYTER · EVENTS · BLUESKY" in panels


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


def test_the_machine_group_reads_connector_archiver_and_channels(
    exemplar_lines: list[str],
) -> None:
    control = line_with(exemplar_lines, "control ")
    # The baseline connector type, spelled the way the card spells one
    # (underscores to spaces), then the two simulator ports the preset
    # declares: the sandbox on 5064 and the stand-in the baseline names, which
    # `live_standin: true` places at the layout's stand-in slot.
    assert "live standin" in control
    assert "EPICS :5064" in control
    assert f"live stand-in :{default_port('va_standin')}" in control
    archiver = line_with(exemplar_lines, "archiver")
    assert "mongodb · 30 d retention" in archiver
    channels = line_with(exemplar_lines, "channels")
    assert "graph finder · tier 3" in channels


def test_the_services_group_names_the_injected_stack(exemplar_lines: list[str]) -> None:
    bluesky = line_with(exemplar_lines, "bluesky ", f":{_PORTS['bluesky']}")
    assert f"tiled :{_PORTS['tiled']}" in bluesky
    assert f"web :{_PORTS['bluesky_web']}" in bluesky
    dispatch = line_with(exemplar_lines, "dispatch")
    assert "1 worker · triggers " in dispatch


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


class RecordingReporter(PhaseReporter):
    """A real reporter whose console is a buffer, not the terminal."""

    def __init__(self, console: Console, *, color: bool) -> None:
        super().__init__(color=color)
        self._console = console

    def out(self) -> Console:
        return self._console


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
    previous = install_reporter(RecordingReporter(console, color=False))
    try:
        print_profile_card(profile, deltas)
    finally:
        install_reporter(previous)
    assert buffer.getvalue().splitlines() == format_profile_card(profile, deltas)


def test_the_styled_card_strips_to_the_plain_renderer(
    exemplar: tuple[BuildProfile, dict],
) -> None:
    profile, deltas = exemplar
    console, buffer = recording_console(terminal=True)
    previous = install_reporter(RecordingReporter(console, color=True))
    try:
        print_profile_card(profile, deltas)
    finally:
        install_reporter(previous)
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
# read_persona_deltas: the persona facts the card derives from, read off disk
# ---------------------------------------------------------------------------


@pytest.fixture
def exemplar_config(lifecycle_repo: Path) -> dict:
    """The exemplar repo's ``config:`` block, straight off its ``profile.yml``."""
    document = yaml.safe_load((lifecycle_repo / "profile.yml").read_text(encoding="utf-8"))
    return document["config"]


def test_read_persona_deltas_reads_one_delta_per_catalog_entry(
    lifecycle_repo: Path, exemplar_config: dict
) -> None:
    deltas = read_persona_deltas(lifecycle_repo, exemplar_config)

    assert set(deltas) == {"readonly", "readwrite", "admin", "logbook", "knowledge"}
    # Every one was really read: a delta the reader could not open is empty.
    assert all(delta for delta in deltas.values())
    # And it is the file's own content, not the host profile's.
    assert deltas["logbook"] == yaml.safe_load(
        (lifecycle_repo / "personas" / "logbook.yml").read_text(encoding="utf-8")
    )


def test_read_persona_deltas_ignores_a_file_the_catalog_does_not_name(
    lifecycle_repo: Path, exemplar_config: dict
) -> None:
    # A persona the profile dropped leaves its file behind. The catalog is the
    # list, never the directory, so the stale file is invisible.
    (lifecycle_repo / "personas" / "old-admin.yml").write_text("provider: gone\n", encoding="utf-8")

    deltas = read_persona_deltas(lifecycle_repo, exemplar_config)

    assert "old-admin" not in deltas
    assert len(deltas) == 5


def test_read_persona_deltas_warns_when_a_persona_file_is_missing(
    lifecycle_repo: Path, exemplar_config: dict, capsys: pytest.CaptureFixture[str]
) -> None:
    (lifecycle_repo / "personas" / "logbook.yml").unlink()

    deltas = read_persona_deltas(lifecycle_repo, exemplar_config)

    assert deltas["logbook"] == {}
    # The persona is still deployed, so it keeps its place in the result.
    assert len(deltas) == 5
    assert all(delta for name, delta in deltas.items() if name != "logbook")
    err = capsys.readouterr().err
    assert "no persona file at personas/logbook.yml" in err
    assert "'logbook'" in err
    assert err.count("no persona file") == 1


def test_read_persona_deltas_warns_for_a_preset_reference(
    lifecycle_repo: Path, exemplar_config: dict, capsys: pytest.CaptureFixture[str]
) -> None:
    # A catalog entry may name a bundled preset instead of a file in this repo.
    # There is nothing under the repo root to read, and that is not an error.
    catalog = exemplar_config["modules.web_terminals"]["personas"]
    catalog["readonly"]["build_profile"] = "control-assistant-readonly"

    deltas = read_persona_deltas(lifecycle_repo, exemplar_config)

    assert deltas["readonly"] == {}
    assert len(deltas) == 5
    err = capsys.readouterr().err
    assert "'readonly'" in err
    assert "'control-assistant-readonly' is not a file in this repo" in err


def test_read_persona_deltas_refuses_an_unparsable_file(
    lifecycle_repo: Path, exemplar_config: dict
) -> None:
    (lifecycle_repo / "personas" / "admin.yml").write_text("config:\n  - [\n", encoding="utf-8")

    with pytest.raises(click.ClickException) as excinfo:
        read_persona_deltas(lifecycle_repo, exemplar_config)

    message = str(excinfo.value)
    # The path named is the one the operator opens, and the parser's own
    # complaint rides along with it.
    assert "personas/admin.yml" in message
    assert "not valid YAML" in message


def test_read_persona_deltas_refuses_a_file_that_is_not_a_mapping(
    lifecycle_repo: Path, exemplar_config: dict
) -> None:
    # Valid YAML, but nothing a delta can be merged from. Carried as an empty
    # delta it would read as a persona that overrides nothing.
    (lifecycle_repo / "personas" / "admin.yml").write_text("- readonly\n", encoding="utf-8")

    with pytest.raises(click.ClickException) as excinfo:
        read_persona_deltas(lifecycle_repo, exemplar_config)

    assert "personas/admin.yml is not a YAML mapping" in str(excinfo.value)


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


def test_card_rows_carry_the_persona_deltas_read_off_disk(
    lifecycle_repo: Path, exemplar_config: dict
) -> None:
    """A deployed repo end to end: its own profile, its own persona files.

    The rights on a roster row come from the persona's own file, not from the
    host profile — so the same profile read with no deltas says something
    different about the same login.
    """
    profile = BuildProfile(name="als-exemplar", config=exemplar_config)
    deltas = read_persona_deltas(lifecycle_repo, exemplar_config)

    rows = card_rows(profile, deltas)

    roster = {row["label"]: row["value"] for row in rows if row["group"] == "web terminal"}
    assert {"alice", "bob"} <= set(roster)
    assert roster["alice"].startswith("readwrite")
    assert roster["bob"].startswith("readonly")
    # The readonly persona pins both connectors read-only in its own file.
    assert "read-only" in roster["bob"]
    bare = {row["label"]: row["value"] for row in card_rows(profile, {})}
    assert bare["bob"] != roster["bob"]


# ---------------------------------------------------------------------------
# `osprey profile card`: the same card, on demand, for a repo that already exists
# ---------------------------------------------------------------------------


@pytest.fixture
def deployed(lifecycle_repo: Path) -> tuple[BuildProfile, dict]:
    """The exemplar repo's own profile, and the deltas its persona files hold.

    Resolved the way the command resolves them, so a test asserting on what the
    command printed is comparing against the same two inputs it printed from.
    """
    profile, _profile_dir = resolve_build_profile(lifecycle_repo / "profile.yml", None)
    return profile, read_persona_deltas(lifecycle_repo, profile.config)


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
    # One delta per persona the exemplar deploys — the facts the roster rows
    # are made of, so an empty read would print a card that says less.
    assert len(deltas) == 5
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


def test_a_persona_warning_never_reaches_the_json_document(
    lifecycle_repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A missing persona file warns, and the warning lands on stderr alone."""
    (lifecycle_repo / "personas" / "logbook.yml").unlink()
    monkeypatch.chdir(lifecycle_repo)

    result = CliRunner().invoke(cli, ["profile", "card", "--json"], catch_exceptions=False)

    assert result.exit_code == 0, result.stderr
    rows = json.loads(result.stdout)
    assert rows
    assert result.stdout.strip() == json.dumps(rows)
    assert "no persona file at personas/logbook.yml" in result.stderr


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
