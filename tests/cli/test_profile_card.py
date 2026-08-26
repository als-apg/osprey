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
import re
from pathlib import Path

import pytest
from click.testing import CliRunner
from rich.console import Console

from osprey.cli.build_profile import resolve_build_profile
from osprey.cli.build_profile_model import BuildProfile
from osprey.cli.main import cli
from osprey.cli.phase_reporter import PhaseReporter, install_reporter
from osprey.cli.profile_card import format_profile_card, print_profile_card
from osprey.cli.profile_cmd import _parsed_persona_deltas, _persona_profile_texts
from osprey.cli.styles import osprey_theme

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
    assert titles == ["web terminal  :9080", "agent", "machine", "services"]


def test_a_blank_line_stands_before_every_group(exemplar_lines: list[str]) -> None:
    assert exemplar_lines[0] == ""
    for index, line in enumerate(exemplar_lines):
        if line and not line.startswith("    "):
            assert exemplar_lines[index - 1] == ""


def test_each_user_row_carries_rights_auth_and_port(exemplar_lines: list[str]) -> None:
    alice = line_with(exemplar_lines, "alice")
    assert "readwrite · rights approval-gated" in alice
    assert "password" in alice
    assert alice.rstrip().endswith(":9091")

    bob = line_with(exemplar_lines, "bob")
    assert "readonly" in bob
    assert "rights approval-gated" not in bob
    assert bob.rstrip().endswith(":9092")


def test_a_single_target_render_keeps_one_unqualified_write_right(
    exemplar_lines: list[str],
) -> None:
    # The readwrite tier's render builds the simulator and nothing else, so a
    # session on it reaches one machine. Naming that machine on the row would
    # describe a target-by-target posture the render has no second half of.
    alice = line_with(exemplar_lines, "alice")
    assert "readwrite · rights approval-gated" in alice


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
            "modules.web_terminals.web_base_port": 9091,
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


def test_a_login_free_user_says_no_login(exemplar_lines: list[str]) -> None:
    ariel = line_with(exemplar_lines, "ariel", ":909")
    assert "no login" in ariel
    assert "password" not in ariel


def test_the_panels_row_is_the_union_across_personas(exemplar_lines: list[str]) -> None:
    # EVENTS and BLUESKY are declared by the readwrite persona, not the host
    # profile; a card that read only the host would miss them. Their labels
    # come from `web.panels.<id>.label` — they are not built-ins.
    panels = line_with(exemplar_lines, "panels")
    assert "ARIEL · CHANNELS · KNOWLEDGE · SYSTEM · EVENTS · BLUESKY" in panels


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
    assert "virtual accelerator" in control
    assert "EPICS :5064" in control
    archiver = line_with(exemplar_lines, "archiver")
    assert "mongodb · 30 d retention" in archiver
    channels = line_with(exemplar_lines, "channels")
    assert "hierarchical finder · tier 3" in channels


def test_the_services_group_names_the_injected_stack(exemplar_lines: list[str]) -> None:
    bluesky = line_with(exemplar_lines, "bluesky ", ":8090")
    assert "tiled :8091" in bluesky
    assert "web :8095" in bluesky
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
    assert "  web terminal  :9080" in result.output
    assert "no login" in result.output
    report_at = result.output.index("✓ Created")
    assert result.output.index("  web terminal  :9080") > report_at
