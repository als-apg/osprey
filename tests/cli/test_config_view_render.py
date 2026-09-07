"""How ``osprey config`` puts a configuration on screen, and into a file.

The verb has one body and two audiences, and the split is the terminal check in
``_emit``. An interactive reader gets a header, the file's location and syntax
colouring, all of it through the CLI renderer. A pipe gets the file's BYTES:
``osprey config > deployment.yml`` has to write the configuration it just
showed, so that branch stays on :func:`click.echo` rather than going through
Rich, which expands tabs and can pad a line out to the console width.

Both halves are pinned here because only one of them is reachable from a
``CliRunner``: the runner is never a terminal, so every other config test in the
suite exercises the byte path and none of them exercise the rendered one.

The third view has its own class at the bottom. ``--defaults`` is not a file
the verb reads back but a ledger it BUILDS, out of the config-key manifest's
``default:`` column, so what it puts on screen is this module's own work and is
pinned accordingly.
"""

from __future__ import annotations

import importlib.resources
from io import StringIO
from pathlib import Path
from unittest.mock import patch

import click
import pytest
import yaml
from rich.console import Console
from rich.syntax import Syntax
from rich.text import Text

from osprey.cli import config_cmd, phase_reporter, styles

#: A configuration with the two things Rich would not leave alone: a tab inside
#: a quoted scalar, and a line long enough that an 80-column console would wrap
#: it. Both are legal YAML and both must survive the pipe untouched.
_LONG_VALUE = "x" * 120
AWKWARD_YAML = f'# what the facility wrote\ngreeting: "a\tb"\nnote: {_LONG_VALUE}\nport: 8080\n'


@pytest.fixture
def terminal(monkeypatch: pytest.MonkeyPatch) -> StringIO:
    """Make the CLI's console a terminal, painting into a buffer.

    Two names, one console. ``config_cmd`` asks ``styles.console`` whether it is
    talking to a terminal, and the renderer prints through the reporter's, which
    ``phase_reporter`` bound from ``styles`` at import; patching only one of them
    would pin a path the CLI never takes.

    Returns:
        The buffer the console paints into, escape sequences and all.
    """
    buffer = StringIO()
    console = Console(
        file=buffer,
        force_terminal=True,
        color_system="standard",
        no_color=False,
        width=100,
        theme=styles.osprey_theme,
        legacy_windows=False,
    )
    monkeypatch.setattr(styles, "console", console)
    monkeypatch.setattr(phase_reporter, "console", console)
    return buffer


def _plain(buffer: StringIO) -> str:
    """What the buffer shows, with its escape sequences removed."""
    return Text.from_ansi(buffer.getvalue()).plain


class TestThePipeGetsTheFile:
    """Off a terminal the verb writes bytes, not copy."""

    def test_the_text_goes_out_unchanged(self, capsys):
        config_cmd._emit(AWKWARD_YAML, label="Source profile", source=Path("/etc/profile.yml"))

        # click.echo supplies the one trailing newline, as it always has.
        assert capsys.readouterr().out == AWKWARD_YAML + "\n"

    def test_the_tab_survives(self, capsys):
        """The reason this branch is not the renderer's: Rich expands tabs."""
        config_cmd._emit(AWKWARD_YAML, label="Source profile", source=None)

        assert '"a\tb"' in capsys.readouterr().out

    def test_the_long_line_is_not_wrapped(self, capsys):
        config_cmd._emit(AWKWARD_YAML, label="Source profile", source=None)

        assert f"note: {_LONG_VALUE}" in capsys.readouterr().out

    def test_no_header_and_no_location(self, capsys):
        """A header in the file would make the file invalid."""
        config_cmd._emit(AWKWARD_YAML, label="Source profile", source=Path("/etc/profile.yml"))

        out = capsys.readouterr().out
        assert "Source profile" not in out
        assert "/etc/profile.yml" not in out


class TestTheTerminalGetsTheView:
    """On a terminal the verb prints through the renderer."""

    def test_label_and_location_and_content_all_appear(self, terminal):
        config_cmd._emit("port: 8080\n", label="Source profile", source=Path("/etc/profile.yml"))

        rendered = _plain(terminal)
        assert "Source profile" in rendered
        assert "/etc/profile.yml" in rendered
        assert "port: 8080" in rendered

    def test_the_location_is_subordinate_to_the_label(self, terminal):
        """It answers "where", which is detail under the view's name."""
        config_cmd._emit("port: 8080\n", label="Source profile", source=Path("/etc/profile.yml"))

        lines = [line.rstrip() for line in _plain(terminal).splitlines()]
        assert "Source profile" in lines
        assert "  /etc/profile.yml" in lines

    def test_the_defaults_view_shows_no_location(self, terminal):
        """The key ledger is built text, not a file in anyone's deployment."""
        config_cmd._emit("port: 8080\n", label="Framework default configuration", source=None)

        rendered = _plain(terminal)
        assert "Framework default configuration" in rendered
        assert "port: 8080" in rendered

    def test_the_content_is_painted(self, terminal):
        """Syntax colouring is half of why the interactive view exists."""
        config_cmd._emit("port: 8080\n", label="Source profile", source=None)

        assert "\x1b[" in terminal.getvalue()


class TestTheCodeBlockGoesThroughTheRenderer:
    """The YAML block is a built renderable, printed by the one primitive."""

    def test_the_syntax_block_is_handed_to_output_table(self, terminal):
        recorded: list[object] = []
        with patch("osprey.cli.output.table", side_effect=recorded.append):
            config_cmd._emit(AWKWARD_YAML, label="Source profile", source=None)

        assert len(recorded) == 1
        block = recorded[0]
        assert isinstance(block, Syntax)
        assert "yaml" in block.lexer.name.lower()
        assert block.code == AWKWARD_YAML
        assert block.word_wrap is True

    def test_the_pipe_reaches_the_renderer_not_at_all(self):
        """Off a terminal there is no renderable, and no renderer call."""
        recorded: list[object] = []
        with patch("osprey.cli.output.table", side_effect=recorded.append):
            config_cmd._emit(AWKWARD_YAML, label="Source profile", source=None)

        assert recorded == []


class TestTheDefaultsLedger:
    """``--defaults`` renders the config-key manifest's ``default:`` column.

    The view answers "what happens when profile.yml omits this key?", which no
    preset and no rendered build output can answer — a preset shows a key's
    VALUE. Its source is therefore the manifest, whose column was read out of
    each key's reader, and what is pinned here is that every declared key
    reaches the screen and that the ways of having no literal default stay
    distinguishable from one another and from a real value.
    """

    def test_defaults_lists_every_manifest_key(self):
        """A key the framework reads but the ledger omits is a key an operator
        can only discover by reading the source, which is the failure this
        view exists to prevent."""
        manifest = config_cmd._load_key_manifest()

        listed = {
            line.split(":", 1)[0]
            for line in config_cmd._render_defaults_ledger().splitlines()
            if line and not line.startswith("#")
        }

        assert listed == set(manifest)

    def test_the_ledger_still_parses_as_yaml(self):
        """Dotted keys are legal YAML keys and the markers are legal scalars.

        Nothing downstream loads this view. But a ledger that does not parse
        has a quoting bug somewhere in its values, and this is the cheapest
        place to notice one.
        """
        loaded = yaml.safe_load(config_cmd._render_defaults_ledger())

        assert set(loaded) == set(config_cmd._load_key_manifest())

    def test_a_literal_default_prints_as_yaml(self):
        """``build_dir``'s reader falls back to ``./build``, so the ledger says so."""
        assert "\nbuild_dir: ./build\n" in config_cmd._render_defaults_ledger()

    def test_the_empty_string_default_stays_quoted(self):
        """Unquoted, an empty value would read back as null rather than as ''."""
        loaded = yaml.safe_load(config_cmd._render_defaults_ledger())

        assert loaded["facility.prefix"] == ""

    def test_a_required_key_is_marked_rather_than_given_a_value(self):
        """The posture floor has no fallback. Inventing one for this view would
        be the same wrong answer the build itself refuses to make."""
        assert "\ncontrol_system.type: <required>\n" in config_cmd._render_defaults_ledger()

    def test_a_derived_key_carries_its_note_above_it(self):
        """``<derived>`` alone says nothing; the note is the whole answer."""
        ledger = config_cmd._render_defaults_ledger()

        assert "# the project directory's own name\nproject_name: <derived>\n" in ledger

    def test_a_covered_leaf_names_the_key_that_covers_it(self):
        """A leaf with no reader of its own points at the block that has one,
        rather than dropping out of the ledger."""
        ledger = config_cmd._render_defaults_ledger()

        assert "\ncontrol_system.connector.epics: <covered by control_system.connector>\n" in ledger

    def test_a_covered_leaf_with_nothing_to_cite_says_so(self):
        """The manifest's contract makes this unreachable today. It is still the
        branch that must not print a key that is not in the ledger."""
        marker = config_cmd._default_marker("orphan.leaf", {"default": "n/a"}, {})

        assert marker == "<no separate default>"

    def test_every_marker_is_explained_before_the_first_section(self):
        """A marker the header does not define is a word the reader guesses at."""
        ledger = config_cmd._render_defaults_ledger()
        preamble = ledger.split("\n# ── ", 1)[0]

        for marker in config_cmd._ABSENT_LITERAL_MARKERS.values():
            assert marker in preamble
        assert "<covered by KEY>" in preamble
        assert "<no separate default>" in preamble

    def test_keys_are_grouped_under_their_top_level_block(self):
        """Grouping in manifest order is what makes the dotted keys navigable."""
        ledger = config_cmd._render_defaults_ledger()

        sections = [
            line.removeprefix("# ── ").split(" ", 1)[0]
            for line in ledger.splitlines()
            if line.startswith("# ── ")
        ]
        assert sections == list(
            dict.fromkeys(key.partition(".")[0] for key in config_cmd._load_key_manifest())
        )

    def test_a_near_miss_sentinel_is_not_taken_for_a_marker(self):
        """The column's vocabulary is closed and spelled exactly. A near miss
        prints as the literal string it is, which is visibly wrong, rather than
        quietly becoming one of the four markers."""
        marker = config_cmd._default_marker("some.key", {"default": "no_fallback"}, {})

        assert marker == "no_fallback"

    def test_a_missing_manifest_reports_a_broken_installation(self, monkeypatch, tmp_path):
        """It is not the operator's deployment that is wrong, and the message has
        to say so or the next hour goes into the wrong file."""
        monkeypatch.setattr(config_cmd, "packaged_manifest_path", lambda: tmp_path / "absent.yml")

        with pytest.raises(click.ClickException) as caught:
            config_cmd._load_key_manifest()

        assert "broken installation" in str(caught.value)

    def test_a_keyless_manifest_reports_a_broken_installation(self, monkeypatch, tmp_path):
        """A manifest that parses but declares nothing would otherwise render an
        empty ledger, which reads as "the framework has no configuration"."""
        empty = tmp_path / config_cmd.MANIFEST_FILENAME
        empty.write_text("deleted: {}\n", encoding="utf-8")
        monkeypatch.setattr(config_cmd, "packaged_manifest_path", lambda: empty)

        with pytest.raises(click.ClickException) as caught:
            config_cmd._load_key_manifest()

        assert "declares no keys" in str(caught.value)


class TestTheManifestShipsInsideThePackage:
    """The ledger's source has to survive a wheel install.

    ``--defaults`` is the one view documented as needing no deployment repo, so
    it is also the one view a plain ``pip install osprey-framework`` has to be
    able to render. The wheel packages ``src/osprey`` and nothing else, so a
    manifest kept outside the package is a manifest this view cannot open —
    the failure would show up only in an installed environment, never in a
    checkout, which is exactly why it is pinned here.
    """

    def test_the_manifest_resolves_as_package_data(self):
        """Located through the package, not through a path out of the repo root."""
        path = config_cmd.packaged_manifest_path()

        assert path.is_file(), path
        assert path.name == config_cmd.MANIFEST_FILENAME

    def test_the_manifest_lives_under_the_osprey_package(self):
        """A path that climbs out of ``src/osprey`` is a path a wheel drops."""
        package_root = Path(str(importlib.resources.files("osprey"))).resolve()

        assert package_root in config_cmd.packaged_manifest_path().resolve().parents

    def test_the_packaged_manifest_is_the_one_the_ledger_renders(self):
        """Two copies would let the view and the guard drift apart silently.

        The expected path is spelled out here rather than taken from
        ``packaged_manifest_path``: reading the same helper on both sides would
        pass for any path that helper returns, including one that no longer
        points inside ``osprey.profiles``.
        """
        packaged_path = (
            Path(str(importlib.resources.files("osprey"))) / "profiles" / "config_key_manifest.yml"
        )

        packaged = yaml.safe_load(packaged_path.read_text(encoding="utf-8"))

        assert config_cmd._load_key_manifest() == packaged["keys"]
