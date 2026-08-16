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
"""

from __future__ import annotations

from io import StringIO
from pathlib import Path
from unittest.mock import patch

import pytest
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
        """The framework template is not a file in the operator's deployment."""
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
