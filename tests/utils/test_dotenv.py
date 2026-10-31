"""Tests for the dependency-free ``.env`` parser.

``parse_dotenv_file`` is the read side of the build lifecycle env injection: the
build subprocess environment is ``{**os.environ, **parse_dotenv_file(env)}``
(``cli/build_cmd.py``). That bulk merge is LOAD-BEARING — every ``KEY`` in the
file must land, not just auth-looking ones, because generated ``.mcp.json``
files reference arbitrary ``${VAR}`` names that Claude Code expands at MCP
server launch. Narrowing the parser to "known" keys would silently break those
references, so the full-passthrough contract is tested explicitly here.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from osprey.deployment.dotenv import env_lock_path
from osprey_connectors.dotenv import parse_dotenv_file


class TestParseDotenvFile:
    """``parse_dotenv_file`` — KEY=VALUE parsing with .env conventions."""

    def _write(self, tmp_path, text):
        p = tmp_path / ".env"
        p.write_text(text, encoding="utf-8")
        return p

    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            pytest.param("KEY=value\n", {"KEY": "value"}, id="basic_key_value"),
            pytest.param(
                "# a comment\n\nKEY=value\n   \n# trailing comment\n",
                {"KEY": "value"},
                id="comments_and_blank_lines_skipped",
            ),
            pytest.param("export KEY=value\n", {"KEY": "value"}, id="export_prefix_stripped"),
            pytest.param(
                'KEY="quoted value"\n', {"KEY": "quoted value"}, id="double_quotes_stripped"
            ),
            pytest.param(
                "KEY='quoted value'\n", {"KEY": "quoted value"}, id="single_quotes_stripped"
            ),
            # Only matching surrounding quotes are stripped.
            pytest.param(
                'KEY="unterminated\n', {"KEY": '"unterminated'}, id="mismatched_quotes_preserved"
            ),
            # A lone quote char is a value, not an empty quoted string.
            pytest.param('KEY="\n', {"KEY": '"'}, id="single_char_value_not_dequoted"),
            # Partition on the first ``=`` keeps later ``=`` in the value.
            pytest.param(
                "URL=postgres://u:p@h/db?a=b\n",
                {"URL": "postgres://u:p@h/db?a=b"},
                id="value_with_equals_sign",
            ),
            pytest.param('KEY="a=b=c"\n', {"KEY": "a=b=c"}, id="quoted_value_containing_equals"),
            pytest.param("KEY=\n", {"KEY": ""}, id="empty_value"),
            pytest.param(
                "  KEY  =  value  \n", {"KEY": "value"}, id="key_and_value_whitespace_trimmed"
            ),
            pytest.param(
                "NOT_A_PAIR\nKEY=value\n", {"KEY": "value"}, id="line_without_equals_skipped"
            ),
            # A leading ``=`` yields an empty key, which is dropped.
            pytest.param("=orphan\nKEY=value\n", {"KEY": "value"}, id="empty_key_skipped"),
            pytest.param(
                "KEY=first\nKEY=second\n", {"KEY": "second"}, id="later_duplicate_key_wins"
            ),
            pytest.param("NAME=Grüße\n", {"NAME": "Grüße"}, id="utf8_value"),
            pytest.param("", {}, id="empty_file_yields_empty_dict"),
        ],
    )
    def test_parses(self, tmp_path, text, expected):
        assert parse_dotenv_file(self._write(tmp_path, text)) == expected

    def test_full_passthrough_not_narrowed_to_auth(self, tmp_path):
        """LOAD-BEARING: every key lands, not just auth-looking ones.

        The build subprocess env is ``{**os.environ, **parse_dotenv_file(...)}``
        and generated ``.mcp.json`` files reference arbitrary ``${VAR}`` names,
        so the parser must not filter to ``*_API_KEY`` / ``*_TOKEN`` style keys.
        """
        text = (
            "ANTHROPIC_API_KEY=sk-secret\n"
            "OSPREY_DISPATCH_TOKEN=tok\n"
            "EPICS_CA_ADDR_LIST=1.2.3.4\n"
            "SOME_RANDOM_SETTING=42\n"
            "facility_name=als\n"
            "PLAIN=value\n"
        )
        env = parse_dotenv_file(self._write(tmp_path, text))
        assert env == {
            "ANTHROPIC_API_KEY": "sk-secret",
            "OSPREY_DISPATCH_TOKEN": "tok",
            "EPICS_CA_ADDR_LIST": "1.2.3.4",
            "SOME_RANDOM_SETTING": "42",
            "facility_name": "als",
            "PLAIN": "value",
        }

    def test_missing_file_raises(self, tmp_path):
        """The parser does not guard existence — callers check ``is_file()``."""
        with pytest.raises(FileNotFoundError):
            parse_dotenv_file(tmp_path / "nonexistent.env")


class TestEnvLockPath:
    """``env_lock_path`` — the one derivation of a ``.env``'s sibling lock name."""

    def test_the_lock_is_the_env_s_name_plus_lock(self):
        """The single spelling every call site now shares."""
        assert env_lock_path(Path("a/.env")) == Path("a/.env.lock")

    def test_a_symlinked_path_is_not_resolved(self, tmp_path):
        """Callers pass this a path they already chose to resolve, or not.

        ``env_file_lock`` resolves first and locks the real inode; the
        name-shaped callers list files inside a directory they hold by name and
        would be handed a path outside it if this resolved on their behalf.
        """
        real = tmp_path / "real"
        real.mkdir()
        link = tmp_path / "link"
        link.symlink_to(real, target_is_directory=True)

        assert env_lock_path(link / ".env") == link / ".env.lock"
