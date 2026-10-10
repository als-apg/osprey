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

from osprey.utils.dotenv import (
    env_lock_path,
    parse_dotenv_file,
)


class TestParseDotenvFile:
    """``parse_dotenv_file`` — KEY=VALUE parsing with .env conventions."""

    def _write(self, tmp_path, text):
        p = tmp_path / ".env"
        p.write_text(text, encoding="utf-8")
        return p

    def test_basic_key_value(self, tmp_path):
        env = parse_dotenv_file(self._write(tmp_path, "KEY=value\n"))
        assert env == {"KEY": "value"}

    def test_comments_and_blank_lines_skipped(self, tmp_path):
        text = "# a comment\n\nKEY=value\n   \n# trailing comment\n"
        env = parse_dotenv_file(self._write(tmp_path, text))
        assert env == {"KEY": "value"}

    def test_export_prefix_stripped(self, tmp_path):
        env = parse_dotenv_file(self._write(tmp_path, "export KEY=value\n"))
        assert env == {"KEY": "value"}

    def test_double_quotes_stripped(self, tmp_path):
        env = parse_dotenv_file(self._write(tmp_path, 'KEY="quoted value"\n'))
        assert env == {"KEY": "quoted value"}

    def test_single_quotes_stripped(self, tmp_path):
        env = parse_dotenv_file(self._write(tmp_path, "KEY='quoted value'\n"))
        assert env == {"KEY": "quoted value"}

    def test_mismatched_quotes_preserved(self, tmp_path):
        """Only matching surrounding quotes are stripped."""
        env = parse_dotenv_file(self._write(tmp_path, 'KEY="unterminated\n'))
        assert env == {"KEY": '"unterminated'}

    def test_single_char_value_not_dequoted(self, tmp_path):
        """A lone quote char is a value, not an empty quoted string."""
        env = parse_dotenv_file(self._write(tmp_path, 'KEY="\n'))
        assert env == {"KEY": '"'}

    def test_value_with_equals_sign(self, tmp_path):
        """Partition on the first ``=`` keeps later ``=`` in the value."""
        env = parse_dotenv_file(self._write(tmp_path, "URL=postgres://u:p@h/db?a=b\n"))
        assert env == {"URL": "postgres://u:p@h/db?a=b"}

    def test_quoted_value_containing_equals(self, tmp_path):
        env = parse_dotenv_file(self._write(tmp_path, 'KEY="a=b=c"\n'))
        assert env == {"KEY": "a=b=c"}

    def test_empty_value(self, tmp_path):
        env = parse_dotenv_file(self._write(tmp_path, "KEY=\n"))
        assert env == {"KEY": ""}

    def test_key_and_value_whitespace_trimmed(self, tmp_path):
        env = parse_dotenv_file(self._write(tmp_path, "  KEY  =  value  \n"))
        assert env == {"KEY": "value"}

    def test_line_without_equals_skipped(self, tmp_path):
        env = parse_dotenv_file(self._write(tmp_path, "NOT_A_PAIR\nKEY=value\n"))
        assert env == {"KEY": "value"}

    def test_empty_key_skipped(self, tmp_path):
        """A leading ``=`` yields an empty key, which is dropped."""
        env = parse_dotenv_file(self._write(tmp_path, "=orphan\nKEY=value\n"))
        assert env == {"KEY": "value"}

    def test_later_duplicate_key_wins(self, tmp_path):
        env = parse_dotenv_file(self._write(tmp_path, "KEY=first\nKEY=second\n"))
        assert env == {"KEY": "second"}

    def test_utf8_value(self, tmp_path):
        env = parse_dotenv_file(self._write(tmp_path, "NAME=Grüße\n"))
        assert env == {"NAME": "Grüße"}

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

    def test_empty_file_yields_empty_dict(self, tmp_path):
        assert parse_dotenv_file(self._write(tmp_path, "")) == {}


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
