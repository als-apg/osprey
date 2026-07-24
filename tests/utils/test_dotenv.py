"""Tests for the dependency-free ``.env`` parser and preserving merge.

``parse_dotenv_file`` is the read side of the build lifecycle env injection: the
build subprocess environment is ``{**os.environ, **parse_dotenv_file(env)}``
(``cli/build_cmd.py``). That bulk merge is LOAD-BEARING — every ``KEY`` in the
file must land, not just auth-looking ones, because generated ``.mcp.json``
files reference arbitrary ``${VAR}`` names that Claude Code expands at MCP
server launch. Narrowing the parser to "known" keys would silently break those
references, so the full-passthrough contract is tested explicitly here.

``merge_env_preserving_existing`` is the write side of ``osprey build --force``
and template ``.env`` shipping: rendered structure, existing values win.
"""

from __future__ import annotations

import pytest

from osprey.utils.dotenv import (
    _dotenv_raw_lines,
    merge_env_preserving_existing,
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


class TestDotenvRawLines:
    """``_dotenv_raw_lines`` maps KEY -> its raw line with quoting intact."""

    def test_maps_key_to_raw_line(self):
        raw = _dotenv_raw_lines('KEY="quoted"\nOTHER=plain\n')
        assert raw == {"KEY": 'KEY="quoted"', "OTHER": "OTHER=plain"}

    def test_export_key_extracted_but_line_preserved(self):
        """The key drops the export prefix; the stored line keeps it verbatim."""
        raw = _dotenv_raw_lines("export KEY=value\n")
        assert raw == {"KEY": "export KEY=value"}

    def test_comments_and_blanks_ignored(self):
        raw = _dotenv_raw_lines("# comment\n\nKEY=value\n")
        assert raw == {"KEY": "KEY=value"}

    def test_last_duplicate_line_wins(self):
        raw = _dotenv_raw_lines("KEY=first\nKEY=second\n")
        assert raw == {"KEY": "KEY=second"}


class TestMergeEnvPreservingExisting:
    """``merge_env_preserving_existing`` — rendered structure, existing values win."""

    def test_existing_value_overrides_rendered(self):
        rendered = "API_KEY=RENDERED_PLACEHOLDER\n"
        existing = "API_KEY=user-secret\n"
        merged = merge_env_preserving_existing(rendered, existing)
        assert "API_KEY=user-secret" in merged
        assert "RENDERED_PLACEHOLDER" not in merged

    def test_new_rendered_key_is_added(self):
        rendered = "OLD=keep\nNEW_VAR=introduced\n"
        existing = "OLD=user\n"
        merged = merge_env_preserving_existing(rendered, existing)
        assert "OLD=user" in merged
        assert "NEW_VAR=introduced" in merged

    def test_existing_only_keys_appended_with_banner(self):
        rendered = "SHARED=rendered\n"
        existing = "SHARED=user\nEXTRA_SECRET=custom\n"
        merged = merge_env_preserving_existing(rendered, existing)
        assert "# Preserved from existing .env" in merged
        assert "EXTRA_SECRET=custom" in merged
        # The preserved leftover comes after the rendered body.
        assert merged.index("SHARED=user") < merged.index("EXTRA_SECRET=custom")

    def test_no_leftovers_means_no_banner(self):
        rendered = "A=rendered\nB=rendered\n"
        existing = "A=user\n"
        merged = merge_env_preserving_existing(rendered, existing)
        assert "# Preserved from existing .env" not in merged

    def test_rendered_comments_and_structure_preserved(self):
        rendered = "# header comment\n\n# section\nKEY=rendered\n"
        existing = "KEY=user\n"
        merged = merge_env_preserving_existing(rendered, existing)
        assert "# header comment" in merged
        assert "# section" in merged
        assert "KEY=user" in merged

    def test_existing_quoting_preserved_verbatim(self):
        rendered = "TOKEN=PLACEHOLDER\n"
        existing = 'TOKEN="quoted with spaces"\n'
        merged = merge_env_preserving_existing(rendered, existing)
        assert 'TOKEN="quoted with spaces"' in merged

    def test_export_prefixed_keys_match_across_render_and_existing(self):
        """A rendered ``export KEY=`` matches an existing ``KEY=`` by key name."""
        rendered = "export KEY=rendered\n"
        existing = "KEY=user\n"
        merged = merge_env_preserving_existing(rendered, existing)
        assert "KEY=user" in merged
        assert "rendered" not in merged
        assert "# Preserved from existing .env" not in merged

    def test_output_ends_with_single_newline(self):
        merged = merge_env_preserving_existing("A=x\n", "A=y\n")
        assert merged.endswith("\n")
        assert not merged.endswith("\n\n")

    def test_empty_existing_returns_rendered_body(self):
        rendered = "# header\nKEY=rendered\n"
        merged = merge_env_preserving_existing(rendered, "")
        assert "KEY=rendered" in merged
        assert "# Preserved from existing .env" not in merged

    def test_roundtrip_values_parse_back(self, tmp_path):
        """The merged text parses back to existing-wins values."""
        rendered = "A=r\nB=r\nC=r\n"
        existing = "A=user_a\nD=user_d\n"
        merged = merge_env_preserving_existing(rendered, existing)
        out = tmp_path / ".env"
        out.write_text(merged, encoding="utf-8")
        parsed = parse_dotenv_file(out)
        assert parsed == {"A": "user_a", "B": "r", "C": "r", "D": "user_d"}
