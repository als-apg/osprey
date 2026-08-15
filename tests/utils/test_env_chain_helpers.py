"""Tests for the env-chain helpers in :mod:`osprey.utils.dotenv`.

The deployment's env lives in two files at the repo root: ``.env.shared``
(committed, non-secret defaults) and ``.env`` (host-local overrides and
secrets). The precedence is ascending — local wins — and every delivery
mechanism has to reach that same answer with different mechanics. These
helpers are the one place that decides what the chain IS and what it merges
to; the mechanisms are tested where they live.

The module stays the bottom of the import graph: nothing here reads or writes
:data:`os.environ`, and the only side effect in the whole surface is
``write_env_merged``'s file write.
"""

from __future__ import annotations

import os
import stat

import pytest

from osprey.utils.dotenv import (
    ENV_CHAIN_FILENAMES,
    ENV_LOCAL_FILENAME,
    ENV_MERGED_BANNER,
    ENV_SHARED_FILENAME,
    chain_files,
    compose_unsafe_vars_in_chain,
    format_env_line,
    merge_chain,
    parse_dotenv_file,
    parse_dotenv_text,
    write_env_merged,
)


def write_chain(repo_root, shared: str | None = None, local: str | None = None) -> None:
    """Lay down whichever chain members the case needs."""
    if shared is not None:
        (repo_root / ENV_SHARED_FILENAME).write_text(shared, encoding="utf-8")
    if local is not None:
        (repo_root / ENV_LOCAL_FILENAME).write_text(local, encoding="utf-8")


class TestChainOrder:
    """``ENV_CHAIN_FILENAMES`` / ``chain_files`` — membership and order."""

    def test_chain_is_shared_then_local(self):
        """Ascending precedence: the file that wins comes last."""
        assert ENV_CHAIN_FILENAMES == (".env.shared", ".env")

    def test_both_present_returns_shared_first(self, tmp_path):
        write_chain(tmp_path, shared="A=1\n", local="A=2\n")
        assert chain_files(tmp_path) == [
            tmp_path / ENV_SHARED_FILENAME,
            tmp_path / ENV_LOCAL_FILENAME,
        ]

    def test_only_shared_present(self, tmp_path):
        write_chain(tmp_path, shared="A=1\n")
        assert chain_files(tmp_path) == [tmp_path / ENV_SHARED_FILENAME]

    def test_only_local_present(self, tmp_path):
        write_chain(tmp_path, local="A=1\n")
        assert chain_files(tmp_path) == [tmp_path / ENV_LOCAL_FILENAME]

    def test_neither_present_is_empty_not_an_error(self, tmp_path):
        """A project with no env files at all is a valid shape."""
        assert chain_files(tmp_path) == []

    def test_directory_named_like_a_chain_member_is_not_a_member(self, tmp_path):
        """Only regular files count -- a stray directory is not an env file."""
        (tmp_path / ENV_SHARED_FILENAME).mkdir()
        write_chain(tmp_path, local="A=1\n")
        assert chain_files(tmp_path) == [tmp_path / ENV_LOCAL_FILENAME]

    def test_neighbouring_env_files_are_not_chain_members(self, tmp_path):
        """Membership is by name, not by an ``.env*`` scan."""
        (tmp_path / ".env.example").write_text("A=1\n", encoding="utf-8")
        (tmp_path / ".env.users").write_text("A=1\n", encoding="utf-8")
        (tmp_path / ".env.local").write_text("A=1\n", encoding="utf-8")
        assert chain_files(tmp_path) == []


class TestMergeChain:
    """``merge_chain`` — one dict, later file wins."""

    def test_local_wins_on_a_conflicting_key(self, tmp_path):
        write_chain(tmp_path, shared="SHARED_ONLY=s\nBOTH=from_shared\n", local="BOTH=from_local\n")
        assert merge_chain(tmp_path) == {"SHARED_ONLY": "s", "BOTH": "from_local"}

    def test_local_only_key_is_added(self, tmp_path):
        write_chain(tmp_path, shared="A=1\n", local="B=2\n")
        assert merge_chain(tmp_path) == {"A": "1", "B": "2"}

    def test_shared_only_key_survives(self, tmp_path):
        """A default the local file says nothing about is still delivered."""
        write_chain(tmp_path, shared="DEFAULT=on\n", local="OTHER=x\n")
        assert merge_chain(tmp_path)["DEFAULT"] == "on"

    def test_empty_local_value_still_overrides(self, tmp_path):
        """``KEY=`` is an override to empty, not an absent key."""
        write_chain(tmp_path, shared="KEY=default\n", local="KEY=\n")
        assert merge_chain(tmp_path) == {"KEY": ""}

    def test_key_order_is_shared_positions_then_local_additions(self, tmp_path):
        write_chain(tmp_path, shared="A=1\nB=2\n", local="B=two\nC=3\n")
        assert list(merge_chain(tmp_path)) == ["A", "B", "C"]

    def test_missing_shared_falls_back_to_local_alone(self, tmp_path):
        write_chain(tmp_path, local="A=1\n")
        assert merge_chain(tmp_path) == {"A": "1"}

    def test_missing_local_falls_back_to_shared_alone(self, tmp_path):
        write_chain(tmp_path, shared="A=1\n")
        assert merge_chain(tmp_path) == {"A": "1"}

    def test_no_chain_files_merges_to_empty(self, tmp_path):
        assert merge_chain(tmp_path) == {}

    def test_comments_and_quoting_follow_the_parser(self, tmp_path):
        write_chain(
            tmp_path,
            shared="# a comment\nA='quoted default'\n",
            local='export B="local value"\n',
        )
        assert merge_chain(tmp_path) == {"A": "quoted default", "B": "local value"}

    def test_does_not_touch_the_process_environment(self, tmp_path, monkeypatch):
        """Pure read: the helpers never export what they parsed."""
        monkeypatch.delenv("OSPREY_CHAIN_PROBE", raising=False)
        write_chain(tmp_path, shared="OSPREY_CHAIN_PROBE=shared\n", local="OSPREY_CHAIN_PROBE=l\n")
        merge_chain(tmp_path)
        assert "OSPREY_CHAIN_PROBE" not in os.environ


class TestFormatEnvLine:
    """``format_env_line`` — the exported quoting helper."""

    @pytest.mark.parametrize(
        "value",
        ["plain", "with space", " leading", "trailing ", "has#hash", "", "a'b", 'a"b'],
    )
    def test_round_trips_through_the_parser(self, value):
        line = format_env_line("KEY", value)
        assert parse_dotenv_text(line) == {"KEY": value}

    def test_plain_value_is_written_unquoted(self, tmp_path):
        assert format_env_line("KEY", "plain") == "KEY=plain"

    def test_newline_value_is_refused(self):
        with pytest.raises(ValueError, match="newline"):
            format_env_line("KEY", "one\ntwo")

    def test_both_quote_characters_with_quoting_needed_is_refused(self):
        with pytest.raises(ValueError, match="both single and double quotes"):
            format_env_line("KEY", "needs space and 'both' \"quotes\"")


class TestWriteEnvMerged:
    """``write_env_merged`` — the one-file delivery of the merged chain."""

    def test_writes_the_merged_result_readable_by_the_parser(self, tmp_path):
        repo = tmp_path / "repo"
        repo.mkdir()
        write_chain(repo, shared="A=1\nBOTH=shared\n", local="BOTH=local\nB=2\n")
        dest = write_env_merged(repo, tmp_path / "build" / ".env.merged")
        assert parse_dotenv_file(dest) == {"A": "1", "BOTH": "local", "B": "2"}

    def test_returns_the_destination(self, tmp_path):
        repo = tmp_path / "repo"
        repo.mkdir()
        write_chain(repo, local="A=1\n")
        dest = tmp_path / "build" / ".env.merged"
        assert write_env_merged(repo, dest) == dest

    def test_creates_a_missing_parent_directory(self, tmp_path):
        repo = tmp_path / "repo"
        repo.mkdir()
        write_chain(repo, local="A=1\n")
        dest = write_env_merged(repo, tmp_path / "build" / "nested" / ".env.merged")
        assert dest.is_file()

    def test_carries_the_generated_artifact_banner(self, tmp_path):
        repo = tmp_path / "repo"
        repo.mkdir()
        write_chain(repo, local="A=1\n")
        dest = write_env_merged(repo, tmp_path / ".env.merged")
        assert dest.read_text(encoding="utf-8").startswith(ENV_MERGED_BANNER)

    def test_file_is_created_0600(self, tmp_path):
        """The merged file inherits the local half's secrets, so its mode too."""
        repo = tmp_path / "repo"
        repo.mkdir()
        write_chain(repo, local="TOKEN=s3cret\n")
        dest = write_env_merged(repo, tmp_path / ".env.merged")
        assert stat.S_IMODE(dest.stat().st_mode) == 0o600

    def test_rewrite_tightens_a_loose_mode(self, tmp_path):
        repo = tmp_path / "repo"
        repo.mkdir()
        write_chain(repo, local="TOKEN=s3cret\n")
        dest = tmp_path / ".env.merged"
        dest.write_text("stale\n", encoding="utf-8")
        dest.chmod(0o644)
        write_env_merged(repo, dest)
        assert stat.S_IMODE(dest.stat().st_mode) == 0o600

    def test_rewrite_replaces_rather_than_appends(self, tmp_path):
        """Every write regenerates the file from the chain."""
        repo = tmp_path / "repo"
        repo.mkdir()
        write_chain(repo, local="A=1\n")
        dest = write_env_merged(repo, tmp_path / ".env.merged")
        write_chain(repo, local="B=2\n")
        write_env_merged(repo, dest)
        assert parse_dotenv_file(dest) == {"B": "2"}

    def test_values_are_quoted_so_they_survive(self, tmp_path):
        repo = tmp_path / "repo"
        repo.mkdir()
        write_chain(repo, shared='PADDED="  spaced  "\n', local="INLINE=a#b\n")
        dest = write_env_merged(repo, tmp_path / ".env.merged")
        assert parse_dotenv_file(dest) == {"PADDED": "  spaced  ", "INLINE": "a#b"}

    def test_no_chain_files_writes_a_bannered_empty_file(self, tmp_path):
        repo = tmp_path / "repo"
        repo.mkdir()
        dest = write_env_merged(repo, tmp_path / ".env.merged")
        assert dest.read_text(encoding="utf-8") == ENV_MERGED_BANNER
        assert parse_dotenv_file(dest) == {}

    def test_leaves_no_temp_file_beside_the_destination(self, tmp_path):
        repo = tmp_path / "repo"
        repo.mkdir()
        write_chain(repo, local="A=1\n")
        out = tmp_path / "build"
        write_env_merged(repo, out / ".env.merged")
        assert [p.name for p in out.iterdir()] == [".env.merged"]


class TestComposeUnsafeVarsInChain:
    """``compose_unsafe_vars_in_chain`` — the ``$``-scan over every member."""

    def test_flags_a_dollar_in_the_local_file(self, tmp_path):
        write_chain(tmp_path, shared="A=1\n", local="SECRET=pa$$word\n")
        assert compose_unsafe_vars_in_chain(tmp_path) == ["SECRET"]

    def test_flags_a_dollar_in_the_shared_file(self, tmp_path):
        write_chain(tmp_path, shared="SHARED_VAL=a$b\n", local="A=1\n")
        assert compose_unsafe_vars_in_chain(tmp_path) == ["SHARED_VAL"]

    def test_flags_a_shared_value_even_when_the_local_file_overrides_it(self, tmp_path):
        """Compose reads every file in the list, override or not."""
        write_chain(tmp_path, shared="KEY=a$b\n", local="KEY=clean\n")
        assert compose_unsafe_vars_in_chain(tmp_path) == ["KEY"]

    def test_names_are_sorted_and_deduplicated_across_the_chain(self, tmp_path):
        write_chain(tmp_path, shared="B_KEY=x$y\nA_KEY=p$q\n", local="A_KEY=r$s\n")
        assert compose_unsafe_vars_in_chain(tmp_path) == ["A_KEY", "B_KEY"]

    def test_returns_names_never_values(self, tmp_path):
        write_chain(tmp_path, local="SECRET=pa$$word\n")
        assert "pa$" not in "".join(compose_unsafe_vars_in_chain(tmp_path))

    def test_clean_chain_is_empty(self, tmp_path):
        write_chain(tmp_path, shared="A=1\n", local="B=two words\n")
        assert compose_unsafe_vars_in_chain(tmp_path) == []

    def test_no_chain_files_is_empty(self, tmp_path):
        assert compose_unsafe_vars_in_chain(tmp_path) == []


def test_compose_generator_shares_the_shared_filename():
    """The renderer's constant is the same string the helpers chain on."""
    from osprey.deployment import compose_generator

    assert (
        compose_generator.ENV_SHARED_FILENAME,
        compose_generator.COMPOSE_ENV_FILENAME,
    ) == ENV_CHAIN_FILENAMES
    assert compose_generator.ENV_SHARED_FILENAME == ENV_SHARED_FILENAME
