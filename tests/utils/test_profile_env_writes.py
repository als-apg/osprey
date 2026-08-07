"""Tests for ``append_profile_env`` — the atomic, append-only profile ``.env`` write.

``deploy up`` persists the secrets it minted back into the profile that owns
them. Two properties carry the safety here:

* **Append-only.** A key already on file keeps its value — it is pinned by the
  docker volume initialized with it and the containers already trusting it. A
  supplied value that disagrees is *reported*, never written.
* **Atomic.** Lock file + temp file + ``os.replace``, so two deploys racing on
  the same profile both land their keys and no reader ever sees a half-written
  file.
"""

from __future__ import annotations

import multiprocessing
import os
import stat
from pathlib import Path

import pytest

from osprey.utils.dotenv import (
    DEPLOY_MINTED_BANNER,
    append_profile_env,
    parse_dotenv_file,
)


def _append_in_child(env_path: str, keys: list[str], barrier) -> None:
    """Child-process body for the concurrency test.

    Module-level (not a closure) so a ``spawn`` child can import it by name.
    """
    from osprey.utils.dotenv import append_profile_env as child_append

    barrier.wait(timeout=30)
    for key in keys:
        child_append(Path(env_path), {key: f"value-of-{key}"})


class TestFreshFile:
    """A profile ``.env`` that does not exist yet."""

    def test_creates_file_with_entries(self, tmp_path):
        env_path = tmp_path / ".env"
        result = append_profile_env(
            env_path, {"ARIEL_DB_PASSWORD": "pw", "ZO_ROOT_USER_PASSWORD": "zo"}
        )
        assert result.added == ("ARIEL_DB_PASSWORD", "ZO_ROOT_USER_PASSWORD")
        assert result.unchanged == () and result.conflicts == ()
        assert parse_dotenv_file(env_path) == {
            "ARIEL_DB_PASSWORD": "pw",
            "ZO_ROOT_USER_PASSWORD": "zo",
        }

    def test_created_file_is_0600(self, tmp_path):
        env_path = tmp_path / ".env"
        append_profile_env(env_path, {"TOKEN": "t"})
        assert stat.S_IMODE(env_path.stat().st_mode) == 0o600

    def test_existing_file_tightened_to_0600(self, tmp_path):
        env_path = tmp_path / ".env"
        env_path.write_text("OTHER=x\n", encoding="utf-8")
        os.chmod(env_path, 0o644)
        append_profile_env(env_path, {"TOKEN": "t"})
        assert stat.S_IMODE(env_path.stat().st_mode) == 0o600

    def test_no_entries_writes_nothing(self, tmp_path):
        env_path = tmp_path / ".env"
        result = append_profile_env(env_path, {})
        assert result.added == ()
        assert not env_path.exists()

    def test_missing_parent_directory_raises(self, tmp_path):
        with pytest.raises(OSError):
            append_profile_env(tmp_path / "gone" / ".env", {"TOKEN": "t"})


class TestAppendOnly:
    """Existing keys keep their value; mismatches are reported, not written."""

    def test_existing_key_keeps_its_value(self, tmp_path):
        env_path = tmp_path / ".env"
        env_path.write_text("ARIEL_DB_PASSWORD=volume-pinned\n", encoding="utf-8")
        result = append_profile_env(env_path, {"ARIEL_DB_PASSWORD": "freshly-minted"})
        assert parse_dotenv_file(env_path)["ARIEL_DB_PASSWORD"] == "volume-pinned"
        assert result.added == ()

    def test_mismatch_reported_as_conflict(self, tmp_path):
        env_path = tmp_path / ".env"
        env_path.write_text("ARIEL_DB_PASSWORD=volume-pinned\n", encoding="utf-8")
        result = append_profile_env(env_path, {"ARIEL_DB_PASSWORD": "freshly-minted"})
        assert len(result.conflicts) == 1
        conflict = result.conflicts[0]
        assert conflict.key == "ARIEL_DB_PASSWORD"
        assert conflict.existing == "volume-pinned"
        assert conflict.supplied == "freshly-minted"

    def test_matching_value_is_unchanged_not_conflict(self, tmp_path):
        env_path = tmp_path / ".env"
        env_path.write_text("TOKEN=same\n", encoding="utf-8")
        result = append_profile_env(env_path, {"TOKEN": "same"})
        assert result.unchanged == ("TOKEN",)
        assert result.conflicts == ()
        assert env_path.read_text(encoding="utf-8") == "TOKEN=same\n"

    def test_quoted_existing_value_compared_by_parsed_value(self, tmp_path):
        env_path = tmp_path / ".env"
        env_path.write_text('TOKEN="same"\n', encoding="utf-8")
        result = append_profile_env(env_path, {"TOKEN": "same"})
        assert result.unchanged == ("TOKEN",)

    def test_mixed_batch_partitions_correctly(self, tmp_path):
        env_path = tmp_path / ".env"
        env_path.write_text("KEPT=old\nSAME=v\n", encoding="utf-8")
        result = append_profile_env(env_path, {"KEPT": "new", "SAME": "v", "FRESH": "f"})
        assert result.added == ("FRESH",)
        assert result.unchanged == ("SAME",)
        assert [c.key for c in result.conflicts] == ["KEPT"]
        assert parse_dotenv_file(env_path) == {"KEPT": "old", "SAME": "v", "FRESH": "f"}

    def test_unrelated_content_preserved_verbatim(self, tmp_path):
        env_path = tmp_path / ".env"
        env_path.write_text("# provider keys\nANTHROPIC_API_KEY=sk-1\n", encoding="utf-8")
        append_profile_env(env_path, {"TOKEN": "t"})
        lines = env_path.read_text(encoding="utf-8").splitlines()
        assert lines[:2] == [
            "# provider keys",
            "ANTHROPIC_API_KEY=sk-1",  # gitleaks:allow - fixture value, not a secret
        ]


class TestSectionBanner:
    """Banner placement — one section, headers never stacked."""

    def test_banner_precedes_appended_entries(self, tmp_path):
        env_path = tmp_path / ".env"
        env_path.write_text("OTHER=x\n", encoding="utf-8")
        append_profile_env(env_path, {"TOKEN": "t"})
        lines = env_path.read_text(encoding="utf-8").splitlines()
        assert lines == ["OTHER=x", "", DEPLOY_MINTED_BANNER, "TOKEN=t"]

    def test_second_append_reuses_the_existing_banner(self, tmp_path):
        env_path = tmp_path / ".env"
        append_profile_env(env_path, {"FIRST": "1"})
        append_profile_env(env_path, {"SECOND": "2"})
        text = env_path.read_text(encoding="utf-8")
        assert text.count(DEPLOY_MINTED_BANNER) == 1
        assert text.splitlines() == [DEPLOY_MINTED_BANNER, "FIRST=1", "SECOND=2"]

    def test_custom_banner_used(self, tmp_path):
        env_path = tmp_path / ".env"
        append_profile_env(env_path, {"TOKEN": "t"}, "# ── Seeded by profile new ──")
        assert env_path.read_text(encoding="utf-8").startswith("# ── Seeded by profile new ──\n")

    def test_no_banner_when_nothing_added(self, tmp_path):
        env_path = tmp_path / ".env"
        env_path.write_text("TOKEN=t\n", encoding="utf-8")
        append_profile_env(env_path, {"TOKEN": "other"})
        assert env_path.read_text(encoding="utf-8") == "TOKEN=t\n"


class TestValueFormatting:
    """Written values must parse back to exactly what was supplied."""

    @pytest.mark.parametrize(
        "value",
        ["plain", "with space", "hash#inside", " leading-space", 'has"double', "has'single"],
    )
    def test_value_round_trips(self, tmp_path, value):
        env_path = tmp_path / ".env"
        append_profile_env(env_path, {"TOKEN": value})
        assert parse_dotenv_file(env_path)["TOKEN"] == value

    def test_newline_in_value_rejected(self, tmp_path):
        with pytest.raises(ValueError, match="newline"):
            append_profile_env(tmp_path / ".env", {"TOKEN": "a\nb"})


class TestConcurrentAppends:
    """Two processes appending at once: both land, nothing is corrupted."""

    def test_two_processes_both_land_their_keys(self, tmp_path):
        env_path = tmp_path / ".env"
        env_path.write_text("BASE=0\n", encoding="utf-8")
        ctx = multiprocessing.get_context("spawn")
        barrier = ctx.Barrier(2)
        left = [f"LEFT_{i}" for i in range(15)]
        right = [f"RIGHT_{i}" for i in range(15)]
        procs = [
            ctx.Process(target=_append_in_child, args=(str(env_path), keys, barrier))
            for keys in (left, right)
        ]
        for proc in procs:
            proc.start()
        for proc in procs:
            proc.join(timeout=120)
        assert [proc.exitcode for proc in procs] == [0, 0]

        parsed = parse_dotenv_file(env_path)
        assert parsed["BASE"] == "0"
        for key in left + right:
            assert parsed[key] == f"value-of-{key}"
        # No corruption: every line is a comment, a blank, or one KEY=VALUE,
        # and no key was written twice.
        lines = env_path.read_text(encoding="utf-8").splitlines()
        assignments = [ln for ln in lines if ln.strip() and not ln.strip().startswith("#")]
        assert len(assignments) == len(parsed)
        assert not list(tmp_path.glob(".env*.tmp"))
