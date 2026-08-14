"""The build's write into the deployment's one secret store.

``tests/cli/test_va_build_step.py`` pins this property end to end, through a
real ``osprey build``. This module pins the same function's branches directly,
because two of them are about what the build must NOT do to a file it does not
own — and a test that has to render a whole project to reach "the operator's
value survived" is a test nobody will extend when the next build-derived key
arrives.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from osprey.cli.build_cmd import _wire_build_derived_env
from osprey.services.virtual_accelerator.manifest.build import MANIFEST_FILENAME
from osprey.utils.dotenv import (
    BUILD_DERIVED_BANNER,
    BUILD_DERIVED_KEYS,
    DEPLOY_MINTED_BANNER,
    parse_dotenv_file,
)


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    """A deployment repo whose build produced a channel manifest."""
    (tmp_path / "build" / "data" / "simulation").mkdir(parents=True)
    (tmp_path / "build" / "data" / "simulation" / MANIFEST_FILENAME).write_text('{"channels": []}')
    return tmp_path


@pytest.fixture
def barren_repo(tmp_path: Path) -> Path:
    """…and one whose build could not: the data tree backs no manifest."""
    (tmp_path / "build" / "data" / "simulation").mkdir(parents=True)
    return tmp_path


def _wire(repo_dir: Path) -> dict[str, str]:
    _wire_build_derived_env(repo_dir, repo_dir / "build")
    env_path = repo_dir / ".env"
    return parse_dotenv_file(env_path) if env_path.is_file() else {}


class TestTheKeysLandWhereComposeReads:
    def test_a_repo_with_no_env_yet_gets_one(self, repo):
        assert _wire(repo) == {"VA_CHANNELS_FILE": MANIFEST_FILENAME, "VA_LATTICE": "builtin"}

    def test_the_keys_written_are_the_keys_the_build_claims_to_own(self, repo):
        """One enumeration, or the stale-pointer scan below misses a key.

        :data:`BUILD_DERIVED_KEYS` is what the no-manifest branch scans the
        operator's file for. A key this function writes but that set does not
        name would be written on a good build and never reported on a bad one.
        """
        assert set(_wire(repo)) == set(BUILD_DERIVED_KEYS)

    def test_they_go_under_their_own_banner(self, repo):
        _wire_build_derived_env(repo, repo / "build")

        text = (repo / ".env").read_text()
        assert BUILD_DERIVED_BANNER in text
        # Not under the deploy's section, which answers a different question for
        # whoever opens the file: that one holds secrets no rebuild can
        # reproduce, this one holds pointers at artifacts every build
        # regenerates. Keeping the two apart also keeps the build's keys clear
        # of anything that later treats minted blocks as a unit — nothing reads
        # either banner today, so the separation is worth pinning while it is
        # still free.
        assert DEPLOY_MINTED_BANNER not in text

    def test_the_file_is_born_with_secret_permissions(self, repo):
        _wire_build_derived_env(repo, repo / "build")

        assert (repo / ".env").stat().st_mode & 0o777 == 0o600

    def test_existing_content_is_kept(self, repo):
        (repo / ".env").write_text("ANTHROPIC_API_KEY=sk-real\n")

        env = _wire(repo)

        assert env["ANTHROPIC_API_KEY"] == "sk-real"
        assert env["VA_CHANNELS_FILE"] == MANIFEST_FILENAME

    def test_a_second_build_appends_nothing(self, repo):
        _wire_build_derived_env(repo, repo / "build")
        first = (repo / ".env").read_text()

        _wire_build_derived_env(repo, repo / "build")

        assert (repo / ".env").read_text() == first


class TestTheOperatorsValueWins:
    """The store is hand-edited and written back to by ``osprey up``."""

    def test_a_hand_edited_pointer_is_not_replaced(self, repo, caplog):
        (repo / ".env").write_text("VA_CHANNELS_FILE=my-own.json\n")

        with caplog.at_level(logging.WARNING):
            env = _wire(repo)

        assert env["VA_CHANNELS_FILE"] == "my-own.json"
        assert "VA_CHANNELS_FILE" in caplog.text

    def test_the_other_key_still_lands(self, repo):
        """A conflict on one key is not a reason to skip the rest."""
        (repo / ".env").write_text("VA_CHANNELS_FILE=my-own.json\n")

        assert _wire(repo)["VA_LATTICE"] == "builtin"

    def test_an_agreeing_value_is_not_a_conflict(self, repo, caplog):
        (repo / ".env").write_text(f"VA_CHANNELS_FILE={MANIFEST_FILENAME}\n")

        with caplog.at_level(logging.WARNING):
            _wire(repo)

        assert "disagrees" not in caplog.text

    def test_no_value_is_ever_logged(self, repo, caplog):
        """The file this reads holds the facility's provider keys."""
        (repo / ".env").write_text("VA_CHANNELS_FILE=secret-looking-value.json\n")

        with caplog.at_level(logging.WARNING):
            _wire(repo)

        assert "secret-looking-value.json" not in caplog.text


class TestABuildThatGeneratedNothing:
    def test_nothing_is_written(self, barren_repo):
        assert _wire(barren_repo) == {}
        assert not (barren_repo / ".env").exists()

    def test_an_unrelated_env_is_left_alone(self, barren_repo):
        (barren_repo / ".env").write_text("ANTHROPIC_API_KEY=sk-real\n")

        assert _wire(barren_repo) == {"ANTHROPIC_API_KEY": "sk-real"}

    def test_a_leftover_pointer_is_named_but_kept(self, barren_repo, caplog):
        """The entrypoint raises on an absent manifest — it does not fall back.

        So this is the state that turns the next ``osprey up`` into a container
        that will not boot, and the operator has to be told which line to go
        remove. Told, not edited around: the build does not own that file.
        """
        (barren_repo / ".env").write_text(f"VA_CHANNELS_FILE={MANIFEST_FILENAME}\n")

        with caplog.at_level(logging.WARNING):
            env = _wire(barren_repo)

        assert env["VA_CHANNELS_FILE"] == MANIFEST_FILENAME
        assert "VA_CHANNELS_FILE" in caplog.text
        assert str(barren_repo / ".env") in caplog.text

    def test_every_leftover_key_is_named(self, barren_repo, caplog):
        (barren_repo / ".env").write_text(
            f"VA_CHANNELS_FILE={MANIFEST_FILENAME}\nVA_LATTICE=builtin\n"
        )

        with caplog.at_level(logging.WARNING):
            _wire(barren_repo)

        assert "VA_LATTICE" in caplog.text
        assert "VA_CHANNELS_FILE" in caplog.text
