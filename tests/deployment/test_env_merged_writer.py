"""The merged env chain, regenerated inside the shared env-file-args seam.

A compose provider that keeps only the LAST ``--env-file`` it is handed cannot
be given the chain as a list: everything but ``.env`` would be dropped, and the
shared defaults the deployment relies on would simply not be there. So the
merge happens ahead of it, into one file under the render zone, and the seam
every lifecycle verb resolves its env-file argv through is where that file is
written — which makes "the file compose reads" and "the chain on disk right
now" the same thing on every invocation, rather than only on the ones that
happen to follow a build.

The two ways a cached copy goes wrong are the two these tests pin: a project
built but never brought up has no merged file at all, and a hand-edited
``.env`` leaves a stale one that would start the stack on the superseded value
with nothing on screen to say so.

The other half is the refusal. A ``$`` anywhere in the chain is mangled by
every compose implementation and by each of them differently, so the deploy
stops before the value is copied into a second file — naming the offending
variables and never their values.
"""

from __future__ import annotations

import os
import stat
from pathlib import Path

import pytest

from osprey.deployment import container_lifecycle
from osprey.deployment.container_lifecycle import (
    ENV_MERGED_RELPATH,
    _env_file_args,
    _write_env_merged_for_compose,
)
from osprey.deployment.errors import ComposeInterpolationError
from osprey.deployment.runtime_helper import ComposeProvider
from osprey.utils.dotenv import ENV_MERGED_BANNER, parse_dotenv_file


def write_chain(repo: Path, shared: str | None = None, local: str | None = None) -> Path:
    """Lay down a repo whose env chain has the given ``.env.shared``/``.env``."""
    (repo / "build").mkdir(parents=True, exist_ok=True)
    if shared is not None:
        (repo / ".env.shared").write_text(shared, encoding="utf-8")
    if local is not None:
        (repo / ".env").write_text(local, encoding="utf-8")
    return repo


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    """A deployment repo with both chain files and one key set in both."""
    return write_chain(
        tmp_path,
        shared="SHARED_ONLY=defaults\nBOTH=from-shared\n",
        local="LOCAL_ONLY=secret\nBOTH=from-local\n",
    )


def merged_path(repo: Path) -> Path:
    return repo / ENV_MERGED_RELPATH


class TestMergedFileContents:
    """What the written file says — and what it must not."""

    def test_local_wins_over_shared(self, repo: Path) -> None:
        parsed = parse_dotenv_file(_write_env_merged_for_compose(repo))
        assert parsed == {
            "SHARED_ONLY": "defaults",
            "BOTH": "from-local",
            "LOCAL_ONLY": "secret",
        }

    def test_written_under_the_render_zone_with_a_leading_dot(self, repo: Path) -> None:
        # The name is what puts the file inside every existing `.env*` secret
        # sweep — of the build zone and of the image contexts — for free.
        written = _write_env_merged_for_compose(repo)
        assert written == repo / "build" / ".env.merged"
        assert written.name.startswith(".env")

    def test_carries_the_do_not_edit_banner(self, repo: Path) -> None:
        # It is a machine artifact: whoever opens it is told where the
        # editable originals are, because edits here are erased next invocation.
        written = _write_env_merged_for_compose(repo).read_text(encoding="utf-8")
        assert written.startswith(ENV_MERGED_BANNER)

    def test_born_owner_only(self, repo: Path) -> None:
        # It inherits the local half's secrets, so it inherits its permissions.
        written = _write_env_merged_for_compose(repo)
        assert stat.S_IMODE(written.stat().st_mode) == 0o600

    def test_values_survive_the_round_trip(self, tmp_path: Path) -> None:
        # Written through the quoting helper, so a value with spaces or an
        # inline `#` reaches compose whole rather than truncated.
        awkward = write_chain(tmp_path, local='SPACED=two words\nHASHED=a#b\nPADDED="  keep  "\n')
        parsed = parse_dotenv_file(_write_env_merged_for_compose(awkward))
        assert parsed == {"SPACED": "two words", "HASHED": "a#b", "PADDED": "  keep  "}

    def test_creates_the_render_zone_when_absent(self, tmp_path: Path) -> None:
        fresh = tmp_path / "repo"
        fresh.mkdir()
        (fresh / ".env").write_text("A=1\n", encoding="utf-8")
        assert parse_dotenv_file(_write_env_merged_for_compose(fresh)) == {"A": "1"}


class TestRegeneration:
    """Written on the invocation that needs it, not on the last build."""

    def test_fresh_build_never_upped(self, repo: Path) -> None:
        # Nothing has written the file yet; the first compose invocation must
        # not be the one that discovers that.
        assert not merged_path(repo).exists()
        args = _env_file_args(repo, provider=ComposeProvider.PODMAN_COMPOSE)
        assert args == ["--env-file", str(merged_path(repo))]
        assert parse_dotenv_file(merged_path(repo))["BOTH"] == "from-local"

    def test_stale_after_an_edit_to_the_chain(self, repo: Path) -> None:
        _env_file_args(repo, provider=ComposeProvider.PODMAN_COMPOSE)
        (repo / ".env").write_text("LOCAL_ONLY=secret\nBOTH=edited\n", encoding="utf-8")

        _env_file_args(repo, provider=ComposeProvider.PODMAN_COMPOSE)

        assert parse_dotenv_file(merged_path(repo))["BOTH"] == "edited"

    def test_a_removed_key_is_gone_rather_than_preserved(self, repo: Path) -> None:
        # Regenerated from the chain, not merged into what was there before:
        # the file says what the chain says now.
        _env_file_args(repo, provider=ComposeProvider.PODMAN_COMPOSE)
        (repo / ".env").write_text("BOTH=from-local\n", encoding="utf-8")

        _env_file_args(repo, provider=ComposeProvider.PODMAN_COMPOSE)

        assert "LOCAL_ONLY" not in parse_dotenv_file(merged_path(repo))

    def test_rewrite_retightens_permissions(self, repo: Path) -> None:
        _env_file_args(repo, provider=ComposeProvider.PODMAN_COMPOSE)
        os.chmod(merged_path(repo), 0o644)

        _env_file_args(repo, provider=ComposeProvider.PODMAN_COMPOSE)

        assert stat.S_IMODE(merged_path(repo).stat().st_mode) == 0o600


class TestSeamShape:
    """Which providers get the merged file, and which keep the two-flag chain."""

    def test_podman_compose_gets_the_single_merged_file(self, repo: Path) -> None:
        assert _env_file_args(repo, provider=ComposeProvider.PODMAN_COMPOSE) == [
            "--env-file",
            str(merged_path(repo)),
        ]

    def test_docker_v2_keeps_the_two_flag_chain(self, repo: Path) -> None:
        assert _env_file_args(repo, provider=ComposeProvider.DOCKER_V2) == [
            "--env-file",
            str(repo / ".env.shared"),
            "--env-file",
            str(repo / ".env"),
        ]
        assert not merged_path(repo).exists()

    def test_unprobed_callers_keep_the_two_flag_chain(self, repo: Path) -> None:
        # Call sites that have not threaded a probe result down are unchanged:
        # the merged file is written only for the shape that needs it.
        assert _env_file_args(repo) == [
            "--env-file",
            str(repo / ".env.shared"),
            "--env-file",
            str(repo / ".env"),
        ]
        assert not merged_path(repo).exists()

    def test_repo_root_defaults_to_the_working_directory(
        self, repo: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.chdir(repo)
        assert _env_file_args(provider=ComposeProvider.PODMAN_COMPOSE) == [
            "--env-file",
            str(merged_path(repo)),
        ]

    def test_shared_only_still_merges(self, tmp_path: Path) -> None:
        # A project carrying only committed defaults has a chain, and compose
        # must be handed it — `.env` is not what makes the chain real.
        shared_only = write_chain(tmp_path, shared="SHARED_ONLY=defaults\n")
        assert _env_file_args(shared_only, provider=ComposeProvider.PODMAN_COMPOSE) == [
            "--env-file",
            str(merged_path(shared_only)),
        ]
        assert parse_dotenv_file(merged_path(shared_only)) == {"SHARED_ONLY": "defaults"}

    def test_no_chain_at_all_writes_nothing(self, tmp_path: Path) -> None:
        # Compose hard-fails on an `--env-file` that is not there, and an empty
        # merged file would promise values this repo does not have.
        empty = write_chain(tmp_path)
        assert _env_file_args(empty, provider=ComposeProvider.PODMAN_COMPOSE) == []
        assert not merged_path(empty).exists()


class TestDollarRefusal:
    """The chain-wide `$`-scan, ahead of the write."""

    def test_refuses_a_dollar_in_the_local_file(self, tmp_path: Path) -> None:
        bad = write_chain(tmp_path, local="TOKEN=h0rse$battery\n")
        with pytest.raises(ComposeInterpolationError) as excinfo:
            _env_file_args(bad, provider=ComposeProvider.PODMAN_COMPOSE)
        assert excinfo.value.variables == ["TOKEN"]

    def test_refuses_a_dollar_in_the_shared_file(self, tmp_path: Path) -> None:
        # Compose reads every chain file through the same mangling parser, so a
        # `$` in the committed defaults is delivered just as wrongly.
        bad = write_chain(tmp_path, shared="SHARED_TOKEN=a$b\n", local="OK=fine\n")
        with pytest.raises(ComposeInterpolationError) as excinfo:
            _env_file_args(bad, provider=ComposeProvider.PODMAN_COMPOSE)
        assert excinfo.value.variables == ["SHARED_TOKEN"]

    def test_refusal_names_variables_never_values(self, tmp_path: Path) -> None:
        bad = write_chain(tmp_path, shared="B_VAR=x$y\n", local="A_VAR=h0rse$battery\n")
        with pytest.raises(ComposeInterpolationError) as excinfo:
            _env_file_args(bad, provider=ComposeProvider.PODMAN_COMPOSE)
        assert excinfo.value.variables == ["A_VAR", "B_VAR"]
        message = str(excinfo.value)
        assert "A_VAR" in message and "B_VAR" in message
        for secret in ("h0rse", "battery", "x$y"):
            assert secret not in message

    def test_refusal_names_the_chain_files_it_scanned(self, tmp_path: Path) -> None:
        bad = write_chain(tmp_path, shared="S=ok\n", local="TOKEN=a$b\n")
        with pytest.raises(ComposeInterpolationError) as excinfo:
            _env_file_args(bad, provider=ComposeProvider.PODMAN_COMPOSE)
        assert str(bad / ".env.shared") in excinfo.value.path
        assert str(bad / ".env") in excinfo.value.path

    def test_nothing_is_written_when_the_scan_refuses(self, tmp_path: Path) -> None:
        # The scan runs BEFORE the write, so a value compose would mangle is
        # never copied into a second file on disk.
        bad = write_chain(tmp_path, local="TOKEN=a$b\n")
        with pytest.raises(ComposeInterpolationError):
            _env_file_args(bad, provider=ComposeProvider.PODMAN_COMPOSE)
        assert not merged_path(bad).exists()

    def test_a_stale_merged_file_is_not_left_in_force(self, repo: Path) -> None:
        # A chain that was clean and then acquired a `$` must not leave the
        # previous merged file behind as the one compose would read.
        _env_file_args(repo, provider=ComposeProvider.PODMAN_COMPOSE)
        (repo / ".env").write_text("BOTH=a$b\n", encoding="utf-8")

        with pytest.raises(ComposeInterpolationError):
            _env_file_args(repo, provider=ComposeProvider.PODMAN_COMPOSE)

        assert parse_dotenv_file(merged_path(repo))["BOTH"] == "from-local"

    def test_docker_shape_is_not_gated_by_this_scan(self, tmp_path: Path) -> None:
        # The docker path keeps its own refusal, at the mint; this seam adds no
        # second one for a shape that never writes the merged file.
        bad = write_chain(tmp_path, local="TOKEN=a$b\n")
        assert _env_file_args(bad, provider=ComposeProvider.DOCKER_V2) == [
            "--env-file",
            str(bad / ".env"),
        ]


def test_relpath_is_the_one_spelling_of_the_artifact() -> None:
    # Later tasks (the argv branch, the reset classifier, the docs) name this
    # file too; they all name it from here.
    assert ENV_MERGED_RELPATH == Path("build") / ".env.merged"
    assert container_lifecycle.ENV_MERGED_RELPATH.name == ".env.merged"
