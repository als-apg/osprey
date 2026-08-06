"""Unit tests for the version resolution chain in osprey.version."""

import subprocess

import pytest

from osprey import version as version_module
from osprey.version import get_release_version, get_running_version, is_release


@pytest.fixture(autouse=True)
def _clear_version_cache():
    """Drop the memoised version between tests so each one resolves fresh."""
    get_running_version.cache_clear()
    get_release_version.cache_clear()
    yield
    get_running_version.cache_clear()
    get_release_version.cache_clear()


def _git(*args, cwd):
    subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True)


def _make_repo(path, tag, extra_commits=0):
    """Build a throwaway git repo carrying ``tag`` and ``extra_commits`` past it."""
    path.mkdir(parents=True, exist_ok=True)
    _git("init", "-q", cwd=path)
    _git("config", "user.email", "test@example.com", cwd=path)
    _git("config", "user.name", "Test", cwd=path)
    (path / "pyproject.toml").write_text('[project]\nname = "osprey-framework"\n')
    _git("add", "-A", cwd=path)
    _git("commit", "-qm", "initial", cwd=path)
    _git("tag", tag, cwd=path)
    for index in range(extra_commits):
        (path / f"file{index}.txt").write_text(str(index))
        _git("add", "-A", cwd=path)
        _git("commit", "-qm", f"commit {index}", cwd=path)
    return path


class TestDescribeParsing:
    def test_exact_tag_is_the_bare_release(self):
        assert version_module._pep440_from_describe("v2026.6.2-0-g1234567") == "2026.6.2"

    def test_distance_becomes_a_post_release_with_the_commit(self):
        assert (
            version_module._pep440_from_describe("v2026.6.2-783-g83fda5e60")
            == "2026.6.2.post783+g83fda5e60"
        )

    def test_dirty_tree_is_marked_and_never_reads_as_a_release(self):
        parsed = version_module._pep440_from_describe("v2026.6.2-0-g1234567-dirty")
        assert parsed is not None
        assert parsed.startswith("2026.6.2.post0+g1234567.d")

    def test_unparseable_output_returns_none(self):
        assert version_module._pep440_from_describe("not-a-describe") is None
        assert version_module._pep440_from_describe("") is None


class TestResolutionChain:
    def test_source_checkout_reports_distance_past_the_tag(self, tmp_path, monkeypatch):
        repo = _make_repo(tmp_path / "osprey", "v2026.6.2", extra_commits=3)
        monkeypatch.setattr(version_module, "_SOURCE_ROOT", repo)

        running = get_running_version()
        assert running.startswith("2026.6.2.post3+g")
        assert get_release_version() == "2026.6.2"
        assert is_release() is False

    def test_clean_tag_is_a_release(self, tmp_path, monkeypatch):
        repo = _make_repo(tmp_path / "osprey", "v2026.6.2")
        monkeypatch.setattr(version_module, "_SOURCE_ROOT", repo)

        assert get_running_version() == "2026.6.2"
        assert get_release_version() == "2026.6.2"
        assert is_release() is True

    def test_does_not_pick_up_an_unrelated_repos_tags(self, tmp_path, monkeypatch):
        """The anchor must beat cwd.

        ``osprey build`` runs ``git init`` in the project it generates, and the
        status line imports osprey from the operator's own repo. A cwd-relative
        probe would report that repo's tag as OSPREY's version.
        """
        unrelated = _make_repo(tmp_path / "someone-elses-repo", "v9.9.9", extra_commits=2)
        (unrelated / "pyproject.toml").write_text('[project]\nname = "not-osprey"\n')
        _git("add", "-A", cwd=unrelated)
        _git("commit", "-qm", "rename", cwd=unrelated)

        installed = tmp_path / "site-packages-parent"
        installed.mkdir()
        monkeypatch.setattr(version_module, "_SOURCE_ROOT", installed)
        monkeypatch.setattr(version_module, "_version_from_stamp", lambda: "2026.6.2")
        monkeypatch.chdir(unrelated)

        assert get_running_version() == "2026.6.2"

    def test_pyproject_declaring_another_project_is_rejected(self, tmp_path, monkeypatch):
        repo = _make_repo(tmp_path / "osprey", "v2026.6.2", extra_commits=1)
        (repo / "pyproject.toml").write_text('[project]\nname = "something-else"\n')
        monkeypatch.setattr(version_module, "_SOURCE_ROOT", repo)
        monkeypatch.setattr(version_module, "_version_from_stamp", lambda: "1.2.3")

        assert get_running_version() == "1.2.3"

    def test_stamp_is_used_when_there_is_no_git(self, tmp_path, monkeypatch):
        monkeypatch.setattr(version_module, "_SOURCE_ROOT", tmp_path)
        monkeypatch.setattr(version_module, "_version_from_stamp", lambda: "2026.6.2.post5+gabc123")

        assert get_running_version() == "2026.6.2.post5+gabc123"
        assert get_release_version() == "2026.6.2"
        assert is_release() is False

    def test_metadata_is_used_when_there_is_no_stamp(self, tmp_path, monkeypatch):
        monkeypatch.setattr(version_module, "_SOURCE_ROOT", tmp_path)
        monkeypatch.setattr(version_module, "_version_from_stamp", lambda: None)
        monkeypatch.setattr(version_module, "_version_from_metadata", lambda: "2026.6.2")

        assert get_running_version() == "2026.6.2"

    def test_missing_git_binary_falls_through_without_raising(self, tmp_path, monkeypatch):
        repo = _make_repo(tmp_path / "osprey", "v2026.6.2", extra_commits=1)
        monkeypatch.setattr(version_module, "_SOURCE_ROOT", repo)
        monkeypatch.setattr(version_module, "_version_from_stamp", lambda: "2026.6.2")

        def _no_git(*args, **kwargs):
            raise FileNotFoundError("git")

        monkeypatch.setattr(version_module.subprocess, "run", _no_git)

        assert get_running_version() == "2026.6.2"

    def test_git_timeout_falls_through_without_raising(self, tmp_path, monkeypatch):
        repo = _make_repo(tmp_path / "osprey", "v2026.6.2", extra_commits=1)
        monkeypatch.setattr(version_module, "_SOURCE_ROOT", repo)
        monkeypatch.setattr(version_module, "_version_from_stamp", lambda: "2026.6.2")

        def _slow(*args, **kwargs):
            raise subprocess.TimeoutExpired(cmd="git", timeout=2)

        monkeypatch.setattr(version_module.subprocess, "run", _slow)

        assert get_running_version() == "2026.6.2"

    def test_shallow_clone_without_tags_falls_through(self, tmp_path, monkeypatch):
        repo = tmp_path / "osprey"
        repo.mkdir()
        _git("init", "-q", cwd=repo)
        _git("config", "user.email", "test@example.com", cwd=repo)
        _git("config", "user.name", "Test", cwd=repo)
        (repo / "pyproject.toml").write_text('[project]\nname = "osprey-framework"\n')
        _git("add", "-A", cwd=repo)
        _git("commit", "-qm", "initial", cwd=repo)
        monkeypatch.setattr(version_module, "_SOURCE_ROOT", repo)
        monkeypatch.setattr(version_module, "_version_from_stamp", lambda: "2026.6.2")

        assert get_running_version() == "2026.6.2"


class TestUnresolvableEnvironment:
    def test_falls_back_without_crashing(self, tmp_path, monkeypatch):
        monkeypatch.setattr(version_module, "_SOURCE_ROOT", tmp_path)
        monkeypatch.setattr(version_module, "_version_from_stamp", lambda: None)
        monkeypatch.setattr(version_module, "_version_from_metadata", lambda: None)

        assert get_running_version() == "0.0.0+unknown"

    def test_fails_closed_rather_than_claiming_to_be_a_release(self, tmp_path, monkeypatch):
        """A broken environment must never pin — hence the local segment."""
        monkeypatch.setattr(version_module, "_SOURCE_ROOT", tmp_path)
        monkeypatch.setattr(version_module, "_version_from_stamp", lambda: None)
        monkeypatch.setattr(version_module, "_version_from_metadata", lambda: None)

        assert is_release() is False


class TestLiveCheckout:
    def test_release_version_matches_the_newest_tag(self):
        """The running checkout's release lineage is the newest v* tag reachable."""
        described = subprocess.run(
            ["git", "describe", "--tags", "--abbrev=0", "--match", "v[0-9]*"],
            cwd=version_module._SOURCE_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        if described.returncode != 0:
            pytest.skip("no reachable v* tag in this checkout")

        assert get_release_version() == described.stdout.strip().removeprefix("v")
