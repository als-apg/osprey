"""Tests for :mod:`osprey.cli.project_utils`."""

from __future__ import annotations

from pathlib import Path

from osprey.agent_runner.project_paths import encode_claude_project_path
from osprey.cli.project_utils import resolve_config_path, resolve_project_path

_CONFIG = "project_name: demo\n"


def _repo(root: Path, *, rendered: bool = True) -> Path:
    """A deployment repo: ``profile.yml`` at the root, optionally with a render."""
    root.mkdir(parents=True, exist_ok=True)
    (root / "profile.yml").write_text("name: Demo\ndata_bundle: hello_world\n", encoding="utf-8")
    if rendered:
        (root / "build").mkdir(exist_ok=True)
        (root / "build" / "config.yml").write_text(_CONFIG, encoding="utf-8")
    return root


class TestEncodeClaudeProjectPath:
    def test_replaces_path_separators_with_dash(self):
        assert encode_claude_project_path("/foo/bar/baz") == "-foo-bar-baz"

    def test_preserves_existing_dashes(self):
        assert encode_claude_project_path("/foo/bar-baz/qux") == "-foo-bar-baz-qux"

    def test_normalizes_underscores_to_dashes(self):
        # Regression: pytest tmpdirs and macOS tmp folder names contain
        # underscores, and Claude Code's CLI dash-normalizes them too.
        out = encode_claude_project_path("/foo/my_test_dir/audit")
        assert out == "-foo-my-test-dir-audit"
        assert "_" not in out

    def test_normalizes_dots_and_other_non_alphanumerics(self):
        # Be lenient: anything that's not [A-Za-z0-9-] should become '-'
        # so the encoding tracks Claude Code's actual normalization rule.
        out = encode_claude_project_path("/foo/v1.2.3/proj!name")
        assert "." not in out
        assert "!" not in out

    def test_accepts_pathlib_path(self, tmp_path: Path):
        p = tmp_path / "demo_proj"
        p.mkdir()
        out = encode_claude_project_path(p)
        assert "_" not in out
        assert out.endswith("-demo-proj")

    def test_resolves_relative_paths_to_absolute(self, tmp_path: Path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        out = encode_claude_project_path("rel_dir")
        # Result must look absolute (i.e. begin with '-' from leading '/')
        assert out.startswith("-")
        assert out.endswith("-rel-dir")


class TestResolveProjectPath:
    """One rule: the directory in hand when it is a project, else the repo above it.

    A command that reports on a deployment must report on the same deployment
    from every directory inside it, the way ``git`` acts on the same repository
    from any subdirectory. Standing in ``data/raw`` is not a broken deployment.
    """

    def test_walks_up_to_the_enclosing_repo(self, tmp_path: Path, monkeypatch) -> None:
        repo = _repo(tmp_path / "repo")
        subdir = repo / "data" / "raw"
        subdir.mkdir(parents=True)
        monkeypatch.chdir(subdir)

        assert resolve_project_path() == repo

    def test_repo_root_is_its_own_answer(self, tmp_path: Path, monkeypatch) -> None:
        repo = _repo(tmp_path / "repo")
        monkeypatch.chdir(repo)

        assert resolve_project_path() == repo

    def test_a_named_subdirectory_resolves_to_its_repo(self, tmp_path: Path) -> None:
        repo = _repo(tmp_path / "repo")
        subdir = repo / "personas"
        subdir.mkdir()

        assert resolve_project_path(str(subdir)) == repo

    def test_a_rendered_project_directory_is_taken_as_named(self, tmp_path: Path) -> None:
        """A render holds no profile.yml, so the walk must not swallow it.

        ``--project`` accepts a rendered/container project directory, which has
        a ``config.yml`` at its root and no manifest anywhere. Redirecting one
        that happens to sit inside a repo would report on the wrong project.
        """
        repo = _repo(tmp_path / "repo")
        render = repo / "build" / "demo-readonly"
        render.mkdir(parents=True)
        (render / "config.yml").write_text(_CONFIG, encoding="utf-8")

        assert resolve_project_path(str(render)) == render

    def test_a_directory_outside_any_repo_stays_itself(self, tmp_path: Path, monkeypatch) -> None:
        loose = tmp_path / "loose"
        loose.mkdir()
        monkeypatch.chdir(loose)

        assert resolve_project_path() == loose


class TestResolveConfigPath:
    """The config a resolved project answers with: the render first, then flat."""

    def test_render_is_found_from_a_subdirectory(self, tmp_path: Path, monkeypatch) -> None:
        repo = _repo(tmp_path / "repo")
        subdir = repo / "data" / "raw"
        subdir.mkdir(parents=True)
        monkeypatch.chdir(subdir)

        assert Path(resolve_config_path()) == repo / "build" / "config.yml"

    def test_render_is_found_from_the_repo_root(self, tmp_path: Path, monkeypatch) -> None:
        repo = _repo(tmp_path / "repo")
        monkeypatch.chdir(repo)

        assert Path(resolve_config_path()) == repo / "build" / "config.yml"

    def test_flat_project_keeps_its_root_config(self, tmp_path: Path, monkeypatch) -> None:
        flat = tmp_path / "container-project"
        flat.mkdir()
        (flat / "config.yml").write_text(_CONFIG, encoding="utf-8")
        monkeypatch.chdir(flat)

        assert Path(resolve_config_path()) == flat / "config.yml"

    def test_an_unbuilt_repo_names_the_render_it_would_hold(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        """The "not found" path must be the one a build writes, not the repo root.

        Naming ``<repo>/config.yml`` tells the operator to create a file no
        build ever writes there.
        """
        repo = _repo(tmp_path / "repo", rendered=False)
        monkeypatch.chdir(repo)

        assert Path(resolve_config_path()) == repo / "build" / "config.yml"

    def test_a_directory_that_is_nothing_names_its_own_config(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        loose = tmp_path / "loose"
        loose.mkdir()
        monkeypatch.chdir(loose)

        assert Path(resolve_config_path()) == loose / "config.yml"

    def test_explicit_config_flag_still_wins(self, tmp_path: Path, monkeypatch) -> None:
        repo = _repo(tmp_path / "repo")
        monkeypatch.chdir(repo)

        assert resolve_config_path(config_arg="/elsewhere/custom.yml") == "/elsewhere/custom.yml"
