"""Tests for :mod:`osprey.cli.project_utils`."""

from __future__ import annotations

from pathlib import Path

from osprey.cli.project_utils import encode_claude_project_path


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
