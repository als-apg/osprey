"""``osprey build`` warns for each panel in a render the web terminal will not serve.

The build judges every render it produced after the lifecycle phases (a
``post_build`` step is what writes ``panels/``) and before the swap, and the
warning is advisory: the build still succeeds.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from click.testing import CliRunner

from osprey.cli import build_cmd, output
from osprey.cli.build_cmd import _report_panels_not_served, build
from tests.fixtures.lifecycle_repo import build_exemplar_repo, preserved_environ
from tests.interfaces.web_terminal.test_panel_discovery import (
    _UNMARKED_HTML,
    _manifest,
    _write_panel,
)


def _render_with_an_unmarked_panel(render: Path) -> None:
    render.mkdir(parents=True, exist_ok=True)
    (render / "config.yml").write_text(
        yaml.safe_dump({"web": {"allow_runtime_panels": True}}), encoding="utf-8"
    )
    _write_panel(render / "panels", "unmarked", manifest=_manifest(), html=_UNMARKED_HTML)


def test_each_refused_panel_is_printed_as_a_warning(tmp_path, monkeypatch):
    stage = tmp_path / "stage"
    login = stage / "demo-admin"
    _render_with_an_unmarked_panel(stage)
    _render_with_an_unmarked_panel(login)
    notes: list[str] = []
    monkeypatch.setattr(output, "note", notes.append)

    _report_panels_not_served(stage, [login])

    prefix = "⚠ [web_terminals.panel_not_served] Panel 'unmarked' will not be served: "
    assert [line[: len(prefix)] for line in notes] == [prefix, prefix]
    assert notes[0].startswith(f"{prefix}build/panels/unmarked/index.html: ")
    assert notes[1].startswith(f"{prefix}build/demo-admin/panels/unmarked/index.html: ")


def test_the_build_reports_every_render_before_the_swap(tmp_path, monkeypatch):
    repo = build_exemplar_repo(tmp_path / "exemplar")
    calls: list[tuple[Path, list[Path], bool]] = []

    def record(stage: Path, renders) -> None:
        calls.append((stage, list(renders), (stage / "config.yml").is_file()))

    monkeypatch.setattr(build_cmd, "_report_panels_not_served", record)
    with pytest.MonkeyPatch.context() as patch, preserved_environ():
        patch.chdir(tmp_path)
        result = CliRunner().invoke(build, ["--repo", str(repo), "--skip-deps", "--skip-lifecycle"])

    assert result.exit_code == 0, result.output
    [(stage, renders, staged_config)] = calls
    assert staged_config
    assert renders
    assert all(render.parent == stage for render in renders)
