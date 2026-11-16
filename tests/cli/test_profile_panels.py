"""``osprey profile panels``: which login shows which panel.

The derivation is pinned on a small repo written here, so each rule has one
profile line behind it: a selected panel is a tab, ``hidden: true`` moves it to
the menu, ``default_panel`` marks the tab a login opens on, and a cell names
the file whose line decides it.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from osprey.cli.profile_cmd import _materialize_profile_directory, profile
from osprey.cli.profile_panels import (
    STATE_DEFAULT,
    STATE_MENU,
    STATE_OFF,
    STATE_TAB,
    format_panel_matrix,
    panel_matrix,
    panel_matrix_bundle,
    panel_matrix_html,
)


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    """The demo preset as a repo, with one login file that hides and adds a panel."""
    target = tmp_path / "facility"
    _materialize_profile_directory(target, "demo")
    return target


def _render(matrix: dict, name: str) -> dict:
    return next(r for r in matrix["renders"] if r["file"] == name)


def test_the_first_column_is_the_deployments_own_profile(repo: Path) -> None:
    matrix = panel_matrix(repo)
    assert matrix["renders"][0]["file"] == "profile.yml"
    assert {r["file"] for r in matrix["renders"][1:]} == {
        str(p.relative_to(repo)) for p in (repo / "tiers").iterdir()
    }


def test_a_selected_panel_is_a_tab_and_the_default_is_marked(repo: Path) -> None:
    root = _render(panel_matrix(repo), "profile.yml")
    default = root["default_panel"]
    assert root["panels"][default]["state"] == STATE_DEFAULT
    assert root["panels"]["artifacts"] == {
        "state": STATE_TAB,
        "selected_in": "always",
        "hidden_in": None,
    }


def test_every_render_has_a_cell_for_every_panel(repo: Path) -> None:
    matrix = panel_matrix(repo)
    ids = [panel["id"] for panel in matrix["panels"]]
    for render in matrix["renders"]:
        assert sorted(render["panels"]) == sorted(ids)


def test_a_hidden_panel_is_in_the_menu_and_names_the_file_that_hid_it(repo: Path) -> None:
    layer = sorted((repo / "tiers").iterdir())[0]
    shown = str(layer.relative_to(repo))
    before = _render(panel_matrix(repo), shown)
    panel_id = next(
        pid
        for pid, cell in before["panels"].items()
        if cell["state"] == STATE_TAB and pid != "artifacts"
    )
    text = layer.read_text(encoding="utf-8")
    assert "\nconfig:\n" in text
    layer.write_text(
        text.replace("\nconfig:\n", f"\nconfig:\n  web.panels.{panel_id}.hidden: true\n", 1),
        encoding="utf-8",
    )

    cell = _render(panel_matrix(repo), shown)["panels"][panel_id]
    assert cell["state"] == STATE_MENU
    assert cell["hidden_in"] == shown


def test_a_panel_no_list_names_is_off(repo: Path) -> None:
    matrix = panel_matrix(repo)
    off = [
        (render["file"], pid)
        for render in matrix["renders"]
        for pid, cell in render["panels"].items()
        if cell["state"] == STATE_OFF
    ]
    for _file, _pid in off:
        cell = _render(matrix, _file)["panels"][_pid]
        assert cell["selected_in"] is None and cell["hidden_in"] is None


def test_the_table_has_one_row_per_panel_and_a_legend(repo: Path) -> None:
    matrix = panel_matrix(repo)
    lines = format_panel_matrix(matrix)
    assert lines[0].split()[0] == "panel"
    assert len(lines) >= 2 + len(matrix["panels"])
    assert any("DEFAULT opens first" in line for line in lines)


def test_the_page_is_self_contained_and_escapes_what_it_prints(repo: Path) -> None:
    matrix = panel_matrix(repo)
    matrix["panels"][0]["label"] = "<b>x</b>"
    page = panel_matrix_html(matrix)
    assert "<b>x</b></small>" not in page and "&lt;b&gt;x&lt;/b&gt;" in page
    assert "http://" not in page and "https://" not in page


def test_the_command_prints_the_table_and_writes_the_page(repo: Path, tmp_path: Path) -> None:
    page = tmp_path / "panels.html"
    result = CliRunner().invoke(profile, ["panels", "--repo", str(repo), "--html", str(page)])
    assert result.exit_code == 0, result.output
    assert "DEFAULT" in result.output
    assert page.read_text(encoding="utf-8").startswith("<!doctype html>")


def test_json_is_one_document_on_stdout(repo: Path) -> None:
    result = CliRunner().invoke(profile, ["panels", "--repo", str(repo), "--json"])
    assert result.exit_code == 0, result.output
    document = json.loads(result.stdout)
    assert set(document) == {"panels", "renders"}


def test_the_bundle_is_a_panel_the_terminal_serves(repo: Path, tmp_path: Path) -> None:
    from osprey.interfaces.design_system.panels.validator import validate_panel

    matrix = panel_matrix(repo)
    files = panel_matrix_bundle(matrix)
    for name, content in files.items():
        (tmp_path / name).write_text(content, encoding="utf-8")

    assert validate_panel(tmp_path) == []
    assert json.loads(files["manifest.json"])["id"] == "roster"
    page = files["index.html"]
    assert 'id="theme"' not in page, "the panel follows the terminal's theme"
    for render in matrix["renders"]:
        assert render["file"] in page


def test_a_login_that_selects_the_roster_shows_it_and_no_other_does(repo: Path) -> None:
    layer = repo / "tiers" / "admin.yml"
    text = layer.read_text(encoding="utf-8")
    assert "\nweb_panels:\n" in text
    layer.write_text(text.replace("\nweb_panels:\n", "\nweb_panels:\n  - roster\n", 1))

    matrix = panel_matrix(repo)
    assert {"id": "roster", "label": "Roster"} in matrix["panels"]
    shown = {
        render["file"]
        for render in matrix["renders"]
        if render["panels"]["roster"]["state"] != STATE_OFF
    }
    assert shown == {"tiers/admin.yml"}
