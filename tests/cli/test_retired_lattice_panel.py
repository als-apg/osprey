"""The retired LATTICE panel, in every spelling a profile document can carry it.

Every repo materialized from the demo preset selects ``lattice``,
and a tab that no longer exists has nothing to protect, so each spelling warns
once and is dropped rather than refusing the build.
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

import pytest
import yaml

import osprey
from osprey.cli.build_profile import resolve_build_profile
from osprey.cli.build_profile_document import _WARNED_RETIRED_KEYS, normalize_retired_panel

BUNDLED_LATTICE_APP = Path(osprey.__file__).parent / "profiles" / "presets" / "apps" / "lattice"


@pytest.fixture(autouse=True)
def _warn_afresh():
    _WARNED_RETIRED_KEYS.clear()
    yield
    _WARNED_RETIRED_KEYS.clear()


def test_the_panel_selection_loses_it_with_one_warning(caplog: pytest.LogCaptureFixture) -> None:
    raw = {"web_panels": ["ariel", "lattice", "okf"]}
    with caplog.at_level(logging.WARNING):
        normalize_retired_panel(raw, "profile.yml")
        normalize_retired_panel(raw | {"web_panels": ["lattice"]}, "profile.yml")
    assert raw == {"web_panels": ["ariel", "okf"]}
    assert caplog.text.count("`web_panels: lattice` is no longer read") == 1


def _repo(tmp_path: Path, **keys: object) -> Path:
    repo = tmp_path / "repo"
    (repo / "data").mkdir(parents=True)
    root = repo / "profile.yml"
    document = {"name": "Facility", "data": "data", "deploy_services": True, **keys}
    root.write_text(yaml.safe_dump(document), encoding="utf-8")
    return root.resolve()


def test_a_repo_still_opening_on_it_builds_without_a_default_panel(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    root = _repo(tmp_path, default_panel="lattice")
    with caplog.at_level(logging.WARNING):
        resolved, _ = resolve_build_profile(root, preset=None)
    assert resolved.default_panel is None
    assert "`default_panel: lattice` is no longer read" in caplog.text


def test_a_repo_running_the_lattice_app_opens_on_its_panel(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    root = _repo(tmp_path, default_panel="lattice", apps={"lattice": {}})
    shutil.copytree(BUNDLED_LATTICE_APP, root.parent / "apps" / "lattice")
    with caplog.at_level(logging.WARNING):
        resolved, _ = resolve_build_profile(root, preset=None)
    assert resolved.default_panel == "lattice"
    assert "no longer read" not in caplog.text


@pytest.mark.parametrize(
    ("config", "left", "spelled"),
    [
        pytest.param(
            {"lattice_dashboard.port": 9400, "web.theme": "main"},
            {"web.theme": "main"},
            "lattice_dashboard.port",
            id="dotted_section",
        ),
        pytest.param(
            {"lattice_dashboard": {"host": "127.0.0.1"}},
            {},
            "lattice_dashboard",
            id="nested_section",
        ),
        pytest.param(
            {"web": {"panels": {"lattice": {"label": "OPTICS"}, "okf": True}}},
            {"web": {"panels": {"okf": True}}},
            "web.panels.lattice",
            id="panel_block_beside_one_that_stays",
        ),
        pytest.param(
            {"web.panels.lattice.enabled": True},
            {},
            "web.panels.lattice.enabled",
            id="dotted_panel_leaf",
        ),
    ],
)
def test_its_config_keys_warn_and_are_dropped(
    config, left, spelled, caplog: pytest.LogCaptureFixture
) -> None:
    raw = {"config": config}
    with caplog.at_level(logging.WARNING):
        normalize_retired_panel(raw, "tiers/a.yml")
    assert raw["config"] == left
    assert f"tiers/a.yml: `config: {spelled}` is no longer read" in caplog.text


def test_a_repo_still_selecting_it_builds_without_it(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    repo = tmp_path / "repo"
    (repo / "data").mkdir(parents=True)
    root = repo / "profile.yml"
    root.write_text(
        yaml.safe_dump(
            {
                "name": "Facility",
                "data": "data",
                "deploy_services": True,
                "web_panels": ["lattice"],
                "config": {"lattice_dashboard.auto_launch": True},
            }
        ),
        encoding="utf-8",
    )
    with caplog.at_level(logging.WARNING):
        resolved, _ = resolve_build_profile(root.resolve(), preset=None)
    assert "lattice" not in resolved.web_panels
    assert not any(key.startswith("lattice_dashboard") for key in resolved.config)
    assert "the LATTICE panel was retired" in caplog.text


def test_other_panels_pass_untouched(caplog: pytest.LogCaptureFixture) -> None:
    raw = {"web_panels": ["ariel"], "config": {"web.panels.ariel.label": "LOG"}}
    with caplog.at_level(logging.WARNING):
        normalize_retired_panel(raw, "profile.yml")
    assert raw == {"web_panels": ["ariel"], "config": {"web.panels.ariel.label": "LOG"}}
    assert not caplog.text
