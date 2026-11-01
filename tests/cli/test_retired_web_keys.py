"""The two retired ``web:`` keys, in every spelling and every layer.

``web.expert_view`` said whether a deployment offers a terminal, so a document
that still states it is refused: read as silence, its ``false`` would hand every
login a terminal. ``web.ui_mode`` chose a layout that no longer exists, and every
repo materialized before now states it, so it warns and is dropped.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest
import yaml

from osprey.cli.build_profile import resolve_build_profile
from osprey.cli.build_profile_document import _WARNED_RETIRED_KEYS, normalize_retired_web_keys
from osprey.cli.build_profile_resolve import cli_edit_layer
from osprey_connectors.errors import BuildProfileError


@pytest.fixture(autouse=True)
def _warn_afresh():
    _WARNED_RETIRED_KEYS.clear()
    yield
    _WARNED_RETIRED_KEYS.clear()


def _write_yaml(path: Path, document: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    return path


def _repo(tmp_path: Path, root_config: dict) -> Path:
    repo = tmp_path / "repo"
    (repo / "data").mkdir(parents=True)
    return _write_yaml(
        repo / "profile.yml",
        {"name": "Facility", "data": "data", "deploy_services": True, "config": root_config},
    )


@pytest.mark.parametrize(
    ("config", "spelled"),
    [
        pytest.param({"web.expert_view": False}, "web.expert_view: false", id="dotted"),
        pytest.param({"web": {"expert_view": False}}, "web: expert_view: false", id="nested"),
        pytest.param({"web.expert_view": True}, "web.expert_view: true", id="true_too"),
    ],
)
def test_the_terminal_key_is_refused_naming_its_line_and_successor(config, spelled) -> None:
    with pytest.raises(BuildProfileError) as refusal:
        normalize_retired_web_keys({"config": config}, "profile.yml")
    assert f"profile.yml states `{spelled}`" in str(refusal.value)
    assert f"`web.terminal: {spelled.rsplit(' ', 1)[1]}` in its place" in str(refusal.value)


@pytest.mark.parametrize("layer", ["tiers/readwrite.yml", "apps/logbook.yml"])
def test_an_overlay_stating_the_terminal_key_refuses_the_build(tmp_path: Path, layer: str) -> None:
    _repo(tmp_path, {})
    path = _write_yaml(
        tmp_path / "repo" / layer, {"name": "Layer", "config": {"web.expert_view": False}}
    )
    with pytest.raises(BuildProfileError, match=r"web\.terminal: false"):
        resolve_build_profile(path.resolve(), preset=None)


def test_a_set_pair_stating_the_terminal_key_is_refused() -> None:
    with pytest.raises(BuildProfileError, match=r"web\.terminal: false"):
        cli_edit_layer(("config.web.expert_view=false",))


@pytest.mark.parametrize(
    ("config", "spelled", "left"),
    [
        pytest.param({"web.ui_mode": "simple"}, "web.ui_mode", {}, id="dotted"),
        pytest.param({"web": {"ui_mode": "expert"}}, "web: ui_mode", {}, id="nested"),
        pytest.param(
            {"web": {"ui_mode": "expert", "theme": "main"}},
            "web: ui_mode",
            {"web": {"theme": "main"}},
            id="beside_a_key_that_stays",
        ),
    ],
)
def test_the_layout_key_warns_naming_its_line_and_is_dropped(
    config, spelled, left, caplog: pytest.LogCaptureFixture
) -> None:
    raw = {"config": config}
    with caplog.at_level(logging.WARNING):
        normalize_retired_web_keys(raw, "tiers/a.yml")
    assert raw["config"] == left
    assert f"tiers/a.yml: `{spelled}` is no longer read; delete it" in caplog.text


def test_a_repo_still_stating_the_layout_key_builds_without_it(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    root = _repo(tmp_path, {"web.ui_mode": "expert", "web.theme": "main"})
    with caplog.at_level(logging.WARNING):
        resolved, _ = resolve_build_profile(root.resolve(), preset=None)
    assert "web.ui_mode" not in resolved.config
    assert resolved.config["web.theme"] == "main"
    assert "`web.ui_mode` is no longer read; delete it" in caplog.text


def test_the_new_key_passes_untouched(caplog: pytest.LogCaptureFixture) -> None:
    raw = {"config": {"web.terminal": False}}
    with caplog.at_level(logging.WARNING):
        normalize_retired_web_keys(raw, "profile.yml")
    assert raw == {"config": {"web.terminal": False}}
    assert not caplog.text
