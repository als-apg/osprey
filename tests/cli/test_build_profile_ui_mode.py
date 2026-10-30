"""``web.ui_mode: expert`` beside ``web.expert_view: false`` is refused at build.

``web.expert_view: false`` rules the Expert view out and the web terminal then
resolves the mode to ``simple``; a profile — or a tier or app laid over its
deployment — that still states ``expert`` is two spellings of one fact
disagreeing, and the build names both lines.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from osprey.cli.build_profile_resolve import resolve_build_profile
from osprey.cli.build_profile_ui_mode import expert_view_errors
from osprey_connectors.errors import BuildProfileError


@pytest.mark.parametrize(
    "config",
    [
        pytest.param({"web.expert_view": False, "web.ui_mode": "expert"}, id="dotted"),
        pytest.param({"web": {"expert_view": False, "ui_mode": "expert"}}, id="nested"),
        pytest.param({"web": {"expert_view": False}, "web.ui_mode": "expert"}, id="mixed"),
        pytest.param({"web.expert_view": "false", "web.ui_mode": "expert"}, id="quoted_false"),
    ],
)
def test_expert_beside_no_expert_view_is_refused_naming_both(config) -> None:
    errors = expert_view_errors(config)
    assert len(errors) == 1
    assert "expert_view" in errors[0]
    assert "ui_mode" in errors[0]
    assert "web.ui_mode: simple" in errors[0]


@pytest.mark.parametrize(
    "config",
    [
        pytest.param({}, id="neither"),
        pytest.param({"web.ui_mode": "expert"}, id="expert_with_the_view_offered"),
        pytest.param({"web.expert_view": True, "web.ui_mode": "expert"}, id="view_true"),
        pytest.param({"web.expert_view": False}, id="ui_mode_absent"),
        pytest.param({"web.expert_view": False, "web.ui_mode": "simple"}, id="simple"),
        pytest.param([], id="not_a_mapping"),
    ],
)
def test_agreeing_or_absent_spellings_are_allowed(config) -> None:
    assert expert_view_errors(config) == []


def _write_yaml(path: Path, document: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    return path


def _repo(tmp_path: Path, tier_config: dict) -> Path:
    repo = tmp_path / "repo"
    (repo / "data").mkdir(parents=True)
    _write_yaml(
        repo / "profile.yml",
        {
            "name": "Facility",
            "data": "data",
            "deploy_services": True,
            "config": {"web.expert_view": False},
        },
    )
    return _write_yaml(
        repo / "tiers" / "readwrite.yml",
        {"name": "Facility (Read-Write)", "config": tier_config},
    )


def test_a_tier_stating_expert_over_a_deployment_without_it_is_refused(tmp_path: Path) -> None:
    tier = _repo(tmp_path, {"web.ui_mode": "expert"})
    with pytest.raises(BuildProfileError, match=r"web\.ui_mode: 'expert' but web\.expert_view"):
        resolve_build_profile(tier.resolve(), preset=None)


def test_a_tier_stating_simple_over_a_deployment_without_it_builds(tmp_path: Path) -> None:
    tier = _repo(tmp_path, {"web.ui_mode": "simple"})
    resolved, _ = resolve_build_profile(tier.resolve(), preset=None)
    assert resolved.config["web.expert_view"] is False
