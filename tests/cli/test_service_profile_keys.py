"""Every bundled service's ``services.<name>`` block is closed.

A bundled service's valid sub-keys are the config-key manifest's rows under its
family (``services.<family>.*``) plus the generic axes any service block may
carry (:data:`~osprey.cli.build_profile_model.SERVICE_AXES`). A second plan
lane is an instance of the bluesky service and is closed against its rows; the
live stand-in keeps rows of its own. A custom service — one OSPREY does not
bundle, declared by a ``path`` to the facility's own template — stays open.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

from osprey.bluesky_bridge_connection import SECOND_LANE_KEYS
from osprey.cli.build_profile_load import (
    _parse_profile,
    bundled_service_names,
    manifest_rows,
    reject_unknown_closed_config_keys,
    service_family,
)
from osprey.cli.build_profile_model import SERVICE_AXES
from osprey.cli.build_profile_resolve import resolve_build_profile
from osprey.errors import BuildProfileError

REPO_ROOT = Path(__file__).resolve().parents[2]
PRESET_DIR = REPO_ROOT / "src" / "osprey" / "profiles" / "presets"
FROZEN_FIXTURES = REPO_ROOT / "tests" / "fixtures" / "explicit_config"


def _parse(config: dict[str, Any]) -> None:
    _parse_profile({"name": "probe", "config": config})


def _rows_under(prefix: str) -> set[str]:
    head = f"{prefix}."
    return {key[len(head) :].split(".", 1)[0] for key in manifest_rows() if key.startswith(head)}


def test_an_unknown_key_under_a_bundled_service_is_refused_with_the_valid_set() -> None:
    with pytest.raises(BuildProfileError) as caught:
        _parse({"services.dispatch_worker.worker_cnt": 2})

    message = str(caught.value)
    assert "Unknown services.dispatch_worker key(s): 'worker_cnt'" in message
    assert "did you mean 'worker_count'?" in message
    for key in _rows_under("services.dispatch_worker") | SERVICE_AXES:
        assert key in message


@pytest.mark.parametrize("lane", sorted(SECOND_LANE_KEYS.values()))
def test_a_lane_instance_is_closed_against_its_familys_rows(lane: str) -> None:
    assert service_family(lane) == "bluesky"
    _parse({f"services.{lane}": {"path": "./services/bluesky", "port": 10081, "target": "va"}})

    with pytest.raises(BuildProfileError) as caught:
        _parse({f"services.{lane}.plan_dirs": "plans"})

    message = str(caught.value)
    assert f"Unknown services.{lane} key(s): 'plan_dirs'" in message
    assert "did you mean 'plan_dir'?" in message


def test_a_second_stand_in_instance_is_closed_against_the_live_standin_rows() -> None:
    assert service_family("live_standin") == "live_standin"
    _parse({"services.live_standin": {"path": "./services/virtual_accelerator", "port": 5066}})

    # `pva_port` is a row of the baseline instance, not of the stand-in.
    with pytest.raises(BuildProfileError) as caught:
        _parse({"services.live_standin.pva_port": 5076})

    message = str(caught.value)
    assert "Unknown services.live_standin key(s): 'pva_port'" in message
    for key in _rows_under("services.live_standin") | SERVICE_AXES:
        assert key in message


@pytest.mark.parametrize(
    ("axis", "value"),
    [
        ("path", "./services/x"),
        ("image", "registry.example.org/x:pinned"),
        ("port", 18123),
        ("network", "bridge"),
        ("http", True),
        ("listens", True),
        ("bind_env", "X_BIND"),
        ("env", ["X_EXTRA"]),
    ],
)
def test_a_generic_axis_passes_on_every_bundled_service(axis: str, value: Any) -> None:
    assert axis in SERVICE_AXES
    for name in sorted(bundled_service_names()):
        reject_unknown_closed_config_keys({f"services.{name}.{axis}": value})


def test_an_env_variable_below_the_env_row_passes() -> None:
    assert manifest_rows()["services.bluesky.env"]["data-map"] is True
    _parse({"services.bluesky.env": ["BLUESKY_EXTRA"]})
    reject_unknown_closed_config_keys({"services": {"bluesky": {"env": {"BLUESKY_EXTRA": "1"}}}})


def test_a_custom_service_with_a_template_directory_keeps_open_sub_keys() -> None:
    # Whether `path` resolves to a template is the build's question (a service
    # must be real); the parser only decides that the block is not OSPREY's.
    assert "my_collector" not in bundled_service_names()
    _parse(
        {
            "services.my_collector": {
                "path": "./services/my_collector",
                "retention": 7,
                "anything": {"nested": True},
            }
        }
    )


def test_a_malformed_axis_keeps_its_own_message(tmp_path: Path) -> None:
    (tmp_path / "data").mkdir()
    profile = _parse_profile(
        {"name": "probe", "data": "data", "config": {"services.gchat_bridge.network": "hostnet"}}
    )

    with pytest.raises(BuildProfileError) as caught:
        profile.validate(tmp_path)

    message = str(caught.value)
    assert "services.gchat_bridge.network must be one of bridge, host (got 'hostnet')" in message
    assert "Unknown services" not in message


def _shipped_documents() -> list[tuple[str, str]]:
    presets = [("preset", path.stem) for path in sorted(PRESET_DIR.glob("*.yml"))]
    fixtures = [
        ("fixture", str(path.relative_to(FROZEN_FIXTURES)))
        for path in sorted(FROZEN_FIXTURES.rglob("*.yml"))
    ]
    return presets + fixtures


@pytest.mark.parametrize(("kind", "name"), _shipped_documents())
def test_every_preset_and_frozen_fixture_parses_under_the_closed_services(
    kind: str, name: str
) -> None:
    """Every shipped preset parses, and every frozen render passes the same check."""
    if kind == "preset":
        resolve_build_profile(None, name)
    else:
        rendered = yaml.safe_load((FROZEN_FIXTURES / name).read_text(encoding="utf-8"))
        reject_unknown_closed_config_keys(rendered)
