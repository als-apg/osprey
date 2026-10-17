"""A service must be real before the build renders compose files for it.

Every ``services.<name>`` block and every ``deployed_services`` entry of the
rendered config names a service the build can render: a template OSPREY
bundles, or a block whose ``path`` names a directory holding a
``docker-compose.yml.j2`` in the render. Anything else is refused as an
ordinary build error, on the rendered config, before the compose render — so
the generator's own ``RuntimeError`` for an unknown name is never what an
operator meets.
"""

from __future__ import annotations

import logging
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import yaml
from click.testing import CliRunner

import osprey.deployment.compose_generator as compose_generator
from osprey.cli.build_cmd import (
    _bundled_service_names,
    _render_compose_files,
    _service_existence_errors,
    build,
)
from osprey.cli.init_cmd import init
from osprey_connectors.errors import BuildProfileError


def _stage(root: Path, config: dict[str, Any]) -> SimpleNamespace:
    (root / "config.yml").write_text(yaml.safe_dump(config), encoding="utf-8")
    return SimpleNamespace(stage=root)


def _template_dir(root: Path, relpath: str) -> None:
    directory = root / relpath
    directory.mkdir(parents=True)
    (directory / "docker-compose.yml.j2").write_text("services: {}\n", encoding="utf-8")


@pytest.fixture
def no_render(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fail the test if the compose render is reached."""

    def _refuse(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("the compose render ran before the service check refused")

    monkeypatch.setattr(compose_generator, "prepare_compose_files", _refuse)


@pytest.mark.usefixtures("no_render")
def test_a_deployed_service_nobody_declares_is_refused_before_the_render(tmp_path: Path) -> None:
    zones = _stage(
        tmp_path,
        {
            "deployed_services": ["telemetry_store", "openobserve"],
            "services": {"telemetry_store": {"path": "./services/telemetry_store"}},
        },
    )

    with pytest.raises(BuildProfileError) as caught:
        _render_compose_files(zones)

    message = str(caught.value)
    assert "deployed_services name(s) the service 'openobserve'" in message
    assert "telemetry_store" in message  # the bundled set is listed
    assert "services.openobserve.path" in message


@pytest.mark.usefixtures("no_render")
def test_a_services_block_for_no_bundled_or_pathed_service_is_refused(tmp_path: Path) -> None:
    zones = _stage(tmp_path, {"services": {"openobserve": {"retention_days": 30}}})

    with pytest.raises(BuildProfileError) as caught:
        _render_compose_files(zones)

    message = str(caught.value)
    assert "services.openobserve name(s) the service 'openobserve'" in message
    assert "it states no path" in message


def test_a_custom_service_path_without_a_template_directory_is_refused(tmp_path: Path) -> None:
    errors = _service_existence_errors(
        {
            "deployed_services": ["openobserve"],
            "services": {"openobserve": {"path": "./services/openobserve"}},
        },
        tmp_path,
    )

    assert len(errors) == 1
    assert "deployed_services and services.openobserve" in errors[0]
    assert "its path './services/openobserve' holds no docker-compose.yml.j2" in errors[0]


def test_a_custom_service_with_a_template_directory_passes(tmp_path: Path) -> None:
    _template_dir(tmp_path, "services/my_collector")

    assert (
        _service_existence_errors(
            {
                "deployed_services": ["my_collector"],
                "services": {"my_collector": {"path": "./services/my_collector"}},
            },
            tmp_path,
        )
        == []
    )


def test_a_close_misspelling_names_the_bundled_service(tmp_path: Path) -> None:
    errors = _service_existence_errors({"deployed_services": ["telemetry_stor"]}, tmp_path)

    assert "(did you mean 'telemetry_store'?)" in errors[0]


def test_an_undeployed_block_of_a_bundled_service_passes(tmp_path: Path) -> None:
    config = {
        "deployed_services": [],
        "services": {"graphdb": {"uri": "http://graphdb:7200", "port_host": 7200}},
    }

    assert _service_existence_errors(config, tmp_path) == []


def test_every_bundled_service_passes_by_name(tmp_path: Path) -> None:
    bundled = _bundled_service_names()
    # live_standin is a second instance of the bundled virtual accelerator.
    assert {"telemetry_store", "telemetry_gateway", "postgresql", "live_standin"} <= bundled
    assert "openobserve" not in bundled

    config = {
        "deployed_services": sorted(bundled),
        "services": {name: {"path": f"./services/{name}"} for name in bundled},
    }

    assert _service_existence_errors(config, tmp_path) == []


def test_a_deployed_bundled_service_without_a_block_is_refused(tmp_path: Path) -> None:
    errors = _service_existence_errors({"deployed_services": ["postgresql"]}, tmp_path)

    assert errors and "no services.postgresql block" in errors[0]


def test_the_refusal_is_an_ordinary_build_error_not_an_unexpected_one(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    runner = CliRunner()
    repo = tmp_path / "probe"
    created = runner.invoke(
        init,
        [
            str(repo),
            "--preset",
            "hello-world",
            "--no-git",
            "--set",
            "config.deployed_services=[telemetry_store,telemetry_gateway,openobserve]",
        ],
    )
    assert created.exit_code == 0, created.output

    with caplog.at_level(logging.ERROR):
        result = runner.invoke(build, ["--repo", str(repo), "--skip-deps", "--skip-lifecycle"])

    assert result.exit_code != 0
    text = caplog.text
    assert "Build error" in text
    assert "'openobserve'" in text
    assert "Unexpected error" not in text
    assert "not found in configuration" not in text
