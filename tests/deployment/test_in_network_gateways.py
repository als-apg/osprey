"""A bridged container that drives the target names the deployed machines by service name.

The rendered config addresses the simulator and the stand-in as ``localhost``,
which is right on the host and in the host-network web terminals. A container
on the compose network — the optimizer, a projected application — sees its own
container at ``localhost``, so its staged copy names each deployed machine's
service instead.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

from osprey.deployment.compose_generator import in_network_gateways, setup_build_dir


def _config(address: str = "localhost") -> dict[str, Any]:
    row = {"address": address, "use_name_server": True}
    return {
        "control_system": {
            "connector": {
                "virtual_accelerator": {
                    "serving": "served",
                    "gateways": {"read_only": dict(row), "write_access": dict(row)},
                    "pva_gateway": dict(row, port=5075),
                },
                "live_standin": {
                    "gateways": {"read_only": dict(row), "write_access": dict(row)},
                    "pva_gateway": dict(row, port=5075),
                },
            }
        }
    }


def _gateways(doc: dict[str, Any], block: str) -> set[str]:
    rows = doc["control_system"]["connector"][block]["gateways"].values()
    return {row["address"] for row in rows}


def _pva(doc: dict[str, Any], block: str) -> dict[str, Any]:
    return doc["control_system"]["connector"][block]["pva_gateway"]


def test_a_deployed_machines_loopback_gateways_name_its_service() -> None:
    out = in_network_gateways(_config(), ["virtual_accelerator", "live_standin"])
    assert _gateways(out, "virtual_accelerator") == {"virtual-accelerator"}
    assert _gateways(out, "live_standin") == {"live-standin"}
    assert _pva(out, "virtual_accelerator") == {
        "address": "virtual-accelerator",
        "use_name_server": True,
        "port": 5075,  # the port is kept
    }


def test_the_stand_ins_pva_gateway_names_the_simulator() -> None:
    """Only the simulator publishes pvAccess, so a loopback one is its server."""
    out = in_network_gateways(_config(), ["virtual_accelerator", "live_standin"])
    assert _pva(out, "live_standin")["address"] == "virtual-accelerator"


@pytest.mark.parametrize("address", ["127.0.0.2", "::1"])
def test_every_loopback_spelling_is_readdressed(address: str) -> None:
    out = in_network_gateways(_config(address), ["virtual_accelerator"])
    assert _gateways(out, "virtual_accelerator") == {"virtual-accelerator"}


def test_a_machine_not_deployed_is_left_alone() -> None:
    out = in_network_gateways(_config(), ["virtual_accelerator"])
    assert _gateways(out, "live_standin") == {"localhost"}
    assert _pva(out, "live_standin")["address"] == "localhost"


def test_an_operators_address_is_left_alone() -> None:
    out = in_network_gateways(_config("gateway.example.org"), ["virtual_accelerator"])
    assert _gateways(out, "virtual_accelerator") == {"gateway.example.org"}


def test_the_document_given_is_not_changed() -> None:
    doc = _config()
    in_network_gateways(doc, ["virtual_accelerator"])
    assert _gateways(doc, "virtual_accelerator") == {"localhost"}


def _stage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, services: dict[str, dict[str, Any]]
) -> dict[str, dict[str, Any]]:
    """Stage each of *services* (``services.<key>`` blocks) and return its staged config."""
    (tmp_path / "config.yml").write_text(yaml.safe_dump({"project_name": "p", **_config()}))
    for service in services:
        service_dir = tmp_path / "services" / service
        service_dir.mkdir(parents=True)
        (service_dir / "docker-compose.yml.j2").write_text(
            f"services:\n  {service}:\n    image: test\n"
        )
    monkeypatch.chdir(tmp_path)
    config = {
        "project_name": "p",
        "build_dir": "./build",
        "deployed_services": ["virtual_accelerator", "live_standin"],
        "services": services,
    }

    for service in services:
        setup_build_dir(
            str(Path("services") / service / "docker-compose.yml.j2"),
            config,
            {"copy_src": False},
            service_key=service,
        )

    staged = tmp_path / "build" / "services"
    return {s: yaml.safe_load((staged / s / "config.yml").read_text()) for s in services}


def test_setup_build_dir_readdresses_the_optimizers_staged_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    staged = _stage(tmp_path, monkeypatch, {"optimizer": {}})
    assert _gateways(staged["optimizer"], "virtual_accelerator") == {"virtual-accelerator"}


def test_an_application_container_on_the_bridge_names_the_machines(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A projected app that drives the target reaches the stand-in by service name."""
    staged = _stage(tmp_path, monkeypatch, {"trajectory": {"drives_target": True}})
    assert _gateways(staged["trajectory"], "live_standin") == {"live-standin"}
    assert _gateways(staged["trajectory"], "virtual_accelerator") == {"virtual-accelerator"}


def test_a_container_on_the_host_network_keeps_loopback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """On the host network ``localhost`` is the host, as in the web terminals."""
    staged = _stage(
        tmp_path, monkeypatch, {"trajectory": {"drives_target": True, "network": "host"}}
    )
    assert _gateways(staged["trajectory"], "live_standin") == {"localhost"}


def test_a_container_that_does_not_drive_the_target_is_left_alone(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    staged = _stage(tmp_path, monkeypatch, {"worker": {}})
    assert _gateways(staged["worker"], "virtual_accelerator") == {"localhost"}
