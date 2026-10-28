"""A machine reached only through a write gateway draws a build warning.

The write switch has two layers: the software refusal (reference monitor and
hooks) and the network one (a read-only gateway the run reads through while
writes are off). A connector block that configures a ``write_access`` gateway
and no ``read_only`` one has the first layer only, so the build says so, and
says how to add the second. The warning rides the deploy-aware warnings path,
so ``osprey build`` and ``osprey profile validate`` print the same one.
"""

from __future__ import annotations

from typing import Any

from osprey.cli.build_profile_deploy import (
    deploy_aware_config_warnings,
    write_gateway_only_warnings,
)

WRITE_GATEWAY = {"address": "cs-gw-write.example.org", "port": 5084}
READ_GATEWAY = {"address": "cs-gw-read.example.org", "port": 5064}


def _config(connector: dict[str, Any]) -> dict[str, Any]:
    return {"control_system": {"type": "epics", "connector": connector}}


def _gateway_warnings(warnings: list[str]) -> list[str]:
    return [w for w in warnings if "gateways.read_only" in w]


def test_write_gateway_only_machine_warns_through_the_build_path() -> None:
    config = _config({"epics": {"gateways": {"write_access": WRITE_GATEWAY}}})

    (warning,) = _gateway_warnings(deploy_aware_config_warnings(None, config))

    assert "control_system.connector.epics" in warning
    assert "network" in warning
    assert "control_system.connector.epics.gateways.read_only" in warning


def test_dotted_spelling_warns_the_same() -> None:
    config = {
        "control_system.type": "epics",
        "control_system.connector.epics.gateways.write_access.address": "gw.example.org",
    }

    (warning,) = _gateway_warnings(deploy_aware_config_warnings(None, config))

    assert "control_system.connector.epics" in warning


def test_both_gateways_draw_no_warning() -> None:
    config = _config(
        {"epics": {"gateways": {"write_access": WRITE_GATEWAY, "read_only": READ_GATEWAY}}}
    )

    assert _gateway_warnings(deploy_aware_config_warnings(None, config)) == []
    assert write_gateway_only_warnings(config) == []


def test_read_only_gateway_alone_draws_no_warning() -> None:
    config = _config({"epics": {"gateways": {"read_only": READ_GATEWAY}}})

    assert write_gateway_only_warnings(config) == []


def test_unconfigured_type_draws_no_warning() -> None:
    # A machine type the profile never configures is not judged: the check walks
    # the connector blocks the config spells, not every control target there is.
    assert write_gateway_only_warnings({"control_system": {"type": "mock"}}) == []
    assert write_gateway_only_warnings({}) == []
    assert write_gateway_only_warnings(_config({"mock": {}})) == []


def test_each_write_only_block_warns_once_and_names_its_machine() -> None:
    config = _config(
        {
            "epics": {"gateways": {"write_access": WRITE_GATEWAY}},
            "virtual_accelerator": {"gateways": {"write_access": WRITE_GATEWAY}},
            "tango": {"gateways": {"write_access": WRITE_GATEWAY, "read_only": READ_GATEWAY}},
        }
    )

    warnings = write_gateway_only_warnings(config)

    assert len(warnings) == 2
    assert "control_system.connector.epics.gateways.read_only" in warnings[0]
    assert "control_system.connector.virtual_accelerator.gateways.read_only" in warnings[1]


def test_empty_write_gateway_is_not_a_write_gateway() -> None:
    config = _config({"epics": {"gateways": {"write_access": {}}}})

    assert write_gateway_only_warnings(config) == []
