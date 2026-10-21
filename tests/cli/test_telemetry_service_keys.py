"""``services.telemetry_store`` and ``services.telemetry_gateway`` are closed blocks.

A bundled service's valid sub-keys are the manifest's rows under
``services.<name>`` (nested rows by prefix) plus the generic axes any service
block may carry (:data:`~osprey.cli.build_profile_model.SERVICE_AXES`). The two
telemetry services are the first closed this way; the manifest rows are pinned
to the keys :mod:`osprey.deployment.telemetry_config` reads.
"""

from __future__ import annotations

from typing import Any

import pytest

from osprey.cli.build_profile_load import _parse_profile, manifest_rows
from osprey.cli.build_profile_model import SERVICE_AXES
from osprey.deployment.telemetry_config import DEFAULT_LIMITS, GATEWAY_KEYS, STORE_KEYS
from osprey_connectors.errors import BuildProfileError


def _parse(config: dict[str, Any]) -> None:
    _parse_profile({"name": "probe", "config": config})


def _rows_under(prefix: str) -> set[str]:
    head = f"{prefix}."
    return {key[len(head) :].split(".", 1)[0] for key in manifest_rows() if key.startswith(head)}


def test_an_unknown_store_key_is_refused_with_the_valid_set() -> None:
    with pytest.raises(BuildProfileError) as caught:
        _parse({"services.telemetry_store.retention_dayz": 30})

    message = str(caught.value)
    assert "Unknown services.telemetry_store key(s): 'retention_dayz'" in message
    assert "did you mean 'retention_days'?" in message
    for key in STORE_KEYS | SERVICE_AXES:
        assert key in message


def test_an_unknown_store_limit_is_refused() -> None:
    with pytest.raises(BuildProfileError) as caught:
        _parse({"services": {"telemetry_store": {"limits": {"rows": 10}}}})

    message = str(caught.value)
    assert "Unknown services.telemetry_store.limits key(s): 'rows'" in message
    for name in DEFAULT_LIMITS:
        assert name in message


def test_an_unknown_gateway_key_is_refused() -> None:
    with pytest.raises(BuildProfileError) as caught:
        _parse({"services.telemetry_gateway": {"path": "./services/telemetry_gateway", "org": "x"}})

    assert "Unknown services.telemetry_gateway key(s): 'org'" in str(caught.value)


@pytest.mark.parametrize(
    ("axis", "value"),
    [
        ("path", "./services/telemetry_store"),
        ("image", "registry.example.org/clickhouse:pinned"),
        ("port", 18123),
        ("network", "host"),
        ("http", True),
        ("listens", True),
        ("bind_env", "STORE_BIND"),
        ("env", ["STORE_EXTRA"]),
    ],
)
def test_a_generic_axis_on_the_store_passes(axis: str, value: Any) -> None:
    assert axis in SERVICE_AXES
    _parse({f"services.telemetry_store.{axis}": value})
    _parse({"services": {"telemetry_gateway": {axis: value}}})


def test_a_custom_services_block_stays_open() -> None:
    _parse({"services.my_collector": {"path": "./services/my_collector", "anything": 1}})


def test_the_manifest_rows_are_the_keys_the_readers_take() -> None:
    assert _rows_under("services.telemetry_store") == STORE_KEYS
    assert _rows_under("services.telemetry_store.limits") == set(DEFAULT_LIMITS)
    assert _rows_under("services.telemetry_gateway") == GATEWAY_KEYS
