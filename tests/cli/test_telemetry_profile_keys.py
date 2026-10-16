"""``claude_code.telemetry`` is a closed block inside the otherwise open ``config:``.

Its readers are Python, so the closed set lives beside them as
:data:`~osprey.build.claude_code_telemetry.TELEMETRY_KEYS`, and the profile
parser refuses any other key over every spelling of the block — the nested
mapping, a partly nested one and dotted keys — in the ordinary unknown-key
wording. An old shape (``claude_code.telemetry.openobserve.*``) meets the same
refusal as any stray key: there is no retired-key list.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

from osprey.build.claude_code_telemetry import (
    TELEMETRY_AUTH_KEYS,
    TELEMETRY_KEYS,
    TelemetryConfigError,
    telemetry_backend,
)
from osprey.cli.build_profile import list_presets, resolve_build_profile
from osprey.cli.build_profile_load import (
    _parse_profile,
    manifest_rows,
    reject_unknown_closed_config_keys,
)
from osprey.errors import BuildProfileError

_FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "explicit_config"


def _parse(config: dict[str, Any]) -> None:
    _parse_profile({"name": "probe", "config": config})


def test_an_unknown_telemetry_key_is_refused_naming_the_closest_spelling() -> None:
    with pytest.raises(BuildProfileError) as caught:
        _parse({"claude_code.telemetry.endpont": "https://collector.example.org:4318"})

    message = str(caught.value)
    assert "Unknown claude_code.telemetry key(s): 'endpont'" in message
    assert "did you mean 'endpoint'?" in message
    # The valid set, in full, so the operator can see what the block takes.
    for key in TELEMETRY_KEYS:
        assert key in message


def test_an_unknown_telemetry_auth_key_is_refused() -> None:
    with pytest.raises(BuildProfileError) as caught:
        _parse({"claude_code.telemetry.auth.token": "COLLECTOR_TOKEN"})

    message = str(caught.value)
    assert "Unknown claude_code.telemetry.auth key(s): 'token'" in message
    assert "did you mean 'token_env'?" in message


@pytest.mark.parametrize(
    "config",
    [
        pytest.param(
            {"claude_code": {"telemetry": {"enabled": True, "openobserve": {"org": "x"}}}},
            id="nested",
        ),
        pytest.param(
            {"claude_code.telemetry": {"enabled": True, "openobserve": {"org": "x"}}},
            id="dotted-block",
        ),
        pytest.param(
            {"claude_code.telemetry.enabled": True, "claude_code.telemetry.openobserve.org": "x"},
            id="dotted-leaves",
        ),
        pytest.param(
            {"claude_code.telemetry.openobserve": {"user": "u", "password": "p"}},
            id="dotted-then-nested",
        ),
    ],
)
def test_the_nested_and_the_dotted_spelling_are_refused_alike(config: dict[str, Any]) -> None:
    with pytest.raises(BuildProfileError) as caught:
        _parse(config)

    assert "Unknown claude_code.telemetry key(s): 'openobserve'" in str(caught.value)


@pytest.mark.parametrize(
    "config",
    [
        pytest.param({"claude_code": {"telemetry": {"auth": {"header": "X"}}}}, id="nested"),
        pytest.param({"claude_code.telemetry.auth.header": "X"}, id="dotted"),
    ],
)
def test_the_auth_block_is_closed_in_either_spelling(config: dict[str, Any]) -> None:
    with pytest.raises(BuildProfileError, match=r"claude_code\.telemetry\.auth key\(s\): 'header'"):
        _parse(config)


def test_resource_attributes_stay_an_open_mapping() -> None:
    _parse(
        {
            "claude_code.telemetry.resource_attributes": {
                "deployment.environment": "staging",
                "facility": "probe",
            }
        }
    )


def test_every_manifest_row_under_the_block_is_a_known_key() -> None:
    rows = manifest_rows()
    head = "claude_code.telemetry."
    under = [key[len(head) :] for key in rows if key.startswith(head)]

    assert under, "the manifest carries no claude_code.telemetry rows"
    assert {key.split(".", 1)[0] for key in under} <= TELEMETRY_KEYS
    assert {key.split(".", 1)[1] for key in under if key.startswith("auth.")} <= (
        TELEMETRY_AUTH_KEYS
    )
    # The two keys read by code and never rendered by a preset have rows.
    assert {"endpoint", "resource_attributes"} <= set(under)


def _preset_names() -> list[str]:
    return [name if isinstance(name, str) else name.name for name in list_presets()]


def _frozen_fixtures() -> list[Path]:
    return sorted(_FIXTURES.glob("*/*/*.yml"))


def test_every_preset_and_frozen_fixture_parses_under_the_closed_blocks() -> None:
    presets = _preset_names()
    fixtures = _frozen_fixtures()
    assert presets and fixtures

    for name in presets:
        # Raises on any key a closed block does not take.
        resolve_build_profile(None, name)

    for path in fixtures:
        rendered = yaml.safe_load(path.read_text(encoding="utf-8"))
        reject_unknown_closed_config_keys(rendered)


def test_the_backend_refusal_names_gateway_and_generic_and_no_service() -> None:
    with pytest.raises(TelemetryConfigError) as caught:
        telemetry_backend({"backend": "openobserve"})

    message = str(caught.value)
    assert "'gateway'" in message
    assert "'generic'" in message
    # The remedy never sends the operator to deploy a service.
    assert "deployed_services" not in message
    assert "services." not in message
    assert "deploy " not in message
