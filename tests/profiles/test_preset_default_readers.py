"""Every shipped preset that deploys the telemetry store names one default reader.

The reader is the deployment's admin roster user where the preset has a roster,
and ``operator`` where it has none, so a fresh deployment always has someone
who may query the store, and nobody else.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

from osprey.cli.build_profile_archiver import _expand_dotted
from osprey.deployment import telemetry_config as tc
from osprey.deployment.telemetry_lint import lint_telemetry

PRESET_DIR = Path(__file__).resolve().parents[2] / "src" / "osprey" / "profiles" / "presets"

#: The documents an operator starts from; persona presets extend control-assistant.
ROOT_PRESETS = ("control-assistant", "hello-world", "ariel-standalone", "channel-finder-standalone")

#: The reader a roster-less preset declares.
ROSTERLESS_READER = "operator"
#: The roster persona whose user is the default reader.
ADMIN_PERSONA = "admin"


def _config(preset: str) -> dict[str, Any]:
    document = yaml.safe_load((PRESET_DIR / f"{preset}.yml").read_text())
    return _expand_dotted(document.get("config"))


def _roster(config: dict[str, Any]) -> list[dict[str, Any]]:
    users = config.get("modules", {}).get("web_terminals", {}).get("users")
    return [user for user in users or [] if isinstance(user, dict)]


def _expected_reader(config: dict[str, Any]) -> str:
    roster = _roster(config)
    if not roster:
        return ROSTERLESS_READER
    admins = [user["name"] for user in roster if user.get("persona") == ADMIN_PERSONA]
    assert len(admins) == 1, f"expected one {ADMIN_PERSONA} roster user, got {admins}"
    return admins[0]


def test_the_store_presets_are_the_ones_named() -> None:
    deploying = {preset for preset in ROOT_PRESETS if tc.store_deployed(_config(preset))}
    assert deploying == {"control-assistant", "hello-world", "ariel-standalone"}


@pytest.mark.parametrize("preset", ROOT_PRESETS)
def test_a_store_preset_names_exactly_its_default_reader(preset: str) -> None:
    config = _config(preset)
    if not tc.store_deployed(config):
        assert tc.readers(config) == []
        return
    assert tc.readers(config) == [{"name": _expected_reader(config)}]


@pytest.mark.parametrize("preset", ROOT_PRESETS)
def test_the_default_reader_passes_the_reader_lint(preset: str) -> None:
    config = _config(preset)
    if not tc.store_deployed(config):
        pytest.skip("deploys no store")
    errors = [
        finding
        for finding in lint_telemetry(config)
        if finding.severity == "error" and "reader" in finding.code
    ]
    assert errors == []
