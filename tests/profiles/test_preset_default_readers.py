"""Every shipped root preset keeps its agent record where it says, read by whom it says.

hello-world keeps the record in the embedded backend and names no reader: it
has no roster and no history page. control-assistant keeps it in ClickHouse for
90 days and names one reader, the deployment's admin roster user, so a fresh
deployment has someone who may read every session, and nobody else. Neither
states a key of the telemetry store the record replaced.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

from osprey.agent_record.config import (
    BACKEND_CLICKHOUSE,
    BACKEND_EMBEDDED,
    RETIRED_CONFIG_KEYS,
    RETIRED_DEPLOYED_SERVICES,
    history_page_deployed,
    parse_agent_record_block,
    reader_errors,
)
from osprey.cli.build_profile_archiver import _expand_dotted

PRESET_DIR = Path(__file__).resolve().parents[2] / "src" / "osprey" / "profiles" / "presets"

#: The documents an operator starts from; bundled tiers and apps are laid over them.
ROOT_PRESETS = ("control-assistant", "hello-world")

#: Each root preset's backend and retention.
EXPECTED_RECORD = {
    "control-assistant": (BACKEND_CLICKHOUSE, 90),
    "hello-world": (BACKEND_EMBEDDED, 14),
}

#: The roster tier whose user is the default reader.
ADMIN_TIER = "admin"

#: Config prefixes the record replaced — the telemetry store, its feeds and
#: the rest — which no preset states.
RETIRED_PREFIXES = tuple(RETIRED_CONFIG_KEYS)


def _document(preset: str) -> dict[str, Any]:
    return yaml.safe_load((PRESET_DIR / f"{preset}.yml").read_text(encoding="utf-8"))


def _config(preset: str) -> dict[str, Any]:
    return _expand_dotted(_document(preset).get("config"))


def _roster(config: dict[str, Any]) -> list[dict[str, Any]]:
    users = config.get("modules", {}).get("web_terminals", {}).get("users")
    return [user for user in users or [] if isinstance(user, dict)]


def _record(preset: str):
    return parse_agent_record_block(_document(preset))


@pytest.mark.parametrize("preset", ROOT_PRESETS)
def test_a_root_preset_states_its_record_block(preset: str) -> None:
    assert isinstance(_document(preset).get("agent_record"), dict)
    record = _record(preset)
    assert (record.backend, record.retention_days) == EXPECTED_RECORD[preset]


@pytest.mark.parametrize("preset", ROOT_PRESETS)
def test_a_root_preset_names_exactly_its_default_reader(preset: str) -> None:
    config = _config(preset)
    readers = [dict(entry) for entry in _record(preset).readers]
    roster = _roster(config)
    if not roster:
        assert readers == []
        return
    admins = [user["name"] for user in roster if user.get("tier") == ADMIN_TIER]
    assert len(admins) == 1, f"expected one {ADMIN_TIER} roster user, got {admins}"
    assert readers == [{"name": admins[0]}]


@pytest.mark.parametrize("preset", ROOT_PRESETS)
def test_the_default_reader_passes_the_reader_checks(preset: str) -> None:
    assert reader_errors(_record(preset), _config(preset)) == []


def test_only_control_assistant_serves_the_history_page() -> None:
    serving = {preset for preset in ROOT_PRESETS if history_page_deployed(_config(preset))}
    assert serving == {"control-assistant"}


@pytest.mark.parametrize("preset", ROOT_PRESETS)
def test_no_root_preset_states_a_telemetry_store_key(preset: str) -> None:
    def paths(node: Any, prefix: str = "") -> list[str]:
        if not isinstance(node, dict):
            return [prefix]
        found = [prefix] if prefix else []
        for key, value in node.items():
            found += paths(value, f"{prefix}.{key}" if prefix else str(key))
        return found

    config = _config(preset)
    stated = [
        path
        for path in paths(config)
        if any(path == p or path.startswith(f"{p}.") for p in RETIRED_PREFIXES)
    ]
    assert stated == []
    services = config.get("deployed_services") or []
    assert RETIRED_DEPLOYED_SERVICES.isdisjoint(services)
