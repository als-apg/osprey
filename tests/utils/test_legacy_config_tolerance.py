"""Legacy-tolerance coverage for the config loader.

Projects rendered against an earlier template carry configuration keys that
have no reader.  Deleting a key from the templates and from the
code that consumed it must never break those projects: their on-disk
``config.yml`` still declares the key, and ``load_osprey_config`` has to keep
accepting it silently — no exception, no ``WARNING``, no deprecation chatter.

``fixtures/legacy_config_all_deleted_keys.yml`` is the shared asset every
deletion task runs against.  It carries each retired dotted path at its most
misleading value, wrapped around a happy-path live-config core, so a failure
here distinguishes "the loader broke" from "the fixture is unrealistic".
"""

from __future__ import annotations

import logging
import shutil
import warnings
from pathlib import Path
from typing import Any

import pytest

from osprey.utils.workspace import load_osprey_config

FIXTURE = Path(__file__).parent / "fixtures" / "legacy_config_all_deleted_keys.yml"

# Every dotted path the fixture carries deliberately, because it is retired.
# Grouped by the ledger section that dispositions it.  Deleting an
# entry from this list — or from the fixture — silently drops the backwards
# compatibility coverage for that key, so both halves are asserted against each
# other by ``test_fixture_still_carries_every_retired_key``.
RETIRED_KEYS: dict[str, tuple[str, ...]] = {
    "P1 — safety surface": (
        "control_system.write_verification.enabled",
        "control_system.write_verification.fail_on_mismatch",
        "control_system.write_verification.timeout",
        "control_system.write_verification.default_level",
        "control_system.write_verification.default_tolerance_percent",
        "approval.tools.channel_limits",
    ),
    "P2 — dead-key deletions": (
        "channel_finder.pipelines.in_context.processing",
        "channel_finder.explicit_validation_mode",
        "channel_finder.pipelines.hierarchical.tree_preview",
        "channel_finder.benchmark.execution",
        "channel_finder.benchmark.output",
        "channel_finder.benchmark.evaluation",
        "machine_state",
        "file_paths.user_memory_dir",
        "file_paths.execution_plans_dir",
        "file_paths.prompts_dir",
        "workspace.base_dir",
        "control_system.connector.timeout",
        "control_system.connector.mock.simulate_delays",
        "api.providers.ollama.host",
        "api.providers.ollama.port",
        "ariel.reasoning",
        "ariel.default_max_results",
        "ariel.cache_embeddings",
        "applications",
    ),
    "P3 — consolidations": (
        "system.facility_name",
        "file_paths.agent_data_dir",
        "ariel.enhancement_modules.semantic_processor.model.provider",
    ),
    "P4 — derive, don't duplicate": (
        "ariel.database.connection_string",
        "ariel.database.uri",
        "control_system.connector.epics.gateways.read_only.port",
        "archiver.mock_archiver.simulation_file",
    ),
    "P5 — behavior honesty": ("logbook.composition.model_id",),
}


def _dotted(config: dict[str, Any], path: str) -> Any:
    """Resolve a dotted path, raising ``KeyError`` naming the full path if absent."""
    value: Any = config
    for key in path.split("."):
        if not isinstance(value, dict) or key not in value:
            raise KeyError(path)
        value = value[key]
    return value


@pytest.fixture
def legacy_config_path(tmp_path, monkeypatch) -> Path:
    """Copy the legacy fixture into an isolated project dir and point the loader at it."""
    project = tmp_path / "legacy-project"
    project.mkdir()
    config_path = project / "config.yml"
    shutil.copyfile(FIXTURE, config_path)

    monkeypatch.setenv("OSPREY_CONFIG", str(config_path))
    monkeypatch.chdir(project)
    return config_path


def test_load_zero_errors_zero_warnings(legacy_config_path, caplog):
    """A config full of retired keys loads cleanly and silently."""
    caplog.set_level(logging.DEBUG, logger="osprey")

    with warnings.catch_warnings(record=True) as python_warnings:
        warnings.simplefilter("always")
        config = load_osprey_config()

    # ``load_osprey_config`` swallows every exception and returns ``{}``, so
    # "no errors" has to be asserted as "the config actually arrived".
    assert config, "loader returned an empty config — the legacy file failed to parse"
    assert config["project_name"] == "legacy-tolerance-project"
    assert config["control_system"]["type"] == "mock"
    assert config["control_system"]["writes_enabled"] is False

    noisy = [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert not noisy, "loading a legacy config emitted: " + "; ".join(
        f"{r.levelname} {r.name}: {r.getMessage()}" for r in noisy
    )
    assert not python_warnings, "loading a legacy config emitted: " + "; ".join(
        f"{w.category.__name__}: {w.message}" for w in python_warnings
    )


def test_fixture_still_carries_every_retired_key(legacy_config_path):
    """The fixture is the coverage — a retired key vanishing from it is a regression."""
    config = load_osprey_config()

    missing = [
        f"{section}: {path}"
        for section, paths in RETIRED_KEYS.items()
        for path in paths
        if not _has(config, path)
    ]
    assert not missing, (
        "the legacy fixture no longer declares: "
        + "; ".join(missing)
        + " — restore them; deleting a key from the templates must not delete it here"
    )


def _has(config: dict[str, Any], path: str) -> bool:
    try:
        _dotted(config, path)
    except KeyError:
        return False
    return True


def test_retired_keys_keep_their_misleading_values(legacy_config_path):
    """The safety knobs are carried at the values that would be dangerous if honored."""
    config = load_osprey_config()

    verification = config["control_system"]["write_verification"]
    assert verification["enabled"] is True
    assert verification["fail_on_mismatch"] is True
    assert verification["default_level"] == "none"
    assert config["channel_finder"]["explicit_validation_mode"] == "strict"
    assert config["control_system"]["connector"]["mock"]["simulate_delays"] is True
