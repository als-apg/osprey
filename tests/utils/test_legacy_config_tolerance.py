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

One retired key is the exception, and it is deliberate: a project that still
declares ``control_system.write_verification.fail_on_mismatch: true`` believes
failed verification aborts its writes, which was never true.  That belief is
only dangerous when the project actually writes, so the ``WARNING`` fires from
the connector's write path — once per process — and *loading* the same config
stays as silent as every other retired key.  Both halves are asserted here.
"""

from __future__ import annotations

import logging
import shutil
import warnings
from pathlib import Path
from typing import Any

import pytest
import yaml

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
    assert config["channel_finder"]["explicit_validation_mode"] == "strict"
    assert config["control_system"]["connector"]["mock"]["simulate_delays"] is True


# --------------------------------------------------------------------------
# fail_on_mismatch — the one retired key that speaks, and only at write time
# --------------------------------------------------------------------------

WRITE_LOGGER = "osprey.connectors.control_system"


def _write_enabled_legacy_project(tmp_path, monkeypatch, **overrides: Any) -> Path:
    """Render the legacy fixture into a project whose writes are enabled.

    The shared fixture ships ``writes_enabled: false`` — with writes off, the
    connector refuses before it ever reads verification config, so the warning
    could never fire and a test built on it would pass vacuously.  Only the
    write gate is flipped here; every retired key is carried through unchanged.
    """
    config = yaml.safe_load(FIXTURE.read_text())
    config["control_system"]["writes_enabled"] = True
    config["control_system"]["write_verification"].update(overrides)

    project = tmp_path / "legacy-writing-project"
    project.mkdir()
    config_path = project / "config.yml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))

    monkeypatch.setenv("OSPREY_CONFIG", str(config_path))
    monkeypatch.chdir(project)
    return config_path


@pytest.fixture
def rearmed_warning(monkeypatch):
    """Re-arm the once-per-process guard, and un-arm it again after the test.

    ``monkeypatch`` restores the flag on teardown, so a test that trips the
    warning cannot silence it for whatever runs next in the same session.
    """
    from osprey.connectors.control_system import base

    monkeypatch.setattr(base, "_fail_on_mismatch_warned", False)


async def _write_twice(channel: str = "LEGACY:TEST:SP") -> list[Any]:
    """Connect a mock connector and perform two real writes through it."""
    from osprey.connectors.control_system.mock_connector import MockConnector

    connector = MockConnector()
    await connector.connect({"response_delay_ms": 0})
    try:
        return [
            await connector.write_channel(channel, 1.0),
            await connector.write_channel(channel, 2.0),
        ]
    finally:
        await connector.disconnect()


def _fail_on_mismatch_warnings(caplog) -> list[str]:
    return [
        r.getMessage()
        for r in caplog.records
        if r.levelno >= logging.WARNING and "fail_on_mismatch" in r.getMessage()
    ]


@pytest.mark.asyncio
async def test_two_writes_emit_exactly_one_warning(tmp_path, monkeypatch, caplog, rearmed_warning):
    """A legacy project is told once, at write time, that the key does nothing."""
    config_path = _write_enabled_legacy_project(tmp_path, monkeypatch)

    # ``load_osprey_config`` swallows every exception and returns ``{}``, so the
    # config the connector is about to read has to be asserted to have arrived.
    config = load_osprey_config()
    assert config, "loader returned an empty config — the legacy file failed to parse"
    assert config["control_system"]["writes_enabled"] is True
    assert config["control_system"]["write_verification"]["fail_on_mismatch"] is True

    with caplog.at_level(logging.WARNING, logger=WRITE_LOGGER):
        results = await _write_twice()

    # Both writes really happened: a refusal would never reach verification config.
    assert [r.success for r in results] == [True, True]
    assert not any(r.blocked for r in results)

    warnings_emitted = _fail_on_mismatch_warnings(caplog)
    assert len(warnings_emitted) == 1, (
        f"expected exactly one warning across two writes, got {len(warnings_emitted)}: "
        + "; ".join(warnings_emitted)
    )
    message = warnings_emitted[0]
    assert "control_system.write_verification.fail_on_mismatch" in message
    assert "write_channel_checked" in message, (
        "the warning must name the path that does enforce verification, "
        f"got: {message} (config: {config_path})"
    )


@pytest.mark.asyncio
async def test_loading_the_same_config_stays_silent(tmp_path, monkeypatch, caplog, rearmed_warning):
    """Loading is not writing — the warning belongs to the write path alone."""
    _write_enabled_legacy_project(tmp_path, monkeypatch)

    caplog.set_level(logging.DEBUG, logger="osprey")
    config = load_osprey_config()

    assert config["control_system"]["write_verification"]["fail_on_mismatch"] is True
    assert not _fail_on_mismatch_warnings(caplog)


@pytest.mark.asyncio
async def test_writes_stay_silent_without_fail_on_mismatch(
    tmp_path, monkeypatch, caplog, rearmed_warning
):
    """The other retired verification keys are inert at every value, so they say nothing."""
    _write_enabled_legacy_project(
        tmp_path, monkeypatch, fail_on_mismatch=False, enabled=True, timeout=5.0
    )

    with caplog.at_level(logging.WARNING, logger=WRITE_LOGGER):
        results = await _write_twice()

    assert [r.success for r in results] == [True, True]
    assert not _fail_on_mismatch_warnings(caplog)
