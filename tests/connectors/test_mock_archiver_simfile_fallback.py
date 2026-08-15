"""The mock archiver derives its simulation file from the control-system config.

The archiver has no machine model of its own — it synthesizes history from the
same file the live connector serves. Declaring the path under both
``control_system.connector.<type>`` and ``archiver.mock_archiver`` meant every
scenario change had to be made twice, and a project that updated only one of
them got archived history that contradicted live reads with nothing said about
it.

So ``archiver.mock_archiver.simulation_file`` is optional: unset, it is
resolved from the control-system side through the same
:func:`osprey.simulation.apply.resolve_simulation_file` the ``sim`` CLI uses
(type-aware, so a virtual-accelerator project resolves its own key). An explicit
archiver-side value still wins — pointing the archiver at a different model is
legitimate — but the divergence is logged at WARNING.

Both directions are covered here, plus the template, which ships no third copy
of the path.

A note on the virtual-accelerator cases below, which would otherwise read as
contradicting the honesty rule: pairing a ``virtual_accelerator`` control system
with the mock archiver is refused at build, at deploy and at MCP startup
(see :mod:`osprey.connectors.honesty`), so no deployment can reach that
configuration. These tests still exercise it, and legitimately — they construct
the connector directly, below the level where any of those three refusals live,
to pin the *resolution* rule: given a mock archiver and a VA-typed control
system, the simulation file resolves through the VA's own config key rather than
the mock's. That resolution has to stay correct because the type-aware lookup is
shared with the ``sim`` CLI and with projects whose control system is a VA while
some *other* archiver is selected. What is forbidden is deploying the pairing,
not computing what it would resolve to.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path

import pytest
import yaml

from osprey.connectors.archiver.mock_archiver_connector import MockArchiverConnector, _anchor

ARCHIVER_LOGGER = "mock_archiver_connector"

CONTROL_SYSTEM_VALUE = 42.0
ARCHIVER_ONLY_VALUE = 7.0

TEMPLATE = (
    Path(__file__).resolve().parents[2]
    / "src"
    / "osprey"
    / "templates"
    / "apps"
    / "control_assistant"
    / "config.yml.j2"
)

START = datetime(2024, 1, 1, 0, 0, 0)
END = datetime(2024, 1, 1, 1, 0, 0)


def _machine(value: float) -> dict:
    """A one-channel machine model whose constant value identifies the file."""
    return {
        "name": f"Rig-{value}",
        "description": "Single-channel machine used to tell two files apart",
        "channels": {
            "T:Q1:CUR:SP": {
                "value": value,
                "units": "A",
                "noise": 0.0,
                "description": "Test quad current setpoint",
            }
        },
        "scenarios": {"nominal": {"description": "All systems nominal."}},
    }


@pytest.fixture
def project(tmp_path, monkeypatch):
    """Build a project dir with a machine file and point the config loader at it.

    Returns a ``write_config(control_system, archiver)`` callable. The process
    never chdirs into the project, so a relative ``simulation_file`` only
    resolves if it is anchored at ``project_root``.
    """
    root = tmp_path / "sim-project"
    (root / "data" / "simulation").mkdir(parents=True)
    (root / "data" / "simulation" / "machine.json").write_text(
        json.dumps(_machine(CONTROL_SYSTEM_VALUE))
    )
    (root / "data" / "simulation" / "other-machine.json").write_text(
        json.dumps(_machine(ARCHIVER_ONLY_VALUE))
    )

    def write_config(control_system: dict, archiver: dict) -> Path:
        config_path = root / "config.yml"
        config_path.write_text(
            yaml.safe_dump(
                {
                    "project_name": "sim-project",
                    "project_root": str(root),
                    "control_system": control_system,
                    "archiver": archiver,
                }
            )
        )
        monkeypatch.setenv("CONFIG_FILE", str(config_path))
        return config_path

    write_config.root = root  # type: ignore[attr-defined]
    return write_config


async def _series(connector: MockArchiverConnector) -> list[float]:
    df = await connector.get_data(channels=["T:Q1:CUR:SP"], start_date=START, end_date=END)
    return df.loc[df["channel"] == "T:Q1:CUR:SP", "value"].tolist()


def _warnings(caplog) -> list[str]:
    """Warning-or-worse messages only — the loaders chat at INFO on every load."""
    return [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]


class TestAnchor:
    """``_anchor`` only rewrites a path that is both relative and has a root to hang on."""

    @pytest.mark.parametrize(
        ("path", "root", "expected"),
        [
            (
                Path("/elsewhere/machine.json"),
                Path("/project/root"),
                Path("/elsewhere/machine.json"),
            ),
            (Path("data/machine.json"), None, Path("data/machine.json")),
            (
                Path("data/machine.json"),
                Path("/project/root"),
                Path("/project/root/data/machine.json"),
            ),
        ],
    )
    def test_only_a_relative_path_with_a_root_is_rewritten(self, path, root, expected):
        assert _anchor(path, root) == expected


class TestDerivedFromControlSystem:
    @pytest.mark.asyncio
    async def test_unset_archiver_key_uses_the_control_system_file(self, project, caplog):
        """The whole point: no archiver-side key, and history still comes from the model."""
        project(
            control_system={
                "type": "mock",
                "connector": {"mock": {"simulation_file": "data/simulation/machine.json"}},
            },
            archiver={"type": "mock_archiver"},
        )

        connector = MockArchiverConnector()
        with caplog.at_level(logging.WARNING, logger=ARCHIVER_LOGGER):
            await connector.connect({})

        assert connector._sim_engine is not None, "no engine — the fallback did not fire"
        assert all(v == CONTROL_SYSTEM_VALUE for v in await _series(connector))
        assert not _warnings(caplog), "deriving the path is the normal case, not a warning"

        await connector.disconnect()

    @pytest.mark.asyncio
    async def test_relative_path_is_anchored_at_project_root(self, project):
        """The config's path is relative and the cwd is elsewhere — anchoring is the test."""
        write_config = project
        write_config(
            control_system={
                "type": "mock",
                "connector": {"mock": {"simulation_file": "data/simulation/machine.json"}},
            },
            archiver={"type": "mock_archiver"},
        )
        expected = write_config.root / "data" / "simulation" / "machine.json"
        assert Path.cwd() != write_config.root

        connector = MockArchiverConnector()
        await connector.connect({})

        assert connector._sim_engine is not None
        assert all(v == CONTROL_SYSTEM_VALUE for v in await _series(connector))
        assert expected.exists()

        await connector.disconnect()

    @pytest.mark.asyncio
    async def test_resolution_is_control_system_type_aware(self, project):
        """A virtual-accelerator project resolves its own key, not the mock block."""
        project(
            control_system={
                "type": "virtual_accelerator",
                "connector": {
                    "mock": {"simulation_file": "data/simulation/other-machine.json"},
                    "virtual_accelerator": {"simulation_file": "data/simulation/machine.json"},
                },
            },
            archiver={"type": "mock_archiver"},
        )

        connector = MockArchiverConnector()
        await connector.connect({})

        assert all(v == CONTROL_SYSTEM_VALUE for v in await _series(connector))

        await connector.disconnect()

    @pytest.mark.asyncio
    async def test_no_project_config_leaves_the_engine_unset(self, monkeypatch, tmp_path):
        """Unchanged behavior with nothing to derive from: procedural synthesis."""
        monkeypatch.delenv("CONFIG_FILE", raising=False)
        monkeypatch.chdir(tmp_path)  # no config.yml here

        connector = MockArchiverConnector()
        await connector.connect({})

        assert connector._sim_engine is None
        df = await connector.get_data(channels=["BEAM:CURRENT"], start_date=START, end_date=END)
        values = df.loc[df["channel"] == "BEAM:CURRENT", "value"]
        assert values.mean() == pytest.approx(500.0, rel=0.1)

        await connector.disconnect()


class TestExplicitArchiverValueWins:
    @pytest.mark.asyncio
    async def test_explicit_value_overrides_the_control_system_file(self, project, caplog):
        project(
            control_system={
                "type": "mock",
                "connector": {"mock": {"simulation_file": "data/simulation/machine.json"}},
            },
            archiver={
                "type": "mock_archiver",
                "mock_archiver": {"simulation_file": "data/simulation/other-machine.json"},
            },
        )

        connector = MockArchiverConnector()
        with caplog.at_level(logging.WARNING, logger=ARCHIVER_LOGGER):
            await connector.connect({"simulation_file": "data/simulation/other-machine.json"})

        assert all(v == ARCHIVER_ONLY_VALUE for v in await _series(connector))

        await connector.disconnect()

    @pytest.mark.asyncio
    async def test_divergence_is_logged(self, project, caplog):
        """Two different models is legal but load-bearing — it must not be silent."""
        write_config = project
        write_config(
            control_system={
                "type": "mock",
                "connector": {"mock": {"simulation_file": "data/simulation/machine.json"}},
            },
            archiver={
                "type": "mock_archiver",
                "mock_archiver": {"simulation_file": "data/simulation/other-machine.json"},
            },
        )

        connector = MockArchiverConnector()
        with caplog.at_level(logging.WARNING, logger=ARCHIVER_LOGGER):
            await connector.connect({"simulation_file": "data/simulation/other-machine.json"})

        warnings = _warnings(caplog)
        assert len(warnings) == 1, f"expected one divergence warning, got {warnings}"
        message = warnings[0]
        assert "archiver.mock_archiver.simulation_file" in message
        assert str(write_config.root / "data" / "simulation" / "other-machine.json") in message
        assert str(write_config.root / "data" / "simulation" / "machine.json") in message

        await connector.disconnect()

    @pytest.mark.asyncio
    async def test_matching_values_do_not_warn(self, project, caplog):
        """A project that declares the duplicate, matching, is honoured silently."""
        project(
            control_system={
                "type": "mock",
                "connector": {"mock": {"simulation_file": "data/simulation/machine.json"}},
            },
            archiver={
                "type": "mock_archiver",
                "mock_archiver": {"simulation_file": "data/simulation/machine.json"},
            },
        )

        connector = MockArchiverConnector()
        with caplog.at_level(logging.WARNING, logger=ARCHIVER_LOGGER):
            await connector.connect({"simulation_file": "data/simulation/machine.json"})

        assert not _warnings(caplog)
        assert all(v == CONTROL_SYSTEM_VALUE for v in await _series(connector))

        await connector.disconnect()


class TestTemplateDropsTheDuplicate:
    def test_archiver_block_declares_no_simulation_file(self):
        archiver_section = TEMPLATE.read_text().split("\narchiver:\n", 1)[1]

        live_keys = [
            line
            for line in archiver_section.splitlines()
            if line.strip().startswith("simulation_file:")
        ]
        assert not live_keys, (
            "the archiver block still declares simulation_file — it is derived from "
            f"the control-system config now: {live_keys}"
        )
        assert "control_system.connector.<type>.simulation_file" in archiver_section, (
            "the archiver block should say where its machine model comes from"
        )
