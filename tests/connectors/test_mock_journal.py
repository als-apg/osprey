"""The mock connector's session-writes journal.

Only a write the mock took is journalled, under the simulation state
directory; every connector on the same view replays what another wrote, and a
reset or a change of the active scenarios empties the journal. An entry that is
not a well-formed write of a writable setpoint inside its band, under the
current active set, keeps the whole journal from being replayed.
"""

from __future__ import annotations

import ast
import json
import logging
import os
import re
import subprocess
import sys
import textwrap
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from osprey.connectors.control_system.base import WriteOutcome
from osprey.connectors.control_system.mock_connector import MockConnector, simulation_state_dir
from tests.facility.served_tree import MOCK_RENDER_CONFIG, mock_config, served_tree

REPO = Path(__file__).resolve().parents[2]
JOURNAL = ("mock", "writes.json")


def _writes_enabled(key, default=None):
    if key == "control_system.writes_enabled":
        return True
    return default


@pytest.fixture(autouse=True)
def writes_enabled(monkeypatch):
    monkeypatch.setattr("osprey.utils.config.get_config_value", _writes_enabled)


def _journal(view: Path) -> Path:
    return simulation_state_dir(view).joinpath(*JOURNAL)


def _rebuild(view: Path) -> Path:
    """Build and render again the facility tree the view was rendered from."""
    from osprey.facility.build import build_facility
    from osprey.facility.render import render_facility_outputs

    root = view.parent.parent.parent
    facility_dir = root / "data" / "facility"
    render_dir = root / "build"
    document = build_facility(facility_dir, project_name="served")
    render_facility_outputs(render_dir, document, MOCK_RENDER_CONFIG, facility_dir)
    return render_dir / "data" / "simulator"


async def _connected(view: Path) -> MockConnector:
    connector = MockConnector()
    await connector.connect(mock_config(view, response_delay_ms=0))
    return connector


@pytest.fixture
def view(tmp_path):
    return served_tree(tmp_path, {"A:SP": "A:RB", "B:SP": None})


async def test_a_write_the_mock_took_is_journalled(view):
    connector = await _connected(view)

    result = await connector.write_channel("A:SP", 4.5)

    document = json.loads(_journal(view).read_text())
    assert result.outcome is WriteOutcome.CONFIRMED
    assert document["seq"] == 1
    assert document["writes"] == [[1, "A:SP", 4.5]]
    assert len(document["active_set_sha256"]) == 64
    await connector.disconnect()


async def test_a_refused_write_is_not_journalled(view, monkeypatch):
    connector = await _connected(view)

    readback = await connector.write_channel("A:RB", 1.0)

    def rejecting_set(_values):
        raise ValueError("refused by the engine")

    monkeypatch.setattr(connector._composite, "set", rejecting_set)
    rejected = await connector.write_channel("A:SP", 1.0)

    assert readback.outcome is WriteOutcome.REFUSED
    assert rejected.outcome is WriteOutcome.REFUSED
    assert not _journal(view).exists()
    await connector.disconnect()


async def test_a_second_connector_reads_what_the_first_wrote(view):
    first = await _connected(view)
    second = await _connected(view)

    await first.write_channel("A:SP", 7.0)

    assert (await second.read_channel("A:SP")).value == 7.0
    assert (await second.read_channel("A:RB")).value == 7.0
    await first.disconnect()
    await second.disconnect()


async def test_a_reset_in_one_is_seen_by_the_other(view):
    first = await _connected(view)
    second = await _connected(view)
    await first.write_channel("A:SP", 7.0)
    assert (await second.read_channel("A:SP")).value == 7.0

    await first.reset()

    assert (await second.read_channel("A:SP")).value == 0.0
    assert json.loads(_journal(view).read_text())["writes"] == []
    await first.disconnect()
    await second.disconnect()


async def test_a_scenario_change_empties_the_journal(view):
    root = view.parent.parent.parent
    (root / "data" / "facility" / "scenarios").mkdir()
    (root / "data" / "facility" / "scenarios" / "quiet.yaml").write_text(
        "description: A quiet machine.\n"
    )
    view = _rebuild(view)
    connector = await _connected(view)
    await connector.write_channel("A:SP", 7.0)

    state = simulation_state_dir(view)
    (state / "active_scenarios").write_text("quiet\n")

    assert (await connector.read_channel("A:SP")).value == 0.0
    assert json.loads(_journal(view).read_text())["writes"] == []
    await connector.disconnect()


async def test_a_second_connector_keeps_what_the_first_wrote_after_a_scenario_change(view):
    root = view.parent.parent.parent
    (root / "data" / "facility" / "scenarios").mkdir()
    (root / "data" / "facility" / "scenarios" / "quiet.yaml").write_text(
        "description: A quiet machine.\n"
    )
    view = _rebuild(view)
    first = await _connected(view)
    second = await _connected(view)

    state = simulation_state_dir(view)
    state.mkdir(parents=True, exist_ok=True)
    (state / "active_scenarios").write_text("quiet\n")

    assert (await first.read_channel("A:SP")).value == 0.0
    await first.write_channel("A:SP", 7.0)
    assert (await second.read_channel("A:SP")).value == 7.0
    assert (await first.read_channel("A:SP")).value == 7.0
    writes = json.loads(_journal(view).read_text())["writes"]
    assert [entry[1:] for entry in writes] == [["A:SP", 7.0]]
    await first.disconnect()
    await second.disconnect()


async def test_a_rewrite_of_the_same_active_set_keeps_the_session_writes(view):
    connector = await _connected(view)
    await connector.write_channel("A:SP", 7.0)

    state = simulation_state_dir(view) / "active_scenarios"
    state.parent.mkdir(parents=True, exist_ok=True)
    state.write_text("")
    later = state.stat().st_mtime + 5
    os.utime(state, (later, later))

    assert (await connector.read_channel("A:SP")).value == 7.0
    other = await _connected(view)
    assert (await other.read_channel("A:SP")).value == 7.0
    assert json.loads(_journal(view).read_text())["writes"] == [[1, "A:SP", 7.0]]
    await connector.disconnect()
    await other.disconnect()


async def test_a_journal_that_cannot_be_written_leaves_a_write_result(view, monkeypatch, caplog):
    connector = await _connected(view)

    def unwritable(_document):
        raise OSError("read-only file system")

    monkeypatch.setattr(connector._journal, "write", unwritable)
    with caplog.at_level(logging.WARNING):
        result = await connector.write_channel("A:SP", 2.0)

    assert result.outcome is WriteOutcome.CONFIRMED
    assert "writes journal not updated: read-only file system" in caplog.text
    await connector.disconnect()


def _subprocess(script: str, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(script), *args],
        capture_output=True,
        text=True,
        cwd=REPO,
    )


_WRITER = """
    import asyncio, sys
    import osprey_connectors.config as config

    real = config.get_config_value
    config.get_config_value = lambda key, default=None, config_path=None: (
        True if key == "control_system.writes_enabled" else real(key, default, config_path)
    )
    from osprey_connectors.control_system.mock_connector import MockConnector

    async def main():
        connector = MockConnector()
        await connector.connect({"simulator_view": sys.argv[1], "response_delay_ms": 0})
        for value in range(50):
            result = await connector.write_channel(sys.argv[2], float(value), confirm=False)
            assert result.outcome.value == "unrequested", result
            await asyncio.sleep(0)
        await connector.disconnect()

    asyncio.run(main())
"""


def test_two_processes_writing_interleaved_values_lose_none(view):
    writers = [
        subprocess.Popen(
            [sys.executable, "-c", textwrap.dedent(_WRITER), str(view), address],
            cwd=REPO,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        for address in ("A:SP", "B:SP")
    ]
    for writer in writers:
        _out, err = writer.communicate(timeout=300)
        assert writer.returncode == 0, err

    document = json.loads(_journal(view).read_text())
    assert document["seq"] == 100
    assert [entry[0] for entry in document["writes"]] == list(range(1, 101))
    for address in ("A:SP", "B:SP"):
        assert [entry[2] for entry in document["writes"] if entry[1] == address] == [
            float(value) for value in range(50)
        ]


async def test_a_composite_written_directly_changes_nothing_another_process_reads(view):
    direct = _subprocess(
        """
        import asyncio, sys
        from osprey_connectors.control_system.mock_connector import MockConnector

        async def main():
            connector = MockConnector()
            await connector.connect({"simulator_view": sys.argv[1], "response_delay_ms": 0})
            connector._composite.set({"A:SP": 9.0})
            await connector.disconnect()

        asyncio.run(main())
        """,
        str(view),
    )
    assert direct.returncode == 0, direct.stderr

    connector = await _connected(view)
    assert (await connector.read_channel("A:SP")).value == 0.0
    await connector.disconnect()


def _limited(view: Path) -> Path:
    """The tree again with a locked setpoint and a band on another."""
    root = view.parent.parent.parent
    (root / "data" / "facility" / "limits.yaml").write_text(
        yaml.safe_dump(
            {
                "records": [
                    {"address": "A:SP", "min_value": 0.0, "max_value": 10.0},
                    {"address": "B:SP", "writable": False},
                ]
            }
        )
    )
    return _rebuild(view)


def _plant(view: Path, writes: list[list], *, sha: str | None = None) -> None:
    connector_view = view
    path = _journal(connector_view)
    path.parent.mkdir(parents=True, exist_ok=True)
    document = {"active_set_sha256": sha, "seq": len(writes), "writes": writes}
    path.write_text(json.dumps(document))


async def _active_sha(view: Path) -> str:
    connector = await _connected(view)
    await connector.write_channel("A:SP", 1.0)
    await connector.reset()
    sha = json.loads(_journal(view).read_text())["active_set_sha256"]
    await connector.disconnect()
    return sha


@pytest.mark.parametrize(
    ("entry", "reason"),
    [
        (["A:RB", 3.0], "A:RB: not a setpoint"),
        (["B:SP", 3.0], "B:SP: not a writable setpoint"),
        (["A:SP", 30.0], "A:SP: 30.0 is outside [0.0, 10.0]"),
        (["C:SP", 3.0], "C:SP: not in build/facility.json"),
    ],
)
async def test_a_forged_entry_is_not_replayed_and_is_logged(view, caplog, entry, reason):
    view = _limited(view)
    sha = await _active_sha(view)
    _plant(view, [[1, "A:SP", 5.0], [2, *entry]], sha=sha)

    with caplog.at_level(logging.WARNING):
        connector = await _connected(view)
        value = (await connector.read_channel("A:SP")).value

    assert value == 0.0
    assert f"writes journal rejected: {reason}" in caplog.text
    await connector.disconnect()


async def test_a_rejected_journal_is_emptied_so_later_writes_carry_across(view, caplog):
    view = _limited(view)
    sha = await _active_sha(view)
    _plant(view, [[1, "A:SP", 30.0]], sha=sha)

    with caplog.at_level(logging.WARNING):
        first = await _connected(view)
        held = (await first.read_channel("A:SP")).value
        await first.read_channel("A:SP")
        second = await _connected(view)
        await second.read_channel("A:SP")

    assert held == 0.0
    assert caplog.text.count("writes journal rejected") == 1
    assert json.loads(_journal(view).read_text())["writes"] == []
    await first.write_channel("A:SP", 4.0)
    fresh = await _connected(view)
    assert (await fresh.read_channel("A:SP")).value == 4.0
    for connector in (first, second, fresh):
        await connector.disconnect()


async def test_an_entry_from_another_active_set_is_not_replayed(view, caplog):
    _plant(view, [[1, "A:SP", 5.0]], sha="0" * 64)

    with caplog.at_level(logging.WARNING):
        connector = await _connected(view)
        value = (await connector.read_channel("A:SP")).value

    assert value == 0.0
    assert "writes journal rejected: A:SP: written under another active scenario set" in (
        caplog.text
    )
    await connector.disconnect()


async def test_an_entry_whose_seq_is_a_bool_is_not_replayed(view, caplog):
    sha = await _active_sha(view)
    _plant(view, [[True, "A:SP", 5.0]], sha=sha)

    with caplog.at_level(logging.WARNING):
        connector = await _connected(view)
        value = (await connector.read_channel("A:SP")).value

    assert value == 0.0
    assert "writes journal rejected: writes.json: malformed entry [True, 'A:SP', 5.0]" in (
        caplog.text
    )
    await connector.disconnect()


def _with_physics_model(view: Path, name: str) -> Path:
    """The view again with one served physics model that owns no channel."""
    variables = json.loads((view / "variables.json").read_text())
    variables["models"].append(
        {
            "name": name,
            "engine": "journal-stub",
            "served": True,
            "settings": {},
            "deck": None,
            "wiring": [],
        }
    )
    (view / "variables.json").write_text(json.dumps(variables))
    served = json.loads((view / "served_models.json").read_text())
    served["models"] = sorted([*served["models"], name])
    (view / "served_models.json").write_text(json.dumps(served))
    return view


async def test_a_rejected_journal_is_appended_to_every_physics_models_log(
    view, caplog, monkeypatch, tmp_path
):
    from osprey_connectors.simulation import composite as composite_module
    from osprey_connectors.simulation.composite import Composite

    stub = SimpleNamespace(build=lambda *args, **kwargs: object())
    real = Composite._engine
    monkeypatch.setattr(
        Composite,
        "_engine",
        staticmethod(lambda name: stub if name == "journal-stub" else real(name)),
    )
    logs = tmp_path / "model-logs"
    monkeypatch.setattr(composite_module, "log_dir", lambda: logs)
    view = _with_physics_model(view, "M")
    _plant(view, [[1, "A:SP", 5.0]], sha="0" * 64)

    with caplog.at_level(logging.WARNING):
        connector = await _connected(view)
        await connector.read_channel("A:SP")

    records = [json.loads(line) for line in (logs / "M.log").read_text().splitlines()]
    rejected = [record for record in records if record["event"] == "journal-rejected"]
    assert rejected == [
        {
            "address": "A:SP",
            "event": "journal-rejected",
            "instance": "inprocess",
            "model": "M",
            "pid": os.getpid(),
            "reason": "written under another active scenario set",
        }
    ]
    assert "writes journal rejected: A:SP: written under another active scenario set" in (
        caplog.text
    )
    await connector.disconnect()


async def test_a_rejected_journal_without_a_physics_model_logs_the_process_line_only(
    view, caplog, monkeypatch, tmp_path
):
    from osprey_connectors.simulation import composite as composite_module

    logs = tmp_path / "model-logs"
    monkeypatch.setattr(composite_module, "log_dir", lambda: logs)
    _plant(view, [[1, "A:SP", 5.0]], sha="0" * 64)

    with caplog.at_level(logging.WARNING):
        connector = await _connected(view)
        await connector.read_channel("A:SP")

    assert "writes journal rejected: A:SP: written under another active scenario set" in (
        caplog.text
    )
    assert not logs.exists()
    await connector.disconnect()


async def test_a_well_formed_entry_is_replayed(view):
    view = _limited(view)
    sha = await _active_sha(view)
    _plant(view, [[1, "A:SP", 5.0]], sha=sha)

    connector = await _connected(view)

    assert (await connector.read_channel("A:SP")).value == 5.0
    await connector.disconnect()


def test_the_journal_is_named_in_the_mock_connector_only():
    """No other module under src/ or packages/ reads or writes the journal."""
    token = re.compile(r"writes\.json(?![\w])")
    hits = sorted(
        str(path.relative_to(REPO))
        for root in ("src", "packages")
        for path in (REPO / root).rglob("*.py")
        if token.search(path.read_text(encoding="utf-8"))
    )
    assert hits == [
        "packages/osprey-connectors/src/osprey_connectors/control_system/mock_connector.py"
    ]


def test_no_other_simulation_module_constructs_a_journal_reader():
    """The composite, the archive composite and the VA never reach the journal."""
    simulation = REPO / "packages/osprey-connectors/src/osprey_connectors/simulation"
    for path in sorted(simulation.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        names = {
            node.id if isinstance(node, ast.Name) else node.attr
            for node in ast.walk(tree)
            if isinstance(node, ast.Name | ast.Attribute)
        }
        assert "MockConnector" not in names, path
        assert "mock_connector" not in path.read_text(encoding="utf-8"), path
