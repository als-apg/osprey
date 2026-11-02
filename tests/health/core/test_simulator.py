"""The core ``simulator`` health category.

A virtual accelerator served from a container keeps serving when one of its
physics models fails: the model's readbacks serve NaN, everything else serves
on, and the container's healthcheck stays green. So the category reads every
served physics model's status channel on each simulator the deployment serves
from a process of its own, and a failed model is an ``error`` row naming it and
its error. A model that names no deck and wires nothing is ``ok``, said to be
not built by design. No view, or a simulator served only in process, gives no
rows.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from osprey.health.core import get_core_category_factory
from osprey.health.models import Status
from tests.cli._simulator_view import add_deck_models, write_simulator_view

AR_ERROR = "scenario ar-errors: FileNotFoundError: /data/simulator/decks/AR.json"

VA_CONTAINER = {
    "control_system": {
        "type": "virtual_accelerator",
        "connector": {"virtual_accelerator": {"host": "localhost"}},
    }
}
VA_AND_STANDIN = {
    "control_system": {
        "type": "live_standin",
        "connector": {"virtual_accelerator": {"host": "localhost"}, "live_standin": {}},
    }
}
IN_PROCESS = {
    "control_system": {
        "type": "virtual_accelerator",
        "connector": {"virtual_accelerator": {"serving": "in_process"}},
    }
}


class _StatusChannels:
    """A connector serving fixed status texts as char arrays, as a CA read returns them."""

    def __init__(self, values: dict[str, str]) -> None:
        self.values = values
        self.disconnected = False

    async def read_channel(self, address: str, timeout: float | None = None) -> Any:
        del timeout
        return SimpleNamespace(value=[*self.values[address].encode("utf-8"), 0])

    async def disconnect(self) -> None:
        self.disconnected = True


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    """A repo root whose render holds the view: ``build/config.yml`` beside ``build/data/``."""
    add_deck_models(write_simulator_view(tmp_path / "build"))
    (tmp_path / "build" / "config.yml").write_text("{}\n", encoding="utf-8")
    return tmp_path


def _serve(monkeypatch: pytest.MonkeyPatch, by_target: dict[str | None, Any]) -> list[str | None]:
    """Answer each target's connector from ``by_target``; returns the targets asked for."""
    from osprey_connectors.factory import ConnectorFactory

    asked: list[str | None] = []

    async def create(section: dict, control_target: str | None = None) -> Any:
        del section
        asked.append(control_target)
        answer = by_target[control_target]
        if isinstance(answer, Exception):
            raise answer
        return answer

    monkeypatch.setattr(ConnectorFactory, "create_control_system_connector", create)
    return asked


def _run(config: dict[str, Any] | None, repo: Path) -> list[Any]:
    factory = get_core_category_factory("simulator")
    return asyncio.run(factory(config, context=None, cwd=repo)())


def test_a_failed_model_is_an_error_row_naming_it_and_its_error(
    repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    connector = _StatusChannels(
        {"ca:SIM:AR:STATUS": AR_ERROR, "ca:SIM:BR:STATUS": "no deck", "ca:SIM:SR:STATUS": "ok"}
    )
    _serve(monkeypatch, {"va": connector})

    rows = {row.name: row for row in _run(VA_CONTAINER, repo)}

    assert set(rows) == {"simulator_va_AR", "simulator_va_BR", "simulator_va_SR"}
    assert rows["simulator_va_AR"].status is Status.ERROR
    assert rows["simulator_va_AR"].message == "AR on va: failed"
    assert AR_ERROR in rows["simulator_va_AR"].details
    assert connector.disconnected


def test_a_model_with_no_deck_is_ok_and_said_to_be_not_built(
    repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    connector = _StatusChannels(
        {"ca:SIM:AR:STATUS": "ok", "ca:SIM:BR:STATUS": "no deck", "ca:SIM:SR:STATUS": "ok"}
    )
    _serve(monkeypatch, {"va": connector})

    rows = {row.name: row for row in _run(VA_CONTAINER, repo)}

    assert {row.status for row in rows.values()} == {Status.OK}
    assert rows["simulator_va_BR"].message == "BR on va: no deck, not built by design"
    assert rows["simulator_va_SR"].message == "SR on va: ok"


def test_every_served_simulator_target_is_read(repo: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    healthy = {"ca:SIM:AR:STATUS": "ok", "ca:SIM:BR:STATUS": "no deck", "ca:SIM:SR:STATUS": "ok"}
    asked = _serve(
        monkeypatch,
        {
            "va": _StatusChannels({**healthy, "ca:SIM:AR:STATUS": AR_ERROR}),
            "standin": _StatusChannels(healthy),
        },
    )

    rows = {row.name: row for row in _run(VA_AND_STANDIN, repo)}

    assert asked == ["va", "standin"]
    assert rows["simulator_va_AR"].status is Status.ERROR
    assert rows["simulator_standin_AR"].status is Status.OK


def test_an_unreadable_target_is_one_warning_row(
    repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _serve(monkeypatch, {"va": ConnectionError("no server at localhost:5064")})

    rows = _run(VA_CONTAINER, repo)

    assert [(row.name, row.status) for row in rows] == [("simulator_va", Status.WARNING)]
    assert "localhost:5064" in rows[0].details


def test_a_simulator_served_in_process_gives_no_rows(
    repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    asked = _serve(monkeypatch, {})

    assert _run(IN_PROCESS, repo) == []
    assert asked == []


def test_a_render_without_a_view_gives_no_rows(tmp_path: Path) -> None:
    assert _run(VA_CONTAINER, tmp_path) == []


def test_no_config_gives_no_rows(repo: Path) -> None:
    assert _run(None, repo) == []
