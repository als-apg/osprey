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
import io
import json
from pathlib import Path
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


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    """A repo root whose render holds the view: ``build/config.yml`` beside ``build/data/``."""
    add_deck_models(write_simulator_view(tmp_path / "build"))
    (tmp_path / "build" / "config.yml").write_text("{}\n", encoding="utf-8")
    return tmp_path


def _serve(monkeypatch: pytest.MonkeyPatch, by_target: dict[str, Any]) -> list[str]:
    """Answer each target's child from ``by_target``; returns the targets asked for."""
    asked: list[str] = []

    async def read(section: Any, target: str, models: Any, addresses: Any, **kwargs: Any) -> Any:
        del section, models, addresses, kwargs
        asked.append(target)
        answer = by_target[target]
        if isinstance(answer, Exception):
            raise answer
        return answer

    monkeypatch.setattr("osprey.simulation.status.read_in_child", read)
    return asked


def _run(config: dict[str, Any] | None, repo: Path) -> list[Any]:
    factory = get_core_category_factory("simulator")
    return asyncio.run(factory(config, context=None, cwd=repo)())


def test_a_failed_model_is_an_error_row_naming_it_and_its_error(
    repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _serve(monkeypatch, {"va": {"AR": AR_ERROR, "BR": "no deck", "SR": "ok"}})

    rows = {row.name: row for row in _run(VA_CONTAINER, repo)}

    assert set(rows) == {"simulator_va_AR", "simulator_va_BR", "simulator_va_SR"}
    assert rows["simulator_va_AR"].status is Status.ERROR
    assert rows["simulator_va_AR"].message == "AR on va: failed"
    assert AR_ERROR in rows["simulator_va_AR"].details


def test_a_model_with_no_deck_is_ok_and_said_to_be_not_built(
    repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _serve(monkeypatch, {"va": {"AR": "ok", "BR": "no deck", "SR": "ok"}})

    rows = {row.name: row for row in _run(VA_CONTAINER, repo)}

    assert {row.status for row in rows.values()} == {Status.OK}
    assert rows["simulator_va_BR"].message == "BR on va: no deck, not built by design"
    assert rows["simulator_va_SR"].message == "SR on va: ok"


def test_every_served_simulator_target_is_read(repo: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    healthy = {"AR": "ok", "BR": "no deck", "SR": "ok"}
    asked = _serve(monkeypatch, {"va": {**healthy, "AR": AR_ERROR}, "standin": healthy})

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


def test_reading_a_target_never_repoints_this_processs_channel_access(
    repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The read leaves for a child at once: no connector is built in the suite's process."""
    from osprey_connectors.factory import ConnectorFactory

    async def refuse(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("a connector was built in the health process")

    monkeypatch.setattr(ConnectorFactory, "create_control_system_connector", refuse)
    monkeypatch.setattr("osprey.health.core.simulator._READ_TIMEOUT_S", 0.001)

    rows = _run(VA_CONTAINER, repo)

    assert [(row.name, row.status) for row in rows] == [("simulator_va", Status.WARNING)]


def test_the_child_prints_each_models_status_read_through_the_connector(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from types import SimpleNamespace

    from osprey.simulation import status as status_module
    from osprey_connectors.factory import ConnectorFactory

    class Channels:
        async def read_channel(self, address: str) -> Any:
            return SimpleNamespace(value=[*f"read {address}".encode(), 0])

        async def disconnect(self) -> None:
            pass

    async def create(section: dict, control_target: str | None = None) -> Channels:
        del section
        assert control_target == "va"
        return Channels()

    class Exited(Exception):
        pass

    def exit_now(code: int) -> None:
        raise Exited(code)

    monkeypatch.setattr(ConnectorFactory, "create_control_system_connector", create)
    monkeypatch.setattr(status_module.os, "_exit", exit_now)
    payload = {"section": {}, "target": "va", "models": ["AR"], "addresses": {"AR": "ca:AR"}}
    monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps(payload)))

    with pytest.raises(Exited):
        status_module._main()

    assert 'OSPREY_MODEL_STATUSES {"AR": "read ca:AR"}' in capsys.readouterr().out


def test_the_child_does_not_import_from_the_working_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A stray ``json.py`` in the cwd never runs: the real child reaches its own failure."""
    from osprey.simulation.status import read_in_child

    marker = tmp_path / "ran"
    (tmp_path / "json.py").write_text(
        f"import pathlib\npathlib.Path({str(marker)!r}).touch()\nraise SystemExit(0)\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)

    # An empty section builds no reachable connector, so the child answers with an error.
    with pytest.raises(RuntimeError) as failure:
        asyncio.run(
            read_in_child({}, "standin", ["M"], {"M": "c:M"}, config_path=None, timeout_s=120.0)
        )

    assert not marker.exists()
    assert "No module named" not in str(failure.value)


def test_the_no_deck_text_is_the_one_the_composite_sends() -> None:
    from osprey.simulation.status import STATUS_NO_DECK
    from osprey_connectors.simulation import composite

    assert composite.STATUS_NO_DECK == STATUS_NO_DECK
