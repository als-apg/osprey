"""The current-value route, ``GET /api/directory/values?name=…``.

Every test runs the real lifespan over a config with the ``pv_directory`` section,
with the value supervisor swapped for a fake that answers the way
``ValueSupervisor.read`` does, so no child starts and no control session opens.
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("osprey_connectors.pv_directory.contract")

from fastapi.testclient import TestClient

from osprey.audit import writer as audit_writer
from osprey.interfaces.channel_finder import directory_values_supervisor
from osprey.interfaces.channel_finder.directory_values_ipc import (
    MAX_NAMES,
    OUTCOME_NO_ANSWER,
    OUTCOME_UNAVAILABLE,
    ValueRow,
)
from osprey.interfaces.common_middleware import HTTP_MUTATION_SURFACE
from osprey_connectors import control_context

_CONFIG_SEAM = "osprey_connectors.workspace.load_osprey_config"
_FIXTURE = Path(__file__).parent / "fixtures" / "directory_routes.json"
_ROUTE = "/api/directory/values"


class FakeSupervisor:
    """Answers reads from fixed rows; ``reason`` set makes every row unavailable."""

    rows: dict[str, ValueRow] = {}
    reason: str | None = None
    reads: list[list[str]] = []

    def __init__(self, config_path: Path) -> None:
        self.config_path = config_path

    def start(self) -> None:
        pass

    def stop(self) -> None:
        pass

    async def read(self, names: Sequence[str]) -> list[ValueRow]:
        names = list(names)
        if len(names) > MAX_NAMES:
            raise ValueError(f"{len(names)} names is over the batch limit of {MAX_NAMES}")
        type(self).reads.append(names)
        if self.reason is not None:
            return [
                ValueRow(name=name, outcome=OUTCOME_UNAVAILABLE, reason=self.reason)
                for name in names
            ]
        return [
            self.rows.get(name)
            or ValueRow(name=name, outcome=OUTCOME_NO_ANSWER, reason="no answer within 2 s")
            for name in names
        ]


@pytest.fixture(autouse=True)
def fake_supervisor(monkeypatch: pytest.MonkeyPatch) -> type[FakeSupervisor]:
    monkeypatch.setattr(directory_values_supervisor, "ValueSupervisor", FakeSupervisor)
    monkeypatch.setattr(FakeSupervisor, "rows", {})
    monkeypatch.setattr(FakeSupervisor, "reason", None)
    monkeypatch.setattr(FakeSupervisor, "reads", [])
    return FakeSupervisor


@pytest.fixture()
def client(
    mock_registry,  # noqa: ARG001
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> Iterator[TestClient]:
    monkeypatch.setenv("OSPREY_CONFIG", str(tmp_path / "config.yml"))
    monkeypatch.setattr(control_context, "read_record", lambda **_: None)
    (tmp_path / "facility.json").write_text(
        json.dumps({"identity": {"code": "test", "name": "Test"}, "channels": []}),
        encoding="utf-8",
    )
    config = {
        "config_dir": str(tmp_path),
        "channel_finder": {
            "pipeline_mode": "in_context",
            "pipelines": {"in_context": {"database": {"path": "/tmp/db.json", "type": "flat"}}},
        },
        "control_system": {"type": "virtual_accelerator"},
        "services": {"virtual_accelerator": {"port": 5064}},
        "pv_directory": {"type": "channelfinder", "settings": {"url": "http://cf.example"}},
    }
    monkeypatch.setattr(_CONFIG_SEAM, lambda: config)
    from osprey.interfaces.channel_finder.app import create_app

    with TestClient(create_app(project_cwd="/tmp/test-project")) as test_client:
        yield test_client


def _fixture_values() -> list[dict[str, Any]]:
    return json.loads(_FIXTURE.read_text(encoding="utf-8"))["values"]


def test_the_values_payload_matches_the_shared_fixture(client: TestClient) -> None:
    expected = _fixture_values()
    found = ValueRow.from_wire(expected[0])
    FakeSupervisor.rows = {found.name: found}

    response = client.get(_ROUTE, params={"name": [row["name"] for row in expected]})

    assert response.status_code == 200
    assert response.json() == expected
    assert "reason" not in response.json()[0]
    assert response.json()[1]["reason"]


def test_a_name_without_an_answer_reads_no_answer_in_request_order(client: TestClient) -> None:
    names = ["SR:B", "SR:A", "SR:C"]

    rows = client.get(_ROUTE, params={"name": names}).json()

    assert [row["name"] for row in rows] == names
    assert {row["outcome"] for row in rows} == {OUTCOME_NO_ANSWER}
    assert FakeSupervisor.reads == [names]


def test_an_unavailable_supervisor_makes_every_row_unavailable_with_the_reason(
    client: TestClient,
) -> None:
    FakeSupervisor.reason = "the value process is not running"

    response = client.get(_ROUTE, params={"name": ["SR:A", "SR:B"]})

    assert response.status_code == 200
    assert response.json() == [
        {
            "name": name,
            "outcome": OUTCOME_UNAVAILABLE,
            "value": None,
            "timestamp": None,
            "severity": None,
            "status": None,
            "reason": "the value process is not running",
        }
        for name in ("SR:A", "SR:B")
    ]


def test_the_cap_admits_exactly_max_names(client: TestClient) -> None:
    names = [f"SR:PV{index}" for index in range(MAX_NAMES)]

    response = client.get(_ROUTE, params={"name": names})

    assert response.status_code == 200
    assert len(response.json()) == MAX_NAMES


def test_more_names_than_the_cap_is_a_422_before_any_read(client: TestClient) -> None:
    names = [f"SR:PV{index}" for index in range(MAX_NAMES + 1)]

    response = client.get(_ROUTE, params={"name": names})

    assert response.status_code == 422
    assert FakeSupervisor.reads == []


def test_no_names_reads_nothing(client: TestClient) -> None:
    response = client.get(_ROUTE)

    assert response.status_code == 200
    assert response.json() == []


def test_the_answer_is_never_cached(client: TestClient) -> None:
    response = client.get(_ROUTE, params={"name": ["SR:A"]})

    directives = {part.strip() for part in response.headers["cache-control"].split(",")}
    assert "no-store" in directives


def test_post_is_not_a_method_of_the_route(client: TestClient) -> None:
    response = client.post(_ROUTE, json={"names": ["SR:A"]})

    assert response.status_code == 405
    assert FakeSupervisor.reads == []


def test_a_read_files_no_http_mutation_record(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    recorded: list[dict[str, Any]] = []

    def fake_record(**fields: Any) -> None:
        recorded.append(fields)

    monkeypatch.setattr(audit_writer, "record", fake_record)

    response = client.get(_ROUTE, params={"name": ["SR:A", "SR:B"]})

    assert response.status_code == 200
    assert [f for f in recorded if f.get("surface") == HTTP_MUTATION_SURFACE] == []
