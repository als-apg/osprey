"""The ``versions`` health category reports the running osprey and its CANOPY pin."""

from __future__ import annotations

from osprey.health.core import versions as versions_mod
from osprey.health.core.versions import versions
from osprey.health.models import Status
from osprey.integrations.canopy.proven import (
    ABSENT,
    NO_RECORD,
    NOT_INSTALLED,
    PROVEN,
    PROVEN_COMMITS,
    UNPROVEN,
    Held,
    Judgement,
    Reading,
)

_CANOPY_ON = {"web": {"panels": {"canopy": {"enabled": True}}}}


def _judgement(held: Held, state: str) -> Judgement:
    pin = PROVEN_COMMITS["canopy"]
    reading = Reading("canopy", pin, held, held.commit == pin)
    return Judgement((reading,), state)


def _rows(
    config: dict,
    monkeypatch,
    *,
    running: str,
    judgement: Judgement | None = None,
    canopy_version: str | None = None,
) -> dict:
    monkeypatch.setattr(versions_mod, "_running_version", lambda: running)
    monkeypatch.setattr(versions_mod, "_canopy_version", lambda: canopy_version)
    if judgement is not None:
        monkeypatch.setattr(versions_mod, "_canopy_judgement", lambda: judgement)
    return {row.name: row for row in versions(config)()}


def test_a_development_build_shows_its_commit(monkeypatch):
    row = _rows({}, monkeypatch, running="2026.9.0b3.post57+gbee9e1557.d20261106")[
        "versions.osprey"
    ]
    assert row.status is Status.OK
    assert row.value == "bee9e1557"
    assert "2026.9.0b3" in row.message


def test_a_release_shows_its_version(monkeypatch):
    assert _rows({}, monkeypatch, running="2026.9.0")["versions.osprey"].value == "2026.9.0"


def test_canopy_row_only_when_the_panel_is_declared(monkeypatch):
    assert "versions.canopy" not in _rows({}, monkeypatch, running="2026.9.0")


def test_canopy_at_the_pin(monkeypatch):
    pin = PROVEN_COMMITS["canopy"]
    row = _rows(
        _CANOPY_ON, monkeypatch, running="1", judgement=_judgement(Held(commit=pin), PROVEN)
    )["versions.canopy"]
    assert row.status is Status.OK
    assert row.value == f"{pin[:12]} = pin"


def test_canopy_off_the_pin_warns(monkeypatch):
    row = _rows(
        _CANOPY_ON,
        monkeypatch,
        running="1",
        judgement=_judgement(Held(commit="deadbeef" * 5), UNPROVEN),
    )["versions.canopy"]
    assert row.status is Status.WARNING
    assert row.value == f"deadbeefdead ≠ pin {PROVEN_COMMITS['canopy'][:12]}"


def test_canopy_not_installed_here_is_a_skip(monkeypatch):
    row = _rows(
        _CANOPY_ON,
        monkeypatch,
        running="1",
        judgement=_judgement(Held(reason=NOT_INSTALLED), ABSENT),
    )["versions.canopy"]
    assert row.status is Status.SKIP
    assert row.value == "not installed here"


def test_canopy_recorded_commit_at_the_pin_ignores_the_label(monkeypatch):
    pin = PROVEN_COMMITS["canopy"]
    row = _rows(
        _CANOPY_ON,
        monkeypatch,
        running="1",
        judgement=_judgement(Held(commit=pin), PROVEN),
        canopy_version="0.0.2.dev291+gdeadbeef",
    )["versions.canopy"]
    assert row.status is Status.OK
    assert row.value == f"{pin[:12]} = pin"


def test_canopy_wheel_without_a_record_at_the_pin_by_its_label(monkeypatch):
    pin = PROVEN_COMMITS["canopy"]
    row = _rows(
        _CANOPY_ON,
        monkeypatch,
        running="1",
        judgement=_judgement(Held(reason=NO_RECORD), UNPROVEN),
        canopy_version=f"0.0.2.dev291+g{pin[:9]}",
    )["versions.canopy"]
    assert row.status is Status.OK
    assert row.value == f"{pin[:12]} = pin"


def test_canopy_wheel_without_a_record_labelled_elsewhere_warns(monkeypatch):
    row = _rows(
        _CANOPY_ON,
        monkeypatch,
        running="1",
        judgement=_judgement(Held(reason=NO_RECORD), UNPROVEN),
        canopy_version="0.0.2.dev291+gdeadbeef1",
    )["versions.canopy"]
    assert row.status is Status.WARNING
    assert row.value == f"deadbeef1 ≠ pin {PROVEN_COMMITS['canopy'][:12]}"


def test_canopy_without_a_record_or_a_label_warns(monkeypatch):
    row = _rows(
        _CANOPY_ON,
        monkeypatch,
        running="1",
        judgement=_judgement(Held(reason=NO_RECORD), UNPROVEN),
        canopy_version="0.0.2",
    )["versions.canopy"]
    assert row.status is Status.WARNING
    assert row.value == f"{NO_RECORD} ≠ pin {PROVEN_COMMITS['canopy'][:12]}"
