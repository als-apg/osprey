"""Tests for the durable audit record written when a readonly write is refused.

The issue's requirement is "log the attempt with the offending source for
audit", so these assert on two things the server log alone cannot give: the
record lands in the durable ``var/audit`` zone, and it carries the source that
was refused.
"""

import json

import pytest

from osprey.services.python_executor import refusal_audit

pytestmark = pytest.mark.unit


@pytest.fixture
def audit_root(tmp_path, monkeypatch):
    """Redirect the audit zone at its single seam."""
    target = tmp_path / "var" / "audit"
    monkeypatch.setattr(refusal_audit, "audit_dir", lambda: target)
    return target


def _records(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def test_record_lands_in_the_audit_zone(audit_root):
    written = refusal_audit.record_refusal(
        layer=refusal_audit.LAYER_IMPORT_DENYLIST,
        trigger=["Import of 'epics' is not allowed in readonly mode"],
        source="import epics\nepics.caput('SR:MAG:QF:01:CURRENT:SP', 150)",
        description="nudge the corrector",
        tool="execute",
    )

    assert written == audit_root / refusal_audit.REFUSAL_LOG_FILENAME
    assert written.is_file()


def test_record_carries_the_offending_source(audit_root):
    code = "w = getattr(epics, 'ca' + 'put')\nw('SR:MAG:QF:01:CURRENT:SP', 150)"
    refusal_audit.record_refusal(
        layer=refusal_audit.LAYER_RUNTIME_GUARD,
        trigger="readonly execution mode: control-system writes are refused",
        source=code,
        description="nudge the corrector",
        tool="execute",
    )

    (record,) = _records(audit_root / refusal_audit.REFUSAL_LOG_FILENAME)
    assert record["source"] == code
    assert record["layer"] == refusal_audit.LAYER_RUNTIME_GUARD
    assert record["mode"] == "readonly"
    assert record["tool"] == "execute"
    assert record["description"] == "nudge the corrector"
    assert record["ts"].endswith("Z")


def test_log_is_append_only(audit_root):
    for index in range(3):
        refusal_audit.record_refusal(
            layer=refusal_audit.LAYER_PATTERN_DETECTION,
            trigger={"writes": [r"\bcaput\s*\("]},
            source=f"caput('PV{index}', 1)",
            tool="execute",
        )

    records = _records(audit_root / refusal_audit.REFUSAL_LOG_FILENAME)
    assert [r["source"] for r in records] == [
        "caput('PV0', 1)",
        "caput('PV1', 1)",
        "caput('PV2', 1)",
    ]


def test_oversized_source_is_truncated_and_says_so(audit_root):
    refusal_audit.record_refusal(
        layer=refusal_audit.LAYER_PATTERN_DETECTION,
        trigger={"writes": [r"\bcaput\s*\("]},
        source="x" * (refusal_audit.MAX_SOURCE_CHARS + 500),
        tool="execute",
    )

    (record,) = _records(audit_root / refusal_audit.REFUSAL_LOG_FILENAME)
    assert len(record["source"]) == refusal_audit.MAX_SOURCE_CHARS
    assert record["source_truncated"] is True


def test_source_at_the_bound_is_not_marked_truncated(audit_root):
    refusal_audit.record_refusal(
        layer=refusal_audit.LAYER_PATTERN_DETECTION,
        trigger=[],
        source="x" * refusal_audit.MAX_SOURCE_CHARS,
        tool="execute",
    )

    (record,) = _records(audit_root / refusal_audit.REFUSAL_LOG_FILENAME)
    assert "source_truncated" not in record


def test_unwritable_audit_zone_does_not_raise(monkeypatch, caplog):
    """A refusal must never become a traceback because the log could not be written."""

    def _explode():
        raise OSError("read-only file system")

    monkeypatch.setattr(refusal_audit, "audit_dir", _explode)

    assert (
        refusal_audit.record_refusal(
            layer=refusal_audit.LAYER_RUNTIME_GUARD,
            trigger="refused",
            source="epics.caput('PV', 1)",
        )
        is None
    )


def test_the_event_reaches_the_server_log_even_without_a_file(monkeypatch, caplog):
    """The durable record is the goal; the log line is the floor under it."""

    def _explode():
        raise OSError("read-only file system")

    monkeypatch.setattr(refusal_audit, "audit_dir", _explode)

    with caplog.at_level("WARNING"):
        refusal_audit.record_refusal(
            layer=refusal_audit.LAYER_IMPORT_DENYLIST,
            trigger=["Import of 'epics' is not allowed in readonly mode"],
            source="import epics",
        )

    assert "Refused a control-system write in readonly mode" in caplog.text
    assert refusal_audit.LAYER_IMPORT_DENYLIST in caplog.text


def test_audit_dir_sits_under_the_state_zone(monkeypatch, tmp_path):
    """The path is derived, not spelled — it must track the repo the executor uses."""
    from osprey.utils.workspace import AUDIT_DIR_RELPATH

    monkeypatch.setattr(
        "osprey.utils.workspace.resolve_project_root", lambda *_args, **_kwargs: tmp_path
    )
    monkeypatch.setattr("osprey.utils.workspace.load_osprey_config", lambda *_a, **_k: {})

    assert refusal_audit.audit_dir() == tmp_path / AUDIT_DIR_RELPATH
    assert refusal_audit.refusal_log_path().name == refusal_audit.REFUSAL_LOG_FILENAME
