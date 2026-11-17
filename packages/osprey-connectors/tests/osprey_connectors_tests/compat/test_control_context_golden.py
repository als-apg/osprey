"""Control-context files a deployment already holds still read the same.

The golden record and report under ``fixtures/control_context/`` were written
once by this package's writer and are never regenerated: a deployment keeps
its record in durable ``var/`` across a rebuild that picks up a newer
connectors, so the newer reader must take the older file as it stands. A
change that needs a new golden is a schema change, and a deployment is
regenerated rather than migrated for one (see ``SCHEMA_VERSION``).

Schema 1 has a normal form. The first write of a record that carries no
``switch_epoch`` -- every golden one -- mints the epoch, stores the switch
empty (every machine's writes off) and writes the ``posture`` mirror out for
every machine in ``CONTROL_TARGETS``; the target, generation, owner and last
switch are carried unchanged (``contracts.check_golden_rewrite``). So an older
record survives a newer writer with the same meaning, not the same bytes, and
the byte guarantee holds from the normal form on: a record already in it
re-writes byte for byte, its epoch not re-minted and its posture not
re-ordered.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from osprey_connectors import control_context
from osprey_connectors_tests.compat import contracts


def test_the_golden_record_and_report_read_as_they_state() -> None:
    assert contracts.check_control_context_golden().startswith("control-context golden OK")


def test_the_golden_record_round_trips_through_the_writer(tmp_path: Path) -> None:
    golden = json.loads((contracts.FIXTURES / control_context.RECORD_FILENAME).read_text())
    record = control_context.parse_record(golden)
    assert record is not None
    path = control_context.write_record(record, path=tmp_path / control_context.RECORD_FILENAME)
    contracts.check_golden_rewrite(json.loads(path.read_text()), golden)


def test_a_record_in_normal_form_rewrites_byte_for_byte(tmp_path: Path) -> None:
    golden = json.loads((contracts.FIXTURES / control_context.RECORD_FILENAME).read_text())
    first = control_context.parse_record(golden)
    assert first is not None
    path = control_context.write_record(first, path=tmp_path / control_context.RECORD_FILENAME)
    normal = path.read_bytes()

    again = control_context.parse_record(json.loads(normal))
    assert again is not None
    control_context.write_record(again, path=path)
    assert path.read_bytes() == normal
    assert json.loads(path.read_bytes())["switch_epoch"] == json.loads(normal)["switch_epoch"]


def test_a_rewrite_that_drops_the_owner_breaks_the_contract(tmp_path: Path) -> None:
    golden = json.loads((contracts.FIXTURES / control_context.RECORD_FILENAME).read_text())
    record = control_context.parse_record(golden)
    assert record is not None
    path = control_context.write_record(record, path=tmp_path / control_context.RECORD_FILENAME)
    rewrite = {**json.loads(path.read_text()), "owner": None}
    with pytest.raises(contracts.ContractError, match="owner"):
        contracts.check_golden_rewrite(rewrite, golden)


def test_a_golden_record_the_reader_refuses_breaks_the_contract(tmp_path: Path) -> None:
    fixtures = tmp_path / "fixtures"
    shutil.copytree(contracts.FIXTURES, fixtures)
    record = fixtures / control_context.RECORD_FILENAME
    record.write_text(json.dumps({**json.loads(record.read_text()), "schema": 2}))
    with pytest.raises(contracts.ContractError, match="no longer parses"):
        contracts.check_control_context_golden(fixtures)


def test_a_golden_report_the_reader_refuses_breaks_the_contract(tmp_path: Path) -> None:
    fixtures = tmp_path / "fixtures"
    shutil.copytree(contracts.FIXTURES, fixtures)
    (report,) = fixtures.glob(control_context.REPORT_FILE_GLOB)
    report.write_text(json.dumps({**json.loads(report.read_text()), "server_pid": 0}))
    with pytest.raises(contracts.ContractError, match="no longer parses"):
        contracts.check_control_context_golden(fixtures)


def test_the_golden_switch_record_reads_its_legacy_on_as_asked() -> None:
    contracts.check_switch_golden()


def test_a_golden_switch_the_reader_reads_as_off_breaks_the_contract(tmp_path: Path) -> None:
    fixtures = tmp_path / "switch_on"
    shutil.copytree(contracts.SWITCH_ON_FIXTURES, fixtures)
    record = fixtures / control_context.RECORD_FILENAME
    record.write_text(json.dumps({**json.loads(record.read_text()), "switch": {"va": "ON"}}))
    with pytest.raises(contracts.ContractError, match="golden switch record reads as"):
        contracts.check_switch_golden(fixtures)


def test_the_legacy_gesture_word_is_refused_and_leaves_the_record(tmp_path: Path) -> None:
    source = contracts.SWITCH_ON_FIXTURES / control_context.RECORD_FILENAME
    path = contracts.check_set_switch_legacy_word(tmp_path)
    assert path.read_bytes() == source.read_bytes()


def test_a_package_that_takes_the_legacy_gesture_word_breaks_the_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(control_context, "set_switch", lambda *_args, **_kwargs: None)
    with pytest.raises(contracts.ContractError, match="was taken"):
        contracts.check_set_switch_legacy_word(tmp_path)


def test_a_refused_legacy_gesture_that_writes_breaks_the_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def write_then_refuse(_targets, state, *, path, **_kwargs):
        path.write_text("{}", encoding="utf-8")
        raise ValueError(f"set_switch state must be one of the positions, not {state!r}")

    monkeypatch.setattr(control_context, "set_switch", write_then_refuse)
    with pytest.raises(contracts.ContractError, match="wrote the record"):
        contracts.check_set_switch_legacy_word(tmp_path)
