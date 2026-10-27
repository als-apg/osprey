"""Control-context files a deployment already holds still read the same.

The golden record and report under ``fixtures/control_context/`` were written
once by this package's writer and are never regenerated: a deployment keeps
its record in durable ``var/`` across a rebuild that picks up a newer
connectors, so the newer reader must take the older file as it stands. A
change that needs a new golden is a schema change, and a deployment is
regenerated rather than migrated for one (see ``SCHEMA_VERSION``).
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
    assert json.loads(path.read_text()) == golden


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
