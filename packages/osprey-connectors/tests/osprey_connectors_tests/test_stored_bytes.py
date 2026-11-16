"""The bytes each file family is stored as: a record's JSON form is ``model_dump(mode="json")``.

The stdlib hook twins (``src/osprey/templates/claude_code/claude/hooks/``) read
these files, so the bytes a writer stores are a contract, not an implementation
detail. Each expected value below was captured from the writers before the
records' ``to_payload`` methods were replaced by their serializers, and every
case writes through the module's own file writer, so a change to a serializer,
a field order, a key or a number's type shows here as a byte difference.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import BaseModel

from osprey_connectors import control_context
from osprey_connectors import write_authorization as wa
from osprey_connectors.control_context import ControlContext, Owner, ServerReport
from osprey_connectors.write_authorization import (
    Approver,
    ChannelRange,
    Grant,
    Ledger,
    LedgerStep,
    Revocation,
    WriteRequest,
)

ID = "0123456789abcdef0123456789abcdef"

CONTEXT_FULL = ControlContext(
    target="live",
    generation=3,
    owner=Owner(kind="web_terminal", pid=1234, port=8087),
    # Read off disk and never stored: the writer derives the field from ``switch``.
    posture={"live": "sandbox", "extra": "ignored"},
    last_switch={
        "request_id": "r-1",
        "terminus": "applied",
        "target": "live",
        "nested": {"a": [1, 2]},
    },
    switch={"standin": "asked", "live": "allowed"},
    switch_epoch=1700000000000000001,
)
CONTEXT_BARE = ControlContext(target="sandbox", generation=0)
REPORT_DISPLAY = ServerReport(
    server_pid=4321,
    session="s-1",
    applied_target="va",
    applied_generation=7,
    children=(11, 12),
    reachability={"va": {"state": "ok", "at": "2026-01-01T00:00:00+00:00"}},
    last_switch={"state": "applying", "expires_at": "2026-01-01T00:01:00+00:00"},
    last_posture_realign={"at": "x", "posture": {"va": "sandbox"}},
    targets={"va": {"label": "VA", "kind": "simulator"}},
    updated_at="2026-01-01T00:00:00+00:00",
    statement={"reachable": {"state": "yes", "since": "t", "reason": None}, "writes": None},
)
REPORT_SERVER = ServerReport(server_pid=99)
GRANT = Grant(
    id=ID,
    owner="alice",
    holder="bob",
    target="live",
    switch_epoch=5,
    # Built in this process, an ``int`` bound or lease is kept as given.
    channels={"SR:B:SP": ChannelRange(min=0, max=10.5), "SR:A:SP": ChannelRange(min=-1.5, max=2)},
    steps=3,
    issued_at_ns=100,
    expires_at_ns=200,
    lease_s=60,
    approver=Approver(identity="alice", surface="chip", at_ns=100),
    label="run-1",
    revoked=Revocation(at_ns=150, by="alice"),
)
LEDGER = Ledger(
    authorization=ID,
    holder_pid=42,
    holder_start="777",
    pid_ns="pid:[4026531836]",
    lease_until_ns=300,
    steps=(
        LedgerStep(n=1, at_ns=110, channels=("SR:A:SP",)),
        LedgerStep(n=2, at_ns=120, channels=("SR:A:SP", "SR:B:SP")),
    ),
)
REQUEST = WriteRequest(
    id=ID,
    owner="alice",
    holder="bob",
    target="live",
    channels={"SR:A:SP": ChannelRange(min=0.0, max=1.0)},
    steps=4,
    requested_at_ns=90,
    expires_at_ns=200,
    lease_s=30.5,
    label="scan",
    completion="restore",
)

STATE_DIRECTORY_FILES: dict[str, tuple[BaseModel, bytes]] = {
    "context_full": (
        CONTEXT_FULL,
        b"""{
  "schema": 1,
  "owner": {
    "kind": "web_terminal",
    "pid": 1234,
    "port": 8087
  },
  "target": "live",
  "generation": 3,
  "posture": {
    "va": "sandbox"
  },
  "last_switch": {
    "request_id": "r-1",
    "terminus": "applied",
    "target": "live",
    "nested": {
      "a": [
        1,
        2
      ]
    }
  },
  "switch": {
    "standin": "asked",
    "live": "allowed"
  },
  "switch_epoch": 1700000000000000001
}""",
    ),
    "context_bare": (
        CONTEXT_BARE,
        b"""{
  "schema": 1,
  "owner": null,
  "target": "sandbox",
  "generation": 0,
  "posture": {
    "live": "sandbox",
    "va": "sandbox",
    "standin": "sandbox"
  },
  "last_switch": null,
  "switch": {},
  "switch_epoch": null
}""",
    ),
    "report_display": (
        REPORT_DISPLAY,
        b"""{
  "server_pid": 4321,
  "session": "s-1",
  "applied_target": "va",
  "applied_generation": 7,
  "children": [
    11,
    12
  ],
  "reachability": {
    "va": {
      "state": "ok",
      "at": "2026-01-01T00:00:00+00:00"
    }
  },
  "last_switch": {
    "state": "applying",
    "expires_at": "2026-01-01T00:01:00+00:00"
  },
  "last_posture_realign": {
    "at": "x",
    "posture": {
      "va": "sandbox"
    }
  },
  "targets": {
    "va": {
      "label": "VA",
      "kind": "simulator"
    }
  },
  "updated_at": "2026-01-01T00:00:00+00:00",
  "statement": {
    "reachable": {
      "state": "yes",
      "since": "t",
      "reason": null
    },
    "writes": null
  }
}""",
    ),
    "report_server": (
        REPORT_SERVER,
        b"""{
  "server_pid": 99,
  "session": null,
  "applied_target": null,
  "applied_generation": null,
  "children": [],
  "reachability": {},
  "last_switch": null,
  "last_posture_realign": null,
  "targets": {},
  "updated_at": null
}""",
    ),
}

AUTHORIZATION_FILES: dict[str, tuple[BaseModel, bytes]] = {
    "grant": (
        GRANT,
        b"""{
  "approver": {
    "at_ns": 100,
    "identity": "alice",
    "surface": "chip"
  },
  "channels": {
    "SR:A:SP": {
      "max": 2,
      "min": -1.5
    },
    "SR:B:SP": {
      "max": 10.5,
      "min": 0
    }
  },
  "expires_at_ns": 200,
  "holder": "bob",
  "id": "0123456789abcdef0123456789abcdef",
  "issued_at_ns": 100,
  "label": "run-1",
  "lease_s": 60,
  "owner": "alice",
  "revoked": {
    "at_ns": 150,
    "by": "alice"
  },
  "schema": 1,
  "steps": 3,
  "switch_epoch": 5,
  "target": "live"
}
""",
    ),
    "ledger": (
        LEDGER,
        b"""{
  "authorization": "0123456789abcdef0123456789abcdef",
  "holder_pid": 42,
  "holder_start": "777",
  "lease_until_ns": 300,
  "pid_ns": "pid:[4026531836]",
  "schema": 1,
  "steps": [
    {
      "at_ns": 110,
      "channels": [
        "SR:A:SP"
      ],
      "n": 1
    },
    {
      "at_ns": 120,
      "channels": [
        "SR:A:SP",
        "SR:B:SP"
      ],
      "n": 2
    }
  ]
}
""",
    ),
    "request": (
        REQUEST,
        b"""{
  "channels": {
    "SR:A:SP": {
      "max": 1.0,
      "min": 0.0
    }
  },
  "completion": "restore",
  "expires_at_ns": 200,
  "holder": "bob",
  "id": "0123456789abcdef0123456789abcdef",
  "label": "scan",
  "lease_s": 30.5,
  "owner": "alice",
  "requested_at_ns": 90,
  "schema": 1,
  "steps": 4,
  "target": "live"
}
""",
    ),
}


@pytest.mark.parametrize("name", list(STATE_DIRECTORY_FILES))
def test_a_state_directory_file_is_stored_as_before(name: str, tmp_path: Path) -> None:
    record, expected = STATE_DIRECTORY_FILES[name]
    path = tmp_path / f"{name}.json"

    control_context.write_json_atomic(path, record.model_dump(mode="json"))

    assert path.read_bytes() == expected


@pytest.mark.parametrize("name", list(AUTHORIZATION_FILES))
def test_an_authorization_file_is_stored_as_before(name: str, tmp_path: Path) -> None:
    record, expected = AUTHORIZATION_FILES[name]
    path = tmp_path / f"{name}.json"

    wa._write_json(path, record.model_dump(mode="json"))

    assert path.read_bytes() == expected


def test_the_stored_record_and_report_read_back_as_written() -> None:
    record = control_context.parse_record(STATE_DIRECTORY_FILES["context_full"][1])
    assert record is not None
    assert (record.owner, record.switch, record.last_switch) == (
        CONTEXT_FULL.owner,
        CONTEXT_FULL.switch,
        CONTEXT_FULL.last_switch,
    )
    assert record.posture == {"va": "sandbox"}
    for name in ("report_display", "report_server"):
        report, stored = STATE_DIRECTORY_FILES[name]
        assert control_context._parse_report(json.loads(stored)) == report


@pytest.mark.parametrize(
    ("name", "parse"),
    [("grant", wa._parse_grant), ("ledger", wa._parse_ledger), ("request", wa._parse_request)],
)
def test_an_authorization_file_reads_back_as_written(name: str, parse: object) -> None:
    record, stored = AUTHORIZATION_FILES[name]
    assert callable(parse)
    parsed = parse(stored)
    # Read from a file, a number field is a float (``60`` reads as ``60.0``).
    assert parsed.model_dump(mode="json") == record.model_dump(mode="json")
