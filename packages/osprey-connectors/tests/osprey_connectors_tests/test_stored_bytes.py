"""The bytes each file family is stored as: a record's JSON form is ``model_dump(mode="json")``.

The stdlib hook twins (``src/osprey/templates/claude_code/claude/hooks/``) read
these files, so the bytes a writer stores are a contract, not an implementation
detail. Each expected value below was captured from the writers before the
records' ``to_payload`` methods were replaced by their serializers, and every
case writes through the module's own file writer, so a change to a serializer,
a field order, a key or a number's type shows here as a byte difference.

The time fields' bytes were re-captured when instants became aware datetimes:
every stamp a writer stores is ISO-8601 in UTC with an explicit ``+00:00``
(``.ffffff`` when it has microseconds), never ``Z`` and never an integer of
nanoseconds. A file in the previous ns-integer format is unreadable, and each
reader of one lands on its fail-closed path: a write verdict refuses
``auth_unreadable``, a request is listed as unreadable, and a session hold
record holds.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import uuid
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
from pydantic import BaseModel

from osprey_connectors import control_context, session_hold, testing
from osprey_connectors import write_authorization as wa
from osprey_connectors.config_sections import ControlSystemSection
from osprey_connectors.control_context import (
    ControlContext,
    MachineStatement,
    Reachable,
    RecordWriter,
    ServerReport,
    SwitchProgress,
    SwitchTerminus,
    WritesStatement,
)
from osprey_connectors.model import evolve, parse_instant
from osprey_connectors.write_authorization import (
    Approver,
    ChannelRange,
    Grant,
    Ledger,
    LedgerStep,
    Revocation,
    WriteRequest,
)
from osprey_connectors_tests._harness import isolate_verdict_environment, write_control_context
from osprey_connectors_tests._harness._control_context_fixtures import FIXTURE_IDENTITY
from osprey_connectors_tests._paths import REPO_ROOT

ID = "0123456789abcdef0123456789abcdef"

#: The fixed instant every pinned stamp is an offset from.
T0 = datetime(2026, 1, 1, tzinfo=UTC)

CONTEXT_FULL = ControlContext(
    target="live",
    generation=3,
    writer=RecordWriter(kind="web_terminal", pid=1234, port=8087),
    last_switch=SwitchTerminus(
        request_id="r-1",
        target="live",
        requested_at=None,
        requested_by="sess-1",
        status="applied",
        reason=None,
        detail="The control target is now 'live' (generation 3).",
        generation=3,
    ),
    switch={"standin": "asked", "live": "allowed"},
    switch_epoch=1700000000000000001,
)
CONTEXT_BARE = ControlContext(target="va", generation=0)
REPORT_DISPLAY = ServerReport(
    server_pid=4321,
    session="s-1",
    applied_target="va",
    applied_generation=7,
    children=(11, 12),
    reachability={"va": {"state": "ok", "at": "2026-01-01T00:00:00+00:00"}},
    switch_progress=SwitchProgress.applying(7, at=T0, expires_at=T0 + timedelta(minutes=1)),
    last_posture_realign={"at": "x", "posture": {"va": "sandbox"}},
    targets={"va": {"label": "VA", "kind": "simulator"}},
    updated_at=T0 + timedelta(microseconds=500_000),
    statement=MachineStatement(
        reachable=Reachable(state="yes", since=T0 + timedelta(microseconds=250_000)),
        writes=None,
    ),
)
REPORT_SERVER = ServerReport(server_pid=99)
#: A switch answer: the request's own stamp, given in another zone, is stored in UTC.
CONTEXT_TERMINUS = CONTEXT_BARE.answered(
    SwitchTerminus(
        request_id="r-2",
        target="live",
        requested_at=datetime(
            2026, 1, 1, 9, 30, 15, 125_000, tzinfo=timezone(timedelta(hours=9.5))
        ),
        requested_by="pid:4242",
        status="applied",
        reason=None,
        detail="",
        generation=1,
    )
)

#: The record the three answers to a switch request store, each built through
#: its constructor. Their bytes were captured from the writer before the block
#: was typed, when the nine-keyword ``terminus`` built it.
_ANSWERING = ControlContext(
    target="va",
    generation=2,
    writer=RecordWriter(kind="web_terminal", pid=1234, port=8087),
    switch={"standin": "asked", "live": "allowed"},
    switch_epoch=1700000000000000001,
)
_SERVER_WRITTEN = ControlContext(
    target="live", generation=5, writer=RecordWriter(kind="controls_server", pid=77)
)
CONTEXT_ANSWERS = {
    "context_applied": _ANSWERING.answered(
        SwitchTerminus.applied(
            request_id="r-applied",
            target="live",
            generation=3,
            requested_at=datetime(
                2026, 1, 1, 9, 30, 15, 125_000, tzinfo=timezone(timedelta(hours=9.5))
            ),
            requested_by="sess-1",
        )
    ),
    "context_unchanged": _ANSWERING.answered(
        SwitchTerminus.unchanged(
            request_id="r-unchanged",
            target="va",
            generation=2,
            requested_at=T0,
            requested_by="pid:4242",
        )
    ),
    "context_refused": _SERVER_WRITTEN.answered(
        SwitchTerminus.refused(
            request_id="r-refused",
            target="standin",
            reason="target_unconfigured",
            detail="The stand-in is not configured.",
            requested_at=None,
            requested_by="pid:9",
        )
    ),
    "context_refused_blank_reason": _SERVER_WRITTEN.answered(
        SwitchTerminus.refused(
            request_id="r-blank",
            target="va",
            reason="",
            detail="",
            requested_at=T0 + timedelta(seconds=1),
            requested_by="pid:9",
        )
    ),
}

#: A controls server's report at each step of a switch, and a display
#: process's at each machine statement. Captured from the writers before the
#: blocks were typed, when they were the publishers' own dicts.
_AT = T0 + timedelta(seconds=5, microseconds=250)
_PROGRESS = {
    "applying": SwitchProgress.applying(3, at=_AT, expires_at=_AT + timedelta(seconds=30)),
    "applying_unbounded": SwitchProgress.applying(3, at=_AT, expires_at=None),
    "applied": SwitchProgress.applied(3, at=_AT),
    "failed": SwitchProgress.failed(
        3, reason="spawn_failed", detail="The child did not answer.", at=_AT
    ),
    "failed_bare": SwitchProgress.failed(4, reason=None, detail=None, at=_AT),
}
REPORTS = {
    f"report_{name}": ServerReport(
        server_pid=4321,
        session="s-1",
        applied_target="va",
        applied_generation=3,
        children=(11,),
        reachability={
            "published_at": _AT.isoformat(),
            "targets": {"va": {"epics": {"state": "reached"}}},
        },
        switch_progress=progress,
        last_posture_realign={"state": "done", "at": _AT.isoformat()},
        targets={"va": {"label": "VA", "endpoint": "e", "real_machine": False}},
        updated_at=T0 + timedelta(seconds=6),
    )
    for name, progress in _PROGRESS.items()
}
for _name, _writes, _reach in (
    (
        "display_allowed",
        WritesStatement(allowed=True, position="asked", reason=None, kind="real"),
        Reachable(state="yes"),
    ),
    (
        "display_refused",
        WritesStatement(allowed=False, position=None, reason="switch_off", kind="rehearsal"),
        Reachable(state="no", since=T0 + timedelta(seconds=7, microseconds=1)),
    ),
    ("display_unstated", None, None),
):
    REPORTS[f"report_{_name}"] = ServerReport(
        server_pid=4322,
        applied_target="live",
        applied_generation=1,
        updated_at=T0 + timedelta(seconds=8),
        statement=MachineStatement(reachable=_reach, writes=_writes),
    )
GRANT = Grant(
    id=ID,
    owner="alice",
    holder="bob",
    target="live",
    switch_epoch=5,
    # Built in this process, an ``int`` bound or lease is kept as given.
    channels={"SR:B:SP": ChannelRange(min=0, max=10.5), "SR:A:SP": ChannelRange(min=-1.5, max=2)},
    steps=3,
    issued_at=T0 + timedelta(seconds=100),
    expires_at=T0 + timedelta(seconds=200),
    lease_s=60,
    approver=Approver(by="alice", surface="chip", approved_at=T0 + timedelta(seconds=100)),
    label="run-1",
    revoked=Revocation(revoked_at=T0 + timedelta(seconds=150, microseconds=250_000), by="alice"),
)
LEDGER = Ledger(
    authorization=ID,
    holder_pid=42,
    holder_start="777",
    pid_ns="pid:[4026531836]",
    lease_expires_at=T0 + timedelta(seconds=300),
    spent=(
        LedgerStep(n=1, spent_at=T0 + timedelta(seconds=110), channels=("SR:A:SP",)),
        LedgerStep(
            n=2,
            spent_at=T0 + timedelta(seconds=120, microseconds=5),
            channels=("SR:A:SP", "SR:B:SP"),
        ),
    ),
)
REQUEST = WriteRequest(
    id=ID,
    owner="alice",
    holder="bob",
    target="live",
    channels={"SR:A:SP": ChannelRange(min=0.0, max=1.0)},
    steps=4,
    requested_at=T0 + timedelta(seconds=90),
    expires_at=T0 + timedelta(seconds=200),
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
    "target": "live",
    "requested_at": null,
    "requested_by": "sess-1",
    "status": "applied",
    "reason": null,
    "detail": "The control target is now 'live' (generation 3).",
    "generation": 3
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
  "target": "va",
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
    "generation": 7,
    "status": "applying",
    "at": "2026-01-01T00:00:00+00:00",
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
  "updated_at": "2026-01-01T00:00:00.500000+00:00",
  "statement": {
    "reachable": {
      "state": "yes",
      "since": "2026-01-01T00:00:00.250000+00:00",
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
    "context_terminus": (
        CONTEXT_TERMINUS,
        b"""{
  "schema": 1,
  "owner": null,
  "target": "live",
  "generation": 1,
  "posture": {
    "live": "sandbox",
    "va": "sandbox",
    "standin": "sandbox"
  },
  "last_switch": {
    "request_id": "r-2",
    "target": "live",
    "requested_at": "2026-01-01T00:00:15.125000+00:00",
    "requested_by": "pid:4242",
    "status": "applied",
    "reason": null,
    "detail": "",
    "generation": 1
  },
  "switch": {},
  "switch_epoch": null
}""",
    ),
    "context_applied": (
        CONTEXT_ANSWERS["context_applied"],
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
    "request_id": "r-applied",
    "target": "live",
    "requested_at": "2026-01-01T00:00:15.125000+00:00",
    "requested_by": "sess-1",
    "status": "applied",
    "reason": null,
    "detail": "The control target is now 'live' (generation 3).",
    "generation": 3
  },
  "switch": {
    "standin": "asked",
    "live": "allowed"
  },
  "switch_epoch": 1700000000000000001
}""",
    ),
    "context_unchanged": (
        CONTEXT_ANSWERS["context_unchanged"],
        b"""{
  "schema": 1,
  "owner": {
    "kind": "web_terminal",
    "pid": 1234,
    "port": 8087
  },
  "target": "va",
  "generation": 2,
  "posture": {
    "va": "sandbox"
  },
  "last_switch": {
    "request_id": "r-unchanged",
    "target": "va",
    "requested_at": "2026-01-01T00:00:00+00:00",
    "requested_by": "pid:4242",
    "status": "applied",
    "reason": null,
    "detail": "The control target is already 'va' (generation 2).",
    "generation": 2
  },
  "switch": {
    "standin": "asked",
    "live": "allowed"
  },
  "switch_epoch": 1700000000000000001
}""",
    ),
    "context_refused": (
        CONTEXT_ANSWERS["context_refused"],
        b"""{
  "schema": 1,
  "owner": {
    "kind": "controls_server",
    "pid": 77,
    "port": null
  },
  "target": "live",
  "generation": 5,
  "posture": {
    "live": "sandbox",
    "va": "sandbox",
    "standin": "sandbox"
  },
  "last_switch": {
    "request_id": "r-refused",
    "target": "standin",
    "requested_at": null,
    "requested_by": "pid:9",
    "status": "refused",
    "reason": "target_unconfigured",
    "detail": "The stand-in is not configured.",
    "generation": null
  },
  "switch": {},
  "switch_epoch": null
}""",
    ),
    "context_refused_blank_reason": (
        CONTEXT_ANSWERS["context_refused_blank_reason"],
        b"""{
  "schema": 1,
  "owner": {
    "kind": "controls_server",
    "pid": 77,
    "port": null
  },
  "target": "live",
  "generation": 5,
  "posture": {
    "live": "sandbox",
    "va": "sandbox",
    "standin": "sandbox"
  },
  "last_switch": {
    "request_id": "r-blank",
    "target": "va",
    "requested_at": "2026-01-01T00:00:01+00:00",
    "requested_by": "pid:9",
    "status": "refused",
    "reason": "",
    "detail": "",
    "generation": null
  },
  "switch": {},
  "switch_epoch": null
}""",
    ),
    "report_applying": (
        REPORTS["report_applying"],
        b"""{
  "server_pid": 4321,
  "session": "s-1",
  "applied_target": "va",
  "applied_generation": 3,
  "children": [
    11
  ],
  "reachability": {
    "published_at": "2026-01-01T00:00:05.000250+00:00",
    "targets": {
      "va": {
        "epics": {
          "state": "reached"
        }
      }
    }
  },
  "last_switch": {
    "generation": 3,
    "status": "applying",
    "at": "2026-01-01T00:00:05.000250+00:00",
    "expires_at": "2026-01-01T00:00:35.000250+00:00"
  },
  "last_posture_realign": {
    "state": "done",
    "at": "2026-01-01T00:00:05.000250+00:00"
  },
  "targets": {
    "va": {
      "label": "VA",
      "endpoint": "e",
      "real_machine": false
    }
  },
  "updated_at": "2026-01-01T00:00:06+00:00"
}""",
    ),
    "report_applying_unbounded": (
        REPORTS["report_applying_unbounded"],
        b"""{
  "server_pid": 4321,
  "session": "s-1",
  "applied_target": "va",
  "applied_generation": 3,
  "children": [
    11
  ],
  "reachability": {
    "published_at": "2026-01-01T00:00:05.000250+00:00",
    "targets": {
      "va": {
        "epics": {
          "state": "reached"
        }
      }
    }
  },
  "last_switch": {
    "generation": 3,
    "status": "applying",
    "at": "2026-01-01T00:00:05.000250+00:00"
  },
  "last_posture_realign": {
    "state": "done",
    "at": "2026-01-01T00:00:05.000250+00:00"
  },
  "targets": {
    "va": {
      "label": "VA",
      "endpoint": "e",
      "real_machine": false
    }
  },
  "updated_at": "2026-01-01T00:00:06+00:00"
}""",
    ),
    "report_applied": (
        REPORTS["report_applied"],
        b"""{
  "server_pid": 4321,
  "session": "s-1",
  "applied_target": "va",
  "applied_generation": 3,
  "children": [
    11
  ],
  "reachability": {
    "published_at": "2026-01-01T00:00:05.000250+00:00",
    "targets": {
      "va": {
        "epics": {
          "state": "reached"
        }
      }
    }
  },
  "last_switch": {
    "generation": 3,
    "status": "applied",
    "at": "2026-01-01T00:00:05.000250+00:00"
  },
  "last_posture_realign": {
    "state": "done",
    "at": "2026-01-01T00:00:05.000250+00:00"
  },
  "targets": {
    "va": {
      "label": "VA",
      "endpoint": "e",
      "real_machine": false
    }
  },
  "updated_at": "2026-01-01T00:00:06+00:00"
}""",
    ),
    "report_failed": (
        REPORTS["report_failed"],
        b"""{
  "server_pid": 4321,
  "session": "s-1",
  "applied_target": "va",
  "applied_generation": 3,
  "children": [
    11
  ],
  "reachability": {
    "published_at": "2026-01-01T00:00:05.000250+00:00",
    "targets": {
      "va": {
        "epics": {
          "state": "reached"
        }
      }
    }
  },
  "last_switch": {
    "generation": 3,
    "status": "failed",
    "reason": "spawn_failed",
    "detail": "The child did not answer.",
    "at": "2026-01-01T00:00:05.000250+00:00"
  },
  "last_posture_realign": {
    "state": "done",
    "at": "2026-01-01T00:00:05.000250+00:00"
  },
  "targets": {
    "va": {
      "label": "VA",
      "endpoint": "e",
      "real_machine": false
    }
  },
  "updated_at": "2026-01-01T00:00:06+00:00"
}""",
    ),
    "report_failed_bare": (
        REPORTS["report_failed_bare"],
        b"""{
  "server_pid": 4321,
  "session": "s-1",
  "applied_target": "va",
  "applied_generation": 3,
  "children": [
    11
  ],
  "reachability": {
    "published_at": "2026-01-01T00:00:05.000250+00:00",
    "targets": {
      "va": {
        "epics": {
          "state": "reached"
        }
      }
    }
  },
  "last_switch": {
    "generation": 4,
    "status": "failed",
    "reason": null,
    "detail": null,
    "at": "2026-01-01T00:00:05.000250+00:00"
  },
  "last_posture_realign": {
    "state": "done",
    "at": "2026-01-01T00:00:05.000250+00:00"
  },
  "targets": {
    "va": {
      "label": "VA",
      "endpoint": "e",
      "real_machine": false
    }
  },
  "updated_at": "2026-01-01T00:00:06+00:00"
}""",
    ),
    "report_display_allowed": (
        REPORTS["report_display_allowed"],
        b"""{
  "server_pid": 4322,
  "session": null,
  "applied_target": "live",
  "applied_generation": 1,
  "children": [],
  "reachability": {},
  "last_switch": null,
  "last_posture_realign": null,
  "targets": {},
  "updated_at": "2026-01-01T00:00:08+00:00",
  "statement": {
    "reachable": {
      "state": "yes",
      "since": null,
      "reason": null
    },
    "writes": {
      "allowed": true,
      "position": "asked",
      "reason": null,
      "kind": "real"
    }
  }
}""",
    ),
    "report_display_refused": (
        REPORTS["report_display_refused"],
        b"""{
  "server_pid": 4322,
  "session": null,
  "applied_target": "live",
  "applied_generation": 1,
  "children": [],
  "reachability": {},
  "last_switch": null,
  "last_posture_realign": null,
  "targets": {},
  "updated_at": "2026-01-01T00:00:08+00:00",
  "statement": {
    "reachable": {
      "state": "no",
      "since": "2026-01-01T00:00:07.000001+00:00",
      "reason": null
    },
    "writes": {
      "allowed": false,
      "position": null,
      "reason": "switch_off",
      "kind": "rehearsal"
    }
  }
}""",
    ),
    "report_display_unstated": (
        REPORTS["report_display_unstated"],
        b"""{
  "server_pid": 4322,
  "session": null,
  "applied_target": "live",
  "applied_generation": 1,
  "children": [],
  "reachability": {},
  "last_switch": null,
  "last_posture_realign": null,
  "targets": {},
  "updated_at": "2026-01-01T00:00:08+00:00",
  "statement": {
    "reachable": null,
    "writes": null
  }
}""",
    ),
}

AUTHORIZATION_FILES: dict[str, tuple[BaseModel, bytes]] = {
    "grant": (
        GRANT,
        b"""{
  "approver": {
    "approved_at": "2026-01-01T00:01:40+00:00",
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
  "expires_at": "2026-01-01T00:03:20+00:00",
  "holder": "bob",
  "id": "0123456789abcdef0123456789abcdef",
  "issued_at": "2026-01-01T00:01:40+00:00",
  "label": "run-1",
  "lease_s": 60,
  "owner": "alice",
  "revoked": {
    "by": "alice",
    "revoked_at": "2026-01-01T00:02:30.250000+00:00"
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
  "lease_expires_at": "2026-01-01T00:05:00+00:00",
  "pid_ns": "pid:[4026531836]",
  "schema": 1,
  "steps": [
    {
      "channels": [
        "SR:A:SP"
      ],
      "n": 1,
      "spent_at": "2026-01-01T00:01:50+00:00"
    },
    {
      "channels": [
        "SR:A:SP",
        "SR:B:SP"
      ],
      "n": 2,
      "spent_at": "2026-01-01T00:02:00.000005+00:00"
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
  "expires_at": "2026-01-01T00:03:20+00:00",
  "holder": "bob",
  "id": "0123456789abcdef0123456789abcdef",
  "label": "scan",
  "lease_s": 30.5,
  "owner": "alice",
  "requested_at": "2026-01-01T00:01:30+00:00",
  "schema": 1,
  "steps": 4,
  "target": "live"
}
""",
    ),
}

#: The bytes the writers stored before instants were aware datetimes: every
#: stamp an integer of nanoseconds under an ``_ns`` name. Copied from the pins
#: this module held then; no reader of this version reads them.
PREVIOUS_AUTHORIZATION_FILES: dict[str, bytes] = {
    "grant": b"""{
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
    "ledger": b"""{
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
    "request": b"""{
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
    assert (record.writer, record.switch, record.last_switch) == (
        CONTEXT_FULL.writer,
        CONTEXT_FULL.switch,
        CONTEXT_FULL.last_switch,
    )
    answered = control_context.parse_record(STATE_DIRECTORY_FILES["context_terminus"][1])
    assert answered is not None
    assert answered.last_switch == CONTEXT_TERMINUS.last_switch
    assert answered.last_switch is not None
    assert answered.last_switch.requested_at == datetime(2026, 1, 1, 0, 0, 15, 125_000, tzinfo=UTC)
    for name, (model, stored) in STATE_DIRECTORY_FILES.items():
        if name.startswith("context_"):
            assert control_context.parse_record(stored) == model, name
        else:
            assert control_context._parse_report(json.loads(stored)) == model, name


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


# -- a file in the previous format -------------------------------------------

_PARSERS = {"grant": wa._parse_grant, "ledger": wa._parse_ledger, "request": wa._parse_request}

#: A previous stamp's name for each instant field.
_PREVIOUS_NAMES = {
    "issued_at": "issued_at_ns",
    "expires_at": "expires_at_ns",
    "requested_at": "requested_at_ns",
    "approved_at": "at_ns",
    "revoked_at": "at_ns",
    "spent_at": "at_ns",
    "lease_expires_at": "lease_until_ns",
}

_EPOCH = datetime(1970, 1, 1, tzinfo=UTC)


def _previous_format(value: Any) -> Any:
    """A record's JSON form as the previous writer stored it: each instant ns under its old name."""
    if isinstance(value, dict):
        return {
            _PREVIOUS_NAMES.get(key, key): (
                (parse_instant(item) - _EPOCH) // timedelta(microseconds=1) * 1000
                if key in _PREVIOUS_NAMES
                else _previous_format(item)
            )
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_previous_format(item) for item in value]
    return value


def _shape(value: Any) -> Any:
    """*value* with every leaf replaced by its type: the keys and nesting a file has."""
    if isinstance(value, dict):
        return {key: _shape(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_shape(item) for item in value]
    return type(value).__name__


@pytest.mark.parametrize("name", list(PREVIOUS_AUTHORIZATION_FILES))
def test_the_previous_format_of_a_pinned_record_has_the_previous_pins_shape(name: str) -> None:
    """The conversion the end-to-end cases use is the previous writer's, key for key."""
    record, _ = AUTHORIZATION_FILES[name]
    payload = _previous_format(record.model_dump(mode="json"))
    assert _shape(payload) == _shape(json.loads(PREVIOUS_AUTHORIZATION_FILES[name]))


@pytest.mark.parametrize("name", list(PREVIOUS_AUTHORIZATION_FILES))
def test_a_file_in_the_previous_format_does_not_read(name: str) -> None:
    with pytest.raises(ValueError):
        _PARSERS[name](PREVIOUS_AUTHORIZATION_FILES[name])


SECTION = {"type": "epics", "writes": "allowed"}
OPS = [("A:SP", 5.0)]


@pytest.fixture
def host(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A host with this login's record (every switch off, an epoch) and no grant."""
    data = tmp_path / "root"
    isolate_verdict_environment(monkeypatch, data)
    for name in (wa.AUTHORIZATION_ENV_VAR, wa.LEDGER_DIR_ENV_VAR, "OSPREY_SESSION_ID"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.delenv(session_hold.HOLD_DIR_ENV_VAR, raising=False)
    write_control_context(data, switch={})
    return data


def _approved() -> Grant:
    """A grant this login approved for itself, issued and its lease started."""
    testing.invalidate_cache()
    record = control_context.read_record()
    assert record is not None and isinstance(record.switch_epoch, int)
    now = datetime.now(UTC)
    grant = Grant(
        id=uuid.uuid4().hex,
        owner=FIXTURE_IDENTITY,
        holder=FIXTURE_IDENTITY,
        target="va",
        switch_epoch=record.switch_epoch,
        channels={"A:SP": ChannelRange(min=0.0, max=10.0)},
        steps=3,
        issued_at=now,
        expires_at=now + timedelta(hours=12),
        lease_s=60.0,
        approver=Approver(by="alice", surface="chip", approved_at=now),
        label="run-1",
    )
    wa._issue_grant(grant)
    wa.start_lease(grant.id)
    return grant


def _file_of(name: str, grant_id: str) -> Path:
    if name == "grant":
        directory = wa.authorizations_dir()
        suffix = ".json"
    else:
        directory = wa.ledger_dir()
        suffix = ".ledger.json"
    assert directory is not None
    return directory / f"{grant_id}{suffix}"


@pytest.mark.usefixtures("host")
@pytest.mark.parametrize("name", ["grant", "ledger"])
def test_a_write_over_a_file_in_the_previous_format_refuses_unreadable(name: str) -> None:
    """The same grant and lease allow; the file rewritten as the previous writer did refuses."""
    grant = _approved()
    assert (
        wa._verdict(ControlSystemSection.read(SECTION), "va", OPS, grant.id, None, None).allowed
        is True
    )

    path = _file_of(name, grant.id)
    previous = _previous_format(json.loads(path.read_bytes()))
    path.write_text(json.dumps(previous, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    before = path.read_bytes()

    answer = wa._verdict(ControlSystemSection.read(SECTION), "va", OPS, grant.id, None, None)
    assert (answer.allowed, answer.reason) == (False, "auth_unreadable")
    spent = wa._spend(ControlSystemSection.read(SECTION), "va", OPS, grant.id, None, None)
    assert (spent.allowed, spent.reason) == (False, "auth_unreadable")
    assert path.read_bytes() == before


def test_a_request_in_the_previous_format_is_listed_as_unreadable(tmp_path: Path) -> None:
    directory = tmp_path / "requests"
    directory.mkdir()
    old = directory / f"{ID}.json"
    old.write_bytes(PREVIOUS_AUTHORIZATION_FILES["request"])
    current = evolve(REQUEST, id="ab" * 16)
    wa._write_json(directory / f"{current.id}.json", current.model_dump(mode="json"))

    listing = wa.list_requests(directory)

    assert [request.id for request in listing.items] == [current.id]
    assert listing.unreadable == (old,)


def _load_hook_twin(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    """The shipped stdlib hook twin, by path; its ``sys.path`` insert is undone."""
    monkeypatch.setattr(sys, "path", list(sys.path))
    hook = REPO_ROOT / "src/osprey/templates/claude_code/claude/hooks/osprey_target_state.py"
    spec = importlib.util.spec_from_file_location("_stored_bytes_target_state", hook)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


HOLD_KEY = "3f2b8c1e-6a0d-4c55-9a1e-0b7f2d4e9c11"  # gitleaks:allow - a test session id


def test_a_hold_record_in_the_previous_format_holds_on_both_readers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The answered synthetic flag reads as released with ``null``; with the previous ``""``, held."""
    monkeypatch.delenv(session_hold.HOLD_DIR_ENV_VAR, raising=False)
    twin = _load_hook_twin(monkeypatch)
    root = tmp_path / "session_holds"
    path = root / HOLD_KEY / session_hold.RECORD_NAME
    path.parent.mkdir(parents=True)
    path.write_text("{not json", encoding="utf-8")
    assert session_hold.answer_flags(
        HOLD_KEY, [session_hold.UNREADABLE_FLAG_ID], answer="continue", operator="op", root=root
    ) == (session_hold.UNREADABLE_FLAG_ID,)
    payload = json.loads(path.read_text(encoding="utf-8"))
    [entry] = payload["flags"]
    assert (entry["flag_id"], entry["state"], entry["opened_at"]) == (
        session_hold.UNREADABLE_FLAG_ID,
        "closed",
        None,
    )
    assert entry["answered_at"].endswith("+00:00")
    assert session_hold.is_held(HOLD_KEY, root=root) is False
    assert twin.is_held(HOLD_KEY, root=root) is False

    # As the previous writer stored it: an empty ``opened_at`` and a ``Z`` stamp.
    entry.update(opened_at="", answered_at=entry["answered_at"].replace("+00:00", "Z"))
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    assert session_hold.is_held(HOLD_KEY, root=root) is True
    assert twin.is_held(HOLD_KEY, root=root) is True
