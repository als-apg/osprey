"""The contract pages' examples are read by the code each page describes.

Every JSON document, refusal text and Python snippet on the pages under
``docs/source/reference/contracts/`` that shows a shape the code produces or
reads is a file under ``docs/source/reference/contracts/examples/``, shown on
its page with ``literalinclude``. Each file is checked against the code that
owns the shape:

* ``health-report.json`` is rebuilt by ``CheckReport.to_dict()`` from its own
  rows, so its counts, summary line and key order are the ones a run writes,
  and its keys are the recorded ``health --json`` key set. The page's ``jq``
  recipes run over it when ``jq`` is installed;
* the three channel databases pass the checks behind
  ``osprey channel-finder validate``, run as that command runs them;
* the watchdog hold record has the keys the session-hold writer writes and is
  read by its reader as a held session, and the refusal text is the one a
  write to that session gets;
* the bridge dispatch payload has the keys the bridge's payload assembler and
  webhook client send;
* the limits fragment loads through the connectors' limits loader with
  confirmation off;
* the ARIEL service-factory snippet leaves the shared service open after a
  search and closes it at shutdown.
"""

from __future__ import annotations

import asyncio
import base64
import importlib.util
import io
import json
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

import httpx
import pytest
from rich.console import Console

REPO_ROOT = Path(__file__).resolve().parents[2]
CONTRACTS = REPO_ROOT / "docs/source/reference/contracts"
EXAMPLES = CONTRACTS / "examples"

_LITERALINCLUDE = re.compile(r"^[ \t]*\.\. literalinclude:: (?P<path>\S+)", re.MULTILINE)


def _example(name: str) -> Any:
    return json.loads((EXAMPLES / name).read_text())


# --- every example is on a page, every include is a file ---------------------------


def _included_examples() -> dict[str, str]:
    """Each ``examples/`` file a contract page includes, mapped to the page."""
    included: dict[str, str] = {}
    for page in sorted(CONTRACTS.glob("*.rst")):
        for match in _LITERALINCLUDE.finditer(page.read_text()):
            path = match.group("path")
            if path.startswith("examples/"):
                included[path.removeprefix("examples/")] = page.name
    return included


def test_every_example_file_is_on_a_page_and_every_include_exists() -> None:
    included = _included_examples()
    on_disk = {path.name for path in EXAMPLES.iterdir() if path.is_file()}
    assert not set(included) - on_disk, f"included but missing: {set(included) - on_disk}"
    assert not on_disk - set(included), f"on disk but on no page: {on_disk - set(included)}"


# --- health-json.rst -----------------------------------------------------------------


def _rebuilt_report(example: dict[str, Any]) -> dict[str, Any]:
    from osprey.health.models import CheckReport, CheckResult, Status

    rows = [CheckResult(**{**row, "status": Status(row["status"])}) for row in example["results"]]
    report = CheckReport(
        results=rows, elapsed_ms=example["elapsed_ms"], deadline_hit=example["deadline_hit"]
    )
    return report.to_dict()


def test_health_report_is_what_check_report_writes() -> None:
    example = _example("health-report.json")
    rebuilt = _rebuilt_report(example)
    assert example == rebuilt
    assert list(example) == list(rebuilt), "the envelope's keys are out of order"


def test_health_report_keys_are_the_recorded_keyset() -> None:
    recorded = json.loads((REPO_ROOT / "tests/cli/data/json_keysets/health.json").read_text())
    assert sorted(_example("health-report.json")) == recorded


def _jq_programs(page: str) -> list[tuple[list[str], str]]:
    """Every ``jq [flags] '<program>'`` the page spells, in order."""
    pattern = re.compile(r"\bjq((?:\s+-\w+)*)\s+'([^']*)'")
    return [(match.group(1).split(), match.group(2)) for match in pattern.finditer(page)]


#: What each jq program on health-json.rst prints over ``health-report.json``,
#: and its exit code. ``-e`` exits 1 on a ``false`` result.
_JQ_EXPECTED: dict[str, tuple[str, int]] = {
    ".summary": ("2/5 checks passed (1 warning, 1 error, 1 skipped)\n", 0),
    '.results[] | .latency_ms // "n/a"': ("n/a\n12.4\nn/a\nn/a\nn/a\n", 0),
    ".errors == 0": ("false\n", 1),
    '.results[] | select(.status == "error" or .status == "warning") '
    '| "\\(.status)\\t\\(.category)/\\(.name)\\t\\(.message)"': (
        "warning\tarchive/archive_last_pass\tArchive unreadable: [Errno 13] Permission denied: "
        "'/srv/osprey/my-agent/var/archive/2026-10-31/MANIFEST.jsonl'\n"
        "error\tconfiguration/config_file_exists\t"
        "config.yml not found at /srv/osprey/my-agent/config.yml\n",
        0,
    ),
    '.results[] | select(.name == "config_file_exists") | .details // "(none)"': (
        "Searched from: /srv/osprey/my-agent\n"
        "Run this from inside a deployment repository, or pass --project with the "
        "directory holding config.yml.\n",
        0,
    ),
    ".deadline_hit | not": ("true\n", 0),
}


def _normalized(program: str) -> str:
    return " ".join(program.split())


def test_health_page_jq_programs_are_the_tested_ones() -> None:
    page = (CONTRACTS / "health-json.rst").read_text()
    spelled = {_normalized(program) for _, program in _jq_programs(page)}
    assert spelled == {_normalized(program) for program in _JQ_EXPECTED}


@pytest.mark.skipif(shutil.which("jq") is None, reason="jq is not installed")
def test_health_page_jq_recipes_run_over_the_example() -> None:
    page = (CONTRACTS / "health-json.rst").read_text()
    expected = {_normalized(program): result for program, result in _JQ_EXPECTED.items()}
    report = (EXAMPLES / "health-report.json").read_text()
    for flags, program in _jq_programs(page):
        run = subprocess.run(
            ["jq", *flags, program], input=report, capture_output=True, text=True, check=False
        )
        want_stdout, want_code = expected[_normalized(program)]
        assert (run.stdout, run.returncode) == (want_stdout, want_code), (program, run.stderr)


# --- channel-finder.rst --------------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "pipeline"),
    [
        ("channel-finder-in-context.json", "in_context"),
        ("channel-finder-hierarchical.json", "hierarchical"),
        ("channel-finder-middle-layer.json", "middle_layer"),
    ],
)
def test_channel_database_passes_channel_finder_validate(name: str, pipeline: str) -> None:
    from osprey.cli.styles import osprey_theme
    from osprey.services.channel_finder.tools.validate_database import run_validation

    output = io.StringIO()
    code = run_validation(
        database=str(EXAMPLES / name),
        pipeline=pipeline,
        console=Console(file=output, width=200, theme=osprey_theme),
    )
    assert code == 0, output.getvalue()
    assert "No issues found" in output.getvalue(), output.getvalue()


def test_in_context_example_count_is_its_row_count() -> None:
    document = _example("channel-finder-in-context.json")
    assert document["count"] == len(document["channels"])


def test_hierarchical_example_holds_the_channels_the_page_names() -> None:
    from osprey.services.channel_finder.databases import HierarchicalChannelDatabase

    database = HierarchicalChannelDatabase(str(EXAMPLES / "channel-finder-hierarchical.json"))
    channels = [row["channel"] for row in database.get_all_channels()]
    assert channels == ["SR:BPM[01]:X:RB", "SR:BPM[02]:X:RB"]
    page = (CONTRACTS / "channel-finder.rst").read_text()
    assert all(f"``{channel}``" in page for channel in channels)


# --- watchdog.rst --------------------------------------------------------------------

_SESSION_KEY = "session-1"


def _store_example_record(root: Path) -> None:
    """The page's hold record, under a real session key, in a hold store at *root*."""
    record = _example("watchdog-hold-record.json")
    record["session_key"] = _SESSION_KEY
    directory = root / _SESSION_KEY
    directory.mkdir(parents=True)
    (directory / "record.json").write_text(json.dumps(record))


def test_hold_record_has_the_keys_the_writer_writes(tmp_path: Path) -> None:
    from osprey_connectors import session_hold

    assert session_hold.open_flag(
        _SESSION_KEY,
        flag_id="f1",
        source="guard-circumvention",
        reason="r",
        prompt_ref="s:1",
        prompt_excerpt="e",
        root=tmp_path,
    )
    written = json.loads((tmp_path / _SESSION_KEY / session_hold.RECORD_NAME).read_text())
    example = _example("watchdog-hold-record.json")
    assert sorted(example) == sorted(written)
    assert example["schema"] == session_hold._SCHEMA_VERSION
    assert sorted(example["flags"][0]) == sorted(written["flags"][0])


def test_hold_record_is_read_as_a_held_session(tmp_path: Path) -> None:
    from osprey_connectors import session_hold

    _store_example_record(tmp_path)
    assert session_hold.is_held(_SESSION_KEY, root=tmp_path)
    [flag] = session_hold.open_flags(_SESSION_KEY, root=tmp_path)
    assert flag.flag_id == "<id>"
    assert flag.source == "guard-circumvention"


def test_session_held_text_is_the_refusal_a_write_gets(tmp_path: Path) -> None:
    from osprey_connectors.posture_store import session_hold_refusal
    from osprey_connectors.refusals import refusal_text

    _store_example_record(tmp_path)
    verdict = session_hold_refusal(root=tmp_path, session_keys=[_SESSION_KEY])
    assert verdict is not None and verdict.reason == "session_held"
    text = refusal_text(verdict.reason, verdict.kind, detail=verdict.detail)
    assert (EXAMPLES / "watchdog-session-held.txt").read_text() == text + "\n"


# --- bridge-dispatch.rst -------------------------------------------------------------


def _shape(value: Any) -> Any:
    """*value*'s key structure: a dict's keys (each shaped), a list's first item."""
    if isinstance(value, dict):
        return {key: _shape(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_shape(value[0])] if value else []
    return None


def _sent_payload(tmp_path: Path) -> dict[str, Any]:
    """The webhook body a bridge sends for a reply, with history, files and a skip.

    The payload assembler (``build_extra``) runs on real stores and the test
    doubles the bridge tests use; the real ``DispatchClient.fire`` posts the
    question and that context to a mock transport, which keeps the body.
    """
    from osprey.bridges.core import pipeline
    from osprey.bridges.core.config import CoreConfig
    from osprey.bridges.core.dedup import DedupStore
    from osprey.bridges.core.dispatch_client import DispatchClient
    from osprey.bridges.core.ports import InputDownload, RoomMember, RoomPeople
    from tests.bridges.conftest import RosterChannelOps
    from tests.bridges.record_double import MemoryRecord, RecordHistory

    cfg = CoreConfig(dedup_path=str(tmp_path / "dedup.json"), history_answer_limit=100)
    history = RecordHistory(MemoryRecord())
    history.seed(
        "room/A",
        "show the vacuum trace",
        "an earlier answer longer than the replay limit " * 20,
        run_id="run-0",
        asked_by={"id": "users/222", "name": "Carol"},
    )
    image = base64.b64encode(b"png").decode("ascii")
    ops = RosterChannelOps(
        room_people=RoomPeople(
            members=(RoomMember("users/111", "Alice"), RoomMember("users/333")), mentions=True
        ),
        inputs=InputDownload(
            files=[
                {"filename": "trace.png", "mime": "image/png", "content_b64": image, "ingest": True}
            ],
            skipped=[{"filename": "scan.h5", "reason": "too large"}],
        ),
    )
    bodies: list[dict[str, Any]] = []

    def webhook(request: httpx.Request) -> httpx.Response:
        bodies.append(json.loads(request.content))
        return httpx.Response(202, json={"dispatch_id": "d-1"})

    client = DispatchClient(cfg, httpx.Client(transport=httpx.MockTransport(webhook)))
    deps = pipeline.PipelineDeps(
        cfg=cfg,
        ops=ops,
        dedup=DedupStore(cfg.dedup_path),
        dispatcher=client,
        history=history,
        probe_capability=lambda cfg, capability: True,
    )
    entry = {"sender_id": "users/111", "sender_display": "Alice", "history_key": "room/A"}
    extra = pipeline.build_extra(
        deps,
        entry,
        history_key="room/A",
        reply_to={"sender": "Alice", "text": "the vacuum trace from this morning"},
    )
    client.fire("now plot that over 24 hours", extra)
    [body] = bodies
    return body


def test_dispatch_payload_has_the_keys_the_bridge_sends(tmp_path: Path) -> None:
    sent = _sent_payload(tmp_path)
    assert _shape(_example("bridge-dispatch-payload.json")) == _shape(sent)


def test_reply_to_example_is_the_payloads_reply_to() -> None:
    assert (
        _example("bridge-dispatch-reply-to.json")
        == (_example("bridge-dispatch-payload.json")["reply_to"])
    )


# --- connectors.rst ------------------------------------------------------------------


def test_limits_fragment_loads_with_confirmation_off(tmp_path: Path) -> None:
    from osprey_connectors.control_system._limits_validator import LimitsValidator

    limits, raw = LimitsValidator._load_limits_database(str(EXAMPLES / "limits-confirm-off.json"))
    validator = LimitsValidator(limits, {}, raw)
    assert limits["SHUTTER:OPEN_CMD"].writable is True
    assert validator.resolve_confirm("SHUTTER:OPEN_CMD") is False
    assert validator.resolve_confirm("SHUTTER:UNLISTED") is True

    with_defaults = tmp_path / "limits.json"
    with_defaults.write_text(json.dumps({**raw, "defaults": {"writable": True}}))
    with pytest.raises(ValueError, match="'defaults' is not allowed"):
        LimitsValidator._load_limits_database(str(with_defaults))


# --- ariel.rst -----------------------------------------------------------------------


class _Service:
    """Stands in for the ARIEL service the factory hands out."""

    def __init__(self) -> None:
        self.closed = False
        self.queries: list[str] = []

    async def search(self, query: str) -> str:
        assert not self.closed, "searched a closed service"
        self.queries.append(query)
        return "result"

    async def aclose(self) -> None:
        self.closed = True


def test_ariel_factory_snippet_keeps_the_singleton_open(monkeypatch: pytest.MonkeyPatch) -> None:
    from osprey.services.ariel_search import capability

    spec = importlib.util.spec_from_file_location(
        "ariel_service_factory", EXAMPLES / "ariel-service-factory.py"
    )
    assert spec is not None and spec.loader is not None
    snippet = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(snippet)

    service = _Service()
    monkeypatch.setattr(capability, "_ariel_service_instance", service)

    async def scenario() -> None:
        assert await snippet.search_logbook() == "result"
        assert await snippet.search_logbook() == "result"
        assert not service.closed
        await snippet.shutdown()

    asyncio.run(scenario())
    assert service.queries == ["RF cavity fault", "RF cavity fault"]
    assert service.closed
    assert capability._ariel_service_instance is None
