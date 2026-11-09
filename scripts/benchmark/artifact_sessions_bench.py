#!/usr/bin/env python3
"""Artifact-session latency: what an operator waits for on a large store.

Seeds a throwaway project in two sizes and boots the hermetic web-terminal stack
(:func:`docs.screenshots.contact_sheet.hermetic_hub`) over each:

``small``
    150 transcripts, one per key.
``large``
    300 transcripts across 150 keys — every key was ``/clear``-ed once, so it
    points at its second transcript through ``session-transcripts.json``.

Each seed holds at least 6,000 artifacts covering every registered artifact
type, runs with parts (approval preview, execute outputs, notebook, result
record), untagged entries and dispatch entries. A small writer-made subset goes
through the store's real writers — keyword arguments are filtered by
``inspect.signature``, so the same seed runs on a tree whose writers take more
or fewer arguments — and the bulk is copied from it in one index write.

Three events are measured, identically on any tree:

1. ``first_paint`` — the current session's artifacts visible: the listing's
   first page scoped to the current transcript (``cold`` is the first request
   after boot; ``median_ms``/``p90_ms`` over warm repeats).
2. ``scope_change`` — the first page of a non-active session, i.e. the listing
   after a filter change to another session.
3. ``execute_writes`` — wall time ``build_execution_response`` spends inside
   artifact writes for one stubbed ``execute`` call (two figures, two
   subprocess artifacts, notebook, result record), over at least 20 runs.

Beside them, ``sessions_list`` times ``GET /api/sessions`` on the hub (the
sidebar's session list); it is reported, never compared.

The JSON records, per seed, the row counts the seed produced and the timings.

Usage::

    uv run python scripts/benchmark/artifact_sessions_bench.py \\
        --out .claude/scratch/perf/baseline.json
    uv run python scripts/benchmark/artifact_sessions_bench.py --quick \\
        --out .claude/scratch/perf/after-quick.json \\
        --compare .claude/scratch/perf/baseline.json

``--quick`` uses fewer HTTP repeats (the execute event keeps its 20-run floor)
and writes nothing but its ``--out``.

Exit status: 0 when the run completed (and, with ``--compare``, nothing
regressed); 1 when ``--compare`` found a regression of events 1–3; 2 when the
``--compare`` baseline is missing, unreadable, or lacks either seed.
"""

from __future__ import annotations

import argparse
import asyncio
import copy
import inspect
import json
import os
import statistics
import subprocess
import sys
import time
import urllib.parse
import urllib.request
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_OUT = REPO_ROOT / ".claude" / "scratch" / "perf" / "baseline.json"

#: Fixed namespace and clock so two runs seed byte-identical stores.
_NS = uuid.UUID("6b1f0c9e-2a47-4d1b-9e35-5a0f7c3d2e81")
_BASE_TIME = datetime(2026, 10, 1, 8, 0, 0, tzinfo=UTC)

MIN_ARTIFACTS = 6000
SEEDS: dict[str, dict[str, int]] = {
    "small": {"keys": 150, "transcripts_per_key": 1},
    "large": {"keys": 150, "transcripts_per_key": 2},
}
EVENTS = ("first_paint", "scope_change", "execute_writes")
EXEC_RUNS_MIN = 20


def _uid(*parts: object) -> str:
    return str(uuid.uuid5(_NS, "/".join(str(p) for p in parts)))


def _hex12(*parts: object) -> str:
    return uuid.uuid5(_NS, "/".join(str(p) for p in parts)).hex[:12]


def _call(method, **kwargs):
    """Call a store writer with only the keyword arguments it accepts."""
    params = inspect.signature(method).parameters
    if not any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
        kwargs = {k: v for k, v in kwargs.items() if k in params}
    return method(**kwargs)


def _stats(samples: list[float]) -> dict[str, float]:
    ordered = sorted(samples)
    p90 = (
        statistics.quantiles(ordered, n=10, method="inclusive")[-1]
        if len(ordered) > 1
        else ordered[0]
    )
    return {
        "n": len(ordered),
        "median_ms": round(statistics.median(ordered), 3),
        "p90_ms": round(p90, 3),
        "min_ms": round(ordered[0], 3),
        "max_ms": round(ordered[-1], 3),
    }


# ---------------------------------------------------------------------------
# Seeding
# ---------------------------------------------------------------------------


class Plan:
    """The transcripts and keys one seed writes, and who is current."""

    def __init__(self, name: str, keys: int, transcripts_per_key: int) -> None:
        self.name = name
        self.keys: list[str] = []
        self.key_map: dict[str, str] = {}
        self.transcripts: list[str] = []
        for k in range(keys):
            chain = [_uid(name, "transcript", k, t) for t in range(transcripts_per_key)]
            self.keys.append(chain[0])
            self.transcripts.extend(chain)
            if len(chain) > 1:
                self.key_map[chain[0]] = chain[-1]
        # The newest key, at its latest transcript, is the active session.
        self.current = self.key_map.get(self.keys[-1], self.keys[-1])
        self.others = [t for t in self.transcripts if t != self.current]


def _writer_groups(store) -> list[tuple[str, list[dict]]]:
    """Make one of each entry shape through the real writers; return dict groups.

    Each group is copied as a unit by the bulk step. Sessions come from
    ``OSPREY_SESSION_ID`` (writers that take no ``session_id``) or the kwarg;
    dispatch entries carry ``OSPREY_DISPATCH_RUN_ID`` and no session.
    """
    from osprey.stores.artifact_store import RUN_RECORD_KEY
    from osprey.stores.type_registry import get_artifact_types

    tagged = "seed-session"
    env_tagged = {"OSPREY_SESSION_ID": tagged, "OSPREY_DISPATCH_RUN_ID": ""}
    groups: list[tuple[str, list[Any]]] = []

    with mock.patch.dict(os.environ, env_tagged):
        # A run with parts: approval preview, execute outputs, notebook, result.
        run = [
            _call(
                store.save_file,
                file_content=b'{"cells": [], "nbformat": 4, "nbformat_minor": 5}',
                filename="approval_preview.ipynb",
                artifact_type="notebook",
                title="Pending approval: orbit correction",
                mime_type="application/x-ipynb+json",
                tool_source="osprey_approval",
                metadata={"exec_mode": "readwrite", RUN_RECORD_KEY: True},
                session_id=tagged,
            ),
            _call(
                store.save_file,
                file_content=b"\x89PNG\r\n\x1a\n" + b"\0" * 256,
                filename="orbit.png",
                artifact_type="image",
                title="Figure: orbit",
                mime_type="image/png",
                tool_source="execute",
                session_id=tagged,
            ),
            _call(
                store.save_file,
                file_content=b"<html><body>plot</body></html>",
                filename="orbit.html",
                artifact_type="plot_html",
                title="Orbit plot",
                mime_type="text/html",
                tool_source="execute",
                category="visualization",
                session_id=tagged,
            ),
            _call(
                store.save_file,
                file_content=b'{"cells": [], "nbformat": 4, "nbformat_minor": 5}',
                filename="orbit_correction.ipynb",
                artifact_type="notebook",
                title="Notebook: orbit correction",
                mime_type="application/x-ipynb+json",
                tool_source="execute",
                metadata={RUN_RECORD_KEY: True},
                session_id=tagged,
            ),
            _call(
                store.save_data,
                tool="execute",
                data={"stdout": "ok", "has_errors": False},
                title="orbit correction",
                description="orbit correction",
                summary={"status": "Success"},
                category="code_output",
                metadata={RUN_RECORD_KEY: True},
                session_id=tagged,
            ),
        ]
        groups.append(("run", run))

        # Every registered artifact type once.
        types = [
            _call(
                store.save_file,
                file_content=f"{key} content".encode(),
                filename=f"sample_{key}.bin",
                artifact_type=key,
                title=f"Sample {typedef.label}",
                tool_source="execute",
                session_id=tagged,
            )
            for key, typedef in get_artifact_types().items()
        ]
        groups.append(("types", types))

        data = [
            _call(
                store.save_data,
                tool="channel_read",
                data={"SR:DCCT": 500.1},
                title="Beam current reading",
                category="channel_values",
                session_id=tagged,
            ),
            _call(
                store.save_object,
                obj={"rms_orbit_um": 42},
                title="Orbit stats",
                session_id=tagged,
            ),
        ]
        groups.append(("data", data))

        untagged = [
            _call(
                store.save_file,
                file_content=b"# untagged note\n",
                filename="untagged.md",
                artifact_type="markdown",
                title="Untagged note",
                mime_type="text/markdown",
                tool_source="artifact_register",
                session_id="",
            )
        ]
        groups.append(("untagged", untagged))

    dispatch_run = "seed-dispatch-run"
    with mock.patch.dict(
        os.environ, {"OSPREY_SESSION_ID": "", "OSPREY_DISPATCH_RUN_ID": dispatch_run}
    ):
        dispatch = [
            _call(
                store.save_file,
                file_content=b"input bytes",
                filename="input.png",
                artifact_type="image",
                title="Input image",
                mime_type="image/png",
                tool_source="input",
                run_id=dispatch_run,
                origin="input",
                session_id="",
            ),
            _call(
                store.save_file,
                file_content=b"<html>dispatch out</html>",
                filename="dispatch_out.html",
                artifact_type="html",
                title="Dispatch output",
                mime_type="text/html",
                tool_source="execute",
                session_id="",
            ),
        ]
        groups.append(("dispatch", dispatch))

    index = json.loads(_index_file(store.artifact_dir).read_text(encoding="utf-8"))
    by_id = {e["id"]: e for e in index["entries"]}
    return [
        (name, [copy.deepcopy(by_id.get(e.id, e.to_dict())) for e in entries])
        for name, entries in groups
    ]


def _index_file(artifacts_dir: Path) -> Path:
    return artifacts_dir / "artifacts.json"


def _seed_artifacts(project_dir: Path, plan: Plan) -> dict[str, Any]:
    """Writer-made subset, then the bulk in one index write; return row counts."""
    from osprey.stores.artifact_store import ArtifactStore, is_run_record

    template_root = project_dir / "var" / "bench-template"
    template_root.mkdir(parents=True, exist_ok=True)
    store = ArtifactStore(template_root, auto_launch=False)
    groups = _writer_groups(store)

    target_dir = project_dir / "artifacts"
    target_dir.mkdir(parents=True, exist_ok=True)
    for path in store.artifact_dir.iterdir():
        if path.is_file() and path.name != "artifacts.json":
            (target_dir / path.name).write_bytes(path.read_bytes())

    template_envelope = json.loads(_index_file(store.artifact_dir).read_text())
    existing = json.loads(_index_file(target_dir).read_text())["entries"]

    entries: list[dict] = list(existing)
    tagged_groups = [g for g in groups if g[0] not in ("dispatch", "untagged")]
    ordered = tagged_groups * 6 + [g for g in groups if g[0] in ("dispatch", "untagged")]
    copy_n = 0
    tick = 0
    while len(entries) < MIN_ARTIFACTS:
        name, group = ordered[copy_n % len(ordered)]
        session = plan.transcripts[copy_n % len(plan.transcripts)]
        dispatch_run = f"dispatch-{plan.name}-{copy_n}"
        for template in group:
            entry = copy.deepcopy(template)
            entry["id"] = _hex12(plan.name, copy_n, template["id"])
            entry["timestamp"] = (_BASE_TIME + timedelta(seconds=tick)).isoformat()
            tick += 1
            if name == "dispatch":
                entry["run_id"] = dispatch_run
                entry["session_id"] = ""
            elif name == "untagged":
                entry["session_id"] = ""
            else:
                entry["session_id"] = session
            entries.append(entry)
        copy_n += 1

    # The current session's newest work is the newest thing in the store.
    for name, group in groups:
        if name != "run":
            continue
        for template in group:
            entry = copy.deepcopy(template)
            entry["id"] = _hex12(plan.name, "current", template["id"])
            entry["timestamp"] = (_BASE_TIME + timedelta(seconds=tick)).isoformat()
            entry["session_id"] = plan.current
            tick += 1
            entries.append(entry)

    envelope = {k: v for k, v in template_envelope.items() if k != "entries"}
    envelope.update(
        {"updated": (_BASE_TIME + timedelta(seconds=tick)).isoformat(), "entry_count": len(entries)}
    )
    envelope["entries"] = entries
    _index_file(target_dir).write_text(json.dumps(envelope), encoding="utf-8")

    # Counts from the store's own loader, so they mean the same on any tree.
    loaded = ArtifactStore(project_dir, auto_launch=False).list_entries()
    by_type: dict[str, int] = {}
    for e in loaded:
        by_type[e.artifact_type] = by_type.get(e.artifact_type, 0) + 1
    return {
        "artifacts": len(loaded),
        "writer_made": sum(len(g) for _, g in groups),
        "by_type": dict(sorted(by_type.items())),
        "untagged": sum(1 for e in loaded if not e.session_id and not e.run_id),
        "dispatch": sum(1 for e in loaded if e.run_id),
        "run_records": sum(1 for e in loaded if is_run_record(e)),
        "current_session_artifacts": sum(1 for e in loaded if e.session_id == plan.current),
    }


def _seed_transcripts(session_dir: Path, plan: Plan) -> dict[str, Any]:
    from osprey.interfaces.web_terminal import transcript_map

    session_dir.mkdir(parents=True, exist_ok=True)
    for i, tid in enumerate(plan.transcripts):
        line = {
            "type": "user",
            "message": {"role": "user", "content": f"Check the orbit, pass {i}"},
            "sessionId": tid,
            "timestamp": (_BASE_TIME + timedelta(minutes=i)).isoformat(),
        }
        (session_dir / f"{tid}.jsonl").write_text(json.dumps(line) + "\n", encoding="utf-8")
    map_path = transcript_map.store_path()
    if plan.key_map and map_path is not None:
        map_path.parent.mkdir(parents=True, exist_ok=True)
        map_path.write_text(json.dumps(plan.key_map, indent=2), encoding="utf-8")
    return {
        "transcripts": len(plan.transcripts),
        "keys": len(plan.keys),
        "clear_splits": len(plan.key_map),
        "transcript_map_written": bool(plan.key_map and map_path is not None),
    }


# ---------------------------------------------------------------------------
# Events
# ---------------------------------------------------------------------------


class _CookieJar:
    """Stands in for a browser context: collects the operator session cookie."""

    def __init__(self) -> None:
        self.header = ""

    def add_cookies(self, cookies: list[dict]) -> None:
        self.header = "; ".join(f"{c['name']}={c['value']}" for c in cookies)


_AUTH = _CookieJar()


def _authorize() -> None:
    """Mint an in-process operator session, as the browser suites do."""
    from osprey.interfaces._serving import authorize_browser_context

    authorize_browser_context(_AUTH)


def _get_ms(url: str) -> tuple[float, dict]:
    request = urllib.request.Request(url, headers={"Cookie": _AUTH.header})
    start = time.perf_counter()
    with urllib.request.urlopen(request, timeout=60) as resp:
        body = resp.read()
    elapsed = (time.perf_counter() - start) * 1000
    return elapsed, json.loads(body)


def _listing_url(artifact_url: str, session_id: str) -> str:
    query = urllib.parse.urlencode({"run": "false", "session_id": session_id})
    return f"{artifact_url}/api/artifacts?{query}"


def _event_first_paint(artifact_url: str, plan: Plan, reps: int) -> dict[str, Any]:
    url = _listing_url(artifact_url, plan.current)
    cold, page = _get_ms(url)
    warm = [_get_ms(url)[0] for _ in range(reps)]
    return {
        "cold_ms": round(cold, 3),
        **_stats(warm),
        "page_count": page.get("count"),
        "page_total": page.get("total"),
    }


def _event_scope_change(artifact_url: str, plan: Plan, reps: int) -> dict[str, Any]:
    step = max(1, len(plan.others) // reps)
    targets = [plan.others[(i * step) % len(plan.others)] for i in range(reps)]
    samples: list[float] = []
    counts: list[int] = []
    for target in targets:
        ms, page = _get_ms(_listing_url(artifact_url, target))
        samples.append(ms)
        counts.append(page.get("count") or 0)
    return {**_stats(samples), "min_page_count": min(counts)}


def _sessions_list(base_url: str, reps: int) -> dict[str, Any]:
    """``GET /api/sessions`` on the hub: the sidebar's session list, first page.

    Reported beside the events, never gated: the baseline tree has no such
    reader to compare against. ``cold_ms`` is the ``drain=1`` call that waits
    for the record to ingest every seeded transcript; the warm repeats do not
    drain.
    """
    url = f"{base_url}/api/sessions"
    cold, page = _get_ms(f"{url}?drain=1")
    warm = [_get_ms(url)[0] for _ in range(reps)]
    return {
        "cold_ms": round(cold, 3),
        **_stats(warm),
        "page_rows": len(page.get("sessions") or []),
        "more": bool(page.get("more")),
    }


def _event_execute_writes(project_dir: Path, plan: Plan, runs: int) -> dict[str, Any]:
    from osprey.mcp_server.python_executor.executor import ExecutionResult
    from osprey.mcp_server.python_executor.tools._response_builder import (
        build_execution_response,
    )
    from osprey.stores import artifact_store as store_mod

    store = store_mod.ArtifactStore(project_dir, auto_launch=False)
    spent = [0.0]
    calls = [0]

    def timed(fn):
        def wrapper(*args, **kwargs):
            start = time.perf_counter()
            calls[0] += 1
            try:
                return fn(*args, **kwargs)
            finally:
                spent[0] += time.perf_counter() - start

        return wrapper

    for name, member in inspect.getmembers(store, inspect.ismethod):
        if name.startswith(("save_", "record_")):
            setattr(store, name, timed(member))

    write_ms: list[float] = []
    total_ms: list[float] = []
    write_calls: list[int] = []
    work = project_dir / "var" / "bench-exec"
    previous = store_mod._artifact_store
    store_mod._artifact_store = store
    try:
        with mock.patch.dict(
            os.environ, {"OSPREY_SESSION_ID": plan.current, "OSPREY_DISPATCH_RUN_ID": ""}
        ):
            for i in range(runs):
                run_dir = work / f"run-{i}"
                run_dir.mkdir(parents=True, exist_ok=True)
                figures = []
                for f in range(2):
                    p = run_dir / f"figure_{f}.png"
                    p.write_bytes(b"\x89PNG\r\n\x1a\n" + bytes([i % 256, f]) * 512)
                    figures.append(p)
                artifacts = []
                for a in range(2):
                    p = run_dir / f"table_{a}.html"
                    p.write_bytes(f"<table><tr><td>{i}-{a}</td></tr></table>".encode())
                    artifacts.append(
                        {
                            "path": p,
                            "artifact_type": "table_html",
                            "title": f"Table {a}",
                            "description": "stub table",
                            "mime_type": "text/html",
                            "category": "visualization",
                        }
                    )
                result = ExecutionResult(
                    success=True,
                    stdout=f"run {i} ok\n",
                    stderr="",
                    figures=figures,
                    artifacts=artifacts,
                )
                spent[0] = 0.0
                calls[0] = 0
                start = time.perf_counter()
                asyncio.run(
                    build_execution_response(
                        code="print('ok')",
                        description=f"bench run {i}",
                        execution_mode="readonly",
                        exec_result=result,
                        patterns={},
                        save_output=True,
                    )
                )
                total_ms.append((time.perf_counter() - start) * 1000)
                write_ms.append(spent[0] * 1000)
                write_calls.append(calls[0])
    finally:
        store_mod._artifact_store = previous
    stats = _stats(write_ms)
    stats["builder_total_median_ms"] = round(statistics.median(total_ms), 3)
    stats["writes_per_run"] = max(write_calls)
    return stats


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def run_seed(name: str, *, http_reps: int, exec_runs: int) -> dict[str, Any]:
    from docs.screenshots.contact_sheet import hermetic_hub

    spec = SEEDS[name]
    plan = Plan(name, spec["keys"], spec["transcripts_per_key"])
    rows: dict[str, Any] = {}

    def seed(project_dir: Path, session_dir: Path) -> None:
        rows.update(_seed_transcripts(session_dir, plan))
        rows.update(_seed_artifacts(project_dir, plan))

    with hermetic_hub(seed=seed) as hub:
        _authorize()
        events = {
            "first_paint": _event_first_paint(hub.artifact_url, plan, http_reps),
            "scope_change": _event_scope_change(hub.artifact_url, plan, http_reps),
            "execute_writes": _event_execute_writes(hub.project_dir, plan, exec_runs),
        }
        sessions = _sessions_list(hub.base_url, http_reps)
    return {"seed": name, "rows": rows, "events": events, "sessions_list": sessions}


def _git_head() -> str:
    try:
        return subprocess.run(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except Exception:
        return ""


def compare(current: dict, baseline_path: Path, tolerance: float, slack_ms: float) -> int:
    """Print the comparison; 0 nothing regressed, 1 regression, 2 bad baseline."""
    try:
        baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
        base_seeds = baseline["seeds"]
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"compare: baseline {baseline_path} unusable: {exc}", file=sys.stderr)
        return 2
    missing = [s for s in SEEDS if s not in base_seeds]
    if missing:
        print(f"compare: baseline lacks seed(s) {missing}", file=sys.stderr)
        return 2

    regressions: list[str] = []
    for seed_name in SEEDS:
        for event in EVENTS:
            metrics = ("median_ms", "p90_ms") if event == "execute_writes" else ("median_ms",)
            for metric in metrics:
                try:
                    before = float(base_seeds[seed_name]["events"][event][metric])
                except (KeyError, TypeError, ValueError):
                    print(f"compare: baseline lacks {seed_name}.{event}.{metric}", file=sys.stderr)
                    return 2
                after = float(current["seeds"][seed_name]["events"][event][metric])
                bad = after > before * (1 + tolerance) and after - before > slack_ms
                if bad:
                    mark = "REGRESSION"
                elif event != "execute_writes" and after >= before:
                    mark = "ok (not lower)"
                else:
                    mark = "ok"
                print(
                    f"{seed_name:6} {event:15} {metric:9} {before:10.3f} -> {after:10.3f}  {mark}"
                )
                if bad:
                    regressions.append(f"{seed_name}.{event}.{metric}")
    if regressions:
        print(f"compare: regressed: {', '.join(regressions)}", file=sys.stderr)
        return 1
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--out", type=Path, default=None, help=f"JSON result (default {DEFAULT_OUT})"
    )
    parser.add_argument("--quick", action="store_true", help="fewer HTTP repeats; requires --out")
    parser.add_argument("--compare", type=Path, default=None, help="baseline JSON to gate against")
    parser.add_argument(
        "--tolerance", type=float, default=0.25, help="relative slack (0.25 = +25%%)"
    )
    parser.add_argument("--slack-ms", type=float, default=5.0, help="absolute slack in ms")
    parser.add_argument("--seeds", nargs="+", choices=sorted(SEEDS), default=list(SEEDS))
    args = parser.parse_args(argv)

    if args.quick and args.out is None:
        parser.error("--quick needs --out")
    out = args.out or DEFAULT_OUT
    if args.compare is not None and not args.compare.is_file():
        print(f"compare: baseline {args.compare} is missing", file=sys.stderr)
        return 2

    http_reps = 5 if args.quick else 30
    exec_runs = EXEC_RUNS_MIN if args.quick else 40

    started = datetime.now(UTC).isoformat()
    seeds = {name: run_seed(name, http_reps=http_reps, exec_runs=exec_runs) for name in args.seeds}
    result = {
        "script": "scripts/benchmark/artifact_sessions_bench.py",
        "git_head": _git_head(),
        "started": started,
        "quick": args.quick,
        "http_reps": http_reps,
        "exec_runs": exec_runs,
        "seeds": seeds,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    for name, seed in seeds.items():
        ev = seed["events"]
        print(
            f"{name}: {seed['rows'].get('artifacts')} artifacts, "
            f"{seed['rows'].get('transcripts')} transcripts / {seed['rows'].get('keys')} keys | "
            f"first_paint {ev['first_paint']['median_ms']} ms, "
            f"scope_change {ev['scope_change']['median_ms']} ms, "
            f"execute_writes {ev['execute_writes']['median_ms']} / "
            f"p90 {ev['execute_writes']['p90_ms']} ms"
        )
        sl = seed.get("sessions_list")
        if sl:
            print(
                f"{name}: GET /api/sessions {sl['median_ms']} / p90 {sl['p90_ms']} ms "
                f"({sl['page_rows']} rows on the first page; not gated)"
            )
    print(f"wrote {out}")

    if args.compare is not None:
        return compare(result, args.compare, args.tolerance, args.slack_ms)
    return 0


if __name__ == "__main__":
    sys.exit(main())
