"""The gateway's ledger feed: which directories it reads, and what it keeps.

The receivers, ``transform/ledger`` and ``filter/ledger-path`` as rendered;
the regexes themselves are pinned against real writer lines in
``test_ledger_regex.py``, and the whole chain runs in the pinned Collector in
``test_telemetry_gateway_validate.py``.
"""

from __future__ import annotations

import copy
import hashlib
import logging
import os
import re
import threading
import time
from datetime import UTC, datetime
from pathlib import Path

import pytest
import yaml

from osprey.deployment import telemetry_config as tc
from osprey.deployment import telemetry_gateway as gw
from osprey.deployment.telemetry_store import ddl

from .web_terminals.test_golden_render import EXAMPLE_CONFIG


def _config(*, roster: bool = True, bluesky: bool = False, workers: int = 0) -> dict:
    config: dict = copy.deepcopy(EXAMPLE_CONFIG) if roster else {"project_name": "p"}
    if not roster:
        config["modules"] = {}
    deployed = ["telemetry_store", "telemetry_gateway"]
    services: dict = {"telemetry_store": {}}
    if workers:
        deployed.append("dispatch_worker")
        services["dispatch_worker"] = {"worker_count": workers}
    if bluesky:
        deployed += ["bluesky", "bluesky_web"]
    config["deployed_services"] = deployed
    config["services"] = services
    config["deployment"] = {"host_identity": "builder"}
    return config


def _transform(config: dict) -> list[str]:
    rendered = gw.collector_config(config, ledger_receivers=gw.ledger_receivers(config, None))
    (group,) = rendered["processors"][gw.LEDGER_TRANSFORM]["log_statements"]
    assert group["context"] == "log"
    return group["statements"]


# --- the receiver set ---------------------------------------------------------


def test_the_configured_set_names_every_writer() -> None:
    config = _config(bluesky=True, workers=2)
    assert gw.configured_ledger_identities(config) == [
        "alice",
        "bob",
        "dispatch-worker-1",
        "dispatch-worker-2",
        "bluesky-web",
        "queueserver",
        "builder",
    ]


def test_no_gateway_no_receivers(tmp_path: Path) -> None:
    config = _config()
    config["deployed_services"] = ["telemetry_store"]
    (tmp_path / "alice").mkdir()
    assert gw.ledger_receivers(config, tmp_path) == {}


def test_the_rendered_set_is_configured_union_existing_with_no_catch_all(tmp_path: Path) -> None:
    config = _config()
    for name in ("alice", "carol", "dispatch-worker-9"):
        (tmp_path / name).mkdir()
    (tmp_path / "stray.jsonl").write_text("")  # a file at the root is no directory

    receivers = gw.ledger_receivers(config, tmp_path)

    expected = {"alice", "bob", "builder", "carol", "dispatch-worker-9"}
    assert set(receivers) == {gw.ledger_receiver_id(name) for name in expected}
    includes = [include for body in receivers.values() for include in body["include"]]
    assert sorted(includes) == sorted(f"/audit/{name}/*.jsonl" for name in expected)
    assert not [include for include in includes if include.startswith("/audit/*")]


def test_removing_a_roster_user_keeps_its_receiver_id(tmp_path: Path) -> None:
    config = _config()
    (tmp_path / "bob").mkdir()
    before = gw.ledger_receivers(config, tmp_path)

    config["modules"]["web_terminals"]["users"] = ["alice"]
    assert "bob" not in gw.configured_ledger_identities(config)
    after = gw.ledger_receivers(config, tmp_path)

    bob = gw.ledger_receiver_id("bob")
    assert bob in after
    assert after[bob] == before[bob]
    assert set(after) == set(before)


def test_the_receiver_id_depends_on_the_name_alone() -> None:
    assert (
        gw.ledger_receiver_id("alice") == f"file_log/d{hashlib.sha256(b'alice').hexdigest()[:12]}"
    )
    assert re.fullmatch(r"file_log/d[0-9a-f]{12}", gw.ledger_receiver_id("ops team"))


def test_a_directory_outside_the_charset_is_skipped_with_a_warning(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    for name in ("ops team", "a+b", "EXPR(x)", "dave"):
        (tmp_path / name).mkdir()
    with caplog.at_level(logging.WARNING, logger=gw.__name__):
        names = gw.ledger_identities(_config(roster=False), tmp_path)
    assert names == ["builder", "dave"]
    for name in ("ops team", "a+b", "EXPR(x)"):
        assert repr(name) in caplog.text


@pytest.mark.parametrize(
    ("name", "include", "literal"),
    [
        ("ops team", "/audit/ops team/*.jsonl", '"ops team"'),
        ("a+b", "/audit/a+b/*.jsonl", '"a+b"'),
        ("EXPR(x)", "/audit/EXPR(x)/*.jsonl", '"EXPR(x)"'),
        ("w*[1]{x}?", r"/audit/w\*\[1\]\{x\}\?/*.jsonl", '"w*[1]{x}?"'),
    ],
)
def test_any_name_is_escaped_in_the_glob_and_quoted_in_add(
    name: str, include: str, literal: str
) -> None:
    receiver = gw.ledger_receiver({}, name)
    assert receiver["include"] == [include]
    (add,) = [op for op in receiver["operators"] if op["type"] == "add"]
    assert add["field"] == 'resource["osprey.user"]'
    assert add["value"] == f"EXPR({literal})"


def test_each_receiver_reads_whole_lines_from_the_beginning() -> None:
    receiver = gw.ledger_receiver({}, "alice")
    assert receiver["start_at"] == "beginning"
    assert receiver["storage"] == gw.STORAGE
    assert receiver["include_file_path"] is True
    assert receiver["include_file_path_resolved"] is True
    assert receiver["max_concurrent_files"] == 2
    assert receiver["max_log_size"] == tc.gateway_max_log_size({}) == 1024 * 1024
    assert receiver["max_log_size_behavior"] == "truncate"
    assert receiver["on_truncate"] == "read_whole_file"
    assert receiver["force_flush_period"] == "60s"
    assert receiver["retry_on_failure"] == {
        "enabled": True,
        "initial_interval": "1s",
        "max_interval": "30s",
        "max_elapsed_time": 0,
    }
    assert [op["id"] for op in receiver["operators"]] == [
        "ledger_anchor",
        "ledger_route",
        "ledger_tool_call",
        "ledger_envelope",
        "ledger_time",
        "ledger_identity",
    ]
    assert "json_parser" not in {op["type"] for op in receiver["operators"]}
    for op in receiver["operators"]:
        if op["type"] in ("regex_parser", "time_parser"):
            assert op["on_error"] == "send_quiet"


def test_the_line_cap_follows_the_inline_payload_cap() -> None:
    config = {"audit": {"tool_call": {"max_inline_bytes": 3 * 1024 * 1024}}}
    assert gw.ledger_receiver(config, "alice")["max_log_size"] == 7 * 1024 * 1024


# --- the pipeline -------------------------------------------------------------


def test_the_ledger_pipeline_runs_transform_path_filter_then_clamp() -> None:
    config = _config()
    rendered = gw.collector_config(config, ledger_receivers=gw.ledger_receivers(config, None))
    pipeline = rendered["service"]["pipelines"][gw.LEDGER_PIPELINE]
    assert pipeline["processors"] == ["transform/ledger", "filter/ledger-path", "filter/clamp"]
    assert pipeline["receivers"] == list(gw.ledger_receivers(config, None))
    assert "memory_limiter" not in pipeline["processors"]


def test_the_path_filter_is_one_pinned_condition() -> None:
    config = _config()
    rendered = gw.collector_config(config, ledger_receivers=gw.ledger_receivers(config, None))
    assert rendered["processors"][gw.LEDGER_PATH_FILTER] == {
        "error_mode": "ignore",
        "logs": {
            "log_record": [
                'log.attributes["log.file.path_resolved"] != log.attributes["log.file.path"] '
                'or log.attributes["log.file.path_resolved"] == nil'
            ]
        },
    }


def test_the_transform_scrubs_first_and_drops_its_scratch_keys_last() -> None:
    statements = _transform(_config())
    assert statements[0] == 'delete_matching_keys(log.attributes, "^osprey\\\\.")'
    assert statements[-1] == 'delete_matching_keys(log.attributes, "^ledger_")'
    assert 'set(log.attributes["osprey.user"], resource.attributes["osprey.user"])' in statements
    assert "set(log.time, log.observed_time) where log.time_unix_nano == 0" in statements
    unparsed_time = next(s for s in statements if "osprey.audit.time_unparsed" in s)
    assert statements.index(unparsed_time) < statements.index(
        "set(log.time, log.observed_time) where log.time_unix_nano == 0"
    )


def test_the_markers_follow_the_truncated_and_incomplete_rule() -> None:
    statements = _transform(_config())
    cap = tc.gateway_max_log_size({})
    truncated = next(s for s in statements if '"osprey.audit.truncated"' in s)
    assert truncated.endswith(f"where Len(log.body) >= {cap}")
    incomplete = next(s for s in statements if '"osprey.audit.incomplete"' in s)
    assert 'log.attributes["ledger_surface"] != nil' in incomplete
    assert 'not IsMatch(log.body, "\\\\}$")' in incomplete
    assert f"Len(log.body) < {cap}" in incomplete
    unparsed = next(s for s in statements if '"osprey.audit.unparsed"' in s)
    assert unparsed.endswith('where not (log.attributes["ledger_surface"] != nil)')


def _trust(statements: list[str]) -> str:
    return next(
        s
        for s in statements
        if s.startswith('delete_key(log.attributes, "osprey.audit.on_behalf_of")')
    )


def test_on_behalf_of_is_kept_from_the_sidecar_only_without_a_queueserver() -> None:
    assert _trust(_transform(_config())) == (
        'delete_key(log.attributes, "osprey.audit.on_behalf_of") '
        'where not (resource.attributes["osprey.user"] == "sidecar")'
    )


def test_a_bluesky_deployment_trusts_its_queueserver_on_preflight_only() -> None:
    config = _config(bluesky=True)
    assert gw.trusted_on_behalf_of(config) == ("sidecar", ("queueserver",))
    trust = _trust(_transform(config))
    assert trust == (
        'delete_key(log.attributes, "osprey.audit.on_behalf_of") '
        'where not (resource.attributes["osprey.user"] == "sidecar" or '
        '((resource.attributes["osprey.user"] == "queueserver") and '
        'log.attributes["ledger_surface"] == "preflight"))'
    )
    # Exact names, never a pattern; the web app is not a trusted writer.
    assert "bluesky-web" not in trust
    assert "IsMatch" not in trust


# --- the column contract ----------------------------------------------------------

_AUDIT_KEY = re.compile(r"LogAttributes\['(osprey\.audit\.[a-z_]+)'\]")


def test_every_key_the_transform_sets_is_read_by_exactly_one_column() -> None:
    statements = _transform(_config(bluesky=True))
    set_keys = {
        match
        for statement in statements
        if statement.startswith("set(")
        for match in re.findall(r'^set\(log\.attributes\["(osprey\.audit\.[a-z_]+)"\]', statement)
    }
    assert set_keys == set(gw.ledger_transform_keys())

    readers: dict[str, list[str]] = {}
    for column in ddl.AUDIT_LEDGER.columns:
        if " MATERIALIZED " not in column.definition:
            continue
        for key in _AUDIT_KEY.findall(column.definition):
            readers.setdefault(key, []).append(column.name)

    assert set(readers) == set_keys, "a column reads a key the transform never sets, or not"
    for key, columns in readers.items():
        assert len(columns) == 1, (key, columns)


# --- hostile files ------------------------------------------------------------------


def test_a_fifo_and_a_symlink_are_moved_aside(tmp_path: Path) -> None:
    (tmp_path / "alice").mkdir()
    (tmp_path / "alice" / "tool_call.jsonl").write_text("{}\n")
    os.mkfifo(tmp_path / "alice" / ".hidden.jsonl")
    (tmp_path / "eve").mkdir()
    os.symlink("/dev/zero", tmp_path / "eve" / "x.jsonl")
    os.symlink(tmp_path / "alice", tmp_path / "link-dir")  # not walked into

    moved = gw.quarantine_hostile_ledgers(tmp_path, now=datetime(2026, 10, 11, 12, 0, tzinfo=UTC))

    assert [(m.path.name, m.moved_to.name) for m in moved] == [
        (".hidden.jsonl", ".hidden.jsonl.refused-20261011T120000Z"),
        ("x.jsonl", "x.jsonl.refused-20261011T120000Z"),
    ]
    assert (tmp_path / "alice" / "tool_call.jsonl").is_file()
    assert sorted(p.name for p in (tmp_path / "alice").iterdir()) == [
        ".hidden.jsonl.refused-20261011T120000Z",
        "tool_call.jsonl",
    ]
    assert (tmp_path / "eve" / "x.jsonl.refused-20261011T120000Z").is_symlink()
    assert gw.quarantine_hostile_ledgers(tmp_path) == []


def test_a_fifo_with_a_blocked_reader_is_released(tmp_path: Path) -> None:
    (tmp_path / "alice").mkdir()
    fifo = tmp_path / "alice" / "p.jsonl"
    os.mkfifo(fifo)
    seen: list[bytes] = []

    def reader() -> None:
        with open(fifo, "rb") as handle:  # blocks in open() until a writer comes
            seen.append(handle.read())

    thread = threading.Thread(target=reader, daemon=True)
    thread.start()
    for _ in range(100):
        if thread.is_alive():
            break
        time.sleep(0.01)
    time.sleep(0.05)
    gw.quarantine_hostile_ledgers(tmp_path)
    thread.join(timeout=5)
    assert not thread.is_alive()
    assert seen == [b""]


def test_an_unrenameable_path_refuses_naming_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "alice").mkdir()
    os.symlink("/etc/hostname", tmp_path / "alice" / "x.jsonl")

    def refuse(*_: str) -> None:
        raise PermissionError(13, "Permission denied")

    monkeypatch.setattr(gw.os, "rename", refuse)
    with pytest.raises(RuntimeError, match=re.escape(str(tmp_path / "alice" / "x.jsonl"))):
        gw.quarantine_hostile_ledgers(tmp_path)


# --- osprey up ------------------------------------------------------------------------


def test_up_quarantines_and_lists_the_unfed_directories(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from osprey.deployment import container_lifecycle as cl

    config = _config(roster=False)
    audit = tmp_path / "var" / "audit"
    (audit / "builder").mkdir(parents=True)
    rendered = gw.render_collector_config(
        config, ledger_receivers=gw.ledger_receivers(config, audit)
    )
    built = tmp_path / "build" / "services" / tc.GATEWAY_SERVICE / gw.COLLECTOR_CONFIG_FILE
    built.parent.mkdir(parents=True)
    built.write_text(rendered)
    (audit / "late").mkdir()  # created after the render
    os.mkfifo(audit / "late" / "x.jsonl")

    facts: list[str] = []
    warnings: list[str] = []
    monkeypatch.setattr(cl, "_report_fact", lambda message, **_: facts.append(message))
    monkeypatch.setattr(cl, "_warn_fact", lambda *words: warnings.append(" ".join(words)))

    cl._guard_ledger_feed(config, tmp_path)

    assert facts == ["Audit directories not fed until the next build: late"]
    assert len(warnings) == 1 and str(audit / "late" / "x.jsonl") in warnings[0]
    assert not (audit / "late" / "x.jsonl").exists()
    assert yaml.safe_load(built.read_text())  # untouched


def test_up_without_the_gateway_touches_nothing(tmp_path: Path) -> None:
    from osprey.deployment import container_lifecycle as cl

    config = _config(roster=False)
    config["deployed_services"] = ["telemetry_store"]
    audit = tmp_path / "var" / "audit" / "alice"
    audit.mkdir(parents=True)
    os.mkfifo(audit / "x.jsonl")
    cl._guard_ledger_feed(config, tmp_path)
    assert (audit / "x.jsonl").exists()
