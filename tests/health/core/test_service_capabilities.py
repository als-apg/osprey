"""The ``service_capabilities`` health category reads capability surfaces nothing else reads."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
from pathlib import Path

import httpx

from osprey.build.manifest import MANIFEST_FILENAME
from osprey.health.core.service_capabilities import service_capabilities
from osprey.health.models import Status

_NOW = 1_800_000_000.0
_BLUESKY = {"services": {"bluesky": {"port": 10080}}}


def _capability(**overrides) -> dict:
    cap = {
        "can_execute": True,
        "reason": "executable",
        "detail": "Plans execute against the 'virtual_accelerator' connector.",
        "lane": "bluesky",
        "lane_target": "va",
        "retryable": False,
    }
    cap.update(overrides)
    return {"status": "ok", "capability": cap}


def _bridge(body: dict | None = None, status: int = 200):
    seen: list[httpx.Request] = []

    def handle(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(status, json=body if body is not None else _capability())

    return handle, seen


def _rows(config, *, handler=None, cwd=None) -> dict:
    transport = httpx.MockTransport(handler) if handler else None
    run = service_capabilities(
        config,
        cwd=cwd,
        transport=transport,
        now=lambda: _NOW,
    )
    return {row.name: row for row in asyncio.run(run())}


# -- nothing configured ------------------------------------------------------


def test_no_rows_when_nothing_is_configured():
    assert _rows({}) == {}


# -- bluesky bridge ------------------------------------------------------------


def test_an_executing_bridge(monkeypatch):
    monkeypatch.delenv("BLUESKY_BRIDGE_URL", raising=False)
    handle, seen = _bridge()
    row = _rows(_BLUESKY, handler=handle)["bluesky.capability"]

    assert row.status is Status.OK
    assert row.value == "can execute · va"
    assert row.probed and row.latency_ms > 0
    assert str(seen[0].url) == "http://127.0.0.1:10080/health"


def test_a_browse_only_bridge_is_not_a_fault(monkeypatch):
    monkeypatch.delenv("BLUESKY_BRIDGE_URL", raising=False)
    handle, _ = _bridge(_capability(can_execute=False, reason="browse_only_connector"))
    row = _rows(_BLUESKY, handler=handle)["bluesky.capability"]

    assert row.status is Status.OK
    assert row.value == "browse only · browse_only_connector"


def test_a_bridge_that_cannot_reach_its_manager_warns(monkeypatch):
    monkeypatch.delenv("BLUESKY_BRIDGE_URL", raising=False)
    handle, _ = _bridge(
        _capability(can_execute=False, reason="manager_unreachable", detail="no manager")
    )
    row = _rows(_BLUESKY, handler=handle)["bluesky.capability"]

    assert row.status is Status.WARNING
    assert row.value == "cannot execute · manager_unreachable"
    assert "no manager" in row.message


def test_an_unreachable_bridge_warns(monkeypatch):
    monkeypatch.delenv("BLUESKY_BRIDGE_URL", raising=False)

    def refuse(request):
        raise httpx.ConnectError("refused", request=request)

    row = _rows(_BLUESKY, handler=refuse)["bluesky.capability"]
    assert row.status is Status.WARNING
    assert row.value == "unreachable"


def test_the_bridge_url_override_is_honoured(monkeypatch):
    monkeypatch.setenv("BLUESKY_BRIDGE_URL", "http://bridge:9000/")
    handle, seen = _bridge()
    _rows(_BLUESKY, handler=handle)
    assert str(seen[0].url) == "http://bridge:9000/health"


# -- OKF bundle ------------------------------------------------------------------


def _bundle(render: Path) -> Path:
    root = render / "data" / "knowledge"
    (root / "devices").mkdir(parents=True)
    (root / "index.md").write_text("# Index\n")
    (root / "devices" / "rf.md").write_text("# RF\n")
    for path in root.rglob("*.md"):
        os.utime(path, (_NOW - 7200, _NOW - 7200))
    return root


def _stamp(render: Path) -> None:
    checksums = {
        str(p.relative_to(render)): "sha256:" + hashlib.sha256(p.read_bytes()).hexdigest()
        for p in render.rglob("*")
        if p.is_file()
    }
    (render / MANIFEST_FILENAME).write_text(json.dumps({"file_checksums": checksums}))


_OKF = {"facility_knowledge": {"bundle_path": "data/knowledge"}}


def test_a_bundle_matching_the_render(tmp_path):
    _bundle(tmp_path)
    _stamp(tmp_path)
    rows = _rows(_OKF, cwd=tmp_path)

    bundle = rows["okf.bundle"]
    assert bundle.status is Status.OK
    assert bundle.value.startswith("1 doc · digest ")
    assert bundle.value.endswith(" = render")
    assert rows["okf.bundle_age"].status is Status.OK
    assert rows["okf.bundle_age"].value == "2.0 h"


def test_an_edited_bundle_differs_from_the_render(tmp_path):
    root = _bundle(tmp_path)
    _stamp(tmp_path)
    (root / "devices" / "rf.md").write_text("# RF edited\n")
    bundle = _rows(_OKF, cwd=tmp_path)["okf.bundle"]

    assert bundle.status is Status.WARNING
    assert bundle.value.endswith("· 1 file differs from render")
    assert bundle.fix == "osprey build"


def test_a_bundle_without_a_manifest_reports_its_digest(tmp_path):
    _bundle(tmp_path)
    bundle = _rows(_OKF, cwd=tmp_path)["okf.bundle"]
    assert bundle.status is Status.OK
    assert "= render" not in bundle.value


def test_the_digest_follows_the_content(tmp_path):
    root = _bundle(tmp_path)
    first = _rows(_OKF, cwd=tmp_path)["okf.bundle"].value
    (root / "index.md").write_text("# Changed\n")
    assert _rows(_OKF, cwd=tmp_path)["okf.bundle"].value != first


def test_a_missing_bundle_warns(tmp_path):
    rows = _rows(_OKF, cwd=tmp_path)
    assert rows["okf.bundle"].status is Status.WARNING
    assert rows["okf.bundle"].value == "missing"
    assert "okf.bundle_age" not in rows


# -- ARIEL enhancement modules are not polled ------------------------------------


def test_the_poll_category_never_runs_an_ariel_module_check():
    """A module's health_check can send a billed completion or create a directory."""
    called: list[str] = []

    class _Billed:
        name = "semantic_processor"

        async def health_check(self):
            called.append(self.name)

    run = service_capabilities(
        {"ariel": {"enhancement_modules": {"semantic_processor": {"enabled": True}}}},
        now=lambda: _NOW,
    )
    rows = asyncio.run(run())

    assert called == []
    assert not any(r.name.startswith("ariel.") for r in rows)
