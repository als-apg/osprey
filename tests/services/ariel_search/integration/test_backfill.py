"""``osprey ariel attachments backfill``: dry run, probe, refusal and the printed hint.

The dry-run and probe tests run on a fresh scratch database brought to
today's schema and seeded with entries whose attachments were never recorded,
so every count is known. The dry run must make no fetch at all: the loud
fetch fake stays installed and un-opted, so any fetch fails the test.
"""

from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime, timedelta
from typing import Any
from unittest.mock import patch

import psycopg
import pytest
from click.testing import CliRunner

from osprey.services.ariel_search import cli_operations as ops
from osprey.services.ariel_search.config import ARIELConfig

# xdist_group("docker"): every container-starting test file shares one worker, so a
# run has a single testcontainers session and the shared database is serialized.
pytestmark = [pytest.mark.asyncio, pytest.mark.xdist_group("docker"), pytest.mark.timeout(120)]

SOURCE = "https://h.example/logbook.json"
BASE = datetime(2026, 9, 1, tzinfo=UTC)


def _config_dict(
    uri: str,
    *,
    source: str = SOURCE,
    adapter: str = "generic_json",
    proxy: str | None = None,
) -> dict[str, Any]:
    ingestion: dict[str, Any] = {"adapter": adapter, "source_url": source}
    if proxy is not None:
        ingestion["proxy_url"] = proxy
    return {
        "database": {"uri": uri},
        "ingestion": ingestion,
        "attachments": {"copy_on_ingest": "images"},
    }


async def _migrate(config_dict: dict[str, Any]) -> None:
    """Bring the scratch database to today's schema, as ``osprey ariel migrate`` does."""
    from osprey.services.ariel_search.database.connection import create_connection_pool
    from osprey.services.ariel_search.database.migrations import run_migrations

    config = ARIELConfig.from_dict(json.loads(json.dumps(config_dict)))
    pool = await create_connection_pool(config.database)
    try:
        await run_migrations(pool, config)
    finally:
        await pool.close()


async def _to_b1_schema(config_dict: dict[str, Any]) -> None:
    """Roll the copy state and the entry text columns back, leaving a B1-era store."""
    from osprey.services.ariel_search.database.attachment_migration import (
        AttachmentFilesCopyStateMigration,
    )
    from osprey.services.ariel_search.database.attachment_text_migration import (
        AttachmentTextColumnsMigration,
    )
    from osprey.services.ariel_search.database.connection import create_connection_pool

    config = ARIELConfig.from_dict(json.loads(json.dumps(config_dict)))
    pool = await create_connection_pool(config.database)
    try:
        for migration in (AttachmentFilesCopyStateMigration(), AttachmentTextColumnsMigration()):
            async with pool.connection() as conn, conn.transaction():
                await migration.down(conn)
                await conn.execute(
                    "DELETE FROM ariel_migrations WHERE name = %s", (migration.name,)
                )
    finally:
        await pool.close()


def _seed(uri: str, entry_id: str, attachments: list, *, minutes: int = 0) -> None:
    with psycopg.connect(uri, autocommit=True) as conn:
        conn.execute(
            """
            INSERT INTO enhanced_entries (
                entry_id, source_system, timestamp, author, raw_text,
                attachments, metadata, enhancement_status
            ) VALUES (%s, 'test', %s, 'tester', 'x', %s::jsonb, '{}'::jsonb, '{}'::jsonb)
            """,
            (entry_id, BASE + timedelta(minutes=minutes), json.dumps(attachments)),
        )


def _insert_row(uri: str, attachment_id: str, entry_id: str, url: str, status: str, reason=None):
    with psycopg.connect(uri, autocommit=True) as conn:
        conn.execute(
            """
            INSERT INTO attachment_files (
                attachment_id, entry_id, filename, mime_type, source_url,
                copy_status, skip_reason
            ) VALUES (%s, %s, 'f.png', 'image/png', %s, %s, %s)
            """,
            (attachment_id, entry_id, url, status, reason),
        )


def _row_count(uri: str) -> int:
    with psycopg.connect(uri) as conn:
        row = conn.execute("SELECT count(*) FROM attachment_files").fetchone()
    assert row is not None
    return int(row[0])


def _png(name: str, host: str = "h.example") -> dict[str, str]:
    return {"url": f"https://{host}/files/{name}", "type": "image/png"}


def _seed_census(uri: str) -> None:
    """Three entries whose dry-run counts are known exactly.

    * ``census-new``: one PNG with no row, one ``/rel/x.png`` (never fetchable
      on an http source), one PDF (stays ``copy_on_ingest_mode`` in images
      mode) and one PNG on a foreign host (stays ``origin_not_allowed``).
    * ``census-mid``: 21 PNGs with no row; the 21st exceeds the per-entry limit.
    * ``census-old``: one ``pending`` row and one ``source_gone`` row.
    """
    from osprey.services.ariel_search.attachments import attachment_id_for

    _seed(
        uri,
        "census-new",
        [
            _png("a.png"),
            {"url": "/rel/x.png", "type": "image/png"},
            {"url": "https://h.example/files/b.pdf", "type": "application/pdf"},
            _png("c.png", host="other.example"),
        ],
        minutes=30,
    )
    _seed(uri, "census-mid", [_png(f"m{i:02d}.png") for i in range(21)], minutes=20)
    old = [_png("p.png"), _png("g.png")]
    _seed(uri, "census-old", old, minutes=10)
    for item, status, reason in ((old[0], "pending", None), (old[1], "skipped", "source_gone")):
        aid = attachment_id_for("census-old", item)
        assert aid is not None
        _insert_row(uri, aid, "census-old", item["url"], status, reason)


class TestBackfillDryRun:
    async def test_backfill_dry_run_counts_by_type_and_host_with_zero_fetches(
        self, scratch_database, attachment_fetch
    ):
        config_dict = _config_dict(scratch_database)
        await _migrate(config_dict)
        _seed_census(scratch_database)
        rows_before = _row_count(scratch_database)

        result = await ops.run_backfill(config_dict, dry_run=True)

        assert attachment_fetch.calls == []
        assert _row_count(scratch_database) == rows_before
        assert result.status == "done" and result.dry_run
        plan = result.plan
        assert plan is not None
        assert plan.entries == 3
        png_h = ("image/png", "h.example")
        assert plan.no_row == {
            png_h: 1 + 21,
            ("application/pdf", "h.example"): 1,
            ("image/png", "other.example"): 1,
        }
        assert plan.pending == {png_h: 1}
        assert plan.skipped == {"source_gone": {png_h: 1}}
        assert plan.would_fetch == {png_h: 1 + 20 + 2}
        assert plan.per_entry_limit == {png_h: 1}
        assert plan.still_skipped == {
            "copy_on_ingest_mode": {("application/pdf", "h.example"): 1},
            "origin_not_allowed": {("image/png", "other.example"): 1},
        }
        assert plan.not_fetchable == {("image/png", "(none)"): 1}

    async def test_backfill_dry_run_counts_relative_path_on_http_source_apart(
        self, scratch_database, attachment_fetch
    ):
        config_dict = _config_dict(scratch_database)
        await _migrate(config_dict)
        _seed(scratch_database, "rel-1", [{"url": "/rel/x.png"}])

        result = await ops.run_backfill(config_dict, dry_run=True)

        plan = result.plan
        assert plan is not None
        assert plan.would_fetch == {}
        assert plan.no_row == {}
        assert plan.not_fetchable == {("(none)", "(none)"): 1}
        assert attachment_fetch.calls == []

    async def test_backfill_dry_run_honours_limit(self, scratch_database):
        config_dict = _config_dict(scratch_database)
        await _migrate(config_dict)
        _seed_census(scratch_database)

        result = await ops.run_backfill(config_dict, dry_run=True, limit=1)

        assert result.plan is not None
        assert result.plan.entries == 1
        assert sum(result.plan.would_fetch.values()) == 1  # census-new's a.png only


class TestBackfillRefusal:
    async def test_backfill_on_b1_schema_store_refuses(self, scratch_database, attachment_fetch):
        config_dict = _config_dict(scratch_database)
        await _migrate(config_dict)
        await _to_b1_schema(config_dict)
        _seed(scratch_database, "b1-1", [_png("a.png")])

        result = await ops.run_backfill(config_dict)

        assert result.status == "no_copy_state"
        assert attachment_fetch.calls == []

    def test_backfill_cli_on_b1_schema_store_says_migrate_first_and_exits_nonzero(
        self, scratch_database
    ):
        config_dict = _config_dict(scratch_database)
        asyncio.run(_migrate(config_dict))
        asyncio.run(_to_b1_schema(config_dict))

        result = _invoke_cli(config_dict, ["attachments", "backfill"])

        assert result.exit_code != 0
        assert "run osprey ariel migrate first" in result.output


def _invoke_cli(config_dict: dict[str, Any], args: list[str], **top: Any):
    from osprey.cli.ariel import ariel_group

    values = {"ariel": config_dict, **top}

    def _get(key, default=None, *_a, **_k):
        return values.get(key, default)

    with patch("osprey.cli.ariel.get_config_value", side_effect=_get):
        return CliRunner().invoke(ariel_group, args, catch_exceptions=False)


class TestBackfillCliOutput:
    def test_backfill_cli_dry_run_prints_counts_exec_line_and_redacted_proxy(
        self, scratch_database, attachment_fetch
    ):
        config_dict = _config_dict(scratch_database, proxy="socks5://op:s3cret@proxy.example:1080")
        asyncio.run(_migrate(config_dict))
        _seed_census(scratch_database)

        with patch("shutil.which", return_value=None):
            result = _invoke_cli(
                config_dict, ["attachments", "backfill", "--dry-run"], project_name="demo"
            )

        assert result.exit_code == 0, result.output
        out = result.output
        assert "docker exec demo-ariel-sync osprey ariel attachments backfill --dry-run" in out
        assert "-it" not in out.split()
        assert "socks5://***@proxy.example:1080" in out
        assert "s3cret" not in out
        assert "would fetch: 23" in out
        assert "image/png from h.example: 23" in out
        assert "per_entry_limit (not fetched): 1" in out
        assert attachment_fetch.calls == []


class TestBackfillExecLine:
    @pytest.mark.parametrize(
        ("env", "config", "runtime"),
        [
            ("docker", {}, "docker"),
            ("podman", {}, "podman"),
            (None, {"container_runtime": "podman"}, "podman"),
            (None, {"container_runtime": "docker"}, "docker"),
            (None, {"container_runtime": "auto"}, "docker"),
            ("auto", {"container_runtime": "podman"}, "podman"),
            (None, {}, "docker"),
        ],
    )
    def test_backfill_exec_line_second_token_is_exec(self, monkeypatch, env, config, runtime):
        if env is None:
            monkeypatch.delenv("CONTAINER_RUNTIME", raising=False)
        else:
            monkeypatch.setenv("CONTAINER_RUNTIME", env)

        line = ops.backfill_exec_line({**config, "project_name": "demo"}, ["--limit", "5"])

        tokens = line.split()
        assert tokens[0] == runtime
        assert tokens[1] == "exec"
        assert tokens[2] == "demo-ariel-sync"
        assert tokens[3:] == ["osprey", "ariel", "attachments", "backfill", "--limit", "5"]

    def test_backfill_exec_line_renders_without_any_runtime_on_path(self, monkeypatch):
        monkeypatch.delenv("CONTAINER_RUNTIME", raising=False)
        with patch("shutil.which", return_value=None):
            line = ops.backfill_exec_line({"project_root": "/srv/als-ops"})
        assert line == "docker exec als-ops-ariel-sync osprey ariel attachments backfill"


@pytest.mark.real_fetch
class TestBackfillProbe:
    async def test_backfill_probe_sends_exactly_n_heads_inside_the_origin_set(
        self, scratch_database
    ):
        from aiohttp import web

        seen: list[tuple[str, str]] = []

        async def _handler(request: web.Request) -> web.Response:
            seen.append((request.method, request.path))
            return web.Response(body=b"", content_type="image/png")

        app = web.Application()
        app.router.add_route("*", "/{tail:.*}", _handler)
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        try:
            port = site._server.sockets[0].getsockname()[1]  # type: ignore[union-attr]
            origin = f"http://127.0.0.1:{port}"
            config_dict = _config_dict(scratch_database, source=f"{origin}/logbook.json")
            await _migrate(config_dict)
            inside = [{"url": f"{origin}/files/{i}.png", "type": "image/png"} for i in range(6)]
            outside = [{"url": f"http://localhost:{port}/files/x.png", "type": "image/png"}]
            _seed(scratch_database, "probe-1", inside + outside)

            result = await ops.run_backfill(config_dict, probe=3)
        finally:
            await runner.cleanup()

        assert [method for method, _ in seen] == ["HEAD"] * 3
        assert all(path.startswith("/files/") and path != "/files/x.png" for _, path in seen)
        assert result.probe is not None
        assert result.probe.sampled == 3
        assert result.probe.reachable == 3
        assert result.probe.estimate == 6
        assert _row_count(scratch_database) == 0

    def test_backfill_cli_probe_prints_estimate(self, scratch_database, monkeypatch):
        from osprey.services.ariel_search.attachments import copy as copy_mod
        from osprey.services.ariel_search.attachments.fetch import FetchOutcome

        heads: list[tuple[str, str]] = []

        async def _head(url, _cap, _origins, _adapter, method="GET", **_kwargs):
            heads.append((method, url))
            return FetchOutcome(data=b"")

        monkeypatch.setattr(copy_mod, "fetch_attachment_bytes", _head)
        config_dict = _config_dict(scratch_database)
        asyncio.run(_migrate(config_dict))
        _seed(scratch_database, "probe-cli", [_png(f"{i}.png") for i in range(4)])

        result = _invoke_cli(config_dict, ["attachments", "backfill", "--probe", "2"])

        assert result.exit_code == 0, result.output
        assert "estimate" in result.output
        assert len(heads) == 2 and all(m == "HEAD" for m, _ in heads)
