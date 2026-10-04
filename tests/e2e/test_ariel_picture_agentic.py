"""Agentic e2e: the logbook agent finds an answer in an attached plot unprompted.

One agent run against a deployment built from the control-assistant preset
unchanged. The question never mentions a plot or picture. The logbook it searches is a pgvector test container holding one
entry whose only attachment is a plot. The plot carries the nonsense token
``QX-7713`` in its pixels and nowhere else: not in the entry's text, not in
the stored filename, not in the PNG's metadata. An answer that names the token
therefore read the picture, and the only route to the picture is the
``attachment_view`` tool inside the ``logbook-search`` subagent.

The entry is seeded through the native write path ``entry_create`` uses
(:func:`~osprey.services.ariel_search.attachments.store_native_attachment`
with the PNG bytes), which prepares the rendition in-process. A raw row insert
would leave the picture ``pending`` and ``attachment_view`` answering
not-ready, because viewers only read finished copies; the test checks the row
is ``copied`` with a viewable rendition before the agent starts.

The presets turn ``image_caption`` and ``image_embedding`` on, and this lane
has neither a caption model nor an embedding server. The run therefore also
proves the agent turn completes with no error from the picture feature on the
agent path. Those tools run in the MCP server Claude Code launches as a
subprocess, out of reach of pytest's log capture, so that check reads the agent
trace: no ARIEL tool result carries an error envelope, save ``hybrid_search``
reporting the qmd sidecar the harness never starts (see
:func:`_absent_qmd_sidecar`). The log check covers the one step that runs in
this process, the seeding.

The plot fixture, ``fixtures/ariel_picture/orbit_drift.png``, is a matplotlib
line plot titled ``Run QX-7713`` with the same token boxed inside the axes,
saved with no text metadata.

Prerequisites: a provider named by ``OSPREY_E2E_PROVIDER`` (resolved through
``tests/e2e/provider.py``) whose route carries images, and Docker for the
database container. Under ``GITHUB_ACTIONS`` a missing database fails rather
than skips; a provider whose route carries no images skips everywhere, since
that is a property of the provider and not a missing service.
"""

from __future__ import annotations

import json
import logging
import os
import re
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
import yaml

from tests._container_support import is_docker_available, start_or_fail, stop_quietly
from tests.e2e.provider import e2e_provider
from tests.e2e.sdk_helpers import HAS_SDK, init_project, render_dir, run_sdk_query

pytestmark = [
    pytest.mark.e2e,
    pytest.mark.agentic_benchmark,
    pytest.mark.requires_e2e_provider,
    pytest.mark.skipif(not HAS_SDK, reason="claude_agent_sdk not installed"),
]

#: The token the plot carries in its pixels and nowhere else.
TOKEN = "QX-7713"

#: The plot, rendered once with matplotlib (see the module docstring).
PLOT_PATH = Path(__file__).parent / "fixtures" / "ariel_picture" / "orbit_drift.png"

#: The name the picture is stored under: no part of the token in it.
STORED_FILENAME = "orbit_drift.png"

#: The database container's image, the one the ARIEL service tests start.
PGVECTOR_IMAGE = "pgvector/pgvector:pg16"

SUBJECT = "Sector 9 horizontal orbit drift study"
DETAILS = (
    "Recorded the overnight drift of the horizontal orbit in sector 9 during the "
    "corrector study. The trend is in the attached plot."
)

#: A neutral question: it names no plot, picture or attachment, and the run
#: identifier it asks for is only in the plot's pixels. The lane has no caption
#: model, so the agent must decide on its own, from the attachment summary
#: (``viewable: true``, no caption), that the picture is worth a look.
PROMPT = "What is the run identifier of the sector 9 horizontal orbit drift study in the logbook?"

LOGBOOK_SUBAGENT = "logbook-search"
VIEW_TOOL = "mcp__ariel__attachment_view"

#: Where the picture feature's in-process code lives, for the seeding log check.
#: The shared ``ariel`` logger also carries the semantic-search ERROR an absent
#: Ollama raises, which is not this feature's, so records are told apart by path.
_FEATURE_PATH_PARTS = ("/ariel_search/attachments/", "/ariel_search/enhancement/")
_FEATURE_PATH_SUFFIXES = ("search/image_lane.py", "search/fusion.py")


def _effective_supports_images(provider: str) -> bool:
    """Whether *provider*'s route carries images, as the translation proxy decides it.

    The packaged catalog entry's own ``supports_images`` when it declares one,
    else the adapter class's; an unregistered provider carries none.
    """
    from osprey.models.provider_registry import get_provider_registry
    from osprey.profiles.providers import load_provider_catalog

    entry = load_provider_catalog(None).entries.get(provider) or {}
    if isinstance(entry, dict) and "supports_images" in entry:
        return bool(entry["supports_images"])
    provider_class = get_provider_registry().get_provider(provider)
    return bool(getattr(provider_class, "supports_images", False))


def _no_database(reason: str) -> None:
    """Fail under GitHub Actions, skip elsewhere: CI must never report a run it did not make."""
    if os.environ.get("GITHUB_ACTIONS") == "true":
        pytest.fail(f"ARIEL picture e2e needs its database container: {reason}")
    pytest.skip(reason)


def _start_logbook_db(request: pytest.FixtureRequest) -> str:
    """Start a pgvector container for this test and return its URI."""
    if not is_docker_available():
        _no_database("Docker is not available to start the pgvector container")

    from testcontainers.postgres import PostgresContainer

    container, port = start_or_fail(
        lambda: PostgresContainer(
            image=PGVECTOR_IMAGE, username="ariel", password="ariel", dbname="ariel"
        ),
        label="ariel-picture-e2e-postgres",
        port=5432,
    )
    request.addfinalizer(lambda: stop_quietly(container))
    host = container.get_container_host_ip()
    return f"postgresql://ariel:ariel@{host}:{port}/ariel"


def _rendered_ariel(repo: Path) -> dict[str, Any]:
    config = yaml.safe_load((render_dir(repo) / "config.yml").read_text(encoding="utf-8")) or {}
    ariel = config.get("ariel")
    assert isinstance(ariel, dict), "the built project's config.yml has no ariel section"
    return ariel


def _feature_errors(records: list[logging.LogRecord]) -> list[logging.LogRecord]:
    """ERROR records raised by the picture feature's own modules."""
    return [
        r
        for r in records
        if r.levelno >= logging.ERROR
        and (
            any(part in r.pathname for part in _FEATURE_PATH_PARTS)
            or r.pathname.endswith(_FEATURE_PATH_SUFFIXES)
        )
    ]


async def _seed_picture_entry(ariel: dict[str, Any]) -> tuple[str, str]:
    """Migrate the database and write one entry with the plot the way ``entry_create`` does.

    Returns:
        The entry id and the picture's attachment id.
    """
    from osprey.mcp_server.ariel.server import ARIEL_NATIVE_SOURCE_SYSTEM
    from osprey.services.ariel_search.attachments import store_native_attachment
    from osprey.services.ariel_search.config import ARIELConfig
    from osprey.services.ariel_search.database import create_connection_pool, run_migrations
    from osprey.services.ariel_search.database.repository import ARIELRepository

    config = ARIELConfig.from_dict(ariel)
    pool = await create_connection_pool(config.database)
    try:
        await run_migrations(pool, config)
        repository = ARIELRepository(pool, config)

        entry_id = f"ariel-{uuid.uuid4().hex[:12]}"
        now = datetime.now(UTC)
        entry: Any = {
            "entry_id": entry_id,
            "source_system": ARIEL_NATIVE_SOURCE_SYSTEM,
            "timestamp": now,
            "author": "e2e operator",
            "raw_text": f"{SUBJECT}\n\n{DETAILS}",
            "attachments": [],
            "metadata": {"logbook": "Operations", "tags": ["orbit"], "created_via": "e2e"},
            "created_at": now,
            "updated_at": now,
        }
        await repository.upsert_entry(entry)
        info = await store_native_attachment(
            repository,
            entry_id,
            filename=STORED_FILENAME,
            declared_mime="image/png",
            data=PLOT_PATH.read_bytes(),
        )
        entry["attachments"] = [info]
        await repository.upsert_entry(entry)

        rows = await repository.get_copy_rows(entry_id)
        assert len(rows) == 1, f"expected one attachment row for {entry_id}, got {rows}"
        row = rows[0]
        assert row["copy_status"] == "copied", (
            f"the seeded picture is {row['copy_status']!r}, not 'copied' "
            f"(skip_reason={row['skip_reason']!r}); attachment_view would answer not-ready"
        )
        attachment_id = row["attachment_id"]
        assert await repository.get_rendition(attachment_id) is not None, (
            f"the seeded picture {attachment_id} has no viewable rendition"
        )
        return entry_id, attachment_id
    finally:
        await pool.close()


def _logbook_parent_ids(traces: list[Any]) -> set[str]:
    """Every id a logbook-search subagent's tool calls can carry as their parent.

    A streamed subagent call names the delegating ``Agent`` tool_use id; one
    harvested from the side-file transcript names the subagent's agent id, which
    the ``Agent`` tool result reports as ``agentId: <id>``.
    """
    ids: set[str] = set()
    for t in traces:
        if (t.name == "Agent" or t.name.startswith("Task")) and (
            (t.input or {}).get("subagent_type") == LOGBOOK_SUBAGENT
        ):
            if t.tool_use_id:
                ids.add(t.tool_use_id)
            ids.update(re.findall(r"agentId:\s*([\w-]+)", t.result or ""))
    return ids


def _delegated_types(traces: list[Any]) -> set[str]:
    return {
        str((t.input or {}).get("subagent_type"))
        for t in traces
        if t.name == "Agent" or t.name.startswith("Task")
    }


def _absent_qmd_sidecar(trace: Any) -> bool:
    """Whether *trace* is ``hybrid_search`` reporting the qmd sidecar this harness never starts.

    The hybrid module answers through a qmd sidecar the deployment runs as a
    service. Projects built here run without their services, so the sidecar is
    absent and ``hybrid_search`` answers ``service_unavailable`` naming it. That
    is the search stack's honest report of a missing service, the same kind as
    the semantic-search ERROR an absent Ollama raises, and not an error from the
    picture feature. Every other ARIEL error, this tool's included, still fails.
    """
    result = trace.result or ""
    return (
        trace.name == "mcp__ariel__hybrid_search"
        and '"error_type": "service_unavailable"' in result
        and "qmd sidecar" in result
    )


def _error_envelope(result: str | None) -> bool:
    """Whether a tool result is an OSPREY MCP error envelope."""
    if not result:
        return False
    if '"error_type"' in result:
        return True
    try:
        payload = json.loads(result)
    except (TypeError, ValueError):
        return False
    return isinstance(payload, dict) and payload.get("error") is True


def _trace_excerpt(result: Any, final: str) -> dict[str, Any]:
    return {
        "tools": [
            {
                "name": t.name,
                "input": t.input,
                "parent_tool_use_id": t.parent_tool_use_id,
                "is_error": t.is_error,
                "result": (t.result or "")[:400],
            }
            for t in result.tool_traces
        ],
        "answer": final,
        "num_turns": result.num_turns,
        "cost_usd": result.cost_usd,
    }


@pytest.mark.flaky(reruns=2)  # agentic; absorbs a rare miss of the delegation directive
@pytest.mark.asyncio
async def test_agent_reads_the_attached_plot(
    tmp_path: Path,
    request: pytest.FixtureRequest,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Asked a neutral question, the logbook agent opens the plot and reports its token."""
    provider = e2e_provider()
    if not _effective_supports_images(provider):
        pytest.skip(f"provider {provider} does not carry images on its route")

    db_uri = _start_logbook_db(request)
    # init_project pins ariel.database.uri to this and checks the render names it.
    monkeypatch.setenv("OSPREY_ARIEL_DB_URI", db_uri)
    repo = init_project(tmp_path, "picture_demo", template="control_assistant", provider=provider)

    ariel = _rendered_ariel(repo)
    modules = ariel.get("enhancement_modules") or {}
    for module in ("image_caption", "image_embedding"):
        assert (modules.get(module) or {}).get("enabled") is True, (
            f"the preset's rendered config does not enable ariel.enhancement_modules.{module}"
        )
    assert (ariel.get("database") or {}).get("uri") == db_uri

    with caplog.at_level(logging.WARNING):
        _entry_id, attachment_id = await _seed_picture_entry(ariel)
    feature_errors = _feature_errors(caplog.records)
    assert not feature_errors, "the picture feature logged ERROR while seeding: " + "; ".join(
        f"{r.pathname}:{r.lineno} {r.getMessage()}" for r in feature_errors
    )

    result = await run_sdk_query(repo, PROMPT, max_turns=30, max_budget_usd=2.0)

    final = (result.result.result if result.result is not None else None) or ""
    excerpt = _trace_excerpt(result, final)
    (tmp_path / "trace_excerpt.json").write_text(
        json.dumps(excerpt, indent=2, default=str), encoding="utf-8"
    )
    print(json.dumps(excerpt, indent=2, default=str))

    assert result.result is not None, "no ResultMessage received"
    assert not result.result.is_error, f"the agent turn ended in error: {final}"

    traces = result.tool_traces
    views = [
        t
        for t in traces
        if t.name == VIEW_TOOL and (t.input or {}).get("attachment_id") == attachment_id
    ]
    assert views, (
        f"attachment_view was never called with the seeded id {attachment_id}. "
        f"Tools: {result.tool_names}"
    )

    logbook_ids = _logbook_parent_ids(traces)
    assert logbook_ids, (
        f"the agent never delegated to {LOGBOOK_SUBAGENT}. Tools: {result.tool_names}"
    )
    only_logbook = _delegated_types(traces) == {LOGBOOK_SUBAGENT}
    in_logbook_turn = [
        t
        for t in views
        if t.parent_tool_use_id is not None
        and (t.parent_tool_use_id in logbook_ids or only_logbook)
    ]
    assert in_logbook_turn, (
        f"attachment_view({attachment_id}) ran outside a {LOGBOOK_SUBAGENT} subagent turn: "
        f"parents {[t.parent_tool_use_id for t in views]}, logbook ids {sorted(logbook_ids)}"
    )

    ariel_errors = [
        t
        for t in traces
        if t.name.startswith("mcp__ariel__")
        and (t.is_error or _error_envelope(t.result))
        and not _absent_qmd_sidecar(t)
    ]
    assert not ariel_errors, "ARIEL tool results carried errors: " + "; ".join(
        f"{t.name}: {(t.result or '')[:300]}" for t in ariel_errors
    )

    assert TOKEN.lower() in final.lower(), (
        f"the answer does not name {TOKEN}, the token only the plot carries: {final!r}"
    )
