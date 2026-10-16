"""The shared ARIEL picture scenario: one logbook entry whose answer is only in its plot.

A case (:class:`PictureCase`) names the plot, the target entry's text, the
text-only entries seeded beside it, the operator's question, and the answer
only the plot carries. :func:`run_picture_case` runs it as one agent session
against a deployment built from the control-assistant preset unchanged, and
asserts the same chain for every case: the agent delegates to the
``logbook-search`` subagent, that subagent views the plot with
``attachment_view``, the answer names what only the plot shows, and a follow-up
turn opens the entry together with the plot.

The entry is seeded through the native write path ``entry_create`` uses
(:func:`~osprey.services.ariel_search.attachments.store_native_attachment`
with the PNG bytes), which prepares the rendition in-process. A raw row insert
would leave the picture ``pending`` and ``attachment_view`` answering
not-ready, because viewers only read finished copies; the run checks the row
is ``copied`` with a viewable rendition before the agent starts.

The presets turn ``image_caption`` and ``image_embedding`` on, and these lanes
have neither a caption model nor an embedding server. A run therefore also
proves the agent turn completes with no error from the picture feature on the
agent path. Those tools run in the MCP server Claude Code launches as a
subprocess, out of reach of pytest's log capture, so that check reads the agent
trace: no ARIEL tool result carries an error envelope, save ``hybrid_search``
reporting the qmd sidecar the harness never starts (see
:func:`absent_qmd_sidecar`). The log check covers the one step that runs in
this process, the seeding.

The second turn asks to open "that entry" in the logbook panel. The operator
names the entry, not the picture, yet the conversation is about the plot, so
``entry_open`` must carry the plot's ``attachment_id`` as well as the entry id.

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
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
import yaml

from tests._container_support import is_docker_available, start_or_fail, stop_quietly
from tests.e2e.provider import e2e_provider
from tests.e2e.sdk_helpers import init_project, render_dir, run_sdk_query

#: The plot fixtures' directory.
FIXTURES = Path(__file__).parent / "fixtures" / "ariel_picture"

#: The name every case's picture is stored under: nothing of any answer in it.
STORED_FILENAME = "orbit_drift.png"

#: The database container's image, the one the ARIEL service tests start.
PGVECTOR_IMAGE = "pgvector/pgvector:pg16"


@dataclass(frozen=True)
class PictureCase:
    """One question whose answer only a logbook plot carries.

    Attributes:
        plot_path: The PNG attached to the target entry; it carries ``answer``
            in its pixels and in no text metadata.
        raw_text: The target entry's text; it never carries ``answer``.
        prompt: The operator's question; it names no plot, picture or
            attachment.
        answer: The text the final answer must contain, matched case-blind.
        distractors: Text-only entries seeded beside the target; none
            carries ``answer`` and none has a picture.
    """

    plot_path: Path
    raw_text: str
    prompt: str
    answer: str
    distractors: tuple[str, ...] = ()


LOGBOOK_SUBAGENT = "logbook-search"
VIEW_TOOL = "mcp__ariel__attachment_view"
OPEN_TOOL = "mcp__ariel__entry_open"

#: The follow-up turn: it names the entry only.
FOLLOW_UP = "Open that entry in the logbook panel."

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


async def _seed_picture_entry(ariel: dict[str, Any], case: PictureCase) -> tuple[str, str]:
    """Migrate the database, write the case's distractors, and write its target entry with the plot.

    The target and its plot are written the way ``entry_create`` does.

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

        for raw_text in case.distractors:
            created = datetime.now(UTC)
            distractor: Any = {
                "entry_id": f"ariel-{uuid.uuid4().hex[:12]}",
                "source_system": ARIEL_NATIVE_SOURCE_SYSTEM,
                "timestamp": created,
                "author": "e2e operator",
                "raw_text": raw_text,
                "attachments": [],
                "metadata": {"logbook": "Operations", "created_via": "e2e"},
                "created_at": created,
                "updated_at": created,
            }
            await repository.upsert_entry(distractor)

        entry_id = f"ariel-{uuid.uuid4().hex[:12]}"
        now = datetime.now(UTC)
        entry: Any = {
            "entry_id": entry_id,
            "source_system": ARIEL_NATIVE_SOURCE_SYSTEM,
            "timestamp": now,
            "author": "e2e operator",
            "raw_text": case.raw_text,
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
            data=case.plot_path.read_bytes(),
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


def logbook_parent_ids(traces: list[Any]) -> set[str]:
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


def delegated_types(traces: list[Any]) -> set[str]:
    return {
        str((t.input or {}).get("subagent_type"))
        for t in traces
        if t.name == "Agent" or t.name.startswith("Task")
    }


def absent_qmd_sidecar(trace: Any) -> bool:
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


def error_envelope(result: str | None) -> bool:
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


def assert_viewed_in_logbook_turn(
    traces: list[Any], attachment_id: str, tool_names: list[str]
) -> None:
    """Assert ``attachment_view`` ran on *attachment_id* inside a ``logbook-search`` turn.

    Three facts, each its own message: the view was called with the seeded id,
    the agent delegated to the logbook subagent at all, and the view's parent is
    one of that subagent's turns (or the logbook subagent is the only one the
    agent used, when the harvested transcript cannot name the parent).
    """
    views = [
        t
        for t in traces
        if t.name == VIEW_TOOL and (t.input or {}).get("attachment_id") == attachment_id
    ]
    assert views, (
        f"attachment_view was never called with the seeded id {attachment_id}. Tools: {tool_names}"
    )
    logbook_ids = logbook_parent_ids(traces)
    assert logbook_ids, f"the agent never delegated to {LOGBOOK_SUBAGENT}. Tools: {tool_names}"
    only_logbook = delegated_types(traces) == {LOGBOOK_SUBAGENT}
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


def assert_no_ariel_errors(traces: list[Any]) -> None:
    """Assert no ARIEL tool result carried an error envelope (an absent qmd sidecar excepted)."""
    ariel_errors = [
        t
        for t in traces
        if t.name.startswith("mcp__ariel__")
        and (t.is_error or error_envelope(t.result))
        and not absent_qmd_sidecar(t)
    ]
    assert not ariel_errors, "ARIEL tool results carried errors: " + "; ".join(
        f"{t.name}: {(t.result or '')[:300]}" for t in ariel_errors
    )


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


async def run_picture_case(
    case: PictureCase,
    tmp_path: Path,
    request: pytest.FixtureRequest,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Run *case* as one agent session plus a follow-up turn, and assert the chain.

    See the module docstring for what each assertion proves.
    """
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
        entry_id, attachment_id = await _seed_picture_entry(ariel, case)
    feature_errors = _feature_errors(caplog.records)
    assert not feature_errors, "the picture feature logged ERROR while seeding: " + "; ".join(
        f"{r.pathname}:{r.lineno} {r.getMessage()}" for r in feature_errors
    )

    result = await run_sdk_query(repo, case.prompt, max_turns=30, max_budget_usd=2.0)

    final = (result.result.result if result.result is not None else None) or ""
    excerpt = _trace_excerpt(result, final)
    (tmp_path / "trace_excerpt.json").write_text(
        json.dumps(excerpt, indent=2, default=str), encoding="utf-8"
    )
    print(json.dumps(excerpt, indent=2, default=str))

    assert result.result is not None, "no ResultMessage received"
    assert not result.result.is_error, f"the agent turn ended in error: {final}"

    traces = result.tool_traces
    assert_viewed_in_logbook_turn(traces, attachment_id, result.tool_names)
    assert_no_ariel_errors(traces)

    assert case.answer.lower() in final.lower(), (
        f"the answer does not name {case.answer!r}, which only the plot carries: {final!r}"
    )

    session_id = result.result.session_id
    assert session_id, "the first turn reported no session id to resume"
    follow = await run_sdk_query(
        repo, FOLLOW_UP, max_turns=10, max_budget_usd=1.0, resume=session_id
    )
    follow_final = (follow.result.result if follow.result is not None else None) or ""
    (tmp_path / "follow_up_excerpt.json").write_text(
        json.dumps(_trace_excerpt(follow, follow_final), indent=2, default=str), encoding="utf-8"
    )

    assert follow.result is not None, "no ResultMessage received for the follow-up"
    assert not follow.result.is_error, f"the follow-up turn ended in error: {follow_final}"
    opens = [t for t in follow.tool_traces if t.name == OPEN_TOOL]
    assert opens, f"entry_open was never called on the follow-up. Tools: {follow.tool_names}"
    assert any(
        (t.input or {}).get("entry_id") == entry_id
        and (t.input or {}).get("attachment_id") == attachment_id
        for t in opens
    ), (
        f"entry_open never carried the plot under discussion ({entry_id}, {attachment_id}): "
        f"{[t.input for t in opens]}"
    )
