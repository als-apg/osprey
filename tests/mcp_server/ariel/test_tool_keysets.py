"""Key/type goldens for the ARIEL MCP tools: compatibility as an enumerated diff.

Each tool runs against the ``keyset_harness`` fixture (``conftest.py``): one
``config.yml`` with an ``entry_url`` template, a real ``ARIELSearchService``
over an in-memory repository seeded with one entry that has long text, a
summary, metadata tags, one PNG and one PDF attachment, plus a second short
entry without attachments.

What is pinned per tool:

* ``keyword_search``, ``semantic_search`` and ``hybrid_search`` run through the
  real service and its search modules (the repository, the embedder and the qmd
  client are faked at their boundaries), so keys the service or repository
  layer adds reach the comparison.
* ``browse``, ``entry_get``, ``entries_by_ids``, ``capabilities``, ``status``,
  ``entry_open`` and ``attachment_to_artifact`` pin only the tool layer: they
  read the fake repository, the context config or a patched
  ``service.get_status`` directly, and ``attachment_to_artifact`` saves into an
  artifact store under the test's directory.

The comparison is "superset plus the listed changes": every golden path keeps
its type, and a new path is accepted only when ``ALLOWED_ADDITIONS`` or
``ALLOWED_CHANGES`` names it. An empty list is recorded as ``list(empty)``, so
a list that is empty in the golden must stay empty and keeps no element shape.
The ``keyword_search`` and ``hybrid_search`` outputs are additionally pinned by
value: after removing only the allowed keys, the output equals
``data/json_values/<tool>.json`` byte for byte.

Regenerating: ``ARIEL_KEYSETS_REGEN=1`` writes both golden sets instead of
comparing, and refuses unless ``git rev-parse HEAD`` equals
``ARIEL_KEYSETS_B1_SHA`` (read from line 1 of ``.claude/scratch/b1-merged-sha``
when unset). A golden change is regenerated in its own commit.
"""

from __future__ import annotations

import json
import os
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from tests.mcp_server.ariel.conftest import KEYSET_ENTRY_ID, get_tool_fn, keyset_entries

REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = Path(__file__).resolve().parent / "data"
KEYSET_DIR = DATA_DIR / "json_keysets"
VALUE_DIR = DATA_DIR / "json_values"
B1_SHA_FILE = REPO_ROOT / ".claude" / "scratch" / "b1-merged-sha"

TOOLS = (
    "keyword_search",
    "semantic_search",
    "hybrid_search",
    "browse",
    "entry_get",
    "entries_by_ids",
    "capabilities",
    "status",
    "entry_open",
    "attachment_to_artifact",
)
VALUE_TOOLS = ("keyword_search", "hybrid_search")

_SEARCH_TOOLS = ("keyword_search", "semantic_search", "hybrid_search")
_LISTING_TOOLS = (*_SEARCH_TOOLS, "browse", "entries_by_ids")

# Requirement 5: the only paths a later phase may add. A trailing ``*`` matches
# any continuation of the prefix; any other pattern matches exactly. The
# ``hybrid_search`` ``include_images`` parameter is an input and has no path.
_LISTING_ADDITIONS = (
    "entries[].attachment_count",
    "entries[].attachments",
    "entries[].attachments[]*",
    "entries[].caption_truncated",
    "entries[].visible_text_truncated",
)
_MATCH_ADDITIONS = (
    "entries[].matched_via",
    "entries[].matched_via[]*",
    "entries[].matched_attachment_ids",
    "entries[].matched_attachment_ids[]*",
)
ALLOWED_ADDITIONS: dict[str, tuple[str, ...]] = {
    "keyword_search": _LISTING_ADDITIONS + _MATCH_ADDITIONS,
    "semantic_search": _LISTING_ADDITIONS + _MATCH_ADDITIONS,
    "hybrid_search": _LISTING_ADDITIONS + _MATCH_ADDITIONS,
    "browse": _LISTING_ADDITIONS,
    "entries_by_ids": _LISTING_ADDITIONS,
    "entry_get": ("attachment_count",),
    "capabilities": ("attachments", "attachments.*"),
    "status": (),
    "entry_open": (),
    "attachment_to_artifact": (),
}

# Requirement 5: golden paths whose shape a later phase may change. The stored
# attachment elements of ``entry_get`` become summaries (``url`` only when
# absolute), so every path below the element may be added, dropped or retyped.
ALLOWED_CHANGES: dict[str, tuple[str, ...]] = {
    "entry_get": ("attachments[].*",),
}

# Keys removed from each ``entries[]`` element before the value comparison.
VALUE_STRIP_KEYS = (
    "attachment_count",
    "attachments",
    "caption_truncated",
    "visible_text_truncated",
    "matched_via",
    "matched_attachment_ids",
)

# Lists whose element shape a later phase changes: seeded non-empty so the
# golden records that shape instead of ``list(empty)``.
SEEDED_LISTS: dict[str, tuple[str, ...]] = {
    **dict.fromkeys(_LISTING_TOOLS, ("entries",)),
    "entry_get": ("attachments",),
    "status": ("embedding_tables", "errors"),
    "capabilities": ("search_modes",),
}


# ---------------------------------------------------------------------------
# Key/type sets
# ---------------------------------------------------------------------------


def _type_name(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int):
        return "int"
    if isinstance(value, float):
        return "float"
    if isinstance(value, str):
        return "str"
    if isinstance(value, dict):
        return "dict"
    if isinstance(value, list):
        return "list" if value else "list(empty)"
    return type(value).__name__


def keyset(value: Any, path: str = "$") -> dict[str, str]:
    """Return ``{path: type}`` for *value* and everything below it.

    Dict keys extend the path with ``.key`` (``key`` at the top level), list
    elements with ``[]``; the element sets of one list are merged, and a path
    whose type differs between elements records the sorted ``a|b`` union.
    """
    found: dict[str, set[str]] = {}

    def walk(node: Any, at: str) -> None:
        found.setdefault(at, set()).add(_type_name(node))
        if isinstance(node, dict):
            for key, child in node.items():
                walk(child, key if at == "$" else f"{at}.{key}")
        elif isinstance(node, list):
            for child in node:
                walk(child, f"{at}[]")

    walk(value, path)
    return {key: "|".join(sorted(types)) for key, types in sorted(found.items())}


def _matches(path: str, patterns: tuple[str, ...]) -> bool:
    for pattern in patterns:
        if pattern.endswith("*"):
            if path.startswith(pattern[:-1]):
                return True
        elif path == pattern:
            return True
    return False


def compare_keysets(tool: str, actual: dict[str, str], golden: dict[str, str]) -> list[str]:
    """Return every way *actual* departs from *golden* beyond the allowed diff."""
    additions = ALLOWED_ADDITIONS.get(tool, ())
    changes = ALLOWED_CHANGES.get(tool, ())
    problems: list[str] = []
    for path, kind in golden.items():
        if _matches(path, changes):
            continue
        if path not in actual:
            problems.append(f"missing {path} ({kind})")
        elif actual[path] != kind:
            problems.append(f"type of {path}: golden {kind}, now {actual[path]}")
    for path, kind in actual.items():
        if path in golden or _matches(path, changes) or _matches(path, additions):
            continue
        problems.append(f"unlisted addition {path} ({kind})")
    return problems


def strip_allowed_values(payload: dict[str, Any]) -> dict[str, Any]:
    """Return *payload* without the requirement-5 keys of its ``entries[]``."""
    stripped = dict(payload)
    stripped["entries"] = [
        {key: value for key, value in entry.items() if key not in VALUE_STRIP_KEYS}
        for entry in payload.get("entries", [])
    ]
    return stripped


def _dump(value: Any) -> str:
    return json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n"


# ---------------------------------------------------------------------------
# Regeneration gate
# ---------------------------------------------------------------------------


class RegenRefused(RuntimeError):
    """Raised when goldens would be written from a tree other than B1."""


def _regen_requested() -> bool:
    return os.environ.get("ARIEL_KEYSETS_REGEN") == "1"


def expected_b1_sha() -> str | None:
    sha = os.environ.get("ARIEL_KEYSETS_B1_SHA")
    if sha:
        return sha.strip()
    if B1_SHA_FILE.is_file():
        lines = B1_SHA_FILE.read_text().splitlines()
        if lines and lines[0].strip():
            return lines[0].strip()
    return None


def check_regen_permitted() -> None:
    """Refuse unless ``HEAD`` is exactly the recorded B1 commit."""
    expected = expected_b1_sha()
    if not expected:
        raise RegenRefused("no B1 SHA: set ARIEL_KEYSETS_B1_SHA or .claude/scratch/b1-merged-sha")
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    if head != expected:
        raise RegenRefused(f"HEAD {head} is not the B1 commit {expected}; goldens stay B1's")


def _write_golden(path: Path, text: str) -> None:
    check_regen_permitted()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


# ---------------------------------------------------------------------------
# Tool runs
# ---------------------------------------------------------------------------


def _status_result() -> Any:
    from osprey.services.ariel_search.models import ARIELStatusResult, EmbeddingTableInfo

    return ARIELStatusResult(
        healthy=True,
        database_connected=True,
        database_uri="postgresql://***@localhost:5432/ariel",
        entry_count=1,
        embedding_tables=[
            EmbeddingTableInfo(
                table_name="text_embeddings_nomic_embed_text",
                entry_count=1,
                dimension=4,
                is_active=True,
            )
        ],
        active_embedding_model="nomic-embed-text",
        enabled_search_modules=["keyword", "semantic", "hybrid"],
        enabled_enhancement_modules=["text_embedding"],
        last_ingestion=datetime(2024, 6, 2, 9, 0, tzinfo=UTC),
        errors=["x"],
    )


def _keyset_picture_id() -> str:
    """The ``attachment_id`` of the seeded entry's viewable PNG."""
    from osprey.services.ariel_search.attachments import attachment_id_for

    return attachment_id_for(KEYSET_ENTRY_ID, keyset_entries()[0]["attachments"][0])


async def run_tool(tool: str, harness: Any) -> str:
    """Call *tool* the way an MCP client would and return its raw JSON text."""
    from osprey.mcp_server.ariel.tools.browse import browse
    from osprey.mcp_server.ariel.tools.capabilities import capabilities
    from osprey.mcp_server.ariel.tools.entry import entries_by_ids, entry_get
    from osprey.mcp_server.ariel.tools.hybrid_search import hybrid_search
    from osprey.mcp_server.ariel.tools.keyword_search import keyword_search
    from osprey.mcp_server.ariel.tools.semantic_search import semantic_search
    from osprey.mcp_server.ariel.tools.status import status

    if tool == "keyword_search":
        return await get_tool_fn(keyword_search)(query="injection")
    if tool == "semantic_search":
        return await get_tool_fn(semantic_search)(query="injection")
    if tool == "hybrid_search":
        return await get_tool_fn(hybrid_search)(query="injection")
    if tool == "browse":
        return await get_tool_fn(browse)()
    if tool == "entry_get":
        return await get_tool_fn(entry_get)(entry_id=KEYSET_ENTRY_ID)
    if tool == "entries_by_ids":
        return await get_tool_fn(entries_by_ids)(entry_ids=[KEYSET_ENTRY_ID])
    if tool == "capabilities":
        return await get_tool_fn(capabilities)()
    if tool == "status":
        with patch.object(harness.service, "get_status", AsyncMock(return_value=_status_result())):
            return await get_tool_fn(status)()
    if tool == "entry_open":
        from osprey.mcp_server.ariel.tools.entry import entry_open

        with patch("osprey.mcp_server.http.notify_panel_focus"):
            return await get_tool_fn(entry_open)(
                entry_id=KEYSET_ENTRY_ID, attachment_id=_keyset_picture_id()
            )
    if tool == "attachment_to_artifact":
        from osprey.mcp_server.ariel.tools.attachment import attachment_to_artifact
        from osprey.stores import artifact_store as artifact_store_module

        store = artifact_store_module.ArtifactStore(
            workspace_root=Path.cwd() / "agent_data", auto_launch=False
        )
        with (
            patch.object(artifact_store_module, "_artifact_store", store),
            patch("osprey.mcp_server.http._post_json_with_response", return_value=(200, {})),
        ):
            return await get_tool_fn(attachment_to_artifact)(attachment_id=_keyset_picture_id())
    raise AssertionError(f"no run for {tool}")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("tool", TOOLS)
async def test_tool_keyset_matches_golden(tool, keyset_harness):
    """The tool's key/type set is the golden's plus only the listed changes."""
    raw = await run_tool(tool, keyset_harness)
    assert "Mock" not in raw, f"{tool} serialized a mock: {raw[:400]}"
    actual = keyset(json.loads(raw))
    golden_path = KEYSET_DIR / f"{tool}.json"

    if _regen_requested():
        _write_golden(golden_path, _dump(actual))

    golden = json.loads(golden_path.read_text())
    problems = compare_keysets(tool, actual, golden)
    assert not problems, f"{tool} departs from its golden:\n" + "\n".join(problems)


@pytest.mark.parametrize("keyset_harness", [{"view_enabled": False}], indirect=True)
@pytest.mark.parametrize(
    "tool", [tool for tool in TOOLS if tool not in ("capabilities", "attachment_to_artifact")]
)
async def test_tool_keyset_equals_golden_with_view_off(tool, keyset_harness):
    """With ``ariel.attachments.view.enabled: false`` every tool emits B1's shape exactly.

    Listings (search, browse, ``entries_by_ids``) carry no ``attachments``,
    ``attachment_count`` or ``matched_attachment_ids``, and ``entry_get``
    carries the stored ``attachments`` unchanged. ``capabilities`` reports the
    switch itself, and ``attachment_to_artifact`` is refused with the view off.
    """
    raw = await run_tool(tool, keyset_harness)
    actual = keyset(json.loads(raw))
    golden = json.loads((KEYSET_DIR / f"{tool}.json").read_text())
    assert actual == golden, f"{tool} with the view off departs from B1's golden"


@pytest.mark.parametrize("tool", VALUE_TOOLS)
async def test_tool_values_match_golden(tool, keyset_harness):
    """With only the allowed keys removed, the output equals the B1 value golden."""
    payload = json.loads(await run_tool(tool, keyset_harness))
    assert len(payload["entries"]) == 2, "both seeded entries must be in the value golden"
    actual = _dump(strip_allowed_values(payload))
    golden_path = VALUE_DIR / f"{tool}.json"

    if _regen_requested():
        _write_golden(golden_path, actual)

    assert actual == golden_path.read_text()


@pytest.mark.parametrize("tool", TOOLS)
def test_golden_is_mock_free_and_seeded(tool):
    """A golden never freezes a mock, and every reshaped list is captured non-empty."""
    text = (KEYSET_DIR / f"{tool}.json").read_text()
    assert "Mock" not in text
    golden = json.loads(text)
    for list_path in SEEDED_LISTS.get(tool, ()):
        assert golden.get(list_path) == "list", f"{tool}.{list_path} must be seeded non-empty"
    if tool in VALUE_TOOLS:
        assert "Mock" not in (VALUE_DIR / f"{tool}.json").read_text()


async def test_capabilities_lists_hybrid_mode(keyset_harness):
    """The mode list is observed, so a regression to ``[]`` fails instead of freezing."""
    payload = json.loads(await run_tool("capabilities", keyset_harness))
    assert payload["search_modes"], "search_modes must be non-empty"
    assert "hybrid" in payload["search_modes"]
    golden = json.loads((KEYSET_DIR / "capabilities.json").read_text())
    assert golden["search_modes[]"] == "str"


def test_regen_refuses_off_b1(monkeypatch):
    """Regeneration refuses when HEAD is not the recorded B1 commit."""
    monkeypatch.setenv("ARIEL_KEYSETS_B1_SHA", "0" * 40)
    with pytest.raises(RegenRefused):
        check_regen_permitted()


def test_comparison_flags_type_changes_and_unlisted_additions():
    """The comparator rejects what requirement 5 does not list and accepts what it does."""
    golden = {"$": "dict", "entries": "list", "entries[]": "dict", "entries[].score": "float"}
    allowed = {**golden, "entries[].matched_via": "list", "entries[].matched_via[]": "str"}
    assert compare_keysets("keyword_search", allowed, golden) == []
    assert compare_keysets("browse", allowed, golden) == [
        "unlisted addition entries[].matched_via (list)",
        "unlisted addition entries[].matched_via[] (str)",
    ]
    retyped = {**golden, "entries[].score": "str"}
    assert compare_keysets("keyword_search", retyped, golden) == [
        "type of entries[].score: golden float, now str"
    ]
    emptied = {"$": "dict", "entries": "list(empty)"}
    assert compare_keysets("keyword_search", emptied, golden)[0] == (
        "type of entries: golden list, now list(empty)"
    )
