"""Byte-level pin of what the four read tools of the facility-knowledge server answer.

Six calls — ``capabilities``, ``list_concepts``, ``read_concept`` (one id that exists,
one that does not) and ``search`` (one query with hits, one without) — run in process
against the checked-in bundle in :mod:`tests.fixtures.qmd_okf`, with no search
sidecar, so ``search`` is the substring path. Their answers, parsed, are compared with
``golden/facility_knowledge_tools.json``. The bundle's absolute path, the one volatile
value in the answers, is replaced by a fixed marker.

A failure means the agent's knowledge tools answer differently. A later change to the
golden is a declared delta: the PR that makes it names the delta. Regenerate from the
tools, in the same change as the code that moved them, then read the diff::

    PYTHONPATH=src:.:packages/osprey-connectors/tests ./.venv/bin/python \
        tests/mcp_server/test_facility_knowledge_tool_json_golden.py
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest
from fastmcp.exceptions import ToolError

from tests.fixtures.qmd_okf import BUNDLE_ROOT
from tests.mcp_server._facility_bundle import install_bundle
from tests.mcp_server.conftest import get_tool_fn

_GOLDEN = Path(__file__).parent / "golden" / "facility_knowledge_tools.json"

#: The marker the bundle's absolute path is replaced by.
_BUNDLE_MARKER = "<BUNDLE_ROOT>"

#: The header keys the golden leads with.
_HEADER: dict[str, str] = {
    "_owner": "tests/mcp_server/test_facility_knowledge_tool_json_golden.py",
    "_rule": (
        "A change to this file is a declared delta: the PR that makes it names the "
        "delta. Regenerate with: PYTHONPATH=src:.:packages/osprey-connectors/tests "
        "./.venv/bin/python tests/mcp_server/test_facility_knowledge_tool_json_golden.py"
    ),
}

#: The six calls, in the order the golden lists them: tool name and arguments.
_CALLS: tuple[tuple[str, dict[str, str]], ...] = (
    ("capabilities", {}),
    ("list_concepts", {}),
    ("read_concept", {"concept_id": "magnets/quadrupoles"}),
    ("read_concept", {"concept_id": "no/such/concept"}),
    ("search", {"query": "steering magnets"}),
    ("search", {"query": "how do we keep the stored current constant during user shifts"}),
)


def _without_bundle_path(text: str) -> str:
    """*text* with every spelling of the bundle's absolute path replaced by the marker."""
    for spelling in sorted({str(BUNDLE_ROOT.resolve()), str(BUNDLE_ROOT)}, key=len, reverse=True):
        text = text.replace(spelling, _BUNDLE_MARKER)
    return text


async def _answers() -> list[dict[str, Any]]:
    """Each call with what the tool answered, or the error envelope it raised."""
    import osprey.mcp_server.facility_knowledge.server as srv

    records: list[dict[str, Any]] = []
    for name, args in _CALLS:
        record: dict[str, Any] = {"tool": name, "args": args}
        try:
            text = await get_tool_fn(getattr(srv, name))(**args)
        except ToolError as exc:
            record["error"] = json.loads(_without_bundle_path(str(exc)))
        else:
            record["answer"] = json.loads(_without_bundle_path(text))
        records.append(record)
    return records


def _document(records: list[dict[str, Any]]) -> dict[str, Any]:
    """The golden document: the header keys, then the calls."""
    return {**_HEADER, "calls": records}


async def test_tool_answers_equal_the_golden(monkeypatch: pytest.MonkeyPatch) -> None:
    install_bundle(monkeypatch, BUNDLE_ROOT)

    golden = json.loads(_GOLDEN.read_text(encoding="utf-8"))

    assert golden == _document(await _answers())


def test_golden_lists_all_six_calls() -> None:
    golden = json.loads(_GOLDEN.read_text(encoding="utf-8"))

    assert [(call["tool"], call["args"]) for call in golden["calls"]] == [
        (name, args) for name, args in _CALLS
    ]
    assert "error" in golden["calls"][3], "the missing concept must be pinned as an error"
    assert golden["calls"][4]["answer"]["count"] > 0, "the first search must have hits"
    assert golden["calls"][5]["answer"]["count"] == 0, "the second search must have none"


def _regenerate() -> None:
    """Overwrite the golden from what the tools answer today."""
    with pytest.MonkeyPatch.context() as monkeypatch:
        install_bundle(monkeypatch, BUNDLE_ROOT)
        records = asyncio.run(_answers())
    _GOLDEN.parent.mkdir(parents=True, exist_ok=True)
    _GOLDEN.write_text(
        json.dumps(_document(records), indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"wrote {_GOLDEN}")


if __name__ == "__main__":
    _regenerate()
