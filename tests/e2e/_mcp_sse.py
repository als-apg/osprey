"""Reading a tool's answer off the dispatcher's MCP transport.

The deploy-backed suites call ``manual_fire`` over the dispatcher's streamable
HTTP transport, directly or through the web terminal's panel proxy, and both
read the answer the same way. The reading lives here once because it has a trap
in it (see :func:`tool_result`) that a second copy is free to fall back into.
"""

from __future__ import annotations

import json


def sse_payloads(body: str) -> list[dict]:
    """Every JSON object an SSE body's ``data:`` lines carry.

    The dispatcher's MCP transport answers in ``text/event-stream`` frames even
    for a single response, so a caller reading the result unwraps the frame
    rather than json-loading the body. A line that is not JSON is skipped rather
    than raised on: the caller asserts on what it found, and a parse error here
    would replace that assertion with a traceback about framing.
    """
    payloads: list[dict] = []
    for line in body.splitlines():
        if not line.startswith("data:"):
            continue
        try:
            payloads.append(json.loads(line[len("data:") :].strip()))
        except json.JSONDecodeError:
            continue
    return payloads


def tool_result(payloads: list[dict]) -> dict:
    """The tool's own answer, parsed out of the JSON-RPC response.

    Three layers have to come off, and each one is a place an earlier draft of
    this helper got it wrong. The SSE frame carries a JSON-RPC envelope; the
    envelope's ``result`` carries MCP content parts; and the text part is
    ITSELF a JSON document, serialised as a string. Searching the envelope for a
    substring cannot work: re-serialising it escapes the inner document's quotes
    (``\\"dispatched\\": true``), so the match fails on a call that succeeded.

    Read from ``structuredContent.result`` when FastMCP wrapped it there and
    from the first text content part otherwise, because which one appears is a
    property of the server's wrapping rather than of the answer.
    """
    envelope = next((item for item in payloads if "result" in item), None)
    assert envelope is not None, f"no JSON-RPC result among the SSE payloads: {payloads}"
    result = envelope["result"]
    assert not result.get("isError"), f"the dispatcher reported a tool error: {result}"

    raw = (result.get("structuredContent") or {}).get("result")
    if raw is None:
        parts = result.get("content") or []
        text_parts = [part.get("text") for part in parts if part.get("type") == "text"]
        assert text_parts, f"the tool answered with no text content: {result}"
        raw = text_parts[0]
    if isinstance(raw, dict):
        return raw
    return json.loads(raw)


def answer_document(answer: str) -> dict | None:
    """The JSON document a persisted tool answer carries, or ``None`` if it carries none.

    A run record stores each tool answer as text, and that text is not always
    the tool's own document: FastMCP may wrap it as ``{"result": "<document>"}``
    with the document serialised as a string inside. Substring matching on the
    stored text then fails on the wrapped form, because the inner quotes are
    escaped — the same trap :func:`tool_result` describes, one layer further in.
    So the wrapping is taken off, however many times it was put on, and the
    caller asserts on keys rather than on spelling.

    A refusal is prose, not JSON, and answers ``None``: the caller's clause
    checks are what read those.
    """
    document: object = answer
    while isinstance(document, str):
        try:
            document = json.loads(document)
        except json.JSONDecodeError:
            return None
        if isinstance(document, dict) and set(document) == {"result"}:
            document = document["result"]
    return document if isinstance(document, dict) else None


def any_answer_succeeded(answers: list[str]) -> bool:
    """Whether any persisted tool answer is the tool's own success envelope."""
    return any((answer_document(answer) or {}).get("status") == "success" for answer in answers)
