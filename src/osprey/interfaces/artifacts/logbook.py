"""Logbook Entry Composer — compose and submit ARIEL logbook entries from the gallery.

Gathers metadata about artifacts/context entries + the session audit trail,
calls Claude Haiku to compose a narrative logbook entry, then submits as an
ARIEL draft for human review in the ARIEL web form.
"""

from __future__ import annotations

import json
import logging
import os
import re
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from osprey.mcp_server.common import (
    gather_session_metadata,
    notify_panel_focus,
    resolve_workspace_root,
)

logger = logging.getLogger("osprey.interfaces.artifacts.logbook")

logbook_router = APIRouter(prefix="/api/logbook")

# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class ComposeRequest(BaseModel):
    artifact_id: str | None = None
    context_id: int | None = None


class ComposeResponse(BaseModel):
    subject: str
    details: str
    tags: list[str]
    artifact_ids: list[str]


class SubmitRequest(BaseModel):
    subject: str
    details: str
    author: str | None = None
    logbook: str | None = None
    shift: str | None = None
    tags: list[str] | None = None
    artifact_ids: list[str] | None = None


class SubmitResponse(BaseModel):
    draft_id: str
    url: str
    message: str


# ---------------------------------------------------------------------------
# Context gathering
# ---------------------------------------------------------------------------


@dataclass
class ComposedContext:
    artifact_meta: dict[str, Any] | None
    context_meta: dict[str, Any] | None
    audit_trail: list[dict[str, Any]]


async def gather_context(
    artifact_id: str | None,
    context_id: int | None,
    artifact_store: Any,
    context_store: Any,
    project_dir: Path,
) -> ComposedContext:
    """Gather metadata from artifact/context stores and the session transcript."""
    artifact_meta: dict[str, Any] | None = None
    context_meta: dict[str, Any] | None = None

    if artifact_id is not None:
        entry = artifact_store.get_entry(artifact_id)
        if entry is None:
            raise HTTPException(status_code=404, detail=f"Artifact {artifact_id} not found")
        artifact_meta = entry.to_dict()

    if context_id is not None:
        # DataContext has been removed; context_id lookups always 404
        raise HTTPException(status_code=404, detail=f"Context entry {context_id} not found")

    # Read audit trail from transcript
    audit_trail: list[dict[str, Any]] = []
    try:
        from osprey.mcp_server.workspace.transcript_reader import TranscriptReader

        reader = TranscriptReader(project_dir)
        events = reader.read_current_session()
        audit_trail = events[-20:] if len(events) > 20 else events
    except Exception:
        logger.warning("Could not read session transcript", exc_info=True)

    return ComposedContext(
        artifact_meta=artifact_meta,
        context_meta=context_meta,
        audit_trail=audit_trail,
    )


# ---------------------------------------------------------------------------
# LLM composition
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are a logbook entry composer for a particle accelerator control room. \
Write concise, technical logbook entries suitable for operator shift logs. \
Focus on what was done, what was observed, and any anomalies. \
Use clear, factual language. Avoid speculation.

Respond with a JSON object containing exactly these fields:
- "subject": a short title (max 120 chars)
- "details": the narrative body (1-3 paragraphs, plain text)
- "tags": a list of 2-5 relevant tags (lowercase, no spaces)
"""


def _build_user_prompt(ctx: ComposedContext) -> str:
    """Build the user prompt from gathered context."""
    sections: list[str] = []

    if ctx.artifact_meta:
        sections.append(
            "## Artifact\n"
            f"Title: {ctx.artifact_meta.get('title', 'N/A')}\n"
            f"Type: {ctx.artifact_meta.get('artifact_type', 'N/A')}\n"
            f"Description: {ctx.artifact_meta.get('description', 'N/A')}\n"
            f"Created: {ctx.artifact_meta.get('timestamp', 'N/A')}"
        )

    if ctx.context_meta:
        sections.append(
            "## Data Context\n"
            f"Tool: {ctx.context_meta.get('tool', 'N/A')}\n"
            f"Description: {ctx.context_meta.get('description', 'N/A')}\n"
            f"Data type: {ctx.context_meta.get('data_type', 'N/A')}\n"
            f"Summary: {json.dumps(ctx.context_meta.get('summary', {}), default=str)}"
        )

    if ctx.audit_trail:
        trail_lines = []
        for evt in ctx.audit_trail[-10:]:
            tool = evt.get("tool", evt.get("type", "?"))
            ts = evt.get("timestamp", "")[:19]
            trail_lines.append(f"  {ts} {tool}")
        sections.append("## Recent session activity\n" + "\n".join(trail_lines))

    return "\n\n".join(sections) or "No context available."


def _clean_llm_json(text: str) -> str:
    """Strip markdown code fences and preamble from an LLM JSON response."""
    # Strip markdown code fences
    if text.startswith("```json"):
        text = text[7:]
    elif text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    text = text.strip()

    # If text still doesn't start with { or [, try to extract JSON object
    if not text.startswith("{") and not text.startswith("["):
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            text = match.group(0)

    return text


async def compose_entry(ctx: ComposedContext) -> ComposeResponse:
    """Call the configured LLM provider to compose a logbook entry.

    Uses ``aget_chat_completion()`` with model config from ``config.yml``
    (``logbook_composition`` role, falling back to ``response``).
    """
    from osprey.models.completion import aget_chat_completion
    from osprey.models.messages import ChatCompletionRequest, ChatMessage
    from osprey.utils.config import get_model_config

    # Resolve model config: logbook_composition → response (fallback)
    # get_model_config returns {} (not raises) when the key is missing,
    # so check for a non-empty dict with at least a "provider" key.
    model_config = get_model_config("logbook_composition")
    if not model_config or "provider" not in model_config:
        model_config = get_model_config("response")
    if not model_config or "provider" not in model_config:
        raise HTTPException(
            status_code=503,
            detail=(
                "No model config for 'logbook_composition' or 'response' in config.yml. "
                "Add a logbook_composition entry under 'models:' or set a response model."
            ),
        )

    try:
        chat_request = ChatCompletionRequest(messages=[
            ChatMessage(role="system", content=SYSTEM_PROMPT),
            ChatMessage(role="user", content=_build_user_prompt(ctx)),
        ])

        text = await aget_chat_completion(
            chat_request=chat_request,
            model_config=model_config,
        )

        # Clean markdown fences / preamble before parsing JSON
        raw = str(text).strip()
        raw = _clean_llm_json(raw)
        parsed = json.loads(raw)

        artifact_ids: list[str] = []
        if ctx.artifact_meta:
            artifact_ids.append(ctx.artifact_meta["id"])

        return ComposeResponse(
            subject=parsed.get("subject", "Untitled entry"),
            details=parsed.get("details", ""),
            tags=parsed.get("tags", []),
            artifact_ids=artifact_ids,
        )
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=503, detail=f"LLM returned invalid JSON: {exc}") from exc
    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(status_code=503, detail=f"LLM provider error: {exc}") from exc
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"LLM API error: {exc}") from exc


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@logbook_router.post("/compose", response_model=ComposeResponse)
async def compose(req: ComposeRequest, request: Request):
    """Gather context and compose a logbook entry draft via LLM."""
    if req.artifact_id is None and req.context_id is None:
        raise HTTPException(
            status_code=422,
            detail="At least one of artifact_id or context_id is required",
        )

    store = request.app.state.artifact_store
    project_dir = Path.cwd()

    ctx = await gather_context(
        artifact_id=req.artifact_id,
        context_id=req.context_id,
        artifact_store=store,
        context_store=None,
        project_dir=project_dir,
    )

    return await compose_entry(ctx)


@logbook_router.post("/submit", response_model=SubmitResponse)
async def submit(req: SubmitRequest):
    """Submit the composed entry as an ARIEL draft."""
    try:
        draft_id = f"draft-{uuid.uuid4().hex[:12]}"
        drafts_dir = resolve_workspace_root() / "drafts"
        drafts_dir.mkdir(parents=True, exist_ok=True)

        draft_data: dict[str, Any] = {
            "draft_id": draft_id,
            "subject": req.subject.strip(),
            "details": req.details.strip(),
            "author": req.author,
            "logbook": req.logbook,
            "shift": req.shift,
            "tags": req.tags or [],
            "metadata": {
                "session_metadata": gather_session_metadata("gallery-compose"),
            },
        }

        # Resolve artifact attachments if provided
        if req.artifact_ids:
            try:
                from osprey.interfaces.ariel.mcp.tools.entry import _resolve_artifacts

                artifact_paths = await _resolve_artifacts(req.artifact_ids)
                draft_data["attachment_paths"] = artifact_paths
            except Exception as exc:
                raise HTTPException(
                    status_code=500, detail=f"Failed to resolve artifacts: {exc}"
                ) from exc

        filepath = drafts_dir / f"{draft_id}.json"
        filepath.write_text(json.dumps(draft_data, indent=2))

        base_url = os.environ.get("ARIEL_WEB_URL", "http://127.0.0.1:8085")
        url = f"{base_url}/#create?draft={draft_id}"

        # Notify web terminal to switch to ARIEL panel (non-fatal)
        try:
            notify_panel_focus("ariel", url=url)
        except Exception:
            pass

        return SubmitResponse(
            draft_id=draft_id,
            url=url,
            message=f"Draft {draft_id} created. Open the URL to review and submit: {url}",
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=500, detail=f"Failed to create draft: {exc}"
        ) from exc
