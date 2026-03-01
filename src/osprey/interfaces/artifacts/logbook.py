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


class AssembleRequest(BaseModel):
    purpose: str = "general"
    detail_level: str = "standard"
    nudge: str | None = None


class AssembleResponse(BaseModel):
    prompt: str


class ComposeRequest(BaseModel):
    artifact_id: str | None = None
    purpose: str | None = None
    detail_level: str | None = None
    nudge: str | None = None
    custom_prompt: str | None = None
    # Context controls
    include_session_log: bool = True
    artifact_ids: list[str] | None = None  # overrides artifact_id when provided
    # Model selection
    model: str | None = None  # "haiku" | "sonnet" | "opus" → maps to config


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
    artifacts_meta: list[dict[str, Any]]  # multi-artifact support
    audit_trail: list[dict[str, Any]]
    chat_history: list[dict[str, Any]]


async def gather_context(
    artifact_id: str | None,
    artifact_store: Any,
    project_dir: Path,
    *,
    artifact_ids: list[str] | None = None,
    include_session_log: bool = True,
) -> ComposedContext:
    """Gather metadata from artifact store and the session transcript.

    If *artifact_ids* is provided, those artifacts are loaded (multi-artifact
    mode) and *artifact_id* is ignored.  If *include_session_log* is False
    the audit trail is skipped.
    """
    artifact_meta: dict[str, Any] | None = None
    artifacts_meta: list[dict[str, Any]] = []

    if artifact_ids is not None:
        # Multi-artifact mode
        for aid in artifact_ids:
            entry = artifact_store.get_entry(aid)
            if entry is not None:
                artifacts_meta.append(entry.to_dict())
        # Set artifact_meta to first for backward compat (artifact_ids in response)
        if artifacts_meta:
            artifact_meta = artifacts_meta[0]
    elif artifact_id is not None:
        entry = artifact_store.get_entry(artifact_id)
        if entry is None:
            raise HTTPException(status_code=404, detail=f"Artifact {artifact_id} not found")
        artifact_meta = entry.to_dict()
        artifacts_meta = [artifact_meta]

    # Read audit trail and chat history from transcript
    audit_trail: list[dict[str, Any]] = []
    chat_history: list[dict[str, Any]] = []
    if include_session_log:
        try:
            from osprey.mcp_server.workspace.transcript_reader import TranscriptReader

            reader = TranscriptReader(project_dir)
            events = reader.read_current_session()
            audit_trail = events[-20:] if len(events) > 20 else events
            chat_history = reader.read_current_chat_history()
        except Exception:
            logger.warning("Could not read session transcript", exc_info=True)

    return ComposedContext(
        artifact_meta=artifact_meta,
        artifacts_meta=artifacts_meta,
        audit_trail=audit_trail,
        chat_history=chat_history,
    )


# ---------------------------------------------------------------------------
# Prompt templates & assembly
# ---------------------------------------------------------------------------

JSON_FORMAT_INSTRUCTIONS = (
    "Respond with a JSON object containing exactly these fields:\n"
    '- "subject": a short title (max 120 chars)\n'
    '- "details": the narrative body (1-3 paragraphs, plain text)\n'
    '- "tags": a list of 2-5 relevant tags (lowercase, no spaces)'
)

# Legacy fixed prompt (backward compatibility when no steering fields provided)
SYSTEM_PROMPT = (
    "You are a logbook entry composer for a particle accelerator control room. "
    "Write concise, technical logbook entries suitable for operator shift logs.\n\n"
    "STRICT RULES:\n"
    "- State ONLY what was done and what the data shows. Never interpret intent.\n"
    "- Never speculate about WHY something was done or what it might mean.\n"
    "- Never claim to see trends, anomalies, drift, or stability unless the "
    "data explicitly quantifies them.\n"
    "- If you cannot verify a claim from the provided data, do not make it.\n\n"
    + JSON_FORMAT_INSTRUCTIONS
)

BASE_PREAMBLE = (
    "You are a logbook entry composer for a particle accelerator control room. "
    "Write clean, technical entries suitable for operator shift logs.\n\n"
    "STRICT RULES — violations make the entry useless:\n"
    "- State ONLY verifiable facts: what was requested, what tools were called, "
    "what data was produced. Never infer operator intent or motivation.\n"
    "- Do NOT interpret results. Do not say data looks 'stable', 'nominal', "
    "'within tolerance', or 'anomalous' unless the operator explicitly said so.\n"
    "- Do NOT add analysis or conclusions. A plot was created — you do not know "
    "what the operator sees in it.\n"
    "- Describe the artifact (what it shows, what channels, what time range) "
    "without editorializing about what it means.\n"
    "- Use the conversation log to understand what was asked for, not to "
    "speculate about why."
)

PURPOSE_FRAGMENTS = {
    "observation": (
        "This is an OBSERVATION entry. Write ONLY what was done and what was "
        "produced. Example of correct tone: 'Created 3D scatter plot of BPM18, "
        "BPM19, BPM20 horizontal positions over 24h, color-coded by time.'\n"
        "Example of WRONG tone: 'Plot was generated to assess beam trajectory "
        "stability.' — you do not know the operator's intent.\n"
        "Example of WRONG tone: 'No anomalies observed.' — you cannot see the plot."
    ),
    "action_taken": (
        "This is an ACTION TAKEN entry. Record what action was performed and "
        "what the direct result was. State the action, the parameters used, and "
        "the outcome. Do not speculate about why the action was taken unless the "
        "operator explicitly stated the reason in the conversation."
    ),
    "anomaly": (
        "This is an ANOMALY entry. Describe the unexpected behavior using only "
        "facts from the conversation and data. State what was observed, what "
        "values were seen, and what the expected values were. Do not speculate "
        "about causes unless the operator identified them."
    ),
    "investigation": (
        "This is an INVESTIGATION entry. Record what was examined, what data "
        "was pulled, and what the operator found. Only state conclusions the "
        "operator explicitly reached in the conversation."
    ),
    "routine_check": (
        "This is a ROUTINE CHECK entry. List what was checked and the readings "
        "obtained. State values without characterizing them as normal or abnormal "
        "unless the operator said so."
    ),
    "general": (
        "Write a factual logbook entry based on the provided context. Describe "
        "what was done and what was produced. Do not add interpretation."
    ),
}

DETAIL_FRAGMENTS = {
    "brief": "Keep the entry to 1-2 sentences. Just the essential facts.",
    "standard": (
        "Write 1 short paragraph covering: what was requested, what data/artifacts "
        "were produced, and the key parameters (channels, time range, etc.)."
    ),
    "detailed": (
        "Write 2-3 paragraphs covering: the operator's request, the data sources "
        "and parameters used, and what artifacts were produced. Include specific "
        "channel names, time ranges, and numerical values from the data. Do not "
        "add interpretation beyond what the operator stated."
    ),
}


def assemble_prompt(
    purpose: str = "general",
    detail_level: str = "standard",
    nudge: str | None = None,
) -> str:
    """Deterministically assemble a system prompt from steering selections."""
    purpose_text = PURPOSE_FRAGMENTS.get(purpose, PURPOSE_FRAGMENTS["general"])
    detail_text = DETAIL_FRAGMENTS.get(detail_level, DETAIL_FRAGMENTS["standard"])

    parts = [BASE_PREAMBLE, purpose_text, detail_text, JSON_FORMAT_INSTRUCTIONS]
    if nudge and nudge.strip():
        parts.append(f"Additional operator guidance: {nudge.strip()}")

    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# LLM composition
# ---------------------------------------------------------------------------


def _format_artifact_section(label: str, meta: dict[str, Any]) -> str:
    """Format a single artifact's metadata for the LLM prompt."""
    lines = [
        f"## {label}",
        f"Title: {meta.get('title', 'N/A')}",
        f"Type: {meta.get('artifact_type', 'N/A')}",
        f"Description: {meta.get('description', 'N/A')}",
        f"Created: {meta.get('timestamp', 'N/A')}",
    ]
    if meta.get("category"):
        lines.append(f"Category: {meta['category']}")
    if meta.get("summary"):
        lines.append(f"Summary: {json.dumps(meta['summary'], default=str)}")
    return "\n".join(lines)


def _build_user_prompt(ctx: ComposedContext) -> str:
    """Build the user prompt from gathered context."""
    sections: list[str] = []

    if ctx.artifacts_meta:
        for i, meta in enumerate(ctx.artifacts_meta):
            label = "Artifact" if len(ctx.artifacts_meta) == 1 else f"Artifact {i + 1}"
            sections.append(_format_artifact_section(label, meta))
    elif ctx.artifact_meta:
        sections.append(_format_artifact_section("Artifact", ctx.artifact_meta))

    if ctx.chat_history:
        chat_lines = []
        for turn in ctx.chat_history[-30:]:
            role = turn.get("role", "?").upper()
            ts = turn.get("timestamp", "")[:19]
            content = turn.get("content", "")
            chat_lines.append(f"[{ts}] {role}: {content}")
        sections.append("## Conversation log\n" + "\n\n".join(chat_lines))

    if ctx.audit_trail:
        trail_lines = []
        for evt in ctx.audit_trail[-10:]:
            tool = evt.get("tool", evt.get("type", "?"))
            ts = evt.get("timestamp", "")[:19]
            args = evt.get("arguments", {})
            result = evt.get("result_summary", "")
            line = f"  {ts} {tool}"
            if args:
                line += f" args={json.dumps(args, default=str)}"
            if result:
                line += f" → {result[:200]}"
            trail_lines.append(line)
        sections.append("## Recent session activity\n" + "\n".join(trail_lines))

    return "\n\n".join(sections) or "No context available."


def _clean_llm_json(text: str) -> str:
    """Strip markdown code fences and preamble from an LLM JSON response."""
    if text.startswith("```json"):
        text = text[7:]
    elif text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    text = text.strip()

    if not text.startswith("{") and not text.startswith("["):
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            text = match.group(0)

    return text


# Valid model tier names for the model selector
VALID_TIERS = {"haiku", "sonnet", "opus"}

# Default composition config when logbook.composition is absent from config.yml
_DEFAULT_COMPOSITION = {
    "provider": "cborg",
    "model_id": "haiku",
    "default_tier": "haiku",
}


def _resolve_composition_model(
    model: str | None = None,
) -> tuple[str, str]:
    """Resolve provider and model_id for logbook composition.

    Resolution order:
    1. If *model* is a tier name (haiku/sonnet/opus), look up model_id
       from ``api.providers[provider].models[tier]``.
    2. Otherwise use ``logbook.composition.model_id`` from config.yml.
    3. Falls back to built-in defaults (cborg / anthropic/claude-haiku).

    Returns:
        (provider, model_id) tuple
    """
    from osprey.utils.config import get_config_value, get_provider_config

    # Read logbook.composition section (or fall back to defaults)
    comp = get_config_value("logbook.composition", {})
    if not isinstance(comp, dict):
        comp = {}
    provider = comp.get("provider", _DEFAULT_COMPOSITION["provider"])
    default_model_id = comp.get("model_id", _DEFAULT_COMPOSITION["model_id"])
    default_tier = comp.get("default_tier", _DEFAULT_COMPOSITION["default_tier"])

    # Determine which tier to use
    tier = model if model and model in VALID_TIERS else default_tier

    # Look up tier → model_id from the provider's models mapping
    provider_cfg = get_provider_config(provider)
    if not provider_cfg:
        raise HTTPException(
            status_code=503,
            detail=(
                f"Provider '{provider}' not found in api.providers. "
                "Check logbook.composition.provider in config.yml."
            ),
        )
    provider_models = provider_cfg.get("models", {})
    model_id = provider_models.get(tier, default_model_id)

    return provider, model_id


async def compose_entry(
    ctx: ComposedContext,
    system_prompt: str | None = None,
    model: str | None = None,
) -> ComposeResponse:
    """Call the configured LLM provider to compose a logbook entry.

    Uses ``aget_chat_completion()`` with provider + model_id resolved from
    ``logbook.composition`` in config.yml and ``api.providers`` tier mappings.

    If *system_prompt* is provided it is used directly; otherwise the
    legacy ``SYSTEM_PROMPT`` is used for backward compatibility.

    If *model* is a tier name (haiku/sonnet/opus) the corresponding model_id
    is looked up from the provider's models mapping.
    """
    from osprey.models.completion import aget_chat_completion
    from osprey.models.messages import ChatCompletionRequest, ChatMessage

    provider, model_id = _resolve_composition_model(model)

    effective_prompt = system_prompt if system_prompt is not None else SYSTEM_PROMPT

    try:
        chat_request = ChatCompletionRequest(
            messages=[
                ChatMessage(role="system", content=effective_prompt),
                ChatMessage(role="user", content=_build_user_prompt(ctx)),
            ]
        )

        text = await aget_chat_completion(
            chat_request=chat_request,
            provider=provider,
            model_id=model_id,
        )

        raw = str(text).strip()
        raw = _clean_llm_json(raw)
        parsed = json.loads(raw)

        artifact_ids: list[str] = []
        for meta in ctx.artifacts_meta:
            aid = meta.get("id")
            if aid:
                artifact_ids.append(aid)

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


@logbook_router.post("/assemble-prompt", response_model=AssembleResponse)
async def assemble_prompt_endpoint(req: AssembleRequest):
    """Assemble a system prompt from steering selections (no LLM call)."""
    return AssembleResponse(
        prompt=assemble_prompt(
            purpose=req.purpose,
            detail_level=req.detail_level,
            nudge=req.nudge,
        )
    )


@logbook_router.post("/compose", response_model=ComposeResponse)
async def compose(req: ComposeRequest, request: Request):
    """Gather context and compose a logbook entry draft via LLM."""
    # artifact_ids overrides artifact_id; at least one source required
    if not req.artifact_ids and not req.artifact_id:
        raise HTTPException(
            status_code=422,
            detail="At least one of artifact_id or artifact_ids is required",
        )

    store = request.app.state.artifact_store
    project_dir = Path.cwd()

    # Resolve "all" sentinel → load every artifact from store
    effective_artifact_ids = req.artifact_ids
    if effective_artifact_ids == ["all"]:
        effective_artifact_ids = [e.id for e in store.list_entries()]

    ctx = await gather_context(
        artifact_id=req.artifact_id,
        artifact_store=store,
        project_dir=project_dir,
        artifact_ids=effective_artifact_ids,
        include_session_log=req.include_session_log,
    )

    # Resolve system prompt: custom_prompt > steering fields > legacy default
    system_prompt: str | None = None
    if req.custom_prompt:
        system_prompt = req.custom_prompt
    elif req.purpose or req.detail_level:
        system_prompt = assemble_prompt(
            purpose=req.purpose or "general",
            detail_level=req.detail_level or "standard",
            nudge=req.nudge,
        )

    return await compose_entry(ctx, system_prompt=system_prompt, model=req.model)


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

        # Create metadata.json attachment (belt-and-suspenders persistence)
        metadata_json = json.dumps(draft_data["metadata"], indent=2)
        meta_file = drafts_dir / f"{draft_id}-metadata.json"
        meta_file.write_text(metadata_json)
        draft_data["attachment_paths"] = [str(meta_file)]

        # Resolve artifact attachments if provided
        if req.artifact_ids:
            try:
                from osprey.interfaces.ariel.mcp.tools.entry import _resolve_artifacts

                artifact_paths = await _resolve_artifacts(req.artifact_ids)
                draft_data["attachment_paths"].extend(artifact_paths)
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
        raise HTTPException(status_code=500, detail=f"Failed to create draft: {exc}") from exc
