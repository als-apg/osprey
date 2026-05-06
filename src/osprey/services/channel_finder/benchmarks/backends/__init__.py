"""Backend protocol and dispatch for benchmark execution.

Decouples *which model* from *which agent harness* in the cross-paradigm
benchmark. The Backend protocol lets us run the same model through either
harness so cell scores attribute cleanly.

Backends (the harness axis):
    sdk    — claude_agent_sdk.query() (Anthropic-native tool-use loop)
    react  — manual ReAct loop on top of litellm.acompletion()
    direct — single MCP tool call, no outer agent loop. Only valid for
             the ``in_context`` paradigm, whose query_channels tool already
             performs the full retrieval inside the MCP subprocess.

The ``direct`` backend is *not* a third option for hierarchical or
middle_layer; those paradigms expose multi-tool surfaces that require
SDK or ReAct to orchestrate.

Backends are constructed from a LiteLLM-form ``provider/wire_id`` model
string. Each backend splits that into provider + wire id and formats the
wire id into the grammar its consumer expects (bare wire id for the Claude
SDK CLI; provider-prefixed slug for LiteLLM).
"""

from __future__ import annotations

from pathlib import Path

from .base import Backend, WorkflowOutput
from .in_context_backend import InContextBackend
from .react_backend import ReactBackend
from .sdk_backend import SdkBackend

__all__ = [
    "Backend",
    "InContextBackend",
    "ReactBackend",
    "SdkBackend",
    "WorkflowOutput",
    "create_backend",
]


def _read_pipeline_mode(project_dir: Path) -> str | None:
    """Return ``channel_finder.pipeline_mode`` from config.yml, or None."""
    config_path = project_dir / "config.yml"
    if not config_path.exists():
        return None
    try:
        import yaml

        config = yaml.safe_load(config_path.read_text()) or {}
        return config.get("channel_finder", {}).get("pipeline_mode")
    except Exception:
        return None


def create_backend(
    name: str,
    project_dir: Path,
    model: str,
    *,
    max_turns: int = 25,
    max_budget_usd: float = 2.0,
) -> Backend:
    """Construct a backend by name.

    Args:
        name: One of ``"auto"``, ``"sdk"``, ``"react"``, ``"direct"``.
            ``"auto"`` checks ``channel_finder.pipeline_mode`` in config.yml
            first; if the project is configured for the ``in_context``
            paradigm, returns ``InContextBackend`` (i.e. the ``direct``
            backend). Otherwise selects ``react`` for ollama providers,
            ``sdk`` for all others.
        project_dir: OSPREY project root.
        model: LiteLLM-form ``provider/wire_id`` (e.g. ``"anthropic/claude-haiku-4-5"``,
            ``"ollama/gemma3:4b"``).
        max_turns: Max agentic turns per query.
        max_budget_usd: Per-query budget (sdk backend only).

    Raises:
        ValueError: For unknown backend names or invalid combinations
            (e.g. SDK + ollama).
    """
    provider = model.split("/", 1)[0]

    if name == "direct":
        return InContextBackend(project_dir, model)

    is_ollama = provider == "ollama"

    if name == "auto":
        if _read_pipeline_mode(project_dir) == "in_context":
            return InContextBackend(project_dir, model)
        name = "react" if is_ollama else "sdk"

    if name == "sdk":
        if is_ollama:
            raise ValueError(
                f"SDK backend does not support Ollama provider (model={model!r})"
            )
        return SdkBackend(project_dir, model, max_turns, max_budget_usd)

    if name == "react":
        return ReactBackend(project_dir, model, max_turns)

    raise ValueError(f"Unknown backend: {name!r}")
