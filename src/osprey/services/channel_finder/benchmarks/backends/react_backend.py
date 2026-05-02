"""ReAct backend — wraps the manual ``litellm.acompletion()`` ReAct loop."""

from __future__ import annotations

import logging
import os
from pathlib import Path

from osprey.services.channel_finder.benchmarks.harness import (
    combined_text_from_react,
    mcp_client_session,
    run_react_query,
)
from osprey.services.channel_finder.benchmarks.sdk import _read_agent_prompt
from osprey.services.channel_finder.rate_limiter import configure_rate_limiter

# Per-provider LiteLLM call rate caps (calls per minute). Set conservatively
# below the documented limit to leave a small safety margin. ``None`` disables
# throttling for that provider.
_PROVIDER_RATE_LIMIT_RPM: dict[str, int | None] = {
    "cborg": 18,  # CBORG free tier is 20 req/min/key
    "anthropic": None,  # Direct Anthropic — no proxy throttle needed
    "als-apg": None,
}

from .base import Backend, WorkflowOutput

logger = logging.getLogger(__name__)


def _resolve_provider_name(project_dir: Path) -> str | None:
    """Read the configured ``claude_code.provider`` from project config.yml."""
    config_path = project_dir / "config.yml"
    if not config_path.exists():
        return None
    try:
        import yaml

        config = yaml.safe_load(config_path.read_text()) or {}
        return config.get("claude_code", {}).get("provider")
    except Exception:
        return None


def _resolve_litellm_endpoint(project_dir: Path, model: str) -> dict | None:
    """Resolve provider routing kwargs for a non-ollama model.

    The SDK path injects ``ANTHROPIC_BASE_URL`` + ``ANTHROPIC_AUTH_TOKEN``
    into the subprocess environment via ``inject_provider_env``. LiteLLM
    does NOT read ``ANTHROPIC_BASE_URL`` (it reads ``ANTHROPIC_API_BASE``),
    so env inheritance can't carry the override — we have to pass
    ``api_base`` / ``api_key`` explicitly to ``litellm.acompletion()``.

    Returns ``None`` for ollama (already handled by ``_litellm_call_kwargs``)
    and for direct Anthropic (LiteLLM's default routing is correct).
    """
    if model.startswith(("ollama/", "ollama_chat/")):
        return None

    try:
        import yaml

        from osprey.cli.claude_code_resolver import ClaudeCodeModelResolver
    except ImportError:  # pragma: no cover — optional dep
        logger.warning("Could not import ClaudeCodeModelResolver; skipping endpoint resolution")
        return None

    config_path = project_dir / "config.yml"
    if not config_path.exists():
        return None

    config = yaml.safe_load(config_path.read_text()) or {}
    cc_config = config.get("claude_code", {})
    api_providers = config.get("api", {}).get("providers", {})
    spec = ClaudeCodeModelResolver.resolve(cc_config, api_providers)
    if spec is None:
        return None

    # ``upstream_base_url`` is only set when the proxy is needed; cborg/als-apg
    # are Anthropic-native so it stays None there. Read the literal from the
    # env_block instead — it's set whenever the provider has a base_url.
    base_url = spec.env_block.get("ANTHROPIC_BASE_URL")
    if not base_url:
        return None  # direct Anthropic — LiteLLM default routing works

    secret = os.environ.get(spec.auth_secret_env)
    if not secret:
        # Fall back to project .env so users don't need to export shell vars
        env_file = project_dir / ".env"
        if env_file.is_file():
            try:
                from dotenv import dotenv_values

                secret = dotenv_values(env_file).get(spec.auth_secret_env)
            except ImportError:
                pass

    if not secret:
        logger.warning(
            "No %s found in env or project .env; LiteLLM auth will likely fail",
            spec.auth_secret_env,
        )
        return None

    return {"api_base": base_url, "api_key": secret}


class ReactBackend(Backend):
    """Run queries via a manual ReAct loop on top of ``litellm.acompletion()``."""

    name = "react"

    def __init__(self, project_dir: Path, model: str, max_turns: int) -> None:
        self.project_dir = project_dir
        self.model = model
        self.max_turns = max_turns
        self.system_prompt = _read_agent_prompt(project_dir)
        self._call_kwargs_override = _resolve_litellm_endpoint(project_dir, model)

        # Arm the global rate limiter based on which provider the project
        # is configured to hit. Ollama models bypass this (the override
        # resolver returned None earlier and the provider is local).
        if not model.startswith(("ollama/", "ollama_chat/")):
            provider = _resolve_provider_name(project_dir)
            rpm = _PROVIDER_RATE_LIMIT_RPM.get(provider, None)
            configure_rate_limiter(rpm)

    async def run_query(self, prompt: str, pipeline_mode: str) -> WorkflowOutput:
        async with mcp_client_session(self.project_dir, pipeline_mode) as client:
            result = await run_react_query(
                client=client,
                prompt=prompt,
                model=self.model,
                system_prompt=self.system_prompt,
                max_turns=self.max_turns,
                call_kwargs_override=self._call_kwargs_override,
            )
        return WorkflowOutput(
            response_text=combined_text_from_react(result),
            tool_traces=result.tool_traces,
            cost_usd=result.cost_usd or 0.0,
            num_turns=result.num_turns or 1,
            input_tokens=result.input_tokens or 0,
            output_tokens=result.output_tokens or 0,
        )
