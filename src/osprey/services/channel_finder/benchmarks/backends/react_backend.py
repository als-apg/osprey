"""ReAct backend — wraps the manual ``litellm.acompletion()`` ReAct loop."""

from __future__ import annotations

import logging
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from osprey.models.config import provider_requests_per_minute
from osprey.models.providers.litellm_adapter import get_litellm_model_name
from osprey.services.channel_finder.benchmarks.harness import (
    combined_text_from_react,
    mcp_client_session,
    run_react_query,
)
from osprey.services.channel_finder.benchmarks.sdk import _read_agent_prompt
from osprey.services.channel_finder.rate_limiter import configure_rate_limiter

from ..project_env import expand_api_providers, project_config, project_dotenv
from .base import Backend, WorkflowOutput

logger = logging.getLogger(__name__)


def _resolve_litellm_endpoint(
    project_dir: Path, config: Mapping[str, Any] | None, provider: str
) -> dict | None:
    """Resolve provider routing kwargs for a non-ollama provider.

    The SDK path injects ``ANTHROPIC_BASE_URL`` + ``ANTHROPIC_AUTH_TOKEN``
    into the subprocess environment via ``inject_provider_env``. LiteLLM
    does NOT read ``ANTHROPIC_BASE_URL`` (it reads ``ANTHROPIC_API_BASE``),
    so env inheritance can't carry the override — we have to pass
    ``api_base`` / ``api_key`` explicitly to ``litellm.acompletion()``.

    Returns ``None`` for ollama (already handled by ``_litellm_call_kwargs``),
    for a project with no ``config.yml`` (``config`` is ``None``), and for
    direct Anthropic (LiteLLM's default routing is correct).

    This benchmark-only path takes the project's config as
    :func:`~osprey.services.channel_finder.benchmarks.project_env.project_config`
    read it and calls ``ClaudeCodeModelResolver.resolve`` rather than going through
    ``load_provider_spec``, because the contract differs (synthetic
    ``{"provider": provider}`` config + litellm ``api_base``). It does expand
    ``${VAR}`` in a provider's ``base_url``, against the overlay
    ``project_env`` defines (``os.environ`` over the project ``.env``) that the
    auth secret is read from, and refuses a reference that resolves to nothing
    rather than handing litellm a placeholder as a hostname — the shipped
    catalog spells gateway endpoints that way.
    """
    if provider == "ollama":
        return None

    from osprey.agent_runner.provider_env import ClaudeCodeModelResolver
    from osprey_connectors.config import is_unresolved_placeholder

    if config is None:
        return None
    # os.environ wins over the project .env, so a sweep can redirect a provider
    # for one run without editing the deployment's file.
    dotenv = project_dotenv(project_dir)
    api_providers = expand_api_providers(config, {**dotenv, **os.environ})
    spec = ClaudeCodeModelResolver.resolve({"provider": provider}, api_providers)
    if spec is None:
        return None

    base_url = spec.env_block.get("ANTHROPIC_BASE_URL")
    if is_unresolved_placeholder(base_url):
        raise ValueError(
            f"Provider '{provider}' names its endpoint as {base_url}, and that variable "
            f"is not set. Export it, or put it in {project_dir / '.env'}, before "
            "benchmarking this provider."
        )
    if not base_url:
        return None  # direct Anthropic — LiteLLM default routing works

    secret = os.environ.get(spec.auth_secret_env) or dotenv.get(spec.auth_secret_env)
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

    def __init__(
        self,
        project_dir: Path,
        model: str,
        max_turns: int,
    ) -> None:
        self.project_dir = project_dir
        self.model = model
        self.provider, self.wire_id = model.split("/", 1)
        # Format the slug for LiteLLM's grammar. Critically, OpenAI-compat
        # proxies (als-apg, cborg) need ``openai/<wire>`` even though the
        # endpoint is reached via ``ANTHROPIC_BASE_URL`` — the prefix tells
        # LiteLLM which wire protocol to speak; the proxy is selected via
        # ``api_base`` resolved below.
        self.litellm_model = get_litellm_model_name(self.provider, self.wire_id)
        self.max_turns = max_turns
        self.system_prompt = _read_agent_prompt(project_dir)
        config = project_config(project_dir)
        self._call_kwargs_override = _resolve_litellm_endpoint(project_dir, config, self.provider)

        # Pace calls to the provider's catalog cap; a provider without one is not paced.
        configure_rate_limiter(provider_requests_per_minute(config or {}, self.provider))

    async def run_query(self, prompt: str, pipeline_mode: str) -> WorkflowOutput:
        async with mcp_client_session(self.project_dir, pipeline_mode) as client:
            result = await run_react_query(
                client=client,
                prompt=prompt,
                model=self.litellm_model,
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
