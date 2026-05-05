"""Channel Finder In-Context MCP Server Context -- singleton config and database management.

Provides centralized configuration access and channel database lifecycle
management for all in-context channel finder MCP tools.

The server context conditionally imports the appropriate database class based
on the configured database type (``template`` or ``flat``).

Usage in tools:
    from osprey.mcp_server.channel_finder_in_context.server_context import (
        get_cf_ic_context,
    )

    ctx = get_cf_ic_context()
    db = ctx.database
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import yaml

from osprey.services.channel_finder.core.base_database import BaseDatabase
from osprey.services.channel_finder.rate_limiter import configure_rate_limiter

logger = logging.getLogger("osprey.mcp_server.channel_finder_in_context.server_context")

PROVIDER_RPM: dict[str, int | None] = {
    "cborg": 18,
    "anthropic": None,
    "als-apg": None,
}


class ChannelFinderICContext:
    """Singleton registry for in-context channel finder MCP server state.

    Responsibilities:
      1. Load and cache config.yml once at startup
      2. Parse channel_finder.pipelines.in_context config section
      3. Conditionally load the correct database class (flat or template)
      4. Provide a cached database instance for all tools
    """

    def __init__(self) -> None:
        self._raw_config: dict[str, Any] = {}
        self._database: BaseDatabase | None = None
        self._facility_name: str = "control system"
        self._subagent_model_id: str = ""
        self._subagent_provider: str = ""
        self._system_prompt_with_db: str = ""
        self._system_prompt_input_tokens: int = 0
        self._initialized = False

    def initialize(self) -> None:
        """Load config, select database type, and instantiate the database.

        Called once during create_server(). Subsequent calls are no-ops.
        """
        if self._initialized:
            return

        self._raw_config = self._load_config()

        cf_config = self._raw_config.get("channel_finder", {})
        ic_config = cf_config.get("pipelines", {}).get("in_context", {})
        db_config = ic_config.get("database", {})
        db_path = db_config.get("path")
        db_type = db_config.get("type", "template")

        if db_path:
            db_path = self._resolve_path(db_path)

            if db_type == "template":
                from osprey.services.channel_finder.databases.template import (
                    ChannelDatabase,
                )
            else:
                from osprey.services.channel_finder.databases.flat import (
                    ChannelDatabase,
                )

            self._database = ChannelDatabase(db_path)
            logger.info(
                "ChannelFinderICContext: loaded %s database from %s (%d channels)",
                db_type,
                db_path,
                len(self._database.get_all_channels()),
            )
        else:
            logger.warning(
                "No database path configured at "
                "channel_finder.pipelines.in_context.database.path -- "
                "channel finder tools will fail until config is provided"
            )

        facility = self._raw_config.get("facility", {})
        self._facility_name = facility.get("name", "control system")

        # Resolve subagent model and provider.
        #
        # The pipeline-local override (subagent_model / subagent_provider) wins
        # because the outer agent's provider — often an OpenAI-compatible
        # router like cborg — cannot dispatch every model the inner subagent
        # might run (e.g. an ollama/* model must reach a local Ollama daemon,
        # not cborg's /chat/completions).
        #
        # When no override is given, fall through to the same resolver that
        # `osprey build` uses for the outer Claude Code config so the in-context
        # server agrees with what the rest of the toolchain considers "the
        # configured model" — that's `claude_code.default_model` (a tier) +
        # `claude_code.models` / `api.providers[name].models` (tier→model map).
        cc_config = self._raw_config.get("claude_code", {})
        ic_model = ic_config.get("subagent_model")
        ic_provider = ic_config.get("subagent_provider")

        if ic_model:
            self._subagent_model_id = ic_model
            self._subagent_provider = ic_provider if ic_provider else cc_config.get("provider", "")
        else:
            from osprey.cli.claude_code_resolver import ClaudeCodeModelResolver

            api_providers = self._raw_config.get("api", {}).get("providers", {})
            spec = ClaudeCodeModelResolver.resolve(cc_config, api_providers)
            if spec is None:
                raise RuntimeError(
                    "No subagent model configured. Set either "
                    "'channel_finder.pipelines.in_context.subagent_model' or "
                    "'claude_code.provider' (with 'default_model' tier) in config.yml."
                )
            self._subagent_model_id = spec.tier_to_model[spec.default_model_tier]
            self._subagent_provider = ic_provider if ic_provider else spec.provider

        # Arm rate limiter for providers with a known RPM cap
        rpm_cap = PROVIDER_RPM.get(self._subagent_provider)
        if rpm_cap is not None:
            configure_rate_limiter(rpm_cap, window=60.0)
            logger.info(
                "ChannelFinderICContext: rate limiter armed for provider '%s' (%d rpm)",
                self._subagent_provider,
                rpm_cap,
            )

        # Build and cache system prompt with serialized channel database
        if self._database is not None:
            channels = self._database.get_all_channels()
            lines = []
            for ch in channels:
                name = ch.get("channel", ch.get("name", ""))
                addr = ch.get("address", "")
                desc = ch.get("description", "")
                lines.append(f"{name} | {addr} | {desc}")
            db_text = "\n".join(lines)
        else:
            db_text = "(no channel database loaded)"

        self._system_prompt_with_db = (
            "You are a channel-finder assistant. Below is the complete list of "
            "available control-system channels in the format:\n"
            "  CHANNEL_NAME | ADDRESS | DESCRIPTION\n\n"
            f"{db_text}\n\n"
            "Answer the user's query by identifying the most relevant channels. "
            "Wrap your final answer in <final> and </final> XML tags, like:\n"
            "<final>YOUR ANSWER HERE</final>"
        )

        # Pre-count system-prompt tokens once. The system prompt is identical
        # across every query in a cell — counting it per-query would waste
        # work on a 24k-token tier-3 prompt. Per-query input_tokens is then
        # this base + a small count for the user query.
        try:
            import litellm

            self._system_prompt_input_tokens = litellm.token_counter(
                model=self._subagent_model_id,
                text=self._system_prompt_with_db,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "ChannelFinderICContext: token_counter failed for model %s "
                "(%s) — input_tokens will be reported as user-query tokens only",
                self._subagent_model_id,
                exc,
            )
            self._system_prompt_input_tokens = 0

        self._initialized = True
        logger.info(
            "ChannelFinderICContext: initialized (system_prompt_tokens=%d)",
            self._system_prompt_input_tokens,
        )

    @property
    def database(self) -> BaseDatabase:
        """The loaded channel database instance.

        Raises:
            RuntimeError: If no database has been configured/loaded.
        """
        if self._database is None:
            raise RuntimeError(
                "Channel database not available. Check that config.yml has "
                "channel_finder.pipelines.in_context.database.path configured."
            )
        return self._database

    @property
    def facility_name(self) -> str:
        """Name of the facility from config."""
        return self._facility_name

    @property
    def subagent_model_id(self) -> str:
        """Resolved model ID for the inner LLM call."""
        return self._subagent_model_id

    @property
    def subagent_provider(self) -> str:
        """Provider name for the inner LLM call."""
        return self._subagent_provider

    @property
    def system_prompt_with_db(self) -> str:
        """Cached system prompt with the serialized channel database."""
        return self._system_prompt_with_db

    @property
    def system_prompt_input_tokens(self) -> int:
        """Tokens in the system prompt, counted once at init for the configured model.

        Used by query_channels to compute per-query input_tokens without
        re-tokenizing the (large) channel-database prompt on every call.
        """
        return self._system_prompt_input_tokens

    @property
    def raw_config(self) -> dict[str, Any]:
        """Full raw config dict."""
        return self._raw_config

    @staticmethod
    def _load_config() -> dict[str, Any]:
        """Load config.yml from OSPREY_CONFIG env var or cwd."""
        config_path = Path(
            os.path.expandvars(os.environ.get("OSPREY_CONFIG", str(Path.cwd() / "config.yml")))
        )
        raw: dict[str, Any] = {}
        if config_path.exists():
            with open(config_path) as f:
                raw = yaml.safe_load(f) or {}
            logger.info("ChannelFinderICContext: config loaded from %s", config_path)
        else:
            logger.warning("Config file not found: %s", config_path)

        return raw

    @staticmethod
    def _resolve_path(path_str: str) -> str:
        """Resolve path relative to config file directory."""
        config_path = Path(
            os.path.expandvars(os.environ.get("OSPREY_CONFIG", str(Path.cwd() / "config.yml")))
        )
        p = Path(path_str)
        if not p.is_absolute():
            p = config_path.parent / p
        return str(p.resolve())


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_registry: ChannelFinderICContext | None = None


def get_cf_ic_context() -> ChannelFinderICContext:
    """Get the channel finder IC MCP registry singleton.

    Raises RuntimeError if initialize_cf_ic_context() hasn't been called.
    """
    if _registry is None:
        raise RuntimeError(
            "Channel Finder IC MCP registry not initialized. Call initialize_cf_ic_context() first."
        )
    return _registry


def initialize_cf_ic_context() -> ChannelFinderICContext:
    """Create and initialize the channel finder IC MCP registry singleton."""
    global _registry
    _registry = ChannelFinderICContext()
    _registry.initialize()
    return _registry


def reset_cf_ic_context() -> None:
    """Reset the registry (for testing)."""
    global _registry
    _registry = None
