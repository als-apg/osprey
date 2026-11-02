"""Settings for the Jev decision client, read from the ``api.jev`` config block.

``JevSettings`` holds every ``api.jev`` default and validates the block. The
shape of the block::

    api:
      jev:
        route: typesafe                 # typesafe | openrouter | openai
        max_choice_options: 255         # 1..255
        max_concurrency: 4              # requests in flight, client-wide
        input_cost_per_million_tokens: 0.042
        budgets:
          interactive: {timeout_s: 2, max_retries: 1, total_s: 3}
          batch:       {timeout_s: 10, max_retries: 2, total_s: 30}
        routes:
          typesafe:
            model: jev-1.13.0
            base_url: https://api.typesafe.ai
            api_key: null               # null resolves TYPESAFE_API_KEY
            max_request_tokens: 60000
            max_state_plus_question_tokens: 30000
          openrouter:
            model: typesafe/jev-1.13
            max_request_tokens: 30000
            max_state_plus_question_tokens: 30000
          openai:
            model: null                 # the served model id; required for this route
            base_url: null              # e.g. http://127.0.0.1:8000; required
            api_key: null               # optional on a loopback or private address
            max_request_tokens: 30000
            max_state_plus_question_tokens: 30000

The openrouter route carries no base URL or key of its own: both come from the
``api.providers.openrouter`` block. The typesafe route is configured here only
and is never an ``api.providers`` entry. The openai route points at any
OpenAI-compatible chat server (a stock vLLM, for a self-hosted model) and asks
each question as one chat completion (see ``openai_route``); its key is
optional when ``base_url`` names a loopback or private address, and has no
environment fallback, so an ambient ``OPENAI_API_KEY`` is never sent to it.

``resolve_route`` turns the settings plus the full config into the one base
URL, key and model a request uses. The openrouter base follows the OpenRouter
provider's own rule (``OPENROUTER_BASE_URL`` > ``api.providers.openrouter.base_url``
> the public default), so the chat route and the decision route always move
together. A key that is missing, empty or an unresolved ``${VAR}`` reference
raises ``DecisionAuthError(error_type="no_key")`` before any request is made.

Model ids are versioned request ids. Floating aliases are refused, because a
calibrated threshold file is only valid for the model it was fitted on.

``max_state_plus_question_tokens`` is an estimate that already includes the
fixed request overhead, so keeping it at or below ``max_request_tokens`` means
a lone question that passes the local size refusal always fits one request.
"""

from __future__ import annotations

import ipaddress
import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Literal
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from osprey.models.jev.errors import DecisionAuthError, DecisionRequestError
from osprey_connectors.config import is_unresolved_placeholder, resolve_env_vars

__all__ = [
    "REFUSED_MODEL_ALIASES",
    "Budget",
    "Budgets",
    "JevSettings",
    "OPENROUTER_KEY_ENV",
    "OpenAIRoute",
    "OpenRouterRoute",
    "ResolvedRoute",
    "RouteSettings",
    "Routes",
    "TYPESAFE_KEY_ENV",
    "TypesafeRoute",
    "chat_endpoint",
    "endpoint",
    "is_local_address",
    "openrouter_provider_block",
    "resolve_base_url",
    "resolve_key",
    "resolve_route",
]

Route = Literal["typesafe", "openrouter", "openai"]

REFUSED_MODEL_ALIASES: frozenset[str] = frozenset(
    {"jev-latest", "jev-preview", "~typesafe/jev-latest"}
)

_MAX_CHOICE_OPTIONS_CEILING = 255

# Environment variables that supply a key when config names none.
TYPESAFE_KEY_ENV = "TYPESAFE_API_KEY"
OPENROUTER_KEY_ENV = "OPENROUTER_API_KEY"


def endpoint(base: str) -> str:
    """The decision endpoint for a base URL given with or without ``/v1``."""
    return base.rstrip("/").removesuffix("/v1") + "/v1/systemone"


def chat_endpoint(base: str) -> str:
    """The chat completions endpoint for a base URL given with or without ``/v1``."""
    return base.rstrip("/").removesuffix("/v1") + "/v1/chat/completions"


class _Settings(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Budget(_Settings):
    """Time and retry budget for one kind of caller."""

    timeout_s: float = Field(gt=0)
    max_retries: int = Field(ge=0)
    total_s: float = Field(gt=0)

    @model_validator(mode="after")
    def _total_covers_one_call(self) -> Budget:
        if self.total_s < self.timeout_s:
            raise ValueError(
                f"total_s ({self.total_s}) must be at least timeout_s ({self.timeout_s})"
            )
        return self


class Budgets(_Settings):
    """Time and retry budgets for interactive and batch requests."""

    interactive: Budget = Budget(timeout_s=2.0, max_retries=1, total_s=3.0)
    batch: Budget = Budget(timeout_s=10.0, max_retries=2, total_s=30.0)

    @field_validator("interactive", "batch", mode="before")
    @classmethod
    def _merge_partial(cls, value: Any, info: Any) -> Any:
        # A partial budget block overrides only the keys it names.
        if isinstance(value, Mapping):
            default = cls.model_fields[info.field_name].default
            return {**default.model_dump(), **value}
        return value


class RouteSettings(_Settings):
    """Model pin and size limits shared by every route."""

    model: str
    max_request_tokens: int = Field(gt=0)
    max_state_plus_question_tokens: int = Field(default=30_000, gt=0)

    @field_validator("model")
    @classmethod
    def _versioned_model_only(cls, value: str | None) -> str | None:
        if value is None:
            return None
        value = value.strip()
        if not value:
            raise ValueError("model must be a non-empty model id")
        if value in REFUSED_MODEL_ALIASES:
            raise ValueError(
                f"model {value!r} is a floating alias; pin a versioned model id instead"
            )
        return value

    @model_validator(mode="after")
    def _question_fits_one_request(self) -> RouteSettings:
        if self.max_state_plus_question_tokens > self.max_request_tokens:
            raise ValueError(
                f"max_state_plus_question_tokens ({self.max_state_plus_question_tokens}) "
                f"must not exceed max_request_tokens ({self.max_request_tokens})"
            )
        return self


class TypesafeRoute(RouteSettings):
    """The direct Typesafe route: model, endpoint, key and request size cap."""

    model: str = "jev-1.13.0"
    base_url: str = "https://api.typesafe.ai"
    api_key: str | None = Field(default=None, repr=False)
    max_request_tokens: int = Field(default=60_000, gt=0)


class OpenRouterRoute(RouteSettings):
    """The OpenRouter route: model and request size cap."""

    model: str = "typesafe/jev-1.13"
    max_request_tokens: int = Field(default=30_000, gt=0)


class OpenAIRoute(RouteSettings):
    """An OpenAI-compatible chat server: served model, endpoint, optional key, size cap."""

    model: str | None = None  # type: ignore[assignment]
    base_url: str | None = None
    api_key: str | None = Field(default=None, repr=False)
    max_request_tokens: int = Field(default=30_000, gt=0)


class Routes(_Settings):
    """Settings for every supported route."""

    typesafe: TypesafeRoute = TypesafeRoute()
    openrouter: OpenRouterRoute = OpenRouterRoute()
    openai: OpenAIRoute = OpenAIRoute()


class JevSettings(_Settings):
    """Every ``api.jev`` default, validated."""

    route: Route = "typesafe"
    max_choice_options: int = Field(
        default=_MAX_CHOICE_OPTIONS_CEILING, ge=1, le=_MAX_CHOICE_OPTIONS_CEILING
    )
    max_concurrency: int = Field(default=4, ge=1)
    input_cost_per_million_tokens: float = Field(default=0.042, ge=0)
    budgets: Budgets = Budgets()
    routes: Routes = Routes()

    @model_validator(mode="after")
    def _openai_route_is_named(self) -> JevSettings:
        if self.route == "openai":
            block = self.routes.openai
            missing = [
                name for name in ("model", "base_url") if not (getattr(block, name) or "").strip()
            ]
            if missing:
                raise ValueError(
                    "route openai needs "
                    + " and ".join(f"api.jev.routes.openai.{name}" for name in missing)
                )
            base = block.base_url or ""
            if "${" not in base and urlsplit(base.strip()).scheme not in ("http", "https"):
                raise ValueError(
                    "api.jev.routes.openai.base_url must start with http:// or https://"
                )
        return self

    @property
    def active_route(self) -> TypesafeRoute | OpenRouterRoute | OpenAIRoute:
        """The settings block of the selected route."""
        if self.route == "openrouter":
            return self.routes.openrouter
        if self.route == "openai":
            return self.routes.openai
        return self.routes.typesafe

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> JevSettings:
        """Read ``api.jev`` from a full config mapping; a missing block gives the defaults."""
        api = config.get("api") or {}
        block = api.get("jev") if isinstance(api, Mapping) else None
        if block is None:
            return cls()
        return cls.model_validate(block)


# ---------------------------------------------------------------------------
# Route resolution: endpoint base and key
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ResolvedRoute:
    """The base URL, key and model one decision request is sent with."""

    base_url: str
    api_key: str | None = field(repr=False)
    model: str
    route: Route

    @property
    def endpoint(self) -> str:
        """The decision endpoint under :attr:`base_url`; chat completions on the openai route."""
        if self.route == "openai":
            return chat_endpoint(self.base_url)
        return endpoint(self.base_url)


def openrouter_provider_block(config: Mapping[str, Any]) -> Mapping[str, Any]:
    """The ``api.providers.openrouter`` block, or an empty mapping when absent."""
    api = config.get("api") or {}
    providers = api.get("providers") if isinstance(api, Mapping) else None
    block = providers.get("openrouter") if isinstance(providers, Mapping) else None
    return block if isinstance(block, Mapping) else {}


def _configured(value: Any) -> str | None:
    """A config string after ``${VAR}`` expansion; ``None`` when empty or unresolved."""
    if value is None:
        return None
    resolved = resolve_env_vars(value)
    if not isinstance(resolved, str) or is_unresolved_placeholder(resolved):
        return None
    resolved = resolved.strip()
    return resolved or None


def resolve_base_url(settings: JevSettings, config: Mapping[str, Any]) -> str:
    """The base URL of the active route.

    The openrouter route takes the OpenRouter provider's effective base, so
    ``OPENROUTER_BASE_URL`` moves chat and decision traffic together. The
    typesafe route uses its own ``base_url``, falling back to the default when
    that is empty or an unresolved reference. The openai route has no default:
    an empty or unresolved ``base_url`` raises ``DecisionRequestError``.
    """
    if settings.route == "openai":
        base = _configured(settings.routes.openai.base_url)
        if base is None:
            raise DecisionRequestError(
                "api.jev.routes.openai.base_url is empty or an unresolved reference",
                error_type="invalid_request",
            )
        return base
    if settings.route == "openrouter":
        from osprey.models.providers.openrouter import OpenRouterProviderAdapter

        block = openrouter_provider_block(config)
        base = OpenRouterProviderAdapter.effective_base_url(_configured(block.get("base_url")))
        # requires_base_url + default_base_url on the adapter make this non-None.
        assert base is not None
        return base
    return _configured(settings.routes.typesafe.base_url) or TypesafeRoute().base_url


def is_local_address(base_url: str) -> bool:
    """Whether ``base_url`` names ``localhost`` or a loopback or private IP address.

    A hostname other than ``localhost`` is never local here: it is not resolved.
    """
    host = urlsplit(base_url).hostname or ""
    if host == "localhost":
        return True
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        return False
    return address.is_loopback or address.is_private


def _openai_key(settings: JevSettings) -> str | None:
    key = _configured(settings.routes.openai.api_key)
    where = "api.jev.routes.openai.api_key"
    if key is None:
        base = _configured(settings.routes.openai.base_url) or ""
        if is_local_address(base):
            return None
        raise DecisionAuthError(
            f"no API key for the openai Jev route: set {where}; it may be omitted only"
            " when base_url is a loopback or private address",
            error_type="no_key",
        )
    if not key.isascii():
        raise DecisionAuthError(
            f"the API key for the openai Jev route is not ASCII: check {where}",
            error_type="no_key",
        )
    return key


def resolve_key(settings: JevSettings, config: Mapping[str, Any]) -> str | None:
    """The API key of the active route; ``None`` only on the openai route.

    Config first (``routes.typesafe.api_key`` or ``api.providers.openrouter.api_key``),
    then the route's environment variable. A key that is missing, empty or an
    unresolved ``${VAR}`` reference raises ``DecisionAuthError`` with
    ``error_type="no_key"``, and so does a key that is not ASCII, which no HTTP
    header can carry; the message names only where a key was looked for.

    The openai route reads ``routes.openai.api_key`` only, with no environment
    fallback; it returns ``None`` (no ``Authorization`` header) when that is
    unset and ``base_url`` is a loopback or private address.
    """
    if settings.route == "openai":
        return _openai_key(settings)
    if settings.route == "openrouter":
        configured = _configured(openrouter_provider_block(config).get("api_key"))
        env_var = OPENROUTER_KEY_ENV
        where = "api.providers.openrouter.api_key"
    else:
        configured = _configured(settings.routes.typesafe.api_key)
        env_var = TYPESAFE_KEY_ENV
        where = "api.jev.routes.typesafe.api_key"
    from_env = (os.environ.get(env_var) or "").strip()
    key = configured or (None if is_unresolved_placeholder(from_env) else from_env) or None
    if key is None:
        raise DecisionAuthError(
            f"no API key for the {settings.route} Jev route: set {where} or {env_var}",
            error_type="no_key",
        )
    if not key.isascii():
        raise DecisionAuthError(
            f"the API key for the {settings.route} Jev route is not ASCII: "
            f"check {where} or {env_var}",
            error_type="no_key",
        )
    return key


def resolve_route(settings: JevSettings, config: Mapping[str, Any]) -> ResolvedRoute:
    """Resolve the active route's base URL, key and model.

    The key is checked first, except on the openai route, whose key rule reads
    the base URL.
    """
    if settings.route == "openai":
        base_url = resolve_base_url(settings, config)
        key = resolve_key(settings, config)
    else:
        key = resolve_key(settings, config)
        base_url = resolve_base_url(settings, config)
    model = settings.active_route.model
    # JevSettings refuses route openai without a model; the others default one.
    assert model is not None
    return ResolvedRoute(base_url=base_url, api_key=key, model=model, route=settings.route)
