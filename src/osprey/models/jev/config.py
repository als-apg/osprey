"""Settings for the Jev decision client, read from the ``api.jev`` config block.

``JevSettings`` holds every ``api.jev`` default and validates the block. The
shape of the block::

    api:
      jev:
        route: typesafe                 # typesafe | openrouter
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

The openrouter route carries no base URL or key of its own: both come from the
``api.providers.openrouter`` block. The typesafe route is configured here only
and is never an ``api.providers`` entry.

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

import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from osprey.models.jev.errors import DecisionAuthError
from osprey_connectors.config import is_unresolved_placeholder, resolve_env_vars

__all__ = [
    "REFUSED_MODEL_ALIASES",
    "Budget",
    "Budgets",
    "JevSettings",
    "OPENROUTER_KEY_ENV",
    "OpenRouterRoute",
    "ResolvedRoute",
    "RouteSettings",
    "Routes",
    "TYPESAFE_KEY_ENV",
    "TypesafeRoute",
    "endpoint",
    "openrouter_provider_block",
    "resolve_base_url",
    "resolve_key",
    "resolve_route",
]

Route = Literal["typesafe", "openrouter"]

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
    def _versioned_model_only(cls, value: str) -> str:
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


class Routes(_Settings):
    """Settings for every supported route."""

    typesafe: TypesafeRoute = TypesafeRoute()
    openrouter: OpenRouterRoute = OpenRouterRoute()


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

    @property
    def active_route(self) -> TypesafeRoute | OpenRouterRoute:
        """The settings block of the selected route."""
        if self.route == "openrouter":
            return self.routes.openrouter
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
    api_key: str = field(repr=False)
    model: str
    route: Route

    @property
    def endpoint(self) -> str:
        """The decision endpoint under :attr:`base_url`."""
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
    that is empty or an unresolved reference.
    """
    if settings.route == "openrouter":
        from osprey.models.providers.openrouter import OpenRouterProviderAdapter

        block = openrouter_provider_block(config)
        base = OpenRouterProviderAdapter.effective_base_url(_configured(block.get("base_url")))
        # requires_base_url + default_base_url on the adapter make this non-None.
        assert base is not None
        return base
    return _configured(settings.routes.typesafe.base_url) or TypesafeRoute().base_url


def resolve_key(settings: JevSettings, config: Mapping[str, Any]) -> str:
    """The API key of the active route.

    Config first (``routes.typesafe.api_key`` or ``api.providers.openrouter.api_key``),
    then the route's environment variable. A key that is missing, empty or an
    unresolved ``${VAR}`` reference raises ``DecisionAuthError`` with
    ``error_type="no_key"``, and so does a key that is not ASCII, which no HTTP
    header can carry; the message names only where a key was looked for.
    """
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
    """Resolve the active route's base URL, key and model; the key is checked first."""
    key = resolve_key(settings, config)
    return ResolvedRoute(
        base_url=resolve_base_url(settings, config),
        api_key=key,
        model=settings.active_route.model,
        route=settings.route,
    )
