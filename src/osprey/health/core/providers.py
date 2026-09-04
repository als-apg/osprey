"""Core ``providers`` health category.

Runs a lightweight connectivity canary against every provider under
``api.providers``, composing :func:`osprey.health.probes.provider_canary.run`
once per provider. All items run **concurrently** inside the single category
callable (``asyncio.gather``); each canary bridges its synchronous
``check_health`` onto a daemon thread with a per-item bound of
:data:`_PER_ITEM_TIMEOUT_S` seconds, so the category's wall-clock stays ≈ 5s
regardless of how many providers are configured — within the health poll bound.

Results are advisory for every provider but one: a reachable provider is
``ok``, and every failure mode (bad key, unknown provider, unreachable
endpoint, timeout) is ``warning``. The exception is the deployment's own agent
provider — ``claude_code.provider``, the one every web terminal and the
dispatch worker authenticate with. When that one fails, the deployment is down,
so its row is an ``error`` (and ``osprey health`` exits 2); it is also an error
when it is named but has no ``api.providers`` block to probe. Zero configured
providers yields no rows.
"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from dataclasses import replace
from typing import TYPE_CHECKING, Any

from osprey.health.models import CheckResult, Status
from osprey.health.probes import ProbeContext, provider_canary
from osprey.health.runtime import HealthRuntime

if TYPE_CHECKING:
    from osprey.health.core import CategoryCallable
    from osprey.models.provider_registry import ProviderRegistry

CATEGORY = "providers"

#: Per-provider request bound in seconds, passed to each canary as ``timeout_s``.
#: The canary's daemon-thread bridge grants a small extra margin on top, so a
#: provider that honors its own timeout returns just under this bound.
_PER_ITEM_TIMEOUT_S = 5.0


def providers(
    config: Mapping[str, Any] | None = None,
    context: HealthRuntime | None = None,
    *,
    registry: ProviderRegistry | None = None,
) -> CategoryCallable:
    """Build the ``providers`` category callable.

    Args:
        config: Parsed config mapping (``None`` when config is unavailable). Read
            for the ``api.providers`` block; each provider's ``api_key`` and
            ``base_url`` are forwarded to its canary.
        context: Health runtime. Unused — a canary needs no control-system
            connector — but accepted for a uniform factory signature.
        registry: Optional provider registry for dependency injection in tests;
            ``None`` uses the global provider-registry singleton via the canary.

    Returns:
        A no-argument async callable returning one advisory result per
        configured provider (empty when none are configured).
    """
    cfg: Mapping[str, Any] = config or {}
    # The canary ignores the runtime; supply a never-constructed one when the
    # factory is called without a context so the ProbeContext stays type-correct.
    ctx = ProbeContext(runtime=context if context is not None else HealthRuntime({}))

    async def _run() -> list[CheckResult]:
        api = cfg.get("api", {}) or {}
        api_providers = api.get("providers", {}) or {}
        own = _own_provider(cfg)
        rows: list[CheckResult] = []
        if own and own not in api_providers:
            rows.append(
                CheckResult(
                    own,
                    CATEGORY,
                    Status.ERROR,
                    f"claude_code.provider names {own!r}, which has no api.providers.{own} "
                    "block, so there is no key to check",
                )
            )
        if not api_providers:
            return rows

        names = list(api_providers)
        specs = [_spec(name, api_providers.get(name) or {}) for name in names]
        outcomes = await asyncio.gather(
            *(provider_canary.run(spec, ctx, registry=registry) for spec in specs),
            return_exceptions=True,
        )

        for name, outcome in zip(names, outcomes, strict=True):
            if isinstance(outcome, CheckResult):
                row = outcome
            else:
                # The canary is designed never to raise; convert any surprise
                # into a warning row so one provider can never sink the batch.
                row = CheckResult(
                    name,
                    CATEGORY,
                    Status.WARNING,
                    "health check failed",
                    details=str(outcome),
                )
            if name == own and row.status is not Status.OK:
                # Not advisory: this is the provider the agent runs on, so a
                # failure here is every terminal and the dispatch worker down.
                row = replace(
                    row,
                    status=Status.ERROR,
                    message=(
                        f"{row.message} — {own} is this deployment's agent provider "
                        "(claude_code.provider); every web terminal and the dispatch "
                        "worker authenticate with it"
                    ),
                )
            rows.append(row)
        return rows

    return _run


def _own_provider(cfg: Mapping[str, Any]) -> str | None:
    """The provider the deployment's agent authenticates with, or ``None``."""
    claude_code = cfg.get("claude_code")
    if not isinstance(claude_code, Mapping):
        return None
    provider = claude_code.get("provider")
    return provider if isinstance(provider, str) and provider else None


def _spec(name: str, block: Mapping[str, Any]) -> dict[str, Any]:
    """Build a canary ``spec`` from a provider's ``api.providers`` config block.

    ``provider`` names the provider to resolve; ``name`` is the row identity
    (the same provider name). ``api_key`` and ``base_url`` are passed explicitly
    (as ``None`` when absent) so the canary resolves them from this config rather
    than re-reading the global config singleton.
    """
    return {
        "provider": name,
        "name": name,
        "category": CATEGORY,
        "api_key": block.get("api_key"),
        "base_url": block.get("base_url"),
        "timeout_s": _PER_ITEM_TIMEOUT_S,
    }
