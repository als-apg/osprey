"""Core ``providers`` health category.

Runs a lightweight connectivity canary against every provider under
``api.providers``, composing :func:`osprey.health.probes.provider_canary.run`
once per provider. All items run **concurrently** inside the single category
callable (``asyncio.gather``); each canary bridges its synchronous
``check_health`` onto a daemon thread with a per-item bound of
:data:`_PER_ITEM_TIMEOUT_S` seconds, so the category's wall-clock stays ≈ 5s
regardless of how many providers are configured — within the epic-locked poll
bound.

Results are advisory only: a reachable provider is ``ok``, and every failure
mode (bad key, unknown provider, unreachable endpoint, timeout) is ``warning``.
The category never emits ``error``. Zero configured providers yields no rows,
matching the pre-framework behavior.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

from osprey.health.models import CheckResult, Status
from osprey.health.probes import ProbeContext, provider_canary
from osprey.health.runtime import HealthRuntime

if TYPE_CHECKING:
    from collections.abc import Mapping

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
        if not api_providers:
            return []

        names = list(api_providers)
        specs = [_spec(name, api_providers.get(name) or {}) for name in names]
        outcomes = await asyncio.gather(
            *(provider_canary.run(spec, ctx, registry=registry) for spec in specs),
            return_exceptions=True,
        )

        rows: list[CheckResult] = []
        for name, outcome in zip(names, outcomes, strict=True):
            if isinstance(outcome, CheckResult):
                rows.append(outcome)
            else:
                # The canary is designed never to raise; convert any surprise
                # into a warning row so one provider can never sink the batch.
                rows.append(
                    CheckResult(
                        name,
                        CATEGORY,
                        Status.WARNING,
                        "health check failed",
                        details=str(outcome),
                    )
                )
        return rows

    return _run


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
