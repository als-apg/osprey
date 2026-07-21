"""Provider-canary health probe — a minimal LLM connectivity check.

Resolves a provider by name through the lightweight
:class:`~osprey.models.provider_registry.ProviderRegistry` (which loads provider
classes lazily and works *without* a full ``RegistryManager`` init), instantiates
it, and calls its synchronous
:meth:`~osprey.models.providers.base.BaseProvider.check_health`
``(api_key, base_url, timeout=5.0, model_id=None) -> tuple[bool, str]`` off the
event loop via the daemon-thread bridge (:func:`osprey.health.offload.run_sync`).

Grading is deliberately narrow — the canary is a *poll*-class check and never
produces an ``error`` row:

* ``check_health`` returns ``(True, msg)`` → ``ok``;
* ``check_health`` returns ``(False, msg)`` → ``warning``;
* an unknown provider name → ``warning`` ``"Unknown provider"`` (never a crash);
* a raised exception, or a timeout while the daemon thread is abandoned →
  ``warning`` (an unreachable provider must not fail the suite).

The API key and base URL come from the ``api.providers.<name>`` config block:
``api_key`` and ``base_url`` may be supplied directly in ``spec`` (as the
core ``providers`` category does when it composes this probe per provider), and
fall back to the global config singleton otherwise. Both are passed through
``${VAR}`` resolution against :data:`os.environ` so a placeholder such as
``${CBORG_API_KEY}`` is expanded at check time (after the project ``.env`` load);
an unresolved lone placeholder collapses to an empty key, matching the connector
contract that authentication is checked by ``check_health`` itself.
"""

from __future__ import annotations

import os
import re
from collections.abc import Mapping
from time import perf_counter
from typing import TYPE_CHECKING, Any

from osprey.health.models import CheckResult, Status
from osprey.health.offload import run_sync
from osprey.models.provider_registry import get_provider_registry
from osprey.utils.config import get_config_value, resolve_env_vars

if TYPE_CHECKING:
    from osprey.health.probes import ProbeContext
    from osprey.models.provider_registry import ProviderRegistry

#: Extra seconds granted to the daemon-thread bridge over the provider's own
#: ``timeout`` so ``check_health`` returns its own ``(False, "…timed out")`` row
#: before the bridge abandons the thread on a hard hang.
_OFFLOAD_MARGIN_S = 2.0

#: A string that is nothing but a single unresolved env-var placeholder, e.g.
#: ``"${MISSING}"`` or ``"$MISSING"``. Such a value collapses to ``""``.
_LONE_PLACEHOLDER = re.compile(r"^\$\{[A-Za-z_][A-Za-z0-9_]*\}$|^\$[A-Za-z_][A-Za-z0-9_]*$")


def _resolve_secret(value: str | None) -> str | None:
    """Expand ``${VAR}`` in *value* against ``os.environ``.

    A non-string (typically ``None`` for an absent key/URL) is returned
    unchanged. A string is passed through :func:`resolve_env_vars`; if it is
    left as a single unresolved placeholder (the env var is unset) it collapses
    to ``""`` so the provider sees an empty key rather than a literal
    ``"${VAR}"``.
    """
    if not isinstance(value, str):
        return value
    resolved = resolve_env_vars(value, environ=os.environ)
    if not isinstance(resolved, str):
        return value
    if _LONE_PLACEHOLDER.match(resolved):
        return ""
    return resolved


def _provider_block(provider_name: str) -> Mapping[str, Any]:
    """Return the ``api.providers.<name>`` config block, or ``{}`` if unavailable.

    Any failure to load config (no ``config.yml``, unparseable YAML) is
    swallowed to an empty block: a canary must never crash on a missing config,
    and a spec that already carries ``api_key``/``base_url`` never needs it.
    """
    try:
        block = get_config_value(f"api.providers.{provider_name}", {})
    except Exception:  # noqa: BLE001 - config unavailability degrades to no block
        return {}
    return block if isinstance(block, Mapping) else {}


async def run(
    spec: Mapping[str, Any],
    ctx: ProbeContext,
    *,
    registry: ProviderRegistry | None = None,
) -> CheckResult:
    """Probe a model provider with a minimal ``check_health`` API call.

    Args:
        spec: Parsed check parameters. Recognized keys:

            * ``provider`` (str): provider name to resolve, e.g. ``"cborg"``.
              Falls back to ``name`` when omitted (the core ``providers``
              category names each check after its provider).
            * ``name`` (str): the result name (the runner injects it as the
              check's identity); defaults to ``provider``.
            * ``category`` (str): result category (default ``"providers"``).
            * ``api_key`` (str | None): API key; may contain ``${VAR}``. Falls
              back to ``api.providers.<provider>.api_key`` when absent from
              ``spec``.
            * ``base_url`` (str | None): custom endpoint; may contain ``${VAR}``.
              Falls back to ``api.providers.<provider>.base_url`` when absent.
            * ``model_id`` (str | None): optional model to probe (provider picks
              its cheapest when ``None``).
            * ``timeout_s`` (float): provider request timeout in seconds
              (default ``5.0``); also bounds the off-loaded call.
        ctx: Shared per-run context. Unused by this probe (a canary needs no
            control-system connector) but part of the uniform probe interface.
        registry: Optional provider registry for dependency injection in tests;
            ``None`` uses the global :func:`get_provider_registry` singleton.

    Returns:
        A :class:`~osprey.health.models.CheckResult`. Never ``error``: success
        is ``ok``, and every failure mode (bad key, unknown provider, timeout,
        raised exception) is a ``warning`` with ``latency_ms`` set when an API
        call was attempted.
    """
    # The provider to resolve is a probe param (``provider``). The runner injects
    # ``name`` as the check's result identity, so fall back to it only when no
    # explicit ``provider`` is given (the core ``providers`` category names each
    # check after its provider).
    provider_name = str(spec.get("provider") or spec.get("name") or "")
    result_name = str(spec.get("name") or provider_name or "provider")
    category = str(spec.get("category", "providers"))
    timeout_s = float(spec.get("timeout_s", 5.0))
    model_id = spec.get("model_id")

    reg = registry if registry is not None else get_provider_registry()
    provider_class = reg.get_provider(provider_name)
    if provider_class is None:
        return CheckResult(result_name, category, Status.WARNING, "Unknown provider")

    # Consult the config block only for fields the spec does not already supply,
    # so a fully spec-driven call never touches the global config singleton.
    block: Mapping[str, Any] = {}
    if "api_key" not in spec or "base_url" not in spec:
        block = _provider_block(provider_name)
    api_key = _resolve_secret(spec["api_key"] if "api_key" in spec else block.get("api_key"))
    base_url = _resolve_secret(spec["base_url"] if "base_url" in spec else block.get("base_url"))

    t0 = perf_counter()
    try:
        provider = provider_class()
        success, message = await run_sync(
            provider.check_health,
            api_key,
            base_url,
            timeout_s,
            model_id,
            timeout_s=timeout_s + _OFFLOAD_MARGIN_S,
        )
    except TimeoutError:
        latency_ms = (perf_counter() - t0) * 1000.0
        return CheckResult(
            result_name,
            category,
            Status.WARNING,
            f"health check timed out after {timeout_s:g}s",
            latency_ms=latency_ms,
        )
    except Exception as exc:  # noqa: BLE001 - an unreachable provider is a warning, never error
        latency_ms = (perf_counter() - t0) * 1000.0
        return CheckResult(
            result_name,
            category,
            Status.WARNING,
            "health check failed",
            latency_ms=latency_ms,
            details=str(exc),
        )

    latency_ms = (perf_counter() - t0) * 1000.0
    return CheckResult(
        result_name,
        category,
        Status.OK if success else Status.WARNING,
        message,
        latency_ms=latency_ms,
    )
