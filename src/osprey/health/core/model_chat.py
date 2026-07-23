"""Core ``model_chat`` health category (on_demand).

For every unique ``(provider, model_id)`` pair in ``config.models`` this
category issues one minimal chat completion (``"Reply with exactly: OK"``,
``max_tokens=50``) and grades it:

* a non-empty string response → ``ok``;
* an empty (or non-string) response → ``warning`` "Empty response";
* a timeout — whether raised by the provider (its message contains
  ``"timeout"``/``"timed out"``) or synthesized by the per-item offload budget
  expiring (:func:`~osprey.health.offload.run_sync`) — → ``warning`` "Timeout";
* any other failure → ``error`` (the ``model_chat_*`` rows are error-allowlisted).

Each completion is synchronous provider I/O, so it runs through
:func:`osprey.health.offload.run_sync` on a daemon thread — a hung model can
never wedge process exit. Pairs are graded **sequentially**: the on_demand
category budget is ``60 × max(N, 1)`` seconds (``config.py``'s item-looping
resolution), so the per-item budget is that divided by ``max(N, 1)`` — 60s by
default, or ``override / N`` when the ``model_chat`` category carries an explicit
``timeout_s``. Zero configured models short-circuits to a single ``skip`` row.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from osprey.health.config import resolve_item_looping_on_demand_timeout
from osprey.health.models import CheckResult, Status
from osprey.health.offload import run_sync

if TYPE_CHECKING:
    from collections.abc import Mapping

    from osprey.health.core import CategoryCallable
    from osprey.health.runtime import HealthRuntime

CATEGORY = "model_chat"

_TEST_PROMPT = "Reply with exactly: OK"
_MAX_TOKENS = 50


def model_chat(
    config: Mapping[str, Any] | None = None,
    context: HealthRuntime | None = None,
) -> CategoryCallable:
    """Build the ``model_chat`` category callable.

    Args:
        config: Parsed config mapping (``None`` when config is unavailable). Read
            for ``models`` (the ``(provider, model_id)`` pairs to exercise) and
            an optional ``health.categories.model_chat.timeout_s`` override.
        context: Health runtime. Unused — chat completions need no
            control-system connector.

    Returns:
        A no-argument async callable returning the category's check results.
    """
    cfg: Mapping[str, Any] = config or {}

    async def _run() -> list[CheckResult]:
        pairs = unique_model_pairs(cfg)
        if not pairs:
            return [
                CheckResult(CATEGORY, CATEGORY, Status.SKIP, "no models configured"),
            ]

        _, per_item_timeout_s = resolve_item_looping_on_demand_timeout(
            len(pairs), _override_timeout_s(cfg)
        )

        results: list[CheckResult] = []
        for provider, model_id in pairs:
            results.append(await _check_pair(provider, model_id, per_item_timeout_s))
        return results

    return _run


def unique_model_pairs(cfg: Mapping[str, Any]) -> list[tuple[str, str]]:
    """Return sorted unique ``(provider, model_id)`` pairs from ``config.models``.

    This is the category's single pairing rule; the CLI uses it too when sizing
    the category's on_demand budget, so the two always agree.

    Args:
        cfg: Parsed config mapping; entries without both ``provider`` and
            ``model_id`` (or that are not mappings) are ignored.

    Returns:
        The deduplicated pairs in sorted order.
    """
    models = cfg.get("models")
    if not isinstance(models, dict):
        return []
    pairs: set[tuple[str, str]] = set()
    for model_config in models.values():
        if not isinstance(model_config, dict):
            continue
        provider = model_config.get("provider")
        model_id = model_config.get("model_id")
        if provider and model_id:
            pairs.add((str(provider), str(model_id)))
    return sorted(pairs)


def _override_timeout_s(cfg: Mapping[str, Any]) -> float | None:
    """Read an explicit ``health.categories.model_chat.timeout_s`` override, if any."""
    categories = (cfg.get("health") or {}).get("categories") or {}
    entry = categories.get(CATEGORY) or {}
    value = entry.get("timeout_s")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _complete(provider: str, model_id: str) -> Any:
    """Issue the minimal test completion (synchronous provider I/O)."""
    from osprey.models.completion import get_chat_completion

    return get_chat_completion(
        message=_TEST_PROMPT,
        provider=provider,
        model_id=model_id,
        max_tokens=_MAX_TOKENS,
    )


async def _check_pair(provider: str, model_id: str, timeout_s: float) -> CheckResult:
    """Grade a single ``(provider, model_id)`` chat completion into a result row."""
    name = f"model_chat_{provider}_{model_id}"
    label = f"{provider}/{model_id}"
    try:
        response = await run_sync(_complete, provider, model_id, timeout_s=timeout_s)
    except TimeoutError:
        # Per-item offload budget expired (run_sync's wait_for) — pinned to
        # warning, matching the provider-raised timeout mapping below.
        return CheckResult(name, CATEGORY, Status.WARNING, f"{label}: Timeout")
    except Exception as exc:  # noqa: BLE001 - any completion failure becomes a result row
        message = str(exc)
        lowered = message.lower()
        if "timeout" in lowered or "timed out" in lowered:
            return CheckResult(name, CATEGORY, Status.WARNING, f"{label}: Timeout")
        return CheckResult(name, CATEGORY, Status.ERROR, f"{label}: {message}")

    if isinstance(response, str) and response.strip():
        return CheckResult(name, CATEGORY, Status.OK, f"{label}: Chat completion successful")
    return CheckResult(name, CATEGORY, Status.WARNING, f"{label}: Empty response")
