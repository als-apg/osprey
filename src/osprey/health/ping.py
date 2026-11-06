"""One minimal real request to one provider, on explicit demand.

The poll-tier ``providers`` category asks each provider whether it answers
(its ``check_health``), which costs nothing. A *ping* goes one step further and
sends one real chat completion capped at one output token, so it costs a
request: it runs only when an operator asks for it (the panel's ``ping``
button), never on a schedule.

The model is the one the deployment itself uses with that provider — the
``model_id`` / ``model`` / ``default_model`` next to a ``provider:`` key that
names it, else the provider's own ``default_model`` — so the ping exercises the
route the deployment depends on rather than an arbitrary catalogue model.
"""

from __future__ import annotations

from collections.abc import Mapping
from time import perf_counter
from typing import Any

from osprey.health.offload import run_sync

#: The prompt, and the output cap: one token proves the round trip.
_PROMPT = "Reply with OK"
_MAX_TOKENS = 1

_MODEL_KEYS = ("model_id", "model", "default_model")


def model_for(config: Mapping[str, Any], provider: str) -> str | None:
    """The model this deployment uses with *provider* for chat, or ``None`` when none is named.

    A model named under an embedding role (a path segment containing
    ``embed``) is passed over: a completion sent to an embedding model is
    refused, which would report a working provider as broken.
    """
    found: list[str] = []

    def walk(node: Any, path: tuple[str, ...]) -> None:
        if isinstance(node, Mapping):
            if path == ("api", "providers"):
                return
            if node.get("provider") == provider and not any("embed" in p for p in path):
                for key in _MODEL_KEYS:
                    if isinstance(node.get(key), str) and node[key]:
                        found.append(node[key])
                        break
            for key, value in node.items():
                walk(value, (*path, str(key)))
        elif isinstance(node, list):
            for item in node:
                walk(item, path)

    walk(config, ())
    if found:
        return found[0]
    catalogue = (config.get("api") or {}).get("providers") or {}
    block = catalogue.get(provider) if isinstance(catalogue, Mapping) else None
    default = block.get("default_model") if isinstance(block, Mapping) else None
    return default if isinstance(default, str) and default else None


def _result(provider: str, status: str, message: str, **extra: Any) -> dict[str, Any]:
    return {
        "provider": provider,
        "status": status,
        "kind": extra.pop("kind", "completion"),
        "model": extra.pop("model", None),
        "latency_ms": extra.pop("latency_ms", None),
        "input_tokens": None,
        "output_tokens": None,
        "message": message,
    }


async def ping_provider(
    config: Mapping[str, Any] | None, provider: str, *, timeout_s: float = 30.0
) -> dict[str, Any]:
    """Send one one-token completion to *provider* and report how it went.

    Returns:
        ``provider``, ``status`` (``ok`` / ``warning`` / ``error`` / ``skip``),
        ``kind`` (``completion``, or ``embedding`` for a provider that serves
        no chat), ``model``, ``latency_ms``, ``input_tokens`` and
        ``output_tokens`` (``None``: the completion route does not report
        usage) and a one-line ``message``. Never raises.
    """
    cfg: Mapping[str, Any] = config or {}
    catalogue = (cfg.get("api") or {}).get("providers") or {}
    if provider not in catalogue:
        return _result(provider, "error", f"{provider} has no api.providers block")
    try:
        from osprey.models.provider_registry import get_provider_registry

        if not get_provider_registry().is_chat(provider):
            return _result(
                provider, "skip", "embeddings only; no completion to send", kind="embedding"
            )
    except Exception:  # an unreadable registry is no reason not to try
        pass
    model = model_for(cfg, provider)
    if model is None:
        return _result(provider, "error", f"no model named for {provider}")

    def _complete() -> Any:
        from osprey.models.completion import get_chat_completion

        return get_chat_completion(
            message=_PROMPT, provider=provider, model_id=model, max_tokens=_MAX_TOKENS
        )

    t0 = perf_counter()
    try:
        response = await run_sync(_complete, timeout_s=timeout_s)
    except TimeoutError:
        return _result(provider, "warning", f"no answer within {timeout_s:g} s", model=model)
    except Exception as exc:  # any completion failure is the answer
        return _result(provider, "error", f"{type(exc).__name__}: {exc}", model=model)
    latency_ms = (perf_counter() - t0) * 1000.0
    if isinstance(response, str) and response.strip():
        return _result(provider, "ok", "completion returned", model=model, latency_ms=latency_ms)
    return _result(provider, "warning", "empty completion", model=model, latency_ms=latency_ms)
