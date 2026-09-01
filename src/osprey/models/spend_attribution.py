"""Spend attribution: the identity every request to a LiteLLM gateway carries.

A deployment authenticates to its LLM gateway with ONE credential, so at the
gateway every terminal, the dispatch worker and every headless run look like
the same caller. The gateway prices each call and can attribute it — but only
if the request says who asked. LiteLLM reads two headers for that:

* ``x-litellm-end-user-id`` — the end user the spend is booked to
  (``/customer/info``, ``end_user`` in the spend logs);
* ``x-litellm-tags`` — free-form tags on the spend-log row (``request_tags``).

OSPREY already knows who is asking: :func:`osprey.utils.identity.acting_identity`
resolves the roster user of a terminal container, the framework identity of a
service container, or the local account. This module turns that answer into
those two headers and delivers them on both LLM call paths:

* **The agent.** Claude Code adds ``ANTHROPIC_CUSTOM_HEADERS`` (newline-
  separated ``Name: Value`` pairs) to every request it makes, so the launch
  paths set that variable through :func:`apply_attribution_env`. It merges into
  an operator's own value rather than replacing it — that variable is also the
  one way to carry corporate-proxy headers, which is why the resolver never
  scrubs it.
* **The LiteLLM SDK path** (structured completions from MCP servers and
  services) sets the same identity as the OpenAI ``user`` field plus
  ``extra_headers``; see ``providers/litellm_adapter.py``.

Only providers fronted by a LiteLLM gateway get the headers (:func:`gateway_for`).
A direct provider would ignore them at best, so it gets none.
"""

from __future__ import annotations

import os
from collections.abc import Mapping, MutableMapping

from osprey.utils.identity import AUDIT_IDENTITY_ENV, TERMINAL_USER_ENV, acting_identity

#: The one gateway kind OSPREY knows how to attribute spend on.
LITELLM_GATEWAY = "litellm"

#: LiteLLM's end-user header — the spend is booked to this id.
END_USER_HEADER = "x-litellm-end-user-id"

#: LiteLLM's request-tags header — a comma-separated list on the spend-log row.
TAGS_HEADER = "x-litellm-tags"

#: Claude Code's extra-request-headers variable: ``Name: Value`` per line.
CUSTOM_HEADERS_ENV = "ANTHROPIC_CUSTOM_HEADERS"

#: The tag every OSPREY request carries, so a gateway shared with other
#: consumers can pick OSPREY's spend out of the rest.
_PRODUCT_TAG = "osprey"

#: The ``surface:<name>`` tag values, keyed by how the process is identified.
SURFACE_TERMINAL = "terminal"
SURFACE_DISPATCH = "dispatch"
SURFACE_SERVICE = "service"
SURFACE_LOCAL = "local"

#: Built-in providers that are LiteLLM proxies. A custom ``api.providers``
#: entry declares the same with ``gateway: litellm``.
_BUILTIN_GATEWAYS: Mapping[str, str] = {
    "als-apg": LITELLM_GATEWAY,
    "cborg": LITELLM_GATEWAY,
}


def gateway_for(
    provider_name: str, api_providers: Mapping[str, Mapping] | None = None
) -> str | None:
    """Return the gateway kind fronting *provider_name*, or ``None`` for a direct provider.

    A custom provider's ``gateway`` key wins over nothing: the built-in table
    only names providers that have no ``api.providers`` entry of their own.
    """
    if provider_name in _BUILTIN_GATEWAYS:
        return _BUILTIN_GATEWAYS[provider_name]
    if api_providers:
        declared = api_providers.get(provider_name, {}).get("gateway")
        if isinstance(declared, str) and declared.strip():
            return declared.strip()
    return None


def acting_surface() -> str:
    """Which kind of process is asking — the ``surface:`` tag value.

    Follows the same ladder as :func:`osprey.utils.identity.acting_identity`:
    a terminal container names a person; a service container names a
    framework identity, and the dispatch worker is the one whose runs a
    facility wants to see apart from the rest; everything else is a local
    session on somebody's own machine.
    """
    if (os.environ.get(TERMINAL_USER_ENV) or "").strip():
        return SURFACE_TERMINAL
    audit_identity = (os.environ.get(AUDIT_IDENTITY_ENV) or "").strip()
    if audit_identity:
        return SURFACE_DISPATCH if audit_identity.startswith("dispatch-worker") else SURFACE_SERVICE
    return SURFACE_LOCAL


def attribution_tags() -> str:
    """The comma-separated ``x-litellm-tags`` value for this process."""
    return f"{_PRODUCT_TAG},surface:{acting_surface()}"


def attribution_headers() -> dict[str, str]:
    """The two headers a request from this process should carry."""
    return {
        END_USER_HEADER: acting_identity(),
        TAGS_HEADER: attribution_tags(),
    }


def render_custom_headers(headers: Mapping[str, str]) -> str:
    """Render *headers* in Claude Code's ``ANTHROPIC_CUSTOM_HEADERS`` format."""
    return "\n".join(f"{name}: {value}" for name, value in headers.items())


def merge_custom_headers(existing: str | None, headers: Mapping[str, str]) -> str:
    """Add *headers* to an ``ANTHROPIC_CUSTOM_HEADERS`` value, keeping the operator's own.

    A line already naming one of *headers* (case-insensitively) is replaced —
    a stale attribution must not ride beside a fresh one — and every other
    line is kept in its place. Blank lines are dropped.
    """
    ours = {name.lower() for name in headers}
    kept: list[str] = []
    for line in (existing or "").splitlines():
        line = line.strip()
        if not line:
            continue
        name = line.split(":", 1)[0].strip().lower()
        if name in ours:
            continue
        kept.append(line)
    kept.append(render_custom_headers(headers))
    return "\n".join(kept)


def apply_attribution_env(environ: MutableMapping[str, str], gateway: str | None) -> None:
    """Set ``ANTHROPIC_CUSTOM_HEADERS`` in *environ* when *gateway* attributes spend.

    Mutates *environ* in place. A ``None`` or unknown gateway leaves it
    untouched — including any operator value already there.
    """
    if gateway != LITELLM_GATEWAY:
        return
    environ[CUSTOM_HEADERS_ENV] = merge_custom_headers(
        environ.get(CUSTOM_HEADERS_ENV), attribution_headers()
    )
