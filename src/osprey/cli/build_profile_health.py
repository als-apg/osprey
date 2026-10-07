"""The MCP health probe address, stated by the build for web-terminal renders.

A web terminal runs on the host's network, where a compose service name does
not resolve, and it dials its MCP servers from there. So a render whose
deployment serves web terminals carries ``health.auto.mcp.url_key: host_url``,
written by the build, rather than leaving the health category to guess the
terminal's network position at run time. ``host_url`` is the address the agent
itself dials, so a green probe row means the agent's own connection works.

The pin and the refusal of a contradicting ``config:`` spelling both read one
predicate, :func:`serves_web_terminals`, so the two cannot disagree. A render
whose deployment serves no web terminal gets no pin, and the key stays the
operator's own there.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

#: The dotted ``config:`` key naming which ``network`` address the derived MCP
#: health checks dial.
MCP_URL_KEY = "health.auto.mcp.url_key"

#: The one value a render that serves web terminals carries for
#: :data:`MCP_URL_KEY`: the host-network address the agent itself dials.
WEB_TERMINAL_MCP_URL_KEY = "host_url"


_WEB_TERMINALS_SEGMENTS = ("modules", "web_terminals")


def _on_web_terminals_path(key: str) -> bool:
    """Whether a dotted ``config:`` key is an ancestor of, or inside, ``modules.web_terminals``."""
    segments = tuple(key.split("."))
    depth = min(len(segments), len(_WEB_TERMINALS_SEGMENTS))
    return segments[:depth] == _WEB_TERMINALS_SEGMENTS[:depth]


def serves_web_terminals(config: Any) -> bool:
    """Whether the render of ``config`` serves web terminals.

    True when the effective ``modules.web_terminals`` subtree is enabled (the
    deployment stands up the web stack), or when it carries a non-empty
    ``personas`` catalog. The second arm is how a persona render answers yes:
    a persona render inherits its host's whole catalog with the module
    switched off, and a persona render is only ever a terminal image.

    Args:
        config: The profile's merged ``config:`` block.

    Returns:
        bool: Whether the render serves web terminals.
    """
    if not isinstance(config, Mapping):
        return False

    # Imported in-function: build_profile_emit closes an import cycle with the model.
    from .build_profile_emit import effective_web_terminals

    # Only keys on the subtree's own path take part, so a conflict elsewhere in
    # the block is reported by whatever reads that key, never by this verdict.
    on_path = {
        key: value
        for key, value in config.items()
        if isinstance(key, str) and _on_web_terminals_path(key)
    }
    web_terminals = effective_web_terminals(on_path)
    if web_terminals.get("enabled"):
        return True
    personas = web_terminals.get("personas")
    return isinstance(personas, Mapping) and bool(personas)


def health_config_overrides(config: Any) -> dict[str, str]:
    """The health key the build writes into a render of ``config``.

    Args:
        config: The profile's merged ``config:`` block.

    Returns:
        dict[str, str]: ``{MCP_URL_KEY: "host_url"}`` when the render serves web
        terminals, else empty.
    """
    if serves_web_terminals(config):
        return {MCP_URL_KEY: WEB_TERMINAL_MCP_URL_KEY}
    return {}


def health_url_key_errors(config: Any) -> list[str]:
    """One message per spelling of the probe address that contradicts the build.

    Only a render that serves web terminals is judged. There, a spelling of
    ``host_url`` agrees with what the build writes and is accepted; any other
    value is refused.

    Args:
        config: The profile's merged ``config:`` block.

    Returns:
        list[str]: Messages naming the key as written, the reason and the fix.
    """
    if not serves_web_terminals(config):
        return []

    # Imported in-function: the reach registry behind `spelled_values` pulls
    # the whole service-resolution package in with it.
    from .build_profile_reach import spelled_values

    errors: list[str] = []
    for spelling, value in spelled_values(config, MCP_URL_KEY):
        if value == WEB_TERMINAL_MCP_URL_KEY:
            continue
        errors.append(
            f"config: {spelling} is {value!r}, but this deployment serves web terminals, "
            f"which share the host's network: the build sets {MCP_URL_KEY} to "
            f"{WEB_TERMINAL_MCP_URL_KEY}, the address the agent itself dials. "
            "Remove the line from profile.yml."
        )
    return errors
