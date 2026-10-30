"""Read the rendered ``nginx.conf`` the way the web-terminal render tests assert on it."""

from __future__ import annotations

import re

from osprey.deployment.web_terminals.render import render_web_terminals


def render_nginx(config: dict) -> str:
    return render_web_terminals(config)["nginx/nginx.conf"]


def nginx_directives(conf: str) -> str:
    """`conf` with its comment lines dropped.

    Counting and absence assertions have to ignore comment prose, which in this
    template frequently names the very construct being counted or ruled out —
    the comments explain `auth_request`, `$request_uri` and `/verify` at length,
    and the one above `error_page` would otherwise be counted as a second
    `auth_request`.
    """
    return "\n".join(line for line in conf.splitlines() if not line.lstrip().startswith("#"))


def nginx_server_blocks(conf: str) -> list[str]:
    """The top-level `server { … }` bodies, in render order.

    Under TLS there are THREE: the redirect-only plain-port server first, then
    the 443 content server, then the redirect for every other name on the TLS
    listener. Without TLS there are TWO: the content server, then the redirect
    for every other name on the plain port. Split on the closing brace in
    column 0, which only a top-level block has (every nested `location` closes
    at an indent).
    """
    blocks = re.findall(r"^server \{\n(.*?)^\}$", conf, flags=re.DOTALL | re.MULTILINE)
    assert blocks, "no top-level server block in the rendered fragment"
    return blocks


def nginx_location_body(conf: str, header: str) -> str:
    """The directives inside a top-level `location …` block, by its header line."""
    match = re.search(rf"{re.escape(header)} \{{(.*?)\n    \}}", conf, flags=re.DOTALL)
    assert match is not None, f"no `{header}` block in the rendered fragment"
    return match.group(1)


#: The comment that opens the history page's origin, the server a signed-in
#: deployment renders after the terminals' servers for its history page.
HISTORY_ORIGIN_MARKER = "# The history page's origin.\n"


def nginx_terminals_part(conf: str) -> str:
    """The terminals' servers of *conf*: everything before the history page's origin.

    The history page's origin gates its page with an ``auth_request``, an
    ``internal`` verify target and a gate include of its own, which the
    per-user counts are not about.
    """
    return conf.partition(HISTORY_ORIGIN_MARKER)[0]
