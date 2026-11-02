"""Sidecar panel registry — the panels the web terminal starts itself.

A leaf module: it imports ``dataclasses`` and nothing else, and must stay that
way. ``osprey.profiles.web_panels`` re-exports everything here and is where
panel code reads it from, but that module also reads the companion-server
registry and the shared middleware, and so loads starlette and the web
credential module. Two readers cannot go through it: the MCP server registry
(``osprey.registry.mcp``), which every build and the agent runner import, and
the credential module (``osprey.interfaces.web_auth``), which that middleware
itself imports. Both read this module instead.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SidecarPanel:
    """A built-in panel served by a process the web terminal starts itself.

    The companion-server registry (``registry.web``) is scoped to servers
    ``ServerLauncher`` runs from a config section with a host, a port and an
    ``auto_launch`` key. A sidecar has none of those: it is started by the
    terminal's lifespan, listens on an ephemeral loopback port, and is reached
    only through the terminal's own panel proxy. Registering one there would
    hand it a port band, a port env var and a health probe that mean nothing.

    The entry is the one description of the panel. The browser builds the
    panel's sidebar row and tab from it (label, icon, description, endpoints),
    the proxy reads its credential posture from it, and the build reads its
    image checks from it; none of them restates a field.

    Attributes:
        name: One-line description, used as the setup skill's menu entry
            for the panel id.
        factory_path: ``module:attribute`` of the sidecar class the terminal
            imports to start it. A dotted path rather than the object so this
            module stays importable without pulling the sidecar's dependencies
            into every process that reads the panel registry.
        label: The display label of the panel's sidebar row and tab. Empty
            means the upper-cased panel id.
        icon: The name of the design system's ``icons.js`` glyph the sidebar
            row wears. Empty means the sidebar's generic panel glyph.
        description: One plain sentence for the sidebar row's tooltip. Empty
            means a bare label.
        health_endpoint: The path below ``/panel/<id>`` the browser polls for
            the panel's health, or ``None`` for a panel that is not polled and
            is taken as healthy once it is available.
        operator_secret: Whether the panel proxy re-issues the operator secret
            (``X-Osprey-Terminal-Secret``) toward this sidecar. A sidecar that
            authenticates with its own launch credential alone sets it false,
            so the operator's proof never reaches a process that has no use
            for it. The launch credential is injected either way.
        entry_pages: The panel's own application documents, as route
            templates below ``/panel/<id>`` (``/``, ``/lab``, ``/s/*`` — a final
            ``*`` segment matches the rest of the path). The panel proxy serves
            every other document this sidecar returns — HTML, SVG, XML, text, an
            untyped body — into an opaque sandbox origin, because a document
            that is not the panel's own page may be one an agent wrote. Matched
            after percent-decoding and dot-segment removal. Empty means every
            document is sandboxed.
        image_checks: Shell commands an image build runs from the deployment
            directory once the deployment is copied in, one per entry. A
            failing entry fails the build, so each one decides for itself what
            a missing optional package means.
    """

    name: str
    factory_path: str
    label: str = ""
    icon: str = ""
    description: str = ""
    health_endpoint: str | None = None
    operator_secret: bool = True
    entry_pages: tuple[str, ...] = ()
    image_checks: tuple[str, ...] = ()


#: The notebook panel's id. Named rather than spelled, because the terminal
#: reads it on a path that has nothing to do with panels — the session binding
#: an attach writes is only worth writing where this panel is served.
JUPYTER_PANEL_ID = "jupyter"

#: Built-in panels the web terminal serves from a sidecar process rather than a
#: registered companion server. Keyed by panel id, exactly like the ids derived
#: from the companion registry, so every consumer of
#: ``osprey.profiles.web_panels.BUILTIN_PANELS`` covers them without a second
#: lookup. Registry order is sidebar order.
SIDECAR_PANELS: dict[str, SidecarPanel] = {
    JUPYTER_PANEL_ID: SidecarPanel(
        "JupyterLab notebooks",
        "osprey.interfaces.web_terminal.jupyter_sidecar:JupyterSidecar",
        label="JUPYTER",
        icon="notebook",
        description="JupyterLab with kernels that follow the terminal session",
        # The sidecar's own status route, reached through the panel proxy.
        health_endpoint="/api/status",
        # JupyterLab's application documents: the Lab shell's page routes and
        # the classic page routes. A user's file (``/files/``), a rendered
        # export (``/nbconvert/``), the ``/view/`` wrapper and every static or
        # API subtree (``/lab/extensions/``, ``/lab/api/``, ``/static/``) are
        # not the application, and stay sandboxed.
        entry_pages=(
            "/lab",
            "/lab/tree/*",
            "/lab/workspaces/*",
            "/lab/doc/*",
            "/tree",
            "/tree/*",
            "/notebooks/*",
            "/doc",
            "/doc/*",
            "/consoles/*",
            "/edit/*",
        ),
    ),
}
