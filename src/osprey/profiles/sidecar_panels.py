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
        mcp_path: The absolute upstream path of an MCP transport this sidecar
            serves, or ``None``. A sidecar that declares one has that path
            refused by the panel proxy for every caller, along with any
            spelling that could be resolved to it past the proxy. Declaring
            ``agent_hop`` as well is the only thing that changes that.
        agent_hop: Whether the agent in a terminal session may reach
            ``mcp_path`` through that terminal's own panel proxy. Off by
            default, and meaningless without an ``mcp_path``: a transport is
            opened to the agent by saying so, never by being declared.
        mcp_allow: The names of the tools on that transport the agent may call
            without being asked, bare (no ``mcp__<server>__`` prefix).
        mcp_ask: The names of the tools on that transport that need the
            operator's approval for each call, bare. A tool in neither tuple
            is not offered to the agent. On the hop, a call to one of these
            is refused while the calling session is held, like a job fired on
            the dispatcher hop.
        mcp_writes: The subset of ``mcp_ask`` that writes the machine. Such a
            tool is governed: its hook rule carries the writes check ahead of
            the approval prompt, so the write switch's position decides it.
            Every other ask tool keeps its approval policy and is never held
            back by the switch.
        storage_scope_attribute: The ``<html>`` attribute the proxy stamps the
            mount's storage scope under, when the panel reads another name than
            the hub's own. ``None`` stands for the hub's own.
        session_writes: Whether the sidecar writes the machine for the
            terminal's sessions. Its factory is then also given the terminal's
            session hold store (``hold_root``) and a callable naming the
            sessions the terminal runs now (``live_sessions``), so a watchdog's
            hold on any of them holds the panel's writes too.
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
        stop_grace_s: Seconds the sidecar's stop waits between the stop signal
            and the kill: the one number its own ``stop`` hands the process
            stop as its grace, and the one the deployment adds the terminal's
            own stop to when it sizes the compose service's
            ``stop_grace_period``. A sidecar that finishes in-flight work on
            shutdown (a write it has already sent) declares what it needs;
            ``0`` is a sidecar whose stop waits for nothing.
    """

    name: str
    factory_path: str
    label: str = ""
    icon: str = ""
    description: str = ""
    health_endpoint: str | None = None
    operator_secret: bool = True
    mcp_path: str | None = None
    agent_hop: bool = False
    mcp_allow: tuple[str, ...] = ()
    mcp_ask: tuple[str, ...] = ()
    mcp_writes: tuple[str, ...] = ()
    storage_scope_attribute: str | None = None
    session_writes: bool = False
    entry_pages: tuple[str, ...] = ()
    image_checks: tuple[str, ...] = ()
    stop_grace_s: int = 0


#: The notebook panel's id. Named rather than spelled, because the terminal
#: reads it on a path that has nothing to do with panels — the session binding
#: an attach writes is only worth writing where this panel is served.
JUPYTER_PANEL_ID = "jupyter"

#: The CANOPY panel's id: the per-user operator GUI the terminal starts as a
#: sidecar. The integration package names the same id as its ``PANEL_ID``; it
#: is spelled here so reading the panel registry imports nothing from it.
CANOPY_PANEL_ID = "canopy"

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
        # jupyter-server closes its kernels on its own shutdown; this is how
        # long the stop lets it, before the group is killed.
        stop_grace_s=5,
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
    CANOPY_PANEL_ID: SidecarPanel(
        "CANOPY, the operator GUI: machine screens and Trace",
        "osprey.integrations.canopy.sidecar:CanopySidecar",
        label="CANOPY",
        icon="layers",
        description="Machine screens and Trace, which the agent reads and authors",
        health_endpoint="/health",
        # The viewer scopes its browser storage per user by this attribute.
        storage_scope_attribute="data-canopy-storage-scope",
        # A click on a screen writes for the terminal's sessions, and is held
        # with them.
        session_writes=True,
        # CANOPY authenticates the hop with its own launch credential; the
        # operator's proof has no reader there.
        operator_secret=False,
        # Its MCP transport, which the agent of this terminal reaches through
        # the terminal's own panel proxy and nobody else reaches at all.
        mcp_path="/mcp",
        agent_hop=True,
        # The read tools: what a screen is, what it shows and whether it holds,
        # and which collections group the panels. Then the workspace tools: the
        # tabs and tiles inside this user's own CANOPY panel, which the agent
        # reads and rearranges without a prompt; they write no PV and no file.
        mcp_allow=(
            "check",
            "read_pv",
            "search_pvs",
            "screenshot",
            "get_screen",
            "list_drafts",
            "list_widgets",
            "get_widget",
            "get_schema",
            "list_collections",
            "get_workspace",
            "set_workspace",
        ),
        # The authoring tools: each call changes the user's personal store, or
        # asks for a panel to be reviewed for the library.
        mcp_ask=(
            "new_screen",
            "patch_node",
            "add_node",
            "remove_node",
            "convert_legacy",
            "delete_draft",
            "request_push",
        ),
        # CANOPY's own application documents: the viewer's shell, a screen
        # (``/s/<name>``), an embedded screen and Trace. A draft
        # (``/drafts/...``) is a page an agent may have written, and every other
        # document is not the application: they stay sandboxed.
        entry_pages=("/", "/s/*", "/embed/*", "/trace/*"),
        # The CANOPY front-end relay's grace (5 s), then the CANOPY panel
        # backend's at the write budget's ceiling (D 24 s + 2 s), then the
        # relay's margin (3 s): a write already sent is finished, not killed. The sidecar's ``PANEL_STOP_GRACE_S`` derives
        # the same number; a test holds the two equal.
        stop_grace_s=34,
        # Two checks, both run by the CANOPY the image carries: the library
        # the image serves against its library report, then every
        # facility collection against CANOPY's rules and that library. The
        # module owns the probe, the skips and their wording; an image without
        # CANOPY passes both and says so.
        image_checks=(
            "python -I -m osprey.integrations.canopy.image_check library",
            "python -I -m osprey.integrations.canopy.image_check collections",
        ),
    ),
}
