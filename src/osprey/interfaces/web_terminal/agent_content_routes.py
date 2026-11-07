"""Every GET route OSPREY serves to a panel that answers with non-JSON bytes.

The opaque origin for agent pages is enforced by default-deny points, not by
this list: the artifact app sandboxes every content route
(``artifacts/content_sandbox.py``) and the terminal sandboxes every proxied or
discovered document that is not a declared entry page (``panel_sandbox.py``).
This table is the enumeration those points are *held to*:

* the drift test (``tests/interfaces/web_terminal/test_panel_sandbox.py``)
  scans the companion servers' route decorators and fails when a GET route that
  builds a non-JSON response is neither listed here, a shell asset
  (:data:`SHELL_ASSET_ROUTES`) nor one of its panel's entry pages, and fails when a listed route no longer exists — so a new surface is
  classified when it is added, not when it is probed;
* the agent-page protection probe
  (``tests/deployment/test_agent_page_protections_drift.py``) reads
  :data:`AGENT_CONTENT_ROUTES` to request one sample of each surface kind and
  check it is served in an opaque origin.

A row is one **surface kind**: the routes that serve the same kind of bytes,
and one sample browser path (below the terminal's root, ``{placeholders}``
filled by the probe) that exercises it.

A leaf module: stdlib only, so the probe tooling can read it without the web
stack.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RouteRef:
    """One route as its source declares it.

    Attributes:
        module: Dotted module that declares the route.
        template: The path literal in the route decorator, before any
            ``include_router`` prefix.
    """

    module: str
    template: str


@dataclass(frozen=True)
class AgentContentRoute:
    """One surface kind of non-JSON panel content.

    Attributes:
        kind: Stable name of the surface kind.
        panel_id: The panel the surface is served under (``/panel/<id>/``), or
            ``""`` for a route of the terminal itself.
        routes: Every route serving this kind.
        sample_path: A browser path below the terminal root that serves one
            sample of the kind; ``{name}`` segments are filled by the caller.
        sandboxed_by: The default-deny points that put the surface in an opaque
            origin (``artifact_sandbox``, ``panel_sandbox``, ``route_sandbox``
            for a route that sends the policy itself, ``upstream_sandbox`` for
            the backend's own policy).
    """

    kind: str
    panel_id: str
    routes: tuple[RouteRef, ...]
    sample_path: str
    sandboxed_by: tuple[str, ...]


_ARTIFACTS = "osprey.interfaces.artifacts.app"
_ARIEL_ROUTES = "osprey.interfaces.ariel.api.routes"
_ARIEL_DRAFTS = "osprey.interfaces.ariel.api.drafts"
_CF_REVIEW = "osprey.interfaces.channel_finder.pending_review_api"
_TERMINAL_PANELS = "osprey.interfaces.web_terminal.routes.panels"
_TERMINAL_PROXY = "osprey.interfaces.web_terminal.routes.proxy"
_JUPYTER_FILES = "jupyter_server.files.handlers"
_JUPYTER_NBCONVERT = "jupyter_server.nbconvert.handlers"

#: The enumeration, one row per surface kind.
AGENT_CONTENT_ROUTES: tuple[AgentContentRoute, ...] = (
    AgentContentRoute(
        kind="artifact-file",
        panel_id="artifacts",
        routes=(RouteRef(_ARTIFACTS, "/files/{artifact_id}/{filename}"),),
        sample_path="/panel/artifacts/files/{artifact_id}/{filename}",
        sandboxed_by=("artifact_sandbox", "panel_sandbox"),
    ),
    AgentContentRoute(
        kind="artifact-rendered-page",
        panel_id="artifacts",
        routes=(
            RouteRef(_ARTIFACTS, "/api/notebooks/{artifact_id}/rendered"),
            RouteRef(_ARTIFACTS, "/api/markdown/{artifact_id}/rendered"),
        ),
        sample_path="/panel/artifacts/api/markdown/{artifact_id}/rendered",
        sandboxed_by=("artifact_sandbox", "panel_sandbox"),
    ),
    AgentContentRoute(
        kind="event-stream",
        panel_id="artifacts",
        routes=(RouteRef(_ARTIFACTS, "/api/events"),),
        sample_path="/panel/artifacts/api/events",
        sandboxed_by=("artifact_sandbox", "panel_sandbox"),
    ),
    AgentContentRoute(
        kind="logbook-attachment",
        panel_id="ariel",
        routes=(
            RouteRef(_ARIEL_ROUTES, "/attachments/{attachment_id}"),
            RouteRef(_ARIEL_ROUTES, "/attachments/{attachment_id}/rendition"),
            RouteRef(_ARIEL_DRAFTS, "/drafts/{draft_id}/attachments/{filename}"),
        ),
        sample_path="/panel/ariel/api/attachments/{attachment_id}",
        sandboxed_by=("route_sandbox", "panel_sandbox"),
    ),
    AgentContentRoute(
        kind="channel-finder-artifact",
        panel_id="channel-finder",
        routes=(RouteRef(_CF_REVIEW, "/artifacts/{filename}"),),
        sample_path="/panel/channel-finder/api/artifacts/{filename}",
        sandboxed_by=("route_sandbox", "panel_sandbox"),
    ),
    AgentContentRoute(
        kind="discovered-panel",
        panel_id="",
        routes=(RouteRef(_TERMINAL_PANELS, "/panel-static/{panel_id}/{path:path}"),),
        sample_path="/panel-static/{panel_id}/",
        sandboxed_by=("panel_sandbox",),
    ),
    AgentContentRoute(
        kind="registered-panel",
        panel_id="{panel_id}",
        routes=(RouteRef(_TERMINAL_PROXY, "/panel/{panel_id}/{path:path}"),),
        sample_path="/panel/{panel_id}/",
        sandboxed_by=("panel_sandbox",),
    ),
    AgentContentRoute(
        kind="notebook-file",
        panel_id="jupyter",
        routes=(
            RouteRef(_JUPYTER_FILES, "/files/{path}"),
            RouteRef(_JUPYTER_NBCONVERT, "/nbconvert/{format}/{path}"),
        ),
        sample_path="/panel/jupyter/files/{path}",
        sandboxed_by=("upstream_sandbox", "panel_sandbox"),
    ),
)

#: Script modules a companion server's shell imports, served from a route
#: rather than its ``/static`` mount: installed facility code, never agent
#: content. Each is ``not_content`` in its app, answers ``text/javascript`` with
#: ``nosniff`` (a type the terminal's panel sandbox passes untouched), and is
#: answered by the hub itself as a first-party asset path. Classified here so
#: the drift test holds them to the code like every other non-JSON route; a
#: route listed here is neither a surface kind nor an entry page.
SHELL_ASSET_ROUTES: tuple[RouteRef, ...] = (
    # An extension type's viewer, the one file resolved for its key at load.
    RouteRef(_ARTIFACTS, "/viewers/{key}.js"),
)

#: Modules whose routes are OSPREY's companion servers: the drift test scans
#: their route decorators. Routes outside them (the terminal's own, and
#: jupyter-server's) are listed above for the probe and pinned by their own tests.
SCANNED_PACKAGES: tuple[str, ...] = (
    "osprey.interfaces.artifacts",
    "osprey.interfaces.ariel",
    "osprey.interfaces.channel_finder",
    "osprey.interfaces.okf_panel",
    "osprey.interfaces.health",
)


def surface_kinds() -> tuple[str, ...]:
    """The surface kinds, in table order."""
    return tuple(row.kind for row in AGENT_CONTENT_ROUTES)
