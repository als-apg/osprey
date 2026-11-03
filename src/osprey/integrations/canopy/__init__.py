"""CANOPY on the deployment's control target.

:mod:`osprey.integrations.canopy.backend` holds the PV backend CANOPY loads by
its ``module:Class`` path, and :mod:`osprey.integrations.canopy.sidecar` runs
one user's CANOPY as two child processes of the web terminal. This package
imports nothing from CANOPY itself, so naming it never requires CANOPY to be
installed; the backend module does.

The names below are what the sidecar puts into its two children's
environments and what the backend reads back from its own. This package imports nothing outside the standard library and
its stdlib-only sibling :mod:`osprey.integrations.canopy.proven`, which holds
:data:`PROVEN_COMMITS`, so that a checker can read them where neither CANOPY
nor the terminal's dependencies are importable.
"""

from __future__ import annotations

from pathlib import Path

from osprey.integrations.canopy.proven import PROVEN_COMMITS

#: The panel id the per-user CANOPY is published under.
PANEL_ID = "canopy"

#: The header carrying the per-launch access credential to the front process.
ACCESS_HEADER = "X-Canopy-Access"

#: The backend class the data process loads, as CANOPY spells a backend path.
BACKEND_PATH = "osprey.integrations.canopy.backend:OspreyPVConnector"

#: The CANOPY modules every data process mounts.
MODULES = "hare,vole,fox,mole,converter"

#: Where saved views live, below the user's agent-data root.
VIEWS_SUBPATH = ("canopy", "views")

#: The user's personal panel store, below the user's agent-data root: the
#: read-write root the authoring tools write into. Its author is the only one
#: who sees it.
PERSONAL_SUBPATH = ("canopy", "drafts")

#: Where a user's push requests are written, below the user's agent-data root:
#: one bundle per request, for an admin to review on the host.
PUSH_SUBPATH = ("canopy", "push")

#: The deployment's screen library, below the project root: the read-only
#: second root handed to the front process when the directory exists.
LIBRARY_SUBPATH = Path("data/canopy/library")

#: The user's personal collection store, below the user's agent-data root:
#: handed to the front process on every render. Its owner is the only one who
#: sees it.
COLLECTIONS_SUBPATH = ("canopy", "collections")

#: The deployment's facility collection store, below the project root, beside
#: the screen library: handed to the front process when the directory exists.
FACILITY_COLLECTIONS_SUBPATH = Path("data/canopy/collections")

#: Names the record the data process's environment was rendered from, as
#: ``<target>@<generation>``, or :data:`NO_RECORD`. The sidecar sets it; the
#: backend compares it with the record its session reads.
RENDERED_FOR_ENV = "OSPREY_CANOPY_RENDERED_FOR"

#: What :data:`RENDERED_FOR_ENV` holds when no record was read.
NO_RECORD = "none"

#: The reachability a data process stated last, carried to its successor on an
#: unchanged record: ``no,<since>`` (epoch seconds) or ``yes``. Anything else
#: is no carry.
REACH_SINCE_ENV = "OSPREY_CANOPY_REACH_SINCE"

#: Where, in the data process's working directory (the sidecar's key
#: directory), the last ``yes`` / ``no`` it stated is kept for its successor:
#: ``{binding_key, state, since, written_at}``.
STATEMENT_FILENAME = "data-statement.json"

#: Where, in the data process's working directory (the sidecar's key
#: directory), the terminal names its live sessions: a JSON list of session
#: keys, rewritten whenever one starts or ends. The data process's writes are
#: held with every session it names; no file names none.
SESSIONS_FILENAME = "sessions.json"

#: ``T,S,D``: the write budget the sidecar computed. The data process takes the
#: smaller of it and its own, component by component.
WRITE_BUDGET_ENV = "OSPREY_CANOPY_WRITE_BUDGET"

#: Where the build stages the CANOPY payload (everything but the two wheels),
#: below the project root.
PAYLOAD_DIRNAME = "canopy_payload"

#: Every name of the archiver block starts with this; the block is rendered for
#: the data process only, and only when the archive is the bound target's.
MOLE_ENV_PREFIX = "CANOPY_MOLE__"

#: The names rendered into the data process's environment, the archiver block
#: aside. ``CANOPY_THEME_FILE`` is present only when a theme file is
#: configured; ``OSPREY_CANOPY_WRITE_BUDGET`` (``T,S,D``) is always rendered.
EXPECTED_DATA_CANOPY_VARS: frozenset[str] = frozenset(
    {
        "CONFIG_FILE",
        "OSPREY_CONFIG",
        "CANOPY_EPICS__BACKEND",
        "CANOPY_MODULES",
        "CANOPY_API_KEYS_FILE",
        "CANOPY_CLOSED_META",
        "CANOPY_VOLE__GLOBAL_BYTES_MAX",
        "OSPREY_CANOPY_RENDERED_FOR",
        "OSPREY_CANOPY_WRITE_BUDGET",
        "CANOPY_THEME_FILE",
    }
)

#: Where a web terminal's container sees the deployment's channel catalog: the
#: snapshot ``osprey build`` stages at ``build/web_terminals/channels.json``,
#: bind-mounted read-only at this one path into every web container when the
#: build wrote it. The front process is always told this path (``CANOPY_CHANNELS_FILE``);
#: without the mount it answers ``/channels`` with 404.
CHANNELS_FILE = "/etc/osprey/channels.json"

#: The stylesheets the embedded CANOPY pages link after their own, in cascade
#: order and relative to the web terminal's static root: OSPREY's design
#: tokens, its font faces, then the sheet that maps both onto CANOPY's host
#: contract. The front process is always told them (``CANOPY_HOST_STYLESHEETS``).
HOST_STYLESHEETS: tuple[str, ...] = (
    "design-system/css/tokens.css",
    "static/fonts/fonts.css",
    "design-system/css/canopy-host.css",
)

#: The environment variable that carries ``HOST_STYLESHEETS`` to the front process, as one
#: space-separated string.
HOST_STYLESHEETS_ENV = "CANOPY_HOST_STYLESHEETS"

#: The names of CANOPY's host contract, in the contract's own order. Each is
#: the custom property ``--canopy-host-<name>`` that the host sheet sets.
HOST_CONTRACT_NAMES: tuple[str, ...] = (
    "font-ui",
    "font-mono",
    "text-sm",
    "text-md",
    "text-lg",
    "radius-sm",
    "radius-md",
    "control-h-xs",
    "control-h-sm",
    "bg-base",
    "bg-panel",
    "bg-elevated",
    "bg-field",
    "text-primary",
    "text-secondary",
    "text-muted",
    "border",
    "border-strong",
    "accent",
    "on-accent",
    "accent-hover",
    "accent-tint",
    "hover-tint",
    "focus",
    "status-success",
    "status-warning",
    "status-error",
    "bar-h",
    "bar-icon",
    "tab-min-w",
    "tab-max-w",
    "tab-underline",
    "weight-medium",
    "shadow-dropdown",
    "z-dropdown",
)

#: The names rendered into the front process's environment.
#: ``CANOPY_THEME_FILE`` is present only when a theme file is configured,
#: ``CANOPY_TRACE_DIST`` only when the payload carries a built Trace,
#: ``CANOPY_LIBRARY_DIR`` only when the deployment has a screen library,
#: ``CANOPY_HOME_SCREEN`` only when ``web.panels.canopy.home_screen`` names a
#: screen that library holds (the home screen is configuration; unset, CANOPY
#: opens its first screen) and ``CANOPY_FACILITY_COLLECTIONS_DIR`` only when
#: the deployment has a facility collection store; the user's
#: ``CANOPY_COLLECTIONS_DIR`` is always rendered. The four write names (the PUT
#: relay, its timeout, the write key file and the confirm flag) are always
#: rendered, and so are the three default queries, the channel catalog's path and the host stylesheets.
EXPECTED_FRONT_CANOPY_VARS: frozenset[str] = frozenset(
    {
        "CANOPY_BASE_URL",
        "CANOPY_SELF_URL",
        "CANOPY_ACCESS_TOKEN_FILE",
        "CANOPY_API_KEY_FILE",
        "CANOPY_DRAFTS_DIR",
        "CANOPY_COLLECTIONS_DIR",
        "CANOPY_FACILITY_COLLECTIONS_DIR",
        "CANOPY_LIBRARY_DIR",
        "CANOPY_HOME_SCREEN",
        "CANOPY_PUSH_DIR",
        "CANOPY_VIEWS_DIR",
        "CANOPY_VIEWER_DIST",
        "CANOPY_TRACE_DIST",
        "CANOPY_URL_PREFIX",
        "CANOPY_PROXY_MARK_HEADERS",
        "CANOPY_LIBRARY_IGNORED_NAMES",
        "CANOPY_AUTHORING",
        "CANOPY_WORKSPACE",
        "CANOPY_BROWSER_SANDBOX",
        "CANOPY_RELAY_CONNECT_RETRY_S",
        "CANOPY_TRACE_DEFAULT_QUERY",
        "CANOPY_CORRELATION_DEFAULT_QUERY",
        "CANOPY_WAVEFORM_DEFAULT_QUERY",
        "CANOPY_CHANNELS_FILE",
        "CANOPY_HOST_STYLESHEETS",
        "CANOPY_PUT_RELAY",
        "CANOPY_PUT_RELAY_TIMEOUT_S",
        "CANOPY_WRITE_API_KEY_FILE",
        "CANOPY_THEME_FILE",
    }
)

#: The names of each set that a correct environment may lack.
CONDITIONAL_DATA_CANOPY_VARS: frozenset[str] = frozenset({"CANOPY_THEME_FILE"})
CONDITIONAL_FRONT_CANOPY_VARS: frozenset[str] = frozenset(
    {
        "CANOPY_TRACE_DIST",
        "CANOPY_THEME_FILE",
        "CANOPY_LIBRARY_DIR",
        "CANOPY_HOME_SCREEN",
        "CANOPY_FACILITY_COLLECTIONS_DIR",
    }
)

__all__ = [
    "ACCESS_HEADER",
    "BACKEND_PATH",
    "CHANNELS_FILE",
    "COLLECTIONS_SUBPATH",
    "CONDITIONAL_DATA_CANOPY_VARS",
    "CONDITIONAL_FRONT_CANOPY_VARS",
    "EXPECTED_DATA_CANOPY_VARS",
    "EXPECTED_FRONT_CANOPY_VARS",
    "FACILITY_COLLECTIONS_SUBPATH",
    "HOST_CONTRACT_NAMES",
    "HOST_STYLESHEETS",
    "HOST_STYLESHEETS_ENV",
    "LIBRARY_SUBPATH",
    "MODULES",
    "MOLE_ENV_PREFIX",
    "NO_RECORD",
    "PANEL_ID",
    "PAYLOAD_DIRNAME",
    "PERSONAL_SUBPATH",
    "PROVEN_COMMITS",
    "PUSH_SUBPATH",
    "REACH_SINCE_ENV",
    "RENDERED_FOR_ENV",
    "SESSIONS_FILENAME",
    "STATEMENT_FILENAME",
    "VIEWS_SUBPATH",
    "WRITE_BUDGET_ENV",
]
