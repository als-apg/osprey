"""MCP tools: native Phoebus interaction via the agent bridge.

Each tool is a thin validation + HTTP-delegation layer over one endpoint of the
Phoebus agent bridge (the in-JVM JSON/HTTP server embedded in a running Phoebus
product):

==========================  =================================================
Tool                        Bridge endpoint
==========================  =================================================
phoebus_open_panel          POST /open
phoebus_panel_lookup        (none — reads the panel registry)
phoebus_list_displays       GET  /displays
phoebus_perceive            GET  /perceive
phoebus_perceive_region     GET  /perceive/region
phoebus_snapshot            GET  /snapshot            (image/png)
phoebus_drive               POST /drive
==========================  =================================================

The HTTP primitives (``_http_get_json`` / ``_http_get_bytes`` /
``_http_post_drive`` / ``_http_post_open``) are module-level so tests can patch
the network boundary.

Control-system target
---------------------
The bridge addresses one running Phoebus product, whose PV context was fixed
when that product started: a session-level control-system target switch does
not re-address it. Phoebus therefore stays pinned to the deployment baseline,
and these tools say so rather than letting the discrepancy pass silently —
``phoebus_drive`` refuses outright while the deployment is switched, and every
read tool prepends one informational line to its normal output. Both strings
come from :mod:`osprey.mcp_server.control_system.target_banner`, shared with
the health runtime's equivalent row. On the baseline nothing is added and the
output is byte-identical.

Display addressing
------------------
Every tool that accepts a ``display`` argument supports three forms:

* ``"active"`` — the currently focused display (default).
* A display name as returned by ``phoebus_list_displays``.
* ``"handle:<id>"`` — the deterministic handle returned by
  ``phoebus_open_panel`` (e.g. ``"handle:d-3"``).  Use this form when
  multiple displays are open and you need to target the one you just opened.

On a backend shared by multiple web terminals, ``"active"`` resolves the
process-global focused display — a race when two terminals perceive/drive
concurrently. A multi-user web-terminal deployment therefore stamps
``PHOEBUS_REQUIRE_HANDLE=1`` on every terminal whose project runs a Phoebus
server, which rejects the implicit ``"active"`` fallback and makes callers
address a specific display (a handle or an explicit name). Elsewhere the switch
is off by default and is turned on with ``phoebus.require_handle: true`` in
config.yml or ``PHOEBUS_REQUIRE_HANDLE=1``. ``phoebus.require_handle: false``
keeps ``"active"`` in a multi-user deployment too.

Agent access
------------
``phoebus_drive`` is offered and served only when ``phoebus.agent_access`` is
``read_write``. Under the default ``read`` the server leaves it out of
``tools/list`` and refuses a call to it by the key's name: a panel runs
whatever its widgets are wired to, so a deployment offers the drive or not.

A drive the deployment offers is a write, so it then follows its machine's
switch like every other write: the write verdict
(:func:`osprey_connectors.posture_store.execution_verdict`) for the baseline, the
only machine the bridge drives, must allow it, and where the approval policy
asks, the approval prompt must have left its record for the call
(:func:`osprey.mcp_server.approval_record.require_approval_record`).

Panel name registry
-------------------
``phoebus_open_panel`` resolves logical panel names to ``.bob`` resources via:

1. ``phoebus.panels.<name>`` in config.yml — value may be an absolute path,
   a ``file:`` URL, or a path relative to the config file's directory. This is
   the deployment path for every facility. The value may instead be a mapping
   with that path as ``file`` and, as ``macros``, the list of macro names the
   panel takes; :func:`_panel_declaration` is the one reader of both forms.
2. An optional framework built-in registry (empty by default; a framework
   built-in, if ever added, is resolved relative to the osprey repository root
   in a development checkout).

``phoebus_panel_lookup`` answers the same registry in the other direction: given
a display path it returns the name that opens it on this terminal, so a server
that indexes the facility's display tree can hand back paths and let the
framework judge them under the caller's own registry.
"""

import asyncio
import contextlib
import json
import logging
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import anyio
from fastmcp.exceptions import ToolError

from osprey.mcp_server.control_system.target_banner import (
    BASELINE_REFUSAL_ERROR_TYPE,
    PHOEBUS_SUBJECT,
    baseline_pinned_line,
    baseline_refusal,
    prepend_line,
)
from osprey.mcp_server.errors import make_error
from osprey.mcp_server.http import (
    _post_json_with_response,
    notify_agent_activity_async,
    panel_op,
    phoebus_bridge_url,
)
from osprey.mcp_server.phoebus.server import mcp
from osprey.phoebus_agent_access import AGENT_ACCESS_KEY, READ, READ_WRITE, agent_access
from osprey.utils.session_hold_api import execution_verdict, refusal_detail
from osprey_connectors import posture_store
from osprey_connectors.types import baseline_target
from osprey_connectors.workspace import (
    agent_data_base_dir,
    anchored_path,
    load_osprey_config,
    resolve_config_path,
    resolve_project_root,
)

logger = logging.getLogger("osprey.mcp_server.tools.phoebus")

# Module-level alias so tests can patch bridge deadline timing without touching
# the global time.monotonic that anyio uses internally for its thread-pool logic.
_monotonic = time.monotonic

_TIMEOUT = 15  # seconds — snapshot/perceive marshal onto the Phoebus FX thread

_UNREACHABLE_HINTS = [
    "Start a Phoebus product with the agent bridge enabled (default port 7979).",
    "Check the PHOEBUS_BRIDGE_URL env var and the phoebus.host/phoebus.port values in config.yml.",
    "To change them, set phoebus.host/phoebus.port in the build profile (profile.yml on the host), then rebuild and redeploy.",
    "Confirm the bridge is reachable, e.g. curl http://127.0.0.1:7979/displays.",
]


# ---------------------------------------------------------------------------
# HTTP boundary (patched in tests)
# ---------------------------------------------------------------------------
def _http_get_json(path: str) -> tuple[int, dict | list]:
    """GET ``path`` on the bridge and return ``(status, parsed_json)``."""
    url = f"{phoebus_bridge_url()}{path}"
    try:
        with urllib.request.urlopen(url, timeout=_TIMEOUT) as resp:
            return resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        body: dict = {}
        with contextlib.suppress(Exception):
            body = json.loads(exc.read())
        return exc.code, body


def _http_get_bytes(path: str) -> tuple[int, dict, bytes]:
    """GET ``path`` and return ``(status, headers, raw_body)`` (for image/png)."""
    url = f"{phoebus_bridge_url()}{path}"
    try:
        with urllib.request.urlopen(url, timeout=_TIMEOUT) as resp:
            return resp.status, dict(resp.headers), resp.read()
    except urllib.error.HTTPError as exc:
        return exc.code, dict(exc.headers), exc.read()


def _http_post_drive(payload: dict) -> tuple[int, dict]:
    """POST a drive request and return ``(status, parsed_json)``."""
    return _post_json_with_response(f"{phoebus_bridge_url()}/drive", payload, timeout=_TIMEOUT)


def _http_post_open(payload: dict) -> tuple[int, dict]:
    """POST an open request and return ``(status, parsed_json)``."""
    return _post_json_with_response(f"{phoebus_bridge_url()}/open", payload, timeout=_TIMEOUT)


def _bridge_error_message(body: object, status: int) -> str:
    """Extract the bridge's JSON ``error`` message, falling back to the status."""
    if isinstance(body, dict) and body.get("error"):
        return str(body["error"])
    return f"Phoebus bridge returned HTTP {status}."


def _snapshot_dir() -> Path:
    """Resolve the directory PNG snapshots are written to (config or default).

    Runtime output, so it belongs under the deployment's agent-data root — read
    from ``agent_data.base_dir`` rather than spelled here — and a configured
    ``phoebus.snapshot_dir`` is anchored on the repo root the same way every
    other configured path is. Both halves matter: neither the default nor the
    configured value may resolve against the working directory, which for an
    MCP server is whatever launched it.
    """
    config = load_osprey_config()
    configured = (config.get("phoebus", {}) or {}).get("snapshot_dir")
    relative = str(configured or f"{agent_data_base_dir(config)}/screenshots")
    out = anchored_path(relative, resolve_project_root(config))
    out.mkdir(parents=True, exist_ok=True)
    return out


# ---------------------------------------------------------------------------
# Require-handle addressing (off by default; see module docstring)
# ---------------------------------------------------------------------------
_TRUTHY = {"1", "true", "yes", "on"}
_FALSY = {"0", "false", "no", "off"}


def _require_handle() -> bool:
    """Whether implicit ``"active"`` display addressing must be rejected.

    Resolution order (mirrors ``osprey.interfaces.vendor.is_offline``):

    1. ``PHOEBUS_REQUIRE_HANDLE`` env var (truthy: 1/true/yes/on; falsy:
       0/false/no/off) — wins outright. The multi-user web-terminal render
       stamps ``PHOEBUS_REQUIRE_HANDLE=1`` on every terminal container whose
       project runs a Phoebus server and does not set rung 2 to false. The
       server definition never sets it, so a single-instance deployment is
       governed by rung 2.
    2. ``phoebus.require_handle`` in config.yml.
    3. ``False`` default — existing ``"active"`` fallback behavior is unchanged.
    """
    env = os.environ.get("PHOEBUS_REQUIRE_HANDLE", "").strip().lower()
    if env in _TRUTHY:
        return True
    if env in _FALSY:
        return False
    config = load_osprey_config()
    return bool(config.get("phoebus", {}).get("require_handle", False))


def _check_drive_offered() -> None:
    """Refuse a drive unless ``phoebus.agent_access`` is ``read_write``.

    The refusal names the key, so a client whose settings still offer the
    tool learns which switch withholds it.
    """
    try:
        access = agent_access(load_osprey_config())
    except ValueError as exc:
        make_error(
            "configuration_error",
            str(exc),
            [
                f"Set {AGENT_ACCESS_KEY} to {READ!r} or {READ_WRITE!r} in the build profile and rebuild."
            ],
            details={"key": AGENT_ACCESS_KEY},
        )
    if access == READ:
        make_error(
            "not_supported",
            f"Driving a Phoebus widget is off on this deployment: {AGENT_ACCESS_KEY} is {READ!r}.",
            [
                f"An administrator can allow it with {AGENT_ACCESS_KEY}: {READ_WRITE} "
                "in the build profile, then rebuild."
            ],
            details={"key": AGENT_ACCESS_KEY, "value": READ},
        )


def _check_drive_switch() -> None:
    """Refuse a drive the write verdict refuses for the deployment baseline.

    The bridge drives the baseline's Phoebus and :func:`baseline_refusal` has
    already refused a session that left it, so the baseline is the machine the
    drive writes to. Asked fresh on every call over the deployment's
    ``control_system`` section, with the owner left to the verdict's ladder,
    so a switch turned off reaches the next drive. A verdict that cannot be
    reached refuses as ``record_unavailable``.

    The message is the verdict's phrase, which says who can lift the refusal
    and names no config key; ``details`` carries the reason and the target.
    """
    from osprey_connectors.refusals import refusal_text

    try:
        config = load_osprey_config()
        section = config.get("control_system") if isinstance(config, dict) else None
        target = baseline_target(section)
        verdict = execution_verdict(section or {}, target)
    except Exception:  # unknown posture must not drive a panel
        logger.warning("Write verdict for phoebus_drive could not be reached", exc_info=True)
        verdict = posture_store.WriteVerdict(
            allowed=False, reason="record_unavailable", target=None, kind=None
        )
    if verdict.allowed:
        return
    reason = verdict.reason or "record_unavailable"
    make_error(
        "writes_disabled",
        refusal_text(
            reason,
            verdict.kind,
            detail=refusal_detail("phoebus_drive was refused before reaching the bridge.", verdict),
        ),
        ["Hand the action to the operator, who can perform it on the Phoebus panel."],
        details={"reason": reason, "target": verdict.target},
    )


def _check_drive_record() -> None:
    """Refuse a drive whose approval prompt left no record, where approval applies.

    :func:`~osprey.mcp_server.approval_record.require_approval_record` decides:
    a drive whose policy does not ask runs unchanged, and an approval switch
    that is not a bool refuses every drive.
    """
    from osprey.mcp_server.approval_record import require_approval_record

    config = load_osprey_config()
    refusal = require_approval_record("phoebus_drive", config if isinstance(config, dict) else {})
    if refusal is not None:
        make_error(
            refusal.error_type,
            refusal.message,
            ["Call phoebus_drive again so the approval prompt can ask."],
        )


def _check_explicit_display(display: str) -> None:
    """Reject the implicit ``"active"`` display when require-handle mode is on.

    ``"active"`` is the only value that resolves process-global focus — a
    handle or an explicit display name always addresses one specific display,
    so both are allowed regardless of this setting.
    """
    if display == "active" and _require_handle():
        make_error(
            "phoebus_handle_required",
            "Implicit 'active' display addressing is disabled on this backend; "
            "call phoebus_open_panel to obtain a handle and pass it as "
            "display='handle:<id>'.",
            [
                "Call phoebus_open_panel to obtain a handle, then pass "
                "display='handle:<id>' to target that display explicitly.",
                "Alternatively pass an explicit display name from phoebus_list_displays.",
            ],
        )


# ---------------------------------------------------------------------------
# Panel name → .bob resource resolution
# ---------------------------------------------------------------------------

# Built-in registry: an optional, framework-level extension point for panels
# shipped with OSPREY itself. Empty by default — the framework works fine this
# way, and phoebus.panels.<name> in config.yml is the real deployment path for
# every facility (see the module docstring). A framework built-in (if ever
# added) is resolved relative to the osprey repo root in a development checkout,
# or the config.yml directory otherwise; a facility panel must never be a
# hardcoded framework default.
_BUILTIN_PANELS: dict[str, str] = {}

# bridge_tools.py is at src/osprey/mcp_server/phoebus/tools/bridge_tools.py
# → parents[5] is the repo root (src/osprey/mcp_server/phoebus/tools → 5 up → repo root)
_OSPREY_REPO_ROOT: Path = Path(__file__).resolve().parents[5]

_OPEN_READY_TIMEOUT = 30  # seconds to wait for ready=True after /open


# A macro name: a plain identifier. Narrower than what Phoebus itself accepts
# (which also allows ".", "-", "[" and "]" for property paths). It is the form
# a panel declares and the form a caller passes.
_MACRO_NAME = re.compile(r"[A-Za-z][A-Za-z0-9_]*\Z")

_PANEL_ENTRY_KEYS = frozenset({"file", "macros"})


@dataclass(frozen=True)
class _PanelDeclaration:
    """One ``phoebus.panels`` entry: the display file and the macros it takes."""

    file: str
    macros: tuple[str, ...] = ()


def _panel_declaration(name: str, panels: object) -> _PanelDeclaration | None:
    """Read the ``phoebus.panels`` entry for *name*; ``None`` when there is none.

    The one reader of an entry's shape. Two forms::

        phoebus.panels.overview: overview.bob        # short: the file, no macros
        phoebus.panels.bpm.file: bpm.bob             # long: the file ...
        phoebus.panels.bpm.macros: [IOC]             # ... and the macro names it takes

    ``macros`` is a list of names today. A panel that lists none, in either
    form, takes none.

    Raises:
        ValueError: The entry is neither form, with what is wrong with it.
    """
    entry = panels.get(name) if isinstance(panels, dict) else None
    if entry is None:
        return None
    if isinstance(entry, str):
        return _PanelDeclaration(entry)
    if not isinstance(entry, dict):
        raise ValueError(
            "the entry must be a display-file path, or a mapping with 'file' and "
            f"optional 'macros' (got {type(entry).__name__})"
        )
    unknown = sorted(str(key) for key in entry if key not in _PANEL_ENTRY_KEYS)
    if unknown:
        raise ValueError(f"unknown key(s) {unknown}; an entry takes 'file' and 'macros'")
    file = entry.get("file")
    if not isinstance(file, str) or not file.strip():
        raise ValueError("'file' must be a display-file path")
    declared = entry.get("macros", [])
    if declared is None:
        declared = []
    if not isinstance(declared, list):
        raise ValueError(f"'macros' must be a list of macro names (got {type(declared).__name__})")
    for macro in declared:
        if not isinstance(macro, str) or not _MACRO_NAME.match(macro):
            raise ValueError(
                f"'macros' names {macro!r}: a macro name is letters, digits and "
                "underscores, starting with a letter"
            )
    return _PanelDeclaration(file, tuple(dict.fromkeys(declared)))


def _panels() -> dict:
    """The ``phoebus.panels`` map of this terminal's config, ``{}`` when unset or malformed."""
    panels = load_osprey_config().get("phoebus", {}).get("panels", {})
    return panels if isinstance(panels, dict) else {}


def _declared_macros(name: str, panels: dict) -> tuple[str, ...]:
    """The macro names panel *name* declares; none for a built-in or a malformed entry."""
    try:
        declaration = _panel_declaration(name, panels)
    except ValueError:
        return ()
    return declaration.macros if declaration is not None else ()


def _panel_resource(name: str, panels: dict) -> str | None:
    """Bridge-ready resource for *name*, or ``None`` when no source registers it.

    The resolution order of :func:`_resolve_panel_resource` without its
    refusal, so the lookup tool answers about exactly the resource an open
    would forward rather than about a second reading of the registry.

    Raises:
        ValueError: The ``phoebus.panels`` entry for *name* is malformed.
    """
    declaration = _panel_declaration(name, panels)
    resource: str | None = declaration.file if declaration is not None else None

    if resource is None:
        builtin_relative = _BUILTIN_PANELS.get(name)
        if builtin_relative is not None:
            # Prefer the development-checkout repo root; fall back to config dir.
            candidate = (_OSPREY_REPO_ROOT / builtin_relative).resolve()
            if candidate.exists():
                resource = str(candidate)
            else:
                config_path = resolve_config_path()
                if config_path.exists():
                    resource = str((config_path.parent / builtin_relative).resolve())
                else:
                    resource = builtin_relative  # pass through; bridge will report missing

    if resource is None:
        return None

    # Resolve relative non-URL config paths against the config file's directory.
    if not resource.startswith("file:") and not Path(resource).is_absolute():
        config_path = resolve_config_path()
        if config_path.exists():
            resource = str((config_path.parent / resource).resolve())

    return resource


def _display_key(value: str) -> str:
    """One comparison form for a display resource: an absolute real path.

    A registration and a caller's path name the same file in different
    spellings — a ``file:`` URL, a path relative to the config file, a
    symlinked display mount — so both sides are reduced to one form before
    they are compared. Only the comparison uses it: what an open forwards to
    the bridge stays as the deployment spelled it.
    """
    text = value
    if text.startswith("file:"):
        text = urllib.request.url2pathname(urllib.parse.urlparse(text).path)
    path = Path(text).expanduser()
    if not path.is_absolute():
        path = resolve_config_path().parent / path
    return str(path.resolve())


def _panel_index() -> dict[str, str]:
    """Comparison key → panel name for every panel this terminal registers.

    Built over the whole registry rather than one entry, so a single
    malformed registration (a value that is not a path) is skipped instead of
    being allowed to break every lookup. Names are walked in sorted order and
    the first one wins, so two names on one file answer with one of them
    deterministically.
    """
    panels = _panels()
    index: dict[str, str] = {}
    for name in sorted({str(name) for name in panels} | set(_BUILTIN_PANELS)):
        try:
            resource = _panel_resource(name, panels)
        except ValueError:
            continue
        if resource is None:
            continue
        index.setdefault(_display_key(resource), name)
    return index


def _panel_listing(name: str, panels: dict) -> str:
    """*name* as a refusal lists it: with the macros it declares, when it declares any."""
    declared = _declared_macros(name, panels)
    return f"{name} (macros: {', '.join(declared)})" if declared else name


def _check_macros_declared(name: str, macros: dict[str, str]) -> None:
    """Refuse macros panel *name* does not declare in its ``phoebus.panels`` entry.

    The facility says which parameters a screen takes; a panel that declares
    none takes none. Runs before the bridge is contacted.
    """
    if not macros:
        return
    declared = _declared_macros(name, _panels())
    undeclared = sorted(macro for macro in macros if macro not in declared)
    if not undeclared:
        return
    takes = f"declares {', '.join(declared)}" if declared else "declares no macros"
    make_error(
        "validation_error",
        f"Panel '{name}' does not take the macro(s) {', '.join(undeclared)}: its "
        f"phoebus.panels entry {takes}.",
        [
            f"Pass only the declared macros: {list(declared)}."
            if declared
            else f"Open '{name}' without macros.",
            f"An administrator can declare a macro with 'phoebus.panels.{name}.macros' "
            "in the build profile, then rebuild and redeploy.",
        ],
        details={"panel": name, "declared": list(declared), "undeclared": undeclared},
    )


def _resolve_panel_resource(name: str) -> str:
    """Map a logical panel name to a bridge-ready resource string.

    Resolution order:

    1. ``phoebus.panels.<name>`` in config.yml — value may be an absolute
       path, a ``file:`` URL, or a path relative to the config file's
       directory (resolved to absolute before forwarding).
    2. An optional framework built-in registry (empty by default), resolved
       relative to the osprey repository root when running from a development
       checkout, or relative to the config file's directory otherwise.

    Args:
        name: Logical panel name registered under ``phoebus.panels`` in
            config.yml (e.g. ``"site_overview"``).

    Raises:
        ToolError(unknown_panel): When *name* is not found in either source.
    """
    panels = _panels()
    try:
        resource = _panel_resource(name, panels)
    except ValueError as exc:
        make_error(
            "configuration_error",
            f"The phoebus.panels entry for '{name}' is malformed: {exc}.",
            [
                f"Write it as 'phoebus.panels.{name}: /path/to/file.bob', or as "
                f"'phoebus.panels.{name}.file: /path/to/file.bob' with "
                f"'phoebus.panels.{name}.macros: [NAME, ...]', in the build profile, "
                "then rebuild and redeploy."
            ],
            details={"key": f"phoebus.panels.{name}"},
        )

    if resource is None:
        known = [
            _panel_listing(known_name, panels)
            for known_name in sorted({str(key) for key in panels} | set(_BUILTIN_PANELS))
        ]
        make_error(
            "unknown_panel",
            f"Unknown panel name '{name}'.",
            [
                f"Known panels: {known}.",
                f"Register a custom panel by adding 'phoebus.panels.{name}: /path/to/file.bob' in the build profile (profile.yml on the host), then rebuild and redeploy.",
                "Paths may be absolute or relative to the config.yml directory.",
            ],
        )
    assert resource is not None  # make_error above raises; assertion satisfies type checker

    return resource


# A macro value the tool forwards: ASCII letters, digits, ".", "_" and "-",
# starting with a letter or digit, and never "..". An allow-list, so that the
# value an approver or a log reader sees is the value Phoebus gets: no nested
# macro reference, no path separator, no URI or query syntax, no look-alike or
# invisible character.
_MACRO_VALUE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*\Z")


def _validated_macros(macros: object) -> dict[str, str]:
    """Return *macros* as a name → string-value dict, or raise a validation error.

    ``None`` and an empty mapping both mean "no macros".

    The panel registry admits display *files*; a macro value is substituted
    into the channel addresses and embedded-display paths inside one. A name
    is a plain identifier and a value a plain token (``_MACRO_VALUE``), so a
    value cannot leave the display's directory or pull in another macro.
    Within that directory it still chooses: where the display builds a file
    name from a macro (``$(TYPE).bob``) the value selects any sibling file,
    and since the caller chooses the name too, it can override a site-wide
    Phoebus preference macro for this display. Which values a parameterised
    panel may be opened with is not decided here.
    """
    if macros is None:
        return {}
    if not isinstance(macros, dict):
        make_error(
            "validation_error",
            "macros must be a mapping of macro name to string value "
            f"(got {type(macros).__name__}).",
            ['Pass macros as an object, e.g. {"IOC": "injbpm"}.'],
        )
    for macro, value in macros.items():
        if not isinstance(macro, str) or not _MACRO_NAME.match(macro):
            make_error(
                "validation_error",
                f"Invalid macro name {macro!r}: a macro name is letters, digits and "
                "underscores, starting with a letter.",
            )
        if not isinstance(value, str):
            make_error(
                "validation_error",
                f"The value of macro {macro!r} must be a string (got {type(value).__name__}).",
                ["Macro values are text: pass a number as a quoted string."],
            )
        if not _MACRO_VALUE.match(value) or ".." in value:
            make_error(
                "validation_error",
                f"The value of macro {macro!r} is not a plain token: a macro value passed "
                "to phoebus_open_panel is ASCII letters, digits, '.', '_' and '-', "
                "starting with a letter or digit, without '..'.",
                [
                    "A value such as an IOC or host name passes; a path, a channel "
                    "prefix with a ':', or a reference to another macro does not.",
                    "If this display needs such a value, ask the operator to open it in Phoebus.",
                ],
            )
    return dict(macros)


# ---------------------------------------------------------------------------
# Tool 1: open panel
# ---------------------------------------------------------------------------
@mcp.tool()
async def phoebus_open_panel(
    name: str, focus: bool = True, macros: dict[str, str] | None = None
) -> str:
    """Open a Phoebus display by logical panel name and return its handle.

    Sends a ``POST /open`` request to the bridge, which loads the named
    ``.bob`` file and returns a deterministic display handle (e.g.
    ``"handle:d-3"``).  The tool polls ``GET /displays`` until the display
    reports ``ready=true`` or a 30-second timeout elapses, so that
    subsequent ``phoebus_perceive`` / ``phoebus_drive`` calls on the same
    handle do not race against model loading.

    Pass the returned ``handle`` string as the ``display`` argument of
    ``phoebus_perceive``, ``phoebus_perceive_region``, ``phoebus_snapshot``,
    and ``phoebus_drive`` to target this display unambiguously even when
    multiple displays are open.

    Panel name resolution (in order):

    1. ``phoebus.panels.<name>`` in config.yml — an absolute path, a
       ``file:`` URL, or a path relative to the config file's directory;
       or a mapping with that path as ``file`` and the macro names the
       panel takes as ``macros``.
    2. An optional framework built-in registry (empty by default).

    A parameterised display (one detail screen for many IOCs, hosts or
    devices) takes its parameters as ``macros``: ``{"IOC": "injbpm"}`` opens
    the display with ``$(IOC)`` resolved in its name, channel addresses and
    embedded displays. A panel takes only the macros its ``phoebus.panels``
    entry declares, and a panel that declares none takes none; a refusal
    names the declared macros, and ``phoebus_panel_lookup`` lists them. The
    same panel opened with different macro values is a different display,
    each with its own handle. Macros the display file
    defines itself keep their values. A bridge too old to apply macros is
    refused with ``not_supported``.

    After a successful open the tool asks the Web Terminal (best-effort,
    never fatal) to focus this instance's panel tab, so the user sees the
    display they just asked for without hunting through tabs.

    Args:
        name: Logical panel name registered under ``phoebus.panels`` in
            config.yml (e.g. ``"site_overview"``).
        focus: Switch the Web Terminal to this instance's panel tab after
            opening (default ``True``). Pass ``False`` for batch or
            background opens that should not steal the user's view.
        macros: Display macros as a mapping of macro name to string value
            (e.g. ``{"IOC": "injbpm"}``). Omit for a display that takes none.
            Every name must be one the panel declares.
            A name is letters, digits and underscores, starting with a
            letter. A value is a plain token: ASCII letters, digits, ``.``,
            ``_`` and ``-``, starting with a letter or digit, without ``..``.
            Anything else is refused before Phoebus is contacted.

    Returns:
        JSON ``{"status": "success", "handle": "handle:d-N", "id": "d-N",
        "resource": "<resolved-path>", "ready": <bool>, "focused": <bool>}``,
        plus ``"macros": {...}``, the macros the bridge reports it applied,
        when any were sent.
    """
    resource = _resolve_panel_resource(name)  # raises ToolError on unknown name
    sent_macros = _validated_macros(macros)  # raises ToolError on a malformed mapping
    _check_macros_declared(name, sent_macros)  # raises ToolError on an undeclared macro

    # "macros" is left out when there are none, so the body stays the one a
    # bridge that predates macro support was always sent.
    payload: dict = {"resource": resource}
    if sent_macros:
        payload["macros"] = sent_macros

    try:
        status, body = await anyio.to_thread.run_sync(_http_post_open, payload)
    except Exception as exc:
        make_error(
            "phoebus_unreachable",
            f"Could not reach the Phoebus bridge: {exc}",
            _UNREACHABLE_HINTS,
        )

    if status != 200:
        make_error(
            "phoebus_open_failed",
            _bridge_error_message(body, status),
            ["Check that the resource path exists and is readable by the Phoebus process."]
            + (
                ["Check the macro names: letters, digits and underscores, starting with a letter."]
                if sent_macros
                else []
            ),
        )

    display_id: str = body.get("id", "")
    if not display_id:
        make_error(
            "phoebus_open_failed",
            "Phoebus bridge returned a success response with no display id.",
            ["This may indicate a bridge version mismatch; check the bridge logs."],
        )

    handle = f"handle:{display_id}"
    ready: bool = bool(body.get("ready", False))

    # A bridge that applies macros echoes them; one that predates macro
    # support ignores the field and has opened the display with its macros
    # unresolved. That is not the display that was asked for, so it is not
    # reported as a success.
    if sent_macros and body.get("macros") != sent_macros:
        applied = body.get("macros")
        make_error(
            "not_supported",
            "The Phoebus bridge is too old for macros: it opened the display without applying them."
            if applied is None
            else "The Phoebus bridge did not apply the macros that were sent.",
            [
                "Update the Phoebus product to an agent bridge whose POST /open applies macros.",
                f"The display is open as {handle} with its macros unresolved; "
                "do not read or drive it as the requested display.",
            ],
            details={"handle": handle, "sent": sent_macros, "applied": applied},
        )

    # Poll until ready=True so downstream perceive/drive calls don't race the
    # JavaFX model-loading phase.  Both the sleep and the GET /displays call are
    # non-blocking so other tool calls and keepalives proceed normally.
    if not ready:
        deadline = _monotonic() + _OPEN_READY_TIMEOUT
        while _monotonic() < deadline:
            await asyncio.sleep(0.5)
            try:
                lst_status, displays = await anyio.to_thread.run_sync(_http_get_json, "/displays")
            except Exception:
                break  # bridge vanished; return what we have
            if lst_status == 200 and isinstance(displays, list):
                for d in displays:
                    if d.get("id") == display_id:
                        ready = bool(d.get("ready", False))
                        break
            if ready:
                break

    # Best-effort UX signal: switch the Web Terminal to this instance's panel
    # tab (panel id == server name; OSPREY_SERVER_NAME is set per-instance by
    # the registry, so a phoebus2 clone focuses the PHOEBUS2 tab). The display
    # is already open at this point — a missing/unreachable web terminal
    # (CLI-only mode) must never turn the open into a failure.
    focused = False
    if focus:
        panel_id = os.environ.get("OSPREY_SERVER_NAME", "phoebus")
        try:
            opened = await anyio.to_thread.run_sync(panel_op, "open", {"member": panel_id})
            focused = bool(opened["ok"])
        except Exception as exc:
            logger.debug("panel open failed (non-fatal): %s", exc)

    result: dict = {
        "status": "success",
        "handle": handle,
        "id": display_id,
        "resource": body.get("resource", resource),
        "ready": ready,
        "focused": focused,
    }
    if sent_macros:
        result["macros"] = sent_macros
    return json.dumps(result)


@mcp.tool()
async def phoebus_panel_lookup(path: str) -> str:
    """Answer whether this terminal may open the display file at *path*.

    A server that indexes the facility's display tree knows the file it
    found; whether that file is openable here is a fact about this terminal's
    own panel registry (``phoebus.panels`` in config.yml), which only the
    framework reads. Pass the path and this tool answers with the panel name
    ``phoebus_open_panel`` takes, or says the path is not registered here.

    A registered file matches however the path is spelled: through a
    symlinked display mount, with ``.`` or ``..`` segments, relative to the
    config file's directory, or as a ``file:`` URL. Registration is the whole
    answer — a registered path whose file is missing is still openable here,
    and the bridge reports the missing file at open time.

    Args:
        path: Filesystem path or ``file:`` URL of a display file, e.g.
            ``"/opt/displays/sr/overview.bob"``. A panel name is not a path.

    Returns:
        JSON ``{"status": "success", "openable": true, "name": "<panel>",
        "path": "<resolved path>"}`` when this terminal registers the file,
        with ``"macros": [...]`` naming the macros that panel takes when it
        declares any,
        and ``{"status": "success", "openable": false, "path": "<resolved
        path>"}`` when it does not.
    """
    if not path.strip():
        return make_error("validation_error", "path must not be empty.")

    key = _display_key(path)
    name = _panel_index().get(key)
    answer: dict = {"status": "success", "openable": name is not None, "path": key}
    if name is not None:
        answer["name"] = name
        declared = _declared_macros(name, _panels())
        if declared:
            answer["macros"] = list(declared)
    return json.dumps(answer)


# ---------------------------------------------------------------------------
# Tool 2: list displays
# ---------------------------------------------------------------------------
@mcp.tool()
async def phoebus_list_displays() -> str:
    """List every currently-open Phoebus display.

    Returns:
        JSON ``{"status": "success", "displays": [{"name", "ready", "active"}, ...]}``.
        An empty list means Phoebus is running but has no display open (or the
        bridge could reach no display).
    """
    try:
        status, body = _http_get_json("/displays")
    except Exception as exc:
        return make_error(
            "phoebus_unreachable", f"Could not reach the Phoebus bridge: {exc}", _UNREACHABLE_HINTS
        )
    if status != 200:
        return make_error("phoebus_error", _bridge_error_message(body, status))
    return json.dumps({"status": "success", "displays": body})


# ---------------------------------------------------------------------------
# Tool 3: perceive
# ---------------------------------------------------------------------------
@mcp.tool()
async def phoebus_perceive(display: str = "active") -> str:
    """Perceive a Phoebus display — walk its widget tree and report each widget.

    For every widget the result carries its type, name, nesting depth, on-screen
    bounds, visibility, and live PV state (name, value, severity, writability).
    Use this to understand what is on screen before snapshotting or driving.

    Args:
        display: Display reference — ``"active"`` (default, the focused display),
                 a display name as returned by ``phoebus_list_displays``, or a
                 handle string returned by ``phoebus_open_panel`` (e.g.
                 ``"handle:d-3"``).

    Returns:
        JSON ``{"status": "success", "display": <ref>, "perception": {...}}`` where
        ``perception`` is the bridge's stable schema (``display`` + ``widgets``).
        While the control target differs from the deployment baseline, one
        informational line naming both targets precedes that JSON (see the
        module docstring).
    """
    _check_explicit_display(display)  # raises phoebus_handle_required if enforced
    path = f"/perceive?display={urllib.parse.quote(display)}"
    try:
        status, body = _http_get_json(path)
    except Exception as exc:
        return make_error(
            "phoebus_unreachable", f"Could not reach the Phoebus bridge: {exc}", _UNREACHABLE_HINTS
        )
    if status != 200:
        return make_error(
            "phoebus_error",
            _bridge_error_message(body, status),
            ["Check the display reference; list displays with phoebus_list_displays."],
        )
    return prepend_line(
        baseline_pinned_line(PHOEBUS_SUBJECT),
        json.dumps({"status": "success", "display": display, "perception": body}),
    )


# ---------------------------------------------------------------------------
# Tool 4: perceive region
# ---------------------------------------------------------------------------
@mcp.tool()
async def phoebus_perceive_region(
    x: float, y: float, w: float, h: float, display: str = "active"
) -> str:
    """Perceive only the widgets whose on-screen bounds intersect a rectangle.

    Args:
        x: Left edge of the query rectangle, in screen pixels.
        y: Top edge of the query rectangle, in screen pixels.
        w: Rectangle width in screen pixels (must be > 0).
        h: Rectangle height in screen pixels (must be > 0).
        display: Display reference (``"active"``, a display name, or a handle
                 string from ``phoebus_open_panel`` e.g. ``"handle:d-3"``).

    Returns:
        JSON ``{"status": "success", "display": <ref>, "perception": {...}}`` whose
        widget list is filtered to the rectangle and sorted top-left first.
        While the control target differs from the deployment baseline, one
        informational line naming both targets precedes that JSON (see the
        module docstring).
    """
    if w <= 0 or h <= 0:
        return make_error("validation_error", f"w and h must be > 0 (got w={w}, h={h}).")
    _check_explicit_display(display)  # raises phoebus_handle_required if enforced
    path = f"/perceive/region?display={urllib.parse.quote(display)}&x={x}&y={y}&w={w}&h={h}"
    try:
        status, body = _http_get_json(path)
    except Exception as exc:
        return make_error(
            "phoebus_unreachable", f"Could not reach the Phoebus bridge: {exc}", _UNREACHABLE_HINTS
        )
    if status != 200:
        return make_error("phoebus_error", _bridge_error_message(body, status))
    return prepend_line(
        baseline_pinned_line(PHOEBUS_SUBJECT),
        json.dumps({"status": "success", "display": display, "perception": body}),
    )


# ---------------------------------------------------------------------------
# Tool 5: snapshot
# ---------------------------------------------------------------------------
@mcp.tool()
async def phoebus_snapshot(
    widget: str, display: str = "active", highlight: bool = False, dpi: float = 1.0
) -> str:
    """Capture a PNG snapshot of one widget and save it for viewing.

    The widget must have a live on-screen representation. The PNG is written to
    the screenshots directory and registered as an artifact; the response carries
    its file path plus the screen-space registration headers (origin + scale) so
    pixels can be mapped back to screen coordinates.

    Args:
        widget: Widget reference — a walk-order index (e.g. ``"0"``) or a widget
                name (e.g. ``"Setpoint"``).
        display: Display reference (``"active"``, a display name, or a handle
                 string from ``phoebus_open_panel`` e.g. ``"handle:d-3"``).
        highlight: When true, overlay an orange-red border on the target widget.
        dpi: Scale factor — 1.0 native, 2.0 HiDPI. Must be in (0, 8].

    Returns:
        JSON artifact response including the saved file path. Open that path with
        your file-reading tool to view the snapshot. While the control target differs from the
        deployment baseline, one informational line naming both targets precedes
        that JSON (see the module docstring).
    """
    if not (0 < dpi <= 8):
        return make_error("validation_error", f"dpi must be in (0, 8] (got {dpi}).")
    _check_explicit_display(display)  # raises phoebus_handle_required if enforced

    path = (
        f"/snapshot?display={urllib.parse.quote(display)}"
        f"&widget={urllib.parse.quote(widget)}"
        f"&highlight={'true' if highlight else 'false'}&dpi={dpi}"
    )
    try:
        status, headers, content = _http_get_bytes(path)
    except Exception as exc:
        return make_error(
            "phoebus_unreachable", f"Could not reach the Phoebus bridge: {exc}", _UNREACHABLE_HINTS
        )
    if status != 200:
        msg = f"Phoebus bridge returned HTTP {status}."
        with contextlib.suppress(Exception):
            msg = json.loads(content).get("error", msg)
        return make_error(
            "phoebus_error",
            msg,
            [
                "Confirm the widget is on screen and rendered (headless widgets "
                "cannot be snapshotted)."
            ],
        )

    safe = re.sub(r"[^A-Za-z0-9_.-]", "_", widget)
    ts = datetime.now(UTC).strftime("%Y%m%d_%H%M%S_%f")
    fpath = _snapshot_dir() / f"phoebus_{safe}_{ts}.png"
    fpath.write_bytes(content)

    # Resolved once so both success paths below — artifact response and the
    # bare-path fallback — carry the identical label.
    label = baseline_pinned_line(PHOEBUS_SUBJECT)

    try:
        from osprey.stores.artifact_store import get_artifact_store

        store = get_artifact_store()
        entry = store.save_data(
            tool="phoebus_snapshot",
            data={
                "display": display,
                "widget": widget,
                "filepath": str(fpath),
                "bytes": len(content),
                "origin_x": headers.get("X-Bridge-Origin-X"),
                "origin_y": headers.get("X-Bridge-Origin-Y"),
                "scale": headers.get("X-Bridge-Scale"),
            },
            title=f"Phoebus snapshot: {widget}",
            description=f"PNG snapshot of widget '{widget}' on display '{display}'",
            summary={
                "widget": widget,
                "file_size_kb": round(len(content) / 1024, 1),
                "filepath": str(fpath),
            },
            access_details={
                "file_format": "PNG",
                "view_hint": f"Open {fpath} with your file-reading tool to view the snapshot.",
            },
        )
        return prepend_line(label, json.dumps(entry.to_tool_response(), default=str))
    except ToolError:
        raise
    except Exception:
        # ArtifactStore unavailable (e.g. minimal context) — still return the path.
        logger.exception("phoebus_snapshot: artifact save failed; returning raw path")
        return prepend_line(
            label,
            json.dumps(
                {
                    "status": "success",
                    "filepath": str(fpath),
                    "bytes": len(content),
                    "view_hint": f"Open {fpath} with your file-reading tool to view the snapshot.",
                }
            ),
        )


# ---------------------------------------------------------------------------
# Tool 6: drive
# ---------------------------------------------------------------------------
@mcp.tool()
async def phoebus_drive(
    widget: str,
    verb: str,
    display: str = "active",
    value: str | None = None,
    mode: str = "synthetic",
) -> str:
    """Drive a Phoebus widget — click an action control or type a value.

    Two drive modes:
      * ``synthetic`` (default) — inject the real GUI event the widget listens
        for, so confirm dialogs, enable/disable rules, and bound scripts all run,
        exactly as if an operator clicked.
      * ``semantic`` — write the widget's primary PV / trigger its action directly
        through the runtime, bypassing the GUI chain. ``semantic`` supports only
        the ``type`` verb (``click`` returns a bypass result).

    Args:
        widget: Widget reference — a walk-order index or a widget name.
        verb: ``"click"`` (fire an action control) or ``"type"`` (commit a value
              into a text control).
        display: Display reference (``"active"``, a display name, or a handle
                 string from ``phoebus_open_panel`` e.g. ``"handle:d-3"``).
        value: Value to commit for the ``"type"`` verb; ignored for ``"click"``.
        mode: ``"synthetic"`` or ``"semantic"``.

    Returns:
        JSON ``{"status": "success", "fired": <bool>, "detail": <str>}``. ``fired``
        is false when no interactive control was resolved or the mode bypassed it.

    Refuses outright while the session's control-system target differs from the
    deployment baseline: the bridge drives the baseline's Phoebus, so the drive
    would land on a target this session has left (see the module docstring).
    Refused unless ``phoebus.agent_access`` is ``read_write``. Refused
    ``writes_disabled`` while the write verdict refuses the baseline (its
    switch is off, its tier allows nothing, the record cannot be read), with
    ``details.reason`` naming why. Where the approval policy asks, refused
    ``approval_missing`` when the approval prompt left no record of this call,
    and ``approval_config_invalid`` under an approval switch that is not a bool.
    """
    # Every check runs before argument validation, in this order: access, the
    # session target, the write verdict, the approval record. Access is a fact
    # about the deployment, the target and the switch facts about the machine,
    # the record a fact about this call; each holds for every argument, so an
    # operator learns them rather than first being sent to fix a typo'd verb,
    # and a deployment that offers no drive says so before anything else.
    _check_drive_offered()
    refusal = baseline_refusal(PHOEBUS_SUBJECT, "Driving a Phoebus widget")
    if refusal is not None:
        return make_error(BASELINE_REFUSAL_ERROR_TYPE, refusal[0], refusal[1])
    _check_drive_switch()
    _check_drive_record()

    verb_l = verb.lower()
    mode_l = mode.lower()
    if verb_l not in ("click", "type"):
        return make_error(
            "validation_error", f"Invalid verb '{verb}'.", ["verb must be 'click' or 'type'."]
        )
    if mode_l not in ("synthetic", "semantic"):
        return make_error(
            "validation_error",
            f"Invalid mode '{mode}'.",
            ["mode must be 'synthetic' or 'semantic'."],
        )
    if verb_l == "type" and value is None:
        return make_error("validation_error", "verb 'type' requires a 'value'.")
    _check_explicit_display(display)  # raises phoebus_handle_required if enforced

    payload = {
        "display": display,
        "widget": widget,
        "verb": verb_l,
        "value": value,
        "mode": mode_l,
    }
    try:
        status, body = _http_post_drive(payload)
    except Exception as exc:
        return make_error(
            "phoebus_unreachable", f"Could not reach the Phoebus bridge: {exc}", _UNREACHABLE_HINTS
        )
    if status != 200:
        return make_error(
            "phoebus_rejected",
            _bridge_error_message(body, status),
            ["Check the widget reference and that verb/mode are valid."],
        )

    # Agent-activity highlight, emitted only when the drive actually reached a
    # control. A synthetic 200 with fired=false means the bridge resolved no
    # interactive control, so nothing was written — that stays silent. Semantic
    # ``type`` is the exception: it writes the widget's PV through the runtime
    # and so reports fired=false even though the value landed. Every refusal
    # (validation, handle enforcement, unreachable bridge, non-200) returns
    # above, so those emit nothing. notify_agent_activity_async never raises; the
    # blocking call runs off the event loop.
    fired = bool(body.get("fired"))
    if fired or (verb_l == "type" and mode_l == "semantic"):
        await notify_agent_activity_async(
            "phoebus_drive", "channel", detail=f"{verb_l} {widget} on {display}"
        )

    return json.dumps(
        {
            "status": "success",
            "fired": body.get("fired"),
            "detail": body.get("detail"),
        }
    )
