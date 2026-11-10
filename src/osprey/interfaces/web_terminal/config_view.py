"""The web terminal's one reader of its rendered ``config.yml``.

Every question the terminal asks its own render goes through here: the
privilege gates, the terminal gate (``web.terminal``), the watchdogs block,
the login's write tier the chip shows, and the baseline the control-context
owner claims at. All of them see one parse —
``${VAR}`` expanded, a non-mapping refused — and one failure answer:
:data:`UNREADABLE`, which arms nothing.

A leaf module on purpose. ``routes.websocket`` imports the control-context
owner, and ``routes`` reaches back into ``app``; a reader the owner, the routes
and the app all share has to import none of them.
"""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from osprey_connectors import posture_store
from osprey_connectors.config import ConfigBuilder, load_project_dotenv
from osprey_connectors.log_filter import quiet_logger

logger = logging.getLogger(__name__)


#: Sentinel telling "the render could not be read" apart from "the render has
#: no ``control_system:`` block". Both answer writes-off, but only the first is
#: a failure worth logging, and the connector helpers have their own opinion
#: about a ``None`` section that the failure case must not borrow.
UNREADABLE = object()

#: ``locks.tier`` when the login's tier cannot be read — the verdict refuses it
#: as ``owner_unknown``, which no chip gesture lifts either.
TIER_UNKNOWN = "unknown"


#: The last render this module parsed: ``(path, signature, config)``. One entry
#: is the whole cache, because one server serves one render — a second path
#: simply replaces it rather than growing a map of stale files.
_CONFIG_MEMO: tuple[Path, tuple[int, int, int], Any] | None = None


def reset_rendered_config_memo() -> None:
    """Forget the parsed render, so the next read parses the file afresh.

    The memo is process-wide: without a reset, a render parsed for one app is
    still answered to the next app in the same process whose file carries the
    same path and stat signature.
    """
    global _CONFIG_MEMO
    _CONFIG_MEMO = None


def _file_signature(path: Path) -> tuple[int, int, int] | None:
    """``(st_mtime_ns, st_size, st_ino)`` of *path*, or ``None`` when it is gone.

    One ``stat`` — the cheapest question that distinguishes "the writer has
    republished" from "nothing has moved", and the one that answers "the file
    disappeared" at the same time. The inode is in the tuple because the render
    this module memoizes is replaced through a rename, and a replacement can
    land with the same size and, on a coarse clock, the same mtime.
    """
    try:
        stat = path.stat()
    except OSError:
        return None
    return (stat.st_mtime_ns, stat.st_size, stat.st_ino)


def rendered_config(config_path: Path | None) -> Any:
    """The WHOLE parsed render, or :data:`UNREADABLE`.

    The whole file rather than the ``control_system:`` block alone because the
    target labels are derived from more than that block — a deployment running a
    stand-in for its live machine says so under ``services:`` — and one caller
    then gets to read the file exactly once for every question it asks.
    Blocking; never call it from the event loop.

    **Parsed the way the deploy parses it**: :class:`ConfigBuilder`, so
    ``${VAR}`` expands and a document that is not a mapping is refused. With
    ``load_env=False``: the server loaded its ``.env`` chain once at start-up,
    and a re-parse must not reload it into a running process.

    **Memoized by the file's signature.** ``config.yml`` is a build artefact: it
    changes when a deployment is re-rendered and not otherwise, while the chip
    polls the roster per open card and every row it draws asks this render three
    or four questions. The key is ``(st_mtime_ns, st_size, st_ino)``.

    The memo is invisible to callers: what comes back is a deep copy, so a
    caller may keep or mutate the render without the next request inheriting it.
    An unreadable render is never memoized — the file may be mid-write — so the
    next call looks again rather than serving a failure until the mtime moves.
    """
    global _CONFIG_MEMO
    try:
        if not config_path:
            return UNREADABLE
        path = Path(config_path)
        signature = _file_signature(path)
        if signature is None:
            return UNREADABLE
        memo = _CONFIG_MEMO
        if memo is not None and memo[0] == path and memo[1] == signature:
            return copy.deepcopy(memo[2])
        with quiet_logger(["registry", "CONFIG"]):
            config = ConfigBuilder(path, load_env=False).raw_config
    except Exception:  # an unreadable config is not a writes-on render
        logger.warning("Could not read the rendered config at %s", config_path)
        return UNREADABLE
    _CONFIG_MEMO = (path, signature, config)
    return copy.deepcopy(config)


def section_of(config: Any) -> Any:
    """The ``control_system:`` block of an ALREADY-READ render.

    A render that could not be read stays :data:`UNREADABLE`; one that
    parsed to something that is not a mapping has no section, which is a
    different (and non-failing) answer the predicates downstream already handle.
    """
    if config is UNREADABLE:
        return UNREADABLE
    return config.get("control_system") if isinstance(config, dict) else None


def login_tier(section: Any, login: str) -> str:
    """The login's write ceiling here: ``never``, ``asked``, ``allowed`` or ``unknown``.

    The two tier terms of the write verdict first, in its order: the
    deployment's tier (:func:`~osprey_connectors.types.tier_writes`), then the
    login's own rung (:func:`~osprey_connectors.posture_store.owner_tier_term`).
    Otherwise the verdict's own ceiling
    (:func:`~osprey_connectors.posture_store.write_ceiling`). A render that
    could not be read is ``unknown``: no tier was read, which is not the answer
    of a render saying ``never``. Never raises.
    """
    if section is UNREADABLE:
        return TIER_UNKNOWN
    try:
        from osprey_connectors.types import tier_writes

        if tier_writes(section) == "never":
            return "never"
        rung = posture_store.owner_tier_term(section, login)
        ceiling = posture_store.write_ceiling(section, login)
    except Exception:  # an unreadable tier is not an allowed one
        logger.warning("Could not read the write tier for login %s", login)
        return TIER_UNKNOWN
    if rung == "never":
        return "never"
    if rung == "unknown":
        return TIER_UNKNOWN
    return ceiling


#: Words a human writes when they mean a boolean but YAML kept a string —
#: ``enabled: "false"`` (quoted) is the one that matters, because for a
#: default-ON switch a bare ``bool("false")`` is ``True``: the deployment
#: would read as having asked for the switch OFF and get it left ON.
_FALSE_WORDS = frozenset({"false", "no", "off", "0"})
_TRUE_WORDS = frozenset({"true", "yes", "on", "1"})


def coerce_config_flag(key: str, value: object, default: bool) -> bool:
    """Return a configured boolean switch, falling back to *default*.

    An **absent** key (``None``) means "use the default" — the shipped
    behaviour of a deployment that never mentions the switch.

    A real YAML boolean is honoured as written. A quoted ``"false"`` is honoured
    too: that spelling is a human writing a boolean, and reading it as truthy
    would leave a switch ON that the config says is OFF.

    Anything else (a number, a mapping from a mis-indented file, an unrecognized
    word) is reported and discarded — a value nobody can interpret must not
    silently become one of the two postures.

    Args:
        key: Dotted config key, used only for the warning message.
        value: Whatever the config reader returned.
        default: The shipped default for *key*.

    Returns:
        The configured boolean, or *default*.
    """
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        token = value.strip().lower()
        if token in _TRUE_WORDS:
            return True
        if token in _FALSE_WORDS:
            return False
    logger.warning("%s is %r, not a boolean; using %r instead", key, value, default)
    return default


def _flag_in(config: Any, key: str, default: bool) -> bool:
    """The flag at dotted *key* of an already-read render, walked as ``ConfigBuilder.get``."""
    value = config
    try:
        for part in key.split("."):
            value = value[part]
    except (KeyError, TypeError):
        value = None
    return coerce_config_flag(key, value, default)


@dataclass(frozen=True)
class PrivilegeGates:
    """The two privilege gates a surface resolves out of its config file.

    Attributes:
        config_panel_enabled: Whether the Config panel's server surface
            (``/api/config`` and ``/api/claude-setup``) is live.
        scaffold_write_enabled: Whether the scaffold gallery's write and delete
            routes under ``/api/scaffold`` are live.
        config_unreadable_path: The resolved config file that exists but
            could not be read, or ``None``.
    """

    config_panel_enabled: bool
    scaffold_write_enabled: bool
    config_unreadable_path: Path | None = None


#: The refusal every terminal-surface route gives when ``web.terminal`` is
#: false. One sentence, so the websocket's error frame and the log say the same.
TERMINAL_REFUSAL = "The terminal is not offered on this deployment (web.terminal: false)."


def unreadable_config_refusal(path: str | Path) -> str:
    """The refusal every surface gives when an existing config file could not be read.

    It names the file and the fix. The privilege gates are resolved once at
    startup, so restarting the server after the file is fixed is what reopens
    them.

    Args:
        path: The config file that exists but could not be read.

    Returns:
        The refusal sentence, naming the file and the fix.
    """
    return f"{path} could not be read; fix it and restart the server"


class TerminalConfigView:
    """The terminal's render, as the questions its gates and its owner ask of it.

    ========================== =========== =========== ============
    case                       ``flag``    ``tier``    ``unreadable``
    ========================== =========== =========== ============
    no file resolved           default     ``unknown`` ``None``
    file unreadable or gone    ``False``   ``unknown`` the path
    key absent or uninterp.    default
    lifespan never ran         ``False``   ``unknown`` ``None``
    ========================== =========== =========== ============

    No file resolved means the shipped defaults: a gate answered out of a file
    the surface was not pointed at is worse than one at its default. A resolved
    file that cannot be read — unparseable, gone, not a mapping — closes every
    gate and gives the tier ``unknown``, because an unreadable config is not
    permission. The view of an app whose lifespan never ran
    (:func:`terminal_config`) has made no decision, and closes everything.

    Every method reads through :func:`rendered_config`, so each sees a
    re-render of the same path on its next call. Blocking; call it off the loop.
    """

    def __init__(self, path: Path | None, *, closed: bool = False) -> None:
        self.path: Path | None = Path(path) if path is not None else None
        self._closed = closed

    def render(self) -> Any:
        """The whole parsed render, or :data:`UNREADABLE` (also with no file)."""
        if self._closed:
            return UNREADABLE
        return rendered_config(self.path)

    @property
    def unreadable(self) -> Path | None:
        """The resolved file when it cannot be read, else ``None``."""
        if self._closed or self.path is None:
            return None
        return self.path if self.render() is UNREADABLE else None

    def flag(self, key: str, default: bool) -> bool:
        """The boolean at dotted *key*: *default* when absent, ``False`` when unreadable."""
        if self._closed:
            return False
        if self.path is None:
            return default
        config = self.render()
        if config is UNREADABLE:
            return False
        return _flag_in(config, key, default)

    def tier(self, login: str) -> str:
        """*login*'s write ceiling here (:func:`login_tier`): never|asked|allowed|unknown."""
        return login_tier(section_of(self.render()), login)

    def gates(self) -> PrivilegeGates:
        """Both privilege gates, and the unreadable file, out of ONE read of the render."""
        if self._closed:
            return PrivilegeGates(False, False)
        if self.path is None:
            return PrivilegeGates(True, True)
        config = self.render()
        if config is UNREADABLE:
            logger.error(
                "Could not read the config file %s; the Config panel and scaffold gallery "
                "writes are closed until it reads cleanly",
                self.path,
            )
            return PrivilegeGates(False, False, self.path)
        return PrivilegeGates(
            _flag_in(config, "web.config_panel.enabled", True),
            _flag_in(config, "web.scaffold_gallery.write_enabled", True),
        )


_CLOSED_VIEW = TerminalConfigView(None, closed=True)


def terminal_config(app: Any) -> TerminalConfigView:
    """The view the lifespan put on ``app.state``, or the closed view when it put none."""
    view = getattr(getattr(app, "state", None), "terminal_config", None)
    return view if isinstance(view, TerminalConfigView) else _CLOSED_VIEW


def resolve_privilege_gates(config_path: str | Path | None) -> PrivilegeGates:
    """Read both privilege gates out of the config file a surface resolved.

    For a surface with no terminal lifespan of its own (ARIEL): loads the
    project's ``.env`` chain first, as the terminal's lifespan does, so a
    ``${VAR}`` reference expands the same way on both. The answers are
    :meth:`TerminalConfigView.gates`'s; a surface refusing because the file
    could not be read words it with :func:`unreadable_config_refusal`.

    Args:
        config_path: The config file this surface resolved, or ``None``.

    Returns:
        The two gates, and the unreadable file when there is one.
    """
    if config_path is None:
        return PrivilegeGates(True, True)
    load_project_dotenv()
    return TerminalConfigView(Path(config_path)).gates()
