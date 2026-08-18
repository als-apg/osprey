"""Authoring-time validator for a session-tier bluesky plan-file BODY.

An agent authoring a new plan (the ``write_plan``/``validate_plan`` MCP
tools) never gets its file exec'd directly —
that would hand arbitrary code execution to whatever produced the body. This
module is the gate a body must pass before anything downstream (the session
directory layer's LOAD gate; the enqueue gate) will treat it as real: :func:`validate_plan` runs three ordered stages, each of
which can reject outright before the next ever runs:

1. **Static AST allowlist** (:func:`_static_allowlist_check`) — a submodule-
   aware import walk (bare top-level matching, as the shared viz sandbox
   uses, is too coarse: ``bluesky.plan_stubs`` must be allowed while
   ``bluesky.utils`` and bare ``bluesky`` must not) plus the shared syntax
   gate and dangerous-substring scan reused from
   :func:`osprey.mcp_server.workspace.execution.sandbox_executor.validate_sandbox_code`,
   plus a positional check (:func:`_check_future_import_position`) that a
   ``from __future__ import ...`` statement can only ever be a body's own
   leading statement — never true once the generated ``PLAN_METADATA``
   assignment is prepended ahead of it.
2. **Narrowed CA/connector pattern scan** (:func:`_ca_pattern_scan`) — reuses
   :func:`osprey.services.python_executor.analysis.pattern_detection.detect_control_system_operations`
   with an override pattern set naming only actual CA/connector constructs
   (``caput``, ``epics.``, ``write_channel``, ``_osprey_connector``, ...) so
   an otherwise-benign body calling ``numpy.put``/``dict.get``/``queue.put``
   is not falsely rejected by the framework's default (bare ``.put(``/
   ``.get(``) patterns.
3. **Mock-RunEngine dry-run** (:func:`_dry_run`) — actually builds and drives
   the plan's generator, in a subprocess whose ``EPICS_CA_*`` variables are
   neutralized to explicit inert values (no address, no auto-discovery — see
   :data:`_EPICS_CA_INERT_ENV`), against in-process mock devices
   (:mod:`osprey.services.bluesky_bridge.devices.mock`) built for exactly the
   channels the plan's ``PARAMS`` declared movable or readable. This is an
   **authoring-quality gate** ("does the body actually run"), not a
   containment boundary — containment comes from stages 1-2 above and the
   downstream load and enqueue gates that key off this module's validation
   record, not from anything the dry-run subprocess itself prevents.
   The same stage also watches what the plan's run(s) declared about their
   own size, and rejects a plan that says it moves channels but leaves an
   operator with no way to know how far along it is — see
   :func:`_point_count_reasons`. That check is enforced HERE, on the session
   tier, and nowhere else: a shipped/preset/facility plan is reviewed by the
   people who install it, while a session plan is written on the spot and
   this validator is the only thing that reads it before it runs.

Every :class:`ValidationResult` (pass or fail) carries a ``content_hash``
computed by :func:`hash_plan_body`, the single normalization the record store
and both gates reuse, so a validation record recorded against one hash of a
body's content is found again by a later re-hash of the same content, byte for
byte.
"""

from __future__ import annotations

import ast
import asyncio
import hashlib
import json
import logging
import os
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from osprey.mcp_server.sandbox_env import scrub_sensitive_env
from osprey.mcp_server.workspace.execution.sandbox_executor import validate_sandbox_code
from osprey.services.python_executor.analysis.pattern_detection import (
    detect_control_system_operations,
)

logger = logging.getLogger("osprey.services.bluesky_bridge.plan_validation")

# ---------------------------------------------------------------------------
# Stage 1: submodule-aware import allowlist
# ---------------------------------------------------------------------------
# `bluesky` is only ever allowed via these three exact, fully-dotted
# submodules — a top-level `name.split(".")[0] in {"bluesky"}` match (as the
# viz sandbox's `validate_sandbox_code` uses) would also let a plan body
# import `bluesky.utils`, or bare `bluesky` itself, neither of which is part
# of the authoring surface an agent-written plan body needs.
_ALLOWED_BLUESKY_SUBMODULES: frozenset[str] = frozenset(
    {"bluesky.plan_stubs", "bluesky.plans", "bluesky.preprocessors"}
)

# `osprey` is narrowed the same way, and for a narrower reason: a plan file may
# declare a `render(window, params)` returning a `Figure` (see `plan_types.py`),
# and there is no way to build one without importing the module that defines it.
# `plans_core/*.py` files are exec'd by `plan_loader` via
# `spec_from_file_location` with no package, so a relative `from ..figure
# import` would fail at load — the absolute path below is the only spelling
# that works, which is what puts it in front of this allowlist at all.
#
# Every entry is inert by construction: `figure` is pydantic-only (models,
# `decimate`, the row adapter), `orm_analysis`/`bump_analysis` are numpy-only,
# and `plan_fields` is pydantic-only (the role-typed channel field helpers a
# plan author uses to declare `movable`/`readable` params) — none performs
# I/O, spawns anything, or touches a control system. Nothing else under
# `osprey` belongs here. In particular the connector, config, queue, and
# executor packages are the capabilities this validator exists to keep out of
# an agent-authored plan body, and admitting bare `osprey` would hand a plan
# every one of them.
_ALLOWED_OSPREY_SUBMODULES: frozenset[str] = frozenset(
    {
        "osprey.services.bluesky_bridge.figure",
        "osprey.services.bluesky_bridge.orm_analysis",
        "osprey.services.bluesky_bridge.bump_analysis",
        "osprey.services.bluesky_bridge.plan_fields",
    }
)

# Everything else a plan body may import, checked top-level only (a plan body
# doing numerical/stdlib bookkeeping around its bluesky calls has no reason to
# reach past these). `pydantic` is a deliberate addition beyond the plain
# numerical/stdlib set: the plan-file contract (`plan_loader.py`) requires
# `PARAMS` to be a `pydantic.BaseModel` subclass whenever a plan declares one
# at all, so a session-authored plan with any typed parameters cannot satisfy
# that contract without importing it. `__future__` and `typing` are zero-
# runtime-capability (no I/O, no import, no exec — `__future__` only toggles
# compiler behavior, `typing` is erased at runtime) and are exactly what the
# shipped exemplars use for `from __future__ import annotations` / type
# hints. `logging` is allowed too (the exemplars log via
# `logging.getLogger(__name__)`), but narrowed further below — see
# `_DENIED_LOGGING_SUBMODULES`.
_ALLOWED_TOP_LEVEL_MODULES: frozenset[str] = frozenset(
    {
        "numpy",
        "scipy",
        "math",
        "statistics",
        "time",
        "collections",
        "itertools",
        "functools",
        "pydantic",
        "__future__",
        "typing",
        "logging",
    }
)

# `logging.config.dictConfig`/`fileConfig` and `logging.handlers` (which
# includes `SMTPHandler`, `SocketHandler`, etc.) do class-instantiation- and
# callable-resolution-by-STRING — an import-by-string bypass of this very
# AST allowlist. Bare `import logging` and `logging.getLogger(...)` attribute
# access (all the shipped exemplars need) require neither submodule, so they
# are denied while plain `logging` stays allowed — the inverse of how
# `bluesky` is narrowed: there, the top level is denied and specific
# submodules are allowed; here, the top level is allowed and these two
# specific submodules are denied.
_DENIED_LOGGING_SUBMODULES: frozenset[str] = frozenset({"logging.config", "logging.handlers"})

# Passed to `validate_sandbox_code`'s own (coarser, top-level-only) import
# check so it never disagrees with `_check_import_allowlist` below: `bluesky`
# and `osprey` are allowed at the top level here, then narrowed to the exact
# submodules above by the finer-grained walk. Every other name in this set is
# identical to `_ALLOWED_TOP_LEVEL_MODULES`, so the two checks agree everywhere
# except those two narrowings, which `validate_sandbox_code` cannot itself
# express. Widening THIS set alone would admit any `osprey` submodule at all —
# it is only ever safe paired with the submodule gate in `_is_allowed_import`.
_VALIDATOR_TOP_LEVEL_MODULES: frozenset[str] = _ALLOWED_TOP_LEVEL_MODULES | {"bluesky", "osprey"}


def _is_allowed_import(dotted_name: str) -> bool:
    """Whether ``dotted_name`` (an ``import``'s own dotted module path) is allowed."""
    top = dotted_name.split(".")[0]
    if top == "bluesky":
        return dotted_name in _ALLOWED_BLUESKY_SUBMODULES
    if top == "osprey":
        return dotted_name in _ALLOWED_OSPREY_SUBMODULES
    if top == "logging":
        return dotted_name not in _DENIED_LOGGING_SUBMODULES
    return top in _ALLOWED_TOP_LEVEL_MODULES


def _check_import_allowlist(code: str) -> list[str]:
    """The authoritative, submodule-aware import walk for a plan body.

    Assumes ``code`` already parses — call after `validate_sandbox_code`'s
    syntax gate has run. A `SyntaxError` here (should the caller ever skip
    that gate) is swallowed by the caller, not this function.

    ``from bluesky import plan_stubs`` and ``from bluesky.plan_stubs import
    mv`` both name the same allowed submodule (similarly ``from logging
    import config`` and ``from logging.config import dictConfig`` both name
    the same denied one), but an `ast.ImportFrom`'s ``module`` only holds the
    parent package name for the first form of each pair — the submodule
    lives in the *imported name*, not the module path. All four forms are
    resolved to the same fully-dotted name before checking the allowlist.
    """
    violations: list[str] = []
    tree = ast.parse(code)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if not _is_allowed_import(alias.name):
                    violations.append(f"Import not allowed for plan authoring: '{alias.name}'")
        elif isinstance(node, ast.ImportFrom):
            if node.module is None:
                violations.append(
                    "Relative import with no module name is not allowed for plan authoring"
                )
            elif node.module == "bluesky":
                for alias in node.names:
                    dotted = f"bluesky.{alias.name}"
                    if not _is_allowed_import(dotted):
                        violations.append(f"Import not allowed for plan authoring: 'from {dotted}'")
            elif node.module == "logging":
                for alias in node.names:
                    dotted = f"logging.{alias.name}"
                    if dotted in _DENIED_LOGGING_SUBMODULES:
                        violations.append(f"Import not allowed for plan authoring: 'from {dotted}'")
            elif not _is_allowed_import(node.module):
                violations.append(f"Import not allowed for plan authoring: 'from {node.module}'")
    return violations


_FUTURE_IMPORT_REJECT_MESSAGE = (
    "session plans cannot use `from __future__` imports because plan "
    "metadata is prepended to the file; omit it — modern type hints "
    "(list[str], dict[str, Any]) work without it on Python 3.9+."
)


def _check_future_import_position(code: str) -> list[str]:
    """Reject a `from __future__ import ...` that is not the file's own leading statement.

    Python requires a `from __future__ import ...` statement to be the
    file's literal first statement (a module docstring may precede it, and
    further future-import statements may follow one — nothing else may).
    Only `compile()`/the import machinery enforces this; `ast.parse` (the
    syntax gate `validate_sandbox_code` runs above) happily parses a
    misplaced one, since it is a semantic rule, not a grammar rule.

    A session-authored body is ALWAYS preceded by a generated
    ``PLAN_METADATA = {...}`` assignment once `write_session_plan` (app.py)
    writes it to disk, so a future-import statement anywhere in that body
    can never end up at a legal position in the assembled file. Left
    uncaught here, it instead surfaces deep in stage 3's dry-run subprocess
    as a `SyntaxError` naming a temp file and line number that point at the
    generated metadata line, not the real cause — this check catches it
    here instead, with a message that explains why and what to do about it.

    This is a positional check, not a blanket ban: a body where the
    future-import genuinely IS the leading statement (docstring aside) is
    unaffected — that covers the shipped `plans_core` exemplars, which this
    same function validates directly (never metadata-prepended) in
    `TestShippedExemplarsPassValidation`.
    """
    tree = ast.parse(code)
    body = tree.body
    future_indices = [
        i
        for i, node in enumerate(body)
        if isinstance(node, ast.ImportFrom) and node.module == "__future__"
    ]
    if not future_indices:
        return []

    def _is_leading_docstring(index: int, node: ast.stmt) -> bool:
        return (
            index == 0
            and isinstance(node, ast.Expr)
            and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, str)
        )

    for index in range(future_indices[-1]):
        node = body[index]
        if _is_leading_docstring(index, node):
            continue
        if isinstance(node, ast.ImportFrom) and node.module == "__future__":
            continue
        return [_FUTURE_IMPORT_REJECT_MESSAGE]
    return []


def _static_allowlist_check(code: str) -> list[str]:
    """Stage 1: syntax gate + submodule-aware import allowlist + dangerous-substring scan.

    Reuses `validate_sandbox_code` for the syntax gate and the dangerous-
    pattern substring scan (its own defaults — this never touches
    `_ALLOWED_TOP_LEVEL`/`_ALLOWED_IMPORTS`/`_DANGEROUS_PATTERNS`); the
    submodule-aware walk above is the authoritative import gate.
    `_check_future_import_position` additionally guards a positional rule
    neither of those checks (nor the syntax gate's own `ast.parse`) enforces
    — see its docstring.
    """
    is_safe, violations = validate_sandbox_code(
        code, allowed_top_level=_VALIDATOR_TOP_LEVEL_MODULES
    )
    if not is_safe:
        # A syntax error short-circuits here — `_check_import_allowlist`
        # would only re-raise the same `SyntaxError` on its own `ast.parse`.
        return violations
    return violations + _check_future_import_position(code) + _check_import_allowlist(code)


# ---------------------------------------------------------------------------
# Stage 2: narrowed CA/connector-only pattern scan
# ---------------------------------------------------------------------------
# Deliberately excludes the framework-default bare `.put(`/`.get(`/
# `.set_value(` regexes (see pattern_detection.py's
# `get_framework_standard_patterns`) — those false-positive on
# `numpy.put(...)`/`dict.get(...)`/`queue.put(...)`, which a benign plan body
# is expected to use freely. Only actual CA/connector constructs are named.
_CA_ONLY_PATTERNS: dict[str, list[str]] = {
    "write": [
        r"\bcaput\s*\(",
        r"epics\.",
        r"\baioca\b",
        r"\bcaproto\b",
        r"\bwrite_channel\s*\(",
        r"_osprey_connector\b",
        r"PV\s*\(",
    ],
    "read": [
        r"\bcaget\s*\(",
        r"\bread_channel\s*\(",
    ],
}


def _ca_pattern_scan(code: str) -> list[str]:
    """Stage 2: reject a body that reaches for CA/connector constructs directly."""
    result = detect_control_system_operations(
        code, patterns=_CA_ONLY_PATTERNS, pattern_mode="override"
    )
    matched = result["detected_patterns"]["writes"] + result["detected_patterns"]["reads"]
    return [f"Control-system operation pattern matched: {pattern}" for pattern in matched]


# ---------------------------------------------------------------------------
# Content hash — the one helper the record store and both gates reuse to
# re-key/re-check a validation record against a plan file's current content.
# ---------------------------------------------------------------------------
def hash_plan_body(body: str) -> str:
    """SHA-256 content hash of a plan-file BODY, over a normalized encoding.

    Normalization: a leading UTF-8 BOM is stripped, CRLF/CR line endings are
    folded to LF, all trailing newlines are collapsed, and the body is left
    with exactly one trailing newline before hashing — so a file that
    round-trips through an editor that rewrites line endings, saves with a
    BOM, or adds/drops trailing blank lines still hashes identically. This is
    the one place that normalization happens; every caller that needs to
    check "does this file content have a passing validation record"
    (the validation-record store, the load gate, the enqueue gate) must hash
    through this function rather than re-deriving its own encoding.
    """
    normalized = body.replace("\r\n", "\n").replace("\r", "\n")
    normalized = normalized.lstrip("﻿").rstrip("\n") + "\n"
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


@dataclass
class ValidationResult:
    """Outcome of validating an authored plan-file BODY.

    ``passed`` is `False` if any stage rejects the body; ``reasons`` lists
    every rejection reported by whichever stage stopped it (later stages
    never run once an earlier one fails — see `validate_plan`).
    ``content_hash`` is always populated, computed before any stage runs, so a
    caller can key a validation record (or a rejection) against the exact
    body content regardless of which stage decided the outcome.
    """

    passed: bool
    reasons: list[str]
    content_hash: str


# ---------------------------------------------------------------------------
# Stage 3: mock-RunEngine dry-run, in a subprocess with EPICS_CA_* neutralized
# ---------------------------------------------------------------------------
# Set (not merely deleted) in the dry-run subprocess's environment, on top of
# the shared `scrub_sensitive_env` deny-list. Deleting these keys outright
# would be actively WORSE than leaving them alone: a CA client that sees
# neither `EPICS_CA_ADDR_LIST` nor an explicit `EPICS_CA_AUTO_ADDR_LIST`
# defaults auto-discovery to YES, which makes it BROADCAST on the local
# subnet looking for IOCs — exactly the unsolicited network traffic this
# scrub exists to prevent, not "no CA address to reach." Setting explicit
# inert values (an empty address list, auto-discovery off, no name server)
# closes that gap: even if some future bluesky/ophyd-async escape hatch let a
# plan body reach real CA machinery despite stages 1-2 rejecting the
# constructs to do so, there is nowhere for it to send a request.
_EPICS_CA_INERT_ENV: dict[str, str] = {
    "EPICS_CA_ADDR_LIST": "",
    "EPICS_CA_AUTO_ADDR_LIST": "NO",
    "EPICS_CA_NAME_SERVERS": "",
}
# Dropped outright rather than set to a value: with no address to reach and
# auto-discovery disabled (see above), nothing ever consults a configured
# server port.
_EPICS_CA_ENV_NAMES_TO_DROP: tuple[str, ...] = ("EPICS_CA_SERVER_PORT",)

# Extra wall-clock time (on top of the caller's `dry_run_timeout`) the parent
# waits for the subprocess to exit after that timeout — long enough for the
# subprocess's own internal deadline (see `_render_dry_run_script`) to fire
# and write a graceful "did not complete" result before the parent falls back
# to a hard `proc.kill()`, which reports only a generic timeout message.
_DRY_RUN_GRACE_SECONDS = 5.0


# Mock devices the dry-run builds when a plan's ``PARAMS`` declares no
# channel role at all — a `PARAMS` with no fields (nothing to declare) still
# has to drive a RunEngine, and a plan that *should* have declared a channel
# gets the dressed "available mock devices" failure naming exactly these two
# rather than an unexplained empty mapping. Injected into the script through
# the same render seam as every other computed value.
_FALLBACK_MOVABLE_MOCKS: tuple[str, ...] = ("sp1",)
_FALLBACK_READABLE_MOCKS: tuple[str, ...] = ("rb1",)

# The start-document key a run's declared point count arrives under, and this
# module's only spelling of it — injected into the dry-run script through the
# render seam so the capture and this file agree by construction. Reading it
# is what a gate does: the same read the progress reporter performs when a run
# starts (`document_plane._on_start`). Authors never write this key by hand —
# `plan_fields.scan_metadata()` is the one place a plan's stamp is built, and
# that is the function the failure message below points them at.
_DECLARED_POINT_COUNT_KEY = "num_points"

# The one remedy sentence both point-count rejections end with. Capability
# vocabulary only: what the plan does (moves channels, declares how many
# points) and the authoring call that says it.
_DECLARE_POINTS_REMEDY = (
    "a plan that moves channels must declare its point count — pass "
    "md=scan_metadata(movable=..., readable=..., points=...) to your run decorator"
)


def _render_dry_run_script(
    *,
    plan_path: Path,
    result_path: Path,
    plan_name: str,
    sample_args: dict[str, Any],
    fallback_movable_names: tuple[str, ...],
    fallback_readable_names: tuple[str, ...],
    point_count_key: str,
    inner_timeout: float,
) -> str:
    """Render the subprocess script that drives the dry-run to completion.

    Loads the plan body as a standalone module, validates ``sample_args``
    against its ``PARAMS`` schema, and drives the resulting generator through a
    bluesky ``RunEngine`` against mock devices built for the channels those
    validated params supply — a `MockSettable` for every channel the plan declared
    movable, a `MockReadable` for every one it declared readable, and nothing
    at all for a string sitting under a field that declared no role (that is a
    plain parameter, not a device). Writes a JSON result to ``result_path`` in a
    ``finally`` so the parent always has something to read even if construction
    itself raised.

    On a successful drive the result also carries ``declared_points``: one entry
    per run the plan opened, holding what that run declared about its own size
    (`None` for a run that declared nothing usable). The script only *reports*
    it — whether a missing declaration is a rejection depends on what the plan's
    own metadata says it does, which the parent reads (`_point_count_reasons`).

    Which mock a channel gets is read from the plan's own declaration via
    `plan_fields.collect_channels`, the single authority every consumer shares.
    That collection runs here, inside the subprocess, because it needs the
    body's `PARAMS` class and its *validated* params — and loading a plan body
    is exactly what the bridge process never does (see the module docstring).
    `plan_fields` is pydantic-only and already on the plan-body import
    allowlist, so a validated body could import it itself; the dry-run
    subprocess reaches for nothing a plan body cannot.

    The RunEngine is driven here rather than through any shared runner class:
    the bridge process runs no plans at all (execution belongs to the
    queueserver worker), so this subprocess is the one place in the codebase
    that still starts a RunEngine, and it owns that wiring directly instead of
    keeping a general-purpose runner alive for a single caller.
    """
    return f"""\
import json
import sys
import threading
import traceback
from pathlib import Path

_PLAN_PATH = Path(r"{plan_path}")
_RESULT_PATH = Path(r"{result_path}")
_PLAN_NAME = {plan_name!r}
_SAMPLE_ARGS = {sample_args!r}
_FALLBACK_MOVABLE = {fallback_movable_names!r}
_FALLBACK_READABLE = {fallback_readable_names!r}
_POINT_COUNT_KEY = {point_count_key!r}
_DEADLINE_S = {inner_timeout!r}

result = {{"success": False, "error": None}}
try:
    import importlib.util

    _MODULE_NAME = "_osprey_plan_dry_run_body"
    spec = importlib.util.spec_from_file_location(_MODULE_NAME, _PLAN_PATH)
    module = importlib.util.module_from_spec(spec)
    # Registered in sys.modules BEFORE exec (mirrors plan_loader.py's own
    # module loader): a plan body with two pydantic models referencing each
    # other by name (e.g. `grid_scan`'s `PARAMS.axes: list[GridAxis]`)
    # relies on `from __future__ import annotations` postponed-evaluation
    # forward refs resolving against `sys.modules[cls.__module__].__dict__`
    # -- skip this and pydantic raises "class not fully defined" the moment
    # the model is instantiated below, even though the class body itself
    # executed without error.
    sys.modules[_MODULE_NAME] = module
    spec.loader.exec_module(module)

    import asyncio

    from bluesky import RunEngine
    from pydantic import BaseModel as _BaseModel

    from osprey.services.bluesky_bridge.devices.mock import build_devices
    from osprey.services.bluesky_bridge.plan_fields import (
        MOVABLE_ROLE,
        READABLE_ROLE,
        collect_channels,
    )

    params_cls = getattr(module, "PARAMS", None)
    if params_cls is None:
        class params_cls(_BaseModel):
            pass

    params = params_cls.model_validate(_SAMPLE_ARGS)

    # Which channel gets which mock comes from the plan's own declaration, not
    # from what a field is named: a channel the plan declared movable gets the
    # settable mock, one it declared readable gets the readable mock, and a
    # string under a field that declared no role is a plain parameter that gets
    # no mock at all. A plan declaring neither role falls back to one of each so
    # a parameterless body still has something to drive.
    _movable_names = list(collect_channels(params_cls, params, MOVABLE_ROLE)) or list(
        _FALLBACK_MOVABLE
    )
    _readable_names = list(collect_channels(params_cls, params, READABLE_ROLE)) or list(
        _FALLBACK_READABLE
    )

    # `context_managers=[]` disables the RunEngine's default SIGINT handling,
    # which only works on the main thread -- required because `RE(...)` is
    # driven from the worker thread below, not this one.
    RE = RunEngine(context_managers=[])

    # `build_devices` is `async def`, and the devices it builds bind to
    # whichever loop connects them -- which must be `RE.loop`, the loop bluesky
    # drives all signal I/O on, not a throwaway one.
    devices = dict(asyncio.run_coroutine_threadsafe(
        build_devices(settable_names=_movable_names, readable_names=_readable_names), RE.loop
    ).result(timeout=30.0))

    try:
        plan_gen = module.build_plan(devices, params)
    except KeyError as exc:
        _missing = exc.args[0] if exc.args else "<unknown>"
        raise KeyError(
            f"plan {{_PLAN_NAME!r}} referenced device {{_missing!r}}, which the dry run "
            f"did not mock; available mock devices: {{sorted(devices)}}. A channel is "
            f"mocked only when PARAMS declares it movable or readable."
        ) from exc

    # What each run this plan opens declares about its own size, captured from
    # the run's own start document -- the same place, and the same read, the
    # progress reporter uses when a real run starts. A run that declares
    # nothing usable records None so the parent can tell "no run at all" from
    # "a run that said nothing".
    _declared_points = []

    def _capture_declared_points(name, doc):
        _value = doc.get(_POINT_COUNT_KEY)
        _usable = isinstance(_value, int) and not isinstance(_value, bool) and _value >= 1
        _declared_points.append(_value if _usable else None)

    RE.subscribe(_capture_declared_points, "start")

    failure = {{}}

    def _drive():
        try:
            RE(plan_gen)
        except Exception as exc:
            failure["exc"] = exc

    thread = threading.Thread(target=_drive, name="plan-dry-run", daemon=True)
    thread.start()
    thread.join(_DEADLINE_S)
    if thread.is_alive():
        raise TimeoutError("dry-run plan did not complete within the timeout")
    if "exc" in failure:
        raise failure["exc"]

    result["declared_points"] = _declared_points
    result["success"] = True
except Exception as exc:
    result["error"] = f"{{type(exc).__name__}}: {{exc}}"
    result["traceback"] = traceback.format_exc()
finally:
    _RESULT_PATH.write_text(json.dumps(result))
"""


def _declares_writes(body: str) -> bool:
    """Whether the body's own ``PLAN_METADATA`` says the plan moves channels.

    Read statically off the assignment's dict literal, never by importing the
    body — the bridge process never loads a plan body (see the module
    docstring), and this runs on the same untrusted text stages 1-2 parse.
    A body with no ``PLAN_METADATA``, a non-literal one, or one whose
    ``writes`` value is anything but the literal `True` reads as "does not
    move channels": the point-count gate then has nothing to enforce, and a
    plan that genuinely moves channels while declaring otherwise is the LOAD
    gate's rejection (`plan_loader`), not this one's.
    """
    try:
        tree = ast.parse(body)
    except SyntaxError:
        return False
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if not any(
            isinstance(target, ast.Name) and target.id == "PLAN_METADATA" for target in node.targets
        ):
            continue
        if not isinstance(node.value, ast.Dict):
            continue
        for key, value in zip(node.value.keys, node.value.values, strict=True):
            if (
                isinstance(key, ast.Constant)
                and key.value == "writes"
                and isinstance(value, ast.Constant)
            ):
                return value.value is True
    return False


def _point_count_reasons(payload: dict[str, Any], *, plan_name: str) -> list[str]:
    """Reject a moving plan whose run leaves an operator no way to track it.

    A plan that moves channels is one an operator watches: the panel's
    progress, the "how much longer" question, and every consumer downstream of
    the run read the point count the run itself declared. A plan that declares
    none reports as an unbounded run forever, so this is a rejection at
    authoring time rather than a surprise at launch.

    Two ways to fail, one remedy: the plan opened no run at all (it said it
    moves channels, so there is a run to declare), or it opened one that
    declared nothing usable. Passing does NOT require calling
    `scan_metadata()` — a plan that hands its work to a stock bluesky plan
    inherits that plan's own declaration and passes on it — but writing the
    stamp yourself is the only route for a plan that builds its own run, which
    is why the remedy names the authoring helper.
    """
    declared = payload.get("declared_points") or []
    if not declared:
        return [
            f"Dry-run gate: plan {plan_name!r} declares that it moves channels but "
            f"opened no run; {_DECLARE_POINTS_REMEDY}."
        ]
    if any(count is None for count in declared):
        return [
            f"Dry-run gate: plan {plan_name!r} opened a run that declares no point "
            f"count; {_DECLARE_POINTS_REMEDY}."
        ]
    return []


async def _dry_run(
    body: str,
    *,
    plan_name: str,
    sample_args: dict[str, Any],
    timeout: float,
) -> list[str]:
    """Stage 3: run the plan body to completion under a mock-device RunEngine.

    The mock devices are the ones the plan's own role declaration asks for —
    see `_render_dry_run_script`, which is where that collection happens and
    why it happens there rather than here.

    Runs in a subprocess (its own interpreter, own event loop) with the
    shared `scrub_sensitive_env` deny-list applied AND every
    `_EPICS_CA_INERT_ENV` variable set to an inert value (never merely
    deleted — see that constant's docstring for why deleting would be worse)
    on top of it. Authoring-QUALITY gate only — "does it actually run" — not
    a containment boundary (see module docstring).

    A body that runs clean is then held to what its own ``PLAN_METADATA``
    claims: a plan declaring that it moves channels must leave a declared
    point count behind on the run(s) it opened (`_point_count_reasons`). A
    read-only plan is not asked for one.
    """
    with tempfile.TemporaryDirectory(prefix="osprey_plan_dry_run_") as tmp:
        tmp_path = Path(tmp)
        plan_path = tmp_path / "plan_body.py"
        plan_path.write_text(body, encoding="utf-8")
        result_path = tmp_path / "result.json"
        script_path = tmp_path / "dry_run_wrapper.py"
        script_path.write_text(
            _render_dry_run_script(
                plan_path=plan_path,
                result_path=result_path,
                plan_name=plan_name,
                sample_args=sample_args,
                fallback_movable_names=_FALLBACK_MOVABLE_MOCKS,
                fallback_readable_names=_FALLBACK_READABLE_MOCKS,
                point_count_key=_DECLARED_POINT_COUNT_KEY,
                inner_timeout=timeout,
            ),
            encoding="utf-8",
        )

        env = scrub_sensitive_env(os.environ.copy())
        env.update(_EPICS_CA_INERT_ENV)
        for name in _EPICS_CA_ENV_NAMES_TO_DROP:
            env.pop(name, None)

        proc = await asyncio.create_subprocess_exec(
            sys.executable,
            str(script_path),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
        )
        try:
            _stdout, stderr = await asyncio.wait_for(
                proc.communicate(), timeout=timeout + _DRY_RUN_GRACE_SECONDS
            )
        except TimeoutError:
            proc.kill()
            await proc.wait()
            return [f"Dry-run subprocess timed out after {timeout} seconds"]

        if not result_path.exists():
            stderr_text = stderr.decode("utf-8", errors="replace").strip()
            return [
                f"Dry-run subprocess produced no result (exit code {proc.returncode}): {stderr_text}"
            ]

        try:
            payload = json.loads(result_path.read_text(encoding="utf-8"))
        except Exception as exc:
            return [f"Dry-run result file unreadable: {exc}"]

        if not payload.get("success"):
            return [f"Dry-run failed: {payload.get('error', 'unknown error')}"]
        if _declares_writes(body):
            return _point_count_reasons(payload, plan_name=plan_name)
        return []


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
async def validate_plan(
    body: str,
    *,
    plan_name: str = "session_plan",
    sample_args: dict[str, Any] | None = None,
    dry_run_timeout: float = 30.0,
) -> ValidationResult:
    """Validate an authored plan-file BODY through all three ordered stages.

    Each stage short-circuits the next on failure: a static-allowlist
    violation is reported without ever running the CA pattern scan, and a CA
    pattern match is reported without ever spawning the dry-run subprocess.

    Args:
        body: The plan file's full source text (``PLAN_METADATA`` + ``PARAMS``
            + ``build_plan``, per the layered directory catalog's file
            contract — see ``plans_core/orm.py``).
        plan_name: Name to register the body's plan under for the stage-3
            dry-run only; unrelated to any name the body's own
            ``PLAN_METADATA`` declares.
        sample_args: Sample ``PARAMS`` field values used to build the stage-3
            dry-run's generator and mock devices. `None` (no sample args)
            still runs the dry-run against an empty-args `PARAMS()` and a
            single default mock setpoint/readback — appropriate only for a body
            whose `PARAMS` has no required fields.
        dry_run_timeout: Seconds the stage-3 subprocess is given to drive the
            plan to completion.

    Returns:
        A `ValidationResult` with `content_hash` always populated (hashed
        before any stage runs), and `passed`/`reasons` reflecting whichever
        stage accepted or rejected the body.
    """
    content_hash = hash_plan_body(body)
    resolved_sample_args = dict(sample_args) if sample_args is not None else {}

    reasons = _static_allowlist_check(body)
    if reasons:
        return ValidationResult(passed=False, reasons=reasons, content_hash=content_hash)

    reasons = _ca_pattern_scan(body)
    if reasons:
        return ValidationResult(passed=False, reasons=reasons, content_hash=content_hash)

    reasons = await _dry_run(
        body,
        plan_name=plan_name,
        sample_args=resolved_sample_args,
        timeout=dry_run_timeout,
    )
    if reasons:
        return ValidationResult(passed=False, reasons=reasons, content_hash=content_hash)

    return ValidationResult(passed=True, reasons=[], content_hash=content_hash)
