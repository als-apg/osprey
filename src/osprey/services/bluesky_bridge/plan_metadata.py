"""Authoring metadata for plans: the ``PlanMetadata`` model and its parser.

A plan module (built-in, preset, or facility-supplied) declares a module-level
``PLAN_METADATA`` dict describing itself to operators and to the agent's
discovery surface (`GET /plans`). This module defines the required shape of
that dict and a fail-closed parser: a malformed, incomplete, or over-declared
block is a typed `PlanMetadataError` naming the offending field(s) or unknown
key(s), never a silently default-filled or silently trimmed object. Provenance (trust tier) is deliberately *not* part of
this model -- it is assigned by the loader based on which layer a file came
from, not self-declared by the plan author.

Kept pydantic-only (no bluesky/ophyd/tiled imports) so it can be imported from
``plan_types.py`` without breaking that module's bluesky-free boundary.
"""

from __future__ import annotations

import ast
import re
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, ValidationError, field_validator


class PlanMetadataError(ValueError):
    """Raised when a plan module's ``PLAN_METADATA`` is missing, malformed, or invalid.

    A single exception type for every failure mode (absent attribute, wrong
    container type, missing/mistyped field, unknown key) so callers only need
    to catch one thing; the message names the offending field(s) or key(s) and
    the module/file the metadata came from.
    """


class PlanMetadata(BaseModel):
    """Authoring-declared metadata for one plan, surfaced via `GET /plans`.

    Three required fields -- ``name``, ``description``, ``writes`` -- and
    optional presentation fields that shape the operator's form. A plan
    lacking a required field is an authoring error to be rejected at load
    time, not a gap to paper over with a default; an optional field a plan
    does not state is ``None``. Unknown keys are rejected too
    (``extra="forbid"``) -- a plan that declares anything outside this
    contract, including keys from an earlier shape, is quarantined loudly
    rather than parsed clean with the surplus dropped. Which channels a plan
    touches is *not* declared here: it is read off the role-typed fields of
    the plan's parameter model. JSON-serializable via `model_dump()` for
    direct inclusion in `PlanSpec.to_dict()`.
    """

    model_config = ConfigDict(extra="forbid")

    name: str
    description: str
    writes: bool
    """Whether the plan moves anything, as declared by its author.

    True means the plan changes machine state through its movable fields;
    false means it only reads. The load gate cross-checks this declaration
    against the plan's role-typed fields, so a mismatch is caught at load
    time rather than at execution.
    """

    layout: list[list[str]] | None = None
    """Rows of parameter names: the fields of one row share a row of the form.

    Parameters the layout does not name follow its rows in schema order. Every
    name appears once, in a non-empty row; whether each name is a parameter of
    the plan is checked by the loader, which has the plan's ``PARAMS``.
    """

    @field_validator("layout")
    @classmethod
    def _layout_names_each_field_once(cls, rows: list[list[str]] | None) -> list[list[str]] | None:
        if rows is None:
            return None
        seen: set[str] = set()
        for row in rows:
            if not row:
                raise ValueError("a layout row is empty")
            for name in row:
                if not name:
                    raise ValueError("a layout name is empty")
                if name in seen:
                    raise ValueError(f"layout names {name!r} more than once")
                seen.add(name)
        return rows

    summary: list[str] | None = None
    """The one-line readout above the form's footer, as a list of parts.

    Each part is text with ``{parameter}`` placeholders; ``{{`` and ``}}`` are
    literal braces. The panel joins the parts it shows with " · ", shows a
    part only once every parameter it names has a value, and fills a list
    parameter in as its count. A placeholder is a bare parameter name: no
    conversion, format spec, index or attribute.
    """

    @field_validator("summary")
    @classmethod
    def _summary_parts_name_bare_parameters(cls, parts: list[str] | None) -> list[str] | None:
        if parts is None:
            return None
        for part in parts:
            _placeholders(part)
        return parts


def parse_plan_metadata_dict(raw: dict[str, Any], *, source: str) -> PlanMetadata:
    """Validate ``raw`` (a plan module's ``PLAN_METADATA`` dict) into a `PlanMetadata`.

    Wraps pydantic's `ValidationError` into `PlanMetadataError`, naming the
    failing field(s) and ``source`` (a path/module label for operator
    debugging), so every caller sees one exception type regardless of whether
    a field is missing, wrong-typed, or outside the contract. Unknown keys are
    reported in their own clause, so a plan carrying a key this model no longer
    accepts says exactly which key to delete.
    """
    try:
        return PlanMetadata.model_validate(raw)
    except ValidationError as exc:
        invalid: list[str] = []
        unknown: list[str] = []
        for error in exc.errors():
            field = ".".join(str(part) for part in error["loc"])
            if error["type"] == "extra_forbidden":
                unknown.append(field)
            elif error["type"] == "value_error":
                reason = str(error["ctx"]["error"]) if "ctx" in error else error["msg"]
                invalid.append(f"{field} ({reason})")
            else:
                invalid.append(field)
        message = f"{source}: PLAN_METADATA is invalid"
        if invalid:
            message += f" for field(s): {', '.join(invalid)}"
        if unknown:
            message += f"; unknown key(s): {', '.join(unknown)}"
        raise PlanMetadataError(message) from exc


_SUMMARY_TOKEN = re.compile(r"\{\{|\}\}|\{([^{}]*)\}")
"""One summary token: a doubled brace, or a placeholder and the text inside it.

The same grammar as the panel's ``SUMMARY_TOKEN`` in ``plan-presentation.js``,
so a part the loader accepts is a part the panel fills.
"""


def _placeholders(part: str) -> list[str]:
    """The parameter names one summary part's placeholders use, in order.

    Raises:
        ValueError: the part is empty, a brace is left unpaired, or a
            placeholder is anything but a bare parameter name.
    """
    if not part:
        raise ValueError("a summary part is empty")
    names: list[str] = []
    for match in _SUMMARY_TOKEN.finditer(part):
        name = match.group(1)
        if name is None:
            continue
        if not name.isidentifier():
            raise ValueError(
                f"summary part {part!r} has a placeholder that is not a bare parameter name"
            )
        names.append(name)
    rest = _SUMMARY_TOKEN.sub("", part)
    if "{" in rest or "}" in rest:
        raise ValueError(f"summary part {part!r} has an unpaired brace")
    return names


def summary_parameters(meta: PlanMetadata) -> set[str]:
    """Every parameter name the summary's placeholders use; empty without a summary."""
    return {name for part in meta.summary or () for name in _placeholders(part)}


def check_presentation_names(
    meta: PlanMetadata, field_names: Iterable[str], *, source: str
) -> None:
    """Refuse ``meta`` when its presentation keys name a parameter the plan lacks.

    ``field_names`` are the plan's ``PARAMS`` fields. The check lives apart
    from the model because only a caller that has imported the plan knows
    them; the source-only reader validates the block without them.

    Raises:
        PlanMetadataError: naming the key and every unknown name, sorted.
    """
    known = set(field_names)
    named = {
        "layout": {name for row in meta.layout or () for name in row},
        "summary": summary_parameters(meta),
    }
    for key, names in named.items():
        unknown = sorted(names - known)
        if unknown:
            raise PlanMetadataError(
                f"{source}: PLAN_METADATA {key} names unknown parameter(s): {', '.join(unknown)}"
            )


def parse_plan_metadata(module: Any, *, source: str | None = None) -> PlanMetadata:
    """Read and validate ``PLAN_METADATA`` from an already-imported plan ``module``.

    Raises `PlanMetadataError` before ever reaching pydantic validation if the
    module has no ``PLAN_METADATA`` attribute or that attribute isn't a dict.
    ``source`` defaults to the module's ``__name__`` for the error message;
    pass it explicitly when the module was imported under a synthetic name
    (e.g. a path-hash `sys.modules` key) that wouldn't mean anything to an
    operator.
    """
    label = source if source is not None else getattr(module, "__name__", repr(module))
    raw = getattr(module, "PLAN_METADATA", None)
    if raw is None:
        raise PlanMetadataError(f"{label}: module has no PLAN_METADATA attribute")
    if not isinstance(raw, dict):
        raise PlanMetadataError(f"{label}: PLAN_METADATA must be a dict, got {type(raw).__name__}")
    return parse_plan_metadata_dict(raw, source=label)


def extract_plan_metadata_from_source(path: Path) -> PlanMetadata:
    """Read and validate ``PLAN_METADATA`` from a plan file *without importing it*.

    Parses ``path`` with `ast` and literal-evaluates the module-level
    ``PLAN_METADATA`` dict, so a plan's metadata can be read by callers that
    must not run plan code -- build-time tooling, and the source-lookup scan
    that has to identify a file before deciding whether to serve it. The
    module under inspection is never imported, exec'd, or otherwise run.

    Only a literal dict counts: ``PLAN_METADATA`` computed at import time
    cannot be read this way and is rejected rather than guessed at. Implicit
    string concatenation is fine -- the parser folds adjacent literals into one
    constant -- which is how every shipped plan writes its ``description``.

    Args:
        path: Plan file to read.

    Returns:
        The validated `PlanMetadata` declared by the file.

    Raises:
        PlanMetadataError: The file is unreadable, unparseable, declares no
            module-level literal ``PLAN_METADATA`` dict, or declares one that
            fails the `PlanMetadata` contract. One error type for every failure
            mode, matching `parse_plan_metadata`.
    """
    label = str(path)
    try:
        # `UnicodeDecodeError` is a `ValueError` subclass, so a file that is not
        # valid UTF-8 is covered by the tuple below without its own entry.
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (OSError, SyntaxError, ValueError) as exc:
        raise PlanMetadataError(f"{label}: could not read plan source: {exc}") from exc

    node = _find_plan_metadata_dict(tree)
    if node is None:
        raise PlanMetadataError(
            f"{label}: file declares no module-level literal PLAN_METADATA dict"
        )

    try:
        # `TypeError` as well as `ValueError`: an unhashable literal key
        # (``{[1]: 2}``) raises the former, and letting it escape would break
        # the one-error-type contract every caller of this function relies on.
        raw = ast.literal_eval(node)
    except (ValueError, TypeError) as exc:
        raise PlanMetadataError(
            f"{label}: PLAN_METADATA is not a literal dict, so it cannot be read "
            f"without running the module: {exc}"
        ) from exc

    return parse_plan_metadata_dict(raw, source=label)


def extract_plan_name_from_source(path: Path) -> str | None:
    """Read *only* the declared plan name from a plan file, without importing it.

    The identity half of `extract_plan_metadata_from_source`, for callers whose
    question is "which file declares this name?" rather than "is this file's
    whole metadata block valid?" -- above all the source-lookup scan behind
    `GET /plans/{name}/source`. It is deliberately as permissive as an import:
    a file whose ``PLAN_METADATA`` carries a non-literal value under some
    *other* key (``"description": _DESC``, say) still imports, still passes the
    loader's validation of the imported object, and is therefore a registered,
    launchable plan -- so its source has to stay reachable. Only the ``name``
    key's own value is read, and only when it is a knowable constant (see
    `_constant_name`); every other value is ignored, literal or not.

    Unknowable is not the same as absent, and this reader returns `None` for
    both rather than guessing: a ``**spread`` entry after the last literal
    ``name`` key can rebind the name to anything, so a name read before it is
    no longer the name an import binds, and reporting it would serve a file
    under a name it does not have.

    Args:
        path: Plan file to read.

    Returns:
        The declared plan name, or `None` when the file is unreadable,
        unparseable, declares no module-level literal ``PLAN_METADATA`` dict,
        or declares one whose ``name`` this reader cannot know. Never raises: a
        file this reader cannot identify is a skip, not an error, for every
        caller.
    """
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (OSError, SyntaxError, ValueError):
        return None

    node = _find_plan_metadata_dict(tree)
    if node is None:
        return None

    found: str | None = None
    for key, value in zip(node.keys, node.values, strict=True):
        if key is None:
            # A `**spread` entry has `None` for its key. Its contents are not
            # statically knowable and a later entry wins in a dict display, so
            # it can silently replace a `name` read above it -- which would
            # mean serving this file under a name it does not have. Drop back
            # to "unknowable"; a literal `name` after the spread re-establishes
            # the name, exactly as it would at import.
            found = None
            continue
        if not (isinstance(key, ast.Constant) and key.value == "name"):
            continue
        # Assigned unconditionally, so a later `name` entry whose value cannot
        # be read (a duplicate key bound to a module constant, say) makes the
        # name unknowable rather than leaving the superseded earlier one.
        found = _constant_name(value)
    return found


def _constant_name(value: ast.expr) -> str | None:
    """The plan name a ``"name":`` entry's value would bind at import, if knowable.

    ``bytes`` counts alongside ``str``: `PlanMetadata.name` is a plain ``str``
    field, which pydantic's default lax mode fills from a ``bytes`` value by
    UTF-8 decoding it -- so importing ``{"name": b"x"}`` really does register
    the plan as ``"x"``, and this reader has to say the same thing or 404 a
    live plan. Bytes that are not valid UTF-8 are `None` here for the same
    reason: pydantic rejects them, so no such plan is ever registered.
    """
    if not isinstance(value, ast.Constant):
        return None
    if isinstance(value.value, str):
        return value.value
    if isinstance(value.value, bytes):
        try:
            return value.value.decode()
        except UnicodeDecodeError:
            return None
    return None


def _find_plan_metadata_dict(tree: ast.Module) -> ast.Dict | None:
    """Return the last module-level ``PLAN_METADATA = {...}`` value, if any.

    Last rather than first: of the bindings this reader can see, the last one
    is the one an import would leave in place, in every shape the reader
    supports. It sees only module-level ``PLAN_METADATA = <literal dict>``
    statements -- a later rebinding to something computed
    (``PLAN_METADATA = _derive()``), or a rebinding nested inside an ``if`` or
    ``try``, is invisible here and deliberately outside the contract, as is any
    assignment inside a function or class.
    """
    found: ast.Dict | None = None
    for node in tree.body:
        if not isinstance(node, ast.Assign) or not isinstance(node.value, ast.Dict):
            continue
        if any(
            isinstance(target, ast.Name) and target.id == "PLAN_METADATA" for target in node.targets
        ):
            found = node.value
    return found
