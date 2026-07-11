"""The panel **manifest** descriptor and its fail-closed schema validator.

A panel manifest is a small JSON object declaring the identity and entry
point of a panel — the self-contained HTML mini-app the web terminal can
mount alongside the chat surface. It is the single shared contract that
the panel validator, the reference panel, and (later) runtime panel
discovery all consume, so the schema here is deliberately **forward-
compatible**: required fields are validated strictly, but unknown keys
are *preserved* rather than rejected, so a future discovery pass can add
fields (a discovery source, an approval state, ...) without breaking any
manifest already on disk.

The validator mirrors the token validator's fail-closed idiom
(``generator/validate.py``): a :class:`StrEnum` of machine-readable rule
ids, a frozen :class:`ManifestError` carrying ``rule``/``message``/
``source``, a :class:`PanelManifestError` that bundles *every* failure,
and a :func:`validate_manifest` that runs every check without short-
circuiting so a caller sees the complete set in one pass.
:func:`assert_valid`, :func:`parse_manifest`, :func:`load_manifest`, and
:func:`load_manifest_file` are the fail-closed doors: they raise
:class:`PanelManifestError` if the manifest carries any error.

Stdlib-only (``json``, ``dataclasses``, ``enum``, ``re``, ``pathlib``,
``typing``). This module is *only* the descriptor + schema; HTML/CSS
token-safety checks belong to the separate panel validator.
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

__all__ = [
    "CURRENT_SCHEMA_VERSION",
    "ManifestRule",
    "ManifestError",
    "PanelManifestError",
    "PanelManifest",
    "validate_manifest",
    "parse_manifest",
    "assert_valid",
    "load_manifest",
    "load_manifest_file",
]

#: The schema version a manifest is assumed to declare when it omits
#: ``version``. Phase 3 can bump this as the descriptor grows fields.
CURRENT_SCHEMA_VERSION = 1

#: A panel ``id`` must be a lowercase kebab slug: it starts with a
#: lowercase letter or digit, then any run of lowercase letters, digits,
#: or single hyphens. This keeps ids safe to use verbatim as URL path
#: segments, directory names, and DOM/token namespaces.
_ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9-]*$")

#: The manifest fields this schema version knows by name. Every other
#: top-level key is preserved on :attr:`PanelManifest.extras` untouched.
_KNOWN_FIELDS = frozenset({"id", "label", "entry", "version"})


class ManifestRule(StrEnum):
    """Machine-readable identifier for each kind of manifest failure."""

    #: The manifest itself is not a JSON object (dict).
    NOT_AN_OBJECT = "not_an_object"
    #: A required field (``id``/``label``/``entry``) is absent.
    MISSING_FIELD = "missing_field"
    #: A required string field is present but empty/blank.
    EMPTY_FIELD = "empty_field"
    #: A field is present but of the wrong JSON type.
    WRONG_TYPE = "wrong_type"
    #: ``id`` is a non-empty string but not a valid kebab slug.
    BAD_ID = "bad_id"


@dataclass(frozen=True)
class ManifestError:
    """A single, located manifest validation failure.

    Attributes:
        rule: Which check produced this error.
        message: Human-readable description of the failure.
        source: Where the manifest came from — a file path string or the
            ``"<manifest>"`` placeholder for an in-memory object.
    """

    rule: ManifestRule
    message: str
    source: str

    def __str__(self) -> str:
        return f"{self.source}: {self.message}"


class PanelManifestError(ValueError):
    """Raised by the fail-closed doors, bundling every :class:`ManifestError`.

    Attributes:
        errors: Every manifest failure, in the order they were found.
    """

    def __init__(self, errors: Sequence[ManifestError]) -> None:
        self.errors = list(errors)
        super().__init__("\n".join(str(error) for error in self.errors))


@dataclass(frozen=True)
class PanelManifest:
    """A parsed, validated panel manifest.

    Attributes:
        id: Stable kebab slug identifying the panel (matches
            :data:`_ID_PATTERN`).
        label: Human-readable display name.
        entry: The panel's HTML entry point — a relative path/filename
            (e.g. ``index.html``).
        version: The declared schema version (defaults to
            :data:`CURRENT_SCHEMA_VERSION` when the manifest omits it).
        extras: Every top-level key not known to this schema version,
            preserved verbatim so forward-compatible fields survive a
            parse/re-serialize round trip.
    """

    id: str
    label: str
    entry: str
    version: int = CURRENT_SCHEMA_VERSION
    extras: dict[str, Any] = field(default_factory=dict)


def _check_string_field(data: dict[str, Any], name: str, source: str) -> list[ManifestError]:
    """Validate one required, non-empty string field, collecting all errors."""
    errors: list[ManifestError] = []
    if name not in data:
        errors.append(
            ManifestError(
                rule=ManifestRule.MISSING_FIELD,
                message=f"missing required field {name!r}",
                source=source,
            )
        )
        return errors
    value = data[name]
    if not isinstance(value, str):
        errors.append(
            ManifestError(
                rule=ManifestRule.WRONG_TYPE,
                message=(f"field {name!r} must be a string, got {type(value).__name__}"),
                source=source,
            )
        )
        return errors
    if not value.strip():
        errors.append(
            ManifestError(
                rule=ManifestRule.EMPTY_FIELD,
                message=f"field {name!r} must not be empty",
                source=source,
            )
        )
    return errors


def validate_manifest(data: Any, *, source: str = "<manifest>") -> list[ManifestError]:
    """Run every manifest check and collect every failure.

    Never stops at the first failure — a caller sees the complete set in
    one pass, mirroring the token validator's contract.

    Args:
        data: The parsed manifest, expected to be a JSON object (dict).
        source: Where the manifest came from, used in error messages
            (a file path string, or the default ``"<manifest>"``).

    Returns:
        Every :class:`ManifestError` found, in check order. Empty if the
        manifest is fully valid.
    """
    if not isinstance(data, dict):
        return [
            ManifestError(
                rule=ManifestRule.NOT_AN_OBJECT,
                message=(f"manifest must be a JSON object, got {type(data).__name__}"),
                source=source,
            )
        ]

    errors: list[ManifestError] = []

    # id — required, non-empty string, and a valid kebab slug.
    id_errors = _check_string_field(data, "id", source)
    errors.extend(id_errors)
    if not id_errors and not _ID_PATTERN.fullmatch(data["id"]):
        errors.append(
            ManifestError(
                rule=ManifestRule.BAD_ID,
                message=(
                    f"field 'id' must be a lowercase kebab slug "
                    f"(matching {_ID_PATTERN.pattern!r}), got {data['id']!r}"
                ),
                source=source,
            )
        )

    # label / entry — required, non-empty strings.
    errors.extend(_check_string_field(data, "label", source))
    errors.extend(_check_string_field(data, "entry", source))

    # version — optional, but must be an int if present. JSON has no int
    # vs bool distinction at the type level, so reject bool explicitly.
    if "version" in data:
        version = data["version"]
        if isinstance(version, bool) or not isinstance(version, int):
            errors.append(
                ManifestError(
                    rule=ManifestRule.WRONG_TYPE,
                    message=(f"field 'version' must be an integer, got {type(version).__name__}"),
                    source=source,
                )
            )

    return errors


def parse_manifest(data: Any, *, source: str = "<manifest>") -> PanelManifest:
    """Validate ``data`` and return the parsed :class:`PanelManifest`.

    A fail-closed door: unknown top-level keys are preserved on
    :attr:`PanelManifest.extras` (forward compatibility), but any schema
    violation raises rather than being silently tolerated.

    Args:
        data: The parsed manifest object (dict).
        source: Where the manifest came from, used in error messages.

    Returns:
        The validated manifest.

    Raises:
        PanelManifestError: If :func:`validate_manifest` found any
            failures; carries all of them, not just the first.
    """
    errors = validate_manifest(data, source=source)
    if errors:
        raise PanelManifestError(errors)

    extras = {key: value for key, value in data.items() if key not in _KNOWN_FIELDS}
    return PanelManifest(
        id=data["id"],
        label=data["label"],
        entry=data["entry"],
        version=data.get("version", CURRENT_SCHEMA_VERSION),
        extras=extras,
    )


def assert_valid(data: Any, *, source: str = "<manifest>") -> None:
    """Validate ``data`` and raise if anything failed.

    Args:
        data: The parsed manifest object.
        source: Where the manifest came from, used in error messages.

    Raises:
        PanelManifestError: If :func:`validate_manifest` found any
            failures; carries all of them, not just the first.
    """
    errors = validate_manifest(data, source=source)
    if errors:
        raise PanelManifestError(errors)


def load_manifest(text: str, *, source: str = "<manifest>") -> PanelManifest:
    """Parse a manifest from a JSON string and validate it.

    Args:
        text: The manifest as a JSON document.
        source: Where the manifest came from, used in error messages.

    Returns:
        The validated manifest.

    Raises:
        PanelManifestError: If the JSON is malformed, or if the parsed
            manifest fails validation (carrying every failure).
    """
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise PanelManifestError(
            [
                ManifestError(
                    rule=ManifestRule.NOT_AN_OBJECT,
                    message=f"manifest is not valid JSON: {exc}",
                    source=source,
                )
            ]
        ) from exc
    return parse_manifest(data, source=source)


def load_manifest_file(path: str | Path) -> PanelManifest:
    """Read a manifest JSON file from disk and validate it.

    Args:
        path: Path to the manifest JSON file.

    Returns:
        The validated manifest.

    Raises:
        PanelManifestError: If the file is not valid JSON, or if the
            parsed manifest fails validation (carrying every failure).
        OSError: If the file cannot be read.
    """
    path = Path(path)
    return load_manifest(path.read_text(encoding="utf-8"), source=str(path))
