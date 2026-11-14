"""PyYAML safe loading and dumping on libyaml when it is installed.

``yaml.safe_load`` always runs PyYAML's pure-Python scanner, even when the
``yaml`` wheel shipped the libyaml bindings — the C scanner is only reached by
naming ``yaml.CSafeLoader`` explicitly. The two loaders share the constructor
and the resolver, so a document parses to the same objects either way; only the
scanning speed differs, by an order of magnitude. Every hot parse of a
deployment's YAML (``osprey build`` re-reads its rendered ``config.yml`` many
times per build) goes through :func:`safe_load` here rather than through
``yaml.safe_load`` directly. The emitter is the same story: ``yaml.safe_dump``
never reaches libyaml, and the two emitters write the same bytes for the
block-style documents a build writes, so every hot write goes through
:func:`safe_dump`.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import IO, TypedDict, overload

import yaml

from osprey_connectors._contract_types import ContractAny

__all__ = [
    "safe_dump",
    "safe_load",
    "safe_loader",
]


def safe_loader() -> type[yaml.CSafeLoader] | type[yaml.SafeLoader]:
    """The safe loader class to parse with: libyaml's when present, else PyYAML's.

    Resolved at call time rather than at import, so a build that runs without
    the C bindings falls back on the same code path a test can exercise by
    removing ``yaml.CSafeLoader``.
    """
    return yaml.CSafeLoader if hasattr(yaml, "CSafeLoader") else yaml.SafeLoader


# The parse is ``yaml.load``'s untyped return, passed through: a document is any
# tree of plain objects, and every caller reads the shape its own document has.
def safe_load(stream: str | bytes | IO[str] | IO[bytes]) -> ContractAny:
    """Parse one YAML document the way ``yaml.safe_load`` does, on the fastest loader.

    Args:
        stream: The document — a string, bytes, or an open file.

    Returns:
        The document as plain Python objects; ``None`` for an empty document.

    Raises:
        yaml.YAMLError: On a document that does not parse.
    """
    return yaml.load(stream, Loader=safe_loader())


class _EmitterOptions(TypedDict):
    """The emitter options :func:`safe_dump` hands ``yaml.dump``, besides ``encoding``."""

    default_style: str | None
    default_flow_style: bool | None
    canonical: bool | None
    indent: int | None
    width: float | None
    allow_unicode: bool | None
    line_break: str | None
    explicit_start: bool | None
    explicit_end: bool | None
    version: tuple[int, int] | None
    tags: Mapping[str, str] | None
    sort_keys: bool


def _safe_dumper() -> type[yaml.CSafeDumper] | type[yaml.SafeDumper]:
    """The safe dumper class to emit with: libyaml's when present, else PyYAML's.

    Resolved at call time, like :func:`safe_loader`.
    """
    return yaml.CSafeDumper if hasattr(yaml, "CSafeDumper") else yaml.SafeDumper


@overload
def safe_dump(
    data: object,
    stream: None = None,
    *,
    default_style: str | None = None,
    default_flow_style: bool | None = False,
    canonical: bool | None = None,
    indent: int | None = None,
    width: float | None = None,
    allow_unicode: bool | None = None,
    line_break: str | None = None,
    encoding: None = None,
    explicit_start: bool | None = None,
    explicit_end: bool | None = None,
    version: tuple[int, int] | None = None,
    tags: Mapping[str, str] | None = None,
    sort_keys: bool = True,
) -> str: ...


@overload
def safe_dump(
    data: object,
    stream: None = None,
    *,
    default_style: str | None = None,
    default_flow_style: bool | None = False,
    canonical: bool | None = None,
    indent: int | None = None,
    width: float | None = None,
    allow_unicode: bool | None = None,
    line_break: str | None = None,
    encoding: str,
    explicit_start: bool | None = None,
    explicit_end: bool | None = None,
    version: tuple[int, int] | None = None,
    tags: Mapping[str, str] | None = None,
    sort_keys: bool = True,
) -> bytes: ...


@overload
def safe_dump(
    data: object,
    stream: IO[str] | IO[bytes],
    *,
    default_style: str | None = None,
    default_flow_style: bool | None = False,
    canonical: bool | None = None,
    indent: int | None = None,
    width: float | None = None,
    allow_unicode: bool | None = None,
    line_break: str | None = None,
    encoding: str | None = None,
    explicit_start: bool | None = None,
    explicit_end: bool | None = None,
    version: tuple[int, int] | None = None,
    tags: Mapping[str, str] | None = None,
    sort_keys: bool = True,
) -> None: ...


def safe_dump(
    data: object,
    stream: IO[str] | IO[bytes] | None = None,
    *,
    default_style: str | None = None,
    default_flow_style: bool | None = False,
    canonical: bool | None = None,
    indent: int | None = None,
    width: float | None = None,
    allow_unicode: bool | None = None,
    line_break: str | None = None,
    encoding: str | None = None,
    explicit_start: bool | None = None,
    explicit_end: bool | None = None,
    version: tuple[int, int] | None = None,
    tags: Mapping[str, str] | None = None,
    sort_keys: bool = True,
) -> str | bytes | None:
    """Emit one YAML document the way ``yaml.safe_dump`` does, on the fastest dumper.

    Args:
        data: Plain Python objects; anything else is refused, as ``yaml.safe_dump`` refuses it.
        stream: An open file to write to; ``None`` returns the document. With an
            ``encoding`` the file is a binary one, as ``yaml.safe_dump`` writes it.
        default_style, default_flow_style, canonical, indent, width,
        allow_unicode, line_break, encoding, explicit_start, explicit_end,
        version, tags, sort_keys: The emitter options ``yaml.safe_dump`` takes,
            with its defaults.

    Returns:
        With no ``stream``, the document: text, or bytes in ``encoding`` when one
        is given. ``None`` when it was written to ``stream``.

    Raises:
        yaml.representer.RepresenterError: On an object the safe dumper cannot represent.
    """
    options: _EmitterOptions = {
        "default_style": default_style,
        "default_flow_style": default_flow_style,
        "canonical": canonical,
        "indent": indent,
        "width": width,
        "allow_unicode": allow_unicode,
        "line_break": line_break,
        "explicit_start": explicit_start,
        "explicit_end": explicit_end,
        "version": version,
        "tags": tags,
        "sort_keys": sort_keys,
    }
    # One call per shape the yaml stubs type separately; each passes yaml.dump
    # exactly the arguments it was given.
    if stream is not None:
        yaml.dump(data, stream, Dumper=_safe_dumper(), encoding=encoding, **options)
        return None
    if encoding is None:
        text: str = yaml.dump(data, None, Dumper=_safe_dumper(), **options)
        return text
    encoded: bytes = yaml.dump(data, None, Dumper=_safe_dumper(), encoding=encoding, **options)
    return encoded
