"""The graph view's IRIs: one per facility record, and the way back to its id.

A place, device, channel or group of the facility file is the node
``https://narad.example.org/<kind>/<code>_<kind>_esc(<id>)``: ``code`` is the
file's identity code, written raw, and ``esc`` writes every UTF-8 byte of the
id outside ``[A-Za-z0-9]`` as ``_xHH_`` in upper-case hex. ``-``, ``_`` and
``.`` are escaped like any other byte, so two ids never share a local name,
the local name always matches :data:`osprey.facility.PN_LOCAL`, and a place and
a device of one id are two nodes.

:func:`decode` strips the literal ``<code>_<kind>_`` prefix and reverses the
escape left to right, so it needs the file's ``code`` and nothing else.
"""

from __future__ import annotations

import re

__all__ = ["KINDS", "NAMESPACE_ROOT", "decode", "escape", "iri", "local_name", "namespace"]

#: The record kinds that are graph nodes, each with its own namespace.
KINDS: tuple[str, ...] = ("place", "device", "channel", "group")

NAMESPACE_ROOT = "https://narad.example.org/"

_ALPHANUMERIC = frozenset(b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789")
_ESCAPED = re.compile(r"_x([0-9A-F]{2})_")


def namespace(kind: str) -> str:
    """Return the namespace the nodes of one kind live in.

    Args:
        kind: One of :data:`KINDS`.

    Returns:
        The namespace IRI, ending in ``/``.

    Raises:
        ValueError: If ``kind`` is not one of :data:`KINDS`.
    """
    if kind not in KINDS:
        raise ValueError(f"{kind!r} is not a graph node kind; expected one of {', '.join(KINDS)}")
    return f"{NAMESPACE_ROOT}{kind}/"


def escape(raw_id: str) -> str:
    """Write an id with every UTF-8 byte outside ``[A-Za-z0-9]`` as ``_xHH_``.

    Args:
        raw_id: The id as the facility file holds it.

    Returns:
        The escaped id, made of ``[A-Za-z0-9_]`` only.
    """
    return "".join(
        chr(byte) if byte in _ALPHANUMERIC else f"_x{byte:02X}_" for byte in raw_id.encode("utf-8")
    )


def local_name(code: str, kind: str, raw_id: str) -> str:
    """Return a record's local name, ``<code>_<kind>_esc(<id>)``.

    Args:
        code: The facility file's identity code.
        kind: One of :data:`KINDS`.
        raw_id: The record's id.

    Returns:
        The local name.

    Raises:
        ValueError: If ``kind`` is not one of :data:`KINDS`.
    """
    namespace(kind)
    return f"{code}_{kind}_{escape(raw_id)}"


def iri(code: str, kind: str, raw_id: str) -> str:
    """Return a record's IRI: its local name under its kind's namespace.

    Args:
        code: The facility file's identity code.
        kind: One of :data:`KINDS`.
        raw_id: The record's id.

    Returns:
        The IRI.

    Raises:
        ValueError: If ``kind`` is not one of :data:`KINDS`.
    """
    return namespace(kind) + local_name(code, kind, raw_id)


def decode(iri: str, code: str) -> str:
    """Return the raw id a record's IRI, or its local name, was minted from.

    Args:
        iri: The IRI, or the local name alone.
        code: The facility file's identity code.

    Returns:
        The record's id as the facility file holds it.

    Raises:
        ValueError: If ``iri`` is not one this module mints for ``code``.
    """
    local, namespace_kind = iri, None
    if iri.startswith(NAMESPACE_ROOT):
        namespace_kind, _, local = iri[len(NAMESPACE_ROOT) :].partition("/")
    for kind in KINDS:
        prefix = f"{code}_{kind}_"
        if local.startswith(prefix) and namespace_kind in (None, kind):
            try:
                return _unescape(local[len(prefix) :])
            except ValueError:
                break
    raise ValueError(f"{iri!r} is not a graph IRI of facility code {code!r}")


def _unescape(escaped: str) -> str:
    """Reverse :func:`escape`, left to right; raise ``ValueError`` on anything it never writes."""
    raw = bytearray()
    position = 0
    while position < len(escaped):
        byte = ord(escaped[position])
        if byte in _ALPHANUMERIC:
            raw.append(byte)
            position += 1
            continue
        match = _ESCAPED.match(escaped, position)
        if match is None:
            raise ValueError(escaped)
        value = int(match.group(1), 16)
        if value in _ALPHANUMERIC:
            raise ValueError(escaped)
        raw.append(value)
        position = match.end()
    return raw.decode("utf-8")
