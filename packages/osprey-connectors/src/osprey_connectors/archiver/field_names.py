"""The field name a channel address is stored under in the archive.

The archive holds one document per instant: ``date`` plus one field per
channel sampled at that instant. MongoDB does not take a field name as a
plain string everywhere it appears: in an update, a query filter or a
projection a ``.`` separates the parts of a path into sub-documents, a leading
``$`` names an operator, and a NUL byte cannot be stored at all. A channel
address carrying any of these is therefore stored under an escaped name.

The escape is ``%XX``, the uppercase hex of the byte: ``%`` is written as
``%25``, ``.`` as ``%2E`` and NUL as ``%00`` wherever they appear, and ``$``
as ``%24`` in the first position only. Every other character is kept.

The archive's documents carry fields of their own: ``_id``, ``date``,
``expireAt``, the densify marker, and the seed manifest's fields. An address
equal to one of them has its first character escaped too (``date`` is stored
as ``%64ate``), so no channel shares a field with the document that holds it.

So an address with no ``%``, no ``.``, no NUL, no leading ``$`` and not equal
to one of those fields is its own field name, byte for byte, and
:func:`channel_address` reverses :func:`field_name` for every address.

Every writer and every reader of the archive collection names a channel's
field through these two functions, so a channel is written and read back under
one name. The module imports only the standard library.
"""

from __future__ import annotations

__all__ = ["channel_address", "field_name"]

_HEX_DIGITS = "0123456789ABCDEF"

# The fields the archive's own documents carry beside channels: every
# document's ``_id``; a sample document's ``date``, ``expireAt`` and the densify
# marker ``osprey_densified``; and the seed manifest's ``fingerprint``,
# ``seeded_at``, ``touched_windows``, ``touched_anchor`` and ``coverage``, which
# live in the same collection. A test pins this set against the code that
# writes those fields.
_RESERVED_FIELDS = frozenset(
    {
        "_id",
        "coverage",
        "date",
        "expireAt",
        "fingerprint",
        "osprey_densified",
        "seeded_at",
        "touched_anchor",
        "touched_windows",
    }
)


def field_name(address: str) -> str:
    """The field name the archive stores ``address`` under.

    Args:
        address: A channel address.

    Returns:
        ``address`` with ``%``, ``.`` and NUL escaped as ``%25``, ``%2E`` and
        ``%00``, a leading ``$`` as ``%24``, and the first character of an
        address equal to one of the archive's own document fields;
        ``address`` itself when none of these applies.

    Raises:
        ValueError: for the empty address, which has no field name.
    """
    if not address:
        raise ValueError("an empty channel address has no archive field name")
    reserved = address in _RESERVED_FIELDS
    parts: list[str] = []
    for position, character in enumerate(address):
        if character in "%.\x00" or (position == 0 and (character == "$" or reserved)):
            parts.append(f"%{ord(character):02X}")
        else:
            parts.append(character)
    return "".join(parts)


def channel_address(field: str) -> str:
    """The channel address stored under the archive field ``field``.

    Args:
        field: A field name written by :func:`field_name`.

    Returns:
        The address, with every ``%XX`` replaced by its character.

    Raises:
        ValueError: when a ``%`` is not followed by two uppercase hex digits,
            which :func:`field_name` never writes.
    """
    parts: list[str] = []
    position = 0
    while position < len(field):
        character = field[position]
        if character != "%":
            parts.append(character)
            position += 1
            continue
        digits = field[position + 1 : position + 3]
        if len(digits) != 2 or any(digit not in _HEX_DIGITS for digit in digits):
            raise ValueError(
                f"{field!r} is not an archive field name: '%' at position {position} "
                f"is not followed by two uppercase hex digits"
            )
        parts.append(chr(int(digits, 16)))
        position += 3
    return "".join(parts)
