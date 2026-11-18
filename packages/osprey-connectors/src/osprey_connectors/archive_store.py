"""The archive's MongoDB store: how a channel is stored in it, and how a process reaches it.

The MongoDB archive is not the only process that opens the store: the archive
recorder writes to it, ``osprey sim apply`` rewrites it and the deploy-time seed
fills it. Every one of them names a channel's field, the store's own fields and
the client it opens from here, so a channel is written and read back under one
name and one configured block can never produce two different clients.

The store holds one document per instant: :data:`DATE_FIELD` plus one field per
channel sampled at that instant. MongoDB does not take a field name as a plain
string everywhere it appears: in an update, a query filter or a projection a
``.`` separates the parts of a path into sub-documents, a leading ``$`` names an
operator, and a NUL byte cannot be stored at all. A channel address carrying any
of these is therefore stored under an escaped name.

The escape is ``%XX``, the uppercase hex of the byte: ``%`` is written as
``%25``, ``.`` as ``%2E`` and NUL as ``%00`` wherever they appear, and ``$``
as ``%24`` in the first position only. Every other character is kept.

The store's documents carry fields of their own, :data:`RESERVED_FIELDS`. An
address equal to one of them has its first character escaped too (``date`` is
stored as ``%64ate``), so no channel shares a field with the document that holds
it.

So an address with no ``%``, no ``.``, no NUL, no leading ``$`` and not equal
to one of those fields is its own field name, byte for byte, and the escape is
injective: two distinct addresses never share a field.

Nothing in this module imports pymongo or the archive, or does I/O beyond
reading two environment variables, so a caller pays for neither by importing
it. It imports only the standard library and the package's ``Any`` alias.
"""

from __future__ import annotations

import os
import re
import ssl

from osprey_connectors._contract_types import ContractAny

__all__ = [
    "CA_FILE_ERRORS",
    "DATE_FIELD",
    "EXPIRE_FIELD",
    "RESERVED_FIELDS",
    "address_overrides",
    "field_name",
    "mongo_client_kwargs",
]

#: Field naming the instant a document's sample was taken. The archive queries
#: and sorts on it; the seed manifest deliberately has none, which is what keeps
#: it out of every archive read without needing a filter.
DATE_FIELD = "date"

#: Field the TTL index acts on. A document without it never expires.
EXPIRE_FIELD = "expireAt"

#: The fields the store's own documents carry beside channels: every document's
#: ``_id``; a sample document's :data:`DATE_FIELD`, :data:`EXPIRE_FIELD` and the
#: densify marker ``osprey_densified``; and the seed manifest's ``fingerprint``,
#: ``seeded_at``, ``touched_windows``, ``touched_anchor`` and ``coverage``, which
#: live in the same collection. A test pins this set against the code that
#: writes those fields.
RESERVED_FIELDS = frozenset(
    {
        "_id",
        "coverage",
        DATE_FIELD,
        EXPIRE_FIELD,
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
    reserved = address in RESERVED_FIELDS
    parts: list[str] = []
    for position, character in enumerate(address):
        if character in "%.\x00" or (position == 0 and (character == "$" or reserved)):
            parts.append(f"%{ord(character):02X}")
        else:
            parts.append(character)
    return "".join(parts)


def _address_of(field: str) -> str:
    """The channel address stored under ``field``: the inverse of :func:`field_name`.

    :func:`field_name` escapes every ``%``, so every ``%XX`` in a field name is
    an escape and decoding them restores the address.
    """
    return re.sub(r"%([0-9A-F]{2})", lambda match: chr(int(match.group(1), 16)), field)


#: Environment overrides for the two connection keys whose configured value is
#: only correct when read from the host.
#:
#: The contract, stated here because every process that opens a socket to the
#: store reads it from here:
#: **the config block carries the host-side truth** — where the agent, running on
#: the host, reaches the store, which for a project deploying its own store is
#: loopback at the published ``port_host``. Inside the compose network that
#: address names the reading container's own loopback and reaches nothing, so a
#: containerized consumer is given the store's network alias and container port
#: through this pair, and **the environment wins**.
#:
#: That is the ordinary environment-override contract, not a container special
#: case: set in a host shell these win there too. Compose templates are simply the
#: caller that has a reason to set them.
#:
#: Read by two consumers — the MongoDB archiver connector and the archiver-recorder service —
#: through :func:`address_overrides`, so both agree on what an empty value means
#: and on how the port is typed.
_HOST_OVERRIDE_ENV = "OSPREY_ARCHIVER_MONGODB_HOST"
_PORT_OVERRIDE_ENV = "OSPREY_ARCHIVER_MONGODB_PORT"


def address_overrides() -> tuple[str | None, int | None]:
    """The in-network store address this process was given, if it was given one.

    Returns:
        ``(host, port)``, either of which is ``None`` when the corresponding
        variable is unset — or set to whitespace, which is treated as unset
        rather than as an address of nothing. A caller applies whichever half it
        got over its own configured value, and keeps its own rules about what to
        do when neither source supplies one.

    Raises:
        ValueError: If the port override is set to something that is not an
            integer. Silently falling back to the configured host-side port
            would send a container to an address nothing serves, and the reason
            would be a typo nobody is looking at.
    """
    host = os.environ.get(_HOST_OVERRIDE_ENV, "").strip() or None

    raw_port = os.environ.get(_PORT_OVERRIDE_ENV, "").strip()
    if not raw_port:
        return host, None
    try:
        return host, int(raw_port)
    except ValueError as exc:
        raise ValueError(
            f"{_PORT_OVERRIDE_ENV} must be an integer port (got {raw_port!r})"
        ) from exc


# ContractAny: the values pymongo.MongoClient takes as ``**kwargs: Any``, past
# host and port; every caller unpacks the result straight into it.
def mongo_client_kwargs(
    *,
    url: str | None,
    host: str | None,
    port: int | None,
    username: str | None,
    password: str | None,
    auth_source: str | None,
    ca_bundle: str | None,
    timeout_s: float,
) -> dict[str, ContractAny]:
    """The keyword arguments for ``pymongo.MongoClient``.

    Every MongoDB client OSPREY opens takes its arguments from here: the agent's
    connector, the archive recorder and the archive rewrite. One block can
    therefore never produce two different clients. This is the only place
    pymongo's keyword names are spelled.

    Pure: it reads no environment and does no I/O. A caller applies the address
    overrides (:func:`address_overrides`) to ``host`` and ``port`` itself,
    before calling, because a host-side caller must take nothing from the
    ambient environment.

    A key whose value is unset is left out rather than passed as ``None``.

    Args:
        url: A connection string; when set it is passed as ``host`` and no
            ``port`` is passed.
        host: The store's host, used when ``url`` is unset.
        port: The store's port, used when ``url`` is unset.
        username: The login user; ``password`` and ``auth_source`` go with it.
            Without one, neither is passed.
        password: The login secret.
        auth_source: The database the user is defined in.
        ca_bundle: A CA file to trust; setting it turns TLS on.
        timeout_s: Seconds to wait for a server.
    """
    kwargs: dict[str, ContractAny] = (
        {"host": url} if url is not None else {"host": host, "port": port}
    )
    if username is not None:
        kwargs["username"] = username
        kwargs["password"] = password
        if auth_source is not None:
            kwargs["authSource"] = auth_source
    if ca_bundle is not None:
        kwargs["tlsCAFile"] = ca_bundle
    kwargs["serverSelectionTimeoutMS"] = int(timeout_s * 1000)
    return kwargs


#: Errors pymongo raises while it loads a CA file.
CA_FILE_ERRORS = (ssl.SSLError, FileNotFoundError, IsADirectoryError, PermissionError)
