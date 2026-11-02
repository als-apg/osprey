"""How an OSPREY process reaches the archive's MongoDB store.

The MongoDB archiver connector is not the only process that opens a client to
the archive store: the archive recorder writes to it and ``osprey sim apply``
rewrites it. All of them take the store's address overrides and pymongo's
keyword arguments from here, so one configured block can never produce two
different clients. Nothing in this module imports pymongo or the connector, or
does I/O beyond reading two environment variables, so a caller pays for neither
by importing it.
"""

import os
import ssl
from typing import Any

__all__ = [
    "CA_FILE_ERRORS",
    "HOST_OVERRIDE_ENV",
    "PORT_OVERRIDE_ENV",
    "address_overrides",
    "mongo_client_kwargs",
]

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
HOST_OVERRIDE_ENV = "OSPREY_ARCHIVER_MONGODB_HOST"
PORT_OVERRIDE_ENV = "OSPREY_ARCHIVER_MONGODB_PORT"


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
    host = os.environ.get(HOST_OVERRIDE_ENV, "").strip() or None

    raw_port = os.environ.get(PORT_OVERRIDE_ENV, "").strip()
    if not raw_port:
        return host, None
    try:
        return host, int(raw_port)
    except ValueError as exc:
        raise ValueError(f"{PORT_OVERRIDE_ENV} must be an integer port (got {raw_port!r})") from exc


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
) -> dict[str, Any]:
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
    kwargs: dict[str, Any] = {"host": url} if url is not None else {"host": host, "port": port}
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
