"""
MongoDB archiver connector for historical channel data retrieval.

Provides interface to MongoDB collections containing archived channel data.
Documents are expected to have a 'date' field and one field per channel, named
by ``field_name`` of the channel address. Waveform traces live in the sibling
collection named by ``waveform_collection``, one document per (channel, instant)
of the same shape, and every read covers both.
"""

from __future__ import annotations

import asyncio
import os
import urllib.parse
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, TypeGuard

import pandas as pd
from typing_extensions import override

from osprey_connectors._contract_types import ContractAny
from osprey_connectors.archive_store import (
    CA_FILE_ERRORS,
    RESERVED_FIELDS,
    _address_of,
    address_overrides,
    field_name,
    mongo_client_kwargs,
)
from osprey_connectors.archiver._base import (
    ArchiveCoverage,
    ArchiveExtent,
    ChannelArchive,
    _NoAnswer,
)
from osprey_connectors.archiver._timerange import (
    Processing,
    ProcessingMode,
    SampleSeries,
    aggregate_series,
    long_frame,
    resolve_processing,
    utc_window,
)
from osprey_connectors.connection import read_connection_settings
from osprey_connectors.errors import ArchiveQueryError, ConnectorUnavailableError
from osprey_connectors.logger import get_logger
from osprey_connectors.service_answers import BackendSpec, Unavailable, UnavailableCause
from osprey_connectors.simulation.archive import stored_date_utc, waveform_collection

if TYPE_CHECKING:
    from pymongo import MongoClient
    from pymongo.collection import Collection

#: A stored document as pymongo decodes it: each field holds any BSON value.
_Document = dict[str, object]

logger = get_logger("mongodb_archiver_connector")

#: The block's key, named in every message the connection reader raises.
_WHERE = "archiver.settings"

# Appended to every failure to open the store. On an OSPREY-deployed stack the store
# is a container the project brings up itself, and "built but never deployed" is
# by far the most common way to reach these errors — the password is minted into
# the project's .env by `osprey up`, so before that first run there is nothing to
# authenticate with and nothing listening. Phrased as a likely remedy rather than
# the only one, since a facility MongoDB may be administered elsewhere.
DEPLOY_HINT = "If this project deploys its own MongoDB, run 'osprey up' to start it."

# pymongo is a dependency of this package, so reaching this message means an
# incomplete environment rather than an unselected option. It names the package
# to reinstall, not an extra to add -- there is no longer an extra to select --
# and it names pymongo too, so the message works whether or not the reader knows
# what backs this connector.
PYMONGO_INSTALL_HINT = (
    "pymongo is required for the MongoDB archiver. It is a dependency of "
    "osprey-connectors, so this environment is incomplete. "
    "Reinstall it with: pip install --upgrade osprey-connectors"
)

#: The url schemes a MongoDB connection string takes.
_MONGO_SCHEMES = ("mongodb", "mongodb+srv")

#: Url options the block has its own key for. A keyword argument overrides the
#: same option in the string without a word, so one fact would have two homes.
_OPTIONS_WITH_A_KEY = {
    "authsource": "auth.source",
    "tlscafile": "tls.ca_bundle",
    "serverselectiontimeoutms": "timeout_s",
}


#: Url options that carry a secret.
_SECRET_OPTIONS = frozenset({"tlscertificatekeyfilepassword"})

#: Url options that turn certificate verification off. No setting does.
_VERIFY_OFF_OPTIONS = frozenset(
    {"tlsinsecure", "tlsallowinvalidcertificates", "tlsallowinvalidhostnames"}
)

_ROUTE_LOGGED = False


def _is_mapping(value: object) -> TypeGuard[Mapping[object, object]]:
    """Whether a config value is a mapping, keys and values unvalidated."""
    return isinstance(value, Mapping)


def check_mongo_url(url: str, *, key: str) -> None:
    """Refuse a connection string the block cannot take.

    Everything else in the url passes to pymongo verbatim: hosts, ``replicaSet``,
    ``tls=true``, ``authMechanism``, ``tlsCertificateKeyFile``. Option names are
    compared case-insensitively, as MongoDB reads them. No message repeats the
    url, whose query can name key files.

    Raises:
        ValueError: The scheme is not ``mongodb`` or ``mongodb+srv``; the url
            is not a valid connection string; it carries a user or password; it
            carries a secret option; it turns certificate verification off; or
            it sets an option this block has its own key for.
    """
    try:
        parts = urllib.parse.urlsplit(url)
    except ValueError:
        raise ValueError(f"`{key}` is not a valid connection string") from None
    if parts.scheme.lower() not in _MONGO_SCHEMES:
        raise ValueError(f"`{key}` must be a mongodb:// or mongodb+srv:// connection string")
    if "@" in parts.netloc:
        raise ValueError(
            f"`{key}` may not carry a user or password; name them under "
            "`auth.username` and `auth.password_env`"
        )
    # MongoDB also takes ';' as the option separator.
    query = parts.query.replace(";", "&")
    for name, _value in urllib.parse.parse_qsl(query, keep_blank_values=True):
        option = name.lower()
        if option in _SECRET_OPTIONS:
            raise ValueError(f"`{key}` may not carry `{name}`, a secret")
        if option in _VERIFY_OFF_OPTIONS:
            raise ValueError(
                f"`{key}` may not set `{name}`; no setting turns certificate verification off"
            )
        if option in _OPTIONS_WITH_A_KEY:
            raise ValueError(
                f"`{key}` may not set `{name}`; this block sets it as "
                f"`{_OPTIONS_WITH_A_KEY[option]}`"
            )


def _log_route_once(by_url: bool, store: str) -> None:
    """Log which key reaches the store, once per process."""
    global _ROUTE_LOGGED
    if _ROUTE_LOGGED:
        return
    _ROUTE_LOGGED = True
    if by_url:
        logger.info(
            "MongoDB archiver reaches its store by `url`; host, port and the "
            "OSPREY_ARCHIVER_MONGODB_* overrides are not applied"
        )
    else:
        logger.info(f"MongoDB archiver reaches its store at {store} by `host` and `port`")


class MongoDBArchiverConnector(ChannelArchive):
    """
    MongoDB archiver connector for historical channel data.

    Provides access to historical channel data stored in MongoDB collections.
    Documents are expected to have the structure:
    {date: ISODate(...), CHANNEL1: value1, CHANNEL2: value2, ...}

    The spec's settings:

    - url: A ``mongodb://`` or ``mongodb+srv://`` connection string
      (optional). When set it names the store verbatim: ``host``, ``port`` and
      the address overrides are not applied, and ``auth:`` is optional,
      ``auth.source`` included.
    - host: MongoDB host (required without ``url``)
    - port: MongoDB host port (required without ``url`` — no default; see the
      ``port`` resolution below for why there is none)
    - name: Database name (required)
    - collection: Collection name (required)
    - auth.source: Authentication database, the one the user is defined in
      (required without ``url``)
    - auth.username: MongoDB username (required without ``url``)
    - auth.password_env: Environment variable name for password (required
      with ``auth.username``)
    - timeout_s: Default timeout in seconds (default: 60)
    - tls.ca_bundle: Absolute path of a CA file to trust (optional); setting
      it turns TLS on. Unset, a ``tls=true`` url checks the store's
      certificate against the image trust store.

    The settings are read when the archive is built; the client is opened, and
    the store pinged, on the first call, and a store that cannot be reached
    answers ``Unavailable`` with the deploy hint.

    Raises:
        ImportError: If pymongo is not installed
        ValueError: If required config *keys* are missing — an authoring
            error in config.yml, not a runtime condition
        ConnectorUnavailableError: If ``auth.password_env`` names a variable
            that is not set. That is the state a not-yet-deployed project is
            in, so it carries the deploy hint and reaches the agent as a
            ``connection_error`` rather than an internal error.

    Example:
        >>> archive = MongoDBArchiverConnector(BackendSpec(
        >>>     kind="archive", type="mongodb_archiver", target="va", rehearsal=False,
        >>>     settings={
        >>>         'host': 'mongodb.example.org',
        >>>         'port': 27017,  # osprey:not-a-port — an external facility store
        >>>                         # on MongoDB's own protocol port. A store this
        >>>                         # deployment publishes is on its `mongo` layout
        >>>                         # slot instead, written by the build.
        >>>         'name': 'my-archiver-database',
        >>>         'collection': 'my-archiver-collection',
        >>>         'auth': {
        >>>             'source': 'database-auth',
        >>>             'username': 'my-username',
        >>>             'password_env': 'MONGODB_READONLY_PASSWORD',
        >>>         },
        >>>         'timeout_s': 60,
        >>>     },
        >>> ))
        >>> frame = await archive.history(
        >>>     ['BEAM:CURRENT'],
        >>>     start_at=datetime(2024, 1, 1, tzinfo=UTC),
        >>>     end_at=datetime(2024, 1, 2, tzinfo=UTC),
        >>> )
    """

    def __init__(
        self, spec: BackendSpec, *, project_dir: Path | None = None, **seams: object
    ) -> None:
        super().__init__(spec, project_dir=project_dir, **seams)
        # The settings block is the archive's extension contract; this backend reads its keys unchecked.
        config: Mapping[str, ContractAny] = spec.settings
        self._client: MongoClient[_Document] | None = None
        self._collection: Collection[_Document] | None = None
        # The waveform sibling of ``_collection``; None on a connector wired to
        # a single collection, which then reads scalars only.
        self._waveforms: Collection[_Document] | None = None
        #: The one open of the store in flight, shared by every call that waits on it.
        self._opening: asyncio.Future[None] | None = None
        try:
            from pymongo.errors import ConnectionFailure
        except ImportError as e:
            raise ImportError(PYMONGO_INSTALL_HINT) from e
        # pymongo's connection-failure class, kept for the readers' handlers.
        self._ConnectionFailure: type[Exception] = ConnectionFailure

        raw_url: object = config.get("url")
        if raw_url is not None and not isinstance(raw_url, str):
            raise ValueError(f"`{_WHERE}.url` must be a string")
        url = raw_url if raw_url and raw_url.strip() else None
        host: str | None = None
        port: int | None = None
        override_port: int | None = None
        if url is not None:
            # A url names the store verbatim: no host, port or address override applies.
            check_mongo_url(url, key=f"{_WHERE}.url")
        else:
            # Where the store is. The configured value is the host-side truth; a
            # containerized consumer is handed the in-network address through the
            # environment and that wins (see _HOST_OVERRIDE_ENV). Resolved before the
            # required-ness check so an environment-only address is a legitimate
            # one, and the error below still fires when neither source names a host.
            override_host, override_port = address_overrides()

            host = override_host or config.get("host")
            if not host:
                raise ValueError("host is required for MongoDB archiver")

        db_name = config.get("name")
        if not db_name:
            raise ValueError("name (database name) is required for MongoDB archiver")

        collection_name = config.get("collection")
        if not collection_name:
            raise ValueError("collection is required for MongoDB archiver")

        auth = config.get("auth")
        auth_block: Mapping[object, object] = auth if _is_mapping(auth) else {}
        if url is None:
            # No literal fallback, and deliberately not one. For a store this
            # deployment publishes, the host port is its ``mongo`` layout slot —
            # ``deployment.port_base + 801`` — which this package cannot compute:
            # osprey-connectors is a separate wheel that does not depend on
            # ``osprey``, so it has no access to ``osprey.port_layout``. The build
            # always writes ``archiver.mongodb_archiver.port`` from the resolved
            # base, and an external facility store names its own port, so a missing
            # key is an authoring error rather than a number to guess: guessing
            # would dial a port belonging to a different deployment's block.
            port = override_port if override_port is not None else config.get("port")
            if port is None:
                raise ValueError(
                    "port is required for MongoDB archiver: set "
                    "archiver.mongodb_archiver.port in config.yml to the store's host "
                    "port (a project that deploys its own MongoDB gets it written by "
                    "'osprey build'), or set OSPREY_ARCHIVER_MONGODB_PORT for a "
                    "container reaching the store on the compose network"
                )

            # The login is required on this path. A bearer-token block skips these
            # checks and is refused by name by the reader below.
            if "token_env" not in auth_block:
                for leaf in ("username", "password_env"):
                    if not auth_block.get(leaf):
                        raise ValueError(f"auth.{leaf} is required for MongoDB archiver")
                if not auth_block.get("source"):
                    raise ValueError(
                        "auth.source (authentication database) is required for MongoDB archiver"
                    )

        settings = read_connection_settings(
            config,
            where=_WHERE,
            logins=frozenset({"password"}),
            unsupported_because=(
                "the MongoDB archiver takes a username and auth.password_env, not a bearer token"
            ),
            extra_auth_keys=frozenset({"source"}),
        )
        self._timeout: float = settings.timeout_or(60)
        login = settings.login
        username = login.username if login is not None else None
        password_env = login.password_env if login is not None else None
        source = auth_block.get("source")
        auth_source = str(source) if source else None
        ca_bundle = str(settings.ca_bundle) if settings.ca_bundle is not None else None

        # Get password from environment variable. An unset variable is a
        # deployment state, not a config error: `osprey up` mints the password
        # into the project's .env, so this is what a built-but-never-deployed
        # project hits. ConnectorUnavailableError so the agent gets an
        # actionable connection_error envelope instead of an opaque internal
        # error.
        password: str | None = None
        if password_env is not None:
            password = os.getenv(password_env)
            if not password:
                raise ConnectorUnavailableError(
                    f"Environment variable '{password_env}' is not set, so the MongoDB "
                    f"archiver has no password to authenticate with. {DEPLOY_HINT}"
                )

        # The url itself is never logged or put in a message: its query can name key files.
        self._store = f"the store named by {_WHERE}.url" if url is not None else f"{host}:{port}"
        _log_route_once(url is not None, self._store)
        self._names = (str(db_name), str(collection_name))
        self._ca_bundle = ca_bundle
        self._client_kwargs: dict[str, ContractAny] = mongo_client_kwargs(
            url=url,
            host=host,
            port=port,
            username=username,
            password=password,
            auth_source=auth_source,
            ca_bundle=ca_bundle,
            timeout_s=self._timeout,
        )

    def _connect(self) -> None:
        """Open the client, ping the store and bind its collections. Runs in a worker thread.

        Raises:
            _NoAnswer: The store cannot be reached or authenticated against
                (``unreachable``), or the client cannot be built from the block
                (``misconfigured``: a pymongo configuration error, or a CA file
                that does not load).
        """
        from pymongo import MongoClient
        from pymongo.errors import ConfigurationError

        store = self._store
        try:
            client: MongoClient[_Document] = MongoClient(**self._client_kwargs)
            try:
                client.admin.command("ping")
            except BaseException:
                client.close()
                raise
        except self._ConnectionFailure as e:
            raise _NoAnswer(
                Unavailable(
                    cause="unreachable",
                    reason=f"Cannot connect to MongoDB at {store}. "
                    f"Please check connectivity and authentication. {DEPLOY_HINT}",
                )
            ) from e
        except ConfigurationError as e:
            raise _NoAnswer(
                Unavailable(cause="misconfigured", reason=f"MongoDB configuration error: {e}")
            ) from e
        except (TimeoutError, OSError) as e:
            # pymongo loads the CA file while it builds the client, before any ping.
            if self._ca_bundle is not None and isinstance(e, CA_FILE_ERRORS):
                raise _NoAnswer(
                    Unavailable(
                        cause="misconfigured",
                        reason=f"MongoDB TLS setup failed: the CA file named by "
                        f"{_WHERE}.tls.ca_bundle could not be loaded: {e}",
                    )
                ) from e
            raise _NoAnswer(
                Unavailable(
                    cause="unreachable", reason=f"MongoDB connection failed: {e}. {DEPLOY_HINT}"
                )
            ) from e
        except Exception as e:
            logger.error(f"Unexpected error connecting to MongoDB: {e}", exc_info=True)
            raise _NoAnswer(
                Unavailable(cause="failed", reason=f"MongoDB connection failed: {e}. {DEPLOY_HINT}")
            ) from e

        db_name, collection_name = self._names
        self._client = client
        self._collection = client[db_name][collection_name]
        self._waveforms = client[db_name][waveform_collection(collection_name)]
        logger.debug(f"MongoDB Archiver connector initialized: {store}/{db_name}.{collection_name}")

    async def _opened(self, bound: float) -> Unavailable | None:
        """Open the store on the first call; ``None`` once it is open, else why it is not.

        One open runs at a time and every call waits on it for at most *bound*
        seconds, its own budget: a store that does not answer costs each call
        its budget, never pymongo's longer server-selection wait, and the open
        keeps running for the next call. A store that could not be opened is
        tried again on the next call, so the archive reopens itself once the
        store comes up.
        """
        if self._is_open():
            return None
        opening = self._opening
        if opening is None or (opening.done() and not self._is_open()):
            opening = self._opening = asyncio.ensure_future(asyncio.to_thread(self._connect))
            # An open nobody waited out still has its failure read, so it is not logged as lost.
            opening.add_done_callback(lambda done: done.cancelled() or done.exception())
        try:
            await asyncio.wait_for(asyncio.shield(opening), timeout=bound)
        except TimeoutError:
            return Unavailable(
                cause="timeout", reason=f"MongoDB at {self._store} did not answer within {bound}s"
            )
        except _NoAnswer as failure:
            return failure.answer
        return None

    def _is_open(self) -> bool:
        return self._collection is not None

    @override
    async def aclose(self) -> None:
        """Close the MongoDB client; the next call opens a new one."""
        client = self._client
        self._client = None
        self._collection = None
        self._waveforms = None
        self._opening = None
        if client:
            try:

                def close_connection() -> None:
                    client.close()

                await asyncio.to_thread(close_connection)
            except Exception as e:
                logger.warning(f"Error closing MongoDB connection: {e}")
        logger.debug("MongoDB Archiver connector closed")

    def _connection_error_types(self) -> tuple[type[BaseException], ...]:
        """pymongo's connection-class exceptions, as an ``except``-ready tuple.

        ``ConnectionFailure`` is the root of pymongo's network-and-server
        family — ``AutoReconnect``, ``NetworkTimeout`` and
        ``ServerSelectionTimeoutError`` all derive from it — so catching it
        covers every way a live connection can go away mid-session.
        """
        return (self._ConnectionFailure,)

    def _stores(self) -> tuple[Collection[_Document], ...]:
        """Every collection a read covers: the snapshots, then the waveform sibling.

        Raises:
            RuntimeError: If the archive holds no collection, which an opened one always does.
        """
        collection = self._collection
        if collection is None:
            raise RuntimeError("the MongoDB archive was read before it was opened")
        if self._waveforms is None:
            return (collection,)
        return (collection, self._waveforms)

    def _unread(self, e: BaseException) -> Unavailable:
        """A read of the store that raised *e*, as the answer it gives."""
        cause: UnavailableCause = (
            "unreachable"
            if isinstance(e, (ConnectionError, *self._connection_error_types()))
            else "failed"
        )
        return Unavailable(
            cause=cause, reason=f"the MongoDB archive could not be read: {type(e).__name__}"
        )

    def _refuse_aggregated_waveforms(self, fields: Mapping[str, str], resolved: Processing) -> None:
        """Refuse a non-raw mode on any requested channel the waveform store holds.

        Asked of the whole store, not of the window: a waveform channel stays one
        outside its stored range, so the answer cannot depend on the dates asked.

        Raises:
            ArchiveQueryError: naming the first waveform channel found.
        """
        if resolved.mode == "raw" or self._waveforms is None:
            return
        for channel, field in fields.items():
            if self._waveforms.count_documents({field: {"$exists": True}}, limit=1):
                raise ArchiveQueryError(
                    f"Cannot apply processing={resolved.mode!r} to waveform channel "
                    f"{channel!r}: only processing='raw' is defined for waveforms",
                    parameter="processing",
                    address=channel,
                )

    @override
    async def history(
        self,
        addresses: Sequence[str],
        *,
        start_at: datetime,
        end_at: datetime,
        precision_ms: int = 1000,
        processing: ProcessingMode = "raw",
        timeout_s: float | None = None,
    ) -> pd.DataFrame | Unavailable:
        """
        Retrieve historical data from MongoDB collection.

        Args:
            addresses: Channel addresses to retrieve
            start_at: Start of the time range, timezone-aware.
            end_at: End of the time range, timezone-aware.
            precision_ms: Time precision in milliseconds (for downsampling)
            processing: Aggregation applied within each precision_ms bin. One of
                "raw", "mean", "min", "max", "median", "std", "count". Applied
                client-side via pandas resampling. Anything else raises
                ArchiveQueryError.
            timeout_s: Optional timeout in seconds

        Returns:
            The canonical long frame — see :meth:`ChannelArchive.history` — or
            ``Unavailable`` when the store cannot be opened or read.

        Raises:
            TypeError: If start_at or end_at is not a datetime
            NaiveDatetimeError: If start_at or end_at is naive
            ArchiveQueryError: If channels is empty, or a non-raw processing
                mode is requested for a channel that carries non-numeric values
                or is a waveform channel
        """
        bound: float = timeout_s if timeout_s is not None else self._timeout

        # pymongo reads naive datetimes as UTC, so the bounds go to it in UTC.
        start_utc, end_utc = utc_window(start_at, end_at)

        if not addresses:
            raise ArchiveQueryError("channels cannot be empty", parameter="addresses")

        resolved = resolve_processing(processing, precision_ms, start_utc)

        unopened = await self._opened(bound)
        if unopened is not None:
            return unopened

        def fetch_data() -> pd.DataFrame:
            """Synchronous data fetch function."""
            fields = {channel: field_name(channel) for channel in addresses}
            # Match any document carrying at least one requested channel: ANDing
            # existence would silently return nothing for channels archived apart.
            query = {
                "date": {"$gte": start_utc, "$lte": end_utc},
                "$or": [{field: {"$exists": True}} for field in fields.values()],
            }

            # Project only the fields we need: date and requested channels.
            projection = {"date": 1, **dict.fromkeys(fields.values(), 1)}

            self._refuse_aggregated_waveforms(fields, resolved)

            # Group documents into one series per requested channel. A channel absent
            # from a given document contributes no sample for that channel.
            timestamps: dict[str, list[datetime]] = {channel: [] for channel in addresses}
            values: dict[str, list[object]] = {channel: [] for channel in addresses}
            traced: set[str] = set()
            found = 0
            for store in self._stores():
                is_waveform_store = store is self._waveforms
                for doc in store.find(query, projection).sort("date", 1):
                    found += 1
                    doc_date = doc.get("date")
                    if not isinstance(doc_date, datetime):
                        logger.warning("Document missing a datetime 'date' field, skipping")
                        continue
                    for channel, field in fields.items():
                        if field in doc:
                            value = doc[field]
                            if is_waveform_store:
                                if not isinstance(value, list):
                                    raise TypeError(f"no trace stored for waveform {channel!r}")
                                traced.add(channel)
                                value = [float(x) for x in value]
                            timestamps[channel].append(doc_date)
                            values[channel].append(value)

            if not found:
                logger.debug(f"No documents found in date range {start_utc} to {end_utc}")

            series: dict[str, SampleSeries] = {}
            for channel in addresses:
                s = pd.Series(
                    values[channel],
                    index=pd.to_datetime(timestamps[channel], utc=True),
                    name=channel,
                    dtype=object if channel in traced else None,
                )
                if channel in traced:
                    # A trace is never binned: raw returns every stored one.
                    series[channel] = s.sort_index(kind="stable")
                else:
                    # No server-side aggregation to defer to, so every scalar mode —
                    # including "raw" — is binned client-side here.
                    series[channel] = aggregate_series(s, resolved)
            return long_frame(series)

        try:
            # Use asyncio.wait_for for timeout, asyncio.to_thread for async execution
            data = await asyncio.wait_for(asyncio.to_thread(fetch_data), timeout=bound)
        except TimeoutError:
            return Unavailable(cause="timeout", reason=f"MongoDB query timed out after {bound}s")
        except ArchiveQueryError:
            raise
        except ConnectionError as e:
            return Unavailable(
                cause="unreachable", reason=f"Network connectivity issue with MongoDB: {e}"
            )
        except self._connection_error_types() as e:
            # A pymongo connection-class failure means the store went away
            # mid-session; the client reconnects on the next call.
            return Unavailable(cause="unreachable", reason=f"Lost connection to MongoDB: {e}")
        except (ValueError, TypeError) as e:
            return Unavailable(cause="bad_reply", reason=f"Error retrieving data from MongoDB: {e}")
        except Exception as e:
            # Log unexpected errors for debugging
            logger.error(f"Unexpected error retrieving data: {e}", exc_info=True)
            return Unavailable(cause="failed", reason=f"Error retrieving data from MongoDB: {e}")

        logger.debug(
            f"Retrieved MongoDB archiver data: {len(data)} rows across {len(addresses)} channels"
        )
        return data

    @override
    async def coverage(
        self, addresses: Sequence[str], *, timeout_s: float | None = None
    ) -> Mapping[str, ArchiveCoverage] | Unavailable:
        """
        Whether each channel is archived, and the span the store holds for it.

        ``first_sample_at`` and ``last_sample_at`` are the instants, in UTC, of
        the oldest and newest documents actually holding the channel, read from the
        store — not a declared or assumed coverage window. An agent that asks
        how far back the history goes gets the real answer, so a query outside
        the stored range reads as "not archived that far back" rather than as
        missing data inside a range it was told existed.

        Args:
            addresses: Names of the process variables
            timeout_s: Optional timeout in seconds

        Returns:
            One :class:`ArchiveCoverage` per address: ``is_archived`` is False
            and both bounds are None when the channel has no stored samples.
            The bounds span the snapshot collection and its waveform sibling,
            so a waveform-only channel is found. ``Unavailable`` when the store
            cannot be opened or read.
        """
        bound: float = timeout_s if timeout_s is not None else self._timeout
        unopened = await self._opened(bound)
        if unopened is not None:
            return unopened

        def stored_extent(address: str) -> ArchiveCoverage:
            """Timestamps of the oldest and newest documents carrying this channel.

            A ``date`` that is not a datetime counts as none, as ``history`` skips it.
            """
            # Sorting on 'date' rides the mandatory {date: 1} index, so this is
            # two index-ordered lookups rather than a collection scan.
            query = {field_name(address): {"$exists": True}}
            projection = {"date": 1}
            firsts: list[datetime] = []
            lasts: list[datetime] = []
            for store in self._stores():
                oldest = store.find_one(query, projection, sort=[("date", 1)])
                if oldest is None:
                    continue
                newest = store.find_one(query, projection, sort=[("date", -1)])
                if isinstance(oldest_at := oldest.get("date"), datetime):
                    firsts.append(stored_date_utc(oldest_at))
                if isinstance(newest_at := (newest or {}).get("date"), datetime):
                    lasts.append(stored_date_utc(newest_at))
            first, last = min(firsts, default=None), max(lasts, default=None)
            return ArchiveCoverage(
                address=address,
                is_archived=first is not None,
                first_sample_at=first,
                last_sample_at=last,
            )

        def read_all() -> dict[str, ArchiveCoverage]:
            return {address: stored_extent(address) for address in addresses}

        try:
            return await asyncio.wait_for(asyncio.to_thread(read_all), timeout=bound)
        except TimeoutError:
            return Unavailable(cause="timeout", reason=f"MongoDB query timed out after {bound}s")
        except Exception as e:
            logger.warning(f"Error checking channel coverage: {e}")
            return self._unread(e)

    @override
    async def extent(self, *, timeout_s: float | None = None) -> ArchiveExtent | Unavailable:
        """The channels the store records now, and its oldest and newest sample.

        ``addresses`` is every channel field of the newest documents of the
        snapshot collection and of its waveform sibling (a waveform document
        holds one channel, so every document at the sibling's newest instant is
        read). ``oldest_at`` and ``newest_at`` span both collections, in UTC. Four or more
        index-ordered reads on ``date``; never a collection scan.

        Returns:
            The extent, or ``Unavailable`` when the store cannot be opened or read.
        """
        bound: float = timeout_s if timeout_s is not None else self._timeout
        unopened = await self._opened(bound)
        if unopened is not None:
            return unopened

        def read_extent() -> ArchiveExtent:
            channels: set[str] = set()
            oldest: list[datetime] = []
            newest: list[datetime] = []
            for store in self._stores():
                first = store.find_one({}, {"date": 1}, sort=[("date", 1)])
                last = store.find_one({}, {"date": 1}, sort=[("date", -1)])
                if first is None or last is None:
                    continue
                if not isinstance(last_at := last.get("date"), datetime):
                    continue
                if isinstance(first_at := first.get("date"), datetime):
                    oldest.append(stored_date_utc(first_at))
                newest.append(stored_date_utc(last_at))
                for document in store.find({"date": last_at}):
                    channels.update(
                        _address_of(field) for field in document if field not in RESERVED_FIELDS
                    )
            return ArchiveExtent(
                addresses=frozenset(channels),
                oldest_at=min(oldest, default=None),
                newest_at=max(newest, default=None),
            )

        try:
            return await asyncio.wait_for(asyncio.to_thread(read_extent), timeout=bound)
        except TimeoutError:
            return Unavailable(cause="timeout", reason=f"MongoDB query timed out after {bound}s")
        except Exception as e:
            logger.warning(f"Error reading the archive's extent: {e}")
            return self._unread(e)

    @override
    async def probe(self, *, timeout_s: float | None = None) -> str | Unavailable:
        """Whether the store answers a ping: it is pinged when the archive opens it."""
        bound: float = timeout_s if timeout_s is not None else self._timeout
        unopened = await self._opened(bound)
        if unopened is not None:
            return unopened
        return f"the MongoDB store at {self._store} answered a ping"
