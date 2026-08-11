"""
MongoDB archiver connector for historical PV data retrieval.

Provides interface to MongoDB collections containing archived PV data.
Documents are expected to have a 'date' field and PV names as fields.
"""

import asyncio
import os
from datetime import datetime
from typing import Any

import pandas as pd

from osprey.connectors.archiver._timerange import (
    aggregate_long_frame,
    resolve_processing,
    utc_window,
)
from osprey.connectors.archiver.base import ArchiverConnector, ArchiverMetadata
from osprey.utils.logger import get_logger

logger = get_logger("mongodb_archiver_connector")

# Appended to every connect()-time failure. On an OSPREY-deployed stack the store
# is a container the project brings up itself, and "built but never deployed" is
# by far the most common way to reach these errors — the password is minted into
# the project's .env by `deploy up`, so before that first run there is nothing to
# authenticate with and nothing listening. Phrased as a likely remedy rather than
# the only one, since a facility MongoDB may be administered elsewhere.
DEPLOY_HINT = "If this project deploys its own MongoDB, run 'osprey deploy up' to start it."

# Names the optional dependency the way it is actually installed, so the message
# works whether or not the reader knows pymongo is what backs this connector.
PYMONGO_INSTALL_HINT = (
    "pymongo is required for the MongoDB archiver. "
    "Install it with: pip install 'osprey-framework[archiver-mongodb]'"
)

#: Environment overrides for the two connection keys whose configured value is
#: only correct when read from the host.
#:
#: The contract, stated here because this is the module that opens the socket:
#: **the config block carries the host-side truth** — where the agent, running on
#: the host, reaches the store, which for a project deploying its own store is
#: loopback at the published ``port_host``. Inside the compose network that
#: address names the reading container's own loopback and reaches nothing, so a
#: containerized consumer is given the store's network alias and container port
#: through this pair, and **the environment wins**.
#:
#: That is the ordinary environment-override contract, not a container special
#: case: set in a host shell these win there too, exactly as
#: ``OSPREY_OTEL_OPENOBSERVE_HOST`` does. Compose templates are simply the
#: caller that has a reason to set them.
#:
#: Read by two consumers — this connector and the archiver-recorder service —
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


class MongoDBArchiverConnector(ArchiverConnector):
    """
    MongoDB archiver connector for historical PV data.

    Provides access to historical PV data stored in MongoDB collections.
    Documents are expected to have the structure:
    {date: ISODate(...), PV1: value1, PV2: value2, ...}

    Example:
        >>> config = {
        >>>     'host': 'mongodb05.nersc.gov',
        >>>     'port': 27017,
        >>>     'name': 'my-archiver-database',
        >>>     'collection': 'my-archiver-collection',
        >>>     'auth': 'database-auth',
        >>>     'username': 'my-username',
        >>>     'password_env': 'MONGODB_READONLY_PASSWORD'
        >>> }
        >>> connector = MongoDBArchiverConnector()
        >>> await connector.connect(config)
        >>> df = await connector.get_data(
        >>>     pv_list=['BEAM:CURRENT'],
        >>>     start_date=datetime(2024, 1, 1),
        >>>     end_date=datetime(2024, 1, 2)
        >>> )
    """

    # pymongo exception classes, bound by connect(). Declared here so every
    # method can reference them on a connector that never completed a connect()
    # — including the stub connectors tests wire up by hand.
    _MongoClient = None
    _ConnectionFailure = None
    _ConfigurationError = None

    def __init__(self):
        self._connected = False
        self._client = None
        self._collection = None
        self._timeout = 60

    async def connect(self, config: dict[str, Any]) -> None:
        """
        Initialize MongoDB connection.

        Args:
            config: Configuration with keys:
                - host: MongoDB host (required)
                - port: MongoDB port (default: 27017)
                - name: Database name (required)
                - collection: Collection name (required)
                - auth: Authentication database (required)
                - username: MongoDB username (required)
                - password_env: Environment variable name for password (required)
                - timeout: Default timeout in seconds (default: 60)

        Raises:
            ImportError: If pymongo is not installed
            ValueError: If required config *keys* are missing — an authoring
                error in config.yml, not a runtime condition
            ConnectionError: If the store cannot be reached or authenticated
                against, including when ``password_env`` names a variable that
                is not set. These are the states a not-yet-deployed project is
                in, so they carry the deploy hint and reach the agent as a
                ``connection_error`` rather than an internal error.
        """
        try:
            from pymongo import MongoClient
            from pymongo.errors import ConfigurationError, ConnectionFailure

            # Store classes in self for lazy import pattern:
            # 1. Allows module import even if pymongo isn't installed (fails only when connect() is called)
            # 2. Makes classes available in exception handlers (import scope is local to this method)
            # 3. Enables reuse in other methods if needed
            self._MongoClient = MongoClient
            self._ConnectionFailure = ConnectionFailure
            self._ConfigurationError = ConfigurationError
        except ImportError as e:
            raise ImportError(PYMONGO_INSTALL_HINT) from e

        # Where the store is. The configured value is the host-side truth; a
        # containerized consumer is handed the in-network address through the
        # environment and that wins (see HOST_OVERRIDE_ENV). Resolved before the
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

        port = override_port if override_port is not None else config.get("port", 27017)
        self._timeout = config.get("timeout", 60)

        # Validate required authentication config
        username = config.get("username")
        if not username:
            raise ValueError("username is required for MongoDB archiver")

        password_env = config.get("password_env")
        if not password_env:
            raise ValueError("password_env is required for MongoDB archiver")

        auth_db = config.get("auth")
        if not auth_db:
            raise ValueError("auth (authentication database) is required for MongoDB archiver")

        # Get password from environment variable. An unset variable is a
        # deployment state, not a config error: `deploy up` mints the password
        # into the project's .env, so this is what a built-but-never-deployed
        # project hits. ConnectionError so the agent gets an actionable
        # connection_error envelope instead of an opaque internal error.
        password = os.getenv(password_env)
        if not password:
            raise ConnectionError(
                f"Environment variable '{password_env}' is not set, so the MongoDB "
                f"archiver has no password to authenticate with. {DEPLOY_HINT}"
            )

        try:
            # Create MongoDB client using direct parameter syntax (more readable than URI)
            self._client = self._MongoClient(
                host=host,
                port=port,
                username=username,
                password=password,
                authSource=auth_db,
                serverSelectionTimeoutMS=self._timeout * 1000,
            )

            # Test connection
            def test_connection():
                self._client.admin.command("ping")

            await asyncio.to_thread(test_connection)

            # Get collection
            self._collection = self._client[db_name][collection_name]

            self._connected = True
            logger.debug(
                f"MongoDB Archiver connector initialized: {host}:{port}/{db_name}.{collection_name}"
            )

        except self._ConnectionFailure as e:
            raise ConnectionError(
                f"Cannot connect to MongoDB at {host}:{port}. "
                f"Please check connectivity and authentication. {DEPLOY_HINT}"
            ) from e
        except self._ConfigurationError as e:
            raise ConnectionError(f"MongoDB configuration error: {e}") from e
        except (TimeoutError, OSError) as e:
            raise ConnectionError(f"MongoDB connection failed: {e}. {DEPLOY_HINT}") from e
        except Exception as e:
            # Last resort - log and re-raise as ConnectionError
            logger.error(f"Unexpected error connecting to MongoDB: {e}", exc_info=True)
            raise ConnectionError(f"MongoDB connection failed: {e}. {DEPLOY_HINT}") from e

    async def disconnect(self) -> None:
        """Cleanup MongoDB connection."""
        if self._client:
            try:

                def close_connection():
                    self._client.close()

                await asyncio.to_thread(close_connection)
            except Exception as e:
                logger.warning(f"Error closing MongoDB connection: {e}")

        self._client = None
        self._collection = None
        self._connected = False
        logger.debug("MongoDB Archiver connector disconnected")

    def _connection_error_types(self) -> tuple[type[BaseException], ...]:
        """pymongo's connection-class exceptions, as an ``except``-ready tuple.

        ``ConnectionFailure`` is the root of pymongo's network-and-server
        family — ``AutoReconnect``, ``NetworkTimeout`` and
        ``ServerSelectionTimeoutError`` all derive from it — so catching it
        covers every way a live connection can go away mid-session. Returns an
        empty tuple when ``connect()`` never ran and the classes were never
        bound; an empty tuple in an ``except`` clause simply never matches,
        which is the correct behaviour there.
        """
        return (self._ConnectionFailure,) if self._ConnectionFailure is not None else ()

    def _require_connected(self) -> None:
        """Raise ``RuntimeError`` unless a live collection is available.

        Raises:
            RuntimeError: If the connector is not connected.
        """
        if not self._connected or self._collection is None:
            raise RuntimeError("MongoDB archiver not connected")

    async def get_data(
        self,
        pv_list: list[str],
        start_date: datetime,
        end_date: datetime,
        precision_ms: int = 1000,
        timeout: int | None = None,
        processing: str = "raw",
    ) -> pd.DataFrame:
        """
        Retrieve historical data from MongoDB collection.

        Args:
            pv_list: List of PV names to retrieve
            start_date: Start of time range
            end_date: End of time range
            precision_ms: Time precision in milliseconds (for downsampling)
            timeout: Optional timeout in seconds
            processing: Aggregation applied within each precision_ms bin. One of
                "raw", "mean", "min", "max", "median", "std", "count". Applied
                client-side via pandas resampling. Anything else raises ValueError.

        Returns:
            The canonical long frame — see :meth:`ArchiverConnector.get_data`.

        Raises:
            RuntimeError: If archiver not connected
            TimeoutError: If operation times out
            ConnectionError: If MongoDB cannot be reached, or the connection is
                lost mid-query — the caller is expected to drop this connector
                and reconnect
            TypeError: If start_date or end_date are not datetime objects
            ValueError: If pv_list is empty, data retrieval fails, or a
                non-raw processing mode is requested for a channel that
                carries non-numeric values
        """
        timeout = timeout if timeout is not None else self._timeout

        self._require_connected()

        # pymongo reads naive datetimes as UTC; normalize so a bare wall-clock
        # means facility-local, as in every other connector.
        start_utc, end_utc = utc_window(start_date, end_date)

        if not pv_list:
            raise ValueError("pv_list cannot be empty")

        resolved = resolve_processing(processing, precision_ms)

        def fetch_data():
            """Synchronous data fetch function."""
            # Match any document carrying at least one requested PV: ANDing
            # existence would silently return nothing for PVs archived apart.
            query = {
                "date": {"$gte": start_utc, "$lte": end_utc},
                "$or": [{pv: {"$exists": True}} for pv in pv_list],
            }

            # Project only the fields we need: date and requested PVs.
            projection = {"date": 1, **dict.fromkeys(pv_list, 1)}

            # Query MongoDB collection
            cursor = self._collection.find(query, projection).sort("date", 1)

            # Convert to list of documents
            documents = list(cursor)

            if not documents:
                logger.debug(f"No documents found in date range {start_date} to {end_date}")

            # Group documents into one series per requested PV. A PV absent
            # from a given document contributes no sample for that channel.
            timestamps: dict[str, list] = {pv: [] for pv in pv_list}
            values: dict[str, list] = {pv: [] for pv in pv_list}
            for doc in documents:
                doc_date = doc.get("date")
                if doc_date is None:
                    logger.warning("Document missing 'date' field, skipping")
                    continue
                for pv in pv_list:
                    if pv in doc:
                        timestamps[pv].append(doc_date)
                        values[pv].append(doc[pv])

            # No server-side aggregation to defer to, so every mode — including
            # "raw" — is binned client-side here.
            return aggregate_long_frame(
                {
                    pv: pd.Series(
                        values[pv], index=pd.to_datetime(timestamps[pv], utc=True), name=pv
                    )
                    for pv in pv_list
                },
                resolved,
            )

        try:
            # Use asyncio.wait_for for timeout, asyncio.to_thread for async execution
            data = await asyncio.wait_for(asyncio.to_thread(fetch_data), timeout=timeout)

            logger.debug(
                f"Retrieved MongoDB archiver data: {len(data)} rows across {len(pv_list)} PVs"
            )
            return data

        except TimeoutError as e:
            raise TimeoutError(f"MongoDB query timed out after {timeout}s") from e
        except ConnectionError as e:
            raise ConnectionError(f"Network connectivity issue with MongoDB: {e}") from e
        except self._connection_error_types() as e:
            # A pymongo connection-class failure means the store went away
            # mid-session. Surfacing it as ConnectionError (rather than the
            # ValueError the generic branch below would produce) is what makes
            # connector_error_handler invalidate the cached connector, so the
            # next tool call reconnects instead of reusing a dead client.
            raise ConnectionError(f"Lost connection to MongoDB: {e}") from e
        except (ValueError, TypeError) as e:
            raise ValueError(f"Error retrieving data from MongoDB: {e}") from e
        except Exception as e:
            # Log unexpected errors for debugging
            logger.error(f"Unexpected error retrieving data: {e}", exc_info=True)
            raise ValueError(f"Error retrieving data from MongoDB: {e}") from e

    async def get_metadata(self, pv_name: str) -> ArchiverMetadata:
        """
        Get archiving metadata for a PV.

        ``archival_start`` and ``archival_end`` are the timestamps of the
        oldest and newest documents actually holding this PV, read from the
        store — not a declared or assumed coverage window. An agent that asks
        how far back the history goes gets the real answer, so a query outside
        the stored range reads as "not archived that far back" rather than as
        missing data inside a range it was told existed.

        Args:
            pv_name: Name of the process variable

        Returns:
            ArchiverMetadata; ``is_archived`` is False and both bounds are None
            when the PV has no stored samples or the store cannot be queried.

        Raises:
            RuntimeError: If archiver not connected
        """
        self._require_connected()

        def stored_extent():
            """Timestamps of the oldest and newest documents carrying this PV."""
            # Sorting on 'date' rides the mandatory {date: 1} index, so this is
            # two index-ordered lookups rather than a collection scan.
            query = {pv_name: {"$exists": True}}
            projection = {"date": 1}
            oldest = self._collection.find_one(query, projection, sort=[("date", 1)])
            if oldest is None:
                return None, None
            newest = self._collection.find_one(query, projection, sort=[("date", -1)])
            return oldest.get("date"), (newest or {}).get("date")

        try:
            first, last = await asyncio.to_thread(stored_extent)
        except Exception as e:
            logger.warning(f"Error checking PV metadata: {e}")
            first = last = None

        return ArchiverMetadata(
            pv_name=pv_name,
            is_archived=first is not None,
            archival_start=first,
            archival_end=last,
            description=f"MongoDB Archived PV: {pv_name}",
        )

    async def check_availability(self, pv_names: list[str]) -> dict[str, bool]:
        """
        Check which PVs are archived in the MongoDB collection.

        Args:
            pv_names: List of PV names to check

        Returns:
            Dictionary mapping PV name to availability status

        Raises:
            RuntimeError: If archiver not connected
        """
        self._require_connected()

        def check_pvs():
            """Check which PVs exist in the collection."""
            availability = {}
            for pv in pv_names:
                query = {pv: {"$exists": True}}
                count = self._collection.count_documents(query, limit=1)
                availability[pv] = count > 0
            return availability

        try:
            availability = await asyncio.to_thread(check_pvs)
        except Exception as e:
            logger.warning(f"Error checking PV availability: {e}")
            # Return all False on error
            availability = dict.fromkeys(pv_names, False)

        return availability
