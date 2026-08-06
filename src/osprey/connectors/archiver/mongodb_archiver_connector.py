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
            ValueError: If required config values are missing
            ConnectionError: If connection cannot be established
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
            raise ImportError(
                "pymongo is required for MongoDB archiver. Install with: pip install pymongo"
            ) from e

        # Validate required config
        host = config.get("host")
        if not host:
            raise ValueError("host is required for MongoDB archiver")

        db_name = config.get("name")
        if not db_name:
            raise ValueError("name (database name) is required for MongoDB archiver")

        collection_name = config.get("collection")
        if not collection_name:
            raise ValueError("collection is required for MongoDB archiver")

        port = config.get("port", 27017)
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

        # Get password from environment variable
        password = os.getenv(password_env)
        if not password:
            raise ValueError(
                f"Environment variable '{password_env}' not set. "
                "Password is required for MongoDB authentication."
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
                "Please check connectivity and authentication."
            ) from e
        except self._ConfigurationError as e:
            raise ConnectionError(f"MongoDB configuration error: {e}") from e
        except (TimeoutError, OSError) as e:
            raise ConnectionError(f"MongoDB connection failed: {e}") from e
        except Exception as e:
            # Last resort - log and re-raise as ConnectionError
            logger.error(f"Unexpected error connecting to MongoDB: {e}", exc_info=True)
            raise ConnectionError(f"MongoDB connection failed: {e}") from e

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
            ConnectionError: If MongoDB cannot be reached
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
        except (ValueError, TypeError) as e:
            raise ValueError(f"Error retrieving data from MongoDB: {e}") from e
        except Exception as e:
            # Log unexpected errors for debugging
            logger.error(f"Unexpected error retrieving data: {e}", exc_info=True)
            raise ValueError(f"Error retrieving data from MongoDB: {e}") from e

    async def get_metadata(self, pv_name: str) -> ArchiverMetadata:
        """
        Get archiving metadata for a PV.

        Note: Basic implementation that checks if PV exists in collection.
        Could be enhanced with actual metadata queries.

        Args:
            pv_name: Name of the process variable

        Returns:
            ArchiverMetadata with basic archiving information

        Raises:
            RuntimeError: If archiver not connected
        """
        self._require_connected()

        def check_pv():
            """Check if PV exists in any document."""
            # Query for any document that has this PV field
            query = {pv_name: {"$exists": True}}
            count = self._collection.count_documents(query, limit=1)
            return count > 0

        try:
            is_archived = await asyncio.to_thread(check_pv)
        except Exception as e:
            logger.warning(f"Error checking PV metadata: {e}")
            is_archived = False

        return ArchiverMetadata(
            pv_name=pv_name,
            is_archived=is_archived,
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
