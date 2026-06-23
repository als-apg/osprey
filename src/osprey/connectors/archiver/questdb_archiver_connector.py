"""
QuestDB archiver connector for historical PV data retrieval.

Provides interface to QuestDB instances containing archived PV data.
Reads data over the PostgreSQL wire protocol (port 8812) using asyncpg.

Assumed schema (table and column names are configurable):

    CREATE TABLE pv_archive (
        ts        TIMESTAMP,   -- designated timestamp
        pv_name   SYMBOL,      -- PV identifier
        value     DOUBLE
    ) TIMESTAMP(ts) PARTITION BY DAY;

Note on identifiers vs values: asyncpg's parameterized queries ($1, $2, ...)
safely bind *values* (PV names, timestamps). Table/column *names* come only
from connector config (never user input at request time) and are validated
against a strict allow-list pattern before being interpolated into SQL.
"""

import asyncio
import re
from datetime import UTC, datetime
from typing import Any

import pandas as pd

from osprey.connectors.archiver.base import ArchiverConnector, ArchiverMetadata
from osprey.utils.logger import get_logger

logger = get_logger("questdb_archiver_connector")

# Identifiers (table/column names) are only ever validated against this
# pattern -- they never come from per-request user input, only from the
# connect() config, but we validate anyway since they're interpolated
# directly into SQL (asyncpg has no way to parameterize identifiers).
_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _validate_identifier(name: str, label: str) -> str:
    """Ensure a table/column name is a safe SQL identifier before use."""
    if not _IDENTIFIER_RE.match(name):
        raise ValueError(f"Invalid {label} '{name}': must be a plain SQL identifier")
    return name


def _to_utc(dt: datetime) -> datetime:
    """Normalize a datetime to UTC, assuming naive datetimes are already UTC."""
    if dt.tzinfo is None:
        return dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC)


def _ms_to_sample_unit(precision_ms: int) -> str:
    """Convert milliseconds to a QuestDB SAMPLE BY unit string."""
    if precision_ms < 1000:
        return f"{precision_ms}T"
    seconds = precision_ms // 1000
    if seconds < 60:
        return f"{seconds}s"
    minutes = seconds // 60
    if minutes < 60:
        return f"{minutes}m"
    hours = minutes // 60
    if hours < 24:
        return f"{hours}h"
    return f"{hours // 24}d"


class QuestDBArchiverConnector(ArchiverConnector):
    """
    QuestDB archiver connector for historical PV data.

    Connects to QuestDB over its PostgreSQL wire protocol using asyncpg.
    Queries a single narrow (long-format) table and pivots results into
    the wide DataFrame shape required by the ArchiverConnector interface.

    All PV names and timestamps are passed as bound query parameters
    ($1, $2, ...) rather than interpolated into the SQL string, so values
    containing quotes or other special characters cannot break or inject
    into the query. Table/column names come only from connector config and
    are validated against a strict identifier pattern.

    Example:
        >>> config = {
        >>>     'host': 'questdb.als.lbl.gov',
        >>>     'port': 8812,
        >>>     'database': 'qdb',
        >>>     'username': 'admin',
        >>>     'password_env': 'QUESTDB_PASSWORD',
        >>>     'table': 'pv_archive',
        >>> }
        >>> connector = QuestDBArchiverConnector()
        >>> await connector.connect(config)
        >>> df = await connector.get_data(
        >>>     pv_list=['BEAM:CURRENT'],
        >>>     start_date=datetime(2024, 1, 1),
        >>>     end_date=datetime(2024, 1, 2)
        >>> )
    """

    def __init__(self):
        self._connected = False
        self._pool = None
        self._timeout = 60

        # Schema config (overridable via connect config)
        self._table = "pv_archive"
        self._pv_col = "pv_name"
        self._val_col = "value"
        self._ts_col = "ts"

    async def connect(self, config: dict[str, Any]) -> None:
        """
        Open an asyncpg connection pool to QuestDB.

        Args:
            config: Configuration with keys:
                - host: QuestDB host (required)
                - port: PostgreSQL wire protocol port (default: 8812)
                - database: Database name (default: 'qdb')
                - username: QuestDB username (required)
                - password_env: Environment variable name for password (required)
                - timeout: Default timeout in seconds (default: 60)
                - table: Table name (default: 'pv_archive')
                - pv_column: PV name column (default: 'pv_name')
                - value_column: Value column (default: 'value')
                - ts_column: Timestamp column (default: 'ts')

        Raises:
            ImportError: If asyncpg is not installed
            ValueError: If required config values are missing or schema
                names are not valid SQL identifiers
            ConnectionError: If connection cannot be established
        """
        try:
            import asyncpg
        except ImportError as e:
            raise ImportError(
                "asyncpg is required for QuestDB archiver. Install with: pip install asyncpg"
            ) from e

        import os

        host = config.get("host")
        if not host:
            raise ValueError("host is required for QuestDB archiver")

        username = config.get("username")
        if not username:
            raise ValueError("username is required for QuestDB archiver")

        password_env = config.get("password_env")
        if not password_env:
            raise ValueError("password_env is required for QuestDB archiver")

        password = os.getenv(password_env)
        if not password:
            raise ValueError(
                f"Environment variable '{password_env}' not set. "
                "Password is required for QuestDB authentication."
            )

        port = config.get("port", 8812)
        database = config.get("database", "qdb")
        self._timeout = config.get("timeout", 60)

        # Schema overrides -- validated since they get interpolated into SQL
        # as identifiers (asyncpg can only parameterize values, not names).
        self._table = _validate_identifier(config.get("table", self._table), "table name")
        self._pv_col = _validate_identifier(config.get("pv_column", self._pv_col), "pv_column")
        self._val_col = _validate_identifier(
            config.get("value_column", self._val_col), "value_column"
        )
        self._ts_col = _validate_identifier(config.get("ts_column", self._ts_col), "ts_column")

        try:
            self._pool = await asyncpg.create_pool(
                host=host,
                port=port,
                database=database,
                user=username,
                password=password,
                command_timeout=self._timeout,
            )

            # Smoke test
            async with self._pool.acquire() as conn:
                await conn.fetchval("SELECT 1")

            self._connected = True
            logger.debug(f"QuestDB archiver connector initialized: {host}:{port}/{database}")

        except asyncpg.PostgresError as e:
            await self.disconnect()
            raise ConnectionError(
                f"Cannot connect to QuestDB at {host}:{port}. "
                "Please check connectivity and authentication."
            ) from e
        except OSError as e:
            await self.disconnect()
            raise ConnectionError(f"QuestDB connection failed: {e}") from e

    async def disconnect(self) -> None:
        """Close the asyncpg connection pool. Safe to call when not connected."""
        if self._pool:
            try:
                await self._pool.close()
            except Exception as e:
                logger.warning(f"Error closing QuestDB connection pool: {e}")

        self._pool = None
        self._connected = False
        logger.debug("QuestDB archiver connector disconnected")

    async def get_data(
        self,
        pv_list: list[str],
        start_date: datetime,
        end_date: datetime,
        precision_ms: int = 1000,
        timeout: int | None = None,
    ) -> pd.DataFrame:
        """
        Retrieve historical data from QuestDB.

        Queries the configured table for the given PVs and time range.
        When precision_ms > 0, uses QuestDB's SAMPLE BY for server-side
        downsampling. Results are pivoted from long to wide format.

        PV names and timestamps are passed as bound parameters, not
        interpolated into the SQL string. start_date/end_date are
        normalized to UTC before querying; naive datetimes are assumed
        to already be in UTC.

        Args:
            pv_list: List of PV names to retrieve
            start_date: Start of time range
            end_date: End of time range
            precision_ms: Time precision in milliseconds (0 = raw rows)
            timeout: Optional timeout in seconds

        Returns:
            DataFrame with a UTC datetime index and one column per PV.
            PVs with no data in range are present as NaN columns.

        Raises:
            RuntimeError: If archiver not connected
            TypeError: If start_date or end_date are not datetime objects
            ValueError: If pv_list is empty or time range is invalid
            TimeoutError: If operation times out
            ConnectionError: If QuestDB cannot be reached
        """
        timeout = timeout or self._timeout

        if not self._connected or self._pool is None:
            raise RuntimeError("QuestDB archiver not connected")

        if not isinstance(start_date, datetime):
            raise TypeError(f"start_date must be a datetime object, got {type(start_date)}")
        if not isinstance(end_date, datetime):
            raise TypeError(f"end_date must be a datetime object, got {type(end_date)}")

        if not pv_list:
            raise ValueError("pv_list cannot be empty")

        start_utc = _to_utc(start_date)
        end_utc = _to_utc(end_date)
        if start_utc >= end_utc:
            raise ValueError("start_date must be before end_date")

        # PV names and timestamps are bound as $1, $2, ... -- never
        # interpolated into the SQL string. Table/column names are
        # identifiers validated in connect() and are safe to interpolate.
        pv_placeholders = ", ".join(f"${i + 3}" for i in range(len(pv_list)))

        if precision_ms > 0:
            sample_unit = _ms_to_sample_unit(precision_ms)
            sql = (
                f"SELECT {self._ts_col}, {self._pv_col}, "
                f"avg({self._val_col}) AS {self._val_col} "
                f"FROM {self._table} "
                f"WHERE {self._ts_col} BETWEEN $1 AND $2 "
                f"  AND {self._pv_col} IN ({pv_placeholders}) "
                f"SAMPLE BY {sample_unit} ALIGN TO CALENDAR;"
            )
        else:
            sql = (
                f"SELECT {self._ts_col}, {self._pv_col}, {self._val_col} "
                f"FROM {self._table} "
                f"WHERE {self._ts_col} BETWEEN $1 AND $2 "
                f"  AND {self._pv_col} IN ({pv_placeholders}) "
                f"ORDER BY {self._ts_col};"
            )

        params = [start_utc, end_utc, *pv_list]

        async def fetch():
            async with self._pool.acquire() as conn:
                return await conn.fetch(sql, *params)

        try:
            records = await asyncio.wait_for(fetch(), timeout=float(timeout))
        except TimeoutError as e:
            raise TimeoutError(f"QuestDB query timed out after {timeout}s") from e
        except Exception as e:
            # Network/connection issues surface here distinctly from a
            # plain timeout so callers can tell the two apart.
            raise ConnectionError(f"QuestDB query failed: {e}") from e

        if not records:
            logger.debug(f"No data found in range {start_utc} to {end_utc}")
            return pd.DataFrame(index=pd.DatetimeIndex([], tz="UTC"), columns=pv_list)

        raw_df = pd.DataFrame(records, columns=[self._ts_col, self._pv_col, self._val_col])
        raw_df[self._ts_col] = pd.to_datetime(raw_df[self._ts_col], utc=True)

        df = raw_df.pivot_table(
            index=self._ts_col,
            columns=self._pv_col,
            values=self._val_col,
            aggfunc="mean",
        )
        df.index.name = "datetime"
        df.columns.name = None

        # Ensure all requested PVs are present
        for pv in pv_list:
            if pv not in df.columns:
                df[pv] = float("nan")

        logger.debug(f"Retrieved QuestDB data: {len(df)} points for {len(pv_list)} PVs")
        return df[pv_list]

    async def get_metadata(self, pv_name: str) -> ArchiverMetadata:
        """
        Get archiving metadata for a PV.

        Args:
            pv_name: Name of the process variable

        Returns:
            ArchiverMetadata with archiving information

        Raises:
            RuntimeError: If archiver not connected
            ValueError: If pv_name is empty
        """
        if not self._connected or self._pool is None:
            raise RuntimeError("QuestDB archiver not connected")

        if not pv_name:
            raise ValueError("pv_name cannot be empty")

        sql = (
            f"SELECT "
            f"  min({self._ts_col}) AS archival_start, "
            f"  max({self._ts_col}) AS archival_end, "
            f"  count() AS sample_count, "
            f"  datediff('ms', min({self._ts_col}), max({self._ts_col})) "
            f"      / nullif(count() - 1, 0) AS avg_period_ms "
            f"FROM {self._table} "
            f"WHERE {self._pv_col} = $1;"
        )

        try:
            async with self._pool.acquire() as conn:
                row = await conn.fetchrow(sql, pv_name)
        except Exception as e:
            # A query failure here means we genuinely don't know the
            # archival status -- report it rather than silently claiming
            # the PV isn't archived, which would be misleading.
            raise ConnectionError(f"QuestDB metadata query failed for {pv_name}: {e}") from e

        if not row or row["sample_count"] == 0:
            return ArchiverMetadata(pv_name=pv_name, is_archived=False)

        start = pd.to_datetime(row["archival_start"], utc=True).to_pydatetime()
        end = pd.to_datetime(row["archival_end"], utc=True).to_pydatetime()
        avg_ms = row["avg_period_ms"]
        sampling_s = float(avg_ms) / 1000.0 if avg_ms is not None else None

        return ArchiverMetadata(
            pv_name=pv_name,
            is_archived=True,
            archival_start=start,
            archival_end=end,
            sampling_period=sampling_s,
        )

    async def check_availability(self, pv_names: list[str]) -> dict[str, bool]:
        """
        Check which PVs have archived data.

        Uses a single batched query for efficiency.

        Args:
            pv_names: List of PV names to check

        Returns:
            Dictionary mapping PV name to availability status

        Raises:
            RuntimeError: If archiver not connected
            ConnectionError: If the availability query fails
        """
        if not self._connected or self._pool is None:
            raise RuntimeError("QuestDB archiver not connected")

        if not pv_names:
            return {}

        placeholders = ", ".join(f"${i + 1}" for i in range(len(pv_names)))
        sql = (
            f"SELECT DISTINCT {self._pv_col} "
            f"FROM {self._table} "
            f"WHERE {self._pv_col} IN ({placeholders});"
        )

        try:
            async with self._pool.acquire() as conn:
                records = await conn.fetch(sql, *pv_names)
        except Exception as e:
            # Surface the failure rather than reporting every PV as
            # unavailable, which would look identical to "checked, none
            # archived" and could mislead a caller.
            raise ConnectionError(f"QuestDB availability query failed: {e}") from e

        found = {row[0] for row in records}
        return {pv: (pv in found) for pv in pv_names}
