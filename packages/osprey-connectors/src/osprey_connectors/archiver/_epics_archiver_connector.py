"""
EPICS Archiver Appliance connector using direct HTTP calls.

Provides interface to EPICS Archiver Appliance for historical data retrieval.
Refactored from existing archiver integration code.

"""

import asyncio
import json
import ssl
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import ClassVar

import pandas as pd
from pydantic import ConfigDict, Field, ValidationError, field_validator, model_validator
from typing_extensions import override

from osprey_connectors._contract_types import ContractAny
from osprey_connectors.archiver._base import ArchiveCoverage, ChannelArchive, _NoAnswer
from osprey_connectors.archiver._timerange import (
    ProcessingMode,
    SampleSeries,
    long_frame,
    reject_non_numeric,
    resolve_processing,
    utc_window,
)
from osprey_connectors.config_sections import ConnectorSettings
from osprey_connectors.connection import ConnectionSettings, read_connection_settings, urllib_opener
from osprey_connectors.errors import ArchiveQueryError, ConfigurationError
from osprey_connectors.logger import get_logger
from osprey_connectors.model import FrozenModel
from osprey_connectors.service_answers import BackendSpec, Unavailable

logger = get_logger("epics_archiver_connector")

#: The block every archive reads its settings from, named in a refusal.
_WHERE = "archiver.settings"

#: Where a bare Archiver Appliance mounts its retrieval servlet.
_DEFAULT_RETRIEVAL_PATH = "/retrieval"


class EpicsArchiverSettings(ConnectorSettings):
    """The EPICS Archiver Appliance's block, ``archiver.settings``, read.

    Attributes:
        connection: ``url`` (required), ``timeout_s``, ``auth`` and ``tls``,
            read by :func:`~osprey_connectors.connection.read_connection_settings`.
        retrieval_path: The prefix the retrieval servlet is reached under, with
            one leading slash and none trailing; ``/retrieval`` when unset.
    """

    connection: ConnectionSettings
    retrieval_path: str = _DEFAULT_RETRIEVAL_PATH

    @model_validator(mode="before")
    @classmethod
    def _read(cls, data: object) -> object:
        if not isinstance(data, Mapping):
            return data
        connection = read_connection_settings(data, where=_WHERE)
        if not connection.url:
            raise ConfigurationError(
                "archiver URL is required for EPICS archiver", key=f"{_WHERE}.url"
            )
        retrieval_path = data.get("retrieval_path") or _DEFAULT_RETRIEVAL_PATH
        if not isinstance(retrieval_path, str):
            raise ConfigurationError(
                f"`{_WHERE}.retrieval_path` must be a string", key=f"{_WHERE}.retrieval_path"
            )
        return {
            **data,
            "connection": connection,
            "retrieval_path": "/" + retrieval_path.strip("/"),
        }


class _Sample(FrozenModel):
    """One sample of a ``getData.json`` reply: ``secs``, ``nanos`` and ``val``.

    The appliance sends more per sample (``severity``, ``status``, ``fields``),
    which the connector does not read, so a key beyond these three is ignored
    rather than refused. ``secs`` and ``nanos`` are JSON numbers, never
    ``true``/``false``, and keep the type they arrived as; ``val`` is whatever
    the channel archives.
    """

    model_config = ConfigDict(extra="ignore")

    secs: int | float
    nanos: int | float
    val: ContractAny


class _Series(FrozenModel):
    """The first series of a ``getData.json`` reply: its samples, oldest first.

    ``meta`` and anything else beside ``data`` is ignored. A ``data`` that is
    absent or empty (``[]``, ``null``, ``{}``, ``0``, ``""``) is a series with
    no samples.
    """

    model_config = ConfigDict(extra="ignore")

    data: list[_Sample] = Field(default_factory=list)

    @field_validator("data", mode="before")
    @classmethod
    def _empty_when_unset(cls, value: object) -> object:
        """An empty or null ``data`` is no samples."""
        return value or []


def _data_points(payload: object, address: str) -> list[_Sample]:
    """The samples of a ``getData.json`` reply, ``[]`` for an empty one.

    The appliance answers ``[{"meta": {...}, "data": [{"secs", "nanos", "val"}, ...]}]``;
    an empty reply is ``[]`` or a series whose ``data`` is empty. Only the
    first series is read.

    Raises:
        _NoAnswer: ``bad_reply``, if the reply is not of that shape, naming the
            PV, and the field and the reason when it is the series that is
            malformed.
    """
    if not payload:
        return []
    if not isinstance(payload, list):
        raise _bad_reply(f"Unexpected archiver reply for {address}: not a list of series objects")
    try:
        return _Series.model_validate(payload[0]).data
    except ValidationError as exc:
        raise _bad_reply(f"Unexpected archiver reply for {address}: {_fault(exc)}") from exc


def _bad_reply(reason: str) -> _NoAnswer:
    return _NoAnswer(Unavailable(cause="bad_reply", reason=reason))


def _fault(exc: ValidationError) -> str:
    """Where the first fault of a malformed series is, and what is wrong there."""
    error = exc.errors()[0]
    loc, reason = error["loc"], error["msg"]
    if loc and loc[-1] in ("int", "float"):
        # A member of ``secs``/``nanos``'s ``int | float``: the field is not a number.
        loc, reason = loc[:-1], "Input should be a number"
    return f"{'.'.join(str(part) for part in loc) or 'series'}: {reason}"


class EPICSArchiverConnector(ChannelArchive):
    """
    EPICS Archiver Appliance connector using direct HTTP calls.

    Provides access to historical PV data from EPICS Archiver Appliance
    via direct HTTP requests using Python stdlib.

    The appliance serves its retrieval servlet at ``/retrieval``; a facility
    that fronts the appliance with a reverse proxy may publish it under another
    prefix, named by ``retrieval_path``.

    The spec's settings:

    - url: Archiver URL (required)
    - timeout_s: Default timeout in seconds (default: 60)
    - retrieval_path: Prefix under which the appliance's retrieval servlet is
      reached (default: ``/retrieval``). Set it when a reverse proxy in front
      of the appliance publishes the servlet under another name.
    - auth: The login a proxy in front of the appliance asks for:
      ``token_env`` (sent as a bearer token), or ``username`` and
      ``password_env`` (sent as HTTP Basic). The secret is read from the named
      environment variable when the archive is built, and is sent only to the
      configured origin.
    - tls: ``ca_bundle``, the absolute path of the CA file the appliance's
      certificate is checked against.

    The settings are read into :class:`EpicsArchiverSettings`.

    Raises:
        ConfigurationError: If URL is not provided, or the block breaks the shape
        ConnectionError: If the variable a login names is unset

    Example:
        >>> archive = EPICSArchiverConnector(BackendSpec(
        >>>     kind="archive", type="epics_archiver", target="live", rehearsal=False,
        >>>     settings={'url': 'https://archiver.example.org:8443', 'timeout_s': 60},
        >>> ))
        >>> frame = await archive.history(
        >>>     ['BEAM:CURRENT'],
        >>>     start_at=datetime(2024, 1, 1, tzinfo=UTC),
        >>>     end_at=datetime(2024, 1, 2, tzinfo=UTC),
        >>> )
    """

    #: Where a bare Archiver Appliance mounts its retrieval servlet.
    DEFAULT_RETRIEVAL_PATH = _DEFAULT_RETRIEVAL_PATH

    settings_model: ClassVar[type[ConnectorSettings]] = EpicsArchiverSettings
    SETTINGS_KEY: ClassVar[str] = _WHERE

    def __init__(
        self, spec: BackendSpec, *, project_dir: Path | None = None, **seams: object
    ) -> None:
        super().__init__(spec, project_dir=project_dir, **seams)
        settings = self._settings(EpicsArchiverSettings)
        connection = settings.connection
        # The login's key and variable name, for messages; never the secret.
        self._login_key: str | None = None
        self._login_env: str | None = None
        self._url = (connection.url or "").rstrip("/")
        self._timeout = connection.timeout_or(60)
        self._retrieval_path = settings.retrieval_path
        self._opener: urllib.request.OpenerDirector = urllib_opener(connection)
        if connection.login is not None:
            self._login_key = connection.login.secret_key
            self._login_env = connection.login.secret_env

        logger.debug(
            f"EPICS Archiver connector initialized: {self._url} (retrieval at {self._retrieval_path})"
        )

    def _fetch_single_pv(
        self, address: str, start_str: str, end_str: str, *, channel: str
    ) -> SampleSeries:
        """
        Fetch archived data for a single PV via direct HTTP.

        Args:
            address: PV name (may include processing operators, e.g. mean_60(SR:DCCT))
            channel: The channel the series is named by: the address the caller asked for
            start_str: ISO 8601 UTC start time string
            end_str: ISO 8601 UTC end time string

        Returns:
            pd.Series with DatetimeIndex; empty Series if no data

        Raises:
            _NoAnswer: The appliance refused the login (``refused``), answered an
                error status or a malformed reply (``bad_reply``), presented a
                certificate this process does not trust (``misconfigured``),
                could not be reached (``unreachable``), or holds a waveform PV
                (``unsupported``).
        """
        params = urllib.parse.urlencode(
            {"pv": address, "from": start_str, "to": end_str, "fetchLatestMetadata": "true"}
        )
        url = f"{self._url}{self._retrieval_path}/data/getData.json?{params}"
        req = urllib.request.Request(url, method="GET")

        try:
            # Every request this connector makes goes through the opener built
            # with the archive, which carries the login and the CA.
            with self._opener.open(req, timeout=self._timeout) as resp:
                payload: object = json.loads(resp.read().decode())
        except urllib.error.HTTPError as e:
            if e.code in (401, 403):
                raise _NoAnswer(
                    Unavailable(cause="refused", reason=self._refused_login_message(e.code))
                ) from e
            raise _bad_reply(f"Cannot connect to archiver at {self._url}: {e}") from e
        except urllib.error.URLError as e:
            if isinstance(e.reason, ssl.SSLCertVerificationError):
                raise _NoAnswer(
                    Unavailable(
                        cause="misconfigured",
                        reason=f"The archiver at {self._url} presented a certificate this "
                        "process does not trust: name its CA in "
                        "`archiver.settings.tls.ca_bundle`, or add it to the image with "
                        "`images.site_ca`.",
                    )
                ) from e
            raise _NoAnswer(
                Unavailable(
                    cause="unreachable", reason=f"Cannot connect to archiver at {self._url}: {e}"
                )
            ) from e
        except json.JSONDecodeError as e:
            raise _bad_reply(f"Unexpected archiver reply for {address}: {e}") from e

        # Empty response: [] or [{"meta": ..., "data": []}]
        samples = _data_points(payload, address)
        if not samples:
            return pd.Series(dtype=float, name=channel)

        # Check for waveform PV (array-valued val)
        if isinstance(samples[0].val, list):
            raise _NoAnswer(
                Unavailable(cause="unsupported", reason=f"Waveform PVs not supported: {address}")
            )

        timestamps = pd.to_datetime(
            [sample.secs for sample in samples], unit="s", utc=True
        ) + pd.to_timedelta([sample.nanos for sample in samples], unit="ns")
        values = [sample.val for sample in samples]

        return pd.Series(values, index=timestamps, name=channel)

    def _refused_login_message(self, code: int) -> str:
        """Name the login the appliance refused, or the keys that would supply one."""
        if self._login_key is not None:
            return (
                f"The archiver at {self._url} refused the login from "
                f"`archiver.settings.auth.{self._login_key}` (HTTP {code}). "
                f"Check the credential in {self._login_env}."
            )
        return (
            f"The archiver at {self._url} asks for a login (HTTP {code}) and "
            "`archiver.settings.auth` names none: set `auth.token_env`, or "
            "`auth.username` and `auth.password_env`."
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
        Retrieve historical data from EPICS archiver.

        Args:
            addresses: Channel addresses to retrieve (EPICS PV names)
            start_at: Start of the time range, timezone-aware.
            end_at: End of the time range, timezone-aware.
            precision_ms: Bin width in milliseconds, applied server-side. Must
                be a whole number of seconds (the Archiver Appliance's operator
                syntax takes seconds); a finer width is rejected rather than
                rounded. ``<= 0`` means full resolution.
            processing: Aggregation applied within each precision_ms bin, pushed
                down to the appliance. One of "raw", "mean", "min", "max",
                "median", "std", "count". Anything else raises ArchiveQueryError.
            timeout_s: Optional timeout in seconds

        Returns:
            The canonical long frame — see :meth:`ChannelArchive.history` — or
            ``Unavailable`` when the appliance gave no answer.

        Raises:
            ArchiveQueryError: If ``precision_ms`` is positive but not a
                multiple of 1000, or if a non-raw ``processing`` mode is
                requested for a channel whose values are non-numeric
        """
        timeout_s = timeout_s if timeout_s is not None else self._timeout

        # The retrieval API's ".000Z" wire format is UTC: convert the bounds
        # rather than relabel them.
        start_utc, end_utc = utc_window(start_at, end_at)
        start_str = start_utc.strftime("%Y-%m-%dT%H:%M:%S.000Z")
        end_str = end_utc.strftime("%Y-%m-%dT%H:%M:%S.000Z")

        resolved = resolve_processing(processing, precision_ms, start_utc)

        # A bin width was asked for but the appliance cannot express it. Reject
        # rather than fall through to the bare PV name, which would answer an
        # aggregate query with full-resolution samples.
        if precision_ms > 0 and resolved.epics_operator is None:
            raise ArchiveQueryError(
                f"precision_ms={precision_ms} is not a whole number of seconds. The EPICS "
                "Archiver Appliance bins server-side in whole seconds only, so this bin "
                "width cannot be requested from it. Use a multiple of 1000, or "
                "precision_ms<=0 with processing='raw' for full resolution.",
                parameter="precision_ms",
            )

        # A None operator here can only mean full resolution.
        operator = resolved.epics_operator

        def fetch_all() -> dict[str, SampleSeries]:
            # Each series is named by its channel, not by the operator-wrapped
            # query, so a refusal below names the address the caller asked for.
            return {
                pv: self._fetch_single_pv(
                    f"{operator}({pv})" if operator else pv, start_str, end_str, channel=pv
                )
                for pv in addresses
            }

        try:
            series_dict = await asyncio.wait_for(asyncio.to_thread(fetch_all), timeout=timeout_s)
        except TimeoutError:
            return Unavailable(
                cause="timeout", reason=f"Archiver request timed out after {timeout_s}s"
            )
        except _NoAnswer as e:
            return e.answer
        except ConnectionRefusedError:
            return Unavailable(
                cause="unreachable",
                reason="Cannot connect to the archiver. "
                "Please check connectivity and SSH tunnels (if required).",
            )
        except Exception as e:
            logger.warning("EPICS archiver read failed", exc_info=True)
            if "connection" in str(e).lower():
                return Unavailable(
                    cause="unreachable", reason=f"Network connectivity issue with archiver: {e}"
                )
            return Unavailable(cause="failed", reason=f"the archiver read failed: {e}")

        # The appliance already binned server-side, so aggregate_series is
        # skipped — but its non-numeric check must run here: the appliance
        # answers an aggregate query on a string-valued PV with the raw
        # values, which would otherwise be handed back labelled as means.
        for s in series_dict.values():
            reject_non_numeric(s, resolved)

        data = long_frame(series_dict)

        logger.debug(f"Retrieved archiver data: {len(data)} rows across {len(addresses)} PVs")
        return data

    @override
    async def coverage(
        self,
        addresses: Sequence[str],
        *,
        timeout_s: float | None = None,
    ) -> Mapping[str, ArchiveCoverage]:
        """
        Whether each channel is archived.

        The connector does not ask the appliance's management interface, so it
        cannot say whether a channel is archived: ``is_archived`` is ``None``
        for every channel, with the reason in ``unknown_reason``.

        Args:
            addresses: Channel addresses (EPICS PV names)
            timeout_s: Unused: nothing is asked.

        Returns:
            One unknown :class:`ArchiveCoverage` per address
        """
        return {
            address: ArchiveCoverage(
                address=address,
                is_archived=None,
                unknown_reason=(
                    "the EPICS Archiver Appliance connector does not ask the appliance "
                    "whether this channel is archived"
                ),
            )
            for address in addresses
        }
