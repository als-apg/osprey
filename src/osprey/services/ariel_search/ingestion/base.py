"""Base ingestion adapter interface.

This module defines the abstract base class for ARIEL ingestion adapters.
"""

import ssl
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from datetime import datetime
from typing import TYPE_CHECKING

import aiohttp

from osprey.services.ariel_search.exceptions import IngestionError
from osprey.services.ariel_search.ingestion.http import build_ssl_context
from osprey.utils.logger import get_logger

if TYPE_CHECKING:
    from osprey.services.ariel_search.config import ARIELConfig
    from osprey.services.ariel_search.models import (
        EnhancedLogbookEntry,
        FacilityEntryCreateRequest,
    )

logger = get_logger(__name__)


class FacilityAdapter(ABC):
    """Abstract base class for ingestion adapters.

    Each facility implements their own adapter to convert facility-specific
    logbook formats into the ARIEL schema. Adapters support both reading
    (fetch_entries) and optionally writing (create_entry) to facility logbooks.

    Attributes:
        config: ARIEL configuration
        metadata_sidecar_names: Attachment filenames this facility uses for the
            JSON sidecar that carries an entry's structured metadata. Declared
            per adapter because ``metadata.json`` is one facility's convention,
            not a standard; a logbook that spells it differently would
            otherwise get no metadata at all and say nothing about it. A name
            that never matches is simply a no-op, so this needs no enable
            switch beside it.
        proxy_url: SOCKS proxy for outbound logbook requests, or ``None`` for a
            direct connection.
        verify_ssl: Whether outbound logbook requests verify TLS certificates.
        ca_bundle: PEM bundle those requests verify against, or ``None`` to use
            the trust store the image ships.

    The three transport attributes are class-level defaults so that every
    adapter answers the TLS and proxy questions the same way, whether or not it
    remembered to read the ingestion config. An adapter that never sets them
    still verifies certificates and still goes direct.
    """

    #: See the class docstring. Lower-case comparison, so case does not matter.
    metadata_sidecar_names: tuple[str, ...] = ("metadata.json",)

    #: See the class docstring. Already resolved by ``IngestionConfig.from_dict``.
    proxy_url: str | None = None

    #: See the class docstring. Defaults on, so silence never means "off".
    verify_ssl: bool = True

    #: See the class docstring. ``None`` leaves aiohttp on the image trust store.
    ca_bundle: str | None = None

    def __init__(self, config: "ARIELConfig") -> None:
        """Initialize the adapter with configuration.

        Args:
            config: ARIEL configuration
        """
        self.config = config
        ingestion = config.ingestion
        if ingestion is not None:
            # A configured value wins; ``None`` falls through to the class
            # default rather than overwriting it. A YAML key written with an
            # empty value (``verify_ssl:``) arrives here as ``None``, and
            # assigning it would turn certificate checking off by silence --
            # exactly what the defaults above exist to prevent.
            if ingestion.proxy_url is not None:
                # Already resolved: IngestionConfig.from_dict folds
                # ARIEL_SOCKS_PROXY in, so no adapter re-reads the environment.
                self.proxy_url = ingestion.proxy_url
            if ingestion.verify_ssl is not None:
                self.verify_ssl = ingestion.verify_ssl
            if ingestion.ca_bundle is not None:
                self.ca_bundle = ingestion.ca_bundle

    def _create_connector(self) -> aiohttp.BaseConnector:
        """Create aiohttp connector with optional SOCKS proxy support.

        Returns:
            aiohttp connector (with proxy if configured)

        Raises:
            IngestionError: If a proxy is configured but aiohttp-socks cannot be imported
        """
        if not self.proxy_url:
            return aiohttp.TCPConnector()

        try:
            from aiohttp_socks import ProxyConnector
        except ImportError as e:
            raise IngestionError(
                "SOCKS proxy support needs aiohttp-socks, a core osprey "
                "dependency, so a missing module means a broken install — "
                "repair it with: pip install --force-reinstall aiohttp-socks",
                source_system=self.source_system_name,
            ) from e

        logger.info(f"Using SOCKS proxy: {self.proxy_url}")
        connector: aiohttp.BaseConnector = ProxyConnector.from_url(self.proxy_url)
        return connector

    def _ssl_context(self) -> ssl.SSLContext | bool:
        """Return the ``ssl=`` argument for this adapter's outbound requests.

        Returns:
            Whatever :func:`build_ssl_context` makes of ``verify_ssl`` and
            ``ca_bundle`` — one place decides, so no adapter can ship a quieter
            answer of its own.
        """
        return build_ssl_context(self.verify_ssl, self.ca_bundle)

    @property
    @abstractmethod
    def source_system_name(self) -> str:
        """Return the source system identifier.

        Examples: 'ALS eLog', 'JLab Logbook', 'ORNL Logbook'
        """

    @property
    def supports_write(self) -> bool:
        """Whether this adapter supports creating entries in the facility logbook.

        Override in subclasses that implement write support.
        """
        return False

    @property
    def requires_write_auth(self) -> bool:
        """Whether ``create_entry`` requires operator credentials to publish.

        Fail-closed default: ``True``. An adapter that publishes without
        credentials (e.g. a local file logbook) must explicitly override this to
        ``False`` — so a misconfigured adapter prompts for credentials rather
        than silently publishing unauthenticated. Adapters that require auth
        raise :class:`AuthenticationRequiredError` from ``create_entry`` when
        credentials are absent.
        """
        return True

    @abstractmethod
    def fetch_entries(
        self,
        since: datetime | None = None,
        until: datetime | None = None,
        limit: int | None = None,
    ) -> AsyncIterator["EnhancedLogbookEntry"]:
        """Fetch entries from the source system.

        Args:
            since: Only fetch entries after this timestamp
            until: Only fetch entries before this timestamp
            limit: Maximum number of entries to fetch

        Yields:
            EnhancedLogbookEntry objects with base fields populated.
            Enhancement fields are added later by enhancement modules.

        Raises:
            IngestionError: If connection to source system fails
        """

    async def create_entry(self, request: "FacilityEntryCreateRequest") -> str:
        """Create an entry in the facility logbook.

        Args:
            request: Entry creation request with subject, details, etc.

        Returns:
            The facility-assigned entry ID.

        Raises:
            NotImplementedError: If this adapter does not support writes.
            IngestionError: If the write fails.
        """
        raise NotImplementedError(
            f"{self.source_system_name} adapter does not support creating entries"
        )

    async def count_entries(
        self,
        since: datetime | None = None,
        until: datetime | None = None,
    ) -> int | None:
        """Count entries available in the source system.

        This is an optional method - adapters may return None if counting
        is not supported or too expensive.

        Args:
            since: Only count entries after this timestamp
            until: Only count entries before this timestamp

        Returns:
            Total count of entries, or None if not available
        """
        return None


# Backwards-compatible alias
BaseAdapter = FacilityAdapter
