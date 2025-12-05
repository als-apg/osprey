"""
Data Source Manager

Unified data source management system that replaces both the registry and integration service
with a cleaner approach supporting core and application-specific data sources.
"""

import asyncio
import logging
import time
import warnings
from dataclasses import dataclass, field
from typing import Any

from .providers import DataSourceContext, DataSourceProvider
from .request import DataSourceRequest

logger = logging.getLogger(__name__)


@dataclass
class DataRetrievalResult:
    """Result of data retrieval from multiple sources."""

    context_data: dict[str, DataSourceContext] = field(default_factory=dict)
    successful_sources: list[str] = field(default_factory=list)
    failed_sources: list[str] = field(default_factory=list)
    total_sources_attempted: int = 0
    retrieval_time_sec: float | None = None

    @property
    def has_data(self) -> bool:
        """Check if any data was successfully retrieved."""
        return bool(self.context_data)

    @property
    def success_rate(self) -> float:
        """Calculate the success rate of data retrieval."""
        if self.total_sources_attempted == 0:
            return 0.0
        return len(self.successful_sources) / self.total_sources_attempted

    @property
    def retrieval_time_ms(self) -> float | None:
        """
        Deprecated: Use retrieval_time_sec instead.
        Returns retrieval time in milliseconds for backwards compatibility.
        """
        warnings.warn(
            "retrieval_time_ms is deprecated, use retrieval_time_sec instead",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.retrieval_time_sec * 1000 if self.retrieval_time_sec is not None else None

    def get_summary(self) -> dict[str, Any]:
        """Get a summary of the retrieval results."""
        return {
            "sources_attempted": self.total_sources_attempted,
            "sources_successful": len(self.successful_sources),
            "sources_failed": len(self.failed_sources),
            "success_rate": self.success_rate,
            "context_types_retrieved": list(
                set(ctx.context_type for ctx in self.context_data.values())
            ),
            "retrieval_time_sec": self.retrieval_time_sec,
        }


class DataSourceManager:
    """
    Unified data source management system.

    Replaces both DataSourceRegistry and DataSourceIntegrationService with a
    cleaner architecture that supports core and application-specific data sources.
    """

    def __init__(self):
        self._providers: dict[str, DataSourceProvider] = {}
        self._initialized = False

    def register_provider(self, provider: DataSourceProvider) -> None:
        """
        Register a data source provider.

        Providers are queried in registration order (framework providers first,
        then application providers).
        """
        self._providers[provider.name] = provider
        logger.info(f"Registered data source: {provider.name}")

    def get_responding_providers(self, request: DataSourceRequest) -> list[DataSourceProvider]:
        """
        Get all providers that should respond to the current request in registration order.

        Args:
            request: Data source request with requester information

        Returns:
            List of providers that should respond in registration order (framework first, then applications)
        """
        return [p for p in self._providers.values() if p.should_respond(request)]

    async def retrieve_all_context(
        self, request: DataSourceRequest, timeout_seconds: float = 30.0
    ) -> DataRetrievalResult:
        """
        Retrieve context from all responding data sources.

        Args:
            request: Data source request with requester information
            timeout_seconds: Maximum time to wait for all data sources

        Returns:
            DataRetrievalResult containing all successfully retrieved data
        """
        start_time = time.time()

        # Get responding providers in registration order
        providers = self.get_responding_providers(request)

        if not providers:
            logger.info("No data sources available for current context")
            return DataRetrievalResult(total_sources_attempted=0)

        logger.info(f"Retrieving context from {len(providers)} data sources")

        # Create retrieval tasks for all providers
        tasks = []
        for provider in providers:
            task = asyncio.create_task(
                self._retrieve_from_provider(provider, request), name=f"retrieve_{provider.name}"
            )
            tasks.append((provider.name, task))

        # Wait for all tasks with timeout
        try:
            results = await asyncio.wait_for(
                asyncio.gather(*[task for _, task in tasks], return_exceptions=True),
                timeout=timeout_seconds,
            )
        except TimeoutError:
            logger.warning(f"Data source retrieval timed out after {timeout_seconds}s")
            # Cancel remaining tasks
            for _, task in tasks:
                if not task.done():
                    task.cancel()
            results = [None] * len(tasks)  # Treat all as failed

        # Process results
        context_data = {}
        successful_sources = []
        failed_sources = []
        empty_sources = []

        for (provider_name, _), result in zip(tasks, results):
            if isinstance(result, Exception):
                logger.warning(f"Data retrieval failed for {provider_name}: {result}")
                failed_sources.append(provider_name)
            elif result is not None:
                context_data[provider_name] = result
                successful_sources.append(provider_name)

                # Check if the result has meaningful content
                has_content = False
                try:
                    # Check if data is truthy (works for UserMemories and similar types)
                    if result.data and (not hasattr(result.data, "__bool__") or bool(result.data)):
                        has_content = True
                    # Also check metadata hints like entry_count
                    elif result.metadata.get("entry_count", 0) > 0:
                        has_content = True
                except Exception:
                    # If we can't determine, assume it has content
                    has_content = True

                if has_content:
                    logger.debug(f"Successfully retrieved data from {provider_name}")
                else:
                    logger.debug(f"Retrieved empty result from {provider_name} (no data available)")
                    empty_sources.append(provider_name)
            else:
                failed_sources.append(provider_name)

        retrieval_time_sec = time.time() - start_time

        retrieval_result = DataRetrievalResult(
            context_data=context_data,
            successful_sources=successful_sources,
            failed_sources=failed_sources,
            total_sources_attempted=len(providers),
            retrieval_time_sec=retrieval_time_sec,
        )

        # Log human-readable summary with better clarity
        sources_with_data = len([s for s in successful_sources if s not in empty_sources])

        if failed_sources or empty_sources:
            details = []
            if sources_with_data > 0:
                details.append(f"{sources_with_data} with data")
            if empty_sources:
                details.append(f"{len(empty_sources)} empty")
            if failed_sources:
                details.append(f"{len(failed_sources)} failed")

            logger.info(
                f"Data sources checked: {len(providers)} ({', '.join(details)}) in {retrieval_time_sec:.2f}s"
            )
        else:
            logger.info(
                f"Retrieved data from {sources_with_data} source{'s' if sources_with_data != 1 else ''} in {retrieval_time_sec:.2f}s"
            )

        return retrieval_result

    def get_provider(self, provider_name: str) -> DataSourceProvider | None:
        """
        Get a specific data source provider by name.

        Args:
            provider_name: Name of the data source provider to retrieve

        Returns:
            DataSourceProvider if found, None otherwise
        """
        return self._providers.get(provider_name)

    async def retrieve_from_provider(
        self, provider_name: str, request: DataSourceRequest
    ) -> DataSourceContext | None:
        """
        Retrieve data from a specific provider by name.

        Args:
            provider_name: Name of the data source provider
            request: Data source request

        Returns:
            DataSourceContext if successful, None if provider not found or retrieval failed
        """
        provider = self.get_provider(provider_name)
        if not provider:
            logger.warning(f"Data source provider '{provider_name}' not found")
            return None

        if not provider.should_respond(request):
            logger.debug(f"Provider '{provider_name}' chose not to respond to request")
            return None

        return await self._retrieve_from_provider(provider, request)

    async def _retrieve_from_provider(
        self, provider: DataSourceProvider, request: DataSourceRequest
    ) -> DataSourceContext | None:
        """
        Retrieve data from a single provider with error handling.

        Args:
            provider: The data source provider to retrieve from
            request: Data source request

        Returns:
            DataSourceContext if successful, None if failed
        """
        try:
            logger.debug(f"Retrieving data from {provider.name}")
            return await provider.retrieve_data(request)
        except Exception as e:
            logger.warning(f"Failed to retrieve data from {provider.name}: {e}")
            return None


# Global manager instance
_data_source_manager: DataSourceManager | None = None


def get_data_source_manager() -> DataSourceManager:
    """
    Get the global data source manager instance.

    Loads all data sources from the registry system. Providers are queried
    in registration order (framework first, then applications).
    """
    global _data_source_manager
    if _data_source_manager is None:
        _data_source_manager = DataSourceManager()

        # Load all data sources from registry
        try:
            from osprey.registry import get_registry

            registry = get_registry()

            # Get all data sources from registry
            registry_data_sources = registry.get_all_data_sources()

            for provider in registry_data_sources:
                _data_source_manager.register_provider(provider)

            logger.info(f"Loaded {len(registry_data_sources)} data sources from registry")

        except Exception as e:
            logger.warning(f"Failed to load data sources from registry: {e}")

    return _data_source_manager
