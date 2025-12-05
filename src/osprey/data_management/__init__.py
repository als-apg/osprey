"""
Osprey Agent Framework - Data Sources

Unified data source management system supporting both core data sources
(always available) and application-specific data sources (domain-specific).

This module provides the complete data source management framework including:
- Base abstractions for data source providers
- Unified manager for core and application-specific sources
- Request abstraction for structured data source requests
- Integration service for orchestrating data retrieval

Key Components:
- DataSourceProvider: Base class for all data source providers
- DataSourceManager: Unified management of all data sources
- DataSourceContext: Standardized format for data source results
- DataSourceRequest: Structured request information for data source providers
"""

from .manager import DataRetrievalResult, DataSourceManager, get_data_source_manager
from .providers import DataSourceContext, DataSourceProvider
from .request import DataSourceRequest, DataSourceRequester, create_data_source_request

__all__ = [
    # Base abstractions
    "DataSourceProvider",
    "DataSourceContext",
    # Request management
    "DataSourceRequest",
    "DataSourceRequester",
    "create_data_source_request",
    # Manager and results
    "DataSourceManager",
    "DataRetrievalResult",
    "get_data_source_manager",
]
