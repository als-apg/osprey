"""
Data Source Request Abstraction

Provides structured request information for data source providers.
"""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from osprey.state import AgentState


@dataclass
class DataSourceRequester:
    """
    Information about the component requesting data from a data source.

    Enables data sources to make decisions about whether to respond
    based on the requesting component and execution context.
    """

    component_type: str  # "task_extraction", "capability", "orchestrator"
    component_name: str  # specific name like "task_extraction", "performance_analysis"


@dataclass
class DataSourceRequest:
    """
    Generic data source request with query and metadata support.

    Provides flexible interface for data source providers to receive
    specific queries and contextual metadata for intelligent retrieval.
    """

    user_id: str | None
    requester: DataSourceRequester
    query: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


def create_data_source_request(
    state: "AgentState",
    requester: DataSourceRequester,
    query: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> DataSourceRequest:
    """
    Create a data source request from AgentState and requester information.

    Args:
        state: AgentState instance (TypedDict)
        requester: Information about the requesting component
        query: Optional specific query for the data source
        metadata: Optional metadata for provider-specific context

    Returns:
        DataSourceRequest with user context and query information
    """
    # Extract user ID from session context
    user_id = None
    try:
        from osprey.utils.config import get_session_info

        session_info = get_session_info()
        user_id = session_info.get("user_id")
    except Exception:
        # Log but don't fail - some contexts might not have session info
        pass

    return DataSourceRequest(
        user_id=user_id, requester=requester, query=query, metadata=metadata or {}
    )
