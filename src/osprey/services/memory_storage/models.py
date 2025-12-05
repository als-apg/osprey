"""
Memory Storage Models

Data models for memory storage operations.
"""

from datetime import datetime

from pydantic import BaseModel, Field


class MemoryContent(BaseModel):
    """Structured memory content with timestamp for persistent user context storage.

    Represents a single memory entry containing user-specified content along with
    creation timestamp for temporal context. This model provides the fundamental
    data structure for all memory storage operations in the framework, ensuring
    consistent format and metadata across the memory storage system.

    The model integrates with Pydantic for automatic validation and serialization,
    supporting both JSON persistence and LLM prompt formatting. Each memory entry
    maintains temporal context through precise timestamps, enabling chronological
    memory retrieval and context-aware memory management.

    :param timestamp: Creation timestamp for the memory entry
    :type timestamp: datetime
    :param content: User content to be stored in memory
    :type content: str

    .. note::
       All memory entries are timestamped automatically during creation to maintain
       temporal context and support chronological memory organization.

    .. seealso::
       :class:`osprey.services.memory_storage.MemoryStorageManager` : Storage backend that manages these entries
       :class:`osprey.services.memory_storage.UserMemoryProvider` : Data source provider that retrieves these entries
       :class:`osprey.state.UserMemories` : State container for memory collections
       :meth:`format_for_llm` : LLM-optimized formatting method
    """

    timestamp: datetime = Field(description="The timestamp of the memory content")
    content: str = Field(description="The content that should be saved to memory")

    def format_for_llm(self) -> str:
        """Format memory content for optimal LLM consumption and prompt integration.

        Creates a standardized, human-readable format that combines timestamp context
        with memory content in a way that's optimized for LLM understanding and
        processing. The format provides temporal context while maintaining readability
        and consistent structure across all memory entries.

        The formatted output uses a bracketed timestamp format followed by the content,
        providing clear temporal markers that help LLMs understand the chronological
        context of stored information.

        :return: Formatted string with timestamp and content optimized for LLM prompts
        :rtype: str

        Examples:
            Basic memory formatting::

                >>> from datetime import datetime
                >>> entry = MemoryContent(
                ...     timestamp=datetime(2025, 1, 15, 14, 30),
                ...     content="User prefers morning meetings"
                ... )
                >>> formatted = entry.format_for_llm()
                >>> print(formatted)
                [2025-01-15 14:30] User prefers morning meetings

            Complex content formatting::

                >>> entry = MemoryContent(
                ...     timestamp=datetime.now(),
                ...     content="Working on project Alpha with team lead Sarah"
                ... )
                >>> print(entry.format_for_llm())
                [2025-01-15 15:45] Working on project Alpha with team lead Sarah

        .. note::
           The timestamp format (YYYY-MM-DD HH:MM) provides sufficient temporal
           resolution while maintaining readability in LLM prompts.

        .. seealso::
           :class:`osprey.services.memory_storage.UserMemoryProvider` : Uses this method for prompt formatting
           :meth:`osprey.services.memory_storage.MemoryStorageManager.get_user_memory` : Alternative formatting approach
        """
        return f"[{self.timestamp.strftime('%Y-%m-%d %H:%M')}] {self.content}"


__all__ = [
    "MemoryContent",
]
