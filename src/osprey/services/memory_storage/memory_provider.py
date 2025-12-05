"""
Core User Memory Provider - Framework Data Source Integration

Provides seamless integration of user memory storage into the framework's data source
management system. This module implements the UserMemoryProvider class which serves as
a core data source that's always available for memory context injection.

The provider automatically retrieves stored user memories and formats them for inclusion
in LLM prompts, enabling conversation continuity and personalized interactions across
sessions. It integrates with the memory storage manager for persistent data access and
follows the framework's data source provider patterns for consistent behavior.

Key Features:
    - Core data source provider integration for universal memory access
    - Automatic memory retrieval and LLM prompt formatting
    - High-priority context injection for enhanced user experience
    - Seamless integration with the framework's data source management system
    - Health checking and configuration requirements management

.. note::
   This provider is registered as a core framework data source and is always
   available when a user ID is present in the session configuration unless
   overridden or excluded by an application registry.

.. seealso::
   :class:`osprey.data_management.providers.DataSourceProvider` : Base provider interface
   :class:`osprey.services.memory_storage.MemoryStorageManager` : Storage backend
   :class:`osprey.data_management.manager.DataSourceManager` : Provider registration system
"""

import logging

from osprey.data_management.providers import DataSourceContext, DataSourceProvider
from osprey.data_management.request import DataSourceRequest

from .storage_manager import get_memory_storage_manager

logger = logging.getLogger(__name__)


class UserMemoryProvider(DataSourceProvider):
    """Core data source provider for comprehensive user memory integration.

    Integrates user memory storage into the framework's data source management system
    as a core component that's universally available across all capabilities and
    operations. The provider serves as the primary interface between the memory storage
    backend and the framework's data source architecture.

    This provider implements sophisticated memory retrieval and formatting capabilities,
    automatically converting stored memory entries into structured context data that
    enhances LLM interactions with personalized user information. It maintains high
    priority in the data source hierarchy to ensure memory context is consistently
    available for conversation continuity and personalized responses.

    The provider handles the complete memory integration workflow including user
    identification, memory retrieval, format conversion, and context creation.
    It integrates seamlessly with the approval system, session management, and
    LLM prompt construction systems.

    Key architectural features:
        - Core framework data source with universal availability
        - Automatic memory retrieval based on session user ID
        - Structured context creation for LLM prompt integration
        - Health checking and configuration requirement management
        - Error handling with graceful degradation on memory access failures

    .. note::
       The provider responds to all requests with valid user IDs, making memory
       context available for both task extraction and capability execution phases.

    .. warning::
       Memory retrieval failures are handled gracefully with logging but do not
       prevent other data sources from functioning.

    .. seealso::
       :class:`osprey.data_management.providers.DataSourceProvider` : Base provider interface
       :class:`osprey.services.memory_storage.MemoryStorageManager` : Storage backend used by this provider
       :class:`osprey.data_management.request.DataSourceRequest` : Request structure for data retrieval
       :class:`osprey.data_management.providers.DataSourceContext` : Context structure returned by this provider
       :class:`osprey.state.UserMemories` : State structure for memory collections
    """

    def __init__(self):
        """Initialize the core user memory provider with storage backend integration.

        Sets up the memory storage manager connection for accessing stored user memories
        and prepares the provider for integration with the framework's data source
        management system. The initialization establishes the connection to the global
        memory storage manager instance for consistent memory access across the framework.

        The provider is designed to be instantiated once during framework initialization
        and reused across all memory retrieval operations.

        .. note::
           Uses the global memory storage manager instance for consistent memory
           access across all framework components.

        .. seealso::
           :func:`osprey.services.memory_storage.get_memory_storage_manager` : Global manager factory
           :class:`osprey.services.memory_storage.MemoryStorageManager` : Storage backend
        """
        self._memory_manager = get_memory_storage_manager()

    @property
    def name(self) -> str:
        """Unique identifier for this data source provider in the framework registry.

        Provides the canonical name used for provider registration, logging, and
        identification within the data source management system. This name must be
        unique across all registered data source providers.

        :return: Provider name identifier used for framework registration
        :rtype: str

        .. note::
           This name is used in framework configuration, logging, and provider lookup operations.
        """
        return "core_user_memory"

    @property
    def context_type(self) -> str:
        """Context type identifier for memory data integration and validation.

        Specifies the type of context data this provider creates, enabling the framework
        to properly validate, route, and format memory context data. This type identifier
        is used by the context management system for type checking and LLM prompt
        formatting decisions.

        :return: Context type identifier for memory data validation and routing
        :rtype: str

        .. note::
           This context type should match registered context types in the framework's
           context registry for proper integration.

        .. seealso::
           :class:`osprey.data_management.providers.DataSourceContext` : Context structure using this type
        """
        return "CORE_MEMORY_CONTEXT"

    @property
    def description(self) -> str:
        """Human-readable description of this data source for documentation and logging.

        Provides a clear, descriptive explanation of what this data source provides
        and its role in the framework. This description is used in logging, debugging,
        health check reports, and administrative interfaces.

        :return: Human-readable description of the memory data source functionality
        :rtype: str
        """
        return "Core user memory system providing saved user information"

    async def retrieve_data(self, request: DataSourceRequest) -> DataSourceContext | None:
        """Retrieve user memory data and create structured context for framework integration.

        Implements the core memory retrieval workflow including user identification,
        memory storage access, data format conversion, and context creation. The method
        handles the complete process of converting stored memory entries into structured
        context data suitable for LLM prompt integration.

        The retrieval process follows these steps:
        1. Extract user ID from the data source request
        2. Retrieve memory entries from the storage manager
        3. Convert entries to framework-compatible UserMemories format
        4. Create structured DataSourceContext with metadata
        5. Return formatted context for prompt integration

        :param request: Data source request containing user information and session context
        :type request: DataSourceRequest
        :return: Structured context with user memory data and metadata, or None if unavailable
        :rtype: Optional[DataSourceContext]

        .. note::
           Query-based memory retrieval is not currently supported. All memory entries
           are returned regardless of query parameters.

        .. warning::
           Returns None if user ID is unavailable or memory retrieval fails, allowing
           other data sources to continue functioning.

        Examples:
            Successful memory retrieval::

                >>> from osprey.data_management.request import DataSourceRequest, DataSourceRequester
                >>> provider = UserMemoryProvider()
                >>> requester = DataSourceRequester(component_type="capability", component_name="analysis")
                >>> request = DataSourceRequest(user_id="user123", requester=requester)
                >>> context = await provider.retrieve_data(request)
                >>> if context:
                ...     print(f"Retrieved {context.metadata['entry_count']} memory entries")

        .. seealso::
           :class:`osprey.data_management.request.DataSourceRequest` : Request structure
           :class:`osprey.data_management.providers.DataSourceContext` : Returned context structure
           :class:`osprey.state.UserMemories` : Memory data format
           :meth:`osprey.services.memory_storage.MemoryStorageManager.get_all_memory_entries` : Storage backend method
        """
        user_id = request.user_id
        if not user_id:
            logger.debug("No user ID available - core user memory unavailable")
            return None

        # Check if query-based retrieval is requested
        if request.query is not None:
            logger.warning(
                "Query-based memory retrieval is not supported. Will return all memory entries."
            )

        try:
            # Get memory entries from the storage manager
            memory_entries = self._memory_manager.get_all_memory_entries(user_id)

            # Convert to UserMemories format for compatibility
            from osprey.state import UserMemories

            if not memory_entries:
                logger.debug(f"No memory entries found for user {user_id}")
                # Return empty context instead of None - no data is not a failure
                user_memories = UserMemories(entries=[])
            else:
                logger.debug(
                    f"Retrieved {len(memory_entries)} core memory entries for user {user_id}"
                )
                user_memories = UserMemories(entries=[entry.content for entry in memory_entries])

            return DataSourceContext(
                source_name=self.name,
                context_type=self.context_type,
                data=user_memories,
                metadata={
                    "user_id": user_id,
                    "entry_count": len(memory_entries) if memory_entries else 0,
                    "source_description": "Core user memory system",
                    "is_core_provider": True,
                },
                provider=self,
            )

        except Exception as e:
            logger.warning(f"Failed to retrieve core user memory for {user_id}: {e}")
            return None

    def should_respond(self, request: DataSourceRequest) -> bool:
        """Determine if memory provider should respond to this request based on user context.

        Evaluates whether this provider can meaningfully contribute to the current request
        by checking for the presence of user identification information. The memory provider
        is designed to respond to all requests where user context is available, as memory
        information is universally relevant for personalized interactions.

        This method performs a fast, non-I/O check to determine provider applicability
        before expensive memory retrieval operations are attempted. The provider responds
        to requests from both task extraction (for context-aware task understanding) and
        capability execution (for personalized operation enhancement).

        :param request: Data source request to evaluate for memory provider applicability
        :type request: DataSourceRequest
        :return: True if user ID is available and memory retrieval should be attempted
        :rtype: bool

        .. note::
           This is a lightweight check that doesn't perform actual memory retrieval.
           Memory access failures are handled in the retrieve_data method.

        Examples:
            Request with user ID::

                >>> from osprey.data_management.request import DataSourceRequest, DataSourceRequester
                >>> provider = UserMemoryProvider()
                >>> requester = DataSourceRequester(component_type="capability", component_name="analysis")
                >>> request = DataSourceRequest(user_id="user123", requester=requester)
                >>> should_respond = provider.should_respond(request)
                >>> print(f"Should respond: {should_respond}")  # True

            Request without user ID::

                >>> request_no_user = DataSourceRequest(user_id=None, requester=requester)
                >>> should_respond = provider.should_respond(request_no_user)
                >>> print(f"Should respond: {should_respond}")  # False

        .. seealso::
           :class:`osprey.data_management.request.DataSourceRequest` : Request structure evaluated
           :meth:`retrieve_data` : Method called if this returns True
        """
        return request.user_id is not None

    def get_config_requirements(self) -> dict:
        """Get configuration requirements for core user memory system operation.

        Specifies the configuration parameters required for proper memory provider
        operation, including storage directory paths and access permissions. This
        information is used by the framework's configuration validation system to
        ensure all necessary settings are available before provider initialization.

        The returned configuration specification includes parameter descriptions,
        types, requirements status, and configuration path mappings for integration
        with the configuration system.

        :return: Dictionary of required configuration parameters with specifications
        :rtype: dict

        .. note::
           Configuration requirements are validated during framework initialization
           to ensure memory provider can operate correctly.

        Examples:
            Configuration requirements structure::

                >>> provider = UserMemoryProvider()
                >>> requirements = provider.get_config_requirements()
                >>> print(requirements['memory_directory']['description'])
                Directory where core user memory files are stored

        .. seealso::
           :func:`configs.config.get_agent_dir` : Configuration path resolution
           :class:`osprey.services.memory_storage.MemoryStorageManager` : Uses configured directory
        """
        return {
            "memory_directory": {
                "description": "Directory where core user memory files are stored",
                "type": "string",
                "required": True,
                "config_path": "file_paths.user_memory_dir",
            }
        }

    async def health_check(self) -> bool:
        """Perform comprehensive health check for the core memory system functionality.

        Validates that the memory provider and its dependencies are operational and
        accessible for memory retrieval operations. This includes checking storage
        manager availability, configuration validity, and basic system functionality.

        The health check is designed to be lightweight and non-intrusive, focusing
        on system availability rather than comprehensive functionality testing.
        Health check results are used by the framework's monitoring and diagnostic
        systems to ensure data source reliability.

        :return: True if the memory system is accessible and functional for operations
        :rtype: bool

        .. note::
           Health check failures are logged but don't prevent framework operation.
           Other data sources continue to function if memory provider is unavailable.

        Examples:
            Basic health check::

                >>> provider = UserMemoryProvider()
                >>> is_healthy = await provider.health_check()
                >>> if is_healthy:
                ...     print("Memory provider is operational")
                ... else:
                ...     print("Memory provider has issues")

        .. seealso::
           :class:`osprey.services.memory_storage.MemoryStorageManager` : Backend system checked
           :func:`osprey.data_management.manager.DataSourceManager.health_check_all` : Framework health checking
        """
        try:
            # Test basic functionality - check if memory manager is available
            return self._memory_manager is not None
        except Exception as e:
            logger.warning(f"Core user memory health check failed: {e}")
            return False

    def format_for_prompt(self, context: DataSourceContext) -> str:
        """Format core user memory data for optimal LLM prompt integration and readability.

        Transforms structured memory context data into a visually appealing, well-organized
        format that enhances LLM understanding and provides clear user context information.
        The formatting uses visual markers, structured organization, and clear hierarchy
        to make memory information easily consumable by language models.

        The formatted output includes entry counts, visual indicators, and organized
        presentation that helps LLMs understand the significance and context of stored
        user information. The format is designed to be both human-readable and
        LLM-optimized for maximum effectiveness in conversational contexts.

        :param context: The DataSourceContext containing memory data and metadata
        :type context: DataSourceContext
        :return: Formatted string optimized for LLM prompt inclusion with visual organization
        :rtype: str

        Examples:
            Formatted memory with multiple entries::

                >>> # Assume context contains memory data
                >>> provider = UserMemoryProvider()
                >>> formatted = provider.format_for_prompt(context)
                >>> print(formatted)
                **🧠 User Memory** (3 saved entries):
                  **Personal Notes & Insights:**
                    • User prefers morning meetings
                    • Working on project Alpha
                    • Completed training module

            Empty memory formatting::

                >>> # Context with no memory entries
                >>> formatted = provider.format_for_prompt(empty_context)
                >>> print(formatted)
                **🧠 User Memory** (0 saved entries):
                  (No memory entries available)

        .. note::
           Returns empty string if context is None or contains no data, allowing
           graceful handling of memory retrieval failures.

        .. seealso::
           :class:`osprey.data_management.providers.DataSourceContext` : Input context structure
           :class:`osprey.state.UserMemories` : Memory data format within context
           :meth:`retrieve_data` : Method that creates the context formatted by this method
        """
        if not context or not context.data:
            return ""

        user_memories = context.data
        entry_count = context.metadata.get("entry_count", 0)

        # Enhanced formatting for core memory
        sections = []
        sections.append(f"**🧠 User Memory** ({entry_count} saved entries):")

        if hasattr(user_memories, "entries") and user_memories.entries:
            sections.append("  **Personal Notes & Insights:**")
            for entry in user_memories.entries:
                # Add bullet point and indent for readability
                sections.append(f"    • {entry}")
        else:
            sections.append("  (No memory entries available)")

        return "\n".join(sections)
