"""
Memory Storage Service - Comprehensive User Memory Infrastructure

Provides complete user memory storage infrastructure including persistent file management,
data source integration, and structured memory operations for the framework. This service
implements a robust, scalable memory system that bridges user context storage with the
framework's data source management and capability execution systems.

The service consists of three primary components working together to provide seamless
memory operations:

1. **MemoryStorageManager**: File-based persistence backend with JSON storage
2. **UserMemoryProvider**: Data source integration for framework-wide memory access
3. **MemoryContent**: Structured data model for memory entries with timestamp context

Key Features:
    - Persistent JSON-based file storage with per-user organization
    - Framework data source integration for universal memory access
    - Structured memory models with timestamp context and LLM formatting
    - Comprehensive error handling and graceful degradation
    - Integration with approval systems and session management
    - Support for both programmatic access and LLM prompt formatting

The memory storage service is designed as a core framework component that's always
available when user context is present. It provides both direct storage operations
and data source provider capabilities for seamless integration with capabilities,
task extraction, and response generation systems.

Architecture Overview:
    - **Storage Layer**: MemoryStorageManager handles all file I/O and persistence
    - **Integration Layer**: UserMemoryProvider bridges storage with data source framework
    - **Data Layer**: MemoryContent provides structured, validated memory representation
    - **Access Layer**: Global factory functions provide consistent instance access

.. note::
   This service is registered as a core framework data source and is automatically
   available in all agent configurations when user identification is present.

.. warning::
   Memory storage contains persistent user data and should be configured according
   to appropriate data privacy and security policies.

.. seealso::
   :class:`osprey.data_management.providers.DataSourceProvider` : Base provider interface
   :class:`osprey.capabilities.memory.MemoryOperationsCapability` : Memory operations capability
   :class:`osprey.state.UserMemories` : Framework state integration
   :func:`configs.config.get_agent_dir` : Configuration system integration
"""

from .memory_provider import UserMemoryProvider
from .models import MemoryContent
from .storage_manager import MemoryStorageManager, get_memory_storage_manager

__all__ = [
    # Core storage layer
    "MemoryStorageManager",
    "get_memory_storage_manager",
    "MemoryContent",
    # Data source integration
    "UserMemoryProvider",
]
