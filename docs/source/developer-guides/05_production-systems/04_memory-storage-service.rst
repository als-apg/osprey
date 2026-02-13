==============
Memory Storage
==============

**What you'll build:** Simple file-based user memory system with data source integration

.. dropdown:: 📚 What You'll Learn
   :color: primary
   :icon: book

   **Key Concepts:**

   - Using :class:`MemoryStorageManager` and :func:`get_memory_storage_manager()` for memory operations
   - Implementing :class:`UserMemoryProvider` as a core data source provider
   - Working with :class:`MemoryContent` models for structured memory entries
   - Understanding automatic framework integration and ``core_user_memory`` provider
   - Managing persistent user context across sessions with file-based storage

   **Prerequisites:** Understanding of :doc:`02_data-source-integration` and basic JSON file operations

   **Time Investment:** 15-20 minutes for complete understanding

Overview
========

The Memory Storage Service provides basic user memory infrastructure for persistent storage of user context across sessions. It implements a simple file-based storage system with JSON persistence and framework data source integration.

**Core Features:**

- **File-based JSON Storage**: Per-user JSON files with basic persistence
- **Data Source Integration**: Automatic registration as ``core_user_memory`` provider
- **Simple Memory Model**: Timestamped entries with content
- **Framework Integration**: Works with existing context and approval systems

.. admonition:: Current Limitations
   :class: warning

   The built-in memory storage service is currently a basic implementation with known limitations:

   - **No advanced querying or search capabilities** - Simple retrieve-all approach only
   - **All memories retrieved together** - No selective retrieval by date, content, or tags
   - **Simple file-based backend only** - No database integration or advanced storage options
   - **No backup or advanced storage features** - Basic JSON persistence without redundancy

   For production systems requiring advanced memory features, consider implementing custom storage providers or extending the existing system.

Architecture
============

The system consists of three core components:

**1. MemoryStorageManager**
   Simple file-based persistence backend with JSON storage

**2. UserMemoryProvider**
   Data source integration for framework-wide memory access

**3. MemoryContent**
   Basic data model with timestamp and content fields

Implementation Guide
====================

Step 1: Configure Memory Storage
--------------------------------

Configure the memory directory in your application's ``config.yml``:

.. code-block:: yaml

   # config.yml - Memory Storage Configuration
   file_paths:
     user_memory_dir: "user_memory"

The framework automatically:

- Creates the directory if it doesn't exist
- Registers ``UserMemoryProvider`` as ``core_user_memory`` data source
- Makes memory available to all capabilities

Step 2: Use Memory in Capabilities
----------------------------------

Access memory through the storage manager:

.. code-block:: python

   """Basic Memory Usage in Capabilities"""

   from osprey.base import BaseCapability, capability_node
   from osprey.state import AgentState
   from osprey.context import ContextManager
   from osprey.services.memory_storage import get_memory_storage_manager, MemoryContent
   from datetime import datetime
   from typing import Dict, Any
   import logging

   logger = logging.getLogger(__name__)

   @capability_node
   class MemoryAwareCapability(BaseCapability):
       """Capability demonstrating basic memory integration."""

       def __init__(self):
           self.memory_manager = get_memory_storage_manager()

       async def execute(self, state: AgentState, context: ContextManager) -> Dict[str, Any]:
           """Execute with memory context."""

           user_id = state.user_id
           if not user_id:
               logger.warning("No user ID available - memory operations unavailable")
               return {"success": True, "memory_available": False}

           try:
               # Retrieve existing memories
               memories = self.memory_manager.get_all_memory_entries(user_id)
               logger.info(f"Retrieved {len(memories)} memories for user {user_id}")

               # Process with memory context
               result = self._process_with_memory(memories, context)

               # Store new memory if needed
               if result.get("new_insight"):
                   memory_entry = MemoryContent(
                       timestamp=datetime.now(),
                       content=result["new_insight"]
                   )
                   success = self.memory_manager.add_memory_entry(user_id, memory_entry)
                   logger.info(f"Stored new memory: {success}")

               return result

           except Exception as e:
               logger.error(f"Memory operation failed: {e}")
               return {"success": False, "error": str(e)}

       def _process_with_memory(self, memories, context):
           """Process capability logic with memory context."""
           # Extract relevant information from stored memories
           memory_context = [m.content for m in memories]

           return {
               "success": True,
               "memory_count": len(memories),
               "memory_context": memory_context,
               "new_insight": "User completed task successfully"
           }

Step 3: Access Memory via Data Sources
--------------------------------------

Memory is automatically available through the data source system:

.. code-block:: python

   """Accessing Memory Through Data Sources"""

   from osprey.data_management import get_data_source_manager, create_data_source_request, DataSourceRequester

   async def get_user_memory_context(state):
       """Retrieve memory through data source system."""

       # Create data source request
       requester = DataSourceRequester("capability", "example_capability")
       request = create_data_source_request(state, requester)

       # Get data source manager and retrieve context
       data_manager = get_data_source_manager()
       result = await data_manager.retrieve_all_context(request, timeout_seconds=10.0)

       # Extract memory context
       memory_context = result.context_data.get("core_user_memory")

       if memory_context:
           user_memories = memory_context.data
           entry_count = memory_context.metadata.get("entry_count", 0)
           logger.info(f"Retrieved {entry_count} memory entries via data source")
           return user_memories
       else:
           logger.info("No memory context available")
           return None

Core API Reference
==================

MemoryStorageManager
--------------------

.. code-block:: python

   class MemoryStorageManager:
       """Simple file-based memory manager."""

       def get_user_memory(self, user_id: str) -> str:
           """Get formatted memory string for user."""

       def get_all_memory_entries(self, user_id: str) -> List[MemoryContent]:
           """Get all memory entries as MemoryContent objects."""

       def add_memory_entry(self, user_id: str, memory_content: MemoryContent) -> bool:
           """Add new memory entry for user."""

MemoryContent Model
-------------------

.. code-block:: python

   class MemoryContent(BaseModel):
       """Memory entry with timestamp and content."""
       timestamp: datetime
       content: str

       def format_for_llm(self) -> str:
           """Format as '[YYYY-MM-DD HH:MM] content'"""

UserMemoryProvider
------------------

Automatically registered data source provider:

- **Name**: ``core_user_memory``
- **Context Type**: ``CORE_MEMORY_CONTEXT``
- **Responds when**: User ID is available
- **Returns**: ``UserMemories`` object with entry list

Testing Memory Integration
==========================

.. code-block:: python

   """Test Memory Storage Integration"""

   from osprey.services.memory_storage import get_memory_storage_manager, MemoryContent
   from datetime import datetime

   async def test_memory_operations():
       """Test basic memory operations."""

       manager = get_memory_storage_manager()
       test_user_id = "test_user_123"

       # Test memory addition
       test_memory = MemoryContent(
           timestamp=datetime.now(),
           content="Test memory entry"
       )

       success = manager.add_memory_entry(test_user_id, test_memory)
       assert success, "Memory addition should succeed"

       # Test memory retrieval
       memories = manager.get_all_memory_entries(test_user_id)
       assert len(memories) > 0, "Should retrieve stored memories"
       assert any(m.content == test_memory.content for m in memories), "Should find test memory"

       # Test formatted output
       formatted = manager.get_user_memory(test_user_id)
       assert test_memory.content in formatted, "Formatted memory should contain test content"

       print("✅ Memory storage tests passed")

Configuration Options
=====================

The memory system supports minimal configuration:

.. code-block:: yaml

   # config.yml
   file_paths:
     user_memory_dir: "user_memory"  # Directory for memory files

**Configuration Details:**

- **user_memory_dir**: Directory where user memory JSON files are stored
- Files are named ``{sanitized_user_id}.json``
- Directory created automatically if it doesn't exist

Troubleshooting
===============

**Common Issues:**

**Issue**: Memory not persisting between sessions
   - **Cause**: User ID not consistent across sessions
   - **Solution**: Verify session management provides stable user identification

**Issue**: Memory not available through data sources
   - **Cause**: UserMemoryProvider not registered
   - **Solution**: Check framework registry initialization

**Issue**: File permission errors
   - **Cause**: Memory directory not writable
   - **Solution**: Verify directory permissions and path accessibility

**Debugging Memory Issues:**

.. code-block:: python

   # Test memory manager availability
   from osprey.services.memory_storage import get_memory_storage_manager
   manager = get_memory_storage_manager()
   print(f"Memory manager available: {manager is not None}")

   # Check memory directory
   from osprey.utils.config import get_agent_dir
   memory_dir = get_agent_dir('user_memory_dir')
   print(f"Memory directory: {memory_dir}")

   # Test data source registration
   from osprey.data_management import get_data_source_manager
   data_manager = get_data_source_manager()
   provider = data_manager.get_provider("core_user_memory")
   print(f"Memory provider registered: {provider is not None}")

Future Enhancements
===================

The current implementation provides basic memory functionality. Planned enhancements include:

- Advanced querying and search capabilities
- Semantic memory retrieval with embeddings
- Memory categorization and organization
- Backup and archival features
- Alternative storage backends

**Related Documentation:**

- :doc:`02_data-source-integration` - Data source system integration
- :doc:`../03_core-framework-systems/02_context-management-system` - Context management patterns
- :doc:`../../api_reference/03_production_systems/04_memory-storage` - Complete API reference
