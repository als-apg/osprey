======================
Core Framework Systems
======================

.. toctree::
   :maxdepth: 2
   :caption: Core Framework Systems
   :hidden:

   01_state-management-architecture
   02_context-management-system
   03_registry-and-discovery
   04_prompt-customization
   05_message-and-execution-flow
   06_configuration-architecture
   07_built-in-capabilities

.. dropdown:: What You'll Learn
   :color: primary
   :icon: book

   **Advanced Framework Internals:**

   - LangGraph-native state management with selective persistence strategies
   - Pydantic-based context objects with intelligent caching and serialization
   - Convention-based registry patterns with lazy loading and component discovery
   - Domain-specific prompt builders with dependency injection patterns
   - Complete message processing pipeline from Gateway entry to response generation
   - Single-file YAML configuration with flat structure and environment variable resolution

   **Prerequisites:** Completion of Quick Start Patterns and understanding of LangGraph concepts

   **Target Audience:** Framework developers and advanced capability authors building sophisticated agentic systems

Master the sophisticated internal systems that enable reliable, type-safe agent development. These core systems provide the foundation for building production-ready conversational agents with proper state management, data sharing, and component orchestration.

System Architecture
===================

The Core Framework Systems implement a **LangGraph-Native, Type-Safe Architecture** with five interconnected components:

.. grid:: 1 1 2 2
   :gutter: 3

   .. grid-item-card:: 🏗️ State Management
      :link: 01_state-management-architecture
      :link-type: doc
      :class-header: bg-primary text-white
      :class-body: text-center
      :shadow: md

      Selective persistence strategy with performance optimization and comprehensive lifecycle management.

   .. grid-item-card:: 🔄 Context Management
      :link: 02_context-management-system
      :link-type: doc
      :class-header: bg-success text-white
      :class-body: text-center
      :shadow: md

      Pydantic-based context objects with automatic serialization and intelligent caching.

.. grid:: 1 1 2 2
   :gutter: 3

   .. grid-item-card:: 📋 Registry & Discovery
      :link: 03_registry-and-discovery
      :link-type: doc
      :class-header: bg-info text-white
      :class-body: text-center
      :shadow: md

      Convention-based loading with lazy initialization and type-safe component access.

   .. grid-item-card:: 💬 Prompt Customization
      :link: 04_prompt-customization
      :link-type: doc
      :class-header: bg-secondary text-white
      :class-body: text-center
      :shadow: md

      Domain-specific prompt builders with dependency injection and debugging tools.

.. grid:: 1 1 2 2
   :gutter: 3

   .. grid-item-card:: 🔀 Message & Execution Flow
      :link: 05_message-and-execution-flow
      :link-type: doc
      :class-header: bg-warning text-white
      :class-body: text-center
      :shadow: md

      Complete message processing from Gateway entry to response generation.

   .. grid-item-card:: ⚙️ Configuration Architecture
      :link: 06_configuration-architecture
      :link-type: doc
      :class-header: bg-dark text-white
      :class-body: text-center
      :shadow: md

      Single-file YAML configuration with self-contained architecture and environment integration.

.. grid:: 1 1 2 2
   :gutter: 3

   .. grid-item-card:: 🧩 Built-in Capabilities
      :link: 07_built-in-capabilities
      :link-type: doc
      :class-header: bg-primary text-white
      :class-body: text-center
      :shadow: md

      Reference for all framework-native capabilities: channel ops, archiver, Python, time parsing, and memory.

System Integration
==================

These systems work together to provide a cohesive development experience:

.. tab-set::

   .. tab-item:: Data Flow

      How information moves through the framework:

      .. code-block:: python

         # State provides the foundation
         state = StateManager.create_fresh_state(user_input)

         # Context enables data sharing
         context = ContextManager(state)
         channel_data = context.get_context('CHANNEL_ADDRESSES', 'beam_current')

         # Registry provides component access
         registry = get_registry()
         capability = registry.get_capability('channel_read')

         # Message flow coordinates execution
         result = await capability.execute(state)

   .. tab-item:: Performance Strategy

      Optimization through intelligent design:

      .. code-block:: python

         # State: Only context persists across conversations
         capability_context_data: Dict[str, Dict[str, Dict[str, Any]]]  # Persists
         execution_step_results: Dict[str, Any]                        # Resets

         # Context: Object caching and efficient serialization
         context = ContextManager(state)  # Cached object reconstruction

         # Registry: Lazy loading and singleton patterns
         initialize_registry()  # One-time component discovery
         registry = get_registry()  # Singleton access

   .. tab-item:: Development Patterns

      Common implementation patterns:

      .. code-block:: python

         # Capability with all systems integration
         @capability_node  # Registry registration
         class MyCapability(BaseCapability):
             @staticmethod
             async def execute(state: AgentState, **kwargs) -> Dict[str, Any]:
                 # Context access
                 context = ContextManager(state)
                 data = context.get_context('INPUT_DATA', 'key')

                 # Processing logic
                 result = process_data(data)

                 # Context storage
                 output = OutputData(results=result)
                 return StateManager.store_context(
                     state, 'OUTPUT_DATA', 'processed', output
                 )
