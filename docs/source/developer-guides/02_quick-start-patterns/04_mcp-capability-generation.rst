=========================
MCP Capability Generation
=========================

.. admonition:: Prototype Feature
   :class: warning

   This is a **prototype feature** under active development. The API and generated code structure may change in future releases.

**Generate Osprey capabilities from Model Context Protocol (MCP) servers** using automated code generation. This guide shows you how to create capabilities that integrate with MCP servers, using the weather demo as a practical example.

.. dropdown:: **Prerequisites**
   :color: info
   :icon: list-unordered

   **Required:**

   - Basic understanding of Osprey capabilities and the registry system
   - An initialized Osprey project (any project will work)
   - MCP adapter libraries installed:

     .. code-block:: bash

        # For running demo MCP servers
        pip install fastmcp

        # Install MCP adapters and LangGraph
        pip install langchain-mcp-adapters langgraph

        # Install LangChain provider based on your config
        # For Anthropic (Claude):
        pip install langchain-anthropic

        # For OpenAI or CBORG:
        pip install langchain-openai


   **Recommended:**

   - Familiarity with the :doc:`../../getting-started/control-assistant` tutorial (we use the control assistant project as our example)
   - Review :doc:`01_building-your-first-capability` to understand capability fundamentals

What You'll Learn
=================

- How to generate a test MCP server for development
- How to generate Osprey capabilities from MCP servers
- Integrating generated capabilities into your project

.. note::

   This guide uses a control assistant project as an example. The commands work with any Osprey project - just replace `my-control-assistant` with your project name. We strongly **recommend using Claude Haiku 4.5** for best results with capability generation.

End-to-End MCP Integration
==========================

This tutorial walks you through generating an MCP server and creating an Osprey capability from it.

Step 1: Generate MCP Server
----------------------------

Create a demo MCP server for testing:

.. code-block:: bash

   # Generate weather demo server (default port 3001)
   osprey generate mcp-server --name weather_demo

   # Or generate on custom port
   osprey generate mcp-server --name weather_demo --port 3002

**Included Tools:**

The generated server includes three weather-related demo tools:

- ``get_current_weather`` - Get current weather conditions
- ``get_forecast`` - Get weather forecast for upcoming days
- ``get_weather_alerts`` - Get active weather alerts and warnings

Step 2: Run the MCP Server
---------------------------

Start the generated MCP server:

.. code-block:: bash

   python weather_demo_server.py  # or python weather_demo_server.py --port 3002

.. dropdown:: **Expected Output**
   :icon: terminal

   .. code-block:: text

      ======================================================================
      Weather Demo MCP Server
      ======================================================================

        Server URL: http://localhost:3001
        SSE Endpoint: http://localhost:3001/sse

      Available Tools:

        1. get_current_weather
           Get current weather conditions for a location
           Parameters:
             • location (string, required)
               City name or coordinates
             • units (string, optional)
               Temperature units (celsius/fahrenheit)

        2. get_forecast
           Get weather forecast for upcoming days
           Parameters:
             • location (string, required)
               City name or coordinates
             • days (integer, optional)
               Number of forecast days (1-7)
             • units (string, optional)
               Temperature units (celsius/fahrenheit)

        3. get_weather_alerts
           Get active weather alerts and warnings for a location
           Parameters:
             • location (string, required)
               City name or coordinates
             • severity (string, optional)
               Filter by alert severity (all/severe/moderate/minor)

      Next Steps:
        1. Keep this server running
        2. In another terminal: osprey generate capability --from-mcp http://localhost:3001 --name weather_demo

Step 3: Generate Capability from Server
----------------------------------------

**In a new terminal**, from your control assistant project:

.. code-block:: bash

   cd my-control-assistant

   # Generate capability from running MCP server
   osprey generate capability --from-mcp http://localhost:3001 --name weather_demo

.. dropdown:: **Expected Output**
   :icon: terminal

   .. code-block:: text

      🎨 Generating MCP Capability

        Capability: weather_demo
        Server: WeatherDemo
        Mode: Real MCP (http://localhost:3001)
        Output: capabilities/weather_demo.py

        ✓ Registry initialized

      📡 Step 1: Discovering MCP tools...
      Connecting to MCP server: http://localhost:3001
      ✓ Discovered 3 tools
        ✓ Found 3 tools
          • get_current_weather
          • get_forecast
          • get_weather_alerts

      🤖 Step 2: Generating guides with LLM...

      🤖 Analyzing tools with LLM...
         Using orchestrator model: cborg/anthropic/claude-haiku
      ✓ Guides generated
        ✓ Generated 11 classifier examples
        ✓ Generated 5 orchestrator examples

      📝 Step 3: Generating capability code...
        ✓ Code generated

      💾 Step 4: Writing output file...
        ✓ Written: capabilities/weather_demo.py (23,057 bytes)

      ======================================================================
      ✅ SUCCESS! MCP Capability Generated
      ======================================================================

      What was created:
        ✓ Capability class: weather_demo
        ✓ MCP client integration
        ✓ Classifier guide with 11 examples
        ✓ Orchestrator guide with 5 examples
        ✓ Context class for results
        ✓ Error handling
        ✓ Registry registration snippet

      Next Steps:
        1. Review: capabilities/weather_demo.py
        2. Customize the context class based on your data structure
        3. Add to your registry.py (see snippet at bottom of file)
        4. Test with: osprey chat


      Registry Integration:
        Found registry: /path/to/my-control-assistant/src/my_control_assistant/registry.py

      ? Add this capability to your registry automatically? Yes

      Preview of changes:

      Capability Registration:
                      CapabilityRegistration(
                          name="weather_demo",
                          module_path="my_control_assistant.capabilities.weather_demo",
                          class_name="WeatherDemoCapability",
                          description="WeatherDemo operations via MCP server",
                          provides=["WEATHERDEMO_RESULTS"],
                          requires=[]
                      ),

      Context Class Registration:
                      ContextClassRegistration(
                          context_type="WEATHERDEMO_RESULTS",
                          module_path="my_control_assistant.capabilities.weather_demo",
                          class_name="WeatherDemoResultsContext"
                      ),

      ? Apply these changes to registry.py? Yes

        ✓ Updated registry.py
        Backup saved to: registry.py.bak

        ℹ️  Capability is now registered! Test with: osprey chat

The generator:

1. **Discovers tools** from the MCP server
2. **Analyzes tools** using LLM to generate classifier/orchestrator examples
3. **Generates code** with complete capability implementation
4. **Offers to update registry** automatically (interactive prompt)

Generated file: ``capabilities/weather_demo.py``

.. dropdown:: **Advanced Options** (Optional)
   :color: info
   :icon: tools

   The automatic registration handles everything needed for the tutorial. These advanced options are available if you need them:

   .. tab-set::

      .. tab-item:: Customizing Context Class

         **For this tutorial:** The generated minimal context class works fine for basic testing.

         **For production/advanced workflows:** This customization becomes **CRITICAL** - well-designed context classes are the foundation of multi-capability workflows. When other Osprey capabilities need to consume MCP results (Python analysis/visualization, memory storage, conditional logic, multi-step workflows), the context class must expose structured data with proper types and access patterns. Without this, downstream capabilities cannot generate accurate code or access the data reliably.

         The generated code currently dumps the entire ReAct agent response into the context:

         .. code-block:: python

            # Generated capability code (lines 267-275 in weather_demo.py)
            # Extract final result
            final_message = response["messages"][-1]
            result_content = final_message.content if hasattr(final_message, 'content') else str(final_message)

            # Format as context - THIS IS TOO GENERIC FOR PRODUCTION
            context = WeatherDemoResultsContext(
                tool="react_agent",
                results={"final_output": result_content, "full_response": response},
                description=f"WeatherDemo ReAct agent: {task_objective}"
            )

         **Production pattern - Customize both capability and context:**

         1. **In the capability code**: Parse MCP tool results to extract structured data
         2. **In the context class**: Define proper fields and access patterns

         .. code-block:: python

            # Example: Production-ready weather context
            class WeatherResultsContext(CapabilityContext):
                """Structured weather data for downstream capabilities."""

                CONTEXT_TYPE: ClassVar[str] = "WEATHER_DATA"
                CONTEXT_CATEGORY: ClassVar[str] = "EXTERNAL_DATA"

                location: str
                temperature: float
                temperature_unit: str
                conditions: str
                humidity: int
                wind_speed: float
                timestamp: datetime

                def get_access_details(self, key: str) -> Dict[str, Any]:
                    """LLM-optimized access for Python code generation."""
                    return {
                        "location": self.location,
                        "data_structure": "Structured weather observation with typed fields",
                        "access_pattern": f"context.WEATHER_DATA.{key}.temperature (float in {self.temperature_unit})",
                        "available_fields": ["temperature", "conditions", "humidity", "wind_speed", "timestamp"],
                        "example_usage": f"temp = context.WEATHER_DATA.{key}.temperature  # {self.temperature} {self.temperature_unit}"
                    }

         This enables downstream capabilities to:

         - Generate accurate Python code: ``temp = context.WEATHER_DATA.current.temperature``
         - Access structured data without parsing strings
         - Perform calculations and visualizations with proper types

         See :doc:`../03_core-framework-systems/02_context-management-system` for comprehensive guidance on context class design patterns.

      .. tab-item:: Prompt Customization

         **Important:** The generated classifier and orchestrator examples are created by an LLM analyzing only the MCP server's tool schemas. While helpful as a starting point, they are likely not production-ready.

         **Why customization matters:**

         - The LLM has no knowledge of your specific application domain
         - Examples are generic and based purely on tool signatures
         - Your app may have specific patterns, terminology, or edge cases
         - Integration with other capabilities requires domain-specific context

         **What to review and customize:**

         1. **Classifier Examples** (in ``_create_classifier_guide()``)

            - Review the positive examples - do they match your users' actual queries?
            - Add domain-specific terminology and patterns
            - Include edge cases specific to your application
            - Consider interactions with other capabilities

            .. code-block:: python

               # Generated example (generic)
               ClassifierExample(
                   query="What's the weather like in London?",
                   result=True,
                   reason="Direct request for current weather..."
               )

               # Production example (domain-specific)
               ClassifierExample(
                   query="Should we delay the telescope observation tonight?",
                   result=True,
                   reason="Observatory operations depend on weather conditions - activate to check forecast and alerts"
               )

         2. **Orchestrator Examples** (in ``_create_orchestrator_guide()``)

            - Ensure context_keys match your naming conventions
            - Verify task_objectives align with how your users phrase requests
            - Add examples showing integration with other capabilities
            - Include multi-step workflow patterns if relevant

            .. code-block:: python

               # Generic generated example
               OrchestratorExample(
                   step=PlannedStep(
                       context_key="current_weather_london",
                       task_objective="Get current weather for London",
                       ...
                   )
               )

               # Domain-specific example (astronomy observatory)
               OrchestratorExample(
                   step=PlannedStep(
                       context_key="observation_conditions_tonight",
                       task_objective="Check current weather and 12-hour forecast for observatory location to assess observation quality",
                       ...
                   )
               )

         3. **Activation Criteria** (classifier instructions)

            - The generated keywords are based on tool names
            - Add domain-specific terms your users actually use
            - Refine when NOT to activate based on your app's other capabilities

         **Testing strategy:**

         1. Start with generated examples (good enough for initial testing)
         2. Collect real user queries during testing
         3. Identify misclassifications or suboptimal planning
         4. Add domain-specific examples addressing those patterns
         5. Iterate based on actual usage

         The auto-generated prompts give you a running start, not a finish line. Budget time for prompt refinement in your production deployment plan.

      .. tab-item:: Registry & Config Integration

         **Manual Registry Setup**

         If you declined automatic registry integration, add to your ``registry.py``:

         .. code-block:: python

            from osprey.registry.base import CapabilityRegistration, ContextClassRegistration

            # In your registry provider's get_registry_config():
            capabilities=[
                # ... existing capabilities ...
                CapabilityRegistration(
                    name="weather_mcp",
                    module_path="my_control_assistant.capabilities.weather_mcp",
                    class_name="WeatherMcpCapability",
                    provides=["WEATHER_MCP_RESULTS"],
                    requires=[]
                ),
            ],
            context_classes=[
                # ... existing context classes ...
                ContextClassRegistration(
                    context_type="WEATHER_MCP_RESULTS",
                    module_path="my_control_assistant.capabilities.weather_mcp",
                    class_name="WeatherMcpResultsContext"
                ),
            ]

         **ReAct Agent Model Configuration**

         If you declined automatic config integration, add a capability-specific model to your ``config.yml``:

         .. code-block:: yaml

            models:
              weather_demo_react:  # Dedicated model for weather_demo MCP capability
                provider: anthropic  # or openai, cborg, etc.
                model_id: claude-haiku-4-5-20251001  # or claude-sonnet, gpt-4o, etc.
                max_tokens: 4096

Step 4: Test Your Capability
-----------------------------

**Your control assistant can now query weather!** With the MCP server running, test the integrated capability:

.. code-block:: bash

   # Make sure MCP server is still running, then:
   osprey chat

Try this example query:

.. code-block:: text

   👤 You: What's the weather right now?

**Result:**

.. code-block:: text

   🤖 # Current Weather Information Report

   ## Summary
   Successfully retrieved current weather data for **San Francisco** as of November 15, 2025 at 14:30 UTC.

   ## Key Weather Data

   | Metric | Value |
   |--------|-------|
   | **Temperature** | 18°C (feels like 16°C) |
   | **Conditions** | Partly Cloudy |
   | **Humidity** | 65% |
   | **Wind Speed** | 12 km/h from Northwest |
   | **Pressure** | 1013 mb |
   | **Visibility** | 10 km |
   | **UV Index** | 5 (Moderate) |

   ## Location Details
   - **City:** San Francisco
   - **Latitude:** 37.7749°
   - **Longitude:** -122.4194°

   ## Analysis
   The weather conditions in San Francisco are **pleasant** with mild temperatures
   and partly cloudy skies. The moderate UV index suggests some sun protection would
   be advisable if spending extended time outdoors, though the scattered cloud cover
   provides some natural protection.

   **Data Quality:** ✓ Complete - All requested information successfully retrieved
   with standard temperature units (Celsius) as specified.

.. dropdown:: **How It Works - Complete Workflow Details**
   :color: info
   :icon: workflow

   Here's what happens behind the scenes when your assistant processes the weather query:

   **🔹 Stage 1: Task Extraction**

   .. code-block:: text

      INFO  Task_Extraction:  * Extracted: 'Retrieve current weather information...'

   The framework extracts the user's intent into a structured task.

   **🔹 Stage 2: Capability Classification**

   .. code-block:: text

      INFO  Classifier: Classifying task: Retrieve current weather information
      INFO  Classifier:  >>> Capability 'memory' >>> False
      INFO  Classifier:  >>> Capability 'channel_finding' >>> False
      INFO  Classifier:  >>> Capability 'python' >>> False
      INFO  Classifier:  >>> Capability 'time_range_parsing' >>> False
      INFO  Classifier:  >>> Capability 'channel_read' >>> False
      INFO  Classifier:  >>> Capability 'channel_write' >>> False
      INFO  Classifier:  >>> Capability 'archiver_retrieval' >>> False
      INFO  Classifier:  >>> Capability 'weather_demo' >>> True ✓
      INFO  Classifier: 3 capabilities required: ['respond', 'clarify', 'weather_demo']

   The classifier evaluates each capability and **activates weather_demo** based on your generated classifier guide.

   **🔹 Stage 3: Orchestration Planning**

   .. code-block:: text

      INFO  Orchestrator: Creating execution plan with orchestrator LLM
      INFO  Orchestrator: ==================================================
      INFO  Orchestrator:  << Step 1
      INFO  Orchestrator:  << ├───── id: 'current_weather_default'
      INFO  Orchestrator:  << ├─── node: 'weather_demo'
      INFO  Orchestrator:  << ├─── task: 'Retrieve current weather information for the
      INFO  Orchestrator:  <<           default location with standard temperature units'
      INFO  Orchestrator:  << └─ inputs: '[]'
      INFO  Orchestrator:  << Step 2
      INFO  Orchestrator:  << ├───── id: 'user_response'
      INFO  Orchestrator:  << ├─── node: 'respond'
      INFO  Orchestrator:  << ├─── task: 'Provide the user with the current weather
      INFO  Orchestrator:  <<           information retrieved from the weather demo capability'
      INFO  Orchestrator:  << └─ inputs: '[{'WEATHERDEMO_RESULTS': 'current_weather_default'}]'
      INFO  Orchestrator: ✅ Final execution plan ready with 2 steps

   The orchestrator creates a 2-step plan using your generated orchestrator guide:

   1. Call weather_demo capability with task objective
   2. Use respond capability to format results for user

   **🔹 Stage 4: MCP Capability Execution**

   .. code-block:: text

      INFO  Router: Executing step 1/2 - capability: weather_demo
      INFO  Weather_Demo: Connected to MCP server: http://localhost:3001/sse
      INFO  Weather_Demo: Loaded 3 tools from MCP server
      INFO  Weather_Demo: ReAct agent initialized
      INFO  Weather_Demo: ReAct agent completed task

   Your generated capability:

   - Connects to the MCP server
   - Initializes its ReAct agent with available tools
   - Autonomously selects and calls appropriate MCP tools
   - Returns results as ``WEATHERDEMO_RESULTS`` context

   **🔹 Stage 5: Response Generation**

   .. code-block:: text

      INFO  Router: Executing step 2/2 - capability: respond
      INFO  Respond: Generated response for: 'Provide the user with the current
                     weather information retrieved from the weather demo capability'

   The respond capability formats the weather data into a user-friendly report.

**Congratulations!** 🎉 You've successfully:

- Generated a demo MCP server
- Created an Osprey capability from the MCP server
- Integrated it into your control assistant
- Tested end-to-end weather queries

Your control assistant now demonstrates the full MCP integration pattern, which you can adapt for any MCP-compatible service.

Removing Capabilities
=====================

If you need to remove a capability, Osprey provides an automated removal command that cleans up all associated files and configurations.

.. code-block:: bash

   # Remove capability interactively (recommended)
   osprey remove capability --name weather_demo

   # Force removal without confirmation
   osprey remove capability --name weather_demo --force

**What Gets Removed:**

- **Registry entries**: Capability and context class registrations from ``registry.py``
- **Config model**: The ``{capability_name}_react`` model configuration from ``config.yml``
- **Capability file**: The capability Python file
- **Automatic backups**: ``.bak`` files are created before modifications

.. seealso::

   - :doc:`01_building-your-first-capability` - Understanding capability fundamentals
   - :doc:`../03_core-framework-systems/02_context-management-system` - Context class design patterns
   - :doc:`../03_core-framework-systems/03_registry-and-discovery` - Registry system details
   - `MCP Specification <https://modelcontextprotocol.io/>`_ - Learn more about Model Context Protocol
   - `FastMCP Documentation <https://github.com/jlowin/fastmcp>`_ - Building MCP servers in Python

