===============================
Hello World Tutorial
===============================

This tutorial builds a simple weather agent to get you from zero to a working Osprey
application quickly. You'll learn the essentials: project structure, capability
implementation, context classes, and running your agent.

We use a single capability with straightforward logic to keep things minimal while
you learn the framework basics. The :doc:`conceptual-tutorial` and
:doc:`control-assistant` demonstrate more complex patterns when you're ready
to scale up.

What You'll Build
=================

A "Hello World Weather" agent is a simple agent that:

* Responds to natural language weather queries
* These queries uses a mock API for realistic weather data.
* By this we demonstrate the complete capability → context → response flow
* Shows framework integration patterns

By the end of this guide, you'll have a working agent that responds to queries like "What's the weather in Prague?" with temperature and conditions data.

.. dropdown:: **Prerequisites**
   :color: info
   :icon: list-unordered

   **Required:**

   - Python 3.11+ with virtual environment
   - Osprey framework installed via ``pip install osprey-framework``
   - API key from your chosen provider (we recommend Claude Haiku 4.5, but any OpenAI-compatible provider works including institutional services)

   If you haven't installed the framework yet, follow the :doc:`installation guide <installation>`.

   **Optional but Recommended:** Basic understanding of Python and async/await patterns.

Step 1: Create the Project
---------------------------

.. tab-set::

   .. tab-item:: Interactive Mode (Recommended)

      The easiest way to create your project is using the interactive menu:

      .. code-block:: bash

         osprey

      This launches an interactive terminal UI that will:

      1. Guide you through template selection (choose ``hello_world_weather``)
      2. Help you select an AI provider and model (we recommend **Claude Haiku 4.5**)
      3. Automatically detect and configure API keys
      4. Create a ready-to-use project

      Just follow the prompts, and you'll have a complete project set up in minutes!

   .. tab-item:: Direct CLI Command

      If you prefer direct commands or are automating project creation:

      .. code-block:: bash

         osprey init weather-agent --template hello_world_weather
         cd weather-agent

      This is perfect for scripts, automation, or when you already know exactly what you want.

Both methods create identical project structures. Use whichever fits your workflow.

**Generated Project Structure**

Either method generates a complete, self-contained project with the following structure:

.. code-block::

   weather-agent/
   ├── src/
   │   └── weather_agent/
   │       ├── __init__.py
   │       ├── mock_weather_api.py         # Mock data source (no external APIs)
   │       ├── context_classes.py          # Data models for weather information
   │       ├── registry.py                 # Component registration
   │       ├── framework_prompts.py        # Domain-specific prompt customizations
   │       └── capabilities/
   │           ├── __init__.py
   │           └── current_weather.py      # Weather retrieval logic
   ├── config.yml                          # Model & provider configuration
   └── .env.example                        # API key template

.. admonition:: Want to see it in action first?
   :class: tip

   If you're the type who likes to play with the toy before reading the manual, jump straight to :ref:`Step 8: Run Your Agent <hello-world-deploy-test>` to get your agent running in minutes! You can always come back here to understand how everything works under the hood.

This tutorial will walk you through understanding how each component works and how they integrate together to create a complete AI agent application.

Step 2: The Mock Data Source
----------------------------

The ``mock_weather_api.py`` file provides a deterministic weather data provider that eliminates external API dependencies while demonstrating the framework's capability integration patterns.

.. code-block:: python

   """
   Simple Mock Weather API
   """

   import random
   from datetime import datetime
   from dataclasses import dataclass

   @dataclass
   class CurrentWeatherReading:
       """Simple weather data model."""
       location: str
       temperature: float
       conditions: str
       timestamp: datetime

   class SimpleWeatherAPI:
       """
       Mock weather API that returns weather data for any location string.
       """

       def get_current_weather(self, location: str) -> CurrentWeatherReading:
           """Get simple current weather for a location."""
           # Implementation details below...

.. dropdown:: Complete Mock API Implementation

   Full implementation of the mock weather service (`view template on GitHub <https://github.com/als-apg/osprey/blob/main/src/osprey/templates/apps/hello_world_weather/mock_weather_api.py>`_)

   .. code-block:: python

      """
      Very simple mock weather API for quick setup.
      Accepts any location string and returns randomized weather data.

      The weather API returns only the type safe data model for the current weather reading.
      """
      import random
      from datetime import datetime
      from dataclasses import dataclass

      @dataclass
      class CurrentWeatherReading:
          """Simple weather data model."""
          location: str
          temperature: float  # Celsius
          conditions: str
          timestamp: datetime

      class SimpleWeatherAPI:
          """
          Very simple mock weather API for quick setup.
          Accepts any location string and returns randomized weather data.

          The weather API returns only the type safe data model for the current weather reading.
          """

          # Weather condition options for random selection
          ALL_CONDITIONS = [
              "Sunny", "Partly Cloudy", "Cloudy", "Overcast", "Foggy",
              "Rainy", "Drizzle", "Thunderstorms", "Snow", "Windy", "Clear"
          ]

          def get_current_weather(self, location: str) -> CurrentWeatherReading:
              """Get simple current weather for a location."""

              # Generate random weather data
              temperature = random.randint(0, 35)  # Temperature range: 0-35°C
              conditions = random.choice(self.ALL_CONDITIONS)

              return CurrentWeatherReading(
                  location=location,  # Preserve exact location string
                  temperature=float(temperature),
                  conditions=conditions,
                  timestamp=datetime.now()
              )

      # Global API instance
      weather_api = SimpleWeatherAPI()

Step 3: Define the Context Class
---------------------------------

Context classes provide structured data storage and enable seamless integration between your agent's capabilities. Define a context class file (we'll call it ``context_classes.py``) to specify how weather information is stored and accessed throughout the framework.

.. admonition:: Requirements

   All context classes must inherit from ``CapabilityContext`` and implement the following required methods:

**Class Structure:**

.. code-block:: python

    class CurrentWeatherContext(CapabilityContext):
        """Context for current weather conditions."""

        # Context type and category identifiers
        CONTEXT_TYPE: ClassVar[str] = "CURRENT_WEATHER"
        CONTEXT_CATEGORY: ClassVar[str] = "LIVE_DATA"

        # Your data fields (must be json serializable)
        location: str = Field(description="Location name")
        temperature: float = Field(description="Temperature in Celsius")
        conditions: str = Field(description="Weather conditions")
        timestamp: datetime = Field(description="When data was retrieved")

**Required Method 1: get_access_details()**

Provides structured access information for LLM consumption. This method is used when LLMs need to write Python code to access this context type:

.. code-block:: python

        def get_access_details(self, key: str) -> Dict[str, Any]:
            """Provide access details for LLM consumption."""
            return {
                "location": self.location,
                "temperature": self.temperature,
                "conditions": self.conditions,
                "temperature_formatted": f"{self.temperature}°C",
                "access_pattern": f"context.{self.CONTEXT_TYPE}.{key}.temperature, context.{self.CONTEXT_TYPE}.{key}.conditions",
                "example_usage": f"The temperature in {self.location} is {{context.{self.CONTEXT_TYPE}.{key}.temperature}}°C with {{context.{self.CONTEXT_TYPE}.{key}.conditions}} conditions",
                "available_fields": ["location", "temperature", "conditions", "timestamp"]
            }

**Required Method 2: get_summary()**

Provides human-readable summaries for user interfaces and debugging:

.. code-block:: python

        def get_summary(self, key: str) -> dict:
            """Get human-readable summary for this weather context."""
            return {
                "summary": f"Weather in {self.location} on {self.timestamp.strftime('%Y-%m-%d')}: {self.temperature}°C, {self.conditions}",
            }

.. dropdown:: Complete Weather Context Implementation

   Full context class showing all required methods (`view context class on GitHub <https://github.com/als-apg/osprey/blob/main/src/osprey/templates/apps/hello_world_weather/context_classes.py.j2>`_)

   .. code-block:: python

      """
      Hello World Weather Context Classes - Quick Start Version

      These classes serve as a simple data structure for exchange of the weather information between the capabilities and the orchestrator.

      It is important to note that the context classes are not used to determine the location, but rather to determine if the task requires current weather information for a specific location.
      The location is determined by the orchestrator based on the user query and the context of the task.

      The context classes are used to store the weather information in a structured format that can be easily used by the capabilities and the orchestrator.
      """

      from datetime import datetime
      from typing import Dict, Any, Optional, ClassVar
      from pydantic import Field
      from osprey.context.base import CapabilityContext

      class CurrentWeatherContext(CapabilityContext):
          """Simple context for current weather conditions."""

          CONTEXT_TYPE: ClassVar[str] = "CURRENT_WEATHER"
          CONTEXT_CATEGORY: ClassVar[str] = "LIVE_DATA"

          # Basic weather data
          location: str = Field(description="Location name")
          temperature: float = Field(description="Temperature in Celsius")
          conditions: str = Field(description="Weather conditions description")
          timestamp: datetime = Field(description="Timestamp of weather data")

          @property
          def context_type(self) -> str:
              """Return the context type identifier."""
              return self.CONTEXT_TYPE

          def get_access_details(self, key: str) -> Dict[str, Any]:
              """Provide access details for LLM consumption."""
              return {
                  "location": self.location,
                  "current_temp": f"{self.temperature}°C",
                  "conditions": self.conditions,
                  "access_pattern": f"context.{self.CONTEXT_TYPE}.{key}.temperature, context.{self.CONTEXT_TYPE}.{key}.conditions",
                  "example_usage": f"The temperature in {self.location} is {{context.{self.CONTEXT_TYPE}.{key}.temperature}}°C with {{context.{self.CONTEXT_TYPE}.{key}.conditions}} conditions",
                  "available_fields": ["location", "temperature", "conditions", "timestamp"]
              }

          def get_summary(self, key: str) -> dict:
              """Get human-readable summary for this weather context."""
              return {
                  "summary": f"Weather in {self.location} on {self.timestamp.strftime('%Y-%m-%d')}: {self.temperature}°C, {self.conditions}",
              }

Step 4: Building the Weather Capability
----------------------------------------

Capabilities are the **business logic units** that perform specific tasks. Our weather capability demonstrates the essential patterns for data retrieval, context storage, and framework integration.

**4.1: The @capability_node Decorator**

The ``@capability_node`` decorator validates required class components and creates a LangGraph-compatible wrapper function with full infrastructure support:

.. code-block:: python

   @capability_node
   class CurrentWeatherCapability(BaseCapability):
       """Get current weather conditions for a location."""

       # Required class attributes for registry configuration
       name = "current_weather"
       description = "Get current weather conditions for a location"
       provides = ["CURRENT_WEATHER"]
       requires = []

.. admonition:: Key Insight

   The ``provides`` field tells the framework what context types this capability generates. The ``requires`` field tells the framework what context types this capability needs to run.

**4.2: Core Business Logic**

The ``execute()`` method contains your main business logic, which you could call the 'tool' in agentic terms. Here's the weather retrieval:

.. code-block:: python

      async def execute(self) -> Dict[str, Any]:
          """Execute weather retrieval."""
          # Get unified logger with automatic streaming
          logger = self.get_logger()

          try:
              # Extract location from user's query
              query = self.get_task_objective()
              location = await _parse_location_from_query(query)

              # Get weather data
              logger.status(f"Getting weather for {location}...")
              weather = weather_api.get_current_weather(location)

              # Create context object
              context = CurrentWeatherContext(
                  location=weather.location,
                  temperature=weather.temperature,
                  conditions=weather.conditions,
                  timestamp=weather.timestamp
              )

              # Store context and return state updates
              logger.success(f"Weather retrieved: {location} - {weather.temperature}°C")
              return self.store_output_context(context)

          except Exception as e:
              logger.error(f"Weather retrieval error: {e}")
              raise

.. admonition:: Key Steps

   1. **Logger Setup** - Get unified logger using self.get_logger()
   2. **Task Retrieval** - Get task of current execution step
   3. **Location Extraction** - Parse user query to find location
   4. **Data Retrieval** - Call your API/service to get actual data
   5. **Context Creation** - Convert raw data to structured context object
   6. **Context Storage** - Store context so other capabilities and LLM can access it

**4.3: Essential Supporting Methods**

Every capability needs basic error handling and retry policies:

.. code-block:: python

       @staticmethod
       def classify_error(exc: Exception, context: dict) -> ErrorClassification:
           """Classify errors for retry decisions."""
           if isinstance(exc, (ConnectionError, TimeoutError)):
               return ErrorClassification(
                   severity=ErrorSeverity.RETRIABLE,
                   user_message="Weather service timeout, retrying...",
                   metadata={"technical_details": str(exc)}
               )

           return ErrorClassification(
               severity=ErrorSeverity.CRITICAL,
               user_message=f"Weather service error: {str(exc)}",
               metadata={
                   "technical_details": f"Error: {type(exc).__name__}"
               }
           )

       @staticmethod
       def get_retry_policy() -> Dict[str, Any]:
           """Retry policy for weather data retrieval."""
           return {
               "max_attempts": 3,
               "delay_seconds": 0.5,
               "backoff_factor": 1.5
           }

.. admonition:: Framework Benefits

   The Framework Handles Everything Else: Error routing, retry logic, user messaging, and execution flow are automatically managed by the framework infrastructure.

.. _hello-world-orchestrator-guide:

**4.4: Orchestrator Guide**

The orchestrator guide teaches the LLM how to plan execution steps and use your capability effectively:

.. code-block:: python

    def _create_orchestrator_guide(self) -> Optional[OrchestratorGuide]:
    """Guide the orchestrator on how to use this capability."""
    example = OrchestratorExample(
              step=PlannedStep(
                  context_key="current_weather",
                  capability="current_weather",
                  task_objective="Get current weather conditions for the specified location",
                  expected_output=registry.context_types.CURRENT_WEATHER,
                  success_criteria="Current weather data retrieved with temperature and conditions",
            inputs=[]
              ),
              scenario_description="Getting current weather for a location",
              notes=f"Output stored as {registry.context_types.CURRENT_WEATHER} with live weather data."
          )

          return OrchestratorGuide(
              instructions=f"""**When to plan "current_weather" steps:**
      - When users ask for current weather conditions
      - For real-time weather information requests
      - When location-specific current conditions are needed

      **Output: {registry.context_types.CURRENT_WEATHER}**
      - Contains: location, temperature, conditions, timestamp
      - Available for immediate display or further analysis

      **Location Handling:**
      - Extracts locations from natural language queries
      - Handles variations and common phrasings naturally""",
        examples=[example],
        order=5
    )

.. admonition:: For Complex Capabilities

   When building more sophisticated capabilities with multiple steps, dependencies, or complex planning logic, providing comprehensive orchestrator examples becomes crucial. The orchestrator uses these examples to understand when and how to integrate your capability into multi-step execution plans.

.. _hello-world-classifier-guide:

**4.5: Classifier Guide**

The classifier guide teaches the LLM when to activate your capability based on user queries:

.. code-block:: python

      def _create_classifier_guide(self) -> Optional[TaskClassifierGuide]:
        """Guide the classifier on when to activate this capability."""
          return TaskClassifierGuide(
              instructions="Determine if the task requires current weather information for a specific location.",
              examples=[
                  ClassifierExample(
                      query="What's the weather like in San Francisco right now?",
                      result=True,
                      reason="Request asks for current weather conditions in a specific location."
                  ),
                  ClassifierExample(
                      query="How's the weather today?",
                      result=True,
                      reason="Current weather request, though location may need to be inferred."
                  ),
                  ClassifierExample(
                      query="What was the weather like last week?",
                      result=False,
                      reason="Request is for historical weather data, not current conditions."
                  ),
                  ClassifierExample(
                    query="What tools do you have?",
                    result=False,
                    reason="Request is for tool information, not weather."
                  ),
              ],
              actions_if_true=ClassifierActions()
          )

.. admonition:: Quality Examples Matter

   The classifier's accuracy depends heavily on the quality and diversity of your examples. Include edge cases, ambiguous queries, and clear negative examples to help the LLM make better classification decisions.

.. dropdown:: Complete Current Weather Capability Implementation

   Full capability showing all required methods and patterns (`view template on GitHub <https://github.com/als-apg/osprey/blob/main/src/osprey/templates/apps/hello_world_weather/capabilities/current_weather.py.j2>`_)

   .. code-block:: python

      """
      Current Weather Capability

      Simple capability to get current weather conditions for a location.
      """

      from typing import Dict, Any, Optional

      from osprey.base import (
          BaseCapability, capability_node,
          OrchestratorGuide, OrchestratorExample, PlannedStep,
          ClassifierActions, ClassifierExample, TaskClassifierGuide
      )
      from osprey.base.errors import ErrorClassification, ErrorSeverity
      from osprey.registry import get_registry

      from weather_agent.context_classes import CurrentWeatherContext
      from weather_agent.mock_weather_api import weather_api

      registry = get_registry()

      @capability_node
      class CurrentWeatherCapability(BaseCapability):
          """Get current weather conditions for a location."""

          # Required class attributes for registry configuration
          name = "current_weather"
          description = "Get current weather conditions for a location"
          provides = ["CURRENT_WEATHER"]
          requires = []

          async def execute(self) -> Dict[str, Any]:
              """Execute weather retrieval."""
              # Get unified logger with automatic streaming support
              logger = self.get_logger()

              try:
                  # Extract location from user's query
                  query = self.get_task_objective()
                  location = await _parse_location_from_query(query)

                  # Get weather data
                  logger.status(f"Getting weather for {location}...")
                  weather = weather_api.get_current_weather(location)

                  # Create context object
                  context = CurrentWeatherContext(
                      location=weather.location,
                      temperature=weather.temperature,
                      conditions=weather.conditions,
                      timestamp=weather.timestamp
                  )

                  # Store context and return
                  logger.status(f"Weather retrieved: {location} - {weather.temperature}°C")
                  return self.store_output_context(context)

              except Exception as e:
                  logger.error(f"Weather retrieval error: {e}")
                  raise

          @staticmethod
          def classify_error(exc: Exception, context: dict) -> ErrorClassification:
              """Classify errors for retry decisions."""
              if isinstance(exc, (ConnectionError, TimeoutError)):
                  return ErrorClassification(
                      severity=ErrorSeverity.RETRIABLE,
                      user_message="Weather service timeout, retrying...",
                      metadata={"technical_details": str(exc)}
                  )

              return ErrorClassification(
                  severity=ErrorSeverity.CRITICAL,
                  user_message=f"Weather service error: {str(exc)}",
                  metadata={"technical_details": f"Error: {type(exc).__name__}"}
              )

          @staticmethod
          def get_retry_policy() -> Dict[str, Any]:
              """Retry policy for weather data retrieval."""
              return {
                  "max_attempts": 3,
                  "delay_seconds": 0.5,
                  "backoff_factor": 1.5
              }

          def _create_orchestrator_guide(self) -> Optional[OrchestratorGuide]:
              """Guide the orchestrator on how to use this capability."""
              example = OrchestratorExample(
                  step=PlannedStep(
                      context_key="current_weather",
                      capability="current_weather",
                      task_objective="Get current weather conditions for the specified location",
                      expected_output=registry.context_types.CURRENT_WEATHER,
                      success_criteria="Current weather data retrieved with temperature and conditions",
                      inputs=[]
                  ),
                  scenario_description="Getting current weather for a location",
                  notes=f"Output stored as {registry.context_types.CURRENT_WEATHER} with live weather data."
              )

              return OrchestratorGuide(
                  instructions=f"""**When to plan "current_weather" steps:**
          - When users ask for current weather conditions
          - For real-time weather information requests
          - When location-specific current conditions are needed

          **Output: {registry.context_types.CURRENT_WEATHER}**
          - Contains: location, temperature, conditions, timestamp
          - Available for immediate display or further analysis

          **Location Handling:**
          - Extracts locations from natural language queries
          - Handles variations and common phrasings naturally""",
                  examples=[example],
                  order=5
              )

          def _create_classifier_guide(self) -> Optional[TaskClassifierGuide]:
              """Guide the classifier on when to activate this capability."""
              return TaskClassifierGuide(
                  instructions="Determine if the task requires current weather information for a specific location.",
                  examples=[
                      ClassifierExample(
                          query="What's the weather like in San Francisco right now?",
                          result=True,
                          reason="Request asks for current weather conditions in a specific location."
                      ),
                      ClassifierExample(
                          query="How's the weather today?",
                          result=True,
                          reason="Current weather request, though location may need to be inferred."
                      ),
                      ClassifierExample(
                          query="What was the weather like last week?",
                          result=False,
                          reason="Request is for historical weather data, not current conditions."
                      ),
                      ClassifierExample(
                          query="What tools do you have?",
                          result=False,
                          reason="Request is for tool information, not weather."
                      ),
                  ],
                  actions_if_true=ClassifierActions()
              )


Step 5: Domain Adaptation
--------------------------

Weather applications need to synthesize information from multi-turn conversations where location, time, and weather concerns are mentioned across different exchanges. The generated project includes ``framework_prompts.py`` with weather-specific examples that teach the framework how to identify and combine these domain-specific elements.

**The Problem**

Consider this conversation:

.. code-block:: text

   User: I'm planning a trip and need to check weather patterns
   Agent: I can help with weather patterns for your trip! Which destination are you considering?
   User: I'm thinking about New York
   Agent: Great! When are you planning to visit New York?
   User: Next weekend, and I'm particularly concerned about rain

The final message should extract a complete weather query: **"Get weather forecast for New York for next weekend with focus on precipitation."** This requires understanding that:

- Location ("New York") was mentioned two exchanges ago
- Time ("next weekend") was just specified
- "Concerned about rain" → emphasize precipitation data (weather-domain knowledge)

Without domain-specific examples, the framework might not recognize that "concerned about rain" should translate into prioritizing precipitation information - this is weather-specific context that doesn't apply to other domains.

**The Solution**

The project includes a custom prompt builder that extends the framework's task extraction with 8 weather-specific examples:

.. code-block:: python

   class WeatherTaskExtractionPromptBuilder(DefaultTaskExtractionPromptBuilder):
       """Weather-specific task extraction with domain examples."""

       def __init__(self):
           super().__init__(include_default_examples=False)
           self._add_weather_examples()

       def get_role(self) -> str:
           return "You are a weather assistant task extraction specialist..."

       def _add_weather_examples(self):
           # Location carry-forward example
           self.examples.append(TaskExtractionExample(
               messages=[
                   MessageUtils.create_user_message("What's the weather in San Francisco?"),
                   MessageUtils.create_assistant_message("Tonight in SF..."),
                   MessageUtils.create_user_message("What about tomorrow?"),
               ],
               expected_output=ExtractedTask(
                   task="Get weather forecast for San Francisco for tomorrow",
                   depends_on_chat_history=True
               )
           ))
           # ... 7 more examples

.. dropdown:: Complete Framework Prompts Implementation

   Full custom prompt builder with 8 weather-specific examples (`view template on GitHub <https://github.com/als-apg/osprey/blob/main/src/osprey/templates/apps/hello_world_weather/framework_prompts.py.j2>`_)

   .. code-block:: python

      """Weather Agent Framework Prompt Customizations."""

      import textwrap
      from osprey.prompts.defaults import DefaultTaskExtractionPromptBuilder, TaskExtractionExample, ExtractedTask
      from osprey.state import MessageUtils, UserMemories


      class WeatherTaskExtractionPromptBuilder(DefaultTaskExtractionPromptBuilder):
          """Weather-specific task extraction prompt builder."""

          def __init__(self):
              """Initialize with weather-specific examples only."""
              super().__init__(include_default_examples=False)
              self._add_weather_examples()

          def get_role(self) -> str:
              """Get the weather-specific role definition."""
              return "You are a weather assistant task extraction specialist that analyzes conversations to extract actionable weather-related tasks."

          def get_instructions(self) -> str:
              """Get the weather-specific task extraction instructions."""
              return textwrap.dedent("""
              Your job is to:
              1. Understand what the user is asking for in the context of weather information
              2. Extract a clear, actionable task related to weather queries
              3. Determine if the task depends on chat history context
              4. Determine if the task depends on user memory

              ## Weather-Specific Guidelines:
              - Create self-contained task descriptions executable without conversation context
              - Resolve temporal references ("tomorrow", "next week") to specific time periods
              - Carry forward location references from previous messages
              - Extract specific locations, times, and weather parameters from previous responses
              - Understand weather-specific concerns ("rain" → precipitation, "good for walking" → temperature + conditions)
              - Set depends_on_chat_history=True if task references previous messages
              - Set depends_on_user_memory=True only when task needs specific information from user memory
              """).strip()

          def _add_weather_examples(self):
              """Add weather-specific examples."""

              # Example 1: Multi-turn progressive refinement - weather planning context
              self.examples.append(TaskExtractionExample(
                  messages=[
                      MessageUtils.create_user_message("I'm planning a trip and need to check weather patterns"),
                      MessageUtils.create_assistant_message(
                          "I can help with weather patterns for your trip! Which destination are you considering?"
                      ),
                      MessageUtils.create_user_message("I'm thinking about New York"),
                      MessageUtils.create_assistant_message(
                          "Great! When are you planning to visit New York? I can check the forecast."
                      ),
                      MessageUtils.create_user_message("Next weekend, and I'm particularly concerned about rain"),
                  ],
                  user_memory=UserMemories(entries=[]),
                  expected_output=ExtractedTask(
                      task="Get weather forecast for New York for next weekend with focus on precipitation probability",
                      depends_on_chat_history=True,
                      depends_on_user_memory=False
                  )
              ))

              # Example 2: Location carry-forward with temporal change
              self.examples.append(TaskExtractionExample(
                  messages=[
                      MessageUtils.create_user_message("What's the weather like in San Francisco tonight?"),
                      MessageUtils.create_assistant_message(
                          "Tonight in San Francisco, expect partly cloudy skies with temperatures "
                          "around 13°C. Light winds from the west at 12 km/h."
                      ),
                      MessageUtils.create_user_message("What about tomorrow?"),
                  ],
                  user_memory=UserMemories(entries=[]),
                  expected_output=ExtractedTask(
                      task="Get weather forecast for San Francisco for tomorrow",
                      depends_on_chat_history=True,
                      depends_on_user_memory=False
                  )
              ))

              # Example 3: Location switching
              self.examples.append(TaskExtractionExample(
                  messages=[
                      MessageUtils.create_user_message("How's the weather in Prague?"),
                      MessageUtils.create_assistant_message(
                          "The current weather in Prague shows 8°C with rainy conditions. "
                          "Humidity is at 85% with moderate winds."
                      ),
                      MessageUtils.create_user_message("What about in Paris?"),
                  ],
                  user_memory=UserMemories(entries=[]),
                  expected_output=ExtractedTask(
                      task="Get current weather conditions for Paris",
                      depends_on_chat_history=True,
                      depends_on_user_memory=False
                  )
              ))

              # Example 4: Implicit location reference ("there")
              self.examples.append(TaskExtractionExample(
                  messages=[
                      MessageUtils.create_user_message("What's the current weather in Paris?"),
                      MessageUtils.create_assistant_message(
                          "Paris is currently experiencing clear weather at 16°C with "
                          "light winds and good visibility."
                      ),
                      MessageUtils.create_user_message("How about tomorrow there?"),
                  ],
                  user_memory=UserMemories(entries=[]),
                  expected_output=ExtractedTask(
                      task="Get weather forecast for Paris for tomorrow",
                      depends_on_chat_history=True,
                      depends_on_user_memory=False
                  )
              ))

              # Example 5: Weather-specific comparison (domain knowledge)
              self.examples.append(TaskExtractionExample(
                  messages=[
                      MessageUtils.create_user_message("What's the weather in San Francisco?"),
                      MessageUtils.create_assistant_message(
                          "San Francisco currently has clear skies at 18°C with light winds."
                      ),
                      MessageUtils.create_user_message("What about Prague?"),
                      MessageUtils.create_assistant_message(
                          "Prague is experiencing rainy conditions at 10°C with moderate winds."
                      ),
                      MessageUtils.create_user_message("Which one is better for an outdoor walk?"),
                  ],
                  user_memory=UserMemories(entries=[]),
                  expected_output=ExtractedTask(
                      task="Compare San Francisco and Prague weather conditions to determine which is better for outdoor walking (considering temperature, precipitation, and wind)",
                      depends_on_chat_history=True,
                      depends_on_user_memory=False
                  )
              ))

              # Example 6: Simple temporal follow-up
              self.examples.append(TaskExtractionExample(
                  messages=[
                      MessageUtils.create_user_message("What's the current weather in Paris?"),
                      MessageUtils.create_assistant_message(
                          "The current weather in Paris is 15°C with overcast skies and light rain. "
                          "Wind speed is 12 km/h from the northwest."
                      ),
                      MessageUtils.create_user_message("What was it like 3 hours ago?"),
                  ],
                  user_memory=UserMemories(entries=[]),
                  expected_output=ExtractedTask(
                      task="Get historical weather conditions for Paris from 3 hours ago",
                      depends_on_chat_history=True,
                      depends_on_user_memory=False
                  )
              ))

              # Example 7: Conversational query
              self.examples.append(TaskExtractionExample(
                  messages=[
                      MessageUtils.create_user_message("Hi, what weather information can you provide?"),
                      MessageUtils.create_assistant_message(
                          "I can help you check current weather conditions and forecasts for various cities! "
                          "I can tell you about temperature, precipitation, wind, and general conditions."
                      ),
                      MessageUtils.create_user_message("Which cities do you cover?"),
                  ],
                  user_memory=UserMemories(entries=[]),
                  expected_output=ExtractedTask(
                      task="List the cities available for weather queries",
                      depends_on_chat_history=False,
                      depends_on_user_memory=False
                  )
              ))

              # Example 8: Fresh request (no conversation context)
              self.examples.append(TaskExtractionExample(
                  messages=[
                      MessageUtils.create_user_message("Can you check the weather in New York?"),
                  ],
                  user_memory=UserMemories(entries=[]),
                  expected_output=ExtractedTask(
                      task="Get current weather conditions for New York",
                      depends_on_chat_history=False,
                      depends_on_user_memory=False
                  )
              ))


Step 6: Understanding the Registry
-----------------------------------

The registry system is how the framework discovers and manages your application's components. It uses a simple pattern where your application provides a configuration that tells the framework what capabilities and context classes you've defined.

.. admonition:: Registry Purpose

   The registry enables loose coupling and lazy loading - the framework can discover your components without importing them until needed, improving startup performance and modularity. The framework provides ``extend_framework_registry()`` helper that lets you declare which framework capabilities your agent needs via ``include_capabilities`` and add your custom ones alongside them.

**The Registry Pattern**

The registry uses a class-based provider pattern. Here's the structure:

.. code-block:: python

   from osprey.registry import (
       extend_framework_registry,
       CapabilityRegistration,
       ContextClassRegistration,
       FrameworkPromptProviderRegistration,
       ExtendedRegistryConfig,
       RegistryConfigProvider
   )

   class WeatherAgentRegistryProvider(RegistryConfigProvider):
       """Registry provider for the weather application."""

       def get_registry_config(self) -> ExtendedRegistryConfig:
           return extend_framework_registry(
               # Only include these framework capabilities
               include_capabilities=[
                   "respond", "clarify", "memory",
                   "time_range_parsing", "python", "state_manager",
               ],
               # Weather-specific capability
               capabilities=[
                  CapabilityRegistration(
                      name="current_weather",
                      module_path="weather_agent.capabilities.current_weather",
                      class_name="CurrentWeatherCapability",
                      description="Get current weather conditions",
                      provides=["CURRENT_WEATHER"],
                      requires=[]
                  )
               ],
               context_classes=[
                   ContextClassRegistration(
                       context_type="CURRENT_WEATHER",
                       module_path="weather_agent.context_classes",
                       class_name="CurrentWeatherContext"
                   )
               ],
               framework_prompt_providers=[
                   FrameworkPromptProviderRegistration(
                       module_path="weather_agent.framework_prompts",
                       prompt_builders={
                           "task_extraction": "WeatherTaskExtractionPromptBuilder"
                       }
                   )
               ]
           )


.. admonition:: Advanced Registry Patterns
   :class: tip

   The ``include_capabilities`` parameter provides a clean whitelist of which framework capabilities your agent needs. For advanced use cases requiring complete control over component registration, see :doc:`Registry and Discovery <../developer-guides/03_core-framework-systems/03_registry-and-discovery>` for alternative patterns including Standalone Mode and :ref:`custom component exclusion <excluding-overriding-components>`.

.. dropdown:: Complete Registry Implementation

   Complete registry file using ``extend_framework_registry()`` (`view template on GitHub <https://github.com/als-apg/osprey/blob/main/src/osprey/templates/apps/hello_world_weather/registry.py.j2>`_)

   .. code-block:: python

      """
      Weather Application Registry

     Registry configuration using extend_framework_registry() to automatically
      include framework components and add weather-specific components.
     """

     from osprey.registry import (
         extend_framework_registry,
         CapabilityRegistration,
         ContextClassRegistration,
          FrameworkPromptProviderRegistration,
         ExtendedRegistryConfig,
         RegistryConfigProvider
     )

     class WeatherAgentRegistryProvider(RegistryConfigProvider):
        """Registry provider for weather tutorial application."""

        def get_registry_config(self) -> ExtendedRegistryConfig:
             """Provide registry configuration with framework + weather components."""
             return extend_framework_registry(
                 # Only include these framework capabilities
                 include_capabilities=[
                     "respond",
                     "clarify",
                     "memory",
                     "time_range_parsing",
                     "python",
                     "state_manager",
                 ],

                 # Add weather-specific capability
                 capabilities=[
                     CapabilityRegistration(
                         name="current_weather",
                         module_path="weather_agent.capabilities.current_weather",
                         class_name="CurrentWeatherCapability",
                         description="Get current weather conditions for a location",
                         provides=["CURRENT_WEATHER"],
                         requires=[]
                     )
                 ],

                 # Add weather-specific context class
                 context_classes=[
                     ContextClassRegistration(
                         context_type="CURRENT_WEATHER",
                         module_path="weather_agent.context_classes",
                         class_name="CurrentWeatherContext"
                     )
                  ],

                  # Add weather-specific prompt customizations
                  framework_prompt_providers=[
                      FrameworkPromptProviderRegistration(
                          module_path="weather_agent.framework_prompts",
                          prompt_builders={
                              "task_extraction": "WeatherTaskExtractionPromptBuilder"
                          }
                      )
                 ]
              )

   The ``include_capabilities`` parameter whitelists only the framework capabilities your agent needs, keeping your weather agent focused. Your weather-specific components are added alongside them.

Step 7: Application Configuration
----------------------------------

The generated project includes a complete, self-contained ``config.yml`` file in the project root with all necessary settings pre-configured. Everything is in one place - no need to edit multiple configuration files.

**Key Configuration Sections:**

The ``config.yml`` includes:

.. code-block:: yaml

   # Project name
   project_name: "weather-agent"

   # Registry discovery - tells framework where your application code is
   registry_path: src/weather_agent/registry.py

   # Model configurations
   models:
     orchestrator:
       provider: anthropic
       model_id: claude-haiku-4-20251015

.. admonition:: Model Recommendation
   :class: tip

   **We recommend Claude Haiku 4.5** for the best experience. It provides excellent performance, low latency, and works very well with the framework's structured outputs. However, any OpenAI-compatible provider works - including institutional services like LBNL CBorg, Stanford AI Playground, or ANL Argo.

**Customization:**

You can customize the configuration by editing ``config.yml``:

1. **Change model providers** - Update ``provider`` fields under ``models``
2. **API keys** - Set in ``.env`` file (not in ``config.yml``)


.. _hello-world-deploy-test:

Step 8: Run Your Agent
-----------------------

Now that you understand the components, let's run and test your agent!

**1. Configure Your API Key**

Set up your ``.env`` file with your API key:

.. code-block:: bash

   # Copy the template
   cp .env.example .env

   # Edit .env and add your API key
   # ANTHROPIC_API_KEY=your-key-here     # If using Anthropic (recommended)
   # CBORG_API_KEY=your-key-here         # If using LBNL CBorg
   # STANFORD_API_KEY=your-key-here      # If using Stanford AI Playground
   # ARGO_API_KEY=your-key-here          # If using ANL Argo
   # OPENAI_API_KEY=your-key-here        # If using OpenAI
   # GOOGLE_API_KEY=your-key-here        # If using Google

.. admonition:: Model Recommendation
   :class: tip

   **We recommend Claude Haiku 4.5** for the best experience. It provides excellent performance, low latency, and works exceptionally well with the framework's structured outputs. However, the framework works with any provider - including institutional services like LBNL CBorg, Stanford AI Playground, ANL Argo, or commercial providers like OpenAI and Google.

.. dropdown:: **Where do I get an API key?**
   :color: info
   :icon: key

   Choose your provider for instructions on obtaining an API key:

   **Anthropic (Claude)** - Recommended

   1. Visit: https://console.anthropic.com/
   2. Sign up or log in with your account
   3. Navigate to 'API Keys' in the settings
   4. Click 'Create Key' and name your key
   5. Copy the key (shown only once!)

   **OpenAI (GPT)**

   1. Visit: https://platform.openai.com/api-keys
   2. Sign up or log in to your OpenAI account
   3. Add billing information if not already set up
   4. Click '+ Create new secret key'
   5. Name your key and copy it (shown only once!)

   **Google (Gemini)**

   1. Visit: https://aistudio.google.com/app/apikey
   2. Sign in with your Google account
   3. Click 'Create API key'
   4. Select a Google Cloud project or create a new one
   5. Copy the generated API key

   **LBNL CBorg**

   1. Visit: https://cborg.lbl.gov
   2. As a Berkeley Lab employee, click 'Request API Key'
   3. Create an API key ($50/month per user allocation)
   4. Copy the key provided

   **Stanford AI Playground**

   1. Visit: https://uit.stanford.edu/service/ai-api-gateway
   2. Requires Stanford University affiliation
   3. Go to 'Get Started' → 'Request the creation of a new API key'
   4. Log in with your Stanford credentials and complete the form
   5. Once approved, copy the API key from the notification email

   **ANL Argo**

   1. Requires Argonne National Laboratory affiliation
   2. Argo uses your ANL username (automatically obtained from the $USER environment variable)
   3. Contact your ANL IT department for access to the Argo proxy service
   4. Base URL: https://argo-bridge.cels.anl.gov

   **Ollama (Local Models)**

   Ollama runs locally and does not require an API key. Simply install Ollama and ensure it's running.

**2. Start the Chat Interface**

Launch the interactive chat using :doc:`osprey chat <../developer-guides/02_quick-start-patterns/00_cli-reference>`:

.. code-block:: bash

   osprey chat

**3. Test Your Agent**

Ask weather-related questions:

.. code-block:: text

   You: What's the weather in San Francisco?
   You: How's the weather in Prague?
   You: Tell me the current conditions in New York
   You: What's the weather like?

When you run your agent, you'll see the framework's decision-making process in action. Here are the key phases to watch for:


**Phase 1: Framework Initialization**

.. code-block::

   🔄 Initializing framework...
   INFO Registry: Registry initialization complete!
        Components loaded:
           • 7 capabilities: memory, time_range_parsing, python, respond, clarify, current_weather, state_manager
           • X context types: MEMORY_CONTEXT, TIME_RANGE, CURRENT_WEATHER ...
   ✅ Framework initialized!

.. admonition:: What's Happening
   :class: important

   The framework loads all available capabilities, including your ``current_weather`` capability and ``CURRENT_WEATHER`` context type. This modular loading system allows you to see exactly which components are active in your agent.

**Phase 2: Task Processing Pipeline**

The user query "What's the weather in San Francisco right now?" is processed by the framework.

.. code-block::

   🔄 Processing: What's the weather in San Francisco right now?
   🔄 Extracting actionable task from conversation
   INFO Task_Extraction: * Extracted: 'Get the current weather conditions in San Francisco...'
   🔄 Analyzing task requirements...
   INFO Classifier: >>> Capability 'current_weather' >>> True
   🔄 Generating execution plan...

.. admonition:: What's Happening
   :class: important

   This is the **core decision-making process**:

   1. **Task Extraction**: Complete chat history gets converted to an actionable task
   2. **Classification**: Each capability is checked if it is needed to complete the current task. Notice how your capability gets activated (``>>> True``).
   3. **Planning**: An execution strategy is formulated, taking the active capabilities into account

**Phase 3: Execution Planning**

.. code-block::

   INFO Orchestrator: ==================================================
   INFO Orchestrator:  << Step 1
   INFO Orchestrator:  << ├───── id: 'sf_weather'
   INFO Orchestrator:  << ├─── node: 'current_weather'
   INFO Orchestrator:  << ├─── task: 'Retrieve current weather conditions for San Francisco
                          including temperature, conditions, and timestamp'
   INFO Orchestrator:  << └─ inputs: '[]'
   INFO Orchestrator:  << Step 2
   INFO Orchestrator:  << ├───── id: 'weather_response'
   INFO Orchestrator:  << ├─── node: 'respond'
   INFO Orchestrator:  << ├─── task: 'Present the current weather conditions for San Francisco to
                          the user in a clear and readable format'
   INFO Orchestrator:  << └─ inputs: '[{'CURRENT_WEATHER': 'sf_weather'}]'
   INFO Orchestrator: ==================================================
   ✅ Orchestrator: Final execution plan ready with 2 steps

.. admonition:: What's Happening
   :class: important

   The orchestrator breaks down the task into logical steps:

   - **Step 1**: Use your ``current_weather`` capability to get data and store it under the key ``sf_weather``
   - **Step 2**: Use the ``respond`` capability to format results and use the ``sf_weather`` context as input, knowing that its a ``CURRENT_WEATHER`` context type.

   This demonstrates how capabilities work together in a coordinated workflow.

**Phase 4: Real-Time Execution**

.. code-block::

   🔄 Executing current_weather... (10%)
   🔄 Extracting location from query...
   🔄 Getting weather for San Francisco...
   🔄 Weather retrieved: San Francisco - 21.0°C
   🔄 Generating response...

.. admonition:: What's Happening
   :class: important

   Your capability is now running! The status messages come from your ``streamer.status()`` and ``logger.info()`` calls, providing real-time feedback as your business logic executes.

**Final Result**

.. code-block::

   🤖 According to the [CURRENT_WEATHER.sf_weather] data, the weather conditions in San Francisco
   for 2025-08-04 are 21.0°C and Partly Cloudy.

.. admonition:: Success Indicators
   :class: important

   - Your weather data was successfully retrieved and stored as ``[CURRENT_WEATHER.sf_weather]``
   - The context reference shows the framework is using your structured data
   - The response is formatted professionally using the framework's response capability

**What You've Built**

By completing this tutorial, you've created an agentic system that demonstrates:

- **Modular Architecture**: Your capability integrates seamlessly with framework components
- **Scalable Orchestration**: The framework can handle multiple capabilities and context types
- **Structured Data Flow**: Information flows through context classes to enable capability coordination
- **Informative UX**: Real-time status updates and structured responses

Next Steps
==========

Experiment with Your Agent
---------------------------

Now that you have a working agent, try these experiments:

**Test framework-provided capabilities:**

- "Save the current weather in Prague to my memories"
- "Calculate the square root of the temperature in San Francisco"

**Try human-in-the-loop mechanics:**

- "/planning What's the weather in Prague?" - See the execution plan before it runs

**Modify your capability:**

- Add support for more cities
- Add new weather attributes (humidity, wind speed)
- Try different response formats

Scale to Production
-------------------

Ready to see Osprey at production scale? The :doc:`control-assistant`
demonstrates a complete industrial control system with 8+ capabilities working
together, complex orchestration patterns, and production deployment with a web UI.

This is where you'll see the modular architecture patterns from the
:doc:`conceptual-tutorial` applied to a real-world application.
