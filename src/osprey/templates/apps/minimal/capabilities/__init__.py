"""{{ app_display_name }} Capabilities.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CAPABILITIES DIRECTORY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

This directory contains your capability implementations.

WHAT IS A CAPABILITY?

A capability is a self-contained module that:
✓ Performs a specific function (API call, computation, data retrieval, etc.)
✓ Accepts inputs (context) from the agent state
✓ Returns outputs (context) back to the agent state
✓ Can be orchestrated by the LLM to accomplish complex tasks

Think of capabilities as the "tools" or "functions" that your LLM agent can use.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DIRECTORY STRUCTURE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

capabilities/
├── __init__.py                 # This file (imports and exports)
├── example_capability.py.j2    # Complete example template (COPY THIS!)
├── my_api_capability.py        # Your API integration capability
├── my_analyzer_capability.py   # Your analysis capability
└── my_other_capability.py      # Additional capabilities

NAMING CONVENTION:

- Use lowercase_with_underscores for file names
- Name files descriptively: weather_api.py, data_analyzer.py
- Match file names to capability names when possible

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
HOW TO CREATE A NEW CAPABILITY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

STEP-BY-STEP GUIDE:

1. Copy the example template:
   cp example_capability.py.j2 my_capability.py

2. Update the class name and attributes:
   @capability_node
   class MyCapability(BaseCapability):
       name = "my_capability"
       description = "What my capability does"
       provides = ["MY_CONTEXT_TYPE"]
       requires = []

3. Implement the execute() method:
   - Extract required inputs from state
   - Call your API / perform your logic
   - Create output context objects
   - Return updated state

4. Define your context classes in context_classes.py:
   class MyContextType(CapabilityContext):
       CONTEXT_TYPE: ClassVar[str] = "MY_CONTEXT_TYPE"
       # ... define fields and methods

5. Register in registry.py:
   CapabilityRegistration(
       name="my_capability",
       module_path="{{ package_name }}.capabilities.my_capability",
       class_name="MyCapability",
       description="What my capability does",
       provides=["MY_CONTEXT_TYPE"],
       requires=[]
   )

6. Test with: osprey chat

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CAPABILITY PATTERNS & EXAMPLES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

PATTERN A: EXTERNAL API INTEGRATION
────────────────────────────────────────────────────────────────────────────────

Use when: You need to call an external REST API, database, or service

Key characteristics:
- Usually requires=[] (no dependencies)
- Calls external service in execute()
- Handles API errors and retries
- Returns data as context

Example:
```python
@capability_node
class WeatherAPICapability(BaseCapability):
    name = "weather_api"
    description = "Fetch weather data from external API"
    provides = ["WEATHER_DATA"]
    requires = []  # No dependencies

    @staticmethod
    async def execute(state: AgentState, **kwargs):
        # Call external API
        response = await weather_client.get_current_weather(city="SF")

        # Convert to context object
        weather_data = WeatherDataContext(
            temperature=response['temp'],
            conditions=response['conditions']
        )

        # Store and return
        return StateManager.store_context(
            state, "WEATHER_DATA",
            step.get("context_key"), weather_data
        )
```

PATTERN B: DATA TRANSFORMATION / ANALYSIS
────────────────────────────────────────────────────────────────────────────────

Use when: You need to process or analyze data from other capabilities

Key characteristics:
- Has requires=["INPUT_CONTEXT"] (depends on other capabilities)
- Extracts input context from state
- Performs computations/transformations
- Returns analysis results as context

Example:
```python
@capability_node
class DataAnalyzerCapability(BaseCapability):
    name = "data_analyzer"
    description = "Analyze weather patterns from historical data"
    provides = ["ANALYSIS_RESULTS"]
    requires = ["WEATHER_DATA"]  # Needs weather data first!

    @staticmethod
    async def execute(state: AgentState, **kwargs):
        # Extract required input
        context_manager = ContextManager(state)
        contexts = context_manager.extract_from_step(
            step, state,
            constraints=["WEATHER_DATA"],
            constraint_mode="hard"
        )
        weather_data = contexts["WEATHER_DATA"]

        # Perform analysis
        avg_temp = calculate_average(weather_data.temperatures)
        trend = detect_trend(weather_data.temperatures)

        # Create results context
        results = AnalysisResultsContext(
            average_temperature=avg_temp,
            trend=trend
        )

        # Store and return
        return StateManager.store_context(
            state, "ANALYSIS_RESULTS",
            step.get("context_key"), results
        )
```

PATTERN C: KNOWLEDGE RETRIEVAL
────────────────────────────────────────────────────────────────────────────────

Use when: You need to query a knowledge base, RAG system, or document store

Key characteristics:
- May have requires=[] or depend on query context
- Queries knowledge source (vector DB, documents, etc.)
- Returns relevant knowledge as context
- Helps LLM answer domain-specific questions

Example:
```python
@capability_node
class KnowledgeRetrievalCapability(BaseCapability):
    name = "knowledge_retrieval"
    description = "Retrieve relevant documentation from knowledge base"
    provides = ["KNOWLEDGE_CONTEXT"]
    requires = []

    async def execute(self):
        # Get query from task objective using helper method
        query = self.get_task_objective()

        # Query knowledge base
        docs = await knowledge_base.search(query, top_k=5)

        # Format as context
        knowledge = KnowledgeContext(
            query=query,
            documents=[doc.content for doc in docs],
            sources=[doc.source for doc in docs]
        )

        # Store and return using helper method
        return self.store_output_context(knowledge)
```

PATTERN D: CHAINED CAPABILITIES
────────────────────────────────────────────────────────────────────────────────

Use when: Multiple capabilities work together to accomplish a complex task

Key characteristics:
- Each capability has specific provides/requires
- Framework automatically sequences them
- Data flows through context between capabilities
- Enables complex workflows

Example flow:
```
User Query: "Analyze weather trends and predict tomorrow"

1. time_range_parser (framework built-in)
   provides: TIME_RANGE

2. weather_api (your capability)
   requires: TIME_RANGE
   provides: WEATHER_DATA

3. data_analyzer (your capability)
   requires: WEATHER_DATA
   provides: ANALYSIS_RESULTS

4. ml_predictor (your capability)
   requires: ANALYSIS_RESULTS
   provides: PREDICTION_RESULTS

Framework automatically sequences: 1 → 2 → 3 → 4
```

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
REQUIRED COMPONENTS OF A CAPABILITY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. @capability_node decorator
   → Registers the class as a capability

2. Inherit from BaseCapability
   → Provides framework integration

3. Class attributes:
   - name: Unique identifier (lowercase_with_underscores)
   - description: What it does (shown to LLM)
   - provides: List of context types it creates
   - requires: List of context types it needs

4. execute() method:
   - Async static method
   - Takes state: AgentState
   - Returns Dict[str, Any] with updated state

5. Optional methods:
   - classify_error(): Custom error handling
   - get_retry_policy(): Retry configuration
   - _create_orchestrator_guide(): Guide LLM orchestration
   - _create_classifier_guide(): Help task classification

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
COMMON INTEGRATION SCENARIOS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

SCENARIO 1: REST API Integration
→ Create one capability per major API endpoint or logical grouping
→ Example: weather_current.py, weather_forecast.py, weather_historical.py

SCENARIO 2: Database Integration
→ Create capabilities for different query types
→ Example: db_fetch.py, db_aggregate.py, db_search.py

SCENARIO 3: ML Model Integration
→ Create capabilities for different models or stages
→ Example: feature_extraction.py, model_inference.py, post_processing.py

SCENARIO 4: Multi-Step Workflow
→ Break down into logical steps, one capability per step
→ Example: data_fetch.py → data_clean.py → data_analyze.py → report_generate.py

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
BEST PRACTICES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

✓ Single Responsibility: Each capability should do ONE thing well
✓ Clear Naming: Use descriptive names that indicate what it does
✓ Good Descriptions: Help the LLM understand when to use this capability
✓ Error Handling: Implement classify_error() for robust error recovery
✓ Logging: Use logger to track execution for debugging
✓ Type Safety: Use Pydantic context classes for type validation
✓ Documentation: Add docstrings explaining what the capability does

✗ Avoid God Capabilities: Don't create one massive capability that does everything
✗ Avoid Circular Dependencies: Check provides/requires don't create loops
✗ Avoid Hardcoding: Use config.yml for API keys, URLs, etc.
✗ Avoid Silent Failures: Always log errors and raise appropriate exceptions

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
QUICK START CHECKLIST
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

For each capability you want to add:

□ 1. Copy example_capability.py.j2 to new file
□ 2. Rename class and update name/description attributes
□ 3. Define context classes in context_classes.py
□ 4. Set provides=[] to match your context types
□ 5. Set requires=[] if you need input from other capabilities
□ 6. Implement execute() with your logic
□ 7. Add error handling in classify_error()
□ 8. Register in registry.py
□ 9. Test with: osprey chat
□ 10. Implement _create_orchestrator_guide() for better LLM guidance

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
NEED HELP?
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. Study example_capability.py.j2 - it has complete inline documentation
2. Look at working examples: wind/ and weather/ directories
3. Check Osprey documentation:
   https://als-apg.github.io/osprey/developer-guides/building-first-capability.html
4. Debug with:
   >>> from osprey.registry import get_registry
   >>> registry = get_registry()
   >>> print([c.name for c in registry.capabilities])

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# IMPORT YOUR CAPABILITIES HERE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#
# After creating your capability files, import them here for easy access:
#
# Example:
# from .weather_api import WeatherAPICapability
# from .data_analyzer import DataAnalyzerCapability
# from .report_generator import ReportGeneratorCapability
#
# Then list them in __all__:
#
# __all__ = [
#     'WeatherAPICapability',
#     'DataAnalyzerCapability',
#     'ReportGeneratorCapability',
# ]
#
# Note: This is optional! The registry loads capabilities by module_path,
# so this is mainly for convenience and documentation.
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# TODO: Import your capabilities here
# from .my_capability import MyCapability

# TODO: List your capabilities here
__all__ = [
    # 'MyCapability',
]
