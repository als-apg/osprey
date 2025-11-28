"""End-to-end tests for tutorial workflows.

These tests validate the complete tutorial experience by:
1. Creating a fresh project from a template
2. Executing tutorial queries
3. Using LLM judges to evaluate results

These tests are the closest simulation to actual user experience.
"""

import pytest


@pytest.mark.e2e
@pytest.mark.e2e_tutorial
@pytest.mark.requires_cborg
@pytest.mark.slow
@pytest.mark.asyncio
async def test_bpm_timeseries_and_correlation_tutorial(e2e_project_factory, llm_judge):
    """Test the BPM analysis tutorial workflow end-to-end.

    This test validates the complete tutorial experience:
    1. Create a control_assistant project with hierarchical channel finder
    2. Query for BPM timeseries and correlation plots
    3. Verify the workflow completes successfully with expected outputs

    This is the canonical example from the Osprey documentation showing:
    - Channel finding (horizontal BPMs)
    - Time range parsing (last 24 hours)
    - Archiver data retrieval
    - Python plotting (timeseries + correlation)
    - Multi-step orchestration
    """
    # Step 1: Create project exactly like the tutorial
    project = await e2e_project_factory(
        name="my-control-assistant",
        template="control_assistant",
        registry_style="extend"
    )

    # Step 2: Initialize framework
    await project.initialize()

    # Step 3: Execute the tutorial query
    result = await project.query(
        "Give me a timeseries and a correlation plot of all "
        "horizontal BPM positions over the last 24 hours."
    )

    # Step 4: Define expectations in plain text
    expectations = """
    The workflow should successfully complete the following:

    1. **Channel Finding**: Identify all horizontal BPM (beam position monitor)
       channels in the facility. This should find approximately 15-20 BPM channels
       with names like 'DIAG:BPM01:POSITION:X', 'DIAG:BPM02:POSITION:X', etc.

    2. **Time Range Parsing**: Parse "last 24 hours" into absolute datetime
       objects representing a 24-hour period ending at the current time.

    3. **Archiver Data Retrieval**: Retrieve historical data from the archiver
       for all identified BPM channels over the parsed time range. The data
       should include thousands of data points per channel.

    4. **Timeseries Plot Generation**: Create a matplotlib visualization showing
       BPM positions over time. This could be either:
       - Multiple lines on one plot (one per BPM)
       - A grid of subplots (one per BPM)
       The plot should clearly show temporal evolution.

    5. **Correlation Plot Generation**: Create a correlation heatmap showing
       the correlation matrix between all BPM positions. This should be a
       square matrix (e.g., 20x20) with correlation coefficients ranging
       from -1 to +1.

    6. **Output Artifacts**: Produce at least two distinct figure files:
       - A timeseries plot (*.png)
       - A correlation heatmap (*.png)
       Optionally may include Jupyter notebooks documenting the analysis.

    7. **User Response**: Provide a coherent markdown response that:
       - Summarizes what was done
       - References the generated plots
       - Mentions key findings (e.g., correlation statistics, data coverage)
       - Includes file paths to the generated artifacts

    8. **No Critical Errors**: The workflow should complete without:
       - Exceptions or crashes
       - Failed capability executions
       - Missing expected outputs
       - Timeout errors

    Minor issues that are acceptable:
    - Using mock data instead of real EPICS archiver (in test environment)
    - Code generation retries (as long as it eventually succeeds)
    - Non-critical warnings in logs

    The overall workflow demonstrates successful multi-capability orchestration
    and should feel like a smooth, professional analysis pipeline.
    """

    # Step 5: Evaluate with LLM judge
    evaluation = await llm_judge.evaluate(
        result=result,
        expectations=expectations
    )

    # Step 6: Assert success with detailed failure info
    assert evaluation.passed, (
        f"Tutorial workflow failed evaluation\n\n"
        f"Confidence: {evaluation.confidence}\n\n"
        f"Reasoning:\n{evaluation.reasoning}\n\n"
        f"Warnings:\n" + "\n".join(f"  - {w}" for w in evaluation.warnings)
    )

    # Additional sanity checks (belt and suspenders)
    assert len(result.artifacts) >= 2, (
        f"Expected at least 2 artifacts (plots), got {len(result.artifacts)}"
    )

    assert result.error is None, (
        f"Workflow encountered error: {result.error}"
    )

    # Verify key capabilities were mentioned in trace (basic smoke check)
    trace_lower = result.execution_trace.lower()
    assert "channel_finding" in trace_lower, "Channel finding capability not executed"
    assert "archiver_retrieval" in trace_lower, "Archiver retrieval not executed"
    assert "python" in trace_lower, "Python execution not performed"


@pytest.mark.e2e
@pytest.mark.e2e_smoke
@pytest.mark.requires_cborg
@pytest.mark.slow
@pytest.mark.asyncio
async def test_simple_query_smoke_test(e2e_project_factory, llm_judge):
    """Quick smoke test to verify basic E2E infrastructure works.

    This is a simpler, faster test that validates:
    - Project creation works
    - Framework initialization works
    - Basic query execution works
    - LLM judge works

    Use this for quick validation before running full tutorial tests.
    """
    # Create minimal project
    project = await e2e_project_factory(
        name="smoke-test-project",
        template="minimal"
    )

    await project.initialize()

    # Simple query
    result = await project.query("Hello, can you help me?")

    # Minimal expectations
    expectations = """
    The framework should:
    - Accept the simple greeting/help request
    - Provide some kind of response (even if asking for clarification)
    - Not crash or error out
    - Complete the interaction
    """

    evaluation = await llm_judge.evaluate(result, expectations)

    assert evaluation.passed, (
        f"Basic smoke test failed: {evaluation.reasoning}"
    )


@pytest.mark.e2e
@pytest.mark.e2e_tutorial
@pytest.mark.requires_cborg
@pytest.mark.slow
@pytest.mark.asyncio
async def test_hello_world_weather_tutorial(e2e_project_factory, llm_judge):
    """Test the Hello World Weather tutorial workflow end-to-end.

    This test validates the complete beginner tutorial experience:
    1. Create a hello_world_weather project
    2. Query for weather in San Francisco
    3. Verify the workflow completes successfully with weather data

    This is the simplest tutorial in Osprey documentation showing:
    - Basic capability creation and registration
    - Mock external API integration
    - Context class usage for structured data
    - Single-step workflow execution
    - Clean registry pattern with extend_framework_registry()
    """
    # Step 1: Create project from hello_world_weather template
    project = await e2e_project_factory(
        name="hello-weather",
        template="hello_world_weather",
        registry_style="extend"
    )

    # Step 2: Initialize framework
    await project.initialize()

    # Step 3: Execute the tutorial query
    result = await project.query(
        "What's the weather in San Francisco?"
    )

    # Step 4: Define expectations in plain text
    expectations = """
    The workflow should successfully complete the following:

    1. **Query Classification**: Correctly identify this as a weather-related request
       requiring the current_weather capability.

    2. **Capability Execution**: Execute the current_weather capability without errors.
       The capability should:
       - Extract "San Francisco" as the location from the query
       - Call the mock weather API service
       - Return structured weather data

    3. **Weather Data Retrieval**: Successfully retrieve weather information for
       San Francisco containing:
       - Location name (should be "San Francisco")
       - Temperature value (numeric, in Celsius)
       - Weather conditions (descriptive string like "Sunny", "Cloudy", etc.)
       - Timestamp of when the data was retrieved

    4. **Context Creation**: Create a CURRENT_WEATHER context object with the
       retrieved data that's properly stored in the agent state.

    5. **User Response**: Provide a clear response to the user that:
       - Confirms the weather for San Francisco was retrieved
       - Mentions the temperature
       - Describes the weather conditions
       - Feels natural and conversational

    6. **No Critical Errors**: The workflow should complete without:
       - Registry initialization errors
       - Capability execution failures
       - Missing context class definitions
       - Exception traces or crashes
       - Framework routing errors

    7. **Mock API Integration**: The mock weather API should be called successfully,
       demonstrating proper external service integration patterns (even though
       it's mocked for the tutorial).

    Expected behavior:
    - Single-step execution (weather retrieval only, or weather + respond)
    - Fast execution (mock API is instant)
    - Structured data output (not just free text)
    - Clean completion without retries

    This is the "Hello World" of Osprey - it should feel smooth, simple, and
    work perfectly as a beginner's first experience with the framework.
    """

    # Step 5: Evaluate with LLM judge
    evaluation = await llm_judge.evaluate(
        result=result,
        expectations=expectations
    )

    # Step 6: Assert success with detailed failure info
    assert evaluation.passed, (
        f"Hello World Weather tutorial failed evaluation\n\n"
        f"Confidence: {evaluation.confidence}\n\n"
        f"Reasoning:\n{evaluation.reasoning}\n\n"
        f"Warnings:\n" + "\n".join(f"  - {w}" for w in evaluation.warnings)
    )

    # Additional sanity checks
    assert result.error is None, (
        f"Workflow encountered error: {result.error}"
    )

    # Verify weather capability was mentioned in trace
    trace_lower = result.execution_trace.lower()
    assert "weather" in trace_lower or "current_weather" in trace_lower, (
        "Weather capability not executed"
    )

    # Verify San Francisco was mentioned (either in trace or response)
    full_output = (result.execution_trace + result.response).lower()
    assert "san francisco" in full_output, (
        "San Francisco not mentioned in workflow output"
    )


# Template for adding new tutorial tests:
#
# @pytest.mark.e2e
# @pytest.mark.e2e_tutorial
# @pytest.mark.requires_cborg
# @pytest.mark.slow
# @pytest.mark.asyncio
# async def test_YOUR_TUTORIAL_NAME(e2e_project_factory, llm_judge):
#     """Test YOUR tutorial workflow."""
#
#     project = await e2e_project_factory(
#         name="your-project",
#         template="appropriate_template",
#         ...
#     )
#
#     await project.initialize()
#
#     result = await project.query("YOUR TUTORIAL QUERY")
#
#     expectations = """
#     YOUR PLAIN TEXT EXPECTATIONS
#     """
#
#     evaluation = await llm_judge.evaluate(result, expectations)
#     assert evaluation.passed, evaluation.reasoning

