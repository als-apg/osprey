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
        name="my-control-assistant", template="control_assistant", registry_style="extend"
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

    6. **Output Artifacts**: Produce at least one figure file (*.png).
       The plots may be combined as subplots in a single figure, or created
       as separate files - either approach is acceptable. Optionally may
       include Jupyter notebooks documenting the analysis.

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
    evaluation = await llm_judge.evaluate(result=result, expectations=expectations)

    # Step 6: Assert success with detailed failure info
    assert evaluation.passed, (
        f"Tutorial workflow failed evaluation\n\n"
        f"Confidence: {evaluation.confidence}\n\n"
        f"Reasoning:\n{evaluation.reasoning}\n\n"
        f"Warnings:\n" + "\n".join(f"  - {w}" for w in evaluation.warnings)
    )

    # Additional sanity checks (belt and suspenders)
    # Check for at least one PNG (plots may be combined in one figure)
    png_files = [a for a in result.artifacts if a.suffix == ".png"]
    assert len(png_files) >= 1, f"Expected at least 1 PNG file, got {len(png_files)}"

    assert result.error is None, f"Workflow encountered error: {result.error}"

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
    project = await e2e_project_factory(name="smoke-test-project", template="minimal")

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

    assert evaluation.passed, f"Basic smoke test failed: {evaluation.reasoning}"


@pytest.mark.e2e
@pytest.mark.e2e_tutorial
@pytest.mark.requires_cborg
@pytest.mark.slow
@pytest.mark.asyncio
async def test_hello_world_weather_tutorial(e2e_project_factory, llm_judge):
    """Test the Hello World Weather tutorial workflow end-to-end.

    This test validates the complete beginner tutorial experience:
    1. Create a hello_world_weather project
    2. Query that exercises BOTH weather AND Python capabilities
    3. Verify multi-step workflow completes successfully

    This is the simplest tutorial in Osprey documentation showing:
    - Basic capability creation and registration
    - Mock external API integration (weather)
    - Python code generation and execution
    - Context class usage for structured data
    - Multi-step workflow execution with context passing
    - Clean registry pattern with extend_framework_registry()
    - Framework configuration defaults (execution, generators)

    The test uses a query that requires calculating the square root of
    San Francisco's temperature, which exercises:
    - Weather capability → retrieves temperature
    - Python capability → generates and executes code
    - Context passing → temperature flows from weather to python
    - Configuration defaults → validates simplified hello_world config works

    This ensures the tutorial works end-to-end including Python execution,
    which would have caught the missing execution config defaults bug.
    """
    # Step 1: Create project from hello_world_weather template
    project = await e2e_project_factory(
        name="hello-weather", template="hello_world_weather", registry_style="extend"
    )

    # Step 2: Initialize framework
    await project.initialize()

    # Step 3: Execute the tutorial query
    # This query exercises BOTH the weather capability AND Python capability
    # in a multi-step workflow, validating the complete tutorial experience
    result = await project.query(
        "Write a Python script that calculates the square root of the temperature in San Francisco"
    )

    # Step 4: Define expectations in plain text
    expectations = """
    The workflow should successfully complete the following:

    1. **Multi-Capability Classification**: Correctly identify this requires both
       the current_weather capability (to get temperature) and the python capability
       (to calculate square root).

    2. **Weather Capability Execution**: Execute the current_weather capability without errors.
       The capability should:
       - Extract "San Francisco" as the location from the query
       - Call the mock weather API service
       - Return structured weather data with temperature (e.g., 18.0°C)

    3. **Python Code Generation**: Successfully generate Python code that:
       - Imports necessary libraries (e.g., math.sqrt)
       - Retrieves the temperature from the CURRENT_WEATHER context
       - Calculates the square root of the temperature
       - Stores results in the 'results' dictionary

    4. **Python Code Execution**: Execute the generated code successfully:
       - No syntax errors
       - No runtime exceptions
       - Code produces the expected numerical result
       - Results are properly formatted and returned

    5. **Multi-Step Orchestration**: The orchestrator should create a plan with
       approximately 2-3 steps:
       - Step 1: Retrieve weather for San Francisco
       - Step 2: Execute Python code to calculate square root
       - Step 3: Respond to user with results

    6. **Context Passing**: The temperature from the weather capability should be
       successfully passed to the Python capability via the CURRENT_WEATHER context.

    7. **Configuration Defaults**: The workflow should work with the simplified
       hello_world_weather config that relies on framework defaults for:
       - Python execution configuration (execution_method, code_generator)
       - Code generator configuration (basic generator with model_config_name)
       - This validates that configuration defaults are complete and functional

    8. **User Response**: Provide a clear response to the user that:
       - Mentions the San Francisco temperature
       - Shows the calculated square root value
       - References the Python code/notebook created
       - Feels natural and complete

    9. **No Critical Errors**: The workflow should complete without:
       - "Unknown provider: None" errors (validates config defaults work)
       - Code generation failures
       - Python execution errors
       - Registry initialization problems
       - Missing context class definitions
       - Framework routing errors

    Expected behavior:
    - Multi-step execution (weather → python → respond)
    - Both mock API and Python code generation succeed
    - Structured data output with numerical results
    - Clean completion without retries
    - Validates that simplified tutorial config works end-to-end

    This test validates the COMPLETE hello_world_weather experience including
    the framework's Python capability, which is essential for any real-world
    tutorial application.
    """

    # Step 5: Evaluate with LLM judge
    evaluation = await llm_judge.evaluate(result=result, expectations=expectations)

    # Step 6: Assert success with detailed failure info
    assert evaluation.passed, (
        f"Hello World Weather tutorial failed evaluation\n\n"
        f"Confidence: {evaluation.confidence}\n\n"
        f"Reasoning:\n{evaluation.reasoning}\n\n"
        f"Warnings:\n" + "\n".join(f"  - {w}" for w in evaluation.warnings)
    )

    # Belt-and-suspenders sanity checks (redundant with LLM judge but provide fast failure signals)
    assert result.error is None, f"Workflow encountered error: {result.error}"

    # Verify BOTH capabilities were executed
    trace_lower = result.execution_trace.lower()
    assert "weather" in trace_lower or "current_weather" in trace_lower, (
        "Weather capability not executed"
    )
    assert "python" in trace_lower, "Python capability not executed"

    # Verify San Francisco was mentioned (either in trace or response)
    full_output = (result.execution_trace + result.response).lower()
    assert "san francisco" in full_output, "San Francisco not mentioned in workflow output"

    # Verify numerical result is present (sqrt calculation produced output)
    # The sqrt of typical SF temps (15-20°C) should be around 3.8-4.5
    assert any(
        keyword in full_output for keyword in ["sqrt", "square root", "result", "calculated"]
    ), "Python calculation results not mentioned in output"

    # =========================================================================
    # DETAILED VERIFICATION: Check LLM Parser API Call Logs
    # =========================================================================
    # Verify the location parser was called correctly by inspecting API logs
    # This ensures the LLM-based location extraction is working as intended

    import re

    api_calls_dir = project.project_dir / "_agent_data" / "api_calls"
    assert api_calls_dir.exists(), f"API calls directory not found: {api_calls_dir}"

    # Find the location parsing API call log
    # The log file should be named with _parse_location_from_query in the name
    api_log_files = list(api_calls_dir.glob("*_parse_location_from_query*.txt"))

    assert len(api_log_files) > 0, (
        f"No API call log found for location parsing in {api_calls_dir}. "
        f"Found files: {list(api_calls_dir.glob('*.txt'))}"
    )

    # Read the most recent log (in case there are multiple)
    api_log_file = sorted(api_log_files, key=lambda p: p.stat().st_mtime)[-1]
    api_log_content = api_log_file.read_text()

    # Verify the prompt contains our location extraction instructions
    assert "location extraction assistant" in api_log_content.lower(), (
        "API log doesn't contain expected location extraction prompt"
    )
    assert "Examples:" in api_log_content, "API log should contain example-based prompt"

    # Verify the prompt includes the user's query
    assert "san francisco" in api_log_content.lower(), "User's query not found in API log"

    # Verify structured output model is used
    assert "_ParsedWeatherQuery" in api_log_content or "ParsedWeatherQuery" in api_log_content, (
        "API log should show structured output model usage"
    )

    # Verify the output contains San Francisco as the extracted location
    # Look for the OUTPUT RESPONSE section
    output_match = re.search(
        r"OUTPUT RESPONSE.*?={10,}(.*?)(?:={10,}|$)", api_log_content, re.DOTALL
    )

    if output_match:
        output_section = output_match.group(1)
        # Check for San Francisco in the output (could be JSON or other format)
        assert "San Francisco" in output_section or "san francisco" in output_section.lower(), (
            f"Output doesn't contain 'San Francisco'. Output section:\n{output_section}"
        )
    else:
        # If we can't parse the output section, at least verify it exists somewhere
        assert "San Francisco" in api_log_content or "san francisco" in api_log_content.lower(), (
            "Parsed location 'San Francisco' not found anywhere in API log"
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
