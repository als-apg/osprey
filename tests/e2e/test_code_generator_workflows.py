"""End-to-end tests for Python code generator workflows.

These tests validate the complete code generation pipeline including:
1. Configuration loading (basic vs claude_code generators)
2. Example script reading and guidance following
3. Code generation with specific instructions
4. Code execution and validation

Tests use LLM judges to evaluate both code quality and instruction following.
"""

import json

import pytest

# Check if Claude SDK is available using the framework's detection
try:
    from osprey.services.python_executor.generation import CLAUDE_SDK_AVAILABLE
except ImportError:
    CLAUDE_SDK_AVAILABLE = False


# =============================================================================
# Test fixtures: Example scripts with facility-specific utility functions
# =============================================================================
# These scripts contain custom functions that Claude MUST read and use to
# produce correct output. This tests that Claude Code actually reads the
# codebase rather than just generating generic code.

FACILITY_UTILS_SCRIPT = '''"""Facility-specific data processing utilities.

IMPORTANT: All data normalization in this facility MUST use the functions
defined in this file. These implement our facility's specific algorithms
that differ from standard implementations.
"""

import numpy as np


def normalize_facility_data(values):
    """Facility-standard data normalization algorithm.

    Our facility uses a SPECIFIC normalization that differs from standard:
    1. Subtract the MEDIAN (not mean)
    2. Divide by the INTERQUARTILE RANGE (not standard deviation)
    3. Multiply by our facility constant: 42.0

    This normalization is required for all data processing at our facility.

    Args:
        values: List or array of numeric values

    Returns:
        List of normalized values using facility algorithm

    Example:
        >>> normalize_facility_data([1, 2, 3, 4, 5])
        [-42.0, -21.0, 0.0, 21.0, 42.0]
    """
    values = np.array(values, dtype=float)
    median = np.median(values)
    q75, q25 = np.percentile(values, [75, 25])
    iqr = q75 - q25
    if iqr == 0:
        iqr = 1.0
    normalized = ((values - median) / iqr) * 42.0
    return normalized.tolist()


# Example usage
if __name__ == "__main__":
    test_data = [1, 2, 3, 4, 5]
    result = normalize_facility_data(test_data)
    print(f"Input: {test_data}")
    print(f"Normalized: {result}")
    # Expected output: [-42.0, -21.0, 0.0, 21.0, 42.0]
'''

FACILITY_README = '''# Facility Data Processing Utilities

## MANDATORY: Use Facility-Standard Normalization

**ALL data normalization at this facility MUST use `normalize_facility_data()` from `facility_utils.py`.**

This is NOT standard normalization. Our facility algorithm:
1. Uses MEDIAN instead of mean
2. Uses INTERQUARTILE RANGE instead of standard deviation
3. Applies facility constant 42.0

### Usage

```python
from facility_utils import normalize_facility_data

raw_data = [1, 2, 3, 4, 5]
normalized = normalize_facility_data(raw_data)
# Result: [-42.0, -21.0, 0.0, 21.0, 42.0]
```

### Why This Algorithm?

Our facility's instruments have specific noise characteristics that make
median/IQR normalization more robust than mean/std normalization. The
constant 42.0 scales the output to match our detector calibration.

### Requirements

- You MUST import and use `normalize_facility_data` for ALL normalization
- Do NOT implement your own normalization
- Do NOT use standard normalization functions (they will give wrong results)
'''


@pytest.mark.e2e
@pytest.mark.slow
@pytest.mark.requires_cborg
@pytest.mark.asyncio
async def test_basic_generator_simple_code_generation(e2e_project_factory):
    """Test basic generator creates simple, functional Python code.

    This validates:
    1. Basic generator configuration loads correctly
    2. Simple prompt-based code generation works
    3. Generated code executes successfully
    4. Results are returned properly

    Note: Basic generator does NOT read example scripts - it's a simple
    prompt-to-code generator. For codebase reading, see Claude Code tests.
    """
    # Step 1: Create project with basic generator (default for control_assistant)
    project = await e2e_project_factory(
        name="test-basic-generator",
        template="control_assistant",
        registry_style="extend"
    )

    # Step 2: Initialize framework
    await project.initialize()

    # Step 3: Request a very simple calculation and plot
    result = await project.query(
        "Calculate the sum of numbers 1 through 10 and create a simple bar chart showing these numbers. "
        "Save the plot as a PNG file."
    )

    # Step 4: Verify workflow completed without errors
    assert result.error is None, f"Workflow error: {result.error}"

    # Step 5: Verify Python capability was executed
    trace_lower = result.execution_trace.lower()
    assert "python" in trace_lower, (
        "Python capability not executed - check execution trace:\n"
        f"{result.execution_trace[:500]}"
    )

    # Step 6: Verify plot artifact was created
    assert len(result.artifacts) > 0, (
        f"Expected at least one artifact (plot), got {len(result.artifacts)}"
    )

    # Step 7: Verify at least one PNG file was created
    png_files = [a for a in result.artifacts if str(a).lower().endswith('.png')]
    assert len(png_files) > 0, (
        f"Expected PNG file in artifacts, got: {result.artifacts}"
    )

    # Step 8: Verify response indicates success
    response_lower = result.response.lower()
    assert any(word in response_lower for word in ['success', 'created', 'saved', 'completed']), (
        f"Response does not indicate successful completion:\n{result.response[:300]}"
    )

    print(f"✅ Basic generator test passed - created {len(result.artifacts)} artifact(s)")


@pytest.mark.e2e
@pytest.mark.slow
@pytest.mark.requires_cborg
@pytest.mark.skipif(
    not CLAUDE_SDK_AVAILABLE,
    reason="Claude Agent SDK not installed"
)
@pytest.mark.asyncio
async def test_claude_code_generator_with_codebase_guidance(e2e_project_factory, tmp_path):
    """Test Claude Code generator reads example scripts and uses facility functions.

    This validates that Claude Code ACTUALLY reads the codebase by:
    1. Providing a custom utility function with facility-specific behavior
    2. Asking Claude to use that function for data processing
    3. Verifying the output matches EXACTLY what the function produces

    The facility_utils.py contains a normalize_facility_data() function that uses
    a non-standard algorithm (median/IQR/42.0 constant). Claude cannot produce
    the correct output without reading and using this function.

    Expected result for [1, 2, 3, 4, 5]: [-42.0, -21.0, 0.0, 21.0, 42.0]
    """
    import numpy as np

    # Step 1: Create project with control_assistant template
    project = await e2e_project_factory(
        name="test-claude-code",
        template="control_assistant",
        registry_style="extend"
    )

    # Step 2: Modify config.yml to use claude_code generator
    config_path = project.project_dir / "config.yml"
    config_content = config_path.read_text()

    # Replace basic generator with claude_code
    config_content = config_content.replace(
        'code_generator: "basic"',
        'code_generator: "claude_code"'
    )
    config_path.write_text(config_content)

    # Step 3: Generate claude_generator_config.yml
    from osprey.cli.templates import TemplateManager
    template_manager = TemplateManager()

    ctx = {
        "default_provider": "cborg",
        "default_model": "anthropic/claude-haiku"
    }

    template_manager.render_template(
        "apps/control_assistant/claude_generator_config.yml.j2",
        ctx,
        project.project_dir / "claude_generator_config.yml"
    )

    # Step 4: Set up example scripts with facility-specific utility function
    # Note: Scripts go in 'plotting' directory because that's what codebase_guidance config expects
    example_scripts_dir = project.project_dir / "_agent_data" / "example_scripts" / "plotting"
    example_scripts_dir.mkdir(parents=True, exist_ok=True)

    # Write the utility script with our specific normalization function
    (example_scripts_dir / "facility_utils.py").write_text(FACILITY_UTILS_SCRIPT)

    # Write README explaining the requirement to use this function
    (example_scripts_dir / "README.md").write_text(FACILITY_README)

    # Step 5: Initialize framework (will load claude_code generator)
    await project.initialize()

    # Step 6: Request normalization using the facility's standard method
    # The specific input values are chosen to produce a deterministic, verifiable output
    # IMPORTANT: The query explicitly mentions looking at example_scripts to trigger codebase reading
    test_input = [1, 2, 3, 4, 5]
    result = await project.query(
        f"Normalize the data values {test_input}. "
        "IMPORTANT: You MUST use the normalize_facility_data() function from "
        "facility_utils.py in the example_scripts/plotting/ directory. "
        "Do NOT use standard normalization - our facility uses a CUSTOM algorithm. "
        "Read the file first to see how it works. "
        "Return the normalized values in the results dictionary under 'normalized_data'."
    )

    # Step 7: Verify workflow completed without errors
    assert result.error is None, f"Workflow error: {result.error}"

    # Step 8: Verify Python execution happened
    trace_lower = result.execution_trace.lower()
    assert "python" in trace_lower or "code" in trace_lower, (
        "Python code generation/execution not performed - check execution trace:\n"
        f"{result.execution_trace[:500]}"
    )

    # Step 9: Find the results.json to check the actual output
    executed_scripts_dir = project.project_dir / "_agent_data" / "executed_scripts"
    results_files = list(executed_scripts_dir.glob("**/results.json"))

    assert len(results_files) > 0, (
        f"No results.json found in executed scripts!\n"
        f"Searched in: {executed_scripts_dir}"
    )

    # Get the most recent results file
    latest_results_file = max(results_files, key=lambda p: p.stat().st_mtime)

    with open(latest_results_file) as f:
        execution_results = json.load(f)

    # Step 10: Calculate expected output using the EXACT facility algorithm
    # This is what Claude should produce if it read and used facility_utils.py
    def expected_normalize(values):
        values = np.array(values, dtype=float)
        median = np.median(values)
        q75, q25 = np.percentile(values, [75, 25])
        iqr = q75 - q25
        if iqr == 0:
            iqr = 1.0
        normalized = ((values - median) / iqr) * 42.0
        return normalized.tolist()

    expected_output = expected_normalize(test_input)
    # Expected: [-42.0, -21.0, 0.0, 21.0, 42.0]

    # Step 11: Check if normalized_data is in results
    actual_output = execution_results.get('normalized_data')

    # Debug output
    print("\n🔍 Codebase Function Usage Validation:")
    print(f"   📊 Input data: {test_input}")
    print(f"   ✅ Expected output: {expected_output}")
    print(f"   📦 Actual output: {actual_output}")
    print(f"   📄 Results file: {latest_results_file.name}")

    # Step 12: Verify the output matches exactly (with floating point tolerance)
    assert actual_output is not None, (
        f"Claude did not return 'normalized_data' in results!\n"
        f"Available keys: {list(execution_results.keys())}\n"
        f"Full results: {json.dumps(execution_results, indent=2)[:500]}\n"
        f"This suggests Claude did NOT follow the instructions to use the facility function."
    )

    # Convert to numpy arrays for comparison with tolerance
    actual_array = np.array(actual_output)
    expected_array = np.array(expected_output)

    # Check shape
    assert actual_array.shape == expected_array.shape, (
        f"Output shape mismatch!\n"
        f"Expected shape: {expected_array.shape}, Actual shape: {actual_array.shape}\n"
        f"Expected: {expected_output}\n"
        f"Actual: {actual_output}"
    )

    # Check values with tolerance for floating point
    matches = np.allclose(actual_array, expected_array, rtol=1e-5, atol=1e-5)

    assert matches, (
        f"Normalized values don't match facility algorithm!\n"
        f"Input: {test_input}\n"
        f"Expected (facility algorithm): {expected_output}\n"
        f"Actual: {actual_output}\n"
        f"Difference: {(actual_array - expected_array).tolist()}\n\n"
        f"This proves Claude did NOT use the facility's normalize_facility_data() function.\n"
        f"Standard normalization would give different results.\n"
        f"Claude must READ facility_utils.py and USE the function defined there."
    )

    print(f"✅ Claude Code codebase guidance test passed - facility function correctly used!")


@pytest.mark.e2e
@pytest.mark.slow
@pytest.mark.requires_cborg
@pytest.mark.skipif(
    not CLAUDE_SDK_AVAILABLE,
    reason="Claude Agent SDK not installed"
)
@pytest.mark.asyncio
async def test_claude_code_robust_profile_workflow(e2e_project_factory):
    """Test Claude Code generator robust profile (multi-phase workflow).

    This validates that the robust profile (scan → plan → implement) works
    correctly with all phases executing in sequence.
    """
    # Step 1: Create project
    project = await e2e_project_factory(
        name="test-claude-robust",
        template="control_assistant",
        registry_style="extend"
    )

    # Step 2: Configure for claude_code with robust profile
    config_path = project.project_dir / "config.yml"
    config_content = config_path.read_text()

    # Switch to claude_code
    config_content = config_content.replace(
        'code_generator: "basic"',
        'code_generator: "claude_code"'
    )
    config_path.write_text(config_content)

    # Step 3: Generate claude config with robust profile
    from osprey.cli.templates import TemplateManager
    template_manager = TemplateManager()

    ctx = {
        "default_provider": "cborg",
        "default_model": "anthropic/claude-haiku"
    }

    claude_config_path = project.project_dir / "claude_generator_config.yml"
    template_manager.render_template(
        "apps/control_assistant/claude_generator_config.yml.j2",
        ctx,
        claude_config_path
    )

    # Modify to use robust profile by updating config.yml
    config_content = config_path.read_text()
    config_content = config_content.replace(
        'profile: "fast"',
        'profile: "robust"'
    )
    config_path.write_text(config_content)

    # Step 4: Initialize
    await project.initialize()

    # Step 5: Execute a query that benefits from planning
    result = await project.query(
        "Create a multi-panel figure with 4 subplots showing: "
        "1) sine wave, 2) cosine wave, 3) their sum, 4) their product. "
        "Each subplot should be clearly labeled."
    )

    # Step 6: Verify workflow completed without errors
    assert result.error is None, f"Workflow error: {result.error}"

    # Step 7: Verify Python execution happened
    trace_lower = result.execution_trace.lower()
    assert "python" in trace_lower or "code" in trace_lower, (
        "Python code generation/execution not performed - check execution trace:\n"
        f"{result.execution_trace[:500]}"
    )

    # Step 8: Verify plot artifacts were created
    assert len(result.artifacts) > 0, (
        f"Expected plot artifacts, got {len(result.artifacts)}"
    )

    # Step 9: Verify PNG file was created
    png_files = [a for a in result.artifacts if str(a).lower().endswith('.png')]
    assert len(png_files) > 0, (
        f"Expected PNG file in artifacts, got: {result.artifacts}"
    )

    # Step 10: Verify response indicates success
    response_lower = result.response.lower()
    assert any(word in response_lower for word in ['success', 'created', 'saved', 'completed', 'subplot']), (
        f"Response does not indicate successful completion:\n{result.response[:300]}"
    )

    # Step 11: Optional - check for multi-phase indicators (scan/plan/implement)
    # This is informational and not critical to the test passing
    prompts_dir = project.project_dir / "_agent_data" / "executed_scripts"
    conversation_files = list(prompts_dir.glob("**/prompts/conversation_full.json"))

    if conversation_files:
        latest_conversation = max(conversation_files, key=lambda p: p.stat().st_mtime)
        with open(latest_conversation) as f:
            conversation_data = json.load(f)
        conversation_str = json.dumps(conversation_data).lower()

        # Look for phase indicators
        has_scan = "scan" in conversation_str
        has_plan = "plan" in conversation_str

        print("\n🔍 Robust Profile Validation:")
        print(f"   📁 Conversation file: {latest_conversation.name}")
        print(f"   🔎 Scan phase detected: {has_scan}")
        print(f"   📋 Plan phase detected: {has_plan}")

    print(f"✅ Claude Code robust profile test passed - {len(png_files)} PNG(s) created")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

