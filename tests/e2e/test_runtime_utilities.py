"""End-to-end tests for runtime utilities in Python executor.

These tests validate:
1. LLM learns to use osprey.runtime API from prompts
2. Runtime utilities work end-to-end with real execution
3. Channel limits safety integrates correctly
4. Context snapshot preservation works
"""

import json
import re
from pathlib import Path

import nbformat
import pytest


def _extract_code_from_notebook(notebook_path: Path) -> str:
    """Extract the user-generated Python code from a notebook.

    The notebook contains wrapper code (imports, context loading, etc.) and the
    actual user code. This function extracts code cells and returns their content.

    Args:
        notebook_path: Path to the .ipynb file

    Returns:
        Combined source code from all code cells in the notebook
    """
    notebook = nbformat.read(notebook_path, as_version=4)
    code_cells = [cell.source for cell in notebook.cells if cell.cell_type == "code"]
    return "\n".join(code_cells)


def _disable_capabilities(project, capability_names: list[str]):
    """Remove specific capabilities from project registry to force alternative paths.

    Works with both registry styles:
    - extend mode: Adds exclude_capabilities parameter to extend_framework_registry()
    - explicit mode: Comments out CapabilityRegistration blocks

    Args:
        project: E2EProject instance
        capability_names: List of capability names to disable (e.g. ["channel_write"])
    """
    # Find the registry file
    registry_files = list(project.project_dir.glob("src/*/registry.py"))
    if not registry_files:
        raise RuntimeError(f"Could not find registry.py in {project.project_dir}")

    registry_path = registry_files[0]
    registry_content = registry_path.read_text()

    if "extend_framework_registry(" in registry_content:
        # Extend mode: add exclude_capabilities parameter
        exclude_list = repr(capability_names)
        registry_content = registry_content.replace(
            "extend_framework_registry(",
            f"extend_framework_registry(\n            exclude_capabilities={exclude_list},",
        )
        registry_path.write_text(registry_content)
    else:
        # Explicit mode: comment out CapabilityRegistration blocks
        for cap_name in capability_names:
            lines = registry_content.split("\n")
            new_lines = []
            i = 0

            while i < len(lines):
                line = lines[i]

                if "CapabilityRegistration(" in line and not line.strip().startswith("#"):
                    block_lines = [line]
                    paren_count = line.count("(") - line.count(")")
                    j = i + 1

                    while j < len(lines) and paren_count > 0:
                        block_lines.append(lines[j])
                        paren_count += lines[j].count("(") - lines[j].count(")")
                        j += 1

                    block_text = "\n".join(block_lines)
                    if re.search(r'name\s*=\s*["\']' + cap_name + r'["\']', block_text):
                        for block_line in block_lines:
                            if block_line.strip():
                                new_lines.append(f"                # {block_line}")
                            else:
                                new_lines.append("#")
                        new_lines[-1] += "  # Disabled for testing"
                        i = j
                    else:
                        new_lines.extend(block_lines)
                        i = j
                else:
                    new_lines.append(line)
                    i += 1

            registry_content = "\n".join(new_lines)

        registry_path.write_text(registry_content)


@pytest.mark.e2e
@pytest.mark.slow
@pytest.mark.requires_cborg
@pytest.mark.asyncio
async def test_runtime_utilities_basic_write(e2e_project_factory):
    """Test that LLM generates code using runtime utilities for control system writes.

    Validates:
    1. LLM learns osprey.runtime API from generator prompts
    2. Generated code uses write_channel() instead of direct EPICS
    3. Code executes successfully
    4. Context snapshot is preserved
    """
    # Create project with writes enabled
    project = await e2e_project_factory(
        name="test-runtime-basic", template="control_assistant", registry_style="extend"
    )

    # Disable channel capabilities to force Python usage for control system ops
    # This validates that LLMs learn the osprey.runtime API from prompts
    # Note: Also disable channel_finding to prevent orchestrator hallucination
    _disable_capabilities(project, ["channel_write", "channel_read", "channel_finding"])

    # Enable writes and disable limits checking
    config_path = project.config_path
    import yaml

    config_data = yaml.safe_load(config_path.read_text())
    config_data["control_system"]["writes_enabled"] = True

    # Disable limits checking - this test validates LLM can use osprey.runtime,
    # not that limits are enforced (that's tested in test_runtime_utilities_respects_channel_limits)
    config_data["control_system"]["limits_checking"]["enabled"] = False

    # Disable approval for e2e testing - we want to test code execution, not approval flow
    if "approval" not in config_data:
        config_data["approval"] = {}
    # Set global mode to disabled to skip all approval workflows
    config_data["approval"]["global_mode"] = "disabled"

    config_path.write_text(yaml.dump(config_data))

    # Initialize framework
    await project.initialize()

    # Request a control system write operation using Python
    # (Explicitly request Python to force classifier to select it)
    result = await project.query(
        "Write a Python script to set the TEST:VOLTAGE channel to 75.5 volts."
    )

    # === DETERMINISTIC ASSERTIONS ===

    # 1. Workflow completed without errors
    assert result.error is None, f"Workflow error: {result.error}"

    # 2. Python capability was executed
    assert "python" in result.execution_trace.lower(), "Python capability not executed"

    # 3. Find execution folder and notebook
    executed_scripts_dir = project.project_dir / "_agent_data" / "executed_scripts"

    # Find execution folders (they're nested inside month folders like 2025-12/)
    execution_folders = []
    for month_dir in executed_scripts_dir.iterdir():
        if month_dir.is_dir():
            execution_folders.extend([d for d in month_dir.iterdir() if d.is_dir()])

    assert len(execution_folders) > 0, "No execution folders created"

    latest_execution = sorted(execution_folders)[-1]

    # 4. Find the notebook and extract generated code
    notebook_files = list(latest_execution.glob("**/*.ipynb"))
    assert len(notebook_files) > 0, "No notebook generated"

    generated_code = _extract_code_from_notebook(notebook_files[0])

    # 5. CRITICAL: Verify LLM used runtime utilities (not direct epics.caput)
    assert "from osprey.runtime import" in generated_code, (
        "Generated code doesn't import osprey.runtime - LLM didn't learn the API"
    )
    assert "write_channel" in generated_code, (
        "Generated code doesn't use write_channel() - LLM didn't follow prompts"
    )

    # 6. Verify code doesn't use deprecated direct EPICS calls
    assert "epics.caput" not in generated_code.lower(), (
        "Generated code uses deprecated epics.caput instead of runtime utilities"
    )

    # 7. Verify context snapshot exists
    context_file = latest_execution / "context.json"
    assert context_file.exists(), f"context.json not created in {latest_execution}"

    # 8. Verify context snapshot contains control system config
    context_data = json.loads(context_file.read_text())
    assert "_execution_config" in context_data, "Context doesn't contain execution config snapshot"
    assert "control_system" in context_data["_execution_config"], (
        "Context snapshot missing control_system config"
    )

    # 9. Verify notebook includes runtime configuration
    notebook = nbformat.read(notebook_files[0], as_version=4)
    notebook_text = "\n".join(cell.source for cell in notebook.cells)

    assert "from osprey.runtime import configure_from_context" in notebook_text, (
        "Notebook missing runtime configuration cell"
    )

    print("✅ Runtime utilities basic write test passed")
    print("   - LLM correctly used osprey.runtime API")
    print("   - Context snapshot preserved")
    print("   - Notebook includes runtime configuration")


@pytest.mark.e2e
@pytest.mark.slow
@pytest.mark.requires_cborg
@pytest.mark.asyncio
async def test_runtime_utilities_respects_channel_limits(e2e_project_factory, tmp_path):
    """CRITICAL SAFETY TEST: Verify runtime utilities respect channel limits.

    This is the most important test - proves that using osprey.runtime
    doesn't bypass the channel limits safety layer.

    Validates:
    1. LLM generates code using runtime utilities
    2. Limits validator catches violations BEFORE they reach control system
    3. Execution fails with ChannelLimitsViolationError
    4. Error message includes violation details
    """
    # Create project
    project = await e2e_project_factory(
        name="test-runtime-limits", template="control_assistant", registry_style="extend"
    )

    # Disable channel capabilities to force Python usage
    _disable_capabilities(project, ["channel_write", "channel_read", "channel_finding"])

    # Configure with limits checking enabled
    config_path = project.config_path
    import yaml

    config_data = yaml.safe_load(config_path.read_text())

    # Enable writes
    config_data["control_system"]["writes_enabled"] = True

    # Enable limits checking
    config_data["control_system"]["limits_checking"] = {
        "enabled": True,
        "policy": {"allow_unlisted_channels": False, "on_violation": "error"},
    }

    # Create limits file
    limits_file = tmp_path / "channel_limits.json"
    limits_data = {
        "TEST:VOLTAGE": {
            "min_value": 0.0,
            "max_value": 100.0,  # ← LIMIT: max is 100V
            "writable": True,
        }
    }
    limits_file.write_text(json.dumps(limits_data, indent=2))
    config_data["control_system"]["limits_checking"]["database_path"] = str(limits_file)

    # Disable approval for e2e testing - we want to test limits enforcement, not approval flow
    if "approval" not in config_data:
        config_data["approval"] = {}
    config_data["approval"]["global_mode"] = "disabled"

    # Write updated config
    config_path.write_text(yaml.dump(config_data))

    # Initialize framework
    await project.initialize()

    # Request a write that VIOLATES the limit using Python
    result = await project.query(
        "Write a Python script to set the TEST:VOLTAGE channel to 150 volts."  # ← VIOLATION: exceeds 100V max
    )

    # === SAFETY VERIFICATION ===

    # 1. Should see Python execution in trace
    trace_lower = result.execution_trace.lower()
    assert "python" in trace_lower, "Python capability not executed"

    # 2. CRITICAL: Should see channel limits violation - either in execution trace OR response
    # The limits violation is caught by the runtime and reported through the response.
    # The capability may report [SUCCESS] because it successfully produced an error report.
    response_lower = result.response.lower()
    limits_detected = (
        "channellimitsviolationerror" in trace_lower
        or "limits" in trace_lower
        or "limits violation" in response_lower
        or "exceeds" in response_lower
        or "maximum" in response_lower and "100" in response_lower
    )
    assert limits_detected, (
        "Channel limits violation not detected!\n"
        f"This means runtime utilities bypassed safety checks!\n"
        f"Trace: {result.execution_trace[:500]}\n"
        f"Response preview: {result.response[:500]}"
    )

    # 4. Find execution folder and check for error details
    executed_scripts_dir = project.project_dir / "_agent_data" / "executed_scripts"
    if executed_scripts_dir.exists():
        # Find execution folders (nested inside month folders)
        execution_folders = []
        for month_dir in executed_scripts_dir.iterdir():
            if month_dir.is_dir():
                execution_folders.extend([d for d in month_dir.iterdir() if d.is_dir()])

        if len(execution_folders) > 0:
            latest_execution = sorted(execution_folders)[-1]

            # 5. Check generated code used runtime utilities (from notebook)
            notebook_files = list(latest_execution.glob("**/*.ipynb"))
            if len(notebook_files) > 0:
                generated_code = _extract_code_from_notebook(notebook_files[0])

                # Verify LLM used runtime API
                assert "from osprey.runtime import" in generated_code, (
                    "Generated code doesn't use runtime utilities"
                )
                assert "write_channel" in generated_code, (
                    "Generated code doesn't use write_channel()"
                )

            # 6. Check for execution metadata with error
            metadata_file = latest_execution / "execution_metadata.json"
            if metadata_file.exists():
                metadata = json.loads(metadata_file.read_text())

                # Verify execution failed (not succeeded)
                assert metadata.get("success") is False, (
                    "Execution metadata shows success despite limits violation!"
                )

                # Check error mentions limits or violation
                error_msg = metadata.get("error", "").lower()
                assert "limit" in error_msg or "violation" in error_msg or "exceed" in error_msg, (
                    f"Error message doesn't mention limits violation: {error_msg}"
                )

    print("✅ Channel limits safety test passed")
    print("   - Runtime utilities correctly blocked violation")
    print("   - Limits validator caught 150V write (max: 100V)")
    print("   - Error propagated correctly to user")


@pytest.mark.e2e
@pytest.mark.slow
@pytest.mark.requires_cborg
@pytest.mark.asyncio
async def test_runtime_utilities_within_limits_succeeds(e2e_project_factory, tmp_path):
    """Positive test: Runtime utilities allow valid writes within limits.

    Validates:
    1. LLM generates code using runtime utilities
    2. Valid writes (within limits) succeed
    3. Results are returned properly
    """
    # Create project
    project = await e2e_project_factory(
        name="test-runtime-valid", template="control_assistant", registry_style="extend"
    )

    # Disable channel capabilities to force Python usage
    _disable_capabilities(project, ["channel_write", "channel_read", "channel_finding"])

    # Configure with limits checking enabled
    config_path = project.config_path
    import yaml

    config_data = yaml.safe_load(config_path.read_text())

    # Enable writes
    config_data["control_system"]["writes_enabled"] = True

    # Enable limits checking
    config_data["control_system"]["limits_checking"] = {
        "enabled": True,
        "policy": {"allow_unlisted_channels": False, "on_violation": "error"},
    }

    # Create limits file
    limits_file = tmp_path / "channel_limits.json"
    limits_data = {"TEST:VOLTAGE": {"min_value": 0.0, "max_value": 100.0, "writable": True}}
    limits_file.write_text(json.dumps(limits_data, indent=2))
    config_data["control_system"]["limits_checking"]["database_path"] = str(limits_file)

    # Disable approval for e2e testing - we want to test limits enforcement, not approval flow
    if "approval" not in config_data:
        config_data["approval"] = {}
    config_data["approval"]["global_mode"] = "disabled"

    # Write updated config
    config_path.write_text(yaml.dump(config_data))

    # Initialize framework
    await project.initialize()

    # Request a VALID write (within limits) using Python
    result = await project.query(
        "Write a Python script to set the TEST:VOLTAGE channel to 75 volts."  # ← Valid: within 0-100V range
    )

    # === SUCCESS VERIFICATION ===

    # 1. Workflow should complete without errors
    assert result.error is None, f"Workflow error: {result.error}"

    # 2. Python capability executed
    assert "python" in result.execution_trace.lower()

    # 3. Should NOT see limits violation
    trace_lower = result.execution_trace.lower()
    assert "channellimitsviolationerror" not in trace_lower, (
        "Valid write was incorrectly blocked by limits checker"
    )

    # 4. Response should indicate success
    response_lower = result.response.lower()
    assert any(word in response_lower for word in ["success", "set", "wrote", "completed"]), (
        f"Response doesn't indicate successful write: {result.response[:300]}"
    )

    # 5. Find execution folder and verify code
    executed_scripts_dir = project.project_dir / "_agent_data" / "executed_scripts"

    # Find execution folders (nested inside month folders)
    execution_folders = []
    for month_dir in executed_scripts_dir.iterdir():
        if month_dir.is_dir():
            execution_folders.extend([d for d in month_dir.iterdir() if d.is_dir()])

    assert len(execution_folders) > 0, "No execution folders created"

    latest_execution = sorted(execution_folders)[-1]

    # 6. Verify generated code used runtime utilities (from notebook)
    notebook_files = list(latest_execution.glob("**/*.ipynb"))
    assert len(notebook_files) > 0, "No notebook generated"

    generated_code = _extract_code_from_notebook(notebook_files[0])
    assert "from osprey.runtime import" in generated_code
    assert "write_channel" in generated_code

    # 7. Verify execution succeeded
    metadata_file = latest_execution / "execution_metadata.json"

    if metadata_file.exists():
        metadata = json.loads(metadata_file.read_text())
        assert metadata.get("success") is True, (
            f"Execution failed despite valid write: {metadata.get('error')}"
        )

    print("✅ Valid write test passed")
    print("   - Runtime utilities allowed valid 75V write (limit: 100V)")
    print("   - Execution completed successfully")
    print("   - LLM correctly used runtime API")
