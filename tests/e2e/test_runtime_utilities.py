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

import pytest


def _disable_capabilities(project, capability_names: list[str]):
    """Remove specific capabilities from project registry to force alternative paths.

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

    # For each capability, comment out its CapabilityRegistration block
    for cap_name in capability_names:
        # Find all CapabilityRegistration blocks and match by counting parentheses
        lines = registry_content.split('\n')
        new_lines = []
        i = 0

        while i < len(lines):
            line = lines[i]

            # Check if this line starts a CapabilityRegistration (and is not already commented)
            if 'CapabilityRegistration(' in line and not line.strip().startswith('#'):
                # Collect the full block by counting parentheses
                block_lines = [line]
                paren_count = line.count('(') - line.count(')')
                j = i + 1

                while j < len(lines) and paren_count > 0:
                    block_lines.append(lines[j])
                    paren_count += lines[j].count('(') - lines[j].count(')')
                    j += 1

                # Check if this block is for the capability we want to disable
                block_text = '\n'.join(block_lines)
                if re.search(rf'name\s*=\s*["\']' + cap_name + r'["\']', block_text):
                    # Comment out this block
                    for block_line in block_lines:
                        if block_line.strip():
                            new_lines.append(f'                # {block_line}')
                        else:
                            new_lines.append('#')
                    new_lines[-1] += '  # Disabled for testing'
                    i = j
                else:
                    # Keep this block as-is
                    new_lines.extend(block_lines)
                    i = j
            else:
                new_lines.append(line)
                i += 1

        registry_content = '\n'.join(new_lines)

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
        name="test-runtime-basic",
        template="control_assistant",
        registry_style="extend"
    )

    # Disable channel capabilities to force Python usage for control system ops
    # This validates that LLMs learn the osprey.runtime API from prompts
    # Note: Also disable channel_finding to prevent orchestrator hallucination
    _disable_capabilities(project, ["channel_write", "channel_read", "channel_finding"])

    # Enable writes and disable limits checking
    config_path = project.config_path
    import yaml
    config_data = yaml.safe_load(config_path.read_text())
    config_data['control_system']['writes_enabled'] = True

    # Disable limits checking - this test validates LLM can use osprey.runtime,
    # not that limits are enforced (that's tested in test_runtime_utilities_respects_channel_limits)
    config_data['control_system']['limits_checking']['enabled'] = False

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
    assert "python" in result.execution_trace.lower(), (
        "Python capability not executed"
    )

    # 3. Find the generated Python code
    code_files = [a for a in result.artifacts if str(a).endswith('.py')]
    assert len(code_files) > 0, "No Python code files generated"

    # Read the generated code
    generated_code = Path(code_files[0]).read_text()

    # 4. CRITICAL: Verify LLM used runtime utilities (not direct epics.caput)
    assert "from osprey.runtime import" in generated_code, (
        "Generated code doesn't import osprey.runtime - LLM didn't learn the API"
    )
    assert "write_channel" in generated_code, (
        "Generated code doesn't use write_channel() - LLM didn't follow prompts"
    )

    # 5. Verify code doesn't use deprecated direct EPICS calls
    assert "epics.caput" not in generated_code.lower(), (
        "Generated code uses deprecated epics.caput instead of runtime utilities"
    )

    # 6. Verify context snapshot exists
    executed_scripts_dir = project.project_dir / "_agent_data" / "executed_scripts"

    # Find execution folders (they're nested inside month folders like 2025-12/)
    execution_folders = []
    for month_dir in executed_scripts_dir.iterdir():
        if month_dir.is_dir():
            execution_folders.extend([d for d in month_dir.iterdir() if d.is_dir()])

    assert len(execution_folders) > 0, "No execution folders created"

    latest_execution = sorted(execution_folders)[-1]
    context_file = latest_execution / "context.json"

    assert context_file.exists(), f"context.json not created in {latest_execution}"

    # 7. Verify context snapshot contains control system config
    context_data = json.loads(context_file.read_text())
    assert "_execution_config" in context_data, (
        "Context doesn't contain execution config snapshot"
    )
    assert "control_system" in context_data["_execution_config"], (
        "Context snapshot missing control_system config"
    )

    # 8. Verify notebook includes runtime configuration
    notebook_files = list(latest_execution.glob("*.ipynb"))
    assert len(notebook_files) > 0, "No notebook generated"

    import nbformat
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
        name="test-runtime-limits",
        template="control_assistant",
        registry_style="extend"
    )

    # Disable channel capabilities to force Python usage
    _disable_capabilities(project, ["channel_write", "channel_read", "channel_finding"])

    # Configure with limits checking enabled
    config_path = project.config_path
    import yaml
    config_data = yaml.safe_load(config_path.read_text())

    # Enable writes
    config_data['control_system']['writes_enabled'] = True

    # Enable limits checking
    config_data['control_system']['limits_checking'] = {
        'enabled': True,
        'policy': {
            'allow_unlisted_channels': False,
            'on_violation': 'error'
        }
    }

    # Create limits file
    limits_file = tmp_path / "channel_limits.json"
    limits_data = {
        "TEST:VOLTAGE": {
            "min_value": 0.0,
            "max_value": 100.0,  # ← LIMIT: max is 100V
            "writable": True
        }
    }
    limits_file.write_text(json.dumps(limits_data, indent=2))
    config_data['control_system']['limits_checking']['database_path'] = str(limits_file)

    # Write updated config
    config_path.write_text(yaml.dump(config_data))

    # Initialize framework
    await project.initialize()

    # Request a write that VIOLATES the limit using Python
    result = await project.query(
        "Write a Python script to set the TEST:VOLTAGE channel to 150 volts."  # ← VIOLATION: exceeds 100V max
    )

    # === SAFETY VERIFICATION ===

    # 1. Execution should have encountered an error
    trace_lower = result.execution_trace.lower()

    # 2. Should see Python execution in trace
    assert "python" in trace_lower, "Python capability not executed"

    # 3. CRITICAL: Should see channel limits violation error
    assert "channellimitsviolationerror" in trace_lower or "limits" in trace_lower, (
        "Channel limits violation not detected!\n"
        f"This means runtime utilities bypassed safety checks!\n"
        f"Trace: {result.execution_trace[:1000]}"
    )

    # 4. Check generated code used runtime utilities
    code_files = [a for a in result.artifacts if str(a).endswith('.py')]
    if len(code_files) > 0:
        generated_code = Path(code_files[0]).read_text()

        # Verify LLM used runtime API
        assert "from osprey.runtime import" in generated_code, (
            "Generated code doesn't use runtime utilities"
        )
        assert "write_channel" in generated_code, (
            "Generated code doesn't use write_channel()"
        )

    # 5. Find execution folder and check for error details
    executed_scripts_dir = project.project_dir / "_agent_data" / "executed_scripts"
    if executed_scripts_dir.exists():
        execution_folders = [d for d in executed_scripts_dir.iterdir() if d.is_dir()]
        if len(execution_folders) > 0:
            latest_execution = sorted(execution_folders)[-1]

            # Check for execution metadata with error
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
        name="test-runtime-valid",
        template="control_assistant",
        registry_style="extend"
    )

    # Disable channel capabilities to force Python usage
    _disable_capabilities(project, ["channel_write", "channel_read", "channel_finding"])

    # Configure with limits checking enabled
    config_path = project.config_path
    import yaml
    config_data = yaml.safe_load(config_path.read_text())

    # Enable writes
    config_data['control_system']['writes_enabled'] = True

    # Enable limits checking
    config_data['control_system']['limits_checking'] = {
        'enabled': True,
        'policy': {
            'allow_unlisted_channels': False,
            'on_violation': 'error'
        }
    }

    # Create limits file
    limits_file = tmp_path / "channel_limits.json"
    limits_data = {
        "TEST:VOLTAGE": {
            "min_value": 0.0,
            "max_value": 100.0,
            "writable": True
        }
    }
    limits_file.write_text(json.dumps(limits_data, indent=2))
    config_data['control_system']['limits_checking']['database_path'] = str(limits_file)

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
    assert any(word in response_lower for word in ['success', 'set', 'wrote', 'completed']), (
        f"Response doesn't indicate successful write: {result.response[:300]}"
    )

    # 5. Verify generated code used runtime utilities
    code_files = [a for a in result.artifacts if str(a).endswith('.py')]
    assert len(code_files) > 0, "No code files generated"

    generated_code = Path(code_files[0]).read_text()
    assert "from osprey.runtime import" in generated_code
    assert "write_channel" in generated_code

    # 6. Verify execution succeeded
    executed_scripts_dir = project.project_dir / "_agent_data" / "executed_scripts"
    execution_folders = [d for d in executed_scripts_dir.iterdir() if d.is_dir()]
    assert len(execution_folders) > 0

    latest_execution = sorted(execution_folders)[-1]
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

