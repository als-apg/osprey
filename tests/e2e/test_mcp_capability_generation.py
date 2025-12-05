"""End-to-end test for MCP capability generation workflow.

This test validates the complete MCP integration pipeline:
1. Generate a demo MCP server
2. Launch the server at a random port
3. Generate a capability from that MCP server
4. Integrate the capability into the project
5. Query the agent using the MCP-backed capability
6. Verify the response using an LLM judge
"""

import os
import socket
import subprocess
import sys
import time

import pytest


def find_free_port() -> int:
    """Find a free port for the MCP server."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        s.listen(1)
        port = s.getsockname()[1]
    return port


def wait_for_server(port: int, timeout: float = 10.0) -> bool:
    """Wait for server to become available.

    Args:
        port: Port number to check
        timeout: Maximum time to wait in seconds

    Returns:
        True if server is reachable, False otherwise
    """
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.settimeout(1.0)
                result = sock.connect_ex(("localhost", port))
                if result == 0:
                    # Port is open, give it a moment to fully initialize
                    time.sleep(0.5)
                    return True
        except Exception:
            pass
        time.sleep(0.2)
    return False


@pytest.mark.e2e
@pytest.mark.e2e_tutorial
@pytest.mark.requires_cborg
@pytest.mark.slow
@pytest.mark.asyncio
async def test_mcp_capability_generation_workflow(e2e_project_factory, llm_judge, tmp_path):
    """Test complete MCP capability generation workflow.

    This test validates:
    1. MCP server generation and launch
    2. Capability generation from live MCP server
    3. Automatic registry integration
    4. End-to-end query execution using MCP capability
    5. LLM judge verification of weather response

    This demonstrates the full developer experience for integrating
    external services via MCP into an Osprey project.
    """
    # =========================================================================
    # Step 1: Create Control Assistant project
    # =========================================================================
    project = await e2e_project_factory(
        name="mcp-test-project", template="control_assistant", registry_style="extend"
    )

    # Change to project directory for all operations
    original_cwd = os.getcwd()
    os.chdir(project.project_dir)

    mcp_server_process = None

    try:
        # =====================================================================
        # Step 2: Generate MCP server at random port
        # =====================================================================
        port = find_free_port()
        server_name = "weather_mcp_test"
        server_file = project.project_dir / f"{server_name}_server.py"

        # Generate MCP server code
        from osprey.generators.mcp_server_template import write_mcp_server_file

        write_mcp_server_file(output_path=server_file, server_name=server_name, port=port)

        assert server_file.exists(), f"MCP server file not created: {server_file}"
        assert server_file.stat().st_size > 0, "MCP server file is empty"

        # =====================================================================
        # Step 3: Launch MCP server in background
        # =====================================================================
        # Start the server process detached
        mcp_server_process = subprocess.Popen(
            [sys.executable, str(server_file)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,  # Detach from current session
            cwd=project.project_dir,
        )

        # Wait for server to become available
        server_ready = wait_for_server(port, timeout=10.0)

        # Verify server is still running
        if mcp_server_process.poll() is not None:
            # Server died, get error output
            _, stderr = mcp_server_process.communicate(timeout=1)
            pytest.fail(
                f"MCP server failed to start.\n"
                f"Exit code: {mcp_server_process.returncode}\n"
                f"Error: {stderr.decode() if stderr else 'No error output'}\n"
                f"Make sure fastmcp is installed: pip install fastmcp"
            )

        assert server_ready, (
            f"MCP server did not become available on port {port} within timeout.\n"
            f"Server process is running: {mcp_server_process.poll() is None}\n"
            f"Make sure the port is not blocked by firewall."
        )

        mcp_url = f"http://localhost:{port}"

        # =====================================================================
        # Step 4: Generate capability from MCP server
        # =====================================================================
        from osprey.cli.generate_cmd import (
            _determine_capabilities_path,
            _generate_capability_async,
            get_mcp_generator,
            initialize_registry,
        )

        capability_name = "weather_mcp"
        server_display_name = "WeatherMCP"

        # Determine output path
        output_path = _determine_capabilities_path(capability_name)

        # Initialize registry (required for LLM providers)
        initialize_registry()

        # Create generator
        MCPCapabilityGenerator = get_mcp_generator()
        generator = MCPCapabilityGenerator(
            capability_name=capability_name,
            server_name=server_display_name,
            verbose=False,
            provider=None,  # Use defaults from config
            model_id=None,
        )

        # Run async generation
        await _generate_capability_async(
            generator=generator,
            mcp_url=mcp_url,
            simulated=False,  # Use real MCP server
            output_path=output_path,
            quiet=True,
        )

        # Verify capability file was created
        assert output_path.exists(), f"Capability file not created: {output_path}"
        assert output_path.stat().st_size > 0, "Capability file is empty"

        # =====================================================================
        # Step 5: Integrate capability into registry
        # =====================================================================
        # The _generate_capability_async function should have offered integration
        # For testing, we'll manually add it to ensure it's registered

        from osprey.generators.registry_updater import (
            add_to_registry,
            find_registry_file,
            is_already_registered,
        )

        registry_path = find_registry_file()
        assert registry_path is not None, "Could not find registry file"

        # Add to registry if not already there (auto-integration might have done it)
        if not is_already_registered(registry_path, capability_name):
            class_name = "WeatherMcpCapability"  # Class name has "Capability" suffix
            context_class_name = "WeatherMcpResultsContext"
            context_type = "WEATHERMCP_RESULTS"
            description = "WeatherMCP operations via MCP server"

            new_content, _ = add_to_registry(
                registry_path,
                capability_name,
                class_name,
                context_type,
                context_class_name,
                description,
            )
            registry_path.write_text(new_content)

        # =====================================================================
        # Step 6: Initialize project with new capability
        # =====================================================================
        # Add project's src directory to sys.path so new capability can be imported
        src_dir = str(project.project_dir / "src")
        if src_dir not in sys.path:
            sys.path.insert(0, src_dir)

        # Reset registry and config caches to pick up new capability
        from osprey.registry import reset_registry
        from osprey.utils import config as config_module

        reset_registry()
        config_module._default_config = None
        config_module._default_configurable = None
        config_module._config_cache.clear()

        # CRITICAL: Remove the project's registry module from sys.modules
        # so it gets re-imported with the new capability registration
        project_name = project.project_dir.name.replace("-", "_")
        registry_module_name = f"{project_name}.registry"
        if registry_module_name in sys.modules:
            del sys.modules[registry_module_name]

        # Also remove the capability module if it was previously imported
        capability_module_name = f"{project_name}.capabilities.{capability_name}"
        if capability_module_name in sys.modules:
            del sys.modules[capability_module_name]

        # Clear CONFIG_FILE env var
        if "CONFIG_FILE" in os.environ:
            del os.environ["CONFIG_FILE"]

        # Now initialize the project (this will load the new capability)
        await project.initialize()

        # =====================================================================
        # Step 7: Query for weather in San Francisco
        # =====================================================================
        result = await project.query("What's the current weather in San Francisco?")

        # =====================================================================
        # Step 8: Define expectations for LLM judge
        # =====================================================================
        expectations = """
        The workflow should successfully complete the following:

        1. **Query Classification**: Correctly identify this as a weather-related request
           that should be handled by the weather_mcp capability.

        2. **MCP Capability Execution**: Execute the weather_mcp capability which:
           - Connects to the running MCP server
           - Uses the ReAct agent to determine which MCP tools to call
           - Calls appropriate weather tools (e.g., get_current_weather)
           - Retrieves weather data for San Francisco

        3. **Weather Data Retrieval**: Successfully retrieve weather information containing:
           - Location confirmation (San Francisco)
           - Temperature value (numeric)
           - Weather conditions (e.g., "Sunny", "Cloudy", "Rainy")
           - Additional weather details (optional: humidity, wind, etc.)

        4. **User Response**: Provide a clear, coherent response that:
           - Confirms this is weather information for San Francisco
           - Includes temperature information
           - Describes current weather conditions
           - Reads like a natural weather report
           - Does NOT say "I don't know" or "I cannot help"

        5. **No Critical Errors**: The workflow should complete without:
           - MCP connection failures
           - Tool execution errors
           - Registry initialization errors
           - Framework routing errors
           - Timeouts or crashes

        6. **Context Creation**: Create appropriate WEATHERMCP_RESULTS context
           with the retrieved weather data.

        The KEY criterion for success is:
        - The response MUST be about actual weather conditions
        - The response MUST NOT be an error message or "I don't know" response
        - The response should demonstrate that the MCP capability successfully
          connected to the server, executed tools, and retrieved weather data

        This test validates the complete MCP integration pipeline from server
        generation through capability execution.
        """

        # =====================================================================
        # Step 9: Evaluate with LLM judge
        # =====================================================================
        evaluation = await llm_judge.evaluate(result=result, expectations=expectations)

        # =====================================================================
        # Step 10: Assert success with detailed failure info
        # =====================================================================
        assert evaluation.passed, (
            f"MCP capability generation workflow failed evaluation\n\n"
            f"Confidence: {evaluation.confidence}\n\n"
            f"Reasoning:\n{evaluation.reasoning}\n\n"
            f"Warnings:\n" + "\n".join(f"  - {w}" for w in evaluation.warnings) + "\n\n"
            f"Response preview:\n{result.response[:500]}\n\n"
            f"Execution trace preview:\n{result.execution_trace[:1000]}"
        )

        # Additional sanity checks
        assert result.error is None, f"Workflow encountered error: {result.error}"

        # Verify MCP capability was mentioned in trace
        trace_lower = result.execution_trace.lower()
        assert "weather" in trace_lower or "mcp" in trace_lower, (
            "Weather/MCP capability not executed - check trace:\n" f"{result.execution_trace[:500]}"
        )

        # Verify San Francisco was mentioned
        full_output = (result.execution_trace + result.response).lower()
        assert "san francisco" in full_output, "San Francisco not mentioned in workflow output"

        # Verify response contains temperature/weather info
        response_lower = result.response.lower()
        has_weather_keywords = any(
            keyword in response_lower
            for keyword in ["temperature", "weather", "sunny", "cloudy", "rain", "degrees", "°"]
        )
        assert (
            has_weather_keywords
        ), f"Response does not contain weather information:\n{result.response}"

    finally:
        # =====================================================================
        # Cleanup: Stop MCP server and restore directory
        # =====================================================================
        if mcp_server_process and mcp_server_process.poll() is None:
            # Server is still running, terminate it
            mcp_server_process.terminate()
            try:
                mcp_server_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                # Force kill if it doesn't terminate gracefully
                mcp_server_process.kill()
                mcp_server_process.wait()

        # Remove src from sys.path
        src_dir = str(project.project_dir / "src")
        if src_dir in sys.path:
            sys.path.remove(src_dir)

        # Restore original working directory
        os.chdir(original_cwd)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s", "--e2e-verbose"])
