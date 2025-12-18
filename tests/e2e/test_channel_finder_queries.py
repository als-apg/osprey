"""End-to-end tests for individual channel finder queries.

These tests validate specific channel finder behaviors using real LLM calls.
Unlike benchmarks which test comprehensive accuracy across large datasets,
these tests focus on individual queries that exercise specific features or
edge cases (e.g., optional levels, direct signals, separator overrides).

Each test case can use a different database and configuration.
"""

import asyncio
import logging
import os
import sys
from pathlib import Path

import pytest
import yaml

logger = logging.getLogger(__name__)


# Test cases - each entry defines a complete test scenario
QUERY_TEST_CASES = [
    {
        "id": "optional_direct_signal",
        "name": "Direct signal at optional level (Heartbeat)",
        "description": (
            "Tests that direct signals (leaf nodes) at optional levels are "
            "correctly presented as options and selected by the LLM. The query "
            "asks for 'heartbeat' which is a direct signal at the device level, "
            "skipping the optional 'subdevice' level entirely."
        ),
        "database": "optional_levels.json",
        "query": "Find the channel address for the main control device heartbeat status signal",
        "expected_channels": ["CTRL:MAIN:MC-01:Heartbeat"],
        "pipeline": "hierarchical",
        "match_type": "exact",  # exact, partial, or any
    },
    {
        "id": "optional_subdevice_signal",
        "name": "Subdevice signal at optional level (PSU Voltage)",
        "description": (
            "Tests that container nodes (subdevices) at optional levels work "
            "correctly alongside direct signals. The query asks for PSU voltage "
            "which requires navigating through the PSU subdevice."
        ),
        "database": "optional_levels.json",
        "query": "What is the power supply voltage for main control device 1?",
        "expected_channels": ["CTRL:MAIN:MC-01:PSU:Voltage"],
        "pipeline": "hierarchical",
        "match_type": "exact",
    },
    {
        "id": "device_direct_signal_with_suffix",
        "name": "Direct signal at device level with suffix (Y_RB)",
        "description": (
            "Tests that direct signals (leaf nodes) at optional levels are "
            "correctly presented as options and selected by the LLM and additionally "
            "can contain a suffix directly at this level. The query "
            "asks for 'BPM position readbacks' which are direct signals at the "
            "device level with the _RB suffix skipping the optional 'subdevice' "
            "level but keeping the suffix level. This also validates that the "
            "underscore separator override is correctly applied."
        ),
        "database": "optional_levels.json",
        "query": "Find the channel address for the vertical position readback of the BPM diagnostic devices.",
        "expected_channels": ["CTRL:DIAG:BPM-01:Y_RB", "CTRL:DIAG:BPM-02:Y_RB"],
        "pipeline": "hierarchical",
        "match_type": "exact",  # exact, partial, or any
    },
    {
        "id": "expansion_at_optional_level",
        "name": "BUG: Expansion at optional tree level shows base container name",
        "description": (
            "Tests that nodes with _expansion at optional tree levels do NOT present "
            "the base container name as a selectable option. Only the expanded instances "
            "should be presented. This prevents the LLM from selecting invalid base names "
            "like 'CH' when it should select 'CH-1' or 'CH-2'. "
            "\n\n"
            "BUG BEHAVIOR: The hierarchical channel finder incorrectly presents 'CH' as "
            "a selectable option at the optional subdevice level. When the LLM selects 'CH', "
            "it builds invalid channels like 'CTRL:MAIN:MC-01:CH:Gain_RB' which get discarded, "
            "resulting in zero channels found even though valid channels exist. "
            "\n\n"
            "EXPECTED BEHAVIOR: Only 'CH-1' and 'CH-2' (the expanded instances) should appear "
            "as selectable options, not the base container 'CH'."
        ),
        "database": "optional_levels.json",
        "query": "Find the gain readback channels for the main control device hardware channel interfaces",
        "expected_channels": [
            "CTRL:MAIN:MC-01:CH-1:Gain_RB",
            "CTRL:MAIN:MC-01:CH-2:Gain_RB",
            "CTRL:MAIN:MC-02:CH-1:Gain_RB",
            "CTRL:MAIN:MC-02:CH-2:Gain_RB",
            "CTRL:MAIN:MC-03:CH-1:Gain_RB",
            "CTRL:MAIN:MC-03:CH-2:Gain_RB",
        ],
        "pipeline": "hierarchical",
        "match_type": "exact",
    },
]


@pytest.mark.e2e
@pytest.mark.requires_anthropic
@pytest.mark.slow
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "test_case",
    QUERY_TEST_CASES,
    ids=[case["id"] for case in QUERY_TEST_CASES],
)
async def test_channel_finder_query(e2e_project_factory, test_case):
    """Test a single channel finder query with expected results.

    This test:
    1. Creates a control-assistant project with specified pipeline
    2. Configures it to use the specified database
    3. Runs the query using the channel finder service
    4. Validates the results match expectations

    Args:
        e2e_project_factory: Fixture that creates test projects
        test_case: Dict with query, database, expected results, etc.
    """
    test_id = test_case["id"]
    logger.info(f"Running query test: {test_case['name']}")
    logger.info(f"Description: {test_case['description']}")

    # Create control-assistant project
    project = await e2e_project_factory(
        name=f"query-test-{test_id}",
        template="control_assistant",
        registry_style="extend",
    )

    # Configure pipeline mode
    config_path = project.config_path
    with open(config_path) as f:
        config = yaml.safe_load(f)

    pipeline_mode = test_case["pipeline"]
    config["channel_finder"]["pipeline_mode"] = pipeline_mode

    # Set database path in the correct location for the pipeline
    database_path = (
        Path(__file__).parent.parent.parent
        / "src/osprey/templates/apps/control_assistant/data/channel_databases/examples"
        / test_case["database"]
    )

    # Ensure the pipeline-specific config structure exists
    if "pipelines" not in config["channel_finder"]:
        config["channel_finder"]["pipelines"] = {}
    if pipeline_mode not in config["channel_finder"]["pipelines"]:
        config["channel_finder"]["pipelines"][pipeline_mode] = {}
    if "database" not in config["channel_finder"]["pipelines"][pipeline_mode]:
        config["channel_finder"]["pipelines"][pipeline_mode]["database"] = {}

    # Set the database path in the correct nested location
    config["channel_finder"]["pipelines"][pipeline_mode]["database"]["path"] = str(database_path)

    with open(config_path, "w") as f:
        yaml.dump(config, f)

    # Run the query
    result = await _run_channel_finder_query(project, test_case["query"])

    # Validate results based on match type
    found_channels = result.get("channels", [])
    expected = test_case["expected_channels"]
    match_type = test_case.get("match_type", "exact")

    if match_type == "exact":
        # Exact match - must find exactly these channels
        assert set(found_channels) == set(expected), (
            f"Query '{test_case['query']}' did not return expected channels.\n"
            f"Expected: {expected}\n"
            f"Got: {found_channels}"
        )
        logger.info(f"✓ Exact match: {found_channels}")

    elif match_type == "partial":
        # Partial match - must find at least these channels
        missing = set(expected) - set(found_channels)
        assert not missing, (
            f"Query '{test_case['query']}' missing expected channels.\n"
            f"Expected (at least): {expected}\n"
            f"Got: {found_channels}\n"
            f"Missing: {list(missing)}"
        )
        logger.info(f"✓ Partial match: {found_channels} (includes {expected})")

    elif match_type == "any":
        # Any match - must find at least one channel
        assert len(found_channels) > 0, (
            f"Query '{test_case['query']}' returned no channels.\nExpected at least one channel."
        )
        logger.info(f"✓ Found channels: {found_channels}")

    else:
        raise ValueError(f"Unknown match_type: {match_type}")


async def _run_channel_finder_query(project, query: str) -> dict:
    """Run a channel finder query programmatically.

    Args:
        project: E2E project object
        query: Natural language query string

    Returns:
        Dict with 'channels' list and other metadata
    """
    # Import the channel finder service from the project
    src_dir = str(project.project_dir / "src")
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)

    # Set environment to use project config
    original_config = os.environ.get("CONFIG_FILE")
    os.environ["CONFIG_FILE"] = str(project.config_path)

    try:
        # Initialize registry for this project context
        from osprey.registry import initialize_registry, reset_registry

        reset_registry()
        initialize_registry(config_path=str(project.config_path))

        # Import the service class from the project
        package_name = project.project_dir.name.replace("-", "_")
        service_module = __import__(
            f"{package_name}.services.channel_finder.service",
            fromlist=["ChannelFinderService"],
        )

        ChannelFinderService = service_module.ChannelFinderService
        # Service gets config from the initialized Osprey config system
        service = ChannelFinderService()

        # Run the query
        result = await service.find_channels(query)

        # Extract channel names from result
        if hasattr(result, "channels"):
            channels = [ch.channel for ch in result.channels]
        else:
            channels = []

        return {
            "channels": channels,
            "result": result,
        }

    finally:
        # Reset registry to prevent state pollution between tests
        from osprey.registry import reset_registry

        reset_registry()

        # Cleanup
        if original_config:
            os.environ["CONFIG_FILE"] = original_config
        elif "CONFIG_FILE" in os.environ:
            del os.environ["CONFIG_FILE"]

        if src_dir in sys.path:
            sys.path.remove(src_dir)

        # Clean up imported modules to prevent state pollution
        package_name = project.project_dir.name.replace("-", "_")
        modules_to_remove = [key for key in sys.modules.keys() if package_name in key]
        for module in modules_to_remove:
            del sys.modules[module]


# Additional helper for interactive testing
if __name__ == "__main__":
    """Run query tests interactively for debugging."""
    import asyncio

    async def run_single_test():
        # This would need the e2e_project_factory fixture
        # Mainly for showing structure during development
        for test_case in QUERY_TEST_CASES:
            print(f"\nTest: {test_case['name']}")
            print(f"Query: {test_case['query']}")
            print(f"Expected: {test_case['expected_channels']}")

    asyncio.run(run_single_test())
