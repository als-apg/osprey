"""Pytest fixtures for end-to-end workflow tests."""

import asyncio
import logging
import os
import sys
from io import StringIO
from pathlib import Path

import pytest
from click.testing import CliRunner
from langgraph.checkpoint.memory import MemorySaver

from osprey.cli.init_cmd import init
from osprey.graph import create_graph
from osprey.infrastructure.gateway import Gateway
from osprey.registry import get_registry, initialize_registry, reset_registry
from osprey.utils.config import get_full_configuration
from tests.e2e.judge import LLMJudge, WorkflowResult


@pytest.fixture(autouse=True, scope="function")
def reset_registry_between_tests():
    """Auto-reset registry before each e2e test to ensure isolation.

    This is critical for e2e tests as they create full framework instances
    with registries that can leak state between tests.

    IMPORTANT: Also clears config cache AND CONFIG_FILE env var to prevent
    stale configuration from leaking between tests that use different config files.
    """
    # Reset before test
    reset_registry()

    # CRITICAL: Clear config cache to prevent stale config from previous tests
    # The config module has global caches that persist across registry resets
    from osprey.utils import config as config_module
    config_module._default_config = None
    config_module._default_configurable = None
    config_module._config_cache.clear()

    # Reset approval manager singleton to prevent approval state pollution
    try:
        import osprey.approval.approval_manager as approval_module
        approval_module._approval_manager = None
    except ImportError:
        pass  # Approval manager might not be available in all test environments

    # Clear CONFIG_FILE environment variable to prevent contamination
    if 'CONFIG_FILE' in os.environ:
        del os.environ['CONFIG_FILE']

    yield

    # Reset after test for good measure
    reset_registry()

    # Clear config cache again after test
    config_module._default_config = None
    config_module._default_configurable = None
    config_module._config_cache.clear()

    # Reset approval manager singleton again
    try:
        import osprey.approval.approval_manager as approval_module
        approval_module._approval_manager = None
    except ImportError:
        pass

    # Clear CONFIG_FILE env var again
    if 'CONFIG_FILE' in os.environ:
        del os.environ['CONFIG_FILE']


class E2EProject:
    """Wrapper for an E2E test project with query execution capabilities."""

    def __init__(self, project_dir: Path, config_path: Path, verbose: bool = False):
        self.project_dir = project_dir
        self.config_path = config_path
        self.gateway: Gateway | None = None
        self.graph = None
        self.base_config = None
        self._thread_id = "e2e_test_session"
        self.verbose = verbose

    async def initialize(self):
        """Initialize the framework for this project."""
        # Set config file environment variable (needed for Python executor)
        os.environ['CONFIG_FILE'] = str(self.config_path)

        # Change to project directory for initialization
        # (registry paths in config.yml are relative to project root)
        original_cwd = os.getcwd()
        os.chdir(self.project_dir)

        try:
            # Initialize framework following CLI pattern
            # 1. Load configuration and get full configurable
            configurable = get_full_configuration(str(self.config_path)).copy()

            # Add session info
            configurable.update({
                "user_id": "e2e_test_user",
                "thread_id": self._thread_id,
                "chat_id": "e2e_test_chat",
                "session_id": self._thread_id,
                "interface_context": "e2e_test"
            })

            # 2. Initialize registry
            initialize_registry(config_path=str(self.config_path))
            registry = get_registry()

            # 3. Create graph with checkpointer
            checkpointer = MemorySaver()
            self.graph = create_graph(registry, checkpointer=checkpointer)

            # 4. Create gateway
            self.gateway = Gateway()

            # 5. Set up base config for graph execution (include full configurable)
            # This matches the CLI pattern and ensures model configs are available
            from osprey.utils.config import get_config_value
            recursion_limit = get_config_value("execution_limits.graph_recursion_limit")

            self.base_config = {
                "configurable": configurable,
                "recursion_limit": recursion_limit
            }
        finally:
            # Restore original working directory
            os.chdir(original_cwd)

    async def query(self, message: str) -> WorkflowResult:
        """Execute a query and return complete results.

        Args:
            message: User query to execute

        Returns:
            WorkflowResult containing response, trace, and artifacts
        """
        if not self.gateway:
            await self.initialize()

        # Change to project directory for query execution
        # (Python executor needs correct working directory for _agent_data)
        original_cwd = os.getcwd()
        os.chdir(self.project_dir)

        # Set up log capture and optional console output
        log_capture = StringIO()
        log_handler = logging.StreamHandler(log_capture)
        log_handler.setLevel(logging.INFO)

        # Attach to root logger to capture all capability logs
        root_logger = logging.getLogger()
        original_level = root_logger.level
        root_logger.setLevel(logging.INFO)
        root_logger.addHandler(log_handler)

        # Add console handler if verbose mode
        console_handler = None
        if self.verbose:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(logging.INFO)
            # Format to show just the important parts
            console_handler.setFormatter(logging.Formatter('  %(message)s'))
            root_logger.addHandler(console_handler)
            print(f"\n🔄 Executing query: '{message}'")

        # Execute query through gateway
        start_time = asyncio.get_event_loop().time()

        try:
            response_text = ""
            error = None

            if self.verbose:
                print("  ⏳ Processing through gateway...")

            # Process message through gateway
            result = await self.gateway.process_message(
                message,
                self.graph,
                self.base_config
            )

            if self.verbose:
                print("  ⏳ Executing agent graph...")

            # Execute the graph based on gateway result
            if result.resume_command:
                # Approval flow resumption
                final_state = await self.graph.ainvoke(
                    result.resume_command,
                    config=self.base_config
                )
            elif result.agent_state:
                # Normal conversation turn
                final_state = await self.graph.ainvoke(
                    result.agent_state,
                    config=self.base_config
                )
            else:
                error = result.error or "Unknown error in gateway"
                final_state = None

            # Extract response from final state
            if final_state and 'messages' in final_state:
                # Get last AI message
                for msg in reversed(final_state['messages']):
                    if hasattr(msg, 'content') and msg.content:
                        response_text = msg.content
                        break

            if self.verbose:
                print(f"  ✅ Query completed in {asyncio.get_event_loop().time() - start_time:.2f}s")
                if response_text:
                    print(f"  📝 Response preview: {response_text[:100]}...")

        except Exception as e:
            error = f"{type(e).__name__}: {str(e)}"
            response_text = f"ERROR: {error}"
            if self.verbose:
                print(f"  ❌ Error: {error}")

        finally:
            # Remove log handlers and restore original level
            root_logger.removeHandler(log_handler)
            if console_handler:
                root_logger.removeHandler(console_handler)
            root_logger.setLevel(original_level)

        execution_time = asyncio.get_event_loop().time() - start_time

        # Get execution trace from logs
        execution_trace = log_capture.getvalue()

        # Collect artifacts (figures, notebooks) while still in project directory
        artifacts = self._collect_artifacts()

        # Restore original working directory
        os.chdir(original_cwd)

        return WorkflowResult(
            query=message,
            response=response_text,
            execution_trace=execution_trace,
            artifacts=artifacts,
            error=error,
            execution_time=execution_time
        )

    def _collect_artifacts(self) -> list[Path]:
        """Collect all artifacts produced by the workflow."""
        artifacts = []
        agent_data_dir = self.project_dir / "_agent_data" / "executed_scripts"

        if agent_data_dir.exists():
            # Find all figures, notebooks, and Python code files
            for pattern in ["**/*.png", "**/*.jpg", "**/*.ipynb", "**/*.py"]:
                artifacts.extend(agent_data_dir.glob(pattern))

        return sorted(artifacts)

    def cleanup(self):
        """Clean up project resources."""
        # Remove from sys.path
        src_dir = str(self.project_dir / "src")
        if src_dir in sys.path:
            sys.path.remove(src_dir)

        # Clean up imported modules
        project_name = self.project_dir.name.replace('-', '_')
        modules_to_remove = [
            key for key in sys.modules.keys()
            if project_name in key
        ]
        for module in modules_to_remove:
            del sys.modules[module]


@pytest.fixture
async def e2e_project_factory(tmp_path, request):
    """Factory fixture for creating E2E test projects.

    Usage:
        async def test_something(e2e_project_factory):
            project = await e2e_project_factory(
                name="test-project",
                template="control_assistant",
                ...
            )
    """
    projects = []
    # Get verbose flag from pytest command line
    verbose = request.config.getoption("--e2e-verbose", default=False)

    async def _create_project(
        name: str,
        template: str = "minimal",
        registry_style: str = "extend",
        provider: str | None = "cborg",
        model: str | None = "anthropic/claude-haiku",
        output_dir: Path | None = None,
    ) -> E2EProject:
        """Create a new E2E test project.

        Args:
            name: Project name
            template: Template to use (minimal, hello_world_weather, control_assistant)
            registry_style: Registry style (extend/standalone)
            provider: AI provider (anthropic, openai, google, cborg, ollama)
            model: Model identifier
            output_dir: Where to create project (defaults to tmp_path)

        Returns:
            Initialized E2EProject instance
        """
        output_dir = output_dir or tmp_path
        runner = CliRunner()

        # Build CLI arguments matching the current init command
        args = [
            name,
            '--template', template,
            '--registry-style', registry_style,
            '--output-dir', str(output_dir),
        ]

        # Add provider and model if specified
        if provider:
            args.extend(['--provider', provider])
        if model:
            args.extend(['--model', model])

        # Create project
        result = runner.invoke(init, args)

        if result.exit_code != 0:
            raise RuntimeError(
                f"Failed to create E2E project: {result.output}\n"
                f"Exception: {result.exception}"
            )

        project_dir = output_dir / name
        config_path = project_dir / "config.yml"

        # Create wrapper with verbose flag
        project = E2EProject(project_dir, config_path, verbose=verbose)
        projects.append(project)

        return project

    yield _create_project

    # Cleanup all projects
    for project in projects:
        project.cleanup()


@pytest.fixture
def llm_judge(request):
    """Fixture providing an LLM judge for evaluation.

    Can be configured via pytest command line:
        pytest --judge-provider=cborg --judge-model=anthropic/claude-haiku
    """
    provider = request.config.getoption("--judge-provider", default="cborg")
    model = request.config.getoption("--judge-model", default="anthropic/claude-haiku")
    verbose = request.config.getoption("--judge-verbose", default=False)

    return LLMJudge(
        provider=provider,
        model=model,
        verbose=verbose
    )


def pytest_addoption(parser):
    """Add custom command-line options for E2E tests."""
    parser.addoption(
        "--judge-provider",
        action="store",
        default="cborg",
        help="AI provider to use for LLM judge evaluation"
    )
    parser.addoption(
        "--judge-model",
        action="store",
        default="anthropic/claude-haiku",
        help="Model to use for LLM judge evaluation"
    )
    parser.addoption(
        "--judge-verbose",
        action="store_true",
        default=False,
        help="Print detailed judge evaluation information"
    )
    parser.addoption(
        "--e2e-verbose",
        action="store_true",
        default=False,
        help="Show real-time progress updates during E2E test execution"
    )


def pytest_configure(config):
    """Register custom markers for E2E tests."""
    config.addinivalue_line(
        "markers",
        "e2e: End-to-end workflow tests (requires API keys, slow)"
    )
    config.addinivalue_line(
        "markers",
        "e2e_smoke: Quick smoke tests for critical workflows"
    )
    config.addinivalue_line(
        "markers",
        "e2e_tutorial: Tutorial workflow validation tests"
    )
    config.addinivalue_line(
        "markers",
        "e2e_benchmark: Channel finder benchmark validation tests"
    )

