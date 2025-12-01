"""
Pytest configuration and shared test utilities.

This module provides shared fixtures and utilities for all Osprey tests.
"""

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from osprey.base.planning import ExecutionPlan, PlannedStep
from osprey.state import AgentState

# ===================================================================
# Auto-reset Registry and Config Between Tests
# ===================================================================

@pytest.fixture(autouse=True, scope="function")
def reset_state_between_tests():
    """Auto-reset registry and config before each test to ensure isolation.

    This prevents state leakage between tests by:
    - Resetting the registry
    - Clearing config caches

    Note: Does NOT clear CONFIG_FILE env var since test fixtures may set it.
    Tests that need a clean CONFIG_FILE state should handle it explicitly.
    """
    # Reset before test
    from osprey.registry import reset_registry
    from osprey.utils import config as config_module

    reset_registry()
    config_module._default_config = None
    config_module._default_configurable = None
    config_module._config_cache.clear()

    yield

    # Reset after test
    reset_registry()
    config_module._default_config = None
    config_module._default_configurable = None
    config_module._config_cache.clear()

# ===================================================================
# Test State Factory
# ===================================================================

def create_test_state(
    user_message: str = "test query",
    task_objective: str = "test objective",
    conversation_history: list[tuple[str, str]] | None = None,
    capability: str = "test_capability",
    context_key: str = "test_context",
    final_objective: str = "Test objective",
    success_criteria: str = "Task completed successfully",
    **overrides
) -> AgentState:
    """Factory function to create test states with minimal boilerplate.

    This is a global test utility that can be used across all test modules
    to create AgentState objects with sensible defaults, reducing test
    verbosity and maintenance burden.

    Args:
        user_message: The user's message (used if no conversation_history provided)
        task_objective: The task objective for the planned step
        conversation_history: List of (role, content) tuples for multi-turn conversations
                            role should be 'user' or 'ai'
        capability: Capability name for the planned step
        context_key: Context key for the planned step
        final_objective: The execution plan's final objective
        success_criteria: Success criteria for the planned step
        **overrides: Any additional state fields to override

    Returns:
        Complete AgentState with sensible defaults for testing

    Examples:
        Simple single-message state::

            state = create_test_state(
                user_message="whats the weather?",
                task_objective="Ask for location"
            )

        Multi-turn conversation::

            state = create_test_state(
                conversation_history=[
                    ("user", "I need data"),
                    ("ai", "What kind of data?"),
                    ("user", "beam current")
                ],
                task_objective="Ask for time range"
            )

        With capability-specific settings::

            state = create_test_state(
                user_message="fetch data",
                capability="data_retrieval",
                context_key="fetch_pv_data",
                task_depends_on_chat_history=True
            )
    """
    # Build messages from conversation history or single message
    messages = []
    if conversation_history:
        for role, content in conversation_history:
            if role == "user":
                messages.append(HumanMessage(content=content))
            else:
                messages.append(AIMessage(content=content))
    else:
        messages = [HumanMessage(content=user_message)]

    # Create execution plan
    execution_plan = ExecutionPlan(
        steps=[
            PlannedStep(
                context_key=context_key,
                capability=capability,
                task_objective=task_objective,
                success_criteria=success_criteria,
                expected_output=None,
                inputs=[]
            )
        ],
        final_objective=final_objective
    )

    # Create state with defaults
    state: AgentState = {
        'messages': messages,
        'planning_execution_plan': execution_plan,
        'planning_current_step_index': 0,
        'capability_context_data': {},
        'agent_control': {},
        'status_updates': [],
        'progress_events': [],
        'task_current_task': 'Test task',
        'task_depends_on_chat_history': False,
        'task_depends_on_user_memory': False,
        'task_custom_message': None,
        'planning_active_capabilities': [capability],
        'execution_step_results': {},
        'execution_last_result': None,
        'execution_pending_approvals': {},
        'execution_start_time': None,
        'execution_total_time': None,
        'approval_approved': None,
        'approved_payload': None,
        'control_reclassification_reason': None,
        'control_reclassification_count': 0,
        'control_plans_created_count': 1,
        'control_current_step_retry_count': 0,
        'control_retry_count': 0,
        'control_has_error': False,
        'control_error_info': None,
        'control_last_error': None,
        'control_max_retries': 3,
        'control_is_killed': False,
        'control_kill_reason': None,
        'control_is_awaiting_validation': False,
        'control_validation_context': None,
        'control_validation_timestamp': None,
        'ui_captured_notebooks': [],
        'ui_captured_figures': [],
        'ui_launchable_commands': [],
        'ui_agent_context': None,
        'runtime_checkpoint_metadata': None,
        'runtime_info': None,
    }

    # Apply any overrides
    state.update(overrides)

    return state


# ===================================================================
# Prompt Testing Helpers
# ===================================================================

class PromptTestHelpers:
    """Helper methods for testing prompt structure and content."""

    @staticmethod
    def extract_section(prompt: str, section_header: str) -> str:
        """Extract a specific section from the prompt by its header.

        Args:
            prompt: The full prompt text
            section_header: The header marking the start of the section

        Returns:
            The extracted section content (without the header)
        """
        lines = prompt.split('\n')
        section_lines = []
        in_section = False

        for line in lines:
            if section_header in line:
                in_section = True
                continue
            if in_section:
                # Stop at next all-caps header with colon
                if line.strip() and line.strip().replace(' ', '').replace("'", '').isupper() and ':' in line:
                    break
                section_lines.append(line)

        return '\n'.join(section_lines).strip()

    @staticmethod
    def get_section_positions(prompt: str, *section_headers: str) -> dict[str, int]:
        """Get the positions of multiple section headers in the prompt.

        Args:
            prompt: The full prompt text
            *section_headers: Variable number of section headers to find

        Returns:
            Dictionary mapping section headers to their positions (-1 if not found)
        """
        positions = {}
        for header in section_headers:
            try:
                positions[header] = prompt.index(header)
            except ValueError:
                positions[header] = -1
        return positions


# ===================================================================
# Pytest Fixtures
# ===================================================================

@pytest.fixture
def test_state():
    """Fixture providing a basic test state for quick testing."""
    return create_test_state()


@pytest.fixture
def prompt_helpers():
    """Fixture providing prompt testing helper methods."""
    return PromptTestHelpers


# ===================================================================
# Test Configuration Helpers
# ===================================================================

@pytest.fixture
def test_config(tmp_path):
    """Fixture providing a minimal test configuration file.

    Creates a temporary config.yml with sensible defaults for testing.
    Tests can override or extend this as needed.

    Returns:
        Path to the created config file

    Examples:
        Basic usage::

            def test_something(test_config):
                # Config file exists at test_config
                assert test_config.exists()
                # Set environment to use it
                os.environ['CONFIG_FILE'] = str(test_config)

        With custom modifications::

            def test_custom(test_config):
                # Read existing config
                import yaml
                with open(test_config) as f:
                    config = yaml.safe_load(f)
                # Modify
                config['execution']['execution_method'] = 'container'
                # Write back
                with open(test_config, 'w') as f:
                    yaml.dump(config, f)
    """
    import yaml

    config_file = tmp_path / "config.yml"

    # Create a test registry using extend_framework_registry helper
    registry_file = tmp_path / "registry.py"
    registry_file.write_text("""
# Test registry for integration tests - extends framework with empty additions
from osprey.registry import RegistryConfigProvider, extend_framework_registry

class TestRegistryProvider(RegistryConfigProvider):
    '''Test registry provider that extends framework defaults.'''

    def get_registry_config(self):
        # Use extend_framework_registry helper - the recommended way
        # This extends the framework registry with no additions
        return extend_framework_registry(
            capabilities=[],
            context_classes=[]
        )
""")

    # Minimal working configuration for tests
    config = {
        'project_root': str(tmp_path),
        'registry_path': str(registry_file),
        'langgraph': {
            'use_postgres': False
        },
        'execution_control': {
            'epics': {
                'writes_enabled': False
            },
            'agent_control': {
                'task_extraction_bypass_enabled': False,
                'capability_selection_bypass_enabled': False
            },
            'limits': {
                'max_reclassifications': 1,
                'max_planning_attempts': 2,
                'max_step_retries': 3,
                'max_execution_time_seconds': 300,
                'graph_recursion_limit': 100,
                'max_concurrent_classifications': 5
            }
        },
        'approval': {
            'global_mode': 'selective',
            'capabilities': {
                'python_execution': {
                    'enabled': False,  # Disable approval for tests
                    'mode': 'all_code'
                },
                'memory': {
                    'enabled': False  # Disable approval for tests
                }
            }
        },
        'control_system': {
            'type': 'epics',
            'patterns': {
                'epics': {
                    'write': [
                        r'\bcaput\s*\(',
                        r'epics\.caput\(',
                        r'\.put\s*\(',
                    ],
                    'read': [
                        r'\bcaget\s*\(',
                        r'epics\.caget\(',
                        r'\.get\s*\(',
                    ]
                },
                'mock': {
                    'write': [r'\.caput\(', r'\.write_pv\('],
                    'read': [r'\.caget\(', r'\.read_pv\(']
                }
            }
        },
        'execution': {
            'execution_method': 'local',  # Fast for tests
            'code_generator': 'mock',  # Use mock by default for tests
            'generators': {
                'legacy': {
                    'provider': 'openai',
                    'model_id': 'gpt-4'
                },
                'mock': {}  # Mock generator config
            },
            'modes': {
                'read_only': {
                    'kernel_name': 'python3',
                    'allows_writes': False,
                    'requires_approval': False
                },
                'write_access': {
                    'kernel_name': 'python3',
                    'allows_writes': True,
                    'requires_approval': True
                }
            },
            'limits': {
                'max_retries': 3,
                'max_execution_time_seconds': 30
            }
        },
        'models': {
            'orchestrator': {
                'provider': 'openai',
                'model_id': 'gpt-4'
            },
            'python_code_generator': {
                'provider': 'openai',
                'model_id': 'gpt-4'
            }
        }
    }

    with open(config_file, 'w') as f:
        yaml.dump(config, f)

    # Do NOT initialize registry here - let each test handle registry initialization
    # to avoid state pollution between tests
    return config_file


@pytest.fixture
def test_config_with_approval(tmp_path):
    """Fixture providing a test configuration WITH approval enabled.

    This is specifically for testing approval workflows.
    """
    import yaml

    config_file = tmp_path / "config.yml"

    # Create a test registry using extend_framework_registry helper
    registry_file = tmp_path / "registry.py"
    registry_file.write_text("""
# Test registry for integration tests - extends framework with empty additions
from osprey.registry import RegistryConfigProvider, extend_framework_registry

class TestRegistryProvider(RegistryConfigProvider):
    '''Test registry provider that extends framework defaults.'''

    def get_registry_config(self):
        # Use extend_framework_registry helper - the recommended way
        # This extends the framework registry with no additions
        return extend_framework_registry(
            capabilities=[],
            context_classes=[]
        )
""")

    # Configuration with approval ENABLED
    config = {
        'project_root': str(tmp_path),
        'registry_path': str(registry_file),
        'langgraph': {
            'use_postgres': False
        },
        'execution_control': {
            'epics': {
                'writes_enabled': False
            },
            'agent_control': {
                'task_extraction_bypass_enabled': False,
                'capability_selection_bypass_enabled': False
            },
            'limits': {
                'max_reclassifications': 1,
                'max_planning_attempts': 2,
                'max_step_retries': 3,
                'max_execution_time_seconds': 300,
                'graph_recursion_limit': 100,
                'max_concurrent_classifications': 5
            }
        },
        'approval': {
            'global_mode': 'selective',
            'capabilities': {
                'python_execution': {
                    'enabled': True,  # ENABLED for approval tests
                    'mode': 'all_code'
                },
                'memory': {
                    'enabled': True
                }
            }
        },
        'control_system': {
            'type': 'epics',
            'patterns': {
                'epics': {
                    'write': [
                        r'\bcaput\s*\(',
                        r'epics\.caput\(',
                        r'\.put\s*\(',
                    ],
                    'read': [
                        r'\bcaget\s*\(',
                        r'epics\.caget\(',
                        r'\.get\s*\(',
                    ]
                },
                'mock': {
                    'write': [r'\.caput\(', r'\.write_pv\('],
                    'read': [r'\.caget\(', r'\.read_pv\(']
                }
            }
        },
        'execution': {
            'execution_method': 'local',
            'code_generator': 'mock',
            'generators': {
                'legacy': {
                    'provider': 'openai',
                    'model_id': 'gpt-4'
                },
                'mock': {}
            },
            'modes': {
                'read_only': {
                    'kernel_name': 'python3',
                    'allows_writes': False,
                    'requires_approval': False
                },
                'write_access': {
                    'kernel_name': 'python3',
                    'allows_writes': True,
                    'requires_approval': True
                }
            },
            'limits': {
                'max_retries': 3,
                'max_execution_time_seconds': 30
            }
        },
        'models': {
            'orchestrator': {
                'provider': 'openai',
                'model_id': 'gpt-4'
            },
            'python_code_generator': {
                'provider': 'openai',
                'model_id': 'gpt-4'
            }
        }
    }

    with open(config_file, 'w') as f:
        yaml.dump(config, f)

    # Do NOT initialize registry here - let the test do it
    # to avoid state pollution between tests

    return config_file


@pytest.fixture
def mock_code_generator():
    """Fixture providing a MockCodeGenerator for testing.

    Creates a fresh MockCodeGenerator with success behavior.
    This is a globally-available fixture for any test that needs
    deterministic code generation.

    Returns:
        MockCodeGenerator configured for successful execution

    Examples:
        Basic usage::

            def test_something(mock_code_generator):
                code = await mock_code_generator.generate_code(request, [])
                assert 'results' in code

        With custom code::

            def test_custom(mock_code_generator):
                mock_code_generator.set_code("results = {'value': 42}")
                code = await mock_code_generator.generate_code(request, [])
                assert code == "results = {'value': 42}"
    """
    from osprey.services.python_executor.generation import MockCodeGenerator
    return MockCodeGenerator(behavior="success")

