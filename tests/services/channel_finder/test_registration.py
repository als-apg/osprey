"""
Tests for Channel Finder pluggable pipeline and database registration system.

Tests:
- Pipeline registration (valid and invalid)
- Database registration (valid and invalid)
- Discovery/listing functions
- Custom pipeline/database initialization
- Error handling
"""

import pytest
import json
import tempfile
from pathlib import Path
from typing import List, Dict, Optional, Any

# Import the service and base classes
import sys
from pathlib import Path

# Add template directory to path for testing
template_dir = Path(__file__).parent.parent.parent.parent / 'src' / 'osprey' / 'templates' / 'apps' / 'control_assistant'
sys.path.insert(0, str(template_dir))

from services.channel_finder import ChannelFinderService
from services.channel_finder.core.base_pipeline import BasePipeline
from services.channel_finder.core.base_database import BaseDatabase
from services.channel_finder.core.models import ChannelFinderResult, ChannelInfo


# ============================================================================
# Mock Implementations for Testing
# ============================================================================

class MockCustomPipeline(BasePipeline):
    """Mock custom pipeline for testing."""

    def __init__(self, database, model_config: dict, custom_param: str = "default", **kwargs):
        super().__init__(database, model_config, **kwargs)
        self.custom_param = custom_param

    @property
    def pipeline_name(self) -> str:
        return "Mock Custom Pipeline"

    async def process_query(self, query: str) -> ChannelFinderResult:
        """Simple mock implementation."""
        return ChannelFinderResult(
            query=query,
            channels=[],
            total_channels=0,
            processing_notes=f"Mock pipeline processed with param: {self.custom_param}"
        )

    def get_statistics(self) -> Dict[str, Any]:
        return {
            'pipeline_type': 'mock_custom',
            'custom_param': self.custom_param
        }


class MockCustomDatabase(BaseDatabase):
    """Mock custom database for testing."""

    def __init__(self, db_path: str, custom_option: str = "default", **kwargs):
        self.custom_option = custom_option
        super().__init__(db_path)

    def load_database(self):
        """Load mock data."""
        self.channels = [
            {
                'channel': 'MockChannel1',
                'address': 'MOCK:CH1',
                'description': 'Mock channel 1'
            },
            {
                'channel': 'MockChannel2',
                'address': 'MOCK:CH2',
                'description': 'Mock channel 2'
            }
        ]
        self.channel_map = {ch['channel']: ch for ch in self.channels}

    def get_all_channels(self) -> List[Dict]:
        return self.channels

    def get_channel(self, channel_name: str) -> Optional[Dict]:
        return self.channel_map.get(channel_name)

    def validate_channel(self, channel_name: str) -> bool:
        return channel_name in self.channel_map

    def get_statistics(self) -> Dict:
        return {
            'total_channels': len(self.channels),
            'format': 'mock_custom',
            'custom_option': self.custom_option
        }

    # InContextPipeline-specific methods
    def chunk_database(self, chunk_size: int = 50) -> List[List[Dict]]:
        return [self.channels]

    def format_chunk_for_prompt(self, chunk: List[Dict], include_addresses: bool = False) -> str:
        formatted = []
        for ch in chunk:
            if include_addresses:
                formatted.append(f"- {ch['channel']} (Address: {ch['address']}): {ch['description']}")
            else:
                formatted.append(f"- {ch['channel']}: {ch['description']}")
        return "\n".join(formatted)


class InvalidPipeline:
    """Invalid pipeline (doesn't inherit from BasePipeline)."""
    pass


class InvalidDatabase:
    """Invalid database (doesn't inherit from BaseDatabase)."""
    pass


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def temp_config_file():
    """Create temporary config file for database initialization."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump({
            'test_config': 'value',
            'custom_option': 'from_file'
        }, f)
        config_path = f.name

    yield config_path

    # Cleanup
    Path(config_path).unlink()


@pytest.fixture(autouse=True)
def reset_registries():
    """Reset custom registries before each test."""
    # Clear custom registrations
    ChannelFinderService._custom_pipelines = {}
    ChannelFinderService._custom_databases = {}
    yield
    # Clear again after test
    ChannelFinderService._custom_pipelines = {}
    ChannelFinderService._custom_databases = {}


# ============================================================================
# Pipeline Registration Tests
# ============================================================================

def test_register_custom_pipeline():
    """Test registering a valid custom pipeline."""
    # Register
    ChannelFinderService.register_pipeline('mock_custom', MockCustomPipeline)

    # Verify registration
    assert 'mock_custom' in ChannelFinderService._custom_pipelines
    assert ChannelFinderService._custom_pipelines['mock_custom'] == MockCustomPipeline


def test_register_multiple_pipelines():
    """Test registering multiple custom pipelines."""
    class AnotherPipeline(BasePipeline):
        @property
        def pipeline_name(self) -> str:
            return "Another"

        async def process_query(self, query: str) -> ChannelFinderResult:
            return ChannelFinderResult(query=query, channels=[], total_channels=0, processing_notes="")

        def get_statistics(self) -> Dict[str, Any]:
            return {}

    # Register multiple
    ChannelFinderService.register_pipeline('mock_custom', MockCustomPipeline)
    ChannelFinderService.register_pipeline('another', AnotherPipeline)

    # Verify both registered
    assert len(ChannelFinderService._custom_pipelines) == 2
    assert 'mock_custom' in ChannelFinderService._custom_pipelines
    assert 'another' in ChannelFinderService._custom_pipelines


def test_register_invalid_pipeline():
    """Test registering an invalid pipeline class (doesn't inherit from BasePipeline)."""
    with pytest.raises(TypeError, match="must inherit from BasePipeline"):
        ChannelFinderService.register_pipeline('invalid', InvalidPipeline)


def test_register_pipeline_overwrites():
    """Test that re-registering a pipeline overwrites the previous registration."""
    class FirstPipeline(BasePipeline):
        @property
        def pipeline_name(self) -> str:
            return "First"
        async def process_query(self, query: str) -> ChannelFinderResult:
            return ChannelFinderResult(query=query, channels=[], total_channels=0, processing_notes="")
        def get_statistics(self) -> Dict[str, Any]:
            return {}

    class SecondPipeline(BasePipeline):
        @property
        def pipeline_name(self) -> str:
            return "Second"
        async def process_query(self, query: str) -> ChannelFinderResult:
            return ChannelFinderResult(query=query, channels=[], total_channels=0, processing_notes="")
        def get_statistics(self) -> Dict[str, Any]:
            return {}

    # Register first
    ChannelFinderService.register_pipeline('test', FirstPipeline)
    assert ChannelFinderService._custom_pipelines['test'] == FirstPipeline

    # Register second (overwrite)
    ChannelFinderService.register_pipeline('test', SecondPipeline)
    assert ChannelFinderService._custom_pipelines['test'] == SecondPipeline


# ============================================================================
# Database Registration Tests
# ============================================================================

def test_register_custom_database():
    """Test registering a valid custom database."""
    # Register
    ChannelFinderService.register_database('mock_db', MockCustomDatabase)

    # Verify registration
    assert 'mock_db' in ChannelFinderService._custom_databases
    assert ChannelFinderService._custom_databases['mock_db'] == MockCustomDatabase


def test_register_multiple_databases():
    """Test registering multiple custom databases."""
    class AnotherDatabase(BaseDatabase):
        def load_database(self):
            pass
        def get_all_channels(self) -> List[Dict]:
            return []
        def get_channel(self, channel_name: str) -> Optional[Dict]:
            return None
        def validate_channel(self, channel_name: str) -> bool:
            return False
        def get_statistics(self) -> Dict:
            return {}

    # Register multiple
    ChannelFinderService.register_database('mock_db', MockCustomDatabase)
    ChannelFinderService.register_database('another_db', AnotherDatabase)

    # Verify both registered
    assert len(ChannelFinderService._custom_databases) == 2
    assert 'mock_db' in ChannelFinderService._custom_databases
    assert 'another_db' in ChannelFinderService._custom_databases


def test_register_invalid_database():
    """Test registering an invalid database class (doesn't inherit from BaseDatabase)."""
    with pytest.raises(TypeError, match="must inherit from BaseDatabase"):
        ChannelFinderService.register_database('invalid', InvalidDatabase)


def test_register_database_overwrites():
    """Test that re-registering a database overwrites the previous registration."""
    class FirstDatabase(BaseDatabase):
        def load_database(self):
            pass
        def get_all_channels(self) -> List[Dict]:
            return []
        def get_channel(self, channel_name: str) -> Optional[Dict]:
            return None
        def validate_channel(self, channel_name: str) -> bool:
            return False
        def get_statistics(self) -> Dict:
            return {}

    class SecondDatabase(BaseDatabase):
        def load_database(self):
            pass
        def get_all_channels(self) -> List[Dict]:
            return []
        def get_channel(self, channel_name: str) -> Optional[Dict]:
            return None
        def validate_channel(self, channel_name: str) -> bool:
            return False
        def get_statistics(self) -> Dict:
            return {}

    # Register first
    ChannelFinderService.register_database('test_db', FirstDatabase)
    assert ChannelFinderService._custom_databases['test_db'] == FirstDatabase

    # Register second (overwrite)
    ChannelFinderService.register_database('test_db', SecondDatabase)
    assert ChannelFinderService._custom_databases['test_db'] == SecondDatabase


# ============================================================================
# Discovery/Listing Tests
# ============================================================================

def test_list_available_pipelines_builtin_only():
    """Test listing pipelines when only built-ins are available."""
    pipelines = ChannelFinderService.list_available_pipelines()

    # Should have built-in pipelines
    assert 'in_context' in pipelines
    assert 'hierarchical' in pipelines
    assert 'Built-in' in pipelines['in_context']
    assert 'Built-in' in pipelines['hierarchical']


def test_list_available_pipelines_with_custom():
    """Test listing pipelines after registering custom ones."""
    # Register custom
    ChannelFinderService.register_pipeline('mock_custom', MockCustomPipeline)

    pipelines = ChannelFinderService.list_available_pipelines()

    # Should have both built-in and custom
    assert 'in_context' in pipelines
    assert 'hierarchical' in pipelines
    assert 'mock_custom' in pipelines
    assert 'Custom' in pipelines['mock_custom']


def test_list_available_databases_builtin_only():
    """Test listing databases when only built-ins are available."""
    databases = ChannelFinderService.list_available_databases()

    # Should have built-in databases
    assert 'legacy' in databases
    assert 'template' in databases
    assert 'hierarchical' in databases
    assert 'Built-in' in databases['legacy']


def test_list_available_databases_with_custom():
    """Test listing databases after registering custom ones."""
    # Register custom
    ChannelFinderService.register_database('mock_db', MockCustomDatabase)

    databases = ChannelFinderService.list_available_databases()

    # Should have both built-in and custom
    assert 'legacy' in databases
    assert 'template' in databases
    assert 'hierarchical' in databases
    assert 'mock_db' in databases
    assert 'Custom' in databases['mock_db']


# ============================================================================
# Integration Tests (Custom Pipeline/Database Usage)
# ============================================================================

@pytest.mark.asyncio
async def test_custom_pipeline_initialization(temp_config_file, monkeypatch):
    """Test that a custom pipeline can be initialized and used."""
    # Register custom pipeline and database
    ChannelFinderService.register_pipeline('mock_custom', MockCustomPipeline)
    ChannelFinderService.register_database('mock_db', MockCustomDatabase)

    # Mock config
    mock_config = {
        'channel_finder': {
            'pipeline_mode': 'mock_custom',
            'pipelines': {
                'mock_custom': {
                    'database': {
                        'type': 'mock_db',
                        'path': temp_config_file,
                        'custom_option': 'test_value'
                    },
                    'processing': {
                        'custom_param': 'test_param'
                    }
                }
            }
        },
        'facility': {'name': 'Test Facility'},
        'project_root': str(Path(temp_config_file).parent)
    }

    # Mock configuration system
    from unittest.mock import Mock
    mock_config_builder = Mock()
    mock_config_builder.raw_config = mock_config
    mock_config_builder.get = lambda key, default=None: {
        'channel_finder.pipeline_mode': 'mock_custom',
        'model.provider': 'mock',
        'model.model_id': 'mock-model',
        'model.max_tokens': 1000,
        'project_root': str(Path(temp_config_file).parent)
    }.get(key, default)

    def mock_get_config():
        return mock_config_builder

    def mock_get_provider_config(provider):
        return {}

    # Patch config functions
    import services.channel_finder.service as service_module
    monkeypatch.setattr(service_module, '_get_config', mock_get_config)
    monkeypatch.setattr(service_module, 'get_provider_config', mock_get_provider_config)

    # Mock prompt loader
    mock_prompts = Mock()
    monkeypatch.setattr(service_module, 'load_prompts', lambda config: mock_prompts)

    # Initialize service with custom pipeline
    service = ChannelFinderService(pipeline_mode='mock_custom')

    # Verify custom pipeline was initialized
    assert isinstance(service.pipeline, MockCustomPipeline)
    assert service.pipeline.custom_param == 'test_param'

    # Verify custom database was initialized
    assert isinstance(service.pipeline.database, MockCustomDatabase)
    assert service.pipeline.database.custom_option == 'test_value'

    # Test query processing
    result = await service.find_channels("test query")
    assert isinstance(result, ChannelFinderResult)
    assert "test_param" in result.processing_notes


@pytest.mark.asyncio
async def test_custom_database_with_builtin_pipeline(temp_config_file, monkeypatch):
    """Test using a custom database with a built-in pipeline."""
    # Register only custom database
    ChannelFinderService.register_database('mock_db', MockCustomDatabase)

    # Mock config for in_context pipeline with custom database
    mock_config = {
        'channel_finder': {
            'pipeline_mode': 'in_context',
            'pipelines': {
                'in_context': {
                    'database': {
                        'type': 'mock_db',
                        'path': temp_config_file,
                        'custom_option': 'builtin_test'
                    },
                    'processing': {
                        'chunk_dictionary': False,
                        'max_correction_iterations': 2
                    }
                }
            }
        },
        'facility': {'name': 'Test Facility'},
        'project_root': str(Path(temp_config_file).parent)
    }

    # Mock configuration system
    from unittest.mock import Mock
    mock_config_builder = Mock()
    mock_config_builder.raw_config = mock_config
    mock_config_builder.get = lambda key, default=None: {
        'channel_finder.pipeline_mode': 'in_context',
        'model.provider': 'mock',
        'model.model_id': 'mock-model',
        'model.max_tokens': 1000,
        'project_root': str(Path(temp_config_file).parent)
    }.get(key, default)

    def mock_get_config():
        return mock_config_builder

    def mock_get_provider_config(provider):
        return {}

    # Patch config functions at service level
    import services.channel_finder.service as service_module
    monkeypatch.setattr(service_module, '_get_config', mock_get_config)
    monkeypatch.setattr(service_module, 'get_provider_config', mock_get_provider_config)

    # Also patch config at pipeline level (InContextPipeline imports _get_config)
    import services.channel_finder.pipelines.in_context.pipeline as pipeline_module
    monkeypatch.setattr(pipeline_module, '_get_config', mock_get_config)

    # Mock prompt loader
    from services.channel_finder.core.models import QuerySplitterOutput, ChannelMatchOutput, ChannelCorrectionOutput

    mock_prompts = Mock()
    mock_prompts.query_splitter = Mock()
    mock_prompts.query_splitter.get_prompt = Mock(return_value="Query split prompt")
    mock_prompts.channel_matcher = Mock()
    mock_prompts.channel_matcher.get_prompt = Mock(return_value="Channel match prompt")
    mock_prompts.correction = Mock()
    mock_prompts.correction.get_prompt = Mock(return_value="Correction prompt")

    # Add system.facility_description attribute (empty string is fine)
    mock_prompts.system = Mock()
    mock_prompts.system.facility_description = ""

    monkeypatch.setattr(service_module, 'load_prompts', lambda config: mock_prompts)
    monkeypatch.setattr(pipeline_module, 'load_prompts', lambda config: mock_prompts)

    # Initialize service
    service = ChannelFinderService(pipeline_mode='in_context')

    # Verify built-in pipeline with custom database
    from services.channel_finder.pipelines.in_context import InContextPipeline
    assert isinstance(service.pipeline, InContextPipeline)
    assert isinstance(service.pipeline.database, MockCustomDatabase)
    assert service.pipeline.database.custom_option == 'builtin_test'


# ============================================================================
# Error Handling Tests
# ============================================================================

def test_unknown_pipeline_error(temp_config_file, monkeypatch):
    """Test error when requesting unknown pipeline."""
    # Mock config
    mock_config = {
        'channel_finder': {
            'pipeline_mode': 'nonexistent',
            'pipelines': {
                'nonexistent': {
                    'database': {
                        'type': 'template',
                        'path': temp_config_file
                    }
                }
            }
        },
        'facility': {'name': 'Test'},
        'project_root': str(Path(temp_config_file).parent)
    }

    from unittest.mock import Mock
    mock_config_builder = Mock()
    mock_config_builder.raw_config = mock_config
    mock_config_builder.get = lambda key, default=None: {
        'channel_finder.pipeline_mode': 'nonexistent',
        'model.provider': 'mock',
        'model.model_id': 'mock',
        'model.max_tokens': 1000,
        'project_root': str(Path(temp_config_file).parent)
    }.get(key, default)

    import services.channel_finder.service as service_module
    monkeypatch.setattr(service_module, '_get_config', lambda: mock_config_builder)
    monkeypatch.setattr(service_module, 'get_provider_config', lambda p: {})

    from services.channel_finder.core.exceptions import PipelineModeError

    with pytest.raises(PipelineModeError, match="Unknown pipeline mode: 'nonexistent'"):
        ChannelFinderService(pipeline_mode='nonexistent')


def test_unknown_database_error(temp_config_file, monkeypatch):
    """Test error when requesting unknown database type."""
    # Mock config with unknown database type
    mock_config = {
        'channel_finder': {
            'pipeline_mode': 'in_context',
            'pipelines': {
                'in_context': {
                    'database': {
                        'type': 'nonexistent_db',
                        'path': temp_config_file
                    },
                    'processing': {}
                }
            }
        },
        'facility': {'name': 'Test'},
        'project_root': str(Path(temp_config_file).parent)
    }

    from unittest.mock import Mock
    mock_config_builder = Mock()
    mock_config_builder.raw_config = mock_config
    mock_config_builder.get = lambda key, default=None: {
        'channel_finder.pipeline_mode': 'in_context',
        'model.provider': 'mock',
        'model.model_id': 'mock',
        'model.max_tokens': 1000,
        'project_root': str(Path(temp_config_file).parent)
    }.get(key, default)

    import services.channel_finder.service as service_module
    monkeypatch.setattr(service_module, '_get_config', lambda: mock_config_builder)
    monkeypatch.setattr(service_module, 'get_provider_config', lambda p: {})
    monkeypatch.setattr(service_module, 'load_prompts', lambda c: Mock())

    from services.channel_finder.core.exceptions import ConfigurationError

    with pytest.raises(ConfigurationError, match="Unknown database type: 'nonexistent_db'"):
        ChannelFinderService(pipeline_mode='in_context')


if __name__ == "__main__":
    pytest.main([__file__, '-v'])

