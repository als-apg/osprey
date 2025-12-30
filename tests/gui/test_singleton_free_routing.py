"""
Test singleton-free routing with SimpleLLMClient.

This test verifies that the routing system works without requiring
the global registry singleton to be initialized.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from osprey.interfaces.pyqt.llm_client import SimpleLLMClient
from osprey.interfaces.pyqt.multi_project_router import MultiProjectRouter


def test_simple_llm_client_initialization():
    """Test SimpleLLMClient can be initialized without singleton."""
    # Should not raise "Registry not initialized" error
    client = SimpleLLMClient(
        provider='anthropic',
        model_id='claude-3-sonnet-20240229',
        api_key='test-key'
    )
    
    assert client.provider == 'anthropic'
    assert client.model_id == 'claude-3-sonnet-20240229'
    assert client.api_key == 'test-key'


def test_simple_llm_client_from_gui_config():
    """Test SimpleLLMClient can read from gui_config.yml."""
    # Mock the gui_config.yml file
    mock_config = {
        'models': {
            'classifier': {
                'provider': 'argo',
                'model_id': 'gpt5'
            }
        },
        'api': {
            'providers': {
                'argo': {
                    'api_key': '${ARGO_API_KEY}',
                    'base_url': 'https://argo-bridge.cels.anl.gov'
                }
            }
        }
    }
    
    with patch('builtins.open', create=True) as mock_open:
        with patch('yaml.safe_load', return_value=mock_config):
            with patch('pathlib.Path.exists', return_value=True):
                with patch.dict('os.environ', {'ARGO_API_KEY': 'test-argo-key'}):
                    client = SimpleLLMClient.from_gui_config()
                    
                    assert client.provider == 'argo'
                    assert client.model_id == 'gpt5'
                    assert client.api_key == 'test-argo-key'
                    assert client.base_url == 'https://argo-bridge.cels.anl.gov'


def test_router_without_singleton():
    """Test MultiProjectRouter works without global registry singleton."""
    # Create mock capability registry
    mock_registry = Mock()
    mock_registry.get_capabilities_by_project.return_value = {}
    mock_registry.get_capability_description.return_value = "Test capability"
    
    # Create router with explicit LLM config (no singleton needed!)
    router = MultiProjectRouter(
        capability_registry=mock_registry,
        llm_config={
            'provider': 'anthropic',
            'model_id': 'claude-3-sonnet-20240229',
            'api_key': 'test-key'
        }
    )
    
    # Verify LLM client was created
    assert router.llm_client is not None
    assert router.llm_client.provider == 'anthropic'
    assert router.llm_client.model_id == 'claude-3-sonnet-20240229'


def test_router_llm_call_without_singleton():
    """Test router can make LLM calls without singleton."""
    # Create mock capability registry
    mock_registry = Mock()
    mock_registry.get_capabilities_by_project.return_value = {}
    mock_registry.get_capability_description.return_value = "Test capability"
    
    # Create router
    router = MultiProjectRouter(
        capability_registry=mock_registry,
        llm_config={
            'provider': 'anthropic',
            'model_id': 'claude-3-sonnet-20240229',
            'api_key': 'test-key'
        }
    )
    
    # Mock the LLM client's call method
    mock_response = """PROJECT: test_project
CONFIDENCE: 0.9
REASONING: This is a test
ALTERNATIVES: other_project"""
    
    with patch.object(router.llm_client, 'call', return_value=mock_response):
        # Create TWO mock projects so LLM routing is actually used
        # (with only 1 project, router shortcuts to confidence 1.0)
        mock_project1 = Mock()
        mock_project1.metadata.name = 'test_project'
        mock_project1.metadata.description = 'Test project'
        mock_project1.metadata.version = '1.0.0'
        
        mock_project2 = Mock()
        mock_project2.metadata.name = 'other_project'
        mock_project2.metadata.description = 'Other project'
        mock_project2.metadata.version = '1.0.0'
        
        # Route query - should work without singleton!
        decision = router.route_query("test query", [mock_project1, mock_project2])
        
        assert decision.project_name == 'test_project'
        assert decision.confidence == 0.9
        assert 'test' in decision.reasoning.lower()


def test_simple_llm_client_call_anthropic():
    """Test SimpleLLMClient can call Anthropic API."""
    client = SimpleLLMClient(
        provider='anthropic',
        model_id='claude-3-sonnet-20240229',
        api_key='test-key'
    )
    
    # Mock the anthropic library
    mock_message = Mock()
    mock_message.content = [Mock(text="Test response")]
    
    mock_anthropic_client = Mock()
    mock_anthropic_client.messages.create.return_value = mock_message
    
    with patch('anthropic.Anthropic', return_value=mock_anthropic_client):
        response = client.call("Test prompt")
        
        assert response == "Test response"
        mock_anthropic_client.messages.create.assert_called_once()


def test_simple_llm_client_call_openai_compatible():
    """Test SimpleLLMClient can call OpenAI-compatible APIs (Argo, etc.)."""
    client = SimpleLLMClient(
        provider='argo',
        model_id='gpt5',
        api_key='test-key',
        base_url='https://argo-bridge.cels.anl.gov'
    )
    
    # Mock the openai library
    mock_response = Mock()
    mock_response.choices = [Mock(message=Mock(content="Test response"))]
    
    mock_openai_client = Mock()
    mock_openai_client.chat.completions.create.return_value = mock_response
    
    with patch('openai.OpenAI', return_value=mock_openai_client):
        response = client.call("Test prompt")
        
        assert response == "Test response"
        mock_openai_client.chat.completions.create.assert_called_once()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])