"""Tests for ARGO provider structured output handling."""

from unittest.mock import MagicMock, patch

import pytest
from pydantic import BaseModel

from osprey.models.providers.argo import (
    ArgoProviderAdapter,
    _clean_json_response,
    _execute_argo_structured_output,
)


class SampleOutput(BaseModel):
    """Sample structured output model for testing."""

    name: str
    value: int
    active: bool


# --- Tests for _clean_json_response ---


class TestCleanJsonResponse:
    """Tests for the _clean_json_response helper."""

    def test_plain_json_unchanged(self):
        """Plain JSON passes through unchanged."""
        raw = '{"name": "test", "value": 42}'
        assert _clean_json_response(raw) == raw

    def test_strips_json_code_fence(self):
        """Strips ```json ... ``` fences."""
        raw = '```json\n{"name": "test"}\n```'
        assert _clean_json_response(raw) == '{"name": "test"}'

    def test_strips_plain_code_fence(self):
        """Strips ``` ... ``` fences without language tag."""
        raw = '```\n{"name": "test"}\n```'
        assert _clean_json_response(raw) == '{"name": "test"}'

    def test_fixes_python_true(self):
        """Converts Python True → JSON true."""
        raw = '{"active": True}'
        result = _clean_json_response(raw)
        assert '"active": true' in result

    def test_fixes_python_false(self):
        """Converts Python False → JSON false."""
        raw = '{"active": False}'
        result = _clean_json_response(raw)
        assert '"active": false' in result

    def test_fixes_python_booleans_after_comma(self):
        """Converts Python booleans after commas."""
        raw = '{"a": True, "b": False}'
        result = _clean_json_response(raw)
        assert '"a": true' in result
        assert '"b": false' in result

    def test_extracts_json_from_surrounding_text(self):
        """Extracts JSON object when preceded by explanation text."""
        raw = 'Here is the result:\n{"name": "test", "value": 1}'
        result = _clean_json_response(raw)
        assert result == '{"name": "test", "value": 1}'

    def test_strips_whitespace(self):
        """Strips leading/trailing whitespace."""
        raw = '   \n  {"name": "test"}  \n  '
        assert _clean_json_response(raw) == '{"name": "test"}'

    def test_combined_fence_and_booleans(self):
        """Handles code fence + Python booleans together."""
        raw = '```json\n{"active": True, "deleted": False}\n```'
        result = _clean_json_response(raw)
        assert '"active": true' in result
        assert '"deleted": false' in result


# --- Tests for _execute_argo_structured_output ---


class TestExecuteArgoStructuredOutput:
    """Tests for the _execute_argo_structured_output function."""

    @patch("osprey.models.providers.argo.httpx.post")
    def test_successful_structured_output(self, mock_post):
        """Successful parse returns validated Pydantic model."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "choices": [{"message": {"content": '{"name": "test", "value": 42, "active": true}'}}]
        }
        mock_response.raise_for_status = MagicMock()
        mock_post.return_value = mock_response

        result = _execute_argo_structured_output(
            model_id="gpt5mini",
            message="Extract info",
            output_format=SampleOutput,
            api_key="test-key",
            base_url="https://test.url",
        )

        assert isinstance(result, SampleOutput)
        assert result.name == "test"
        assert result.value == 42
        assert result.active is True

    @patch("osprey.models.providers.argo.httpx.post")
    def test_typed_dict_output_returns_dict(self, mock_post):
        """When is_typed_dict_output=True, returns dict instead of model."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "choices": [{"message": {"content": '{"name": "test", "value": 1, "active": false}'}}]
        }
        mock_response.raise_for_status = MagicMock()
        mock_post.return_value = mock_response

        result = _execute_argo_structured_output(
            model_id="gpt5mini",
            message="Extract info",
            output_format=SampleOutput,
            api_key="test-key",
            base_url="https://test.url",
            is_typed_dict_output=True,
        )

        assert isinstance(result, dict)
        assert result["name"] == "test"

    @patch("osprey.models.providers.argo.httpx.post")
    def test_empty_response_raises_value_error(self, mock_post):
        """Empty API response raises ValueError."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"choices": [{"message": {"content": ""}}]}
        mock_response.raise_for_status = MagicMock()
        mock_post.return_value = mock_response

        with pytest.raises(ValueError, match="Empty response"):
            _execute_argo_structured_output(
                model_id="gpt5mini",
                message="Extract info",
                output_format=SampleOutput,
                api_key="test-key",
                base_url="https://test.url",
            )

    @patch("osprey.models.providers.argo.httpx.post")
    def test_invalid_json_raises_value_error(self, mock_post):
        """Invalid JSON in response raises ValueError."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "not valid json at all"}}]
        }
        mock_response.raise_for_status = MagicMock()
        mock_post.return_value = mock_response

        with pytest.raises(ValueError, match="Failed to parse structured output"):
            _execute_argo_structured_output(
                model_id="gpt5mini",
                message="Extract info",
                output_format=SampleOutput,
                api_key="test-key",
                base_url="https://test.url",
            )

    @patch("osprey.models.providers.argo.httpx.post")
    def test_cleans_markdown_fenced_response(self, mock_post):
        """Successfully parses response wrapped in markdown code fences."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "choices": [
                {
                    "message": {
                        "content": '```json\n{"name": "fenced", "value": 99, "active": True}\n```'
                    }
                }
            ]
        }
        mock_response.raise_for_status = MagicMock()
        mock_post.return_value = mock_response

        result = _execute_argo_structured_output(
            model_id="gpt5mini",
            message="Extract info",
            output_format=SampleOutput,
            api_key="test-key",
            base_url="https://test.url",
        )

        assert isinstance(result, SampleOutput)
        assert result.name == "fenced"
        assert result.value == 99
        assert result.active is True

    @patch("osprey.models.providers.argo.httpx.post")
    def test_uses_default_base_url(self, mock_post):
        """Falls back to default base URL when none provided."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "choices": [{"message": {"content": '{"name": "test", "value": 1, "active": true}'}}]
        }
        mock_response.raise_for_status = MagicMock()
        mock_post.return_value = mock_response

        with patch.dict("os.environ", {}, clear=True):
            _execute_argo_structured_output(
                model_id="gpt5mini",
                message="Extract info",
                output_format=SampleOutput,
                api_key="test-key",
                base_url=None,
            )

        call_url = mock_post.call_args[1].get("url") or mock_post.call_args[0][0]
        assert "argo-bridge.cels.anl.gov" in call_url

    @patch("osprey.models.providers.argo.httpx.post")
    def test_no_choices_raises_value_error(self, mock_post):
        """Response with no choices raises ValueError."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"choices": []}
        mock_response.raise_for_status = MagicMock()
        mock_post.return_value = mock_response

        with pytest.raises(ValueError, match="Empty response"):
            _execute_argo_structured_output(
                model_id="gpt5mini",
                message="Extract info",
                output_format=SampleOutput,
                api_key="test-key",
                base_url="https://test.url",
            )


# --- Tests for ArgoProviderAdapter.execute_completion with output_format ---


class TestArgoExecuteCompletionStructured:
    """Test that execute_completion routes structured output correctly."""

    @patch("osprey.models.providers.argo._execute_argo_structured_output")
    def test_routes_to_structured_handler(self, mock_structured):
        """output_format routes to direct httpx handler, not LiteLLM."""
        mock_structured.return_value = SampleOutput(name="test", value=1, active=True)

        provider = ArgoProviderAdapter()
        result = provider.execute_completion(
            message="Extract info",
            model_id="gpt5mini",
            api_key="test-key",
            base_url="https://test.url",
            output_format=SampleOutput,
        )

        mock_structured.assert_called_once()
        assert isinstance(result, SampleOutput)

    @patch("osprey.models.providers.litellm_adapter.litellm.completion")
    def test_plain_text_uses_litellm(self, mock_completion):
        """Without output_format, completion goes through LiteLLM."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="plain text"))]
        mock_completion.return_value = mock_response

        provider = ArgoProviderAdapter()
        result = provider.execute_completion(
            message="Hello",
            model_id="gpt5mini",
            api_key="test-key",
            base_url="https://test.url",
        )

        assert result == "plain text"
        mock_completion.assert_called_once()
