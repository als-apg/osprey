"""Tests for chat completion module."""

import pytest
from pydantic import BaseModel
from typing_extensions import TypedDict

from osprey.models.completion import (
    _convert_typed_dict_to_pydantic,
    _handle_output_conversion,
    _is_typed_dict,
    _validate_proxy_url,
)

# =============================================================================
# Test TypedDict Detection
# =============================================================================


class TestIsTypedDict:
    """Test TypedDict detection utility."""

    def test_detects_typed_dict(self):
        """Test that actual TypedDict is correctly identified."""

        class MyTypedDict(TypedDict):
            field1: str
            field2: int

        assert _is_typed_dict(MyTypedDict) is True

    def test_rejects_regular_class(self):
        """Test that regular classes are not identified as TypedDict."""

        class RegularClass:
            field1: str
            field2: int

        assert _is_typed_dict(RegularClass) is False

    def test_rejects_pydantic_model(self):
        """Test that Pydantic models are not identified as TypedDict."""

        class PydanticModel(BaseModel):
            field1: str
            field2: int

        assert _is_typed_dict(PydanticModel) is False

    def test_rejects_none(self):
        """Test that None is not identified as TypedDict."""
        assert _is_typed_dict(None) is False

    def test_rejects_plain_dict(self):
        """Test that plain dict instances are not identified as TypedDict."""
        assert _is_typed_dict(dict) is False


# =============================================================================
# Test TypedDict to Pydantic Conversion
# =============================================================================


class TestConvertTypedDictToPydantic:
    """Test TypedDict to Pydantic model conversion."""

    def test_converts_simple_typed_dict(self):
        """Test conversion of simple TypedDict to Pydantic model."""

        class SimpleTypedDict(TypedDict):
            name: str
            age: int

        pydantic_model = _convert_typed_dict_to_pydantic(SimpleTypedDict)

        # Check that result is a Pydantic model
        assert issubclass(pydantic_model, BaseModel)

        # Check that fields are preserved
        assert "name" in pydantic_model.model_fields
        assert "age" in pydantic_model.model_fields

        # Check that we can instantiate it
        instance = pydantic_model(name="Alice", age=30)
        assert instance.name == "Alice"
        assert instance.age == 30

    def test_converts_nested_typed_dict(self):
        """Test conversion handles nested type annotations."""

        class AddressTypedDict(TypedDict):
            street: str
            city: str

        class PersonTypedDict(TypedDict):
            name: str
            addresses: list

        pydantic_model = _convert_typed_dict_to_pydantic(PersonTypedDict)

        # Check fields exist
        assert "name" in pydantic_model.model_fields
        assert "addresses" in pydantic_model.model_fields

        # Check instantiation
        instance = pydantic_model(name="Bob", addresses=[])
        assert instance.name == "Bob"

    def test_raises_on_non_typed_dict(self):
        """Test that conversion raises ValueError for non-TypedDict classes."""

        class NotATypedDict:
            field: str

        with pytest.raises(ValueError, match="Expected TypedDict"):
            _convert_typed_dict_to_pydantic(NotATypedDict)

    def test_model_name_has_pydantic_suffix(self):
        """Test that generated Pydantic model has 'Pydantic' suffix."""

        class MyData(TypedDict):
            value: str

        pydantic_model = _convert_typed_dict_to_pydantic(MyData)
        assert pydantic_model.__name__ == "MyDataPydantic"

    def test_preserves_field_types(self):
        """Test that field types are preserved during conversion."""

        class TypedData(TypedDict):
            text: str
            count: int
            active: bool
            items: list

        pydantic_model = _convert_typed_dict_to_pydantic(TypedData)

        # Check field types in model fields
        fields = pydantic_model.model_fields
        assert fields["text"].annotation is str
        assert fields["count"].annotation is int
        assert fields["active"].annotation is bool
        assert fields["items"].annotation is list


# =============================================================================
# Test Output Conversion Handling
# =============================================================================


class TestHandleOutputConversion:
    """Test output conversion from Pydantic to dict."""

    def test_converts_pydantic_to_dict_when_typed_dict_output(self):
        """Test Pydantic model is converted to dict when is_typed_dict_output=True."""

        class TestModel(BaseModel):
            name: str
            value: int

        result = TestModel(name="test", value=42)
        converted = _handle_output_conversion(result, is_typed_dict_output=True)

        assert isinstance(converted, dict)
        assert converted == {"name": "test", "value": 42}

    def test_preserves_pydantic_when_not_typed_dict_output(self):
        """Test Pydantic model is preserved when is_typed_dict_output=False."""

        class TestModel(BaseModel):
            name: str
            value: int

        result = TestModel(name="test", value=42)
        converted = _handle_output_conversion(result, is_typed_dict_output=False)

        assert isinstance(converted, TestModel)
        assert converted.name == "test"
        assert converted.value == 42

    def test_preserves_string_result(self):
        """Test string results pass through unchanged."""
        result = "This is a text response"
        converted = _handle_output_conversion(result, is_typed_dict_output=True)
        assert converted == result

    def test_preserves_list_result(self):
        """Test list results pass through unchanged."""
        result = ["item1", "item2", "item3"]
        converted = _handle_output_conversion(result, is_typed_dict_output=True)
        assert converted == result

    def test_preserves_dict_result(self):
        """Test dict results pass through unchanged."""
        result = {"key": "value", "number": 42}
        converted = _handle_output_conversion(result, is_typed_dict_output=True)
        assert converted == result


# =============================================================================
# Test Proxy URL Validation
# =============================================================================


class TestProxyUrlValidation:
    """Test HTTP proxy URL validation."""

    def test_validate_valid_http_proxy(self):
        """Test validation of valid HTTP proxy URL."""
        assert _validate_proxy_url("http://proxy.example.com:8080") is True

    def test_validate_valid_https_proxy(self):
        """Test validation of valid HTTPS proxy URL."""
        assert _validate_proxy_url("https://proxy.example.com:8443") is True

    def test_validate_proxy_with_auth(self):
        """Test validation of proxy URL with authentication."""
        assert _validate_proxy_url("http://user:pass@proxy.example.com:8080") is True

    def test_validate_proxy_with_ipv4(self):
        """Test validation of proxy URL with IPv4 address."""
        assert _validate_proxy_url("http://192.168.1.1:8080") is True

    def test_validate_proxy_https_default_port(self):
        """Test validation of HTTPS proxy without explicit port."""
        assert _validate_proxy_url("https://proxy.example.com") is True

    def test_validate_empty_proxy_url(self):
        """Test validation rejects empty string."""
        assert _validate_proxy_url("") is False

    def test_validate_none_proxy_url(self):
        """Test validation rejects None."""
        assert _validate_proxy_url(None) is False

    def test_validate_invalid_scheme(self):
        """Test validation rejects invalid scheme."""
        assert _validate_proxy_url("ftp://proxy.example.com:8080") is False

    def test_validate_socks_scheme(self):
        """Test validation rejects SOCKS proxy scheme."""
        assert _validate_proxy_url("socks5://proxy.example.com:1080") is False

    def test_validate_no_scheme(self):
        """Test validation rejects URL without scheme."""
        assert _validate_proxy_url("proxy.example.com:8080") is False

    def test_validate_no_netloc(self):
        """Test validation rejects URL without netloc."""
        assert _validate_proxy_url("http://") is False

    def test_validate_malformed_url(self):
        """Test validation handles malformed URLs gracefully."""
        assert _validate_proxy_url("not a valid url at all") is False

    def test_validate_proxy_with_path(self):
        """Test validation accepts proxy URL with path component."""
        assert _validate_proxy_url("http://proxy.example.com:8080/path") is True
