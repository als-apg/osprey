"""Tests for chat completion module."""

import pytest
from pydantic import BaseModel
from typing_extensions import TypedDict

from osprey.models.completion import (
    _convert_typed_dict_to_pydantic,
    _is_typed_dict,
)


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
