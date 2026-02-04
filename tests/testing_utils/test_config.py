"""Tests for JSON schema utilities."""

import logging

import pytest

from extraction_review.config import create_union_schema, get_extraction_schema


def test_union_schema_merges_properties_and_required():
    """Merges properties, discriminator required, common fields required."""
    result = create_union_schema(
        {
            "A": {
                "type": "object",
                "properties": {"x": {"type": "string"}},
                "required": ["x"],
            },
            "B": {
                "type": "object",
                "properties": {"x": {"type": "string"}, "y": {"type": "string"}},
                "required": ["x"],
            },
        }
    )

    assert result["properties"]["document_type"]["enum"] == ["A", "B"]
    assert "x" in result["properties"]
    assert "y" in result["properties"]
    assert result["required"] == ["document_type", "x"]


def test_custom_discriminator_field():
    result = create_union_schema(
        {"A": {"type": "object", "properties": {}, "required": []}},
        discriminator_field="kind",
    )
    assert "kind" in result["properties"]
    assert "kind" in result["required"]


def test_existing_discriminator_field_replaced(caplog):
    """Discriminator field in input schemas is replaced and logged."""
    with caplog.at_level(logging.WARNING):
        result = create_union_schema(
            {
                "A": {
                    "type": "object",
                    "properties": {"document_type": {"type": "string"}},
                    "required": [],
                },
            }
        )

    assert any("document_type" in r.message for r in caplog.records)
    assert result["properties"]["document_type"]["enum"] == ["A"]


def test_conflicting_types_become_anyof():
    """Same field with different types becomes anyOf."""
    result = create_union_schema(
        {
            "A": {
                "type": "object",
                "properties": {"val": {"type": "string"}},
                "required": [],
            },
            "B": {
                "type": "object",
                "properties": {"val": {"type": "number"}},
                "required": [],
            },
        }
    )

    assert "anyOf" in result["properties"]["val"]
    assert {"type": "string"} in result["properties"]["val"]["anyOf"]
    assert {"type": "number"} in result["properties"]["val"]["anyOf"]


def test_identical_fields_no_anyof():
    """Identical field definitions stay as-is."""
    result = create_union_schema(
        {
            "A": {
                "type": "object",
                "properties": {"x": {"type": "string"}},
                "required": [],
            },
            "B": {
                "type": "object",
                "properties": {"x": {"type": "string"}},
                "required": [],
            },
        }
    )

    assert result["properties"]["x"] == {"type": "string"}
    assert "anyOf" not in result["properties"]["x"]


def test_duplicate_types_not_repeated_in_anyof():
    """Third schema with same type as first doesn't add duplicate."""
    result = create_union_schema(
        {
            "A": {
                "type": "object",
                "properties": {"x": {"type": "string"}},
                "required": [],
            },
            "B": {
                "type": "object",
                "properties": {"x": {"type": "number"}},
                "required": [],
            },
            "C": {
                "type": "object",
                "properties": {"x": {"type": "string"}},
                "required": [],
            },
        }
    )

    assert len(result["properties"]["x"]["anyOf"]) == 2


# Tests for get_extraction_schema


def test_get_extraction_schema_creates_pydantic_model():
    """Creates a Pydantic model from JSON schema."""
    schema = {
        "type": "object",
        "properties": {"name": {"type": "string"}},
        "required": ["name"],
    }
    model_class = get_extraction_schema(schema)

    instance = model_class(name="test")
    assert instance.name == "test"


def test_get_extraction_schema_with_discriminator():
    """Adds discriminator field with default value."""
    schema = {
        "type": "object",
        "properties": {"name": {"type": "string"}},
        "required": ["name"],
    }
    model_class = get_extraction_schema(
        schema,
        discriminator_field="doc_type",
        discriminator_value="invoice",
    )

    # Discriminator has default, so we don't need to pass it
    instance = model_class(name="test")
    assert instance.name == "test"
    assert instance.doc_type == "invoice"


def test_get_extraction_schema_discriminator_requires_value():
    """Raises error if discriminator_field set without value."""
    schema = {"type": "object", "properties": {}, "required": []}

    with pytest.raises(ValueError, match="discriminator_value is required"):
        get_extraction_schema(schema, discriminator_field="doc_type")


def test_get_extraction_schema_caches_results():
    """Same schema returns same model class."""
    schema = {
        "type": "object",
        "properties": {"x": {"type": "string"}},
        "required": [],
    }

    model1 = get_extraction_schema(schema)
    model2 = get_extraction_schema(schema)

    assert model1 is model2
