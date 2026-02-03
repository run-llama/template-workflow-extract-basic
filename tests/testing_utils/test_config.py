"""Tests for create_union_schema."""

import logging

from extraction_review.config import create_union_schema


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
