"""Tests for extraction_review.config module."""

import logging

from extraction_review.config import create_union_schema


def test_create_union_schema_basic():
    """Test basic union schema creation with discriminator."""
    schemas = {
        "10-K": {
            "type": "object",
            "properties": {
                "company_name": {"type": "string"},
                "fiscal_year": {"type": "integer"},
            },
            "required": ["company_name"],
        },
        "8-K": {
            "type": "object",
            "properties": {
                "company_name": {"type": "string"},
                "event_date": {"type": "string"},
            },
            "required": ["company_name"],
        },
    }

    result = create_union_schema(schemas)

    assert result["type"] == "object"
    assert "document_type" in result["properties"]
    assert result["properties"]["document_type"]["enum"] == ["10-K", "8-K"]
    assert "company_name" in result["properties"]
    assert "fiscal_year" in result["properties"]
    assert "event_date" in result["properties"]
    # company_name is required in both schemas, so it should be required
    assert "document_type" in result["required"]
    assert "company_name" in result["required"]


def test_create_union_schema_no_common_required():
    """Test union schema when schemas have no common required fields."""
    schemas = {
        "TypeA": {
            "type": "object",
            "properties": {
                "field_a": {"type": "string"},
                "shared_field": {"type": "string"},
            },
            "required": ["field_a"],
        },
        "TypeB": {
            "type": "object",
            "properties": {
                "field_b": {"type": "string"},
                "shared_field": {"type": "string"},
            },
            "required": ["field_b"],
        },
    }

    result = create_union_schema(schemas)

    # Only discriminator should be required
    assert result["required"] == ["document_type"]
    assert "shared_field" in result["properties"]
    assert "field_a" in result["properties"]
    assert "field_b" in result["properties"]


def test_create_union_schema_multiple_common_required():
    """Test union schema with multiple fields required in all schemas."""
    schemas = {
        "Invoice": {
            "type": "object",
            "properties": {
                "invoice_number": {"type": "string"},
                "total_amount": {"type": "number"},
                "date": {"type": "string"},
            },
            "required": ["invoice_number", "total_amount", "date"],
        },
        "Receipt": {
            "type": "object",
            "properties": {
                "invoice_number": {"type": "string"},
                "total_amount": {"type": "number"},
                "date": {"type": "string"},
                "merchant": {"type": "string"},
            },
            "required": ["invoice_number", "total_amount", "date"],
        },
    }

    result = create_union_schema(schemas)

    # All three common fields should be required
    assert "document_type" in result["required"]
    assert "invoice_number" in result["required"]
    assert "total_amount" in result["required"]
    assert "date" in result["required"]
    assert "merchant" not in result["required"]  # Only in Receipt
    assert len(result["required"]) == 4


def test_create_union_schema_empty_required():
    """Test union schema when some schemas have no required fields."""
    schemas = {
        "TypeA": {
            "type": "object",
            "properties": {
                "field_a": {"type": "string"},
            },
            "required": [],
        },
        "TypeB": {
            "type": "object",
            "properties": {
                "field_b": {"type": "string"},
            },
            "required": [],
        },
    }

    result = create_union_schema(schemas)

    # Only discriminator should be required
    assert result["required"] == ["document_type"]


def test_create_union_schema_custom_discriminator():
    """Test union schema with custom discriminator field name."""
    schemas = {
        "A": {
            "type": "object",
            "properties": {"x": {"type": "string"}},
            "required": ["x"],
        },
        "B": {
            "type": "object",
            "properties": {"x": {"type": "string"}},
            "required": ["x"],
        },
    }

    result = create_union_schema(schemas, discriminator_field="doc_type")

    assert "doc_type" in result["properties"]
    assert "doc_type" in result["required"]
    assert result["properties"]["doc_type"]["enum"] == ["A", "B"]
    assert "x" in result["required"]


def test_create_union_schema_partial_common_required():
    """Test union schema where only some schemas require a field."""
    schemas = {
        "TypeA": {
            "type": "object",
            "properties": {
                "common_field": {"type": "string"},
                "field_a": {"type": "string"},
            },
            "required": ["common_field", "field_a"],
        },
        "TypeB": {
            "type": "object",
            "properties": {
                "common_field": {"type": "string"},
                "field_b": {"type": "string"},
            },
            "required": ["common_field"],
        },
        "TypeC": {
            "type": "object",
            "properties": {
                "common_field": {"type": "string"},
                "field_c": {"type": "string"},
            },
            "required": ["common_field"],
        },
    }

    result = create_union_schema(schemas)

    # common_field is required in all three, so it should be required
    assert "document_type" in result["required"]
    assert "common_field" in result["required"]
    # field_a is only required in TypeA, so it shouldn't be required
    assert "field_a" not in result["required"]
    assert "field_b" not in result["required"]
    assert "field_c" not in result["required"]


def test_create_union_schema_ignores_existing_discriminator(caplog):
    """Test that discriminator field in input schemas is ignored."""
    schemas = {
        "TypeA": {
            "type": "object",
            "properties": {
                "document_type": {"type": "string", "description": "Old discriminator"},
                "field_a": {"type": "string"},
            },
            "required": ["document_type", "field_a"],
        },
        "TypeB": {
            "type": "object",
            "properties": {
                "field_b": {"type": "string"},
            },
            "required": [],
        },
    }

    with caplog.at_level(logging.WARNING):
        result = create_union_schema(schemas)

    # Should have logged a warning
    assert len(caplog.records) > 0
    assert any("document_type" in record.message for record in caplog.records)
    assert any("TypeA" in record.message for record in caplog.records)

    # The old document_type property should be ignored
    # The union schema's document_type should be present with enum values
    assert "document_type" in result["properties"]
    assert result["properties"]["document_type"]["enum"] == ["TypeA", "TypeB"]
    assert (
        result["properties"]["document_type"]["description"]
        == "Type of document that was extracted"
    )
    # field_a should still be present
    assert "field_a" in result["properties"]
    # document_type should be required (the union one, not the old one)
    assert "document_type" in result["required"]


def test_create_union_schema_ignores_discriminator_in_multiple_schemas(caplog):
    """Test that discriminator field is ignored when present in multiple schemas."""
    schemas = {
        "TypeA": {
            "type": "object",
            "properties": {
                "document_type": {"type": "string"},
                "field_a": {"type": "string"},
            },
            "required": ["document_type"],
        },
        "TypeB": {
            "type": "object",
            "properties": {
                "document_type": {"type": "string"},
                "field_b": {"type": "string"},
            },
            "required": ["document_type"],
        },
    }

    with caplog.at_level(logging.WARNING):
        result = create_union_schema(schemas)

    # Should have logged warnings mentioning both schemas
    assert any("TypeA" in record.message for record in caplog.records)
    assert any("TypeB" in record.message for record in caplog.records)

    # The union schema's document_type should be present
    assert result["properties"]["document_type"]["enum"] == ["TypeA", "TypeB"]
    # document_type should be required (the union one)
    assert "document_type" in result["required"]
    # field_a and field_b should be present
    assert "field_a" in result["properties"]
    assert "field_b" in result["properties"]


def test_create_union_schema_custom_discriminator_ignored(caplog):
    """Test that custom discriminator field in input schemas is ignored."""
    schemas = {
        "A": {
            "type": "object",
            "properties": {
                "doc_type": {"type": "string", "description": "Existing field"},
                "x": {"type": "string"},
            },
            "required": ["doc_type"],
        },
        "B": {
            "type": "object",
            "properties": {
                "x": {"type": "string"},
            },
            "required": [],
        },
    }

    with caplog.at_level(logging.WARNING):
        result = create_union_schema(schemas, discriminator_field="doc_type")

    # Should have logged a warning
    assert len(caplog.records) > 0
    assert any("doc_type" in record.message for record in caplog.records)
    assert any("A" in record.message for record in caplog.records)

    # The union schema's doc_type should be present with enum values
    assert "doc_type" in result["properties"]
    assert result["properties"]["doc_type"]["enum"] == ["A", "B"]
    # The old doc_type property definition should be ignored
    assert (
        result["properties"]["doc_type"]["description"]
        == "Type of document that was extracted"
    )
    # doc_type should be required
    assert "doc_type" in result["required"]
