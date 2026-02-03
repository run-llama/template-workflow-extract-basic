"""Utilities for working with JSON schemas."""

import json
import logging
from typing import Any

logger = logging.getLogger(__name__)


def _schemas_are_equal(schema1: dict[str, Any], schema2: dict[str, Any]) -> bool:
    """Check if two JSON schemas are structurally equal."""
    return json.dumps(schema1, sort_keys=True) == json.dumps(schema2, sort_keys=True)


def _merge_property_schemas(
    existing: dict[str, Any], new: dict[str, Any]
) -> dict[str, Any]:
    """Merge two property schemas, creating anyOf for conflicting types."""
    if _schemas_are_equal(existing, new):
        return existing

    if "anyOf" in existing:
        for variant in existing["anyOf"]:
            if _schemas_are_equal(variant, new):
                return existing
        return {"anyOf": existing["anyOf"] + [new]}

    return {"anyOf": [existing, new]}


def create_union_schema(
    schemas: dict[str, dict[str, Any]],
    discriminator_field: str = "document_type",
) -> dict[str, Any]:
    """Create a union JSON schema from multiple extraction schemas.

    Merges all properties from input schemas into a single flat schema,
    adding a discriminator field. When the same field appears with different
    types across schemas, creates an anyOf to accommodate both.

    Args:
        schemas: Map of document type names to their JSON schemas.
        discriminator_field: Name of the field to add for document type.

    Returns:
        A JSON schema with all fields from all schemas plus a discriminator.
    """
    schemas_with_discriminator = [
        name
        for name, schema in schemas.items()
        if discriminator_field in schema.get("properties", {})
    ]
    if schemas_with_discriminator:
        logger.warning(
            f"Discriminator field '{discriminator_field}' found in schemas: "
            f"{', '.join(schemas_with_discriminator)}. "
            f"It will be replaced with the union discriminator."
        )

    all_properties: dict[str, Any] = {}
    for schema in schemas.values():
        for prop_name, prop_def in schema.get("properties", {}).items():
            if prop_name == discriminator_field:
                continue
            if prop_name not in all_properties:
                all_properties[prop_name] = prop_def
            else:
                all_properties[prop_name] = _merge_property_schemas(
                    all_properties[prop_name], prop_def
                )

    schema_list = list(schemas.values())
    common_required: set[str] = set()
    if schema_list:
        common_required = set(schema_list[0].get("required", []))
        for schema in schema_list[1:]:
            common_required &= set(schema.get("required", []))

    common_required.discard(discriminator_field)

    required_fields = [discriminator_field] + sorted(common_required)
    return {
        "type": "object",
        "properties": {
            discriminator_field: {
                "type": "string",
                "enum": list(schemas.keys()),
                "description": "Type of document that was extracted",
            },
            **all_properties,
        },
        "required": required_fields,
    }
