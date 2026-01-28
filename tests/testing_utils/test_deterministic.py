"""Tests for the deterministic data generation utilities."""

from __future__ import annotations

import random

import pytest

from extraction_review.testing_utils._deterministic import (
    _generate_value,
    generate_data_from_schema,
)

SEED = 42


def _rng(seed: int = SEED) -> random.Random:
    return random.Random(seed)


# -- type → expected python type mapping for parametrize -----------------------

_TYPE_CASES = [
    ("integer", int),
    ("number", (int, float)),
    ("boolean", bool),
    ("string", str),
]


@pytest.mark.parametrize(
    "schema_type, expected_type",
    _TYPE_CASES,
    ids=[t for t, _ in _TYPE_CASES],
)
def test_basic_types(schema_type, expected_type):
    value = _generate_value({"type": schema_type}, _rng(), depth=0)
    assert isinstance(value, expected_type)


@pytest.mark.parametrize(
    "schema_type, expected_type",
    _TYPE_CASES,
    ids=[f"nullable_{t}" for t, _ in _TYPE_CASES],
)
def test_nullable_types(schema_type, expected_type):
    """``["<type>", "null"]`` must produce the concrete type, not a text blob."""
    value = _generate_value({"type": [schema_type, "null"]}, _rng(), depth=0)
    assert isinstance(value, expected_type)


def test_null_type():
    assert _generate_value({"type": "null"}, _rng(), depth=0) is None


def test_all_null_union_returns_none():
    assert _generate_value({"type": ["null"]}, _rng(), depth=0) is None


def test_multi_type_union_picks_first_concrete():
    value = _generate_value({"type": ["string", "integer"]}, _rng(), depth=0)
    assert isinstance(value, str)


# -- constraints survive nullable wrapping ------------------------------------


@pytest.mark.parametrize(
    "schema, lo, hi",
    [
        ({"type": "integer", "minimum": 10, "maximum": 20}, 10, 20),
        ({"type": "number", "minimum": 0.5, "maximum": 1.5}, 0.5, 1.5),
        ({"type": ["integer", "null"], "minimum": 10, "maximum": 20}, 10, 20),
        ({"type": ["number", "null"], "minimum": 0.5, "maximum": 1.5}, 0.5, 1.5),
    ],
    ids=["int", "float", "nullable_int", "nullable_float"],
)
def test_numeric_bounds(schema, lo, hi):
    value = _generate_value(schema, _rng(), depth=0)
    assert lo <= value <= hi


# -- string formats -----------------------------------------------------------


@pytest.mark.parametrize(
    "fmt, substring",
    [
        ("date-time", "T"),
        ("email", "@example.com"),
        ("uri", "https://example.com/"),
    ],
)
def test_string_formats(fmt, substring):
    value = _generate_value({"type": "string", "format": fmt}, _rng(), depth=0)
    assert isinstance(value, str)
    assert substring in value


# -- composite / container schemas --------------------------------------------


def test_enum():
    value = _generate_value({"enum": ["a", "b", "c"]}, _rng(), depth=0)
    assert value in ("a", "b", "c")


@pytest.mark.parametrize("keyword", ["oneOf", "anyOf"])
def test_composition_keywords(keyword):
    schema = {keyword: [{"type": "integer"}, {"type": "string"}]}
    value = _generate_value(schema, _rng(), depth=0)
    assert isinstance(value, (int, str))


def test_object():
    schema = {
        "type": "object",
        "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
    }
    value = _generate_value(schema, _rng(), depth=0)
    assert isinstance(value["name"], str)
    assert isinstance(value["age"], int)


def test_array():
    value = _generate_value(
        {"type": "array", "items": {"type": "integer"}, "minItems": 2, "maxItems": 4},
        _rng(),
        depth=0,
    )
    assert 2 <= len(value) <= 4
    assert all(isinstance(v, int) for v in value)


def test_nullable_object():
    schema = {
        "type": ["object", "null"],
        "properties": {"name": {"type": "string"}},
    }
    value = _generate_value(schema, _rng(), depth=0)
    assert isinstance(value, dict) and "name" in value


def test_nullable_array():
    schema = {"type": ["array", "null"], "items": {"type": "integer"}}
    value = _generate_value(schema, _rng(), depth=0)
    assert isinstance(value, list)
    assert all(isinstance(v, int) for v in value)


# -- kitchen-sink integration test --------------------------------------------


def test_mixed_nullable_object_end_to_end():
    """Full schema with nullable numerics, enum, and nested nullable fields."""
    schema = {
        "type": "object",
        "properties": {
            "total_revenue": {"type": ["number", "null"]},
            "employee_count": {"type": ["integer", "null"]},
            "filing_type": {"type": "string", "enum": ["10-K", "10-Q", "8-K"]},
            "is_audited": {"type": ["boolean", "null"]},
            "scores": {
                "type": ["array", "null"],
                "items": {"type": ["number", "null"]},
            },
            "metadata": {
                "type": ["object", "null"],
                "properties": {"source": {"type": ["string", "null"]}},
            },
        },
    }
    data = generate_data_from_schema(schema, seed=SEED)

    assert isinstance(data["total_revenue"], (int, float))
    assert isinstance(data["employee_count"], int)
    assert data["filing_type"] in ("10-K", "10-Q", "8-K")
    assert isinstance(data["is_audited"], bool)
    assert all(isinstance(s, (int, float)) for s in data["scores"])
    assert isinstance(data["metadata"]["source"], str)

    # deterministic
    assert generate_data_from_schema(schema, seed=SEED) == data


# -- edge cases ---------------------------------------------------------------


def test_depth_limit_returns_primitive():
    value = _generate_value({"type": "object", "properties": {}}, _rng(), depth=9)
    assert isinstance(value, (int, float, str))


def test_none_schema():
    assert isinstance(_generate_value(None, _rng(), depth=0), str)


def test_bare_string_schema():
    value = _generate_value("some_type", _rng(), depth=0)
    assert value.startswith("some_type-")


def test_list_schema():
    value = _generate_value([{"type": "integer"}, {"type": "string"}], _rng(), depth=0)
    assert isinstance(value, list) and len(value) == 2


def test_empty_enum_falls_through():
    value = _generate_value({"enum": [], "type": "string"}, _rng(), depth=0)
    assert isinstance(value, str)


def test_unknown_mapping_falls_through():
    value = _generate_value({"description": "mystery"}, _rng(), depth=0)
    assert isinstance(value, str)
