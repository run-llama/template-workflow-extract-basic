"""Tests for the FakeAgentDataNamespace mock implementation."""

import pytest
from llama_cloud.core.api_error import ApiError
from llama_cloud_services.beta.agent_data import AsyncAgentDataClient
from pydantic import BaseModel, Field

from extraction_review.testing_utils import FakeLlamaCloudServer
from extraction_review.testing_utils._deterministic import hash_schema


class Receipt(BaseModel):
    merchant: str = Field(description="Vendor name")
    total: float = Field(description="Grand total")


@pytest.fixture
def server():
    """Provide an installed FakeLlamaCloudServer."""
    with FakeLlamaCloudServer() as srv:
        yield srv


@pytest.fixture
def client(server):
    """Provide an AsyncAgentDataClient configured for the fake server."""
    return AsyncAgentDataClient(
        Receipt,
        collection="extracted_data",
        deployment_name="extraction_agent",
        token="fake-api-key",
    )


@pytest.mark.asyncio
async def test_create_item(server, client):
    """Verify items can be created and have expected ID format."""
    data = Receipt(merchant="Test Inc", total=1000)
    item = await client.create_item(data)

    assert item.id == hash_schema(data)[:7]
    assert item.data.merchant == data.merchant
    assert item.data.total == data.total
    assert item.collection == "extracted_data"
    assert item.deployment_name == "extraction_agent"


@pytest.mark.asyncio
async def test_update_item(server, client):
    """Verify items can be updated while preserving metadata."""
    data = Receipt(merchant="Test Inc", total=1000)
    item = await client.create_item(data)
    assert item.id is not None

    updated_data = Receipt(merchant="Testing Inc", total=1100)
    updated_item = await client.update_item(item_id=item.id, data=updated_data)

    assert updated_item.data.merchant == updated_data.merchant
    assert updated_item.data.total == updated_data.total
    assert updated_item.id == item.id
    assert updated_item.collection == item.collection
    assert updated_item.deployment_name == item.deployment_name


@pytest.mark.asyncio
async def test_search_with_eq_filter(server, client):
    """Verify search with equality filter returns matching items."""
    data1 = Receipt(merchant="Test Inc", total=1000)
    data2 = Receipt(merchant="Test Inc", total=1300)
    data3 = Receipt(merchant="Testing Inc", total=1100)
    item1 = await client.create_item(data1)
    item2 = await client.create_item(data2)
    await client.create_item(data3)

    result = await client.search(filter={"merchant": {"eq": "Test Inc"}})

    assert result.total == 2
    assert any(item.id == item1.id for item in result.items)
    assert any(item.id == item2.id for item in result.items)
    assert all(item.data.merchant == "Test Inc" for item in result.items)


@pytest.mark.asyncio
async def test_search_with_lt_filter(server, client):
    """Verify search with less-than filter returns matching items."""
    data1 = Receipt(merchant="Test Inc", total=1000)
    data2 = Receipt(merchant="Test Inc", total=1300)
    data3 = Receipt(merchant="Testing Inc", total=1100)
    item1 = await client.create_item(data1)
    await client.create_item(data2)
    item3 = await client.create_item(data3)

    result = await client.search(filter={"total": {"lt": 1200}})

    assert result.total == 2
    assert any(item.id == item1.id for item in result.items)
    assert any(item.id == item3.id for item in result.items)
    assert all(item.data.total < 1200 for item in result.items)


@pytest.mark.asyncio
async def test_aggregate_with_filter(server, client):
    """Verify aggregation with filter groups correctly."""
    data1 = Receipt(merchant="Test Inc", total=1000)
    data2 = Receipt(merchant="Test Inc", total=1300)
    data3 = Receipt(merchant="Testing Inc", total=1100)
    await client.create_item(data1)
    await client.create_item(data2)
    await client.create_item(data3)

    result = await client.aggregate(
        filter={"merchant": {"eq": "Test Inc"}},
        group_by=["merchant"],
        count=True,
    )

    # Filtering for 'Test Inc' means only one group
    assert len(result.items) == 1
    assert result.items[0].count == 2
    assert result.items[0].first_item is not None
    assert result.items[0].first_item.merchant == data1.merchant
    assert result.items[0].group_key == {"merchant": "Test Inc"}


@pytest.mark.asyncio
async def test_aggregate_without_filter(server, client):
    """Verify aggregation without filter groups all items."""
    data1 = Receipt(merchant="Test Inc", total=1000)
    data2 = Receipt(merchant="Test Inc", total=1300)
    data3 = Receipt(merchant="Testing Inc", total=1100)
    await client.create_item(data1)
    await client.create_item(data2)
    await client.create_item(data3)

    result = await client.aggregate(group_by=["merchant"], count=True)

    assert len(result.items) == 2
    # First group: Test Inc (2 items)
    assert result.items[0].count == 2
    assert result.items[0].group_key == {"merchant": "Test Inc"}
    # Second group: Testing Inc (1 item)
    assert result.items[1].count == 1
    assert result.items[1].group_key == {"merchant": "Testing Inc"}


@pytest.mark.asyncio
async def test_get_item(server, client):
    """Verify items can be retrieved by ID."""
    data1 = Receipt(merchant="Test Inc", total=1000)
    data2 = Receipt(merchant="Test Inc", total=1300)
    item1 = await client.create_item(data1)
    item2 = await client.create_item(data2)

    retrieved = await client.get_item(item1.id)

    assert retrieved.collection == item1.collection
    assert retrieved.deployment_name == item1.deployment_name
    assert retrieved.data.merchant == data1.merchant
    assert retrieved.data.total == data1.total

    # Non-existent ID should raise 404
    with pytest.raises(ApiError) as exc_info:
        await client.get_item(item2.id + "nonexistent")

    assert exc_info.value.status_code == 404
    assert exc_info.value.body == {"detail": f"No data with ID: {item2.id}nonexistent"}


@pytest.mark.asyncio
async def test_delete_by_id(server, client):
    """Verify items can be deleted by ID."""
    data = Receipt(merchant="Test Inc", total=1300)
    item = await client.create_item(data)
    assert item.id is not None

    await client.delete_item(item.id)

    # Item should no longer exist
    with pytest.raises(ApiError) as exc_info:
        await client.get_item(item.id)

    assert exc_info.value.status_code == 404

    # Deleting again should also raise 404
    with pytest.raises(ApiError) as exc_info:
        await client.delete_item(item.id)

    assert exc_info.value.status_code == 404


@pytest.mark.asyncio
async def test_delete_by_query(server, client):
    """Verify items can be deleted by filter query."""
    data1 = Receipt(merchant="Test Inc", total=1000)
    data2 = Receipt(merchant="Test Inc", total=1300)
    data3 = Receipt(merchant="Testing Inc", total=1100)
    item1 = await client.create_item(data1)
    item2 = await client.create_item(data2)
    item3 = await client.create_item(data3)

    result = await client.delete(filter={"merchant": {"eq": "Test Inc"}})

    assert result == 2

    # Deleted items should no longer exist
    for item in (item1, item2):
        with pytest.raises(ApiError) as exc_info:
            await client.get_item(item.id)
        assert exc_info.value.status_code == 404

    # Non-matching item should still exist
    found = await client.get_item(item3.id)
    assert found.id == item3.id
