"""Tests for the FakeAgentDataNamespace mock implementation."""

from typing import cast

import pytest
from extraction_review.testing_utils import FakeLlamaCloudServer
from extraction_review.testing_utils._deterministic import hash_schema
from llama_cloud import AsyncLlamaCloud
from llama_cloud._exceptions import APIStatusError
from pydantic import BaseModel, Field


class Receipt(BaseModel):
    merchant: str = Field(description="Vendor name")
    total: float = Field(description="Grand total")


@pytest.fixture
def server():
    """Provide an installed FakeLlamaCloudServer."""
    with FakeLlamaCloudServer() as srv:
        yield srv


@pytest.fixture
def client(server) -> AsyncLlamaCloud:
    """Provide an AsyncLlamaCloud client configured for the fake server."""
    return AsyncLlamaCloud(api_key="fake-api-key")


@pytest.mark.asyncio
async def test_create_item(server, client: AsyncLlamaCloud):
    """Verify items can be created and have expected ID format."""
    data = Receipt(merchant="Test Inc", total=1000)
    item = await client.beta.agent_data.agent_data(
        data=data.model_dump(),
        deployment_name="extraction_agent",
        collection="extracted_data",
    )

    assert item.id == hash_schema(data)[:7]
    assert item.data["merchant"] == data.merchant
    assert item.data["total"] == data.total
    assert item.collection == "extracted_data"
    assert item.deployment_name == "extraction_agent"


@pytest.mark.asyncio
async def test_update_item(server, client: AsyncLlamaCloud):
    """Verify items can be updated while preserving metadata."""
    data = Receipt(merchant="Test Inc", total=1000)
    item = await client.beta.agent_data.agent_data(
        data=data.model_dump(),
        deployment_name="extraction_agent",
        collection="extracted_data",
    )
    assert item.id is not None

    updated_data = Receipt(merchant="Testing Inc", total=1100)
    updated_item = await client.beta.agent_data.update(
        item_id=item.id, data=updated_data.model_dump()
    )

    assert updated_item.data["merchant"] == updated_data.merchant
    assert updated_item.data["total"] == updated_data.total
    assert updated_item.id == item.id
    assert updated_item.collection == item.collection
    assert updated_item.deployment_name == item.deployment_name


@pytest.mark.asyncio
async def test_search_with_eq_filter(server, client: AsyncLlamaCloud):
    """Verify search with equality filter returns matching items."""
    data1 = Receipt(merchant="Test Inc", total=1000)
    data2 = Receipt(merchant="Test Inc", total=1300)
    data3 = Receipt(merchant="Testing Inc", total=1100)
    item1 = await client.beta.agent_data.agent_data(
        data=data1.model_dump(),
        deployment_name="extraction_agent",
        collection="extracted_data",
    )
    item2 = await client.beta.agent_data.agent_data(
        data=data2.model_dump(),
        deployment_name="extraction_agent",
        collection="extracted_data",
    )
    await client.beta.agent_data.agent_data(
        data=data3.model_dump(),
        deployment_name="extraction_agent",
        collection="extracted_data",
    )

    result = await client.beta.agent_data.search(
        deployment_name="extraction_agent",
        collection="extracted_data",
        filter={"merchant": {"eq": "Test Inc"}},
    )

    assert result.total_size == 2
    assert any(item.id == item1.id for item in result.items)
    assert any(item.id == item2.id for item in result.items)
    assert all(item.data["merchant"] == "Test Inc" for item in result.items)


@pytest.mark.asyncio
async def test_search_with_lt_filter(server, client: AsyncLlamaCloud):
    """Verify search with less-than filter returns matching items."""
    data1 = Receipt(merchant="Test Inc", total=1000)
    data2 = Receipt(merchant="Test Inc", total=1300)
    data3 = Receipt(merchant="Testing Inc", total=1100)
    item1 = await client.beta.agent_data.agent_data(
        data=data1.model_dump(),
        deployment_name="extraction_agent",
        collection="extracted_data",
    )
    await client.beta.agent_data.agent_data(
        data=data2.model_dump(),
        deployment_name="extraction_agent",
        collection="extracted_data",
    )
    item3 = await client.beta.agent_data.agent_data(
        data=data3.model_dump(),
        deployment_name="extraction_agent",
        collection="extracted_data",
    )

    result = await client.beta.agent_data.search(
        deployment_name="extraction_agent",
        collection="extracted_data",
        filter={"total": {"lt": 1200}},
    )

    assert result.total_size == 2
    assert any(item.id == item1.id for item in result.items)
    assert any(item.id == item3.id for item in result.items)
    assert all(cast(int, item.data["total"]) < 1200 for item in result.items)


@pytest.mark.asyncio
async def test_aggregate_with_filter(server, client: AsyncLlamaCloud):
    """Verify aggregation with filter groups correctly."""
    data1 = Receipt(merchant="Test Inc", total=1000)
    data2 = Receipt(merchant="Test Inc", total=1300)
    data3 = Receipt(merchant="Testing Inc", total=1100)
    await client.beta.agent_data.agent_data(
        data=data1.model_dump(),
        deployment_name="extraction_agent",
        collection="extracted_data",
    )
    await client.beta.agent_data.agent_data(
        data=data2.model_dump(),
        deployment_name="extraction_agent",
        collection="extracted_data",
    )
    await client.beta.agent_data.agent_data(
        data=data3.model_dump(),
        deployment_name="extraction_agent",
        collection="extracted_data",
    )

    result = await client.beta.agent_data.aggregate(
        deployment_name="extraction_agent",
        collection="extracted_data",
        filter={"merchant": {"eq": "Test Inc"}},
        group_by=["merchant"],
        count=True,
    )

    # Filtering for 'Test Inc' means only one group
    assert len(result.items) == 1
    assert result.items[0].count == 2
    assert result.items[0].first_item is not None
    assert result.items[0].first_item["merchant"] == data1.merchant
    assert result.items[0].group_key == {"merchant": "Test Inc"}


@pytest.mark.asyncio
async def test_aggregate_without_filter(server, client: AsyncLlamaCloud):
    """Verify aggregation without filter groups all items."""
    data1 = Receipt(merchant="Test Inc", total=1000)
    data2 = Receipt(merchant="Test Inc", total=1300)
    data3 = Receipt(merchant="Testing Inc", total=1100)
    await client.beta.agent_data.agent_data(
        data=data1.model_dump(),
        deployment_name="extraction_agent",
        collection="extracted_data",
    )
    await client.beta.agent_data.agent_data(
        data=data2.model_dump(),
        deployment_name="extraction_agent",
        collection="extracted_data",
    )
    await client.beta.agent_data.agent_data(
        data=data3.model_dump(),
        deployment_name="extraction_agent",
        collection="extracted_data",
    )
    result = await client.beta.agent_data.aggregate(
        deployment_name="extraction_agent",
        collection="extracted_data",
        group_by=["merchant"],
        count=True,
    )

    assert len(result.items) == 2
    # First group: Test Inc (2 items)
    assert result.items[0].count == 2
    assert result.items[0].group_key == {"merchant": "Test Inc"}
    # Second group: Testing Inc (1 item)
    assert result.items[1].count == 1
    assert result.items[1].group_key == {"merchant": "Testing Inc"}


@pytest.mark.asyncio
async def test_get_item(server, client: AsyncLlamaCloud):
    """Verify items can be retrieved by ID."""
    data1 = Receipt(merchant="Test Inc", total=1000)
    data2 = Receipt(merchant="Test Inc", total=1300)
    item1 = await client.beta.agent_data.agent_data(
        data=data1.model_dump(),
        deployment_name="extraction_agent",
        collection="extracted_data",
    )
    item2 = await client.beta.agent_data.agent_data(
        data=data2.model_dump(),
        deployment_name="extraction_agent",
        collection="extracted_data",
    )

    assert item1.id is not None
    retrieved = await client.beta.agent_data.get(item_id=item1.id)

    assert retrieved.collection == item1.collection
    assert retrieved.deployment_name == item1.deployment_name
    assert retrieved.data["merchant"] == data1.merchant
    assert retrieved.data["total"] == data1.total

    assert item2.id is not None
    # Non-existent ID should raise 404
    with pytest.raises(APIStatusError) as exc_info:
        await client.beta.agent_data.get(item_id=item2.id + "nonexistent")

    assert exc_info.value.status_code == 404
    assert exc_info.value.body == {"detail": f"No data with ID: {item2.id}nonexistent"}


@pytest.mark.asyncio
async def test_delete_by_id(server, client: AsyncLlamaCloud):
    """Verify items can be deleted by ID."""
    data = Receipt(merchant="Test Inc", total=1300)
    item = await client.beta.agent_data.agent_data(
        data=data.model_dump(),
        deployment_name="extraction_agent",
        collection="extracted_data",
    )
    assert item.id is not None

    await client.beta.agent_data.delete(item.id)

    # Item should no longer exist
    with pytest.raises(APIStatusError) as exc_info:
        await client.beta.agent_data.get(item.id)

    assert exc_info.value.status_code == 404

    # Deleting again should also raise 404
    with pytest.raises(APIStatusError) as exc_info:
        await client.beta.agent_data.delete(item.id)

    assert exc_info.value.status_code == 404


@pytest.mark.asyncio
async def test_delete_by_query(server, client: AsyncLlamaCloud):
    """Verify items can be deleted by filter query."""
    data1 = Receipt(merchant="Test Inc", total=1000)
    data2 = Receipt(merchant="Test Inc", total=1300)
    data3 = Receipt(merchant="Testing Inc", total=1100)
    item1 = await client.beta.agent_data.agent_data(
        data=data1.model_dump(),
        deployment_name="extraction_agent",
        collection="extracted_data",
    )
    item2 = await client.beta.agent_data.agent_data(
        data=data2.model_dump(),
        deployment_name="extraction_agent",
        collection="extracted_data",
    )
    item3 = await client.beta.agent_data.agent_data(
        data=data3.model_dump(),
        deployment_name="extraction_agent",
        collection="extracted_data",
    )

    result = await client.beta.agent_data.delete_by_query(
        deployment_name="extraction_agent",
        collection="extracted_data",
        filter={"merchant": {"eq": "Test Inc"}},
    )

    assert result.deleted_count == 2

    # Deleted items should no longer exist
    for item in (item1, item2):
        assert item.id is not None
        with pytest.raises(APIStatusError) as exc_info:
            await client.beta.agent_data.get(item.id)
        assert exc_info.value.status_code == 404

    # Non-matching item should still exist
    assert item3.id is not None
    found = await client.beta.agent_data.get(item3.id)
    assert found.id == item3.id
