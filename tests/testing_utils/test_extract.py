"""Tests for the FakeExtractNamespace mock implementation (llama-cloud v2)."""

from pathlib import Path

import pytest
from extraction_review.testing_utils import FakeLlamaCloudServer
from llama_cloud import AsyncLlamaCloud
from pydantic import BaseModel, Field


class Receipt(BaseModel):
    merchant: str = Field(description="Vendor name")
    total: float = Field(description="Grand total")


@pytest.fixture(autouse=True)
def fake_env(monkeypatch):
    monkeypatch.setenv("LLAMA_CLOUD_API_KEY", "unit-test-key")
    monkeypatch.setenv("LLAMA_CLOUD_BASE_URL", FakeLlamaCloudServer.DEFAULT_BASE_URL)


@pytest.fixture
def server():
    with FakeLlamaCloudServer() as srv:
        yield srv


def _write_sample_file(tmp_path: Path, name: str, content: str) -> Path:
    target = tmp_path / name
    target.write_text(content)
    return target


@pytest.mark.asyncio
async def test_stateless_extract_is_deterministic(server, tmp_path):
    """Inline configuration + polling returns deterministic data."""
    client = AsyncLlamaCloud(api_key="unit-test-key")
    sample_path = _write_sample_file(
        tmp_path, "receipt.txt", "Merchant: Lunar Bistro\nTotal: 123.45"
    )

    file_obj = await client.files.create(
        file=sample_path,
        purpose="extract",
        external_file_id=str(sample_path),
    )
    configuration = {
        "data_schema": Receipt.model_json_schema(),
        "tier": "cost_effective",
    }
    first = await client.extract.run(
        file_input=file_obj.id,
        configuration=configuration,
    )
    second = await client.extract.run(
        file_input=file_obj.id,
        configuration=configuration,
    )

    assert first.status == "COMPLETED"
    assert isinstance(first.extract_result, dict)
    assert "merchant" in first.extract_result
    assert second.extract_result == first.extract_result


@pytest.mark.asyncio
async def test_saved_configuration_flow(server, tmp_path):
    """Using a saved configuration_id resolves schema + settings from the config."""
    client = AsyncLlamaCloud(api_key="unit-test-key")

    cfg = await client.configurations.create(
        name="receipt-cfg",
        parameters={
            "product_type": "extract_v2",
            "data_schema": Receipt.model_json_schema(),
            "tier": "agentic",
        },
    )

    sample_path = _write_sample_file(
        tmp_path, "contract.pdf", "Agreement between parties."
    )
    file_obj = await client.files.create(
        file=sample_path,
        purpose="extract",
        external_file_id=str(sample_path),
    )
    job = await client.extract.run(
        file_input=file_obj.id,
        configuration_id=cfg.id,
    )

    assert job.status == "COMPLETED"
    assert isinstance(job.extract_result, dict)
    assert "merchant" in job.extract_result
    assert job.configuration_id == cfg.id

    fetched = await client.configurations.retrieve(cfg.id)
    assert fetched.id == cfg.id
    assert fetched.parameters.product_type == "extract_v2"
