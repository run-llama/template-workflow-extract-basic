"""Tests for the FakeExtractNamespace mock implementation."""

from pathlib import Path

import pytest
from extraction_review.testing_utils import FakeLlamaCloudServer
from llama_cloud import AsyncLlamaCloud
from llama_cloud.types.extraction.extract_config_param import ExtractConfigParam
from pydantic import BaseModel, Field


class Receipt(BaseModel):
    merchant: str = Field(description="Vendor name")
    total: float = Field(description="Grand total")


@pytest.fixture(autouse=True)
def fake_env(monkeypatch):
    """Set up environment variables for the fake server."""
    monkeypatch.setenv("LLAMA_CLOUD_API_KEY", "unit-test-key")
    monkeypatch.setenv("LLAMA_CLOUD_BASE_URL", FakeLlamaCloudServer.DEFAULT_BASE_URL)


@pytest.fixture
def server():
    """Provide an installed FakeLlamaCloudServer."""
    with FakeLlamaCloudServer() as srv:
        yield srv


def _write_sample_file(tmp_path: Path, name: str, content: str) -> Path:
    """Write a sample file for testing."""
    target = tmp_path / name
    target.write_text(content)
    return target


@pytest.mark.asyncio
async def test_stateless_extract_is_deterministic(server, tmp_path):
    """Verify stateless extraction produces deterministic results."""
    client = AsyncLlamaCloud(api_key="unit-test-key")
    config = ExtractConfigParam(extraction_mode="FAST")
    sample_path = _write_sample_file(
        tmp_path, "receipt.txt", "Merchant: Lunar Bistro\nTotal: 123.45"
    )

    file_obj = await client.files.create(
        file=sample_path,
        purpose="extract",
        external_file_id=str(sample_path),
    )
    first_run = await client.extraction.extract(
        data_schema=Receipt.model_json_schema(),
        config=config,
        file_id=file_obj.id,
    )
    second_run = await client.extraction.extract(
        data_schema=Receipt.model_json_schema(),
        config=config,
        file_id=file_obj.id,
    )

    assert second_run.data == first_run.data
    assert isinstance(first_run.data, dict)
    assert "merchant" in first_run.data
    assert server.extract.stateless_run.called


@pytest.mark.asyncio
async def test_agent_flow_uploads_and_processes_files(server, tmp_path):
    """Verify agent flow correctly uploads files and processes them."""
    client = AsyncLlamaCloud(api_key="unit-test-key")
    config = ExtractConfigParam(extraction_mode="FAST")
    agent = await client.extraction.extraction_agents.create(
        name="unit-test-agent", data_schema=Receipt.model_json_schema(), config=config
    )

    sample_path = _write_sample_file(
        tmp_path, "contract.pdf", "Agreement between parties."
    )
    file_obj = await client.files.create(
        file=sample_path,
        purpose="extract",
        external_file_id=str(sample_path),
    )
    run = await client.extraction.jobs.extract(
        extraction_agent_id=agent.id,
        file_id=file_obj.id,
    )

    assert isinstance(run.data, dict)
    assert "merchant" in run.data

    assert server.extract.agent_job.called
