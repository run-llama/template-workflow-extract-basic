"""Tests for the FakeExtractNamespace mock implementation."""

from pathlib import Path

import pytest
from llama_cloud import ExtractConfig
from llama_cloud.types import ExtractMode
from llama_cloud_services.extract import LlamaExtract
from llama_cloud_services.parse import LlamaParse

from extraction_review.testing_utils import FakeLlamaCloudServer
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


def test_stateless_extract_is_deterministic(server, tmp_path):
    """Verify stateless extraction produces deterministic results."""
    extractor = LlamaExtract(api_key="unit-test-key", verify=False)
    config = ExtractConfig(extraction_mode=ExtractMode.FAST)
    sample_path = _write_sample_file(
        tmp_path, "receipt.txt", "Merchant: Lunar Bistro\nTotal: 123.45"
    )

    first_run = extractor.extract(Receipt, config, sample_path)
    second_run = extractor.extract(Receipt, config, sample_path)

    assert first_run.status.value == "SUCCESS"
    assert second_run.data == first_run.data
    assert "merchant" in first_run.data
    assert server.extract.stateless_run.called


def test_agent_flow_uploads_and_processes_files(server, tmp_path):
    """Verify agent flow correctly uploads files and processes them."""
    extractor = LlamaExtract(api_key="unit-test-key", verify=False)
    config = ExtractConfig(extraction_mode=ExtractMode.FAST)
    agent = extractor.create_agent(
        name="unit-test-agent", data_schema=Receipt, config=config
    )

    sample_path = _write_sample_file(
        tmp_path, "contract.pdf", "Agreement between parties."
    )
    run = agent.extract(sample_path)

    assert run.status.value == "SUCCESS"
    assert "merchant" in run.data

    uploaded_bytes = server.files.read(run.file.id)
    assert uploaded_bytes.startswith(b"Agreement")
    assert server.extract.agent_job.called
    assert server.extract.agent_run.called


def test_parse_load_data_returns_documents(server, tmp_path):
    """Verify LlamaParse returns documents with expected content."""
    parser = LlamaParse(
        api_key="unit-test-key", base_url=FakeLlamaCloudServer.DEFAULT_BASE_URL
    )
    sample_path = _write_sample_file(
        tmp_path, "report.pdf", "Executive summary of quarterly goals."
    )

    documents = parser.load_data(sample_path)

    assert documents
    assert "(page 1)" in documents[0].text
