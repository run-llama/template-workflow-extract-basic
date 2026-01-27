"""Tests for the FakeSheetsNamespace mock implementation."""

import httpx
import pytest
from extraction_review.testing_utils import FakeLlamaCloudServer
from llama_cloud import APIStatusError, AsyncLlamaCloud


@pytest.fixture
def server():
    """Provide a server with files and sheets namespaces enabled."""
    with FakeLlamaCloudServer(namespaces=["files", "sheets"]) as srv:
        yield srv


@pytest.mark.asyncio
async def test_sheets_create_and_get(server, tmp_path):
    """Verify a sheets job can be created and retrieved."""
    test_file = tmp_path / "spreadsheet.xlsx"
    test_file.write_bytes(b"fake spreadsheet content")
    file_id = server.files.preload(path=test_file)

    client = AsyncLlamaCloud(api_key="fake-api-key")

    job = await client.beta.sheets.create(file_id=file_id)
    assert job.id.startswith("sheets-job_")
    assert job.status == "SUCCESS"
    assert job.success is True
    assert job.regions is not None
    assert len(job.regions) > 0

    for region in job.regions:
        assert region.region_id is not None
        assert region.region_type in ("table", "extra")
        assert region.sheet_name is not None
        assert region.location is not None

    assert job.worksheet_metadata is not None
    assert len(job.worksheet_metadata) > 0

    # Retrieve the same job
    retrieved = await client.beta.sheets.get(job.id)
    assert retrieved.id == job.id
    assert retrieved.status == "SUCCESS"


@pytest.mark.asyncio
async def test_sheets_file_not_found(server):
    """Verify sheets with non-existent file returns 404."""
    client = AsyncLlamaCloud(api_key="fake-api-key")
    with pytest.raises(APIStatusError) as exc_info:
        await client.beta.sheets.create(file_id="nonexistent-file")
    assert exc_info.value.status_code == 404


@pytest.mark.asyncio
async def test_sheets_job_not_found(server):
    """Verify non-existent sheets job returns 404."""
    client = AsyncLlamaCloud(api_key="fake-api-key")
    with pytest.raises(APIStatusError) as exc_info:
        await client.beta.sheets.get("nonexistent-id")
    assert exc_info.value.status_code == 404


@pytest.mark.asyncio
async def test_sheets_delete_job(server, tmp_path):
    """Verify a sheets job can be deleted."""
    test_file = tmp_path / "spreadsheet.xlsx"
    test_file.write_bytes(b"delete me")
    file_id = server.files.preload(path=test_file)

    client = AsyncLlamaCloud(api_key="fake-api-key")
    job = await client.beta.sheets.create(file_id=file_id)

    # Job should exist
    retrieved = await client.beta.sheets.get(job.id)
    assert retrieved.id == job.id

    # Delete
    await client.beta.sheets.delete_job(job.id)

    # Should no longer exist
    with pytest.raises(APIStatusError) as exc_info:
        await client.beta.sheets.get(job.id)
    assert exc_info.value.status_code == 404


@pytest.mark.asyncio
async def test_sheets_get_result_table(server, tmp_path):
    """Verify presigned URL generation for region results."""
    test_file = tmp_path / "spreadsheet.xlsx"
    test_file.write_bytes(b"spreadsheet with tables")
    file_id = server.files.preload(path=test_file)

    client = AsyncLlamaCloud(api_key="fake-api-key")
    job = await client.beta.sheets.create(file_id=file_id)

    assert job.regions is not None
    assert len(job.regions) > 0

    region = job.regions[0]
    presigned = await client.beta.sheets.get_result_table(
        "table",
        spreadsheet_job_id=job.id,
        region_id=region.region_id,
    )
    assert presigned.url is not None
    assert "token=fake" in presigned.url
    assert job.id in presigned.url


@pytest.mark.asyncio
async def test_sheets_with_config(server, tmp_path):
    """Verify sheets job with custom config."""
    test_file = tmp_path / "spreadsheet.xlsx"
    test_file.write_bytes(b"configured spreadsheet")
    file_id = server.files.preload(path=test_file)

    client = AsyncLlamaCloud(api_key="fake-api-key")

    job = await client.beta.sheets.create(
        file_id=file_id,
        config={
            "sheet_names": ["Revenue", "Expenses"],
            "flatten_hierarchical_tables": True,
            "generate_additional_metadata": True,
        },
    )
    assert job.status == "SUCCESS"
    assert job.worksheet_metadata is not None

    sheet_names = {ws.sheet_name for ws in job.worksheet_metadata}
    assert sheet_names == {"Revenue", "Expenses"}

    if job.regions:
        region_sheets = {r.sheet_name for r in job.regions}
        assert region_sheets.issubset({"Revenue", "Expenses"})


@pytest.mark.asyncio
async def test_sheets_list_jobs(server, tmp_path):
    """Verify listing sheets jobs returns created jobs."""
    test_file = tmp_path / "spreadsheet.xlsx"
    test_file.write_bytes(b"list test")
    file_id = server.files.preload(path=test_file)

    client = AsyncLlamaCloud(api_key="fake-api-key")

    await client.beta.sheets.create(file_id=file_id)
    await client.beta.sheets.create(file_id=file_id)

    jobs = await client.beta.sheets.list()
    items = [j async for j in jobs]
    assert len(items) == 2


@pytest.mark.asyncio
async def test_sheets_presigned_download(server, tmp_path):
    """Verify that the presigned URL for a region result serves parquet content."""
    test_file = tmp_path / "spreadsheet.xlsx"
    test_file.write_bytes(b"spreadsheet for download test")
    file_id = server.files.preload(path=test_file)

    client = AsyncLlamaCloud(api_key="fake-api-key")
    job = await client.beta.sheets.create(file_id=file_id)

    assert job.regions is not None
    assert len(job.regions) > 0

    region = job.regions[0]
    presigned = await client.beta.sheets.get_result_table(
        region.region_type,
        spreadsheet_job_id=job.id,
        region_id=region.region_id,
    )

    # Follow the presigned URL to download the parquet content
    async with httpx.AsyncClient() as http:
        resp = await http.get(presigned.url)
    assert resp.status_code == 200
    content = resp.content
    # Verify PAR1 magic bytes at start and end
    assert content[:4] == b"PAR1"
    assert content[-4:] == b"PAR1"
    assert len(content) > 8  # has actual payload between magic markers


@pytest.mark.asyncio
async def test_sheets_presigned_download_deterministic(server, tmp_path):
    """Verify that repeated downloads for the same region return identical content."""
    test_file = tmp_path / "spreadsheet.xlsx"
    test_file.write_bytes(b"deterministic download test")
    file_id = server.files.preload(path=test_file)

    client = AsyncLlamaCloud(api_key="fake-api-key")
    job = await client.beta.sheets.create(file_id=file_id)

    assert job.regions is not None
    region = job.regions[0]
    presigned = await client.beta.sheets.get_result_table(
        region.region_type,
        spreadsheet_job_id=job.id,
        region_id=region.region_id,
    )

    async with httpx.AsyncClient() as http:
        resp1 = await http.get(presigned.url)
        resp2 = await http.get(presigned.url)
    assert resp1.content == resp2.content
