"""Tests for the FakeFilesNamespace mock implementation."""

from urllib.parse import urlencode

import httpx
import pytest
from extraction_review.testing_utils import FakeLlamaCloudServer
from llama_cloud import APIStatusError, AsyncLlamaCloud
from llama_cloud.types.file_query_params import Filter


@pytest.fixture
def server():
    """Provide a server with files namespace enabled."""
    with FakeLlamaCloudServer(namespaces=["files"]) as srv:
        yield srv


@pytest.mark.asyncio
async def test_preload_and_download_as_presigned_url(server, tmp_path):
    """Verify files can be preloaded and read back."""
    test_file = tmp_path / "test_file.txt"
    test_file.write_bytes(b"test content here")

    file_id = server.files.preload(path=test_file)

    content = server.files.read(file_id)
    assert content == b"test content here"

    client = AsyncLlamaCloud(api_key="fake-api-key")

    presigned_url = await client.files.get(
        file_id=file_id,
    )
    assert (
        presigned_url.url
        == f"{server._download_base_url}/files/{file_id}?{urlencode({'token': 'fake'})}"
    )
    response = httpx.get(presigned_url.url)
    assert response.content == b"test content here"


@pytest.mark.asyncio
async def test_not_found_returns_404(server):
    """Verify non-existent file returns 404."""
    client = AsyncLlamaCloud(api_key="fake-api-key")
    with pytest.raises(APIStatusError) as exc_info:
        await client.files.get(
            "does-not-exist",
        )
    assert exc_info.value.status_code == 404


@pytest.mark.asyncio
async def test_delete_file(server, tmp_path):
    """Verify files can be deleted."""
    test_file = tmp_path / "to_delete.txt"
    test_file.write_bytes(b"delete me")

    file_id = server.files.preload(path=test_file)
    client = AsyncLlamaCloud(api_key="fake-api-key")

    # File should exist
    response = await client.files.get(file_id)
    assert file_id in response.url

    # Delete the file
    await client.files.delete(
        file_id,
    )

    # File should no longer exist
    with pytest.raises(APIStatusError) as exc_info:
        await client.files.get(
            file_id,
        )
    assert exc_info.value.status_code == 404


@pytest.mark.asyncio
async def test_files_native_upload(server, tmp_path):
    """Verify that the client can natively upload the files without having to pass through server.preload"""
    test_file = tmp_path / "test_file.txt"
    test_file.write_bytes(b"test content here")
    client = AsyncLlamaCloud(api_key="fake-api-key")
    file_obj = await client.files.create(
        file=test_file,
        purpose="parse",
        external_file_id=str(test_file),
    )
    assert isinstance(file_obj.id, str)
    assert file_obj.file_type == "application/octet-stream"
    assert file_obj.id.startswith("file_")


@pytest.mark.asyncio
async def test_files_query_by_id(server, tmp_path):
    """Test that you can upload and query files selecting them by file ID"""
    test_file_1 = tmp_path / "test_file1.txt"
    test_file_1.write_bytes(b"test content here 1")
    test_file_2 = tmp_path / "test_file2.txt"
    test_file_2.write_bytes(b"test content here 2")
    client = AsyncLlamaCloud(api_key="fake-api-key")
    file_obj_1 = await client.files.create(
        file=test_file_1,
        purpose="parse",
        external_file_id=str(test_file_1),
    )
    await client.files.create(
        file=test_file_2,
        purpose="parse",
        external_file_id=str(test_file_1),
    )
    response = await client.files.query(filter=Filter(file_ids=[file_obj_1.id]))
    assert len(response.items) == 1
    assert response.total_size == 1
    assert response.items[0].id == file_obj_1.id


@pytest.mark.asyncio
async def test_files_list(server, tmp_path):
    """Test that you can list files using the GET /files endpoint."""
    test_file_1 = tmp_path / "file_a.txt"
    test_file_1.write_bytes(b"content a")
    test_file_2 = tmp_path / "file_b.txt"
    test_file_2.write_bytes(b"content b")

    client = AsyncLlamaCloud(api_key="fake-api-key")
    file_obj_1 = await client.files.create(
        file=test_file_1,
        purpose="extract",
    )
    file_obj_2 = await client.files.create(
        file=test_file_2,
        purpose="extract",
    )

    # List all files
    all_files = await client.files.list()
    items = [f async for f in all_files]
    assert len(items) >= 2
    ids = {f.id for f in items}
    assert file_obj_1.id in ids
    assert file_obj_2.id in ids


@pytest.mark.asyncio
async def test_files_list_by_name(server, tmp_path):
    """Test filtering files by name via the list endpoint."""
    test_file = tmp_path / "unique_name.pdf"
    test_file.write_bytes(b"unique content")

    # Use preload for reliable filename storage
    server.files.preload(path=test_file, filename="unique_name.pdf")
    server.files.preload_from_source("other_file.txt", b"other content")

    client = AsyncLlamaCloud(api_key="fake-api-key")

    results = await client.files.list(file_name="unique_name.pdf")
    items = [f async for f in results]
    assert len(items) == 1
    assert items[0].name == "unique_name.pdf"
