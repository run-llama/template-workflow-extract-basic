"""Tests for the FakeFilesNamespace mock implementation."""

import pytest
import httpx

from extraction_review.testing_utils import FakeLlamaCloudServer


@pytest.fixture
def server():
    """Provide a server with files namespace enabled."""
    with FakeLlamaCloudServer(namespaces=["files"]) as srv:
        yield srv


def test_preload_and_read(server, tmp_path):
    """Verify files can be preloaded and read back."""
    test_file = tmp_path / "test_file.txt"
    test_file.write_bytes(b"test content here")

    file_id = server.files.preload(path=test_file)

    content = server.files.read(file_id)
    assert content == b"test content here"

    response = httpx.get(f"{server.DEFAULT_BASE_URL}/api/v1/files/{file_id}")
    assert response.status_code == 200
    metadata = response.json()
    assert metadata["id"] == file_id
    assert metadata["name"] == "test_file.txt"


def test_not_found_returns_404(server):
    """Verify non-existent file returns 404."""
    response = httpx.get(f"{server.DEFAULT_BASE_URL}/api/v1/files/nonexistent-file-id")
    assert response.status_code == 404


def test_delete_file(server, tmp_path):
    """Verify files can be deleted."""
    test_file = tmp_path / "to_delete.txt"
    test_file.write_bytes(b"delete me")

    file_id = server.files.preload(path=test_file)

    # File should exist
    response = httpx.get(f"{server.DEFAULT_BASE_URL}/api/v1/files/{file_id}")
    assert response.status_code == 200

    # Delete the file
    delete_response = httpx.delete(f"{server.DEFAULT_BASE_URL}/api/v1/files/{file_id}")
    assert delete_response.status_code == 200

    # File should no longer exist
    response = httpx.get(f"{server.DEFAULT_BASE_URL}/api/v1/files/{file_id}")
    assert response.status_code == 404
