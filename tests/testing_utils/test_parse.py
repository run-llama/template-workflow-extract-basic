"""Tests for the FakeParseNamespace mock implementation."""

import pytest
import httpx

from extraction_review.testing_utils import FakeLlamaCloudServer


@pytest.fixture
def server():
    """Provide a server with parse namespace enabled."""
    with FakeLlamaCloudServer(namespaces=["parse"]) as srv:
        yield srv


def _make_multipart_body(
    filename: str, content: bytes = b"fake pdf content"
) -> tuple[bytes, str]:
    """Create a multipart form body with the given filename."""
    boundary = "----TestBoundary123"
    body = (
        (
            f"------{boundary}\r\n"
            f'Content-Disposition: form-data; name="file"; filename="{filename}"\r\n'
            f"Content-Type: application/pdf\r\n"
            f"\r\n"
        ).encode()
        + content
        + f"\r\n------{boundary}--\r\n".encode()
    )
    return body, boundary


def test_job_result_excludes_duplicate_fields(server):
    """Verify job result doesn't include job_id/file_name to avoid duplicate arg errors.

    The llama_cloud_services library's JobResult.__init__ passes job_id and
    file_name as explicit args AND spreads the job_result dict. If both are
    present, it causes "multiple values for keyword argument" errors.
    """
    body, boundary = _make_multipart_body("test.pdf")

    upload_response = httpx.post(
        f"{server.DEFAULT_BASE_URL}/api/parsing/upload",
        content=body,
        headers={"Content-Type": f"multipart/form-data; boundary=----{boundary}"},
    )
    assert upload_response.status_code == 200
    job_id = upload_response.json()["id"]

    result_response = httpx.get(
        f"{server.DEFAULT_BASE_URL}/api/parsing/job/{job_id}/result/json"
    )
    assert result_response.status_code == 200
    result = result_response.json()

    # These fields should NOT be in the result to avoid duplicate argument errors
    assert "job_id" not in result, "job_id should be excluded from result"
    assert "file_name" not in result, "file_name should be excluded from result"

    # But other expected fields should still be present
    assert "status" in result
    assert "is_done" in result
    assert "pages" in result


def test_filename_with_double_quotes(server):
    """Verify filename is correctly parsed from double-quoted Content-Disposition."""
    body, boundary = _make_multipart_body("my_document.pdf")

    response = httpx.post(
        f"{server.DEFAULT_BASE_URL}/api/parsing/upload",
        content=body,
        headers={"Content-Type": f"multipart/form-data; boundary=----{boundary}"},
    )
    assert response.status_code == 200
    job_id = response.json()["id"]

    job = server.parse._jobs[job_id]
    assert job.file_name == "my_document.pdf"


def test_filename_with_single_quotes(server):
    """Verify filename is correctly parsed from single-quoted Content-Disposition."""
    boundary = "----TestBoundary789"
    body = (
        f"------{boundary}\r\n"
        f"Content-Disposition: form-data; name=\"file\"; filename='single_quoted.pdf'\r\n"
        f"Content-Type: application/pdf\r\n"
        f"\r\n"
        f"fake pdf content\r\n"
        f"------{boundary}--\r\n"
    ).encode()

    response = httpx.post(
        f"{server.DEFAULT_BASE_URL}/api/parsing/upload",
        content=body,
        headers={"Content-Type": f"multipart/form-data; boundary=----{boundary}"},
    )
    assert response.status_code == 200
    job_id = response.json()["id"]

    job = server.parse._jobs[job_id]
    assert job.file_name == "single_quoted.pdf"


def test_filename_does_not_capture_subsequent_headers(server):
    """Verify filename parsing stops at header boundary, not capturing Content-Type."""
    body, boundary = _make_multipart_body("test.pdf")

    response = httpx.post(
        f"{server.DEFAULT_BASE_URL}/api/parsing/upload",
        content=body,
        headers={"Content-Type": f"multipart/form-data; boundary=----{boundary}"},
    )
    assert response.status_code == 200
    job_id = response.json()["id"]

    job = server.parse._jobs[job_id]
    # Should be just "test.pdf", not "test.pdf\r\nContent-Type: application/pdf"
    assert job.file_name == "test.pdf"
    assert "Content-Type" not in job.file_name


def test_job_status_endpoint(server):
    """Verify job status endpoint returns correct status."""
    body, boundary = _make_multipart_body("status_test.pdf")

    upload_response = httpx.post(
        f"{server.DEFAULT_BASE_URL}/api/parsing/upload",
        content=body,
        headers={"Content-Type": f"multipart/form-data; boundary=----{boundary}"},
    )
    job_id = upload_response.json()["id"]

    status_response = httpx.get(f"{server.DEFAULT_BASE_URL}/api/parsing/job/{job_id}")
    assert status_response.status_code == 200
    status = status_response.json()
    assert status["id"] == job_id
    assert status["status"] == "SUCCESS"


def test_job_not_found_returns_404(server):
    """Verify non-existent job returns 404."""
    response = httpx.get(
        f"{server.DEFAULT_BASE_URL}/api/parsing/job/nonexistent-job-id"
    )
    assert response.status_code == 404

    result_response = httpx.get(
        f"{server.DEFAULT_BASE_URL}/api/parsing/job/nonexistent-job-id/result/json"
    )
    assert result_response.status_code == 404
