"""Tests for the FakeParseNamespace mock implementation."""

import pytest
import respx
from extraction_review.testing_utils import FakeLlamaCloudServer
from llama_cloud import APIStatusError, AsyncLlamaCloud


@pytest.fixture
def server():
    """Provide a server with parse namespace enabled."""
    with FakeLlamaCloudServer() as srv:
        yield srv


@pytest.fixture()
def client() -> AsyncLlamaCloud:
    return AsyncLlamaCloud(api_key="fake-api-key")


@pytest.fixture()
def data() -> tuple[str, bytes, str]:
    with open("tests/files/test.pdf", "rb") as f:
        content = f.read()
    return ("tests/files/test.pdf", content, "application/pdf")


@pytest.mark.asyncio
async def test_parse_with_upload_file(
    server: FakeLlamaCloudServer, client: AsyncLlamaCloud, data: tuple[str, bytes, str]
) -> None:
    job_create = await client.parsing.create(
        tier="fast",
        version="latest",
        upload_file=data,
    )
    assert job_create.error_message is None
    assert job_create.status == "COMPLETED"
    assert job_create.project_id == server.default_project_id
    job_response = await client.parsing.get(
        job_id=job_create.id, expand=["text", "markdown", "items"]
    )
    assert job_response.job.id == job_create.id
    assert job_response.job.status == job_create.status
    assert job_response.job.project_id == job_create.project_id
    assert job_response.items is not None
    assert job_response.markdown is not None
    assert job_response.text is not None


@pytest.mark.asyncio
async def test_parse_with_different_expand(
    server: FakeLlamaCloudServer, client: AsyncLlamaCloud, data: tuple[str, bytes, str]
) -> None:
    job_create = await client.parsing.create(
        tier="fast",
        version="latest",
        upload_file=data,
    )
    job_response = await client.parsing.get(job_id=job_create.id, expand=["text"])
    assert job_response.items is None
    assert job_response.markdown is None
    assert job_response.text is not None
    job_response = await client.parsing.get(job_id=job_create.id, expand=["markdown"])
    assert job_response.items is None
    assert job_response.markdown is not None
    assert job_response.text is None
    job_response = await client.parsing.get(job_id=job_create.id, expand=["items"])
    assert job_response.items is not None
    assert job_response.markdown is None
    assert job_response.text is None
    # no expands -> defaul to items
    job_response = await client.parsing.get(job_id=job_create.id)
    assert job_response.items is not None
    assert job_response.markdown is None
    assert job_response.text is None


@pytest.mark.asyncio
async def test_parse_with_file_id(
    server: FakeLlamaCloudServer, client: AsyncLlamaCloud, data: tuple[str, bytes, str]
) -> None:
    file_name, _, _ = data
    file_obj = await client.files.create(
        file=file_name,
        purpose="parse",
        external_file_id=file_name,
    )
    job_create = await client.parsing.create(
        tier="fast",
        version="latest",
        file_id=file_obj.id,
    )
    assert job_create.error_message is None
    assert job_create.status == "COMPLETED"
    assert job_create.project_id == server.default_project_id
    job_response = await client.parsing.get(
        job_id=job_create.id, expand=["text", "markdown", "items"]
    )
    assert job_response.job.id == job_create.id
    assert job_response.job.status == job_create.status
    assert job_response.job.project_id == job_create.project_id
    assert job_response.items is not None
    assert job_response.markdown is not None
    assert job_response.text is not None


@pytest.mark.asyncio
async def test_parse_with_file_id_file_not_found(
    server: FakeLlamaCloudServer,
    client: AsyncLlamaCloud,
) -> None:
    with pytest.raises(APIStatusError) as exc_info:
        await client.parsing.create(
            tier="fast",
            version="latest",
            file_id="does-not-exist",
        )
    assert exc_info.value.status_code == 404


@pytest.mark.asyncio
async def test_parse_without_fileid_or_sourceurl(
    server: FakeLlamaCloudServer,
    client: AsyncLlamaCloud,
) -> None:
    with pytest.raises(APIStatusError) as exc_info:
        await client.parsing.create(
            tier="fast",
            version="latest",
        )
    assert exc_info.value.status_code == 400


@pytest.mark.asyncio
@respx.mock(assert_all_mocked=False)
async def test_parse_with_source_url(
    server: FakeLlamaCloudServer,
    client: AsyncLlamaCloud,
) -> None:
    job_create = await client.parsing.create(
        tier="fast",
        version="latest",
        source_url="https://pdfobject.com/pdf/sample.pdf",
    )
    assert job_create.error_message is None
    assert job_create.status == "COMPLETED"
    assert job_create.project_id == server.default_project_id
    job_response = await client.parsing.get(
        job_id=job_create.id, expand=["text", "markdown", "items"]
    )
    assert job_response.job.id == job_create.id
    assert job_response.job.status == job_create.status
    assert job_response.job.project_id == job_create.project_id
    assert job_response.items is not None
    assert job_response.markdown is not None
    assert job_response.text is not None


@pytest.mark.asyncio
async def test_parse_e2e(
    server: FakeLlamaCloudServer, client: AsyncLlamaCloud, data: tuple[str, bytes, str]
) -> None:
    file_name, _, _ = data
    file_obj = await client.files.create(
        file=file_name,
        purpose="parse",
        external_file_id=file_name,
    )
    result = await client.parsing.parse(
        file_id=file_obj.id,
        expand=["markdown"],
        tier="agentic",
        version="latest",
    )
    assert result.markdown is not None
    assert len(result.markdown.pages) == 1
    assert hasattr(result.markdown.pages[0], "markdown")
    assert isinstance(result.markdown.pages[0].markdown, str)  # type: ignore
