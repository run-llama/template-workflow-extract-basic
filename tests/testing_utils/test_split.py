import pytest
from extraction_review.testing_utils import FakeLlamaCloudServer
from llama_cloud import APIStatusError, AsyncLlamaCloud


@pytest.fixture
def server():
    with FakeLlamaCloudServer() as srv:
        yield srv


@pytest.fixture()
def client() -> AsyncLlamaCloud:
    return AsyncLlamaCloud(api_key="fake-api-key")


@pytest.mark.asyncio
async def test_split_end_to_end(
    server: FakeLlamaCloudServer, client: AsyncLlamaCloud
) -> None:
    file_id = server.files.preload(path="tests/files/test.pdf")
    split_job = await client.beta.split.create(
        categories=[
            {"name": "hello", "description": ""},
            {"name": "world", "description": ""},
        ],
        document_input={"type": "file_id", "value": file_id},
    )
    assert split_job.id.startswith("split-")
    assert split_job.status == "pending"
    cts = [c.name for c in split_job.categories]
    cts.sort()
    assert cts == ["hello", "world"]
    split_result = await client.beta.split.wait_for_completion(
        split_job_id=split_job.id,
    )
    assert split_result.result is not None
    cts = [s.category for s in split_result.result.segments]
    cts.sort()
    assert cts == [
        "hello",
        "world",
    ]
    assert any(len(s.pages) > 0 for s in split_result.result.segments)


@pytest.mark.asyncio
async def test_split_no_categories_raises_bad_request(
    server: FakeLlamaCloudServer, client: AsyncLlamaCloud
) -> None:
    file_id = server.files.preload(path="tests/files/test.pdf")
    # should fail because there are no categories
    with pytest.raises(APIStatusError) as exc_info:
        await client.beta.split.create(
            categories=[],
            document_input={"type": "file_id", "value": file_id},
        )
        assert exc_info.value.status_code == 400
        assert (
            "categories field should be non-null and non-empty"
            in exc_info.value.message
        )


@pytest.mark.asyncio
async def test_split_invalid_document_input_type_raises_bad_request(
    server: FakeLlamaCloudServer, client: AsyncLlamaCloud
) -> None:
    file_id = server.files.preload(path="tests/files/test.pdf")
    # should fail because the only allowed document_input type is file_id
    with pytest.raises(APIStatusError) as exc_info:
        await client.beta.split.create(
            categories=[
                {"name": "hello", "description": ""},
                {"name": "world", "description": ""},
            ],
            document_input={"type": "file", "value": file_id},
        )
        assert exc_info.value.status_code == 400
        assert (
            "document_input.type file is invalid. Allowed input types: file_id"
            in exc_info.value.message
        )


@pytest.mark.asyncio
async def test_split_non_existing_file_id_raises_notfound(
    server: FakeLlamaCloudServer, client: AsyncLlamaCloud
) -> None:
    file_id = "file-doesnotexist"
    with pytest.raises(APIStatusError) as exc_info:
        await client.beta.split.create(
            categories=[
                {"name": "hello", "description": ""},
                {"name": "world", "description": ""},
            ],
            document_input={"type": "file_id", "value": file_id},
        )
        assert exc_info.value.status_code == 404
        assert f"file with ID {file_id} not found" in exc_info.value.message


@pytest.mark.asyncio
async def test_split_non_existing_job_id_raises_notfound(
    server: FakeLlamaCloudServer, client: AsyncLlamaCloud
) -> None:
    with pytest.raises(APIStatusError) as exc_info:
        job_id = "split-doesnotexist"
        await client.beta.split.get(job_id)
        assert exc_info.value.status_code == 404
        assert f"job with ID {job_id} does not exist" in exc_info.value.message
