"""Tests for the FakePipelinesNamespace mock implementation."""

import pytest
from extraction_review.testing_utils import FakeLlamaCloudServer
from llama_cloud import APIStatusError, AsyncLlamaCloud


@pytest.fixture
def server():
    """Provide a server with the pipelines namespace enabled."""
    with FakeLlamaCloudServer(namespaces=["pipelines"]) as srv:
        yield srv


@pytest.mark.asyncio
async def test_pipelines_create_and_get(server):
    """Verify a pipeline can be created and retrieved."""
    client = AsyncLlamaCloud(api_key="fake-api-key")

    pipeline = await client.pipelines.create(name="test-pipeline")
    assert pipeline.id.startswith("pipeline_")
    assert pipeline.name == "test-pipeline"
    assert pipeline.project_id == server.default_project_id
    assert pipeline.status == "CREATED"

    retrieved = await client.pipelines.get(pipeline.id)
    assert retrieved.id == pipeline.id
    assert retrieved.name == "test-pipeline"


@pytest.mark.asyncio
async def test_pipelines_create_with_embedding_config(server):
    """Verify a pipeline can be created with explicit embedding config."""
    client = AsyncLlamaCloud(api_key="fake-api-key")

    pipeline = await client.pipelines.create(
        name="custom-pipeline",
        embedding_config={
            "type": "MANAGED_OPENAI_EMBEDDING",
            "component": {},
        },
        pipeline_type="MANAGED",
    )
    assert pipeline.name == "custom-pipeline"
    assert pipeline.embedding_config is not None


@pytest.mark.asyncio
async def test_pipelines_list(server):
    """Verify listing pipelines returns created pipelines."""
    client = AsyncLlamaCloud(api_key="fake-api-key")

    await client.pipelines.create(name="pipeline-1")
    await client.pipelines.create(name="pipeline-2")

    pipelines = await client.pipelines.list()
    assert len(pipelines) == 2
    names = {p.name for p in pipelines}
    assert names == {"pipeline-1", "pipeline-2"}


@pytest.mark.asyncio
async def test_pipelines_get_not_found(server):
    """Verify non-existent pipeline returns 404."""
    client = AsyncLlamaCloud(api_key="fake-api-key")
    with pytest.raises(APIStatusError) as exc_info:
        await client.pipelines.get("nonexistent-id")
    assert exc_info.value.status_code == 404


@pytest.mark.asyncio
async def test_pipelines_delete(server):
    """Verify a pipeline can be deleted."""
    client = AsyncLlamaCloud(api_key="fake-api-key")

    pipeline = await client.pipelines.create(name="to-delete")
    retrieved = await client.pipelines.get(pipeline.id)
    assert retrieved.id == pipeline.id

    await client.pipelines.delete(pipeline.id)

    with pytest.raises(APIStatusError) as exc_info:
        await client.pipelines.get(pipeline.id)
    assert exc_info.value.status_code == 404


@pytest.mark.asyncio
async def test_pipelines_update(server):
    """Verify a pipeline can be updated."""
    client = AsyncLlamaCloud(api_key="fake-api-key")

    pipeline = await client.pipelines.create(name="original-name")
    updated = await client.pipelines.update(pipeline.id, name="new-name")
    assert updated.name == "new-name"
    assert updated.id == pipeline.id

    retrieved = await client.pipelines.get(pipeline.id)
    assert retrieved.name == "new-name"


@pytest.mark.asyncio
async def test_pipelines_get_status(server):
    """Verify pipeline status can be retrieved."""
    client = AsyncLlamaCloud(api_key="fake-api-key")

    pipeline = await client.pipelines.create(name="status-test")
    status = await client.pipelines.get_status(pipeline.id)
    assert status.status == "SUCCESS"


@pytest.mark.asyncio
async def test_pipelines_retrieve(server):
    """Verify pipeline retrieve endpoint works."""
    client = AsyncLlamaCloud(api_key="fake-api-key")

    pipeline = await client.pipelines.create(name="retrieve-test")
    result = await client.pipelines.retrieve(pipeline.id, query="test query")
    assert result.pipeline_id == pipeline.id
    assert result.retrieval_nodes is not None
    assert isinstance(result.retrieval_nodes, list)


# --- Document ingestion tests ---


@pytest.mark.asyncio
async def test_pipelines_ingest_documents_and_retrieve(server):
    """Ingest documents into a pipeline and verify retrieval returns nodes."""
    client = AsyncLlamaCloud(api_key="fake-api-key")

    pipeline = await client.pipelines.create(name="doc-ingest-test")

    docs = await client.pipelines.documents.create(
        pipeline.id,
        body=[
            {
                "text": "The quick brown fox jumps over the lazy dog.",
                "metadata": {"source": "test"},
            },
            {
                "text": "Machine learning is a subset of artificial intelligence.",
                "metadata": {"source": "ml"},
            },
        ],
    )
    assert len(docs) == 2
    assert docs[0].text == "The quick brown fox jumps over the lazy dog."
    assert docs[0].metadata["source"] == "test"
    assert docs[1].text == "Machine learning is a subset of artificial intelligence."

    result = await client.pipelines.retrieve(pipeline.id, query="fox")
    assert result.pipeline_id == pipeline.id
    assert len(result.retrieval_nodes) > 0

    # Nodes should have text content from our documents
    texts = [n.node.text for n in result.retrieval_nodes]
    all_doc_texts = {
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is a subset of artificial intelligence.",
    }
    for text in texts:
        assert text in all_doc_texts

    # Each node should have a score
    for node in result.retrieval_nodes:
        assert node.score is not None
        assert 0.0 <= node.score <= 1.0


@pytest.mark.asyncio
async def test_pipelines_upsert_documents(server):
    """Upsert documents (PUT) into a pipeline and verify they are stored."""
    client = AsyncLlamaCloud(api_key="fake-api-key")

    pipeline = await client.pipelines.create(name="doc-upsert-test")

    docs = await client.pipelines.documents.upsert(
        pipeline.id,
        body=[
            {"text": "First document content.", "metadata": {"order": "1"}},
        ],
    )
    assert len(docs) == 1
    assert docs[0].text == "First document content."

    # Upsert more documents
    docs2 = await client.pipelines.documents.upsert(
        pipeline.id,
        body=[
            {"text": "Second document content.", "metadata": {"order": "2"}},
        ],
    )
    assert len(docs2) == 1

    # Retrieve should pick up both documents
    result = await client.pipelines.retrieve(pipeline.id, query="document")
    assert len(result.retrieval_nodes) == 2


@pytest.mark.asyncio
async def test_pipelines_ingest_documents_with_custom_id(server):
    """Documents with explicit IDs preserve them."""
    client = AsyncLlamaCloud(api_key="fake-api-key")

    pipeline = await client.pipelines.create(name="custom-id-test")

    docs = await client.pipelines.documents.create(
        pipeline.id,
        body=[
            {"id": "my-doc-1", "text": "Custom ID document.", "metadata": {}},
        ],
    )
    assert len(docs) == 1
    assert docs[0].id == "my-doc-1"


# --- File ingestion tests ---


@pytest.mark.asyncio
async def test_pipelines_ingest_files_and_retrieve(server):
    """Add files to a pipeline and verify retrieval returns generated nodes."""
    client = AsyncLlamaCloud(api_key="fake-api-key")

    pipeline = await client.pipelines.create(name="file-ingest-test")

    files = await client.pipelines.files.create(
        pipeline.id,
        body=[
            {"file_id": "file-abc123"},
            {"file_id": "file-def456"},
        ],
    )
    assert len(files) == 2
    assert files[0].pipeline_id == pipeline.id
    assert files[0].status == "SUCCESS"
    assert files[1].file_id == "file-def456"

    result = await client.pipelines.retrieve(pipeline.id, query="search query")
    assert result.pipeline_id == pipeline.id
    assert len(result.retrieval_nodes) > 0

    # Nodes from files should have file metadata
    for node in result.retrieval_nodes:
        assert node.node.text  # Should have generated text
        assert node.score is not None


# --- Retrieval behavior tests ---


@pytest.mark.asyncio
async def test_pipelines_retrieve_empty_pipeline(server):
    """Retrieve on an empty pipeline returns no nodes."""
    client = AsyncLlamaCloud(api_key="fake-api-key")

    pipeline = await client.pipelines.create(name="empty-pipeline")
    result = await client.pipelines.retrieve(pipeline.id, query="anything")
    assert result.retrieval_nodes == []


@pytest.mark.asyncio
async def test_pipelines_retrieve_respects_top_k(server):
    """Retrieve respects the dense_similarity_top_k parameter."""
    client = AsyncLlamaCloud(api_key="fake-api-key")

    pipeline = await client.pipelines.create(name="topk-test")

    # Ingest several documents to have more chunks than top_k
    await client.pipelines.documents.create(
        pipeline.id,
        body=[
            {
                "text": f"Document number {i} with unique content about topic {i}.",
                "metadata": {},
            }
            for i in range(10)
        ],
    )

    result = await client.pipelines.retrieve(
        pipeline.id,
        query="topic",
        dense_similarity_top_k=2,
    )
    assert len(result.retrieval_nodes) == 2


@pytest.mark.asyncio
async def test_pipelines_retrieve_deterministic(server):
    """Same query on same data produces the same results."""
    client = AsyncLlamaCloud(api_key="fake-api-key")

    pipeline = await client.pipelines.create(name="deterministic-test")
    await client.pipelines.documents.create(
        pipeline.id,
        body=[
            {"text": "Alpha bravo charlie delta.", "metadata": {}},
            {"text": "Echo foxtrot golf hotel.", "metadata": {}},
        ],
    )

    result1 = await client.pipelines.retrieve(pipeline.id, query="bravo")
    result2 = await client.pipelines.retrieve(pipeline.id, query="bravo")

    texts1 = [n.node.text for n in result1.retrieval_nodes]
    texts2 = [n.node.text for n in result2.retrieval_nodes]
    assert texts1 == texts2

    scores1 = [n.score for n in result1.retrieval_nodes]
    scores2 = [n.score for n in result2.retrieval_nodes]
    assert scores1 == scores2


@pytest.mark.asyncio
async def test_pipelines_delete_cleans_up_documents_and_files(server):
    """Deleting a pipeline clears its ingested documents and files."""
    client = AsyncLlamaCloud(api_key="fake-api-key")

    pipeline = await client.pipelines.create(name="cleanup-test")

    await client.pipelines.documents.create(
        pipeline.id,
        body=[{"text": "Some content.", "metadata": {}}],
    )
    await client.pipelines.files.create(
        pipeline.id,
        body=[{"file_id": "file-xyz"}],
    )

    # Verify data exists
    result = await client.pipelines.retrieve(pipeline.id, query="content")
    assert len(result.retrieval_nodes) > 0

    # Delete pipeline
    await client.pipelines.delete(pipeline.id)

    # Internal stores should be cleaned
    assert pipeline.id not in server.pipelines._documents
    assert pipeline.id not in server.pipelines._files


@pytest.mark.asyncio
async def test_pipelines_mixed_documents_and_files_retrieval(server):
    """Retrieval combines results from both documents and files."""
    client = AsyncLlamaCloud(api_key="fake-api-key")

    pipeline = await client.pipelines.create(name="mixed-test")

    await client.pipelines.documents.create(
        pipeline.id,
        body=[
            {
                "text": "Document text about important concepts.",
                "metadata": {"type": "doc"},
            }
        ],
    )
    await client.pipelines.files.create(
        pipeline.id,
        body=[{"file_id": "file-mixed-001"}],
    )

    result = await client.pipelines.retrieve(
        pipeline.id,
        query="concepts",
        dense_similarity_top_k=10,
    )
    assert len(result.retrieval_nodes) > 1

    # Should have nodes from both sources
    has_doc_node = any(
        n.node.text == "Document text about important concepts."
        for n in result.retrieval_nodes
    )
    has_file_node = any(
        n.node.extra_info and n.node.extra_info.get("file_id") == "file-mixed-001"
        for n in result.retrieval_nodes
    )
    assert has_doc_node, "Should have a node from the ingested document"
    assert has_file_node, "Should have a node from the ingested file"
