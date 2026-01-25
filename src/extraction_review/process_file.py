import asyncio
import hashlib
import logging
import os
import tempfile
from pathlib import Path
from typing import Annotated, Any, Literal

import httpx
from llama_cloud import AsyncLlamaCloud
from llama_cloud.types.beta.extracted_data import ExtractedData, InvalidExtractionData
from llama_cloud.types.file_query_params import Filter
from pydantic import BaseModel
from workflows import Context, Workflow, step
from workflows.events import Event, StartEvent, StopEvent
from workflows.resource import Resource, ResourceConfig

from .clients import agent_name, get_llama_cloud_client, project_id
from .config import EXTRACTED_DATA_COLLECTION, ExtractConfig, get_extraction_schema

logger = logging.getLogger(__name__)


class FileEvent(StartEvent):
    file_id: str


class DownloadFileEvent(Event):
    pass


class FileDownloadedEvent(Event):
    pass


class Status(Event):
    level: Literal["info", "warning", "error"]
    message: str


class ExtractJobStartedEvent(Event):
    pass


class ExtractJobCompletedEvent(Event):
    pass


class ExtractedEvent(Event):
    data: ExtractedData


class ExtractedInvalidEvent(Event):
    """Event for extraction results that failed validation."""

    data: ExtractedData[dict[str, Any]]


class ExtractionState(BaseModel):
    file_id: str | None = None
    file_path: str | None = None
    filename: str | None = None
    file_hash: str | None = None
    extract_job_id: str | None = None


class ProcessFileWorkflow(Workflow):
    """Extract structured data from a document and save it for review."""

    @step()
    async def run_file(self, event: FileEvent, ctx: Context) -> DownloadFileEvent:
        """Start extraction for the uploaded document."""
        logger.info(f"Running file {event.file_id}")
        async with ctx.store.edit_state() as state:
            state.file_id = event.file_id
        return DownloadFileEvent()

    @step()
    async def download_file(
        self,
        event: DownloadFileEvent,
        ctx: Context[ExtractionState],
        llama_cloud_client: Annotated[
            AsyncLlamaCloud, Resource(get_llama_cloud_client)
        ],
    ) -> FileDownloadedEvent:
        """Retrieve the document from cloud storage for processing."""
        state = await ctx.store.get_state()
        if state.file_id is None:
            raise ValueError("File ID is not set")
        try:
            files = await llama_cloud_client.files.query(
                filter=Filter(file_ids=[state.file_id])
            )
            file_metadata = files.items[0]
            file_url = await llama_cloud_client.files.get(file_id=state.file_id)

            temp_dir = tempfile.gettempdir()
            filename = file_metadata.name
            file_path = os.path.join(temp_dir, filename)
            client = httpx.AsyncClient()
            # Report progress to the UI
            logger.info(f"Downloading file {file_url.url} to {file_path}")

            async with client.stream("GET", file_url.url) as response:
                with open(file_path, "wb") as f:
                    async for chunk in response.aiter_bytes():
                        f.write(chunk)
            logger.info(f"Downloaded file {file_url.url} to {file_path}")
            async with ctx.store.edit_state() as state:
                state.file_path = file_path
                state.filename = filename
            return FileDownloadedEvent()

        except Exception as e:
            logger.error(f"Error downloading file {state.file_id}: {e}", exc_info=True)
            ctx.write_event_to_stream(
                Status(
                    level="error",
                    message=f"Error downloading file {state.file_id}: {e}",
                )
            )
            raise e

    @step()
    async def process_file(
        self,
        event: FileDownloadedEvent,
        ctx: Context[ExtractionState],
        llama_cloud_client: Annotated[
            AsyncLlamaCloud, Resource(get_llama_cloud_client)
        ],
        extract_config: Annotated[
            ExtractConfig,
            ResourceConfig(
                config_file="configs/config.json",
                path_selector="extract",
                label="Extraction Settings",
                description="Configuration for document extraction quality and features",
            ),
        ],
    ) -> ExtractJobStartedEvent:
        """Extract structured data fields from the document."""
        state = await ctx.store.get_state()
        if state.file_path is None or state.filename is None:
            raise ValueError("File path or filename is not set")
        # track the content of the file, so as to be able to de-duplicate
        file_content = Path(state.file_path).read_bytes()
        logger.info(f"Extracting data from file {state.filename}")
        ctx.write_event_to_stream(
            Status(level="info", message=f"Extracting data from file {state.filename}")
        )

        extract_job = await llama_cloud_client.extraction.run(
            config=extract_config.settings.model_dump(),
            data_schema=extract_config.json_schema,
            file_id=state.file_id,
            project_id=project_id,
        )
        async with ctx.store.edit_state() as st:
            st.file_hash = hashlib.sha256(file_content).hexdigest()
            st.extract_job_id = extract_job.id

        return ExtractJobStartedEvent()

    @step()
    async def wait_for_extract_job(
        self,
        event: ExtractJobStartedEvent,
        ctx: Context[ExtractionState],
        llama_cloud_client: Annotated[
            AsyncLlamaCloud, Resource(get_llama_cloud_client)
        ],
    ) -> ExtractJobCompletedEvent:
        state = await ctx.store.get_state()
        if state.extract_job_id is None:
            raise ValueError("Job ID cannot be null when waiting for its completion")
        await llama_cloud_client.extraction.jobs.wait_for_completion(
            state.extract_job_id
        )
        return ExtractJobCompletedEvent()

    @step()
    async def get_extraction_job_result(
        self,
        event: ExtractJobCompletedEvent,
        ctx: Context[ExtractionState],
        llama_cloud_client: Annotated[
            AsyncLlamaCloud, Resource(get_llama_cloud_client)
        ],
        extract_config: Annotated[
            ExtractConfig,
            ResourceConfig(
                config_file="configs/config.json",
                path_selector="extract",
                label="Extraction Settings",
                description="Configuration for document extraction quality and features",
            ),
        ],
    ) -> ExtractedEvent | ExtractedInvalidEvent:
        """Process the completed extraction job and validate results."""
        state = await ctx.store.get_state()
        if state.extract_job_id is None:
            raise ValueError("Job ID cannot be null when getting its result")
        extracted_result = await llama_cloud_client.extraction.jobs.get_result(
            state.extract_job_id
        )
        extract_run = await llama_cloud_client.extraction.runs.get(
            run_id=extracted_result.run_id
        )
        try:
            logger.info(f"Extracted data: {extracted_result}")
            # Create dynamic Pydantic model from JSON schema
            schema_class = get_extraction_schema(extract_config.json_schema)
            # Use from_extraction_result for proper metadata extraction
            data = ExtractedData.from_extraction_result(
                result=extract_run,
                schema=schema_class,
                file_name=state.filename,
                file_id=state.file_id,
                file_hash=state.file_hash,
            )
            return ExtractedEvent(data=data)
        except InvalidExtractionData as e:
            logger.error(f"Error validating extracted data: {e}", exc_info=True)
            return ExtractedInvalidEvent(data=e.invalid_item)
        except Exception as e:
            logger.error(
                f"Error extracting data from file {state.filename}: {e}",
                exc_info=True,
            )
            ctx.write_event_to_stream(
                Status(
                    level="error",
                    message=f"Error extracting data from file {state.filename}: {e}",
                )
            )
            raise e

    @step()
    async def record_extracted_data(
        self,
        event: ExtractedEvent | ExtractedInvalidEvent,
        ctx: Context,
        llama_cloud_client: Annotated[
            AsyncLlamaCloud, Resource(get_llama_cloud_client)
        ],
    ) -> StopEvent:
        """Save extracted data for human review and correction."""
        data = event.data.model_dump()
        # remove past data when reprocessing the same file
        if event.data.file_hash is not None:
            await llama_cloud_client.beta.agent_data.delete_by_query(
                deployment_name=agent_name or "_public",
                collection=EXTRACTED_DATA_COLLECTION,
                filter={
                    "file_hash": {
                        "eq": event.data.file_hash,
                    },
                },
            )
        logger.info(
            f"Removing past data for file {event.data.file_name} with hash {event.data.file_hash}"
        )
        # finally, save the new data
        item = await llama_cloud_client.beta.agent_data.agent_data(
            data=data,
            deployment_name=agent_name or "_public",
            collection=EXTRACTED_DATA_COLLECTION,
        )
        logger.info(f"Recorded extracted data for file {event.data.file_name or ''}")
        ctx.write_event_to_stream(
            Status(
                level="info",
                message=f"Recorded extracted data for file {event.data.file_name or ''}",
            )
        )
        return StopEvent(result=item.id)


workflow = ProcessFileWorkflow(timeout=None)

if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()
    logging.basicConfig(level=logging.INFO)

    async def main():
        file = await get_llama_cloud_client().files.create(
            file=Path("test.pdf").open("rb"),
            purpose="extract",
        )
        await workflow.run(start_event=FileEvent(file_id=file.id))

    asyncio.run(main())
