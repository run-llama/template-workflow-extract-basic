import asyncio
import hashlib
import logging
import os
from pathlib import Path
import tempfile
from typing import Any, Literal, Annotated

import httpx
from llama_cloud import ExtractRun
from llama_cloud.client import AsyncLlamaCloud
from llama_cloud_services.extract import SourceText, LlamaExtract
from llama_cloud_services.beta.agent_data import (
    ExtractedData,
    InvalidExtractionData,
    AsyncAgentDataClient,
)
from pydantic import BaseModel
from workflows.resource import Resource
from workflows import Context, Workflow, step
from workflows.events import Event, StartEvent, StopEvent

from .config import EXTRACT_CONFIG, ExtractionSchema
from .clients import get_llama_cloud_client, get_data_client, get_llama_extract

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


class ExtractedEvent(Event):
    data: ExtractedData


class ExtractedInvalidEvent(Event):
    data: ExtractedData[dict[str, Any]]


class ExtractionState(BaseModel):
    file_id: str | None = None
    file_path: str | None = None
    filename: str | None = None


class ProcessFileWorkflow(Workflow):
    """
    Given a file path, this workflow will process a single file through the custom extraction logic.
    """

    @step()
    async def run_file(self, event: FileEvent, ctx: Context) -> DownloadFileEvent:
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
        """Download the file reference from the cloud storage"""
        state = await ctx.store.get_state()
        if state.file_id is None:
            raise ValueError("File ID is not set")
        try:
            file_metadata = await llama_cloud_client.files.get_file(id=state.file_id)
            file_url = await llama_cloud_client.files.read_file_content(state.file_id)

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
        extractor: Annotated[LlamaExtract, Resource(get_llama_extract)],
    ) -> ExtractedEvent | ExtractedInvalidEvent:
        """Runs the extraction against the file"""
        state = await ctx.store.get_state()
        if state.file_path is None or state.filename is None:
            raise ValueError("File path or filename is not set")
        try:
            # track the content of the file, so as to be able to de-duplicate
            file_content = Path(state.file_path).read_bytes()
            file_hash = hashlib.sha256(file_content).hexdigest()
            source_text = SourceText(
                file=state.file_path,
                filename=state.filename,
            )
            logger.info(f"Extracting data from file {state.filename}")
            ctx.write_event_to_stream(
                Status(
                    level="info", message=f"Extracting data from file {state.filename}"
                )
            )
            extracted_result: ExtractRun = await extractor.aextract(
                data_schema=ExtractionSchema, files=source_text, config=EXTRACT_CONFIG
            )
            try:
                logger.info(f"Extracted data: {extracted_result}")
                data = ExtractedData.from_extraction_result(
                    result=extracted_result,
                    schema=ExtractionSchema,
                    # retain original file name and id, rather than using the extracted duplicate file
                    file_name=state.filename,
                    file_id=state.file_id,
                    file_hash=file_hash,
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
        data_client: Annotated[AsyncAgentDataClient, Resource(get_data_client)],
    ) -> StopEvent:
        """Records the extracted data to the agent data API"""
        try:
            logger.info(f"Recorded extracted data for file {event.data.file_name}")
            ctx.write_event_to_stream(
                Status(
                    level="info",
                    message=f"Recorded extracted data for file {event.data.file_name}",
                )
            )
            # remove past data when reprocessing the same file
            if event.data.file_hash:
                await data_client.delete(
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
            item_id = await data_client.create_item(event.data)
            return StopEvent(
                result=item_id.id,
            )
        except Exception as e:
            logger.error(
                f"Error recording extracted data for file {event.data.file_name}: {e}",
                exc_info=True,
            )
            ctx.write_event_to_stream(
                Status(
                    level="error",
                    message=f"Error recording extracted data for file {event.data.file_name}: {e}",
                )
            )
            raise e


workflow = ProcessFileWorkflow(timeout=None)

if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()
    logging.basicConfig(level=logging.INFO)

    async def main():
        file = await get_llama_cloud_client().files.upload_file(
            upload_file=Path("test.pdf").open("rb")
        )
        await workflow.run(start_event=FileEvent(file_id=file.id))

    asyncio.run(main())
