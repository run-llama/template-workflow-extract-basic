from typing import Any
import jsonref
from workflows import Workflow, step
from workflows.events import StartEvent, StopEvent

from .config import EXTRACTED_DATA_COLLECTION, ExtractionSchema


class MetadataResponse(StopEvent):
    # Given a JSON _object_ schema, the UI will dynamically generate (editable) form
    # rendering the extracted data
    json_schema: dict[str, Any]
    extracted_data_collection: str


async def get_extraction_schema_json() -> dict[str, Any]:
    # If the extracted schema differs from the presentation schema, modify this to be the presentation schema
    json_schema = ExtractionSchema.model_json_schema()
    json_schema = jsonref.replace_refs(json_schema, proxies=False)
    return json_schema


class MetadataWorkflow(Workflow):
    """
    Simple single step workflow to expose configuration to the UI, such as the JSON schema and collection name.
    """

    @step
    async def get_metadata(self, _: StartEvent) -> MetadataResponse:
        json_schema = await get_extraction_schema_json()
        return MetadataResponse(
            json_schema=json_schema,
            extracted_data_collection=EXTRACTED_DATA_COLLECTION,
        )


workflow = MetadataWorkflow(timeout=None)
