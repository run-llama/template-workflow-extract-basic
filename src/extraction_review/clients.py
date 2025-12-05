import os
from typing import Any
import httpx

from llama_cloud_services import LlamaExtract
from llama_cloud_services.beta.agent_data import AsyncAgentDataClient, ExtractedData
from llama_cloud.client import AsyncLlamaCloud
from .testing_utils import FakeLlamaCloudServer
import logging

from extraction_review.config import (
    EXTRACTED_DATA_COLLECTION,
)

logger = logging.getLogger(__name__)

# deployed agents may infer their name from the deployment name
# Note: Make sure that an agent deployment with this name actually exists
# otherwise calls to get or set data will fail. You may need to adjust the `or `
# name for development
agent_name = os.getenv("LLAMA_DEPLOY_DEPLOYMENT_NAME")
# required for all llama cloud calls
api_key = os.getenv("LLAMA_CLOUD_API_KEY")
# get this in case running against a different environment than production
base_url = os.getenv("LLAMA_CLOUD_BASE_URL")
project_id = os.getenv("LLAMA_DEPLOY_PROJECT_ID")

if os.getenv("FAKE_LLAMA_CLOUD"):
    fake = FakeLlamaCloudServer().install()
else:
    fake = None


def get_llama_extract() -> LlamaExtract:
    return LlamaExtract(api_key=api_key, base_url=base_url, project_id=project_id)


def get_data_client() -> AsyncAgentDataClient:
    return AsyncAgentDataClient(
        deployment_name=agent_name,
        collection=EXTRACTED_DATA_COLLECTION,
        type=ExtractedData[Any],
        client=get_llama_cloud_client(),
    )


def get_llama_cloud_client() -> AsyncLlamaCloud:
    return AsyncLlamaCloud(
        base_url=base_url,
        token=api_key,
        httpx_client=httpx.AsyncClient(
            timeout=60, headers={"Project-Id": project_id} if project_id else None
        ),
    )
