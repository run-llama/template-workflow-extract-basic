from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import httpx
from llama_cloud.types.extract_configuration import ExtractConfiguration
from llama_cloud.types.extract_v2_job import ExtractV2Job

from ._deterministic import (
    combined_seed,
    generate_data_from_schema,
    hash_schema,
    utcnow,
)
from .configurations import FakeConfigurationsNamespace
from .files import FakeFilesNamespace, StoredFile
from .matchers import RequestContext, RequestMatcher

if TYPE_CHECKING:
    from .server import FakeLlamaCloudServer


@dataclass
class ExtractRunStub:
    matcher: Optional[RequestMatcher]
    data: Optional[Any]
    status: Optional[str]
    metadata: Optional[Dict[str, Any]]
    error: Optional[str]
    once: bool


@dataclass
class StoredJob:
    job: ExtractV2Job
    data_schema: Dict[str, Any]
    file: StoredFile


class FakeExtractNamespace:
    """Mocks the llama-cloud v2 extract API.

    Endpoints covered:
        POST   /api/v2/extract                               create extract job
        GET    /api/v2/extract/{job_id}                      get job
        GET    /api/v2/extract                               list jobs
        DELETE /api/v2/extract/{job_id}                      delete job
        POST   /api/v2/extract/schema/validation             validate schema
        POST   /api/v2/extract/schema/generate               generate schema
    """

    def __init__(
        self,
        *,
        server: "FakeLlamaCloudServer",
        files: FakeFilesNamespace,
        configurations: FakeConfigurationsNamespace,
    ) -> None:
        self._server = server
        self._files = files
        self._configurations = configurations
        self._jobs: Dict[str, StoredJob] = {}
        self._run_stubs: List[ExtractRunStub] = []
        self.routes: Dict[str, Any] = {}

    # Public stub APIs -----------------------------------------------
    def stub_run(
        self,
        matcher: Optional[RequestMatcher],
        *,
        data: Optional[Any] = None,
        status: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None,
        once: bool = True,
    ) -> None:
        self._run_stubs.append(
            ExtractRunStub(
                matcher=matcher,
                data=data,
                status=status,
                metadata=metadata,
                error=error,
                once=once,
            )
        )

    # Route registration ---------------------------------------------
    def register(self) -> None:
        server = self._server
        create_route = server.add_route(
            "POST",
            "/api/v2/extract",
            self._handle_create_job,
            namespace="extract",
            alias="extract_create",
        )
        self.routes["create"] = create_route
        server.add_route(
            "GET",
            "/api/v2/extract",
            self._handle_list_jobs,
            namespace="extract",
        )
        server.add_route(
            "GET",
            "/api/v2/extract/{job_id}",
            self._handle_get_job,
            namespace="extract",
        )
        server.add_route(
            "DELETE",
            "/api/v2/extract/{job_id}",
            self._handle_delete_job,
            namespace="extract",
        )
        server.add_route(
            "POST",
            "/api/v2/extract/schema/validation",
            self._handle_validate_schema,
            namespace="extract",
        )
        server.add_route(
            "POST",
            "/api/v2/extract/schema/generate",
            self._handle_generate_schema,
            namespace="extract",
        )

    # Handlers -------------------------------------------------------
    def _handle_create_job(self, request: httpx.Request) -> httpx.Response:
        payload = self._server.json(request)
        file_input = payload["file_input"]
        stored_file = self._files.get(file_input)
        if not stored_file:
            return self._server.json_response(
                {"detail": f"File {file_input} not found"}, status_code=404
            )

        configuration = payload.get("configuration")
        configuration_id = payload.get("configuration_id")
        if configuration and configuration_id:
            return self._server.json_response(
                {"detail": "Provide configuration OR configuration_id"},
                status_code=400,
            )

        if configuration_id:
            cfg = self._configurations.get(configuration_id)
            if not cfg:
                return self._server.json_response(
                    {"detail": "Configuration not found"}, status_code=404
                )
            params = cfg.parameters
            data_schema = params["data_schema"]
            extract_config = {
                k: v for k, v in params.items() if k not in ("product_type",)
            }
        else:
            if not configuration:
                return self._server.json_response(
                    {"detail": "configuration or configuration_id required"},
                    status_code=400,
                )
            data_schema = configuration["data_schema"]
            extract_config = dict(configuration)

        context = RequestContext(
            request=request,
            json=payload,
            file_id=stored_file.file.id,
            filename=stored_file.file.name,
            file_sha256=stored_file.sha256,
            schema_hash=hash_schema(data_schema),
            project_id=stored_file.file.project_id,
            organization_id=self._server.default_organization_id,
        )

        stub = self._pop_stub(self._run_stubs, context)
        status = "COMPLETED"
        run_data = self._generate_run_data(data_schema, stored_file.sha256)
        error_message: Optional[str] = None

        if stub:
            if stub.status:
                status = stub.status
            if stub.error:
                error_message = stub.error
                status = "FAILED"
            if stub.data is not None:
                if callable(stub.data):
                    run_data = stub.data(payload)  # type: ignore[assignment]
                else:
                    run_data = stub.data

        job_id = self._server.new_id("exj")
        now = utcnow()
        job = ExtractV2Job(
            id=job_id,
            created_at=now,
            updated_at=now,
            file_input=file_input,
            project_id=stored_file.file.project_id,
            status=status,
            configuration=ExtractConfiguration(**extract_config)
            if not configuration_id
            else None,
            configuration_id=configuration_id,
            error_message=error_message,
            extract_result=run_data if status == "COMPLETED" else None,
            extract_metadata=None,
            metadata=None,
        )
        stored = StoredJob(job=job, data_schema=data_schema, file=stored_file)
        self._jobs[job_id] = stored
        return self._server.json_response(job.model_dump(mode="json"))

    def _handle_get_job(self, request: httpx.Request) -> httpx.Response:
        job_id = request.url.path.rstrip("/").split("/")[-1]
        stored = self._jobs.get(job_id)
        if not stored:
            return self._server.json_response(
                {"detail": "Job not found"}, status_code=404
            )
        return self._server.json_response(stored.job.model_dump(mode="json"))

    def _handle_list_jobs(self, request: httpx.Request) -> httpx.Response:
        items = [stored.job.model_dump(mode="json") for stored in self._jobs.values()]
        return self._server.json_response(
            {"items": items, "next_page_token": None, "has_more": False}
        )

    def _handle_delete_job(self, request: httpx.Request) -> httpx.Response:
        job_id = request.url.path.rstrip("/").split("/")[-1]
        self._jobs.pop(job_id, None)
        return self._server.json_response({}, status_code=200)

    def _handle_validate_schema(self, request: httpx.Request) -> httpx.Response:
        payload = self._server.json(request)
        return self._server.json_response({"data_schema": payload["data_schema"]})

    def _handle_generate_schema(self, request: httpx.Request) -> httpx.Response:
        payload = self._server.json(request)
        name = payload.get("name") or "generated"
        schema = payload.get("data_schema") or {
            "type": "object",
            "properties": {},
        }
        return self._server.json_response(
            {
                "name": name,
                "parameters": {
                    "product_type": "extract_v2",
                    "data_schema": schema,
                },
            }
        )

    # Helpers --------------------------------------------------------
    def _generate_run_data(self, schema: Dict[str, Any], file_hash: str) -> Any:
        seed = combined_seed(file_hash, hash_schema(schema))
        return generate_data_from_schema(schema, seed)

    def _pop_stub(
        self,
        stubs: List[ExtractRunStub],
        context: RequestContext,
    ) -> Optional[ExtractRunStub]:
        for index, stub in enumerate(list(stubs)):
            if context.matches(stub.matcher):
                if stub.once:
                    stubs.pop(index)
                return stub
        return None
