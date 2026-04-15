from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any, Dict, Optional

import httpx
from llama_cloud.types.configuration_response import (
    ConfigurationResponse,
    ExtractV2Parameters,
)

from ._deterministic import utcnow

if TYPE_CHECKING:
    from .server import FakeLlamaCloudServer


@dataclass
class StoredConfiguration:
    id: str
    name: str
    parameters: Dict[str, Any]
    created_at: datetime = field(default_factory=utcnow)
    updated_at: datetime = field(default_factory=utcnow)


class FakeConfigurationsNamespace:
    """Mocks the llama-cloud v2 configurations API.

    Endpoints covered:
        GET    /api/v1/beta/configurations/{config_id}       retrieve
        GET    /api/v1/beta/configurations                   list
        POST   /api/v1/beta/configurations                   create
        PATCH  /api/v1/beta/configurations/{config_id}       update
        DELETE /api/v1/beta/configurations/{config_id}       delete
    """

    def __init__(self, *, server: "FakeLlamaCloudServer") -> None:
        self._server = server
        self._configurations: Dict[str, StoredConfiguration] = {}

    # Public API -----------------------------------------------------
    def create(
        self,
        *,
        name: str,
        parameters: Dict[str, Any],
    ) -> StoredConfiguration:
        config_id = self._server.new_id("cfg")
        stored = StoredConfiguration(id=config_id, name=name, parameters=parameters)
        self._configurations[config_id] = stored
        return stored

    def get(self, config_id: str) -> Optional[StoredConfiguration]:
        return self._configurations.get(config_id)

    # Route registration ---------------------------------------------
    def register(self) -> None:
        server = self._server
        server.add_route(
            "GET",
            "/api/v1/beta/configurations",
            self._handle_list,
            namespace="configurations",
        )
        server.add_route(
            "POST",
            "/api/v1/beta/configurations",
            self._handle_create,
            namespace="configurations",
        )
        server.add_route(
            "GET",
            "/api/v1/beta/configurations/{config_id}",
            self._handle_get,
            namespace="configurations",
        )
        server.add_route(
            "PATCH",
            "/api/v1/beta/configurations/{config_id}",
            self._handle_update,
            namespace="configurations",
        )
        server.add_route(
            "DELETE",
            "/api/v1/beta/configurations/{config_id}",
            self._handle_delete,
            namespace="configurations",
        )

    # Handlers -------------------------------------------------------
    def _handle_create(self, request: httpx.Request) -> httpx.Response:
        payload = self._server.json(request)
        stored = self.create(name=payload["name"], parameters=payload["parameters"])
        return self._server.json_response(self._to_dict(stored))

    def _handle_get(self, request: httpx.Request) -> httpx.Response:
        config_id = request.url.path.rstrip("/").split("/")[-1]
        stored = self._configurations.get(config_id)
        if not stored:
            return self._server.json_response(
                {"detail": "Configuration not found"}, status_code=404
            )
        return self._server.json_response(self._to_dict(stored))

    def _handle_update(self, request: httpx.Request) -> httpx.Response:
        config_id = request.url.path.rstrip("/").split("/")[-1]
        stored = self._configurations.get(config_id)
        if not stored:
            return self._server.json_response(
                {"detail": "Configuration not found"}, status_code=404
            )
        payload = self._server.json(request)
        if "name" in payload and payload["name"] is not None:
            stored.name = payload["name"]
        if "parameters" in payload and payload["parameters"] is not None:
            stored.parameters = payload["parameters"]
        stored.updated_at = utcnow()
        return self._server.json_response(self._to_dict(stored))

    def _handle_delete(self, request: httpx.Request) -> httpx.Response:
        config_id = request.url.path.rstrip("/").split("/")[-1]
        self._configurations.pop(config_id, None)
        return self._server.json_response({}, status_code=200)

    def _handle_list(self, request: httpx.Request) -> httpx.Response:
        product_type = request.url.params.get_list("product_type")
        items = []
        for stored in self._configurations.values():
            pt = stored.parameters.get("product_type")
            if product_type and pt not in product_type:
                continue
            items.append(self._to_dict(stored))
        return self._server.json_response(
            {"items": items, "next_page_token": None, "has_more": False}
        )

    # Helpers --------------------------------------------------------
    def _to_dict(self, stored: StoredConfiguration) -> Dict[str, Any]:
        product_type = stored.parameters.get("product_type", "extract_v2")
        response = ConfigurationResponse(
            id=stored.id,
            name=stored.name,
            parameters=ExtractV2Parameters(**stored.parameters)
            if product_type == "extract_v2"
            else stored.parameters,  # type: ignore[arg-type]
            product_type=product_type,
            version=stored.updated_at.isoformat(),
            created_at=stored.created_at,
            updated_at=stored.updated_at,
        )
        return response.model_dump(mode="json")
