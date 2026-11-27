"""资产与覆盖度 HTTP 路由。"""
from typing import Literal

from fastapi import Depends
from pydantic import BaseModel, ConfigDict

from artex.assets import AssetService
from artex.human_api import create_app as previous_app


class AssetInput(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")

    kind: Literal["domain", "ip", "service", "endpoint"]
    value: str


def create_app(repo=None, api_token=None):
    app = previous_app(repo, api_token)
    service = AssetService(app.state.repo)
    auth = [Depends(app.state.authorize)]

    @app.post("/api/tasks/{task_id}/assets", dependencies=auth)
    async def add(task_id: str, body: AssetInput):
        return service.upsert(task_id, body.kind, body.value)

    @app.get("/api/tasks/{task_id}/assets", dependencies=auth)
    async def assets(task_id: str):
        app.state.repo.get_task(task_id)
        return {
            "items": app.state.repo.list_records(
                "asset", task_id
            )
        }

    @app.get("/api/tasks/{task_id}/coverage", dependencies=auth)
    async def coverage(task_id: str):
        return service.coverage(task_id)

    return app
