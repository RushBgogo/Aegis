"""证据绑定与报告 HTTP 路由。"""
from fastapi import Depends, HTTPException
from pydantic import BaseModel, ConfigDict

from artex.assets_api import create_app as previous_app
from artex.reporting import ReportService


class BindingInput(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")

    finding_node_id: str
    evidence_id: str
    note: str


def create_app(repo=None, api_token=None):
    app = previous_app(repo, api_token)
    service = ReportService(app.state.repo)
    auth = [Depends(app.state.authorize)]

    @app.post(
        "/api/tasks/{task_id}/evidence-bindings",
        dependencies=auth,
    )
    async def bind(task_id: str, body: BindingInput):
        return service.bind(
            task_id,
            body.finding_node_id,
            body.evidence_id,
            body.note,
        )

    @app.post("/api/tasks/{task_id}/report", dependencies=auth)
    async def build(task_id: str):
        return service.build(task_id)

    @app.get("/api/tasks/{task_id}/report", dependencies=auth)
    async def report(task_id: str):
        app.state.repo.get_task(task_id)
        record = app.state.repo.get_record(
            "report", "latest", task_id
        )
        if not record:
            raise HTTPException(404, "报告尚未生成")
        return {
            **record["value"],
            "stale": service.is_stale(task_id),
        }

    return app
