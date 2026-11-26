"""HTTP 协议适配层：身份验证、DTO、生命周期服务与 SSE。"""
import asyncio
import json
import os
import secrets
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, ConfigDict, Field

from artex.db import Repository
from artex.engine import Engine
from artex.events import EventService
from artex.llm_planner import ModelPlanner
from artex.worker import FixtureModel, Worker, WorkerDeepSeekModel


class TaskInput(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    description: str = Field(min_length=1, max_length=2000)
    goal: str = Field(min_length=1, max_length=2000)
    scope: list[str] = Field(min_length=1, max_length=100)


class RunInput(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    model_mode: Literal["fixture", "deepseek"] = "fixture"


def model_factory(mode):
    return WorkerDeepSeekModel if mode == "deepseek" else FixtureModel


def _local_token():
    token = os.getenv("ARTEX_API_TOKEN", "")
    if not token:
        path = Path(__file__).resolve().parents[1] / "config.local.json"
        if path.exists():
            token = json.loads(
                path.read_text(encoding="utf-8-sig")
            ).get("api_token", "")
    if not token or token == "GENERATE_A_LOCAL_RANDOM_TOKEN":
        raise ValueError(
            "先设置 ARTEX_API_TOKEN 或 config.local.json 的 api_token"
        )
    return token


def create_app(
    repo=None,
    api_token=None,
    worker_factory=None,
    planner_factory=None,
):
    owns_repo = repo is None
    repo = repo if repo is not None else Repository()
    token = api_token or _local_token()
    engines, jobs, modes = {}, {}, {}

    def get_engine(task_id, mode=None):
        repo.get_task(task_id)
        saved = repo.get_record("web_runtime", "mode", task_id)
        selected = mode or (
            saved["value"]["model_mode"] if saved else "fixture"
        )
        if selected not in ("fixture", "deepseek"):
            raise HTTPException(409, "持久化模型配置无效")

        if mode is not None and (
            saved is None or saved["value"]["model_mode"] != mode
        ):
            if task_id in jobs and not jobs[task_id].done():
                raise HTTPException(409, "运行期间不能切换模型")
            if not repo.put_record(
                "web_runtime",
                "mode",
                {"model_mode": mode},
                task_id,
                expected_version=saved["version"] if saved else 0,
            ):
                raise HTTPException(409, "模型配置已变化，请重试")
        mode = selected

        if task_id in engines and modes[task_id] != mode:
            if task_id in jobs and not jobs[task_id].done():
                raise HTTPException(409, "运行期间不能切换模型")
            del engines[task_id]

        if task_id not in engines:
            worker = (
                worker_factory(repo, mode)
                if worker_factory
                else Worker(
                    repo, model_factory=model_factory(mode)
                )
            )
            planner = (
                planner_factory(repo, mode)
                if planner_factory
                else (
                    ModelPlanner(repo)
                    if mode == "deepseek" and worker_factory is None
                    else None
                )
            )
            engines[task_id] = Engine(
                repo, worker=worker, planner=planner
            )
            modes[task_id] = mode
        return engines[task_id]

    async def launch(task_id, mode=None, resume=False):
        state = repo.get_task(task_id)["status"]
        if state in ("deleted", "deleting", "cancelled", "failed"):
            raise HTTPException(409, "当前状态不能启动")
        if task_id in jobs and not jobs[task_id].done():
            return {"accepted": True, "already_running": True}

        engine = get_engine(task_id, mode)

        async def run():
            try:
                if resume:
                    await engine.resume(task_id)
                else:
                    await engine.run(task_id)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                task = repo.get_task(task_id)
                if task["status"] not in ("deleting", "deleted"):
                    repo.append_event(
                        task_id,
                        "api_run_failed",
                        {"error_type": type(exc).__name__},
                    )

        jobs[task_id] = asyncio.create_task(run())
        return {"accepted": True, "already_running": False}

    @asynccontextmanager
    async def lifespan(app):
        yield
        for task_id, job in list(jobs.items()):
            if not job.done():
                if repo.get_task(task_id)["status"] in (
                    "created",
                    "running",
                ):
                    await get_engine(task_id).pause(task_id)
                job.cancel()
        if jobs:
            await asyncio.gather(
                *jobs.values(), return_exceptions=True
            )
        if owns_repo:
            repo.close()

    app = FastAPI(
        title="Python ARTEX 学习版", lifespan=lifespan
    )
    bearer = HTTPBearer(auto_error=False)

    async def authorize(
        credentials: HTTPAuthorizationCredentials | None = Depends(
            bearer
        ),
    ):
        if (
            credentials is None
            or credentials.scheme.lower() != "bearer"
            or not secrets.compare_digest(
                credentials.credentials, token
            )
        ):
            raise HTTPException(
                401,
                "需要有效的 Bearer token",
                headers={"WWW-Authenticate": "Bearer"},
            )

    app.state.repo = repo
    app.state.authorize = authorize
    app.state.get_engine = get_engine
    app.state.launch = launch
    app.state.jobs = jobs

    @app.exception_handler(KeyError)
    async def missing(request, exc):
        return JSONResponse(
            status_code=404, content={"detail": "资源不存在"}
        )

    @app.exception_handler(ValueError)
    async def conflict(request, exc):
        return JSONResponse(
            status_code=409, content={"detail": "状态或参数冲突"}
        )

    auth = [Depends(authorize)]

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    @app.get("/", response_class=HTMLResponse)
    async def index():
        return (Path(__file__).with_name("web.html")).read_text(
            encoding="utf-8"
        )

    @app.post("/api/tasks", status_code=201, dependencies=auth)
    async def create_task(body: TaskInput):
        return repo.create_task(
            body.description, body.goal, scope=body.scope
        )

    @app.get("/api/tasks", dependencies=auth)
    async def tasks():
        return {"items": repo.list_tasks()}

    @app.get("/api/tasks/{task_id}", dependencies=auth)
    async def task(task_id: str):
        return repo.get_task(task_id)

    @app.get("/api/tasks/{task_id}/graph", dependencies=auth)
    async def graph(task_id: str):
        repo.get_task(task_id)
        return {
            "nodes": repo.list_nodes(task_id),
            "edges": repo.list_edges(task_id),
        }

    @app.post(
        "/api/tasks/{task_id}/run",
        status_code=202,
        dependencies=auth,
    )
    async def run_task(task_id: str, body: RunInput):
        return await launch(task_id, body.model_mode)

    @app.post("/api/tasks/{task_id}/pause", dependencies=auth)
    async def pause(task_id: str):
        await get_engine(task_id).pause(task_id)
        return repo.get_task(task_id)

    @app.post(
        "/api/tasks/{task_id}/resume",
        status_code=202,
        dependencies=auth,
    )
    async def resume(task_id: str):
        if repo.get_task(task_id)["status"] != "paused":
            raise HTTPException(409, "只有暂停任务可以恢复")
        return await launch(task_id, resume=True)

    @app.get("/api/tasks/{task_id}/events", dependencies=auth)
    async def events(
        task_id: str,
        since: int = Query(0, ge=0),
        limit: int = Query(100, ge=1, le=1000),
    ):
        repo.get_task(task_id)
        items = repo.list_events(
            task_id, since=since, limit=limit
        )
        return {
            "items": items,
            "cursor": items[-1]["id"] if items else since,
        }

    @app.get("/api/tasks/{task_id}/stream", dependencies=auth)
    async def stream(
        task_id: str,
        request: Request,
        since: int = Query(0, ge=0),
    ):
        repo.get_task(task_id)
        cursor = request.headers.get("last-event-id")
        if cursor is not None:
            try:
                since = max(0, int(cursor))
            except ValueError:
                raise HTTPException(
                    400, "Last-Event-ID 必须是整数"
                ) from None
        return StreamingResponse(
            EventService(repo).sse(task_id, since),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )

    return app
