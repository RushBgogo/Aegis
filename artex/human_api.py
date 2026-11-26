"""Human-in-the-loop HTTP 路由：查询、提示、注入、纠偏与旁路问答。"""
from fastapi import Depends
from pydantic import BaseModel, ConfigDict, Field

from artex.human import HumanService, InterventionModel
from artex.llm_planner import ModelPlanner
from artex.web import create_app as base_app, model_factory
from artex.worker import Worker


class HintInput(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")

    request_id: str = Field(min_length=1, max_length=100)
    text: str = Field(min_length=1, max_length=2000)


class IntentInput(HintInput):
    target: str


class SteerInput(HintInput):
    intent_id: str


class QuestionInput(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")

    question: str = Field(min_length=1, max_length=1000)


def create_app(repo=None, api_token=None):
    def workers(store, mode):
        # 每个 Worker 执行单独创建包装器；纠偏已读集合不跨意图共享。
        return Worker(
            store,
            model_factory=lambda: InterventionModel(
                store, model_factory(mode)()
            ),
        )

    app = base_app(
        repo=repo,
        api_token=api_token,
        worker_factory=workers,
        planner_factory=lambda store, mode: (
            ModelPlanner(store) if mode == "deepseek" else None
        ),
    )
    service = HumanService(app.state.repo)
    auth = [Depends(app.state.authorize)]

    @app.get("/api/tasks/{task_id}/summary", dependencies=auth)
    async def summary(task_id: str):
        return service.summary(task_id)

    @app.post("/api/tasks/{task_id}/hints", dependencies=auth)
    async def hint(task_id: str, body: HintInput):
        return service.add_hint(
            task_id, body.text, body.request_id
        )

    @app.post("/api/tasks/{task_id}/intents", dependencies=auth)
    async def intent(task_id: str, body: IntentInput):
        node = await service.inject_intent(
            task_id, body.target, body.text, body.request_id
        )
        state = app.state.repo.get_task(task_id)["status"]
        if state == "running":
            await app.state.launch(task_id)
        return node

    @app.post("/api/tasks/{task_id}/steer", dependencies=auth)
    async def steer(task_id: str, body: SteerInput):
        return service.steer(
            task_id,
            body.intent_id,
            body.text,
            body.request_id,
        )

    @app.post(
        "/api/tasks/{task_id}/questions", dependencies=auth
    )
    async def question(task_id: str, body: QuestionInput):
        return await service.ask(task_id, body.question)

    return app
