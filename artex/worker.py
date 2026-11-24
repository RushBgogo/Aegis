"""Day 3 Worker：把一条 intent 落实到第一课 AgentRuntime，并把真实工具结果写回事实与证据。"""
import asyncio
from contextlib import AsyncExitStack
from copy import deepcopy
import hashlib
import json

from runtime import (
    AgentRuntime,
    Message,
    ModelReply,
    ToolCall,
    ToolInput,
    ToolRegistry,
    ToolSpec,
)

DEFAULT_FIXTURES = {
    "https://lab.example/public": {"status": 200, "body": "public sample", "issue": False},
    "https://lab.example/private": {"status": 200, "body": "fixture-only verified mismatch", "issue": True},
}

WORKER_SYSTEM_PROMPT = (
    "你是意图执行者。用户 JSON 给出 target。必须调用 observe(target)，"
    "仅使用给定目标；根据实际工具结果回答。不要调用 add，不要猜测证据。"
)


class ObserveInput(ToolInput):
    target: str


class FixtureModel:
    async def complete(self, messages: list[Message], tools: list[dict]) -> ModelReply:
        last = messages[-1]
        if last.role == "tool":
            return ModelReply(text=json.dumps(last.result.content, ensure_ascii=False))
        prompt = json.loads(last.content)
        return ModelReply(tool_calls=(ToolCall("observe-1", "observe", {"target": prompt["target"]}),))


class WorkerDeepSeekModel:
    """复用第一课传输；把 Worker 指令作为唯一 system prompt，避免与 add 指令冲突。"""

    def __init__(self, model_factory=None):
        if model_factory is None:
            from deepseek_model import DeepSeekModel
            model_factory = DeepSeekModel
        self.inner = model_factory(system_prompt=WORKER_SYSTEM_PROMPT)

    async def __aenter__(self):
        await self.inner.__aenter__()
        return self

    async def __aexit__(self, *args):
        await self.inner.__aexit__(*args)

    async def complete(self, messages, tools):
        return await self.inner.complete(messages, tools)


class Worker:
    """执行一条 intent；不负责把整个 Task 标记 done，Task 生命周期由 Engine/Planner 汇总。"""

    def __init__(self, repo, model_factory=FixtureModel, executor=None, fixtures=None):
        self.repo = repo
        self.model_factory = model_factory
        self.executor = executor
        self.fixtures = deepcopy(DEFAULT_FIXTURES if fixtures is None else fixtures)

    async def execute(self, task_id, intent_id, owner="worker"):
        if not self.repo.claim_intent(task_id, intent_id, owner):
            return False
        intent = self.repo.get_node(task_id, intent_id)
        target = intent["payload"].get("target")
        observed = []
        trace = []
        call_id = [None]
        operation_id = hashlib.sha256(f"{intent_id}:observe:{target}".encode()).hexdigest()

        def current_run_id():
            return trace[0].run_id if trace else None

        async def observe(arguments):
            task = self.repo.get_task(task_id)
            if task["status"] != "running" or arguments.target not in task["scope"]:
                raise ValueError("target outside exact task scope or task not running")
            if arguments.target != target:
                raise ValueError("tool target differs from claimed intent")

            async def fetch():
                await asyncio.sleep(0)
                if target not in self.fixtures:
                    raise ValueError("fixture missing; no network fallback")
                return {"target": target, **deepcopy(self.fixtures[target])}

            if self.executor:
                value = await self.executor.execute(
                    task_id, operation_id, "observe", {"target": target}, fetch
                )
            else:
                value = await fetch()

            run_id = current_run_id()
            evidence_id = "ev-" + operation_id
            evidence = {
                "task_id": task_id,
                "intent_id": intent_id,
                "run_id": run_id,
                "call_id": call_id[0],
                **value,
            }
            if not self.repo.put_record("evidence", evidence_id, evidence, task_id):
                if self.repo.get_record("evidence", evidence_id, task_id) is None:
                    raise RuntimeError("evidence write failed")
            fact = self.repo.create_node(
                task_id,
                "fact",
                f"Observed {target}: {value['status']}",
                key="fact:" + operation_id,
                payload={
                    "intent_id": intent_id,
                    "run_id": run_id,
                    "call_id": call_id[0],
                    "evidence_id": evidence_id,
                    "observation": value,
                },
            )
            self.repo.add_edge(task_id, intent_id, fact["id"], "yields")
            if value.get("issue"):
                finding = self.repo.create_node(
                    task_id,
                    "finding",
                    "Fixture verified issue",
                    key="finding:" + operation_id,
                    payload={
                        "finding_record_id": "finding-" + operation_id,
                        "evidence_id": evidence_id,
                        "intent_id": intent_id,
                        "run_id": run_id,
                    },
                )
                self.repo.add_edge(task_id, fact["id"], finding["id"], "proves")
            # 只有证据、事实真正落库后才算业务成功；模型口头成功不算。
            observed.append(value)
            return value

        class TracedRegistry(ToolRegistry):
            async def execute(self, call, emit):
                call_id[0] = call.call_id
                return await super().execute(call, emit)

        registry = TracedRegistry()
        registry.register(ToolSpec("observe", "读取本地已授权目标的实验响应", ObserveInput, observe))
        try:
            async with AsyncExitStack() as stack:
                model = self.model_factory()
                if hasattr(model, "__aenter__"):
                    model = await stack.enter_async_context(model)
                result = await AgentRuntime(model, registry).run(
                    json.dumps({"target": target, "intent_id": intent_id, "task_id": task_id}),
                    events=trace,
                )
            for item in trace:
                self.repo.append_event(
                    task_id,
                    item.kind,
                    {"intent_id": intent_id, "run_id": item.run_id, **item.data},
                    call_id=item.call_id,
                )
            state = "done" if result.reason == "completed" and observed else "blocked"
            self.repo.transition_node(task_id, intent_id, "running", state)
            return True
        except asyncio.CancelledError:
            if self.repo.get_task(task_id)["status"] not in {"deleting", "deleted"}:
                self.repo.transition_node(task_id, intent_id, "running", "open")
            raise
        except Exception:
            if self.repo.get_task(task_id)["status"] not in {"deleting", "deleted"}:
                self.repo.transition_node(task_id, intent_id, "running", "blocked")
            raise
