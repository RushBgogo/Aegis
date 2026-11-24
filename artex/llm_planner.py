"""用第一课 Runtime 获取模型建议，代码继续负责范围、去重与完成校验。"""
from contextlib import AsyncExitStack
import hashlib
import json

from runtime import AgentRuntime, ToolInput, ToolRegistry, ToolSpec
from artex.planner import Planner

PLANNER_SYSTEM_PROMPT = (
    "你是评估任务规划者。仅针对用户 JSON 中 scope 的完整目标，"
    "通过 submit_plan 提交一个或多个独立的评估交付目标，并按价值排序目标。"
    "goals 每项 text 描述需要核验的结果，targets 必须属于 scope；"
    "ordered_targets 必须恰好覆盖 scope。不要断言发现漏洞，不要执行探测。"
)


class GoalDraft(ToolInput):
    text: str
    targets: list[str]


class PlanDraft(ToolInput):
    goals: list[GoalDraft]
    ordered_targets: list[str]


class PlanningDeepSeekModel:
    def __init__(self, model_factory=None):
        if model_factory is None:
            from deepseek_model import DeepSeekModel
            model_factory = DeepSeekModel
        self.inner = model_factory(system_prompt=PLANNER_SYSTEM_PROMPT)

    async def __aenter__(self):
        await self.inner.__aenter__()
        return self

    async def __aexit__(self, *args):
        await self.inner.__aexit__(*args)

    async def complete(self, messages, tools):
        return await self.inner.complete(messages, tools)


class ModelPlanner(Planner):
    def __init__(self, repo, model_factory=PlanningDeepSeekModel):
        super().__init__(repo)
        self.model_factory = model_factory

    async def plan(self, task_id):
        task = self.repo.get_task(task_id)
        hints = [
            {"text": hint["text"], "version": hint["version"]}
            for hint in self.repo.list_nodes(task_id, "hint")
        ]
        signature = hashlib.sha256(
            json.dumps([task["goal"], task["scope"], hints], sort_keys=True).encode()
        ).hexdigest()
        previous = self.repo.get_record("planner_policy", "input", task_id)
        needs_plan = previous is None or previous["value"]["signature"] != signature
        if task["status"] == "running" and needs_plan:
            scope = set(task["scope"])
            accepted = False

            async def submit(draft):
                nonlocal accepted
                if not draft.goals or not scope:
                    raise ValueError("nonempty goals and explicit scope required")
                if (
                    set(draft.ordered_targets) != scope
                    or len(draft.ordered_targets) != len(scope)
                ):
                    raise ValueError("ordered targets must cover exactly the allowed scope")
                covered = set()
                for goal in draft.goals:
                    if (
                        not goal.text.strip()
                        or not goal.targets
                        or not set(goal.targets) <= scope
                    ):
                        raise ValueError("goal must have text and allowed explicit targets")
                    covered.update(goal.targets)
                if covered != scope:
                    raise ValueError("goal targets must cover scope")
                goal_ids = []
                for goal in draft.goals:
                    key = "goal:" + hashlib.sha256(
                        json.dumps([goal.text, sorted(goal.targets)]).encode()
                    ).hexdigest()
                    node = self.repo.create_node(
                        task_id,
                        "goal",
                        goal.text,
                        key=key,
                        payload={"targets": goal.targets, "source": "model_proposal"},
                    )
                    goal_ids.append((node["id"], goal.targets))
                for priority, target in enumerate(reversed(draft.ordered_targets), 1):
                    key = "intent:" + hashlib.sha256(target.encode()).hexdigest()
                    goal_id = next(
                        (gid for gid, targets in goal_ids if target in targets), None
                    )
                    payload = {"target": target}
                    if goal_id is not None:
                        payload["goal_id"] = goal_id
                    self.repo.create_node(
                        task_id,
                        "intent",
                        "Observe " + target,
                        key=key,
                        priority=priority,
                        payload=payload,
                    )
                accepted = True
                return {"accepted": True}

            tools = ToolRegistry()
            tools.register(
                ToolSpec(
                    "submit_plan",
                    "提交有明确范围的评估目标与执行顺序",
                    PlanDraft,
                    submit,
                )
            )
            async with AsyncExitStack() as stack:
                model = self.model_factory()
                if hasattr(model, "__aenter__"):
                    model = await stack.enter_async_context(model)
                result = await AgentRuntime(model, tools).run(
                    json.dumps(
                        {
                            "goal": task["goal"],
                            "scope": task["scope"],
                            "hints": hints,
                            "existing_goals": [
                                goal["text"]
                                for goal in self.repo.list_nodes(task_id, "goal")
                            ],
                            "facts": [
                                fact["text"]
                                for fact in self.repo.list_nodes(task_id, "fact")
                            ],
                        },
                        ensure_ascii=False,
                    ),
                    max_turns=3,
                )
            self.repo.put_record(
                "planner_policy",
                "input",
                {"signature": signature},
                task_id,
                previous["version"] if previous else 0,
            )
            self.repo.append_event(
                task_id,
                "model_plan",
                {
                    "reason": result.reason,
                    "accepted": accepted,
                },
            )
            if not self.repo.list_nodes(task_id, "goal"):
                self.repo.append_event(
                    task_id, "planner_fallback", {"reason": "no valid proposal"}
                )
        return await super().plan(task_id)
