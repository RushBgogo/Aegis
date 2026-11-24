import json
import unittest

from runtime import Message, ModelReply, ToolCall
from artex.db import Repository
from artex.llm_planner import PLANNER_SYSTEM_PROMPT, ModelPlanner, PlanningDeepSeekModel
from artex.planner import Planner, sample_task
from artex.worker import Worker


class PlannerTests(unittest.IsolatedAsyncioTestCase):
    async def test_model_proposal_goals_finish_only_after_evidence(self):
        class GoodPlanner:
            async def complete(self, messages, tools):
                if messages[-1].role == "tool":
                    return ModelReply(text="计划已登记")
                scope = json.loads(messages[-1].content)["scope"]
                return ModelReply(tool_calls=(ToolCall("plan-1", "submit_plan", {
                    "goals": [{"text": "评估 " + target, "targets": [target]} for target in scope],
                    "ordered_targets": scope}),))

        repo = Repository("sqlite:///:memory:")
        repo.setup()
        self.addCleanup(repo.close)
        task = sample_task(repo)
        planner = ModelPlanner(repo, GoodPlanner)
        await planner.plan(task["id"])
        self.assertEqual(len(repo.list_nodes(task["id"], "goal")), 2)
        self.assertEqual(repo.get_task(task["id"])["status"], "running")
        for intent in repo.list_nodes(task["id"], "intent", "open"):
            await Worker(repo).execute(task["id"], intent["id"])
        await planner.plan(task["id"])
        self.assertEqual(repo.get_task(task["id"])["status"], "done")
        self.assertTrue(all(goal["state"] == "met" for goal in repo.list_nodes(task["id"], "goal")))

    async def test_model_planner_cannot_expand_scope(self):
        class BadPlanner:
            async def complete(self, messages, tools):
                if messages[-1].role == "tool":
                    return ModelReply(text="建议结束")
                return ModelReply(tool_calls=(ToolCall("plan-1", "submit_plan", {
                    "goals": [{"text": "越界目标", "targets": ["https://outside.example"]}],
                    "ordered_targets": ["https://outside.example"]}),))

        repo = Repository("sqlite:///:memory:")
        repo.setup()
        self.addCleanup(repo.close)
        task = sample_task(repo)
        await ModelPlanner(repo, BadPlanner).plan(task["id"])
        self.assertEqual(
            {node["payload"]["target"] for node in repo.list_nodes(task["id"], "intent")},
            set(task["scope"]),
        )
        self.assertTrue(any(e["kind"] == "planner_fallback" for e in repo.list_events(task["id"])))

    async def test_dedup_and_evidence_backed_completion(self):
        repo = Repository("sqlite:///:memory:")
        repo.setup()
        self.addCleanup(repo.close)
        task = sample_task(repo)
        tid = task["id"]
        planner = Planner(repo)
        self.assertEqual(await planner.plan(tid), 2)
        self.assertEqual(await planner.plan(tid), 0)
        self.assertEqual(repo.get_task(tid)["status"], "running")
        for intent in repo.list_nodes(tid, "intent", "open"):
            await Worker(repo).execute(tid, intent["id"])
        await planner.plan(tid)
        self.assertEqual(repo.get_task(tid)["status"], "done")
        self.assertEqual(repo.list_nodes(tid, "goal")[0]["state"], "met")

    async def test_negative_result_can_complete_assessment(self):
        repo = Repository("sqlite:///:memory:")
        repo.setup()
        self.addCleanup(repo.close)
        task = repo.create_task("negative", "评估是否有问题", ["https://lab.example/public"])
        repo.transition_task(task["id"], "created", "running")
        planner = Planner(repo)
        await planner.plan(task["id"])
        await Worker(repo).execute(task["id"], repo.list_nodes(task["id"], "intent")[0]["id"])
        await planner.plan(task["id"])
        self.assertEqual(repo.get_task(task["id"])["status"], "done")
        self.assertEqual(repo.list_nodes(task["id"], "finding"), [])

    async def test_planning_deepseek_uses_planner_prompt_only(self):
        class FakeInner:
            def __init__(self, system_prompt=None):
                self.system_prompt = system_prompt
                self.messages = None

            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                pass

            async def complete(self, messages, tools):
                self.messages = messages
                return ModelReply(text="ok")

        model = PlanningDeepSeekModel(FakeInner)
        self.assertEqual(model.inner.system_prompt, PLANNER_SYSTEM_PROMPT)
        messages = [Message(role="user", content="{}")]
        reply = await model.complete(messages, [])
        self.assertEqual(reply.text, "ok")
        self.assertEqual(model.inner.messages, messages)
