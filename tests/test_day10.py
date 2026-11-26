import json
import unittest

from artex.context import (
    ContextBuilder,
    ContextOverflow,
    ContextualModel,
    DigestStore,
)
from artex.db import Repository
from runtime import (
    AgentRuntime,
    ModelReply,
    ToolCall,
    ToolInput,
    ToolRegistry,
    ToolSpec,
)


class ObserveInput(ToolInput):
    target: str


async def observe(arguments):
    return {
        "target": arguments.target,
        "status": 403,
        "body": "public sample is protected",
    }


class ScriptedObserveModel:
    async def complete(self, messages, tools):
        last = messages[-1]
        if last.role == "user":
            return ModelReply(
                tool_calls=(
                    ToolCall(
                        "context-1",
                        "observe",
                        {"target": "https://lab.example/public"},
                    ),
                )
            )
        if last.role == "tool" and last.result is not None:
            return ModelReply(
                text=json.dumps(last.result.content, ensure_ascii=False)
            )
        raise ValueError("unexpected message order")


def build_runtime(model):
    tools = ToolRegistry()
    tools.register(
        ToolSpec("observe", "读取固定样例目标", ObserveInput, observe)
    )
    return AgentRuntime(model, tools)


class ContextTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.repo = Repository("sqlite:///:memory:")
        self.repo.setup()
        self.task = self.repo.create_task(
            "context",
            "必须保留的目标",
            scope=["https://lab.example/public"],
        )["id"]

    async def asyncTearDown(self):
        self.repo.close()

    async def test_source_changes_during_generation_reject_digest(self):
        fact = self.repo.create_node(self.task, "fact", "old")

        async def summarize(nodes):
            self.repo.update_node(
                self.task, fact["id"], "new", {}, fact["version"]
            )
            return "stale summary"

        digest = await DigestStore(self.repo).generate(
            self.task, [fact["id"]], summarize
        )
        self.assertIsNone(digest)
        self.assertEqual(
            self.repo.get_node(self.task, fact["id"])["text"], "new"
        )

    async def test_stale_digest_not_read(self):
        fact = self.repo.create_node(self.task, "fact", "old")

        async def summarize(nodes):
            return "old summary"

        store = DigestStore(self.repo)
        digest = await store.generate(
            self.task, [fact["id"]], summarize
        )
        self.assertIsNotNone(digest)
        self.repo.update_node(
            self.task, fact["id"], "new", {}, fact["version"]
        )
        self.assertEqual(store.valid(self.task), [])

    async def test_active_intent_and_constraint_are_pinned(self):
        self.repo.create_node(self.task, "intent", "当前意图不可丢")
        self.repo.create_node(
            self.task,
            "hint",
            "禁止写操作",
            payload={"constraint": True},
        )
        for _ in range(30):
            self.repo.create_node(
                self.task, "fact", "历史结果" * 100
            )

        value = ContextBuilder(self.repo).build(
            self.task, max_chars=900
        )
        self.assertEqual(value["pinned"]["goal"], "必须保留的目标")
        self.assertIn("禁止写操作", value["pinned"]["constraints"])
        self.assertEqual(
            value["pinned"]["active_intents"][0]["text"],
            "当前意图不可丢",
        )
        self.assertGreater(value["omitted"], 0)

    async def test_pinned_overflow_fails_loudly(self):
        with self.assertRaises(ContextOverflow):
            ContextBuilder(self.repo).build(self.task, max_chars=10)

    async def test_active_members_cannot_be_summarized(self):
        intent = self.repo.create_node(
            self.task, "intent", "in progress"
        )

        async def summarize(nodes):
            return "must not be called"

        with self.assertRaises(ValueError):
            await DigestStore(self.repo).generate(
                self.task, [intent["id"]], summarize
            )

    async def test_context_wrapper_exposes_inner_usage(self):
        class UsageModel(ScriptedObserveModel):
            last_usage = 9

        model = ContextualModel(
            UsageModel(),
            ContextBuilder(self.repo),
            self.task,
        )
        await build_runtime(model).run("observe public")
        self.assertEqual(model.last_usage, 9)

    async def test_context_wrapper_preserves_tool_request_result_pair(self):
        model = ContextualModel(
            ScriptedObserveModel(),
            ContextBuilder(self.repo),
            self.task,
        )
        result = await build_runtime(model).run("observe public")

        self.assertEqual(result.reason, "completed")
        request = result.messages[1].tool_calls[0]
        response = result.messages[2].result
        self.assertEqual(request.call_id, response.call_id)
        self.assertEqual(response.content["status"], 403)
