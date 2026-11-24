import unittest

from runtime import Message, ModelReply
from artex.db import Repository
from artex.worker import (
    DEFAULT_FIXTURES,
    WORKER_SYSTEM_PROMPT,
    Worker,
    WorkerDeepSeekModel,
)


class WorkerTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.repo = Repository("sqlite:///:memory:")
        self.repo.setup()
        self.task = self.repo.create_task("fixture", "observe", list(DEFAULT_FIXTURES))
        self.tid = self.task["id"]
        self.repo.transition_task(self.tid, "created", "running")

    def tearDown(self):
        self.repo.close()

    async def test_real_tool_result_and_provenance(self):
        target = "https://lab.example/private"
        intent = self.repo.create_node(self.tid, "intent", "observe", payload={"target": target})
        self.assertTrue(await Worker(self.repo).execute(self.tid, intent["id"]))
        self.assertFalse(await Worker(self.repo).execute(self.tid, intent["id"]))
        self.assertEqual(self.repo.get_node(self.tid, intent["id"])["state"], "done")
        facts = self.repo.list_nodes(self.tid, "fact")
        self.assertEqual(len(facts), 1)
        self.assertEqual(facts[0]["payload"]["call_id"], "observe-1")
        evidence = self.repo.get_record("evidence", facts[0]["payload"]["evidence_id"], self.tid)
        self.assertEqual(evidence["value"]["target"], target)
        self.assertEqual(len(self.repo.list_nodes(self.tid, "finding")), 1)
        events = self.repo.list_events(self.tid)
        finished = [event for event in events if event["kind"] == "tool_finished"]
        self.assertEqual(len(finished), 1)
        self.assertEqual(finished[0]["call_id"], "observe-1")
        self.assertEqual(finished[0]["data"]["intent_id"], intent["id"])
        self.assertIn("run_id", finished[0]["data"])
        self.assertEqual(evidence["value"]["run_id"], finished[0]["data"]["run_id"])
        self.assertTrue(any(event["kind"] == "run_completed" for event in events))

    async def test_model_claim_without_tool_is_not_evidence(self):
        class Liar:
            async def complete(self, messages, tools):
                return ModelReply(text="任务成功，发现漏洞")

        intent = self.repo.create_node(
            self.tid, "intent", "observe",
            payload={"target": next(iter(DEFAULT_FIXTURES))},
        )
        self.assertTrue(await Worker(self.repo, model_factory=Liar).execute(self.tid, intent["id"]))
        self.assertEqual(self.repo.list_nodes(self.tid, "finding"), [])
        self.assertEqual(self.repo.get_node(self.tid, intent["id"])["state"], "blocked")

    async def test_out_of_scope_has_no_fact(self):
        intent = self.repo.create_node(
            self.tid, "intent", "outside",
            payload={"target": "https://other.example/"},
        )
        self.assertTrue(await Worker(self.repo).execute(self.tid, intent["id"]))
        self.assertEqual(self.repo.list_nodes(self.tid, "fact"), [])
        self.assertEqual(self.repo.get_node(self.tid, intent["id"])["state"], "blocked")

    async def test_worker_deepseek_uses_observe_prompt_only(self):
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

        model = WorkerDeepSeekModel(FakeInner)
        self.assertEqual(model.inner.system_prompt, WORKER_SYSTEM_PROMPT)
        messages = [Message(role="user", content="{}")]
        reply = await model.complete(messages, [])
        self.assertEqual(reply.text, "ok")
        self.assertEqual(model.inner.messages, messages)
