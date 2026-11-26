import json
import unittest

from runtime import Message, ModelReply
from artex.db import Repository
from artex.human import HumanService, InterventionModel


class HumanTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.repo = Repository("sqlite:///:memory:")
        self.repo.setup()
        self.task = self.repo.create_task(
            "d", "g", scope=["https://lab.example/public"]
        )
        self.id = self.task["id"]
        self.service = HumanService(self.repo)

    async def asyncTearDown(self):
        self.repo.close()

    async def test_idempotent_hint_and_conflicting_request(self):
        first = self.service.add_hint(self.id, "a", "request")
        second = self.service.add_hint(self.id, "a", "request")
        self.assertEqual(first["id"], second["id"])
        with self.assertRaises(ValueError):
            self.service.add_hint(self.id, "b", "request")

    async def test_intent_scope_and_repeated_request(self):
        with self.assertRaises(ValueError):
            await self.service.inject_intent(
                self.id, "https://other.example/", "x", "1"
            )
        first = await self.service.inject_intent(
            self.id, self.task["scope"][0], "x", "2"
        )
        second = await self.service.inject_intent(
            self.id, self.task["scope"][0], "x", "2"
        )
        self.assertEqual(first["id"], second["id"])

    async def test_question_does_not_write_graph_or_events(self):
        before = (
            self.repo.list_nodes(self.id),
            self.repo.list_events(self.id),
        )

        class Answer:
            async def complete(inner, messages, tools):
                self.assertEqual(tools, [])
                return ModelReply(text="依据快照，任务尚未执行")

        answer = await self.service.ask(
            self.id, "状态？", Answer()
        )
        self.assertIn("尚未", answer["answer"])
        self.assertEqual(
            before,
            (
                self.repo.list_nodes(self.id),
                self.repo.list_events(self.id),
            ),
        )

    async def test_completed_task_reopens_and_delete_barrier_rejects(self):
        self.repo.transition_task(self.id, "created", "running")
        self.repo.transition_task(self.id, "running", "done")
        await self.service.inject_intent(
            self.id, self.task["scope"][0], "manual", "new-1"
        )
        self.assertEqual(
            self.repo.get_task(self.id)["status"], "running"
        )

        self.repo.transition_task(self.id, "running", "deleting")
        with self.assertRaises(ValueError):
            await self.service.inject_intent(
                self.id, self.task["scope"][0], "late", "new-2"
            )
        self.assertIsNone(
            self.repo.get_record(
                "human_request", "new-2", self.id
            )
        )

    async def test_steering_does_not_change_existing_last_message(self):
        intent = self.repo.create_node(
            self.id,
            "intent",
            "check",
            payload={"target": self.task["scope"][0]},
        )
        self.service.steer(
            self.id, intent["id"], "不要偏离当前目标", "steer-1"
        )

        class Observe:
            async def complete(inner, messages, tools):
                self.assertEqual(messages[0].role, "system")
                self.assertIn("不要偏离", messages[0].content)
                self.assertEqual(messages[-1].role, "user")
                return ModelReply(text="已收到")

        wrapped = InterventionModel(self.repo, Observe())
        messages = [
            Message(
                role="user",
                content=json.dumps(
                    {
                        "task_id": self.id,
                        "intent_id": intent["id"],
                        "target": self.task["scope"][0],
                    }
                ),
            )
        ]
        await wrapped.complete(messages, [])
        self.assertEqual(len(messages), 1)
        self.assertEqual(len(wrapped.delivered), 1)


class HumanApiTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        import httpx

        from artex.human_api import create_app

        self.repo = Repository("sqlite:///:memory:")
        self.repo.setup()
        self.app = create_app(
            self.repo, api_token="human-test-token"
        )
        self.life = self.app.router.lifespan_context(self.app)
        await self.life.__aenter__()
        self.client = httpx.AsyncClient(
            transport=httpx.ASGITransport(app=self.app),
            base_url="http://study",
            headers={"Authorization": "Bearer human-test-token"},
        )

    async def asyncTearDown(self):
        await self.client.aclose()
        await self.life.__aexit__(None, None, None)
        self.repo.close()

    async def test_hint_then_summary(self):
        task = self.repo.create_task(
            "api", "g", scope=["https://lab.example/public"]
        )
        response = await self.client.post(
            f"/api/tasks/{task['id']}/hints",
            json={"request_id": "api-hint", "text": "关注证据"},
        )
        self.assertEqual(response.status_code, 200)

        response = await self.client.get(
            f"/api/tasks/{task['id']}/summary"
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json()["counts"]["hint"], 1
        )
