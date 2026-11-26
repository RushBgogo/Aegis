import asyncio
import unittest

import httpx

from artex.db import Repository
from artex.web import create_app


class WebTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.repo = Repository("sqlite:///:memory:")
        self.repo.setup()
        self.app = create_app(self.repo, api_token="test-token")
        self.life = self.app.router.lifespan_context(self.app)
        await self.life.__aenter__()
        self.client = httpx.AsyncClient(
            transport=httpx.ASGITransport(app=self.app),
            base_url="http://study",
        )

    async def asyncTearDown(self):
        await self.client.aclose()
        await self.life.__aexit__(None, None, None)
        self.repo.close()

    async def test_auth_and_strict_input(self):
        response = await self.client.get("/api/tasks")
        self.assertEqual(response.status_code, 401)

        self.client.headers["Authorization"] = "Bearer test-token"
        response = await self.client.post(
            "/api/tasks",
            json={
                "description": "d",
                "goal": "g",
                "scope": ["x"],
                "api_key": "wrong",
            },
        )
        self.assertEqual(response.status_code, 422)

    async def test_create_run_and_read_without_mutation(self):
        self.client.headers["Authorization"] = "Bearer test-token"
        response = await self.client.post(
            "/api/tasks",
            json={
                "description": "d",
                "goal": "g",
                "scope": ["https://lab.example/public"],
            },
        )
        self.assertEqual(response.status_code, 201)
        task_id = response.json()["id"]

        response = await self.client.post(
            f"/api/tasks/{task_id}/run", json={}
        )
        self.assertEqual(response.status_code, 202)
        await asyncio.gather(*self.app.state.jobs.values())
        self.assertEqual(
            self.repo.get_task(task_id)["status"], "done"
        )

        before = self.repo.list_events(task_id)
        response = await self.client.get(
            f"/api/tasks/{task_id}/graph"
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(self.repo.list_events(task_id), before)

        response = await self.client.get(
            f"/api/tasks/{task_id}/events?limit=1"
        )
        self.assertEqual(len(response.json()["items"]), 1)
        self.assertNotIn("test-token", response.text)

    async def test_wrong_task_does_not_leak_and_cursor_validation(self):
        self.client.headers["Authorization"] = "Bearer test-token"
        response = await self.client.get(
            "/api/tasks/missing/graph"
        )
        self.assertEqual(response.status_code, 404)

        response = await self.client.get(
            "/api/tasks/missing/events?since=-1"
        )
        self.assertEqual(response.status_code, 422)

        schema = (await self.client.get("/openapi.json")).json()
        self.assertIn(
            "/api/tasks/{task_id}/stream", schema["paths"]
        )

    async def test_model_mode_is_persisted_and_restored(self):
        from artex.worker import Worker

        seen = []

        def workers(repo, mode):
            seen.append(mode)
            # 测试只验证模式路由，不发送真实 API。
            return Worker(repo)

        task = self.repo.create_task(
            "mode", "g", scope=["https://lab.example/public"]
        )
        first = create_app(
            self.repo,
            api_token="test-token",
            worker_factory=workers,
        )
        first.state.get_engine(task["id"])
        first.state.get_engine(task["id"], "deepseek")

        second = create_app(
            self.repo,
            api_token="test-token",
            worker_factory=workers,
        )
        second.state.get_engine(task["id"])
        self.assertEqual(
            seen, ["fixture", "deepseek", "deepseek"]
        )
