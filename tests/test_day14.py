import unittest

import httpx

from artex.assets import AssetService, normalize
from artex.db import Repository


class AssetTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.repo = Repository("sqlite:///:memory:")
        self.repo.setup()
        self.task = self.repo.create_task(
            "d", "g", scope=["https://lab.example/public"]
        )
        self.id = self.task["id"]
        self.service = AssetService(self.repo)

    async def asyncTearDown(self):
        self.repo.close()

    async def test_normalization_idempotence_and_scope(self):
        first = self.service.upsert(
            self.id,
            "endpoint",
            "https://LAB.example:443/public",
        )
        second = self.service.upsert(
            self.id,
            "endpoint",
            "https://lab.example/public",
            "sync",
        )
        self.assertEqual(
            first["asset_id"], second["asset_id"]
        )
        self.assertEqual(
            second["sources"], ["manual", "sync"]
        )
        self.assertEqual(
            len(self.repo.list_records("asset", self.id)), 1
        )

        with self.assertRaises(ValueError):
            self.service.upsert(self.id, "domain", "example")
        with self.assertRaises(ValueError):
            self.service.upsert(
                self.id,
                "endpoint",
                "https://lab.example/private",
            )
        self.assertEqual(
            normalize("ip", "2001:0db8::1"), "2001:db8::1"
        )

    async def test_sync_and_probe_use_fixed_data(self):
        self.service.upsert(
            self.id, "endpoint", "https://lab.example/public"
        )
        seen = []

        def handler(request):
            seen.append(request.url.path)
            return httpx.Response(
                200, text="<title>sample</title>"
            )

        async with httpx.AsyncClient(
            transport=httpx.MockTransport(handler)
        ) as client:
            result = await self.service.enrich(
                self.id, client
            )

        self.assertEqual(result[0]["title"], "sample")
        self.assertEqual(seen, ["/public"])
        self.assertEqual(
            self.service.coverage(self.id)["tested"], 0
        )

    async def test_bad_row_is_rejected_without_aborting_batch(self):
        result = self.service.import_rows(
            self.id,
            [
                {"kind": "endpoint", "value": None},
                {
                    "kind": "endpoint",
                    "value": "https://lab.example/public",
                },
            ],
        )
        self.assertEqual(len(result["rejected"]), 1)
        self.assertEqual(len(result["imported"]), 1)

    async def test_coverage_requires_matching_fact_and_evidence(self):
        target = self.task["scope"][0]
        self.repo.put_record(
            "evidence", "e1", {"target": target}, self.id
        )
        fact = self.repo.create_node(
            self.id,
            "fact",
            "negative observation",
            payload={
                "observation": {
                    "target": target,
                    "issue": False,
                },
                "evidence_id": "e1",
            },
        )
        self.service.mark_tested(
            self.id, target, fact["id"]
        )
        self.assertEqual(
            self.service.coverage(self.id),
            {"total": 1, "tested": 1, "ratio": 1.0},
        )

        other = self.repo.create_task(
            "other", "g", scope=[target]
        )
        with self.assertRaises(KeyError):
            self.service.mark_tested(
                other["id"], target, fact["id"]
            )


class AssetApiTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        from artex.assets_api import create_app

        self.repo = Repository("sqlite:///:memory:")
        self.repo.setup()
        self.app = create_app(
            self.repo, api_token="asset-test-token"
        )
        self.life = self.app.router.lifespan_context(self.app)
        await self.life.__aenter__()
        self.client = httpx.AsyncClient(
            transport=httpx.ASGITransport(app=self.app),
            base_url="http://study",
            headers={
                "Authorization": "Bearer asset-test-token"
            },
        )

    async def asyncTearDown(self):
        await self.client.aclose()
        await self.life.__aexit__(None, None, None)
        self.repo.close()

    async def test_asset_and_coverage_routes(self):
        task = self.repo.create_task(
            "api", "g", scope=["https://lab.example/public"]
        )
        response = await self.client.post(
            f"/api/tasks/{task['id']}/assets",
            json={
                "kind": "endpoint",
                "value": "https://lab.example/public",
            },
        )
        self.assertEqual(response.status_code, 200)

        response = await self.client.get(
            f"/api/tasks/{task['id']}/assets"
        )
        self.assertEqual(len(response.json()["items"]), 1)

        response = await self.client.get(
            f"/api/tasks/{task['id']}/coverage"
        )
        self.assertEqual(
            response.json(),
            {"total": 1, "tested": 0, "ratio": 0.0},
        )
