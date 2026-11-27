import hashlib
import unittest

import httpx

from artex.db import Repository
from artex.engine import Engine
from artex.reporting import ReportService
from artex.traffic import RecordingClient


class EvidenceTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.repo = Repository("sqlite:///:memory:")
        self.repo.setup()
        self.target = "https://lab.example/private"
        self.task = self.repo.create_task(
            "d", "g", scope=[self.target]
        )
        self.id = self.task["id"]
        self.intent = self.repo.create_node(
            self.id,
            "intent",
            "test",
            payload={"target": self.target},
        )
        self.finding = self.repo.create_node(
            self.id,
            "finding",
            "fixed finding",
            payload={
                "target": self.target,
                "intent_id": self.intent["id"],
                "finding_record_id": "record-1",
            },
        )
        self.service = ReportService(self.repo)

    async def asyncTearDown(self):
        self.repo.close()

    async def capture(self, body="fixture evidence"):
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(
                lambda request: httpx.Response(
                    200, text=body
                )
            ),
            headers={"Authorization": "Bearer offline-token"},
        ) as client:
            return await RecordingClient(
                self.repo, self.id, client
            ).get(
                self.target,
                intent_id=self.intent["id"],
                call_id="http-1",
            )

    async def test_binding_report_and_stale_version(self):
        evidence = await self.capture()
        self.assertNotIn(
            "authorization", evidence["request_headers"]
        )
        self.service.bind(
            self.id,
            self.finding["id"],
            evidence["evidence_id"],
            "固定正文支持结论",
        )
        report = self.service.build(self.id)
        self.assertFalse(self.service.is_stale(self.id, report))
        self.assertIn(
            evidence["evidence_id"], report["markdown"]
        )

        record = self.repo.get_record(
            "evidence", evidence["evidence_id"], self.id
        )
        value = dict(record["value"])
        value["body"] = "new evidence"
        value["body_sha256"] = hashlib.sha256(
            value["body"].encode()
        ).hexdigest()
        self.repo.put_record(
            "evidence",
            evidence["evidence_id"],
            value,
            self.id,
            expected_version=record["version"],
        )
        self.assertTrue(self.service.is_stale(self.id, report))
        with self.assertRaises(ValueError):
            self.service.build(self.id)

        self.service.bind(
            self.id,
            self.finding["id"],
            evidence["evidence_id"],
            "新正文重新核对",
        )
        self.assertFalse(
            self.service.is_stale(
                self.id, self.service.build(self.id)
            )
        )

    async def test_missing_or_other_task_evidence_is_rejected(self):
        with self.assertRaises(ValueError):
            self.service.build(self.id)

        other = self.repo.create_task(
            "other", "g", scope=[self.target]
        )
        self.repo.put_record(
            "evidence",
            "foreign",
            {
                "task_id": other["id"],
                "target": self.target,
                "body": "x",
                "status": 200,
            },
            other["id"],
        )
        with self.assertRaises(ValueError):
            self.service.bind(
                self.id,
                self.finding["id"],
                "foreign",
                "not allowed",
            )

    async def test_canonical_scope_and_matching_intent_are_used(self):
        task = self.repo.create_task(
            "canonical",
            "g",
            scope=["https://LAB.example:443/private"],
        )
        intent = self.repo.create_node(
            task["id"],
            "intent",
            "same",
            payload={"target": "https://LAB.example:443/private"},
        )
        calls = []

        def handler(request):
            calls.append(str(request.url))
            return httpx.Response(
                200, text="canonical evidence"
            )

        async with httpx.AsyncClient(
            transport=httpx.MockTransport(handler)
        ) as client:
            evidence = await RecordingClient(
                self.repo, task["id"], client
            ).get(
                "https://lab.example/private",
                intent_id=intent["id"],
            )

        finding = self.repo.create_node(
            task["id"],
            "finding",
            "test",
            payload={
                "intent_id": intent["id"],
                "evidence_id": evidence["evidence_id"],
            },
        )
        report = ReportService(self.repo).build(task["id"])
        self.assertIn("canonical evidence", report["markdown"])

        other_intent = self.repo.create_node(
            self.id,
            "intent",
            "other",
            payload={"target": "https://lab.example/public"},
        )
        before = len(calls)
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(handler)
        ) as client:
            with self.assertRaises(ValueError):
                await RecordingClient(
                    self.repo, self.id, client
                ).get(
                    self.target,
                    intent_id=other_intent["id"],
                )
        self.assertEqual(len(calls), before)

    async def test_scope_rejects_before_network_and_body_limit(self):
        count = 0

        def handler(request):
            nonlocal count
            count += 1
            return httpx.Response(200, text="large")

        async with httpx.AsyncClient(
            transport=httpx.MockTransport(handler)
        ) as client:
            recorder = RecordingClient(
                self.repo, self.id, client, max_body=2
            )
            with self.assertRaises(ValueError):
                await recorder.get("https://other.example/")
            self.assertEqual(count, 0)
            with self.assertRaises(ValueError):
                await recorder.get(self.target)
        self.assertEqual(
            self.repo.list_records("evidence", self.id), []
        )

    async def test_retest_preserves_original_finding(self):
        before = self.repo.get_node(
            self.id, self.finding["id"]
        )

        async def probe(target):
            return {
                "status": 403,
                "body": "denied",
                "issue": False,
            }

        result = await self.service.retest(
            self.id, self.finding["id"], probe
        )
        self.assertEqual(result["status"], "not_reproduced")
        self.assertEqual(
            before,
            self.repo.get_node(self.id, self.finding["id"]),
        )
        self.assertEqual(
            len(self.repo.list_records("retest", self.id)), 1
        )

    async def test_original_worker_evidence_can_build_and_retest(self):
        await Engine(self.repo).run(self.id)
        worker_finding = next(
            node
            for node in self.repo.list_nodes(
                self.id, "finding"
            )
            if node["payload"].get("evidence_id")
        )
        self.service.bind(
            self.id,
            self.finding["id"],
            worker_finding["payload"]["evidence_id"],
            "固定样例已核实",
        )
        report = self.service.build(self.id)
        self.assertFalse(self.service.is_stale(self.id, report))

        async def probe(target):
            return {
                "status": 403,
                "body": "denied",
                "issue": False,
            }

        result = await self.service.retest(
            self.id, worker_finding["id"], probe
        )
        self.assertEqual(result["status"], "not_reproduced")
