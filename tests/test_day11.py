import asyncio
import unittest

from artex.db import Repository
from artex.events import EventService, merge_events
from artex.evaluation import evaluate


class EventTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.repo = Repository("sqlite:///:memory:")
        self.repo.setup()
        self.task = self.repo.create_task("events", "test")["id"]

    async def asyncTearDown(self):
        self.repo.close()

    async def test_slow_subscriber_reads_all_300(self):
        expected = [
            self.repo.append_event(
                self.task, "step", {"n": index}
            )
            for index in range(300)
        ]
        stream = EventService(self.repo).tail(self.task)
        received = []
        try:
            for index in range(300):
                received.append(await anext(stream))
                if index == 50:
                    await asyncio.sleep(0.01)
        finally:
            await stream.aclose()
        self.assertEqual(
            [event["id"] for event in expected],
            [event["id"] for event in received],
        )

    async def test_resume_gapful_cursor_and_deduplicate(self):
        other = self.repo.create_task("other", "test")["id"]
        first = self.repo.append_event(self.task, "one", {})
        self.repo.append_event(other, "ignored", {})
        second = self.repo.append_event(self.task, "two", {})
        stream = EventService(self.repo).tail(
            self.task, first["id"]
        )
        try:
            received = await anext(stream)
        finally:
            await stream.aclose()
        self.assertEqual(received["id"], second["id"])
        self.assertEqual(
            len(merge_events([first, second], [second])), 2
        )

    async def test_sse_encodes_untrusted_text_as_json(self):
        event = self.repo.append_event(
            self.task, "step", {"text": "x\nid: 999"}
        )
        frame = EventService.format_sse(event)
        self.assertEqual(frame.count("\nid:"), 0)
        self.assertIn("event: activity", frame)

    async def test_fixed_evaluation_suite(self):
        report = await evaluate()
        self.assertEqual(report["count"], 20)
        self.assertEqual(report["passed"], 20)
        self.assertEqual(report["evidence_valid_rate"], 1.0)
        self.assertEqual(report["live_api_calls"], 0)
