import asyncio
import json
from pathlib import Path
import tempfile
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor

import httpx
from openai import AsyncOpenAI

from artex.db import Repository
from artex.model_policy import (
    Budget,
    BudgetExceeded,
    DeepSeekUsageModel,
    GovernedModel,
    ProviderFailure,
)
from runtime import Message, ModelReply


class PolicyTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.repo = Repository("sqlite:///:memory:")
        self.repo.setup()
        self.task = self.repo.create_task("budget", "test")["id"]
        self.repo.transition_task(self.task, "created", "running")

    async def asyncTearDown(self):
        self.repo.close()

    async def test_reservations_share_remaining_budget(self):
        budget = Budget(self.repo, self.task, 100)

        async def attempt(name):
            try:
                budget.reserve(name, 80)
                return True
            except BudgetExceeded:
                return False

        self.assertEqual(
            sum(await asyncio.gather(attempt("a"), attempt("b"))), 1
        )
        self.assertEqual(sum(budget.snapshot()["holds"].values()), 80)

    async def test_independent_connections_cannot_spend_same_balance(self):
        work = Path(__file__).resolve().parents[1] / ".work"
        work.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=work) as folder:
            url = "sqlite:///" + (Path(folder) / "budget.db").as_posix()
            main = Repository(url)
            main.setup()
            task = main.create_task("two connections", "budget")["id"]
            main.transition_task(task, "created", "running")
            budget = Budget(main, task, 100)
            ready = threading.Barrier(2)

            def attempt(name):
                other = Repository(url)
                try:
                    local = Budget(other, task, 100)
                    ready.wait(timeout=5)
                    try:
                        local.reserve(name, 80)
                        return True
                    except BudgetExceeded:
                        return False
                finally:
                    other.close()

            try:
                with ThreadPoolExecutor(2) as pool:
                    outcomes = list(pool.map(attempt, ["first", "second"]))
                self.assertEqual(sum(outcomes), 1)
                self.assertEqual(sum(budget.snapshot()["holds"].values()), 80)
            finally:
                main.close()

    async def test_success_without_usage_retains_uncertain_reservation(self):
        class Fixture:
            async def complete(self, messages, tools):
                return ModelReply(text="answer without usage")

        budget = Budget(self.repo, self.task, 100)
        reply = await GovernedModel(
            Fixture(), budget, reservation=40
        ).complete([], [])
        self.assertEqual(reply.text, "answer without usage")
        state = budget.snapshot()
        self.assertEqual(state["spent"], 0)
        self.assertEqual(sum(state["holds"].values()), 40)
        self.assertEqual(len(state["uncertain"]), 1)

    async def test_retry_rate_limit_then_settle_actual_usage(self):
        class Fixture:
            calls, last_usage = 0, 7

            async def complete(self, messages, tools):
                self.calls += 1
                if self.calls == 1:
                    raise ProviderFailure(429)
                return ModelReply(text="ok")

        inner = Fixture()
        budget = Budget(self.repo, self.task, 100)
        reply = await GovernedModel(
            inner, budget, reservation=40, delay=0
        ).complete([], [])
        self.assertEqual(reply.text, "ok")
        self.assertEqual(inner.calls, 2)
        self.assertEqual(budget.snapshot()["spent"], 7)
        self.assertFalse(budget.snapshot()["holds"])

    async def test_authentication_failure_does_not_retry(self):
        class Fixture:
            calls = 0

            async def complete(self, messages, tools):
                self.calls += 1
                raise ProviderFailure(401)

        inner = Fixture()
        budget = Budget(self.repo, self.task, 100)
        with self.assertRaises(ProviderFailure):
            await GovernedModel(inner, budget, reservation=40).complete([], [])
        self.assertEqual(inner.calls, 1)
        self.assertFalse(budget.snapshot()["holds"])

    async def test_cancellation_preserves_uncertain_usage(self):
        class Fixture:
            async def complete(self, messages, tools):
                raise asyncio.CancelledError()

        budget = Budget(self.repo, self.task, 100)
        with self.assertRaises(asyncio.CancelledError):
            await GovernedModel(
                Fixture(), budget, reservation=40
            ).complete([], [])
        self.assertEqual(sum(budget.snapshot()["holds"].values()), 40)
        self.assertEqual(len(budget.snapshot()["uncertain"]), 1)

    async def test_truncated_or_non_object_arguments_rejected(self):
        for raw in ('{"a":', "[1,2]"):
            with self.assertRaises(ValueError):
                DeepSeekUsageModel.parse(raw)

    async def test_three_transient_failures_trip_circuit(self):
        class Fixture:
            async def complete(self, messages, tools):
                raise ProviderFailure(503)

        budget = Budget(self.repo, self.task, 1000)
        model = GovernedModel(
            Fixture(), budget, reservation=10, delay=0
        )
        with self.assertRaises(ProviderFailure):
            await model.complete([], [])
        with self.assertRaisesRegex(RuntimeError, "熔断"):
            await model.complete([], [])

    async def test_stream_fragments_form_one_valid_tool_call(self):
        def frame(delta, finish=None, usage=None):
            body = {
                "id": "fixture",
                "object": "chat.completion.chunk",
                "created": 0,
                "model": "fixture",
                "choices": [
                    {"index": 0, "delta": delta, "finish_reason": finish}
                ],
            }
            if usage:
                body["usage"] = usage
            return "data: " + json.dumps(body) + "\n\n"

        content = frame(
            {
                "tool_calls": [
                    {
                        "index": 0,
                        "id": "stream-1",
                        "type": "function",
                        "function": {
                            "name": "observe",
                            "arguments": '{"target":"',
                        },
                    }
                ]
            }
        )
        content += frame(
            {
                "tool_calls": [
                    {
                        "index": 0,
                        "function": {
                            "arguments": 'https://lab.example/public"}'
                        },
                    }
                ]
            }
        )
        content += frame(
            {},
            "tool_calls",
            {
                "prompt_tokens": 10,
                "completion_tokens": 4,
                "total_tokens": 14,
            },
        )
        content += "data: [DONE]\n\n"

        class Body(httpx.AsyncByteStream):
            async def __aiter__(self):
                yield content.encode()

        def handler(request):
            return httpx.Response(
                200,
                headers={"content-type": "text/event-stream"},
                stream=Body(),
            )

        inner = object.__new__(DeepSeekUsageModel)
        inner.model, inner.stream, inner.last_usage = "fixture", True, None
        inner.client = AsyncOpenAI(
            api_key="offline-test",
            base_url="https://fixture.invalid",
            max_retries=0,
            http_client=httpx.AsyncClient(
                transport=httpx.MockTransport(handler)
            ),
        )
        async with inner:
            reply = await inner.complete(
                [Message(role="user", content="observe public")], []
            )

        self.assertEqual(
            reply.tool_calls[0].arguments,
            {"target": "https://lab.example/public"},
        )
        self.assertEqual(reply.tool_calls[0].call_id, "stream-1")
        self.assertEqual(inner.last_usage, 14)
