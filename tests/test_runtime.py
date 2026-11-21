import asyncio
import unittest

from demo import AddInput, ScriptedModel, build_runtime
from runtime import ModelReply, ToolCall, ToolRegistry, ToolSpec


class Replies:
    """用于测试，依次返回指定的模型响应。"""

    def __init__(self, *replies):
        self.replies = iter(replies)
        self.calls = 0

    async def complete(self, messages, tools):
        self.calls += 1
        return next(self.replies)


class RuntimeTests(unittest.IsolatedAsyncioTestCase):
    async def test_success_and_replay(self):
        executions = 0

        async def counted(arguments):
            nonlocal executions
            executions += 1
            return arguments.a + arguments.b

        first = await build_runtime(
            handler=counted
        ).run("计算 7 与 5 的和")

        second = await build_runtime().run("计算 7 与 5 的和")

        self.assertEqual(
            (first.text, first.reason, first.turns),
            ("12", "completed", 2),
        )
        self.assertEqual(executions, 1)

        self.assertEqual(
            [m.role for m in first.messages],
            ["user", "assistant", "tool", "assistant"],
        )

        self.assertEqual(
            first.messages[1].tool_calls[0].call_id,
            first.messages[2].result.call_id,
        )

        def normalize(result):
            return [
                (e.seq, e.kind, e.call_id, e.data)
                for e in result.events
            ]

        # run_id 可以变化，其余行为应该一致。
        self.assertEqual(normalize(first), normalize(second))

    async def test_direct_answer(self):
        result = await build_runtime(
            Replies(ModelReply(text="你好"))
        ).run("你好")

        self.assertEqual((result.text, result.turns), ("你好", 1))

        self.assertFalse(
            any(e.kind == "tool_started" for e in result.events)
        )

    async def test_invalid_calls_never_execute_handler(self):
        async def forbidden(arguments):
            self.fail("非法调用不应执行 handler")

        cases = [
            (
                ToolCall("x", "missing", {}),
                "unknown_tool",
            ),
            (
                ToolCall("x", "add", {"b": 5}),
                "invalid_arguments",
            ),
            (
                ToolCall("x", "add", {"a": "7", "b": 5}),
                "invalid_arguments",
            ),
            (
                ToolCall("x", "add", {"a": True, "b": 5}),
                "invalid_arguments",
            ),
            (
                ToolCall("x", "add", {"a": 7, "b": 5, "extra": 1}),
                "invalid_arguments",
            ),
        ]

        for call, code in cases:
            with self.subTest(call=call):
                model = Replies(
                    ModelReply(tool_calls=(call,)),
                    ModelReply(text="收到错误"),
                )

                result = await build_runtime(
                    model, forbidden
                ).run("test")

                error = result.messages[2].result

                self.assertTrue(error.is_error)
                self.assertEqual(
                    (error.call_id, error.content["code"]),
                    ("x", code),
                )

    async def test_tool_exception_becomes_result(self):
        async def broken(arguments):
            raise RuntimeError("boom")

        result = await build_runtime(
            ScriptedModel(), broken
        ).run("test")

        self.assertEqual(result.reason, "completed")
        self.assertEqual(
            result.messages[2].result.content["code"],
            "tool_error",
        )

    async def test_max_turns(self):
        model = Replies(
            *[
                ModelReply(
                    tool_calls=(
                        ToolCall(
                            str(i), "add", {"a": 1, "b": 2}
                        ),
                    )
                )
                for i in range(3)
            ]
        )

        result = await build_runtime(model).run(
            "test", max_turns=2
        )

        self.assertEqual(
            (result.reason, result.turns, model.calls),
            ("max_turns", 2, 2),
        )

    async def test_model_and_protocol_errors(self):
        cases = [
            (Replies(), "model_error"),
            (Replies(ModelReply()), "protocol_error"),
            (
                Replies(
                    ModelReply(
                        text="bad",
                        tool_calls=(ToolCall("x", "add", {}),),
                    )
                ),
                "protocol_error",
            ),
        ]

        for model, reason in cases:
            result = await build_runtime(model).run("test")
            self.assertEqual(result.reason, reason)

    async def test_duplicate_call_id(self):
        reply = ModelReply(
            tool_calls=(
                ToolCall("same", "add", {"a": 1, "b": 2}),
            )
        )

        result = await build_runtime(
            Replies(reply, reply)
        ).run("test")

        self.assertEqual(result.reason, "protocol_error")

        self.assertEqual(
            sum(e.kind == "tool_started" for e in result.events),
            1,
        )

    async def test_cancellation_cleans_up(self):
        started = asyncio.Event()
        cleaned = asyncio.Event()

        async def waiting(arguments):
            try:
                started.set()
                await asyncio.Event().wait()
            finally:
                cleaned.set()

        events = []

        task = asyncio.create_task(
            build_runtime(handler=waiting).run(
                "test", events=events
            )
        )

        try:
            await asyncio.wait_for(started.wait(), timeout=2)
        finally:
            task.cancel()

            with self.assertRaises(asyncio.CancelledError):
                await task

        self.assertTrue(cleaned.is_set())
        self.assertEqual(events[-1].kind, "run_cancelled")

    def test_duplicate_tool_registration(self):
        async def handler(arguments):
            return 0

        tools = ToolRegistry()
        tool = ToolSpec("add", "test", AddInput, handler)

        tools.register(tool)

        with self.assertRaises(ValueError):
            tools.register(tool)