import unittest

from runtime import (
    AgentRuntime,
    Message,
    ModelReply,
    ToolCall,
    ToolInput,
    ToolRegistry,
    ToolSpec,
)
from artex.db import Repository
from artex.recovery import (
    OperationLedger,
    UncertainOperation,
    load_checkpoint,
    save_checkpoint,
)
from artex.worker import DEFAULT_FIXTURES, Worker


class AddInput(ToolInput):
    a: int
    b: int


async def add(arguments: AddInput) -> int:
    return arguments.a + arguments.b


class ScriptedAddModel:
    async def complete(self, messages, tools):
        last = messages[-1]
        if last.role == "user":
            return ModelReply(
                tool_calls=(ToolCall("call-1", "add", {"a": 7, "b": 5}),)
            )
        if last.role == "tool" and last.result is not None:
            return ModelReply(text=str(last.result.content))
        raise ValueError("unexpected message order")


def build_runtime():
    tools = ToolRegistry()
    tools.register(ToolSpec("add", "计算两个整数之和", AddInput, add))
    return AgentRuntime(ScriptedAddModel(), tools)


class RecoveryTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.repo = Repository("sqlite:///:memory:")
        self.repo.setup()
        self.task = self.repo.create_task("recovery", "observe")

    def tearDown(self):
        self.repo.close()

    async def test_restart_after_side_effect_before_completion(self):
        tid = self.task["id"]

        async def idempotent_effect():
            self.repo.put_record("local-effect", "op-1", {"count": 1}, tid)
            return {"count": 1}

        async def crash_after_effect():
            await idempotent_effect()
            raise RuntimeError("simulated process exit before completion record")

        with self.assertRaises(RuntimeError):
            await OperationLedger(self.repo).execute(
                tid, "op-1", "write", {}, crash_after_effect
            )

        fresh = OperationLedger(self.repo)
        self.assertEqual(
            await fresh.execute(tid, "op-1", "write", {}, idempotent_effect),
            {"count": 1},
        )

        async def forbidden():
            raise AssertionError("completed operation must not run again")

        await fresh.execute(tid, "op-1", "write", {}, forbidden)
        self.assertEqual(len(self.repo.list_records("local-effect", tid)), 1)

    async def test_uncertain_external_write_and_argument_conflict(self):
        tid = self.task["id"]

        async def fail():
            raise RuntimeError("connection lost after external send")

        with self.assertRaises(RuntimeError):
            await OperationLedger(self.repo).execute(
                tid, "op-2", "send", {}, fail, idempotent=False
            )

        with self.assertRaises(UncertainOperation):
            await OperationLedger(self.repo).execute(
                tid, "op-2", "send", {}, fail, idempotent=False
            )

        # 首次登记为非幂等后，后续调用即使声称幂等也不能盲目重试。
        with self.assertRaises(UncertainOperation):
            await OperationLedger(self.repo).execute(
                tid, "op-2", "send", {}, fail, idempotent=True
            )

        with self.assertRaises(ValueError):
            await OperationLedger(self.repo).execute(
                tid, "op-2", "send", {"changed": True}, fail
            )

    async def test_checkpoint_resume_uses_tool_result(self):
        first = await build_runtime().run("计算", max_turns=1)
        self.assertEqual(first.reason, "max_turns")

        save_checkpoint(self.repo, self.task["id"], "stable-session", first.messages)
        history = load_checkpoint(self.repo, self.task["id"], "stable-session")
        self.assertIsNotNone(history)

        resumed = await build_runtime().run(
            "ignored because history exists", history=history
        )
        self.assertEqual(resumed.text, "12")
        self.assertFalse(
            any(event.kind == "tool_started" for event in resumed.events)
        )

    async def test_worker_observe_uses_operation_ledger(self):
        task = self.repo.create_task(
            "worker-ledger", "observe", list(DEFAULT_FIXTURES)
        )
        task_id = task["id"]
        self.repo.transition_task(task_id, "created", "running")
        target = next(iter(DEFAULT_FIXTURES))
        intent = self.repo.create_node(
            task_id, "intent", "observe", payload={"target": target}
        )

        worker = Worker(self.repo, executor=OperationLedger(self.repo))
        self.assertTrue(await worker.execute(task_id, intent["id"], "worker-1"))

        self.assertEqual(len(self.repo.list_nodes(task_id, "fact")), 1)
        self.assertEqual(len(self.repo.list_records("evidence", task_id)), 1)
        operations = self.repo.list_records("operation", task_id)
        self.assertEqual(len(operations), 1)
        self.assertEqual(operations[0]["value"]["state"], "done")

    async def test_checkpoint_rejects_dangling_tool_call(self):
        messages = [
            Message(
                role="assistant",
                tool_calls=(ToolCall("call-1", "add", {"a": 1, "b": 2}),),
            )
        ]
        with self.assertRaises(ValueError):
            save_checkpoint(
                self.repo, self.task["id"], "dangling-session", messages
            )
