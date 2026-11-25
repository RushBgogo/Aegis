import asyncio
import unittest

from artex.approval import (
    ApprovalDenied,
    ApprovalExecutor,
    ApprovalExpired,
    ApprovalService,
)
from artex.db import Repository


class ApprovalTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.repo = Repository("sqlite:///:memory:")
        self.repo.setup()
        self.task = self.repo.create_task("test", "test")["id"]
        self.repo.transition_task(self.task, "created", "running")

    async def asyncTearDown(self):
        self.repo.close()

    async def test_deny_never_calls_handler(self):
        called = []

        async def handler():
            called.append(True)
            return {"value": 1}

        executor = ApprovalExecutor(
            self.repo, policy=lambda tool, args: "deny"
        )
        with self.assertRaises(ApprovalDenied):
            await executor.execute(self.task, "op", "write", {}, handler)

        self.assertEqual(called, [])

    async def test_approval_binds_exact_parameters(self):
        service = ApprovalService(self.repo)
        service.request(self.task, "op", "write", {"a": 1})
        service.decide(self.task, "op", True)

        with self.assertRaises(ValueError):
            service.request(self.task, "op", "write", {"a": 2})

    async def test_expired_approval_fails(self):
        now = [10.0]
        service = ApprovalService(self.repo, clock=lambda: now[0])
        service.request(self.task, "op", "write", {}, ttl=1)
        now[0] = 12.0

        with self.assertRaises(ApprovalExpired):
            service.decide(self.task, "op", True)

    async def test_wait_can_cancel_without_execution(self):
        called = []

        async def handler():
            called.append(True)
            return {}

        executor = ApprovalExecutor(
            self.repo, policy=lambda tool, args: "ask"
        )
        work = asyncio.create_task(
            executor.execute(self.task, "op", "write", {}, handler)
        )
        await asyncio.sleep(0.04)
        work.cancel()

        with self.assertRaises(asyncio.CancelledError):
            await work
        self.assertFalse(called)

    async def test_execute_and_cached_replay(self):
        called = []

        async def handler():
            called.append(True)
            return {"value": 7}

        executor = ApprovalExecutor(
            self.repo, policy=lambda tool, args: "allow"
        )
        first = await executor.execute(self.task, "op", "observe", {}, handler)
        second = await executor.execute(self.task, "op", "observe", {}, handler)

        self.assertEqual(first, second)
        self.assertEqual(len(called), 1)

    async def test_paused_task_is_rejected(self):
        self.repo.transition_task(self.task, "running", "paused")
        called = []

        async def handler():
            called.append(True)
            return {}

        executor = ApprovalExecutor(
            self.repo, policy=lambda tool, args: "allow"
        )
        with self.assertRaises(ApprovalDenied):
            await executor.execute(self.task, "op", "write", {}, handler)

        self.assertFalse(called)
