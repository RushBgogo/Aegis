import asyncio
import time
import unittest

from artex.db import Repository
from artex.engine import Engine
from artex.planner import sample_task
from artex.worker import FixtureModel, Worker


class LifecycleTests(unittest.IsolatedAsyncioTestCase):
    async def test_pause_resume_delete(self):
        repo = Repository("sqlite:///:memory:")
        repo.setup()
        self.addCleanup(repo.close)
        task = sample_task(repo)
        task_id = task["id"]
        started, cleaned = asyncio.Event(), asyncio.Event()

        class Waiting(FixtureModel):
            async def complete(self, messages, tools):
                try:
                    started.set()
                    await asyncio.Event().wait()
                finally:
                    cleaned.set()

        worker = Worker(repo, model_factory=Waiting)
        engine = Engine(repo, worker)

        run = asyncio.create_task(engine.run(task_id))
        await asyncio.wait_for(started.wait(), 2)
        await engine.pause(task_id)
        await run

        self.assertTrue(cleaned.is_set())
        self.assertEqual(repo.list_nodes(task_id, "intent", "running"), [])
        self.assertEqual(engine.causes[task_id], "paused_by_user")

        worker.model_factory = FixtureModel
        await engine.resume(task_id)
        self.assertEqual(repo.get_task(task_id)["status"], "done")

        await engine.delete(task_id)
        with self.assertRaises(ValueError):
            repo.create_node(task_id, "hint", "too late")
        self.assertEqual(engine.active, {})

    async def test_cancel_sets_cause(self):
        repo = Repository("sqlite:///:memory:")
        repo.setup()
        self.addCleanup(repo.close)
        task = sample_task(repo)
        task_id = task["id"]
        started = asyncio.Event()

        class Waiting(FixtureModel):
            async def complete(self, messages, tools):
                started.set()
                await asyncio.Event().wait()

        engine = Engine(repo, Worker(repo, model_factory=Waiting))
        run = asyncio.create_task(engine.run(task_id))
        await asyncio.wait_for(started.wait(), 2)
        await engine.cancel(task_id)
        await run

        self.assertEqual(repo.get_task(task_id)["status"], "cancelled")
        self.assertEqual(engine.causes[task_id], "cancelled_by_user")
        self.assertEqual(engine.active, {})

    async def test_delete_while_running_installs_barrier(self):
        repo = Repository("sqlite:///:memory:")
        repo.setup()
        self.addCleanup(repo.close)
        task = sample_task(repo)
        task_id = task["id"]
        started = asyncio.Event()

        class Waiting(FixtureModel):
            async def complete(self, messages, tools):
                started.set()
                await asyncio.Event().wait()

        engine = Engine(repo, Worker(repo, model_factory=Waiting))
        run = asyncio.create_task(engine.run(task_id))
        await asyncio.wait_for(started.wait(), 2)
        await engine.delete(task_id)
        await run

        self.assertEqual(repo.get_task(task_id)["status"], "deleted")
        self.assertEqual(engine.causes[task_id], "task_deleted")
        self.assertEqual(engine.active, {})
        with self.assertRaises(ValueError):
            repo.create_node(task_id, "hint", "too late")

    async def test_deadline_is_separate_cause(self):
        repo = Repository("sqlite:///:memory:")
        repo.setup()
        self.addCleanup(repo.close)
        task = sample_task(repo)
        task_id = task["id"]
        repo.set_deadline(task_id, time.time() + 0.01)

        class Slow(FixtureModel):
            async def complete(self, messages, tools):
                await asyncio.sleep(1)
                return await super().complete(messages, tools)

        engine = Engine(repo, Worker(repo, model_factory=Slow))
        await engine.run(task_id)

        self.assertEqual(repo.get_task(task_id)["status"], "failed")
        self.assertEqual(engine.causes[task_id], "deadline_exceeded")
