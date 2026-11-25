import asyncio
import unittest

from artex.db import Repository
from artex.engine import Engine
from artex.planner import Planner
from artex.worker import Worker


class EngineTests(unittest.IsolatedAsyncioTestCase):
    async def test_ten_intents_and_planner_not_reentrant(self):
        repo = Repository("sqlite:///:memory:")
        repo.setup()
        self.addCleanup(repo.close)

        class CheckedPlanner(Planner):
            def __init__(self, repo):
                super().__init__(repo)
                self.busy = False

            async def plan(self, task_id):
                if self.busy:
                    raise AssertionError("planner reentered")
                self.busy = True
                try:
                    await asyncio.sleep(0)
                    return await super().plan(task_id)
                finally:
                    self.busy = False

        fixtures = {
            f"fixture:{index}": {"status": 200, "body": str(index), "issue": False}
            for index in range(10)
        }
        task = repo.create_task("concurrency", "assess", list(fixtures))
        engine = Engine(repo, Worker(repo, fixtures=fixtures), CheckedPlanner(repo))

        await asyncio.gather(engine.run(task["id"]), engine.run(task["id"]))

        self.assertEqual(len(repo.list_nodes(task["id"], "fact")), 10)
        self.assertEqual(len(repo.list_nodes(task["id"], "intent", "done")), 10)
        self.assertEqual(repo.get_task(task["id"])["status"], "done")

    async def test_batch_size_limits_each_round(self):
        repo = Repository("sqlite:///:memory:")
        repo.setup()
        self.addCleanup(repo.close)

        fixtures = {
            f"fixture:{index}": {"status": 200, "body": str(index), "issue": False}
            for index in range(10)
        }
        task = repo.create_task("bounded", "assess", list(fixtures))
        engine = Engine(
            repo,
            Worker(repo, fixtures=fixtures),
            workers=2,
            batch_size=3,
        )
        await engine.run(task["id"])

        self.assertEqual(len(repo.list_nodes(task["id"], "fact")), 10)
        self.assertEqual(len(repo.list_nodes(task["id"], "intent", "done")), 10)
        self.assertEqual(repo.get_task(task["id"])["status"], "done")

    async def test_engine_rejects_zero_workers(self):
        repo = Repository("sqlite:///:memory:")
        repo.setup()
        self.addCleanup(repo.close)

        with self.assertRaises(ValueError):
            Engine(repo, workers=0)
