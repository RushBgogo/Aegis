import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import tempfile
import unittest

from artex.db import Repository

PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORK_DIR = PROJECT_ROOT / ".work"


class RepositoryTests(unittest.TestCase):
    def setUp(self):
        WORK_DIR.mkdir(parents=True, exist_ok=True)
        self.tmp = tempfile.TemporaryDirectory(dir=WORK_DIR)
        default_url = "sqlite:///" + (Path(self.tmp.name) / "test.db").as_posix()
        self.url = os.getenv("ARTEX_TEST_DATABASE_URL") or default_url
        self.repo = Repository(self.url)
        self.repo.setup()
        self.task = self.repo.create_task("fixture", "observe")
        self.tid = self.task["id"]
        self.repo.transition_task(self.tid, "created", "running")

    def tearDown(self):
        self.repo.close()
        self.tmp.cleanup()

    def test_two_connections_only_one_claim(self):
        intent = self.repo.create_node(self.tid, "intent", "observe")
        def claim(owner):
            other = Repository(self.url)
            try:
                return other.claim_intent(self.tid, intent["id"], owner)
            finally:
                other.close()
        with ThreadPoolExecutor(2) as pool:
            self.assertEqual(sorted(pool.map(claim, ["A", "B"])), [False, True])

    def test_scope_and_version(self):
        other = self.repo.create_task("other", "other")
        node = self.repo.create_node(self.tid, "fact", "first")
        with self.assertRaises(KeyError):
            self.repo.get_node(other["id"], node["id"])
        self.assertTrue(self.repo.update_node(self.tid, node["id"], "second", {}, 1))
        self.assertFalse(self.repo.update_node(self.tid, node["id"], "stale", {}, 1))
        foreign = self.repo.create_node(other["id"], "fact", "foreign")
        with self.assertRaises(ValueError):
            self.repo.add_edge(self.tid, node["id"], foreign["id"], "yields")

    def test_node_priority_update(self):
        node = self.repo.create_node(self.tid, "intent", "priority")
        self.assertTrue(self.repo.set_node_priority(self.tid, node["id"], 5))
        self.assertEqual(self.repo.get_node(self.tid, node["id"])["priority"], 5)

    def test_node_state_vocabulary(self):
        node = self.repo.create_node(self.tid, "intent", "state")
        self.assertTrue(
            self.repo.transition_node(self.tid, node["id"], "open", "blocked")
        )
        with self.assertRaises(ValueError):
            self.repo.create_node(self.tid, "fact", "invalid", state="unknown")
        with self.assertRaises(ValueError):
            self.repo.transition_node(self.tid, node["id"], "blocked", "unknown")

    def test_restart_and_record_cas(self):
        self.assertTrue(self.repo.put_record("note", "x", {"a": 1}, self.tid))
        self.assertFalse(self.repo.put_record("note", "x", {"a": 2}, self.tid))
        other = Repository(self.url)
        try:
            self.assertEqual(other.get_task(self.tid)["status"], "running")
            self.assertTrue(other.put_record("note", "x", {"a": 3}, self.tid, 1))
            self.assertFalse(self.repo.put_record("note", "x", {}, self.tid, 1))
        finally:
            other.close()
