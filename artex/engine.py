"""单进程任务调度器：Planner 串行，Worker 并发，数据库负责最终裁决。"""
import asyncio

from artex.planner import Planner
from artex.worker import Worker


class Engine:
    """在单个进程内调度一个 Task。

    设计要点：
    - 同一个 Engine 对同一个 task 只允许一个 run() 进入；
    - 每一批 open intent 交给多个 Worker 并发领取；
    - 一批全部提交后只规划一次，避免每个 Worker 都触发一轮 Planner；
    - 数据库的 claim_intent 仍然是最终裁决者，队列只表示“待尝试”。
    """

    def __init__(
        self,
        repo,
        worker=None,
        planner=None,
        workers=3,
        max_rounds=100,
        batch_size=100,
    ):
        if workers < 1:
            raise ValueError("workers must be positive")
        if max_rounds < 1:
            raise ValueError("max_rounds must be positive")
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        self.repo = repo
        self.worker = worker or Worker(repo)
        self.planner = planner or Planner(repo)
        self.workers = workers
        self.max_rounds = max_rounds
        self.batch_size = batch_size
        self._task_locks: dict[str, asyncio.Lock] = {}

    async def run(self, task_id: str) -> None:
        """运行直到任务离开 running 状态，或达到调度轮数上限。"""
        lock = self._task_locks.setdefault(task_id, asyncio.Lock())
        async with lock:
            self._ensure_running(task_id)
            for _ in range(self.max_rounds):
                if not self._is_running(task_id):
                    return
                await self.planner.plan(task_id)
                if not self._is_running(task_id):
                    return

                frontier = self.repo.list_nodes(task_id, "intent", "open")
                if not frontier:
                    return
                # 只处理有限批次，避免一次拉爆大量意图；剩余意图下一轮继续。
                batch = frontier[: self.batch_size]
                await self._run_batch(task_id, [intent["id"] for intent in batch])

            raise RuntimeError(f"scheduler round limit reached for task {task_id}")

    def _ensure_running(self, task_id: str) -> None:
        """调用方可以直接创建 created 任务；Engine 负责把它推进到 running。"""
        if self.repo.get_task(task_id)["status"] == "created":
            self.repo.transition_task(task_id, "created", "running")

    def _is_running(self, task_id: str) -> bool:
        return self.repo.get_task(task_id)["status"] == "running"

    async def _run_batch(self, task_id: str, intent_ids: list[str]) -> None:
        """把一个有限批次分给 Worker；真正的唯一领取由 claim_intent 保证。"""
        queue: asyncio.Queue[str] = asyncio.Queue()
        for intent_id in intent_ids:
            queue.put_nowait(intent_id)

        async def consume(owner: str) -> None:
            while True:
                try:
                    intent_id = queue.get_nowait()
                except asyncio.QueueEmpty:
                    return
                await self.worker.execute(task_id, intent_id, owner)

        async with asyncio.TaskGroup() as group:
            for index in range(self.workers):
                group.create_task(consume(f"worker-{index + 1}"))
