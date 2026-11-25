"""单进程任务调度器：Planner 串行，Worker 并发，生命周期由 Engine 管理。"""
import asyncio
import time

from artex.planner import Planner
from artex.worker import Worker


class Engine:
    """在单个进程内调度一个 Task。

    设计要点：
    - 同一个 Engine 对同一个 task 只允许一个 run() 进入；
    - 每一批 open intent 交给多个 Worker 并发领取；
    - 一批全部提交后只规划一次，避免每个 Worker 都触发 Planner；
    - 数据库 claim_intent 是最终裁决者，队列只表示“待尝试”；
    - pause/cancel/delete 先关闭新工作准入，再取消在途任务并等待清理。
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
        # 公开的可观测状态：正在执行的子任务、最近一次停止原因。
        self.active: dict[str, set[asyncio.Task]] = {}
        self.causes: dict[str, str] = {}
        self._task_locks: dict[str, asyncio.Lock] = {}

    async def run(self, task_id: str) -> None:
        """运行到任务离开 running，或触发截止时间/轮数上限。"""
        lock = self._task_locks.setdefault(task_id, asyncio.Lock())
        async with lock:
            self._ensure_running(task_id)
            deadline = self.repo.get_task(task_id)["deadline"]
            seconds = None if deadline is None else max(0.0, deadline - time.time())
            timeout = asyncio.timeout(seconds)
            try:
                async with timeout:
                    await self._drain(task_id)
            except TimeoutError:
                # 区分“任务截止时间到了”和工具自己抛出的 TimeoutError。
                if not timeout.expired():
                    self._fail_if_running(task_id, "TimeoutError")
                    raise
                self._fail_if_running(task_id, "deadline_exceeded")
            except Exception as exc:
                self._fail_if_running(task_id, type(exc).__name__)
                raise

    def _ensure_running(self, task_id: str) -> None:
        """调用方可以直接创建 created 任务；Engine 负责推进到 running。"""
        if self.repo.get_task(task_id)["status"] == "created":
            self.repo.transition_task(task_id, "created", "running")

    def _is_running(self, task_id: str) -> bool:
        return self.repo.get_task(task_id)["status"] == "running"

    def _fail_if_running(self, task_id: str, cause: str) -> None:
        if not self._is_running(task_id):
            return
        self.causes[task_id] = cause
        self.repo.append_event(task_id, "task_stopped", {"cause": cause})
        self.repo.transition_task(task_id, "running", "failed")

    async def _drain(self, task_id: str) -> None:
        """规划 → 取有限批次 → 并发执行 → 再规划，直到任务离开 running。"""
        try:
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
        finally:
            self.active.pop(task_id, None)

    async def _run_batch(self, task_id: str, intent_ids: list[str]) -> None:
        """把一个有限批次分给 Worker；真正的唯一领取由 claim_intent 保证。"""
        queue: asyncio.Queue[str] = asyncio.Queue()
        for intent_id in intent_ids:
            queue.put_nowait(intent_id)

        async def consume(owner: str) -> None:
            while self._is_running(task_id):
                try:
                    intent_id = queue.get_nowait()
                except asyncio.QueueEmpty:
                    return
                await self.worker.execute(task_id, intent_id, owner)

        running = self.active.setdefault(task_id, set())
        async with asyncio.TaskGroup() as group:
            for index in range(self.workers):
                child = group.create_task(consume(f"worker-{index + 1}"))
                running.add(child)
                child.add_done_callback(running.discard)
        # 给 done 回调一个事件循环 tick，保证 active 集合及时清理。
        await asyncio.sleep(0)

    async def _stop(self, task_id: str, cause: str) -> None:
        """关闭准入后取消在途子任务，并等待它们完成清理。"""
        self.causes[task_id] = cause
        running = list(self.active.get(task_id, ()))
        for child in running:
            child.cancel(cause)
        if running:
            await asyncio.gather(*running, return_exceptions=True)

    async def pause(self, task_id: str) -> None:
        state = self.repo.get_task(task_id)["status"]
        if state == "paused":
            return
        if state not in {"created", "running"}:
            raise ValueError("task cannot be paused")
        if not self.repo.transition_task(task_id, state, "paused"):
            raise ValueError("concurrent lifecycle transition")
        await self._stop(task_id, "paused_by_user")

    async def resume(self, task_id: str) -> None:
        if not self.repo.transition_task(task_id, "paused", "running"):
            raise ValueError("task not paused")
        await self.run(task_id)

    async def cancel(self, task_id: str) -> None:
        state = self.repo.get_task(task_id)["status"]
        if state not in {"created", "running", "paused"}:
            raise ValueError("task cannot be cancelled")
        if not self.repo.transition_task(task_id, state, "cancelled"):
            raise ValueError("concurrent lifecycle transition")
        await self._stop(task_id, "cancelled_by_user")

    async def delete(self, task_id: str) -> None:
        state = self.repo.get_task(task_id)["status"]
        if state == "deleted":
            return
        if state != "deleting" and not self.repo.transition_task(
            task_id, state, "deleting"
        ):
            raise ValueError("concurrent lifecycle transition")
        await self._stop(task_id, "task_deleted")
        self.repo.transition_task(task_id, "deleting", "deleted")
