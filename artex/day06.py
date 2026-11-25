"""Day 6 生命周期演示：暂停、恢复、截止时间与删除。"""
import argparse
import asyncio

from artex.db import Repository
from artex.engine import Engine
from artex.planner import sample_task
from artex.worker import FixtureModel, Worker


class SlowModel(FixtureModel):
    async def complete(self, messages, tools):
        await asyncio.sleep(0.1)
        return await super().complete(messages, tools)


def build_parser():
    parser = argparse.ArgumentParser(description="Aegis Day6 生命周期演示")
    parser.add_argument("--workers", type=int, default=3, help="并发 Worker 数量")
    return parser


async def main():
    args = build_parser().parse_args()
    if args.workers < 1:
        raise SystemExit("--workers must be >= 1")

    repo = Repository("sqlite:///:memory:")
    repo.setup()  # 演示专用内存库；生产数据库使用 alembic upgrade head
    try:
        task = sample_task(repo)
        task_id = task["id"]
        engine = Engine(repo, Worker(repo, model_factory=SlowModel), workers=args.workers)

        run = asyncio.create_task(engine.run(task_id))
        await asyncio.sleep(0.02)
        await engine.pause(task_id)
        await run
        print("暂停:", repo.get_task(task_id)["status"], engine.causes.get(task_id))

        await engine.resume(task_id)
        print("恢复完成:", repo.get_task(task_id)["status"])
    finally:
        repo.close()


if __name__ == "__main__":
    asyncio.run(main())
