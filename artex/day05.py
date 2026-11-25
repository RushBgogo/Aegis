"""Day 5 多 Worker 演示：有限批次 + 每任务串行 Planner。"""
import argparse
import asyncio

from artex.db import Repository
from artex.engine import Engine
from artex.worker import Worker


def build_fixtures(count: int) -> dict[str, dict]:
    return {
        f"https://lab.example/{index}": {
            "status": 200,
            "body": str(index),
            "issue": False,
        }
        for index in range(count)
    }


def build_parser():
    parser = argparse.ArgumentParser(description="Aegis Day5 多 Worker 调度演示")
    parser.add_argument("--workers", type=int, default=3, help="并发 Worker 数量")
    parser.add_argument("--count", type=int, default=10, help="fixture 目标数量")
    parser.add_argument("--batch-size", type=int, default=100, help="每轮最多处理的意图数")
    return parser


async def main():
    args = build_parser().parse_args()
    if args.workers < 1:
        raise SystemExit("--workers must be >= 1")
    if args.count < 1:
        raise SystemExit("--count must be >= 1")
    if args.batch_size < 1:
        raise SystemExit("--batch-size must be >= 1")

    fixtures = build_fixtures(args.count)
    repo = Repository("sqlite:///:memory:")
    repo.setup()  # 演示专用内存库；生产数据库使用 alembic upgrade head
    try:
        task = repo.create_task("并发实验", "评估固定样例", list(fixtures))
        engine = Engine(
            repo,
            Worker(repo, fixtures=fixtures),
            workers=args.workers,
            batch_size=args.batch_size,
        )
        await engine.run(task["id"])

        intents = repo.list_nodes(task["id"], "intent")
        facts = repo.list_nodes(task["id"], "fact")
        done = [intent for intent in intents if intent["state"] == "done"]
        status = repo.get_task(task["id"])["status"]

        print("意图:", len(intents), "完成:", len(done))
        print("事实:", len(facts), "状态:", status)
        if status != "done":
            raise SystemExit(1)
    finally:
        repo.close()


if __name__ == "__main__":
    asyncio.run(main())
