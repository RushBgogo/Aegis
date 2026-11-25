"""Day 8 人工审批演示：ask 策略、等待决定、审批后执行 observe。"""
import argparse
import asyncio

from artex.approval import ApprovalExecutor
from artex.db import Repository
from artex.engine import Engine
from artex.worker import Worker


def build_parser():
    parser = argparse.ArgumentParser(description="Aegis Day8 人工审批演示")
    parser.add_argument(
        "--wait-seconds",
        type=float,
        default=2.0,
        help="等待审批记录出现的秒数",
    )
    return parser


async def wait_for_approval(repo, task_id, timeout):
    deadline = asyncio.get_running_loop().time() + timeout
    while asyncio.get_running_loop().time() < deadline:
        rows = repo.list_records("approval", task_id)
        if rows:
            return rows[0]["key"]
        await asyncio.sleep(0.01)
    raise RuntimeError("没有收到审批请求")


async def main():
    args = build_parser().parse_args()
    if args.wait_seconds <= 0:
        raise SystemExit("--wait-seconds must be > 0")

    repo = Repository("sqlite:///:memory:")
    repo.setup()  # 演示专用内存库；生产数据库使用 alembic upgrade head
    try:
        task = repo.create_task(
            "审批演示", "核实本地样例", scope=["https://lab.example/public"]
        )
        executor = ApprovalExecutor(
            repo, policy=lambda tool, args: "ask"
        )
        worker = Worker(repo, executor=executor)
        engine = Engine(repo, worker=worker)

        run = asyncio.create_task(engine.run(task["id"]))
        try:
            operation_id = await wait_for_approval(
                repo, task["id"], args.wait_seconds
            )
        except Exception:
            run.cancel()
            await asyncio.gather(run, return_exceptions=True)
            raise

        executor.approvals.decide(task["id"], operation_id, True)
        await run

        print("审批决定：approved；任务状态：", repo.get_task(task["id"])["status"])
        print("事实数：", len(repo.list_nodes(task["id"], "fact")))
    finally:
        repo.close()


if __name__ == "__main__":
    asyncio.run(main())
