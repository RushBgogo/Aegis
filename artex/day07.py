"""Day 7 操作账本演示：已完成操作只执行一次，重复调用直接重放结果。"""
import argparse
import asyncio

from artex.db import Repository
from artex.recovery import OperationLedger


def build_parser():
    parser = argparse.ArgumentParser(description="Aegis Day7 操作账本演示")
    parser.add_argument("--operation-id", default="stable-op", help="稳定操作编号")
    return parser


async def main():
    args = build_parser().parse_args()
    repo = Repository("sqlite:///:memory:")
    repo.setup()  # 演示专用内存库；生产数据库使用 alembic upgrade head
    try:
        task = repo.create_task("recovery", "show replay")
        ledger = OperationLedger(repo)
        executions = 0

        async def handler():
            nonlocal executions
            executions += 1
            return {"answer": 65}

        for _ in range(2):
            result = await ledger.execute(
                task["id"],
                args.operation_id,
                "local-demo",
                {},
                handler,
            )
            print("操作结果:", result)
        print("实际执行次数:", executions)
    finally:
        repo.close()


if __name__ == "__main__":
    asyncio.run(main())
