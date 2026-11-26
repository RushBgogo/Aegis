"""Day 13 人工协作演示：幂等提示、旁路问答。"""
import argparse
import asyncio

from artex.db import Repository
from artex.engine import Engine
from artex.human import HumanService


def build_parser():
    return argparse.ArgumentParser(
        description="Aegis Day13 人工协作演示"
    )


async def main():
    build_parser().parse_args()
    repo = Repository("sqlite:///:memory:")
    repo.setup()
    try:
        task = repo.create_task(
            "人工协作",
            "评估样例",
            scope=["https://lab.example/public"],
        )
        service = HumanService(repo)
        first = service.add_hint(
            task["id"], "关注证据来源", "hint-1"
        )
        second = service.add_hint(
            task["id"], "关注证据来源", "hint-1"
        )
        assert first["id"] == second["id"]

        await Engine(repo).run(task["id"])
        print(
            await service.ask(task["id"], "任务进度如何？")
        )
    finally:
        repo.close()


if __name__ == "__main__":
    asyncio.run(main())
