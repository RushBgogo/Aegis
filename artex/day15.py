"""Day 15 流量证据、报告与复测演示。"""
import argparse
import asyncio

import httpx

from artex.db import Repository
from artex.reporting import ReportService
from artex.traffic import RecordingClient


def build_parser():
    return argparse.ArgumentParser(
        description="Aegis Day15 流量证据与报告演示"
    )


async def main():
    build_parser().parse_args()
    repo = Repository("sqlite:///:memory:")
    repo.setup()
    try:
        target = "https://lab.example/private"
        task = repo.create_task(
            "证据交付",
            "评估固定访问控制样例",
            scope=[target],
        )
        task_id = task["id"]
        intent = repo.create_node(
            task_id,
            "intent",
            "验证固定样例",
            payload={"target": target},
        )
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(
                lambda request: httpx.Response(
                    200,
                    text="fixture: private record visible",
                )
            )
        ) as client:
            evidence = await RecordingClient(
                repo, task_id, client
            ).get(
                target,
                intent_id=intent["id"],
                call_id="fixture-http-1",
            )

        finding = repo.create_node(
            task_id,
            "finding",
            "固定样例中的访问控制问题",
            payload={
                "target": target,
                "intent_id": intent["id"],
                "finding_record_id": "finding-record-1",
            },
        )
        service = ReportService(repo)
        service.bind(
            task_id,
            finding["id"],
            evidence["evidence_id"],
            "固定样例正文出现预期私有记录，仅用于本地验证",
        )
        print(service.build(task_id)["markdown"])

        async def fixed(target):
            return {
                "status": 403,
                "body": "fixture: access denied",
                "issue": False,
            }

        print(
            "复测：",
            await service.retest(task_id, finding["id"], fixed),
        )
    finally:
        repo.close()


if __name__ == "__main__":
    asyncio.run(main())
